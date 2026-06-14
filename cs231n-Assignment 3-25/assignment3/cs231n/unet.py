import copy
from einops import rearrange
from torch import einsum

from torch import nn
import torch
import torch.nn.functional as F
import math


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def Upsample(dim, dim_out=None):
    """将图像特征分辨率上采样（放大） 2 倍。"""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),  # type: ignore
    )


def Downsample(dim, dim_out=None):
    """将图像特征分辨率下采样（缩小） 2 倍。"""
    return nn.Conv2d(dim, default(dim_out, dim), kernel_size=2, stride=2)  # type: ignore


class RMSNorm(nn.Module):
    """RMSNorm 层，它是 LayerNorm 的一种计算效率更高的简化变体。"""

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale


class SinusoidalPosEmb(nn.Module):
    """用于时间步的正弦位置嵌入（Sinusoidal position embedding）。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):
    """带有特征调制（feature modulation）的卷积块。"""

    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.GELU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        # Scale 和 shift 用于调制输出。这是特征融合（feature fusion）的一种变体，
        # 比简单地将特征图相加更强大。
        if exists(scale_shift):
            scale, shift = scale_shift # type: ignore
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """一个类似 ResNet 的块，带有依赖于上下文的特征调制。"""

    def __init__(self, dim, dim_out, context_dim):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.context_dim = context_dim

        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(context_dim, dim_out * 2))
            if exists(context_dim)
            else None
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, context=None):

        scale_shift = None
        if exists(self.mlp) and exists(context):
            context = self.mlp(context) # type: ignore
            context = rearrange(context, "b c -> b c 1 1")
            scale_shift = context.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.dropout(h)
        h = self.block2(h)
        return h + self.res_conv(x)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        condition_dim,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        uncond_prob=0.2,
    ):
        super().__init__()

        self.init_conv = nn.Conv2d(channels, dim, 3, padding=1)
        self.channels = channels

        # 每一层的通道数，即 [d1, d2, ..., dn]
        dims = [dim] + [dim * m for m in dim_mults]
        # 下采样层中每个 U-Net 块的输入和输出通道数
        # 例如 [(d1, d2), (d2, d3), ..., (dn-1, dn)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # 上采样层中每个 U-Net 块的输入和输出通道数
        # 例如 [(dn, dn-1), (dn-1, dn-2), ..., (d2, d1)]
        in_out_ups = [(b, a) for a, b in reversed(in_out)]

        # 将时间步编码为上下文（context）
        context_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

        # 将条件（即文本嵌入 text embedding）编码为上下文
        self.condition_dim = condition_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

        # 在训练期间丢弃条件的概率（用于无分类器引导 classifier-free guidance）
        self.uncond_prob = uncond_prob

        # UNet 下采样和上采样块。
        # self.downs 是一个包含 ModuleList 的 ModuleList。
        self.downs = nn.ModuleList([])
        # self.ups 是一个包含 ModuleList 的 ModuleList。
        self.ups = nn.ModuleList([])

        ####################################################################
        # 下采样块 (Downsampling blocks)
        ####################################################################
        for ind, (dim_in, dim_out) in enumerate(in_out):
            ##################################################################
            # TODO: 创建一个名为 `down_block` 的 UNet 下采样层，将其设为 ModuleList。
            # 它应该是一个包含 3 个块的 ModuleList：[ResnetBlock, ResnetBlock, Downsample]。
            # 每个 ResnetBlock 接收 dim_in 个通道，并输出 dim_in 个通道。
            # 确保将 context_dim 传递给每个 ResnetBlock。
            # Downsample 块接收 dim_in 个通道，并输出 dim_out 个通道。
            # 请务必严格遵循这种 ModuleList 结构，以便能够加载预训练权重。
            ##################################################################
            down_block = nn.ModuleList([
                ResnetBlock(dim_in, dim_in, context_dim=context_dim),
                ResnetBlock(dim_in, dim_in, context_dim=context_dim),
                Downsample(dim_in, dim_out)
            ])
            ##################################################################
            self.downs.append(down_block) # type: ignore

        # 中间块 (Middle blocks)
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, context_dim=context_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, context_dim=context_dim)

        ####################################################################
        # 上采样块 (Upsampling blocks)
        ####################################################################
        # 通过完全镜像下采样块来创建上采样块。
        # self.ups 也将是一个包含 ModuleList 的 ModuleList。
        # 每个块列表将包含 3 个块：[Upsample, ResnetBlock, ResnetBlock]。
        for ind, (dim_in, dim_out) in enumerate(in_out_ups):
            ##################################################################
            # TODO: 创建一个 UNet 上采样层作为 ModuleList。
            # 它应该是一个包含 3 个块的 ModuleList：[Upsample, ResnetBlock, ResnetBlock]。
            # 这将与对应的下采样块镜像对称。
            # 不要忘记处理跳跃连接（skip connections），因此两个 ResnetBlock 的输入通道数都应为 2 x dim_out。
            ##################################################################
            up_block = nn.ModuleList([
                Upsample(dim_in, dim_out),
                ResnetBlock(dim_out * 2, dim_out, context_dim=context_dim),
                ResnetBlock(dim_out * 2, dim_out, context_dim=context_dim)
            ])
            self.ups.append(up_block) # type: ignore
            ##################################################################

        # 最后的卷积层，映射到输出通道数
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def cfg_forward(self, x, time, model_kwargs={}):
        """使用无分类器引导（Classifier-free guidance）的前向传播。model_kwargs 应包含 `cfg_scale`。"""

        cfg_scale = model_kwargs.pop("cfg_scale")
        print("Classifier-free guidance scale:", cfg_scale)
        model_kwargs = copy.deepcopy(model_kwargs)

        ##################################################################
        # TODO: 应用论文 https://arxiv.org/pdf/2207.12598 中的公式 (6) 来实现无分类器引导，即：
        # x = (scale + 1) * eps(x_t, cond) - scale * eps(x_t, empty)
        #
        # 你将需要调用两次 self.forward。
        # 对于无条件采样（unconditional sampling），请在 `text_emb` 中传入 None。
        ##################################################################
        eps_cond = self.forward(x, time, model_kwargs)

        uncond_kwargs = copy.deepcopy(model_kwargs)
        uncond_kwargs["text_emb"] = None
        eps_empty = self.forward(x, time, uncond_kwargs)

        x = (cfg_scale + 1) * eps_cond - cfg_scale * eps_empty
        ##################################################################

        return x

    def forward(self, x, time, model_kwargs={}):
        """通过 U-Net 的前向传播。
        参数:
            x: 形状为 (batch_size, channels, height, width) 的输入张量。
            time: 形状为 (batch_size,) 的时间步张量。
            model_kwargs: 包含额外模型输入的字典，其中包括形状为 (batch_size, condition_dim) 的 "text_emb"（文本嵌入）。

        返回:
            x: 形状为 (batch_size, channels, height, width) 的输出张量。
        """

        if "cfg_scale" in model_kwargs:
            return self.cfg_forward(x, time, model_kwargs)

        # 嵌入时间步 (Embed time step)
        context = self.time_mlp(time)

        # 嵌入条件并添加到上下文中 (Embed condition and add to context)
        cond_emb = model_kwargs["text_emb"]
        if cond_emb is None:
            cond_emb = torch.zeros(x.shape[0], self.condition_dim, device=x.device)
        if self.training:
            # 在训练时随机丢弃条件
            mask = (torch.rand(cond_emb.shape[0]) > self.uncond_prob).float()
            mask = mask[:, None].to(cond_emb.device)  # B x 1
            cond_emb = cond_emb * mask
        context = context + self.condition_mlp(cond_emb)

        # 初始卷积
        x = self.init_conv(x)

        ##################################################################
        # TODO: 将 `x` 通过受上下文条件约束的 U-Net 处理。
        #
        # 1. 下采样 (Downsampling):
        #    - 将 `x` 以及上下文传递给每个下采样块进行处理。
        #    - 在经过每个 ResNet 块之后，将输出（特征图）保存在列表或字典中，
        #      以便在上采样路径中作为跳跃连接（skip connections）使用。
        #    - 确保将上下文 (context) 传递给每个 ResNet 块。
        #
        # 2. 中间部分 (Middle):
        #    - 将 `x` 以及上下文传递给中间块进行处理。
        #
        # 3. 上采样 (Upsampling):
        #    - 将 `x` 以及上下文传递给每个上采样块进行处理。
        #    - 在进入每个 ResNet 块之前，将输入与下采样路径中对应的跳跃连接特征进行拼接（concatenate）。
        #    - 确保将上下文 (context) 传递给每个 ResNet 块。
        ##################################################################
        skips = []

        # 1. 下采样 (Downsampling)
        for down_block in self.downs:
            res1, res2, down = down_block
            x = res1(x, context)
            skips.append(x)
            x = res2(x, context)
            skips.append(x)
            x = down(x)

        # 2. 中间部分 (Middle)
        x = self.mid_block1(x, context)
        x = self.mid_block2(x, context)

        # 3. 上采样 (Upsampling)
        for up_block in self.ups:
            upsample, res1, res2 = up_block
            x = upsample(x)

            skip = skips.pop()
            x = torch.cat((x, skip), dim=1)
            x = res1(x, context)

            skip = skips.pop()
            x = torch.cat((x, skip), dim=1)
            x = res2(x, context)
        ##################################################################

        # 最后的卷积块
        x = self.final_conv(x)

        return x
