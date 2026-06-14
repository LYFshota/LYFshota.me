import torch
import torch.nn as nn
from tqdm.auto import tqdm
import math


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        objective="pred_noise",
        beta_schedule="sigmoid",
    ):
        super().__init__()

        self.model = model
        self.channels = 3
        self.image_size = image_size
        self.objective = objective
        assert objective in {
            "pred_noise",
            "pred_x_start",
        }, "objective must be either pred_noise (predict noise) or pred_x_start (predict image start)"

        # 这是一个辅助函数，用于将一些常量注册为 buffer（缓冲区），以确保
        # 它们与模型参数位于同一个设备（device）上。
        # 参见 https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        # 每个 buffer 都可以通过 `self.name` 来访问
        register_buffer = lambda name, val: self.register_buffer(name, val.float())

        #############################################################################
        # 噪声调度（Noise schedule）的 beta 和 alpha 值
        #############################################################################
        betas = get_beta_schedule(beta_schedule, timesteps)
        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # alpha_bar_t
        register_buffer("betas", betas)  # can be accessed as self.betas
        register_buffer("alphas", alphas)  # can be accessed as self.alphas
        register_buffer("alphas_cumprod", alphas_cumprod)  # self.alphas_cumprod

        #############################################################################
        # 在 x_t、x_0 和噪声（noise）之间转换所需的其他系数
        # 注意，根据公式 (4) 及其在公式 (14) 中的重参数化：
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        # 其中 noise 从标准正态分布 N(0, 1) 中采样
        #############################################################################
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        # register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        # register_buffer(
        #     "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        # )

        #############################################################################
        # 用于后验分布 q(x_{t-1} | x_t, x_0)，根据论文中的公式 (6) 和 (7)。
        #############################################################################
        # alpha_bar_{t-1}
        alphas_cumprod_prev = nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_std = torch.sqrt(posterior_var.clamp(min=1e-20))
        register_buffer("posterior_std", posterior_std)

        #################################################################
        # 损失函数权重（loss weight）
        #################################################################
        snr = alphas_cumprod / (1 - alphas_cumprod)
        loss_weight = torch.ones_like(snr) if objective == "pred_noise" else snr
        register_buffer("loss_weight", loss_weight)

    def normalize(self, img):
        return img * 2 - 1

    def unnormalize(self, img):
        return (img + 1) * 0.5

    def predict_start_from_noise(self, x_t, t, noise):
        """根据论文中的公式 (14)，利用 x_t 和 noise 计算出 x_start。
        参数：
            x_t: 形状为 (b, *) 的张量。加噪后的图像。
            t: 形状为 (b,) 的张量。时间步。
            noise: 形状为 (b, *) 的张量。来自 N(0, 1) 的噪声。
        返回：
            x_start: 形状为 (b, *) 的张量。起始图像（即 x_0）。
        """
        ####################################################################
        # 待办（TODO）:
        # 根据公式 (4) 和公式 (14)，对 x_t 和 noise 进行变换以获得 x_start。
        # 请查看 `__init__` 方法中的系数，并使用 `extract` 函数。
        ####################################################################
        sqrt_alphas_cumprod = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        x_start = (x_t - sqrt_one_minus_alphas_cumprod * noise) / sqrt_alphas_cumprod
        ####################################################################
        return x_start

    def predict_noise_from_start(self, x_t, t, x_start):
        """根据论文中的公式 (14)，利用 x_t 和 x_start 计算出 noise。
        参数：
            x_t: 形状为 (b, *) 的张量。加噪后的图像。
            t: 形状为 (b,) 的张量。时间步。
            x_start: 形状为 (b, *) 的张量。起始图像（即 x_0）。
        返回：
            pred_noise: 形状为 (b, *) 的张量。预测的噪声。
        """
        ####################################################################
        # 待办（TODO）:
        # 根据公式 (4) 和公式 (14)，对 x_t 和 x_start 进行变换以获得预测的 noise。
        # 请查看 `__init__` 方法中的系数，并使用 `extract` 函数。
        ####################################################################
        sqrt_alphas_cumprod = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        pred_noise = (x_t - sqrt_alphas_cumprod * x_start) / sqrt_one_minus_alphas_cumprod
        ####################################################################
        return pred_noise

    def q_posterior(self, x_start, x_t, t):
        """根据论文中的公式 (6) 和 (7) 获取后验分布 q(x_{t-1} | x_t, x_0)。
        参数：
            x_start: 形状为 (b, *) 的张量。预测的起始图像。
            x_t: 形状为 (b, *) 的张量。加噪后的图像。
            t: 形状为 (b,) 的张量。时间步。
        返回：
            posterior_mean: 形状为 (b, *) 的张量。后验分布的均值。
            posterior_std: 形状为 (b, *) 的张量。后验分布的标准差。
        """
        posterior_mean = None
        posterior_std = None
        ####################################################################
        # 我们已经为你实现了这个方法。
        c1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        c2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = c1 * x_start + c2 * x_t
        posterior_std = extract(self.posterior_std, t, x_t.shape)
        ####################################################################
        return posterior_mean, posterior_std

    @torch.no_grad()
    def p_sample(self, x_t, t: int, model_kwargs={}):
        """根据论文中的公式 (6)，从 p(x_{t-1} | x_t) 中采样。仅在推理期间使用。
        参数：
            x_t: 形状为 (b, *) 的张量。加噪后的图像。
            t: 整数。采样的时间步。
            model_kwargs: 模型的额外参数。
        返回：
            x_tm1: 形状为 (b, *) 的张量。采样得到的图像（即 x_{t-1}）。
        """
        t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)  # (b,)
        ##################################################################
        # 待办（TODO）: 根据公式 (6) 实现采样步骤 p(x_{t-1} | x_t)：
        #
        # - 步骤：
        #   1. 通过传递合适的参数调用 self.model 获取模型预测结果。
        #   2. 根据 self.objective 的不同，模型的输出可能是 noise 也可能是 x_start。
        #      你可以根据需要调用 self.predict_start_from_noise 或
        #      self.predict_noise_from_start 来恢复另一个值。
        #   3. 将预测得到的 x_start 限制（clamp）在有效范围 [-1, 1] 内。
        #      这可以确保在去噪迭代过程中生成保持稳定。
        #   4. 使用 self.q_posterior 获取 q(x_{t-1} | x_t, x_0) 的均值和标准差，
        #      并采样出 x_{t-1}。
        ##################################################################
        model_out = self.model(x_t, t, model_kwargs=model_kwargs)
        if self.objective == "pred_noise":
            x_start = self.predict_start_from_noise(x_t, t, model_out)
        elif self.objective == "pred_x_start":
            x_start = model_out
            
        x_start = torch.clamp(x_start, -1.0, 1.0)
        posterior_mean, posterior_std = self.q_posterior(x_start, x_t, t)
        
        noise = torch.randn_like(x_t) if t[0].item() > 0 else 0.0
        x_tm1 = posterior_mean + posterior_std * noise
        ##################################################################

        return x_tm1

    @torch.no_grad()
    def sample(self, batch_size=16, return_all_timesteps=False, model_kwargs={}):

        shape = (batch_size, self.channels, self.image_size, self.image_size)
        img = torch.randn(shape, device=self.betas.device)
        imgs = [img]

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img = self.p_sample(img, t, model_kwargs=model_kwargs)
            imgs.append(img)

        res = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        res = self.unnormalize(res)
        return res

    def q_sample(self, x_start, t, noise):
        """根据论文中的公式 (4)，从 q(x_t | x_0) 中采样。

        参数：
            x_start: 形状为 (b, *) 的张量。起始图像（即 x_0）。
            t: 形状为 (b,) 的张量。时间步。
            noise: 形状为 (b, *) 的张量。来自 N(0, 1) 的噪声。
        返回：
            x_t: 形状为 (b, *) 的张量。加噪后的图像。
        """

        ####################################################################
        # 待办（TODO）:
        # 根据论文中的公式 (4)，实现从 q(x_t | x_0) 的采样。
        # 提示：(1) 查看 `__init__` 方法以获取预先计算好的系数。
        # (2) 使用上面定义的 `extract` 函数来提取给定时间步 `t` 的系数。
        # (3) 回想一下，从 N(mu, sigma^2) 中采样可以通过这种方式完成：
        # x_t = mu + sigma * noise，其中 noise 是从 N(0, 1) 采样的。
        # 大约需要 3 行代码。
        ####################################################################
        sqrt_alphas_cumprod = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        x_t = sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise
        ####################################################################
        return x_t

    def p_losses(self, x_start, model_kwargs={}):
        b, nts = x_start.shape[0], self.num_timesteps
        t = torch.randint(0, nts, (b,), device=x_start.device).long()  # (b,)
        x_start = self.normalize(x_start)  # (b, *)
        noise = torch.randn_like(x_start)  # (b, *)
        target = noise if self.objective == "pred_noise" else x_start  # (b, *)
        loss_weight = extract(self.loss_weight, t, target.shape)  # (b, *)
        ####################################################################
        # 待办（TODO）:
        # 根据论文中的公式 (14) 实现损失函数。
        # 首先，使用 `q_sample` 函数从 q(x_t | x_0) 中采样 x_t。
        # 然后，通过使用适当的参数调用 self.model 获取模型预测。
        # 最后，计算加权的均方误差（MSE）损失。
        # 大约需要 3-4 行代码。
        ####################################################################
        x_t = self.q_sample(x_start, t, noise)
        model_out = self.model(x_t, t, model_kwargs=model_kwargs)
        loss = torch.mean(loss_weight * (model_out - target) ** 2)
        ####################################################################

        return loss


def extract(a, t, x_shape):
    """
    根据给定的时间步提取合适的系数值。

    该函数根据给定的时间步 `t` 从系数张量 `a` 中收集值，
    并将其重塑（reshape）为匹配所需的形状，以便支持与给定形状 `x_shape` 的张量进行广播。

    参数：
        a (torch.Tensor): 形状为 (T,) 的张量，包含所有时间步的系数值。
        t (torch.Tensor): 形状为 (b,) 的张量，表示批次中每个样本的时间步。
        x_shape (tuple): 输入图像张量的形状，通常为 (b, c, h, w)。

    返回：
        torch.Tensor: 形状为 (b, 1, 1, 1) 的张量，包含为批次中每个元素对应的
                      时间步从 a 中提取出来的系数值，并相应地进行了重塑。
    """
    b, *_ = t.shape  # 从时间步张量中提取批次大小 (batch size)
    out = a.gather(-1, t)  # 根据时间步 `t` 从系数 `a` 中收集值
    out = out.reshape(
        b, *((1,) * (len(x_shape) - 1))
    )  # 重塑为 (b, 1, 1, 1) 以便进行广播 (broadcasting)
    return out


def linear_beta_schedule(timesteps):
    """
    线性调度（linear schedule），在原始 ddpm 论文中提出
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    余弦调度（cosine schedule）
    在 https://openreview.net/forum?id=-NEXDKk8gZ 中提出
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    Sigmoid 调度（sigmoid schedule）
    在 https://arxiv.org/abs/2212.11972 - Figure 8 中提出
    在训练期间使用时，对于大于 64x64 的图像效果更好
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def get_beta_schedule(beta_schedule, timesteps):
    if beta_schedule == "linear":
        beta_schedule_fn = linear_beta_schedule
    elif beta_schedule == "cosine":
        beta_schedule_fn = cosine_beta_schedule
    elif beta_schedule == "sigmoid":
        beta_schedule_fn = sigmoid_beta_schedule
    else:
        raise ValueError(f"unknown beta schedule {beta_schedule}")

    betas = beta_schedule_fn(timesteps)
    return betas
