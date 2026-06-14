import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
此文件定义了通常用于 Transformer 的层类型。
"""

class PositionalEncoding(nn.Module):
    """
    编码序列中词元(token)位置的信息。在这种情况下，
    该层没有可学习的参数，因为它是一个简单的正弦和余弦函数。
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        构建 PositionalEncoding（位置编码）层。

        输入:
         - embed_dim: 嵌入维度的尺寸
         - dropout: dropout 的值
         - max_len: 输入序列的最大可能长度
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # 创建一个“批次维度”为 1 的数组（它将在批次中的
        # 所有样本上进行广播）。
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO: 按照 Transformer_Captioning.ipynb 中的描述构建位置编码数组。       #
        # 目标是使每一行交替出现正弦和余弦，并且具有 0, 0, 2, 2, 4, 4 等指数，     #
        # 一直到 embed_dim。当然，这个确切的规范有些随意，但这是自动评分器所期望的。#
        # 作为参考，我们的解决方案少于 5 行代码。                                  #
        ############################################################################
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        ############################################################################
        #                             你的代码结束                                 #
        ############################################################################

        # 确保位置编码将与模型参数一起保存
        # （主要是为了完整性）。
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        将位置嵌入逐元素地加到输入序列中。

        输入:
         - x: 馈入位置编码器模型的序列，形状为
              (N, S, D)，其中 N 是批次大小，S 是序列长度，
              D 是嵌入维度
        返回:
         - output: 输入序列 + 位置编码，形状为 (N, S, D)
        """
        N, S, D = x.shape
        # 创建一个占位符，将被下面你的代码覆盖。
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO: 索引到你的位置编码数组中，并将合适的位置编码                       #
        # 加到输入序列中。之后不要忘记应用 dropout。这只需要几行代码。               #
        ############################################################################
        output = x + self.pe[:, :S, :]
        output = self.dropout(output)
        ############################################################################
        #                             你的代码结束                                 #
        ############################################################################
        return output


class MultiHeadAttention(nn.Module):
    """
    一个实现了简化版掩码注意力机制的模型层，
    如 "Attention Is All You Need" (https://arxiv.org/abs/1706.03762) 中所介绍的。

    用法:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # 自注意力 (self-attention)
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # 使用两个输入的注意力
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        构建一个新的 MultiHeadAttention（多头注意力）层。

        输入:
         - embed_dim: 词元(token)嵌入的维度
         - num_heads: 注意力头的数量
         - dropout: Dropout 概率
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # 我们将为您初始化这些层，因为交换顺序
        # 会影响随机数的生成（从而影响您相对于自动评分器的精确输出）。
        # 注意，这些层使用了偏置项，但这并不是绝对必要的
        # （并且因实现而异）。
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        计算所提供数据的掩码注意力输出，并行计算所有注意力头。

        在下面的形状定义中，N 是批次大小，S 是源序列长度，T 是目标序列长度，
        而 E 是嵌入维度。

        输入:
        - query: 用作查询(query)的输入数据，形状为 (N, S, E)
        - key: 用作键(key)的输入数据，形状为 (N, T, E)
        - value: 用作值(value)的输入数据，形状为 (N, T, E)
        - attn_mask: 形状为 (S, T) 的数组，其中 mask[i,j] == 0 表示源序列中的
          词元 i 不应影响目标序列中的词元 j。

        返回:
        - output: 形状为 (N, S, E) 的张量，给出了根据使用键和查询计算出的
          注意力权重对值(value)中的数据进行加权组合的结果。
        """
        N, S, E = query.shape
        N, T, E = value.shape
        # 创建一个占位符，将被下面你的代码覆盖。
        output = torch.empty((N, S, E))
        ############################################################################
        # TODO: 使用 Transformer_Captioning.ipynb 中给出的方程实现多头注意力机制。 #
        # 一些提示：                                                               #
        #  1) 你需要将形状从 (N, T, E) 拆分为 (N, T, H, E/H)，                     #
        #     其中 H 是注意力头的数量。                                            #
        #  2) torch.matmul 函数允许你进行批量矩阵乘法。                            #
        #     例如，你可以将 (N, H, T, E/H) 乘以 (N, H, E/H, T) 从而产生           #
        #     形状 (N, H, T, T)。更多示例请参见：                                  #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) 关于应用 attn_mask，请思考应如何修改分数以阻止某个值影响输出。       #
        #     具体来说，PyTorch 的 masked_fill 函数可能会派上用场。                #
        ############################################################################
        H = self.n_head
        D = self.head_dim

        # 1. 线性映射并分离出各个注意力头
        # query, key, value 经过各自的 Linear 层后形状变为 (N, L, E)
        # 通过 view 重塑为 (N, L, H, D)，然后 transpose 为 (N, H, L, D) 方便后续矩阵乘法
        q = self.query(query).view(N, S, H, D).transpose(1, 2)
        k = self.key(key).view(N, T, H, D).transpose(1, 2)
        v = self.value(value).view(N, T, H, D).transpose(1, 2)

        # 2. 计算缩放点积注意力 (Scaled Dot-Product Attention)
        # q: (N, H, S, D), k 的转置: (N, H, D, T)
        # 结果 scores 的形状为 (N, H, S, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)

        # 3. 应用注意力掩码 (Attention Mask)
        if attn_mask is not None:
            # attn_mask 形状为 (S, T), 使用 masked_fill_ 将 mask 为 0 的位置填充为 -inf
            # 这样在 softmax 后这些位置的权重就会接近 0
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        # 4. 计算注意力权重 (Softmax) 并应用 dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # 5. 使用注意力权重对 value 进行加权求和
        # attn_weights: (N, H, S, T), v: (N, H, T, D)
        # 结果 context 的形状为 (N, H, S, D)
        context = torch.matmul(attn_weights, v)

        # 6. 将各个头的输出拼接起来并恢复原本形状
        # 将 context 形状从 (N, H, S, D) 转置回 (N, S, H, D)，然后 view 重塑回 (N, S, E)
        # 注意: 转置后内存不连续，需要用 contiguous() 才能调用 view()
        context = context.transpose(1, 2).contiguous().view(N, S, E)

        # 7. 通过最后的线性映射层
        output = self.proj(context)
        ############################################################################
        #                             你的代码结束                                 #
        ############################################################################
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        """
        带有 dropout 和 ReLU 激活函数的简单双层前馈网络。

        输入:
         - embed_dim: 输入和输出嵌入的维度
         - ffn_dim: 前馈网络中的隐藏层维度
         - dropout: Dropout 概率
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        """
        前馈网络的前向传播。

        输入:
        - x: 形状为 (N, T, D) 的输入张量

        返回:
        - out: 与输入形状相同的输出张量
        """
        out = torch.empty_like(x)

        out = self.fc1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class TransformerDecoderLayer(nn.Module):
    """
    Transformer 解码器的单层，用于与 TransformerDecoder 配合使用。
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        构建一个 TransformerDecoderLayer 实例。

        输入:
         - input_dim: 输入中预期的特征数量。
         - num_heads: 注意力头的数量。
         - dim_feedforward: 前馈网络模型的维度。
         - dropout: dropout 的值。
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(input_dim, dim_feedforward, dropout)

        self.norm_self = nn.LayerNorm(input_dim)
        self.norm_cross = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)


    def forward(self, tgt, memory, tgt_mask=None):
        """
        将输入（和掩码）传递过解码器层。

        输入:
        - tgt: 输入到解码器层的序列，形状为 (N, T, D)
        - memory: 来自编码器最后一层的序列，形状为 (N, S, D)
        - tgt_mask: 目标序列中要掩码的部分，形状为 (T, T)

        返回:
        - out: Transformer 特征，形状为 (N, T, W)
        """

        # 自注意力块（参考实现）
        shortcut = tgt
        tgt = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask)
        tgt = self.dropout_self(tgt)
        tgt = tgt + shortcut
        tgt = self.norm_self(tgt)

        ############################################################################
        # TODO: 通过实现剩余的两个子层来完成解码器层：                             #
        # (1) 使用编码器输出作为 memory 的交叉注意力块，以及 (2) 前馈网络块。      #
        # 每个块应该遵循与上面刚刚实现的自注意力相同的结构。                       #
        ############################################################################
        
        # (1) 交叉注意力块
        shortcut = tgt
        tgt = self.cross_attn(query=tgt, key=memory, value=memory, attn_mask=None)
        tgt = self.dropout_cross(tgt)
        tgt = tgt + shortcut
        tgt = self.norm_cross(tgt)

        # (2) 前馈网络块
        shortcut = tgt
        tgt = self.ffn(tgt)
        tgt = self.dropout_ffn(tgt)
        tgt = tgt + shortcut
        tgt = self.norm_ffn(tgt)
        ############################################################################
        #                             你的代码结束                                 #
        ############################################################################

        return tgt


class PatchEmbedding(nn.Module):
    """
    将图像分割成小块(patch)并将每个小块投影到嵌入向量的层。
    用作视觉 Transformer (ViT) 的输入层。

    输入:
    - img_size: 表示输入图像高度/宽度的整数（假设为方形图像）。
    - patch_size: 表示每个小块高度/宽度的整数（方形小块）。
    - in_channels: 输入图像通道数（例如，RGB 为 3）。
    - embed_dim: 线性嵌入空间的维度。
    """
    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=128):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels

        # 将展平的小块线性投影到嵌入维度
        self.proj = nn.Linear(self.patch_dim, embed_dim)


    def forward(self, x):
        """
        小块嵌入(patch embedding)的前向传播。

        输入:
        - x: 形状为 (N, C, H, W) 的输入图像张量

        返回:
        - out: 小块嵌入，形状为 (N, num_patches, embed_dim)
        """
        N, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Expected image size ({self.img_size}, {self.img_size}), but got ({H}, {W})"
        out = torch.zeros(N, self.embed_dim)

        ############################################################################
        # TODO: 将图像划分为形状为 (C x patch_size x patch_size) 的非重叠小块，    #
        # 并将它们重新排列为形状为 (N, num_patches, patch_dim) 的张量。不要使用    #
        # for 循环。相反，你可能会发现 torch.reshape 和 torch.permute 对这一步   #
        # 有帮助。一旦小块被展平，就使用投影层将它们嵌入到隐向量中。               #
        ############################################################################
        P = self.patch_size
        H_p, W_p = H // P, W // P
        
        x_reshaped = x.reshape(N, C, H_p, P, W_p, P)
        x_permuted = x_reshaped.permute(0, 2, 4, 1, 3, 5)
        x_flattened = x_permuted.reshape(N, self.num_patches, self.patch_dim)
        
        out = self.proj(x_flattened)
        ############################################################################
        #                             你的代码结束                                 #
        ############################################################################
        return out




class TransformerEncoderLayer(nn.Module):
    """
    Transformer 编码器的单层，用于与 TransformerEncoder 配合使用。
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        构建一个 TransformerEncoderLayer 实例。

        输入:
         - input_dim: 输入中预期的特征数量。
         - num_heads: 注意力头的数量。
         - dim_feedforward: 前馈网络模型的维度。
         - dropout: dropout 的值。
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(input_dim, dim_feedforward, dropout)

        self.norm_self = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        将输入（和掩码）传递过编码器层。

        输入:
        - src: 输入到编码器层的序列，形状为 (N, S, D)
        - src_mask: 源序列中要掩码的部分，形状为 (S, S)

        返回:
        - out: Transformer 特征，形状为 (N, S, D)
        """
        ############################################################################
        # TODO: 通过应用自注意力然后接一个前馈网络块来实现编码器层。               #
        # 这段代码将与解码器层非常相似。                                           #
        ############################################################################
        
        # 自注意力块
        shortcut = src
        src = self.self_attn(query=src, key=src, value=src, attn_mask=src_mask)
        src = self.dropout_self(src)
        src = src + shortcut
        src = self.norm_self(src)

        # 前馈网络块
        shortcut = src
        src = self.ffn(src)
        src = self.dropout_ffn(src)
        src = src + shortcut
        src = self.norm_ffn(src)

        ############################################################################
        #                             你的代码结束                                 #
        ############################################################################
        return src
