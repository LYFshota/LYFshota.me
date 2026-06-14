import numpy as np
import copy

import torch
import torch.nn as nn

from ..transformer_layers import *


class CaptioningTransformer(nn.Module):
    """
    CaptioningTransformer 使用 Transformer 解码器从图像特征生成字幕。

    该 Transformer 接收大小为 D 的输入向量，词汇表大小为 V，
    处理长度为 T 的序列，使用维度为 W 的词向量，
    并且在大小为 N 的小批量（minibatch）上运行。
    """
    def __init__(self, word_to_idx, input_dim, wordvec_dim, num_heads=4,
                 num_layers=2, max_length=50):
        """
        构建一个新的 CaptioningTransformer 实例。

        输入：
        - word_to_idx：包含词汇表的字典。它包含 V 个条目，
          并将每个字符串映射到范围 [0, V) 内的唯一整数。
        - input_dim：输入图像特征向量的维度 D。
        - wordvec_dim：词向量的维度 W。
        - num_heads：注意力头（attention heads）的数量。
        - num_layers：Transformer 层的数量。
        - max_length：最大可能的序列长度。
        """
        super().__init__()

        vocab_size = len(word_to_idx)
        self.vocab_size = vocab_size
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        self.visual_projection = nn.Linear(input_dim, wordvec_dim)
        self.embedding = nn.Embedding(vocab_size, wordvec_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(wordvec_dim, max_len=max_length)

        decoder_layer = TransformerDecoderLayer(input_dim=wordvec_dim, num_heads=num_heads)
        self.transformer = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.apply(self._init_weights)

        self.output = nn.Linear(wordvec_dim, vocab_size)

    def _init_weights(self, module):
        """
        初始化网络的权重。
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, features, captions):
        """
        给定图像特征和字幕词元（caption tokens），返回每个时间步上可能词元的概率分布。
        注意，由于整个字幕序列是一次性提供的，我们需要屏蔽（mask out）未来的时间步。

        输入：
         - features：图像特征，形状为 (N, D)
         - captions：真实字幕（ground truth captions），形状为 (N, T)

        返回：
         - scores：每个时间步每个词元的得分，形状为 (N, T, V)
        """
        N, T = captions.shape
        # 创建一个占位符，将被你下面的代码覆盖。
        scores = torch.empty((N, T, self.vocab_size))
        ############################################################################
        # TODO: 实现 CaptioningTransformer 的 forward 函数。                       #
        # 一些提示：                                                               #
        #  1) 你首先需要对字幕进行嵌入（embed）并添加位置编码。然后你需要将图像    #
        #     特征投影到相同的维度。                                               #
        #  2) 你需要准备一个掩码（tgt_mask），用于屏蔽字幕中未来的时间步。         #
        #     torch.tril() 函数可能有助于准备这个掩码。                            #
        #  3) 最后，将解码器特征应用于文本和图像嵌入，并结合 tgt_mask。将输出      #
        #     投影为每个词元的得分（scores）。                                     #
        ############################################################################
        tgt = self.embedding(captions)
        tgt = self.positional_encoding(tgt)
        
        memory = self.visual_projection(features).unsqueeze(1)
        
        tgt_mask = torch.tril(torch.ones((T, T), device=features.device))
        
        out = self.transformer(tgt, memory, tgt_mask=tgt_mask)
        
        scores = self.output(out)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return scores

    def sample(self, features, max_length=30):
        """
        给定图像特征，使用贪心解码（greedy decoding）来预测图像字幕。

        输入：
         - features：图像特征，形状为 (N, D)
         - max_length：最大可能的字幕长度

        返回：
         - captions：每个样本的字幕，形状为 (N, max_length)
        """
        with torch.no_grad():
            features = torch.Tensor(features)
            N = features.shape[0]

            # 创建一个空的字幕张量（其中所有词元都是 NULL）。
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # 创建一个部分字幕，仅包含开始词元（start token）。
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption)
            # [N] -> [N, 1]
            partial_caption = partial_caption.unsqueeze(1)

            for t in range(max_length):

                # 预测下一个词元（忽略所有其他时间步）。
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]

                # 从词汇表中选择最可能的单词 ID。
                # [N, V] -> [N]
                word = torch.argmax(output_logits, dim=1)

                # 更新我们的整体字幕和当前的部分字幕。
                captions[:, t] = word.numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)

            return captions


def clones(module, N):
    "生成 N 个相同的层。"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask)

        return output


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, src_mask=None):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=src_mask)

        return output



class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) 的实现。
    """
    def __init__(self, img_size=32, patch_size=8, in_channels=3,
                 embed_dim=128, num_layers=6, num_heads=4,
                 dim_feedforward=256, num_classes=10, dropout=0.1):
        """
        输入：
         - img_size：输入图像的尺寸（假设为正方形）。
         - patch_size：每个图块（patch）的尺寸（假设为正方形）。
         - in_channels：图像通道数。
         - embed_dim：每个图块的嵌入维度。
         - num_layers：Transformer 编码器（encoder）层的数量。
         - num_heads：注意力头的数量。
         - dim_feedforward：前馈网络（feedforward network）的隐藏层大小。
         - num_classes：分类标签的数量。
         - dropout：Dropout 概率。
        """
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)

        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 最后的分类层，根据池化后的词元预测类别得分。
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        """
        初始化网络的权重。
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Vision Transformer 的前向传播。

        输入：
         - x：输入图像张量，形状为 (N, C, H, W)

        返回：
         - logits：输出的分类 logits，形状为 (N, num_classes)
        """
        N = x.size(0)
        logits = torch.zeros(N, self.num_classes, device=x.device)
        
        ############################################################################
        # TODO: 实现 Vision Transformer 的前向传播。                               #
        # 1. 将输入图像转换为一系列图块（patch）向量的序列。                       #
        # 2. 添加位置编码以保留空间信息。                                          #
        # 3. 将该序列传递给 Transformer 编码器。                                   #
        # 4. 对图块向量进行平均池化（Average pool），得到每张图像的特征向量。      #
        #    你可能会发现 torch.mean 很有用。                                      #
        # 5. 将其输入到线性层以产生类别 logits。                                   #
        ############################################################################
        x = self.patch_embed(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        logits = self.head(x)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


        return logits
