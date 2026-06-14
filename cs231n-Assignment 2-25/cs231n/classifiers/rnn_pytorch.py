import numpy as np
import torch
from ..rnn_layers_pytorch import *


class CaptioningRNN:
    """
    CaptioningRNN 使用循环神经网络从图像特征生成字幕。

    RNN 接收大小为 D 的输入向量，词汇表大小为 V，处理长度为 T 的序列，
    RNN 隐藏层维度为 H，使用维度为 W 的词向量，并在大小为 N 的小批量上操作。

    请注意，我们不对 CaptioningRNN 使用任何正则化。
    """

    def __init__(
        self,
        word_to_idx,
        input_dim=512,
        wordvec_dim=128,
        hidden_dim=128,
        cell_type="rnn",
        dtype=torch.float32,
    ):
        """
        构建一个新的 CaptioningRNN 实例。

        输入：
        - word_to_idx: 包含词汇表的字典。它包含 V 个条目，
          并将每个字符串映射到 [0, V) 范围内的一个唯一整数。
        - input_dim: 输入图像特征向量的维度 D。
        - wordvec_dim: 词向量的维度 W。
        - hidden_dim: RNN 隐藏状态的维度 H。
        - cell_type: 使用哪种类型的 RNN；'rnn' 或 'lstm'。
        - dtype: 要使用的 numpy 数据类型；训练时使用 float32，
          进行数值梯度检查时使用 float64。
        """
        if cell_type not in {"rnn", "lstm"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        # 初始化词向量
        self.params["W_embed"] = torch.randn(vocab_size, wordvec_dim)
        self.params["W_embed"] /= 100

        # 初始化 CNN -> 隐藏状态投影参数
        self.params["W_proj"] = torch.randn(input_dim, hidden_dim)
        self.params["W_proj"] /= np.sqrt(input_dim)
        self.params["b_proj"] = torch.zeros(hidden_dim)

        # 初始化 RNN 的参数
        dim_mul = {"lstm": 4, "rnn": 1}[cell_type]
        self.params["Wx"] = torch.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params["Wx"] /= np.sqrt(wordvec_dim)
        self.params["Wh"] = torch.randn(hidden_dim, dim_mul * hidden_dim)
        self.params["Wh"] /= np.sqrt(hidden_dim)
        self.params["b"] = torch.zeros(dim_mul * hidden_dim)

        # 初始化输出到词汇表的权重
        self.params["W_vocab"] = torch.randn(hidden_dim, vocab_size)
        self.params["W_vocab"] /= np.sqrt(hidden_dim)
        self.params["b_vocab"] = torch.zeros(vocab_size)

        # 将参数转换为正确的数据类型
        for k, v in self.params.items():
            self.params[k] = v.to(self.dtype)

    def loss(self, features, captions):
        """
        计算 RNN 的训练时损失。我们输入图像特征和这些图像的
        真实字幕（ground-truth），并使用 RNN（或 LSTM）来计算
        所有参数的损失和梯度。

        输入：
        - features: 输入图像特征，形状为 (N, D)
        - captions: 真实字幕；一个形状为 (N, T + 1) 的整数数组，其中
          每个元素的范围是 0 <= y[i, t] < V

        返回以下元组：
        - loss: 标量损失
        """
        # 将 captions 切割成两部分：captions_in 包含除最后一个词以外的所有词，
        # 并将作为 RNN 的输入；captions_out 包含除第一个词以外的所有词，
        # 这是我们期望 RNN 生成的内容。它们彼此偏移了一个位置，
        # 因为 RNN 在接收到单词 t 后应该产生单词 (t+1)。
        # captions_in 的第一个元素将是 START 标记，而 captions_out 的第一个元素将是第一个词。
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # 你会用到这个
        mask = captions_out != self._null

        # 从图像特征到初始隐藏状态的仿射变换的权重和偏置
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]

        # 词嵌入矩阵
        W_embed = self.params["W_embed"]

        # RNN 的输入到隐藏层、隐藏层到隐藏层的权重，以及偏置
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]

        # 隐藏层到词汇表变换的权重和偏置。
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        loss = 0.0
        ############################################################################
        # TODO: 实现 CaptioningRNN 的前向传播。                                    #
        # 在前向传播中，你需要执行以下操作：                                       #
        # (1) 使用仿射变换从图像特征计算初始隐藏状态。                             #
        #     这将产生一个形状为 (N, H) 的数组。                                   #
        # (2) 使用词嵌入层将 captions_in 中的单词从索引转换为向量，                #
        #     得到一个形状为 (N, T, W) 的数组。                                    #
        # (3) 根据 self.cell_type，使用普通的 RNN 或 LSTM 处理                     #
        #     输入词向量序列，并为所有时间步生成隐藏状态向量，                     #
        #     产生一个形状为 (N, T, H) 的数组。                                    #
        # (4) 使用（时间）仿射变换，利用隐藏状态计算每个时间步在词汇表上的得分，   #
        #     得到一个形状为 (N, T, V) 的数组。                                    #
        # (5) 使用 captions_out 计算（时间）softmax 损失，并利用上面的 mask 忽略   #
        #     输出词为 <NULL> 的位置。                                             #
        #                                                                          #
        # 请确保你的实现与输入张量的数据类型无关。                                 #
        #                                                                          #
        # 不要担心对权重或它们的梯度进行正则化！                                   #
        #                                                                          #
        # 你也不需要实现反向传播。                                                 #
        ############################################################################

        # (1) 计算初始隐藏状态 h0
        h0 = affine_forward(features, W_proj, b_proj)
        
        # (2) 词嵌入层
        x = word_embedding_forward(captions_in, W_embed)
        
        # (3) RNN / LSTM 处理
        if self.cell_type == 'rnn':
            h = rnn_forward(x, h0, Wx, Wh, b)
        elif self.cell_type == 'lstm':
            h = lstm_forward(x, h0, Wx, Wh, b)
            
        # (4) 时序仿射变换计算得分
        scores = temporal_affine_forward(h, W_vocab, b_vocab)
        
        # (5) 计算损失
        loss = temporal_softmax_loss(scores, captions_out, mask)

        ############################################################################
        #                             你的代码结束                                 #
        ############################################################################

        return loss

    def sample(self, features, max_length=30):
        """
        运行模型的测试时前向传播，对输入特征向量采样字幕。

        在每个时间步，我们嵌入当前词，将它和上一个隐藏状态传递给 RNN
        以获得下一个隐藏状态，利用隐藏状态获得所有词汇的分数，并选择
        得分最高的词作为下一个词。初始隐藏状态是通过对输入图像特征应用
        仿射变换计算得到的，初始词是 <START> 标记。

        对于 LSTM，你还需要跟踪细胞状态 (cell state)；在这种情况下，
        初始的细胞状态应为全零。

        输入：
        - features: 形状为 (N, D) 的输入图像特征数组。
        - max_length: 生成的字幕的最大长度 T。

        返回：
        - captions: 形状为 (N, max_length) 的数组，给出采样的字幕，
          其中每个元素是 [0, V) 范围内的整数。captions 的第一个元素
          应该是采样的第一个词，而不是 <START> 标记。
        """
        N = features.shape[0]
        captions = self._null * torch.ones((N, max_length), dtype=torch.long)

        # 解包参数
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]
        W_embed = self.params["W_embed"]
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        ###########################################################################
        # TODO: 实现模型的测试时采样。你需要通过对输入图像特征应用学习到的        #
        # 仿射变换来初始化 RNN 的隐藏状态。你喂给 RNN 的第一个词应该是 <START>      #
        # 标记；它的值存储在变量 self._start 中。在每个时间步，你需要执行以下操作： #
        # (1) 使用学习到的词嵌入来嵌入上一个词                                      #
        # (2) 使用上一个隐藏状态和当前词的嵌入向量执行一步 RNN 计算，               #
        #     以获得下一个隐藏状态。                                                #
        # (3) 对下一个隐藏状态应用学习到的仿射变换，以获得词汇表中所有词的得分      #
        # (4) 选择得分最高的词作为下一个词，将其（词索引）写入 captions 变量        #
        #     的适当位置。                                                          #
        #                                                                         #
        # 为了简单起见，你不需要在采样到 <END> 标记后停止生成，                     #
        # 但如果你愿意也可以这么做。                                                #
        #                                                                         #
        # 提示：你将无法使用 rnn_forward 或 lstm_forward 函数；                     #
        # 你需要在循环中调用 rnn_step_forward 或 lstm_step_forward。                #
        #                                                                         #
        # 注意：在这个函数中，我们仍在对小批量(minibatch)进行操作。                 #
        # 此外，如果你正在使用 LSTM，请将第一个细胞状态(cell state)初始化为零。     #
        ###########################################################################

        # 初始化隐藏状态 h0
        prev_h = affine_forward(features, W_proj, b_proj)
        
        # LSTM 的初始细胞状态
        if self.cell_type == 'lstm':
            prev_c = torch.zeros_like(prev_h)
            
        # 初始词 <START>
        prev_word = torch.full((N,), self._start, dtype=torch.long, device=features.device)
        
        for t in range(max_length):
            # (1) 嵌入上一个词
            x = word_embedding_forward(prev_word, W_embed)
            
            # (2) 执行 RNN/LSTM 一步计算
            if self.cell_type == 'rnn':
                next_h = rnn_step_forward(x, prev_h, Wx, Wh, b)
            elif self.cell_type == 'lstm':
                next_h, prev_c = lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)
                
            # (3) 应用仿射变换获得分数
            scores = affine_forward(next_h, W_vocab, b_vocab)
            
            # (4) 选择得分最高的词
            next_word = torch.argmax(scores, dim=1)
            captions[:, t] = next_word
            
            # 更新状态为下一步准备
            prev_h = next_h
            prev_word = next_word

        ############################################################################
        #                             你的代码结束                                 #
        ############################################################################
        return captions
