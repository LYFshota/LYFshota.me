from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """多层全连接神经网络类。

    该网络包含任意数量的隐藏层、ReLU 非线性激活函数以及一个 softmax 损失函数。
    网络还可以选择实现 dropout 以及批量/层归一化（batch/layer normalization）。
    对于一个拥有 L 层的网络，其架构将是：

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    其中批量/层归一化和 dropout 是可选的，并且 {...} 块重复 L - 1 次。

    可学习的参数存储在 self.params 字典中，并将使用 Solver 类进行训练学习。
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """初始化一个新的 FullyConnectedNet (全连接网络)。

        输入：
        - hidden_dims: 一个整数列表，给出每个隐藏层的大小。
        - input_dim: 一个整数，给出输入数据的大小。
        - num_classes: 一个整数，给出要分类的类别数量。
        - dropout_keep_ratio: 介于 0 和 1 之间的标量，表示 dropout 强度（保留神经元的比例）。
            如果 dropout_keep_ratio=1，则网络完全不使用 dropout。
        - normalization: 网络应使用哪种类型的归一化。有效的值
            为 "batchnorm"、"layernorm" 或用 None 表示不进行归一化（默认值）。
        - reg: 一个标量，给出 L2 正则化的强度。
        - weight_scale: 一个标量，给出随机初始化权重时的标准差。
        - dtype: 一个 numpy 的数据类型对象；所有计算都将使用此数据类型执行。
            float32 速度更快但不太精确，因此进行数值梯度检查时应该使用 float64。
        - seed: 如果不为 None，则将此随机种子传递给 dropout 层。
            这将使 dropout 层具有确定性，从而能对模型进行梯度检查。
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: 初始化网络的参数，将所有值存储在 self.params 字典中。              #
        # 将第一层的权重和偏置存储在 W1 和 b1 中；对于第二层使用 W2 和 b2，依此类推。#
        # 权重应从以 0 为中心、标准差等于 weight_scale 的正态分布中初始化。        #
        # 偏置应初始化为零。                                                       #
        #                                                                          #
        # 使用批量归一化时，将第一层的缩放和平移参数分别存储在                     #
        # gamma1 和 beta1 中；第二层使用 gamma2 和 beta2，依此类推。               #
        # 缩放参数（scale）应初始化为 1，平移参数（shift）应初始化为 0。           #
        ############################################################################
        
        # 把所有层的维度（维度数组）串联在一起以便用循环赋值
        layer_dims = [input_dim] + hidden_dims + [num_classes]
        
        for i in range(1, self.num_layers + 1):
            # 依均值0，标准差 weight_scale 初始化 W；b 初始化为 0
            self.params[f'W{i}'] = np.random.normal(0, weight_scale, (layer_dims[i-1], layer_dims[i]))
            self.params[f'b{i}'] = np.zeros(layer_dims[i])
            
            # 若不是最后一层且使用了归一化，分配 gamma 和 beta
            if self.normalization is not None and i < self.num_layers:
                self.params[f'gamma{i}'] = np.ones(layer_dims[i])
                self.params[f'beta{i}'] = np.zeros(layer_dims[i])

        ############################################################################
        #                             END OF YOUR CODE (代码结束)                  #
        ############################################################################

        # 当使用 dropout 时，我们需要将 dropout_param 字典传递给每个
        # dropout 层，以便该层知道 dropout 概率（保留比例）以及当前模式
        # （train / test）。你可以将同一个 dropout_param 传递给每一个 dropout 层。
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # 使用批量归一化时，我们需要跟踪运行时的均值和方差，
        # 因此我们需要将一个特殊的 bn_param 对象传递给每个批量归一化层。
        # 你应该将 self.bn_params[0] 传递给第一层批量归一化的前向传播，
        # 将 self.bn_params[1] 传递给第二层批量归一化的前向传播，依此类推。
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.ln_params = [{} for i in range(self.num_layers - 1)]

        # 将所有参数转换为正确的数据类型。
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """计算全连接网络的损失和梯度。"""
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # 为 batchnorm 参数和 dropout 参数设置 train/test 模式
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

        scores = None


        curr_X = X
        self.caches = {} # 统一使用字典进行层号标记存储


        for i in range(1, self.num_layers):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']


            if self.normalization == "batchnorm":
                gamma = self.params[f'gamma{i}']
                beta = self.params[f'beta{i}']

                curr_X, layer_cache = affine_bn_relu_forward(
                    curr_X, W, b, gamma, beta, self.bn_params[i-1]
                )
                self.caches[i] = layer_cache
                
            elif self.normalization == "layernorm":
                gamma = self.params[f'gamma{i}']
                beta = self.params[f'beta{i}']

                curr_X, fc_cache = affine_forward(curr_X, W, b)
                curr_X, norm_cache = layernorm_forward(curr_X, gamma, beta, self.ln_params[i-1])
                curr_X, relu_cache = relu_forward(curr_X)
                self.caches[i] = (fc_cache, norm_cache, relu_cache)
                
            else:

                curr_X, layer_cache = affine_relu_forward(curr_X, W, b)
                self.caches[i] = layer_cache

            if self.use_dropout:
                curr_X, drop_cache = dropout_forward(curr_X, self.dropout_param)

                self.caches[f'drop_cache{i}'] = drop_cache


        W_last = self.params[f'W{self.num_layers}']
        b_last = self.params[f'b{self.num_layers}']
        scores, out_cache = affine_forward(curr_X, W_last, b_last)
        self.caches[f'fc_cache{self.num_layers}'] = out_cache

        if mode == "test":
            return scores

        loss, grads = 0.0, {}

        loss, dscores = softmax_loss(scores, y)

        for i in range(1, self.num_layers + 1):
            W = self.params[f'W{i}']
            loss += 0.5 * self.reg * np.sum(W * W)


        dout, dw, db = affine_backward(dscores, self.caches[f'fc_cache{self.num_layers}'])
        grads[f'W{self.num_layers}'] = dw + self.reg * self.params[f'W{self.num_layers}']
        grads[f'b{self.num_layers}'] = db

        for i in range(self.num_layers - 1, 0, -1):
            

            if self.use_dropout:
                dout = dropout_backward(dout, self.caches[f'drop_cache{i}'])


            if self.normalization == "batchnorm":
                dout, dw, db, dgamma, dbeta = affine_bn_relu_backward(dout, self.caches[i])
                grads[f'W{i}'] = dw + self.reg * self.params[f'W{i}']
                grads[f'b{i}'] = db
                grads[f'gamma{i}'] = dgamma
                grads[f'beta{i}'] = dbeta
                
            elif self.normalization == "layernorm":

                fc_cache, norm_cache, relu_cache = self.caches[i]
                dout = relu_backward(dout, relu_cache)
                dout, dgamma, dbeta = layernorm_backward(dout, norm_cache)
                dout, dw, db = affine_backward(dout, fc_cache)
                grads[f'W{i}'] = dw + self.reg * self.params[f'W{i}']
                grads[f'b{i}'] = db
                grads[f'gamma{i}'] = dgamma
                grads[f'beta{i}'] = dbeta
                
            else:
                dout, dw, db = affine_relu_backward(dout, self.caches[i])
                grads[f'W{i}'] = dw + self.reg * self.params[f'W{i}']
                grads[f'b{i}'] = db

        return loss, grads
