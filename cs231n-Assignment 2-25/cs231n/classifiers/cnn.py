from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    一个三层卷积神经网络，具有以下网络架构：

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    该网络在形状为 (N, C, H, W) 的小批量数据（minibatch）上运行，
    包含 N 张图像，每张图像的高度为 H，宽度为 W，且具有 C 个输入通道。
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        初始化一个新的网络。

        输入：
        - input_dim: 元组 (C, H, W)，给出输入数据的尺寸
        - num_filters: 卷积层中使用的滤波器（卷积核）数量
        - filter_size: 卷积层中使用的滤波器（卷积核）的宽度/高度
        - hidden_dim: 全连接隐藏层中使用的单元数量
        - num_classes: 最终仿射（全连接）层输出的得分数量
        - weight_scale: 标量，给出权重随机初始化的标准差
        - reg: 标量，给出 L2 正则化强度
        - dtype: 用于计算的 numpy 数据类型。
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: 初始化三层卷积神经网络的权重和偏置。                               #
        # 权重应从以 0.0 为中心、标准差等于 weight_scale 的高斯分布中进行初始化；     #
        # 偏置应初始化为零。所有的权重和偏置都应存储在 self.params 字典中。         #
        # 使用键 'W1' 和 'b1' 存储卷积层的权重和偏置；使用键 'W2' 和 'b2'           #
        # 存储隐藏仿射层的权重和偏置，使用键 'W3' 和 'b3' 存储输出仿射层的权重和偏置。#
        #                                                                          #
        # 重要提示：对于本次作业，你可以假设第一层卷积层的填充和步幅被选择为       #
        # **输入的高度和宽度得以保留**。请查看 loss() 函数的开头，了解这是如何实现的。 #
        ############################################################################

        C, H, W = input_dim

        # W1: 卷积层权重，形状 (num_filters, C, filter_size, filter_size)
        # b1: 卷积层偏置，形状 (num_filters,)
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)

        # 卷积后空间尺寸保持 H x W（由 pad 和 stride=1 保证），
        # 经过 2x2 max pool (stride=2) 后变为 H/2 x W/2
        # W2: 隐藏全连接层权重，形状 (num_filters * H/2 * W/2, hidden_dim)
        # b2: 隐藏全连接层偏置，形状 (hidden_dim,)
        self.params['W2'] = weight_scale * np.random.randn(num_filters * H * W // 4, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)

        # W3: 输出全连接层权重，形状 (hidden_dim, num_classes)
        # b3: 输出全连接层偏置，形状 (num_classes,)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE (代码结束)                  #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        评估三层卷积神经网络的损失和梯度。

        输入 / 输出：与 fc_net.py 中的 TwoLayerNet 具有相同的 API。
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # 将 conv_param 传递给卷积层的前向传播
        # 选择的填充和步幅以保留输入的空间尺寸大小
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # 将 pool_param 传递给最大池化层的前向传播
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: 实现三层卷积神经网络的前向传播，计算 X 的类别得分并将其存储在       #
        # scores 变量中。                                                           #
        #                                                                          #
        # 请记住，你可以在实现中使用在 cs231n/fast_layers.py 和                      #
        # cs231n/layer_utils.py 中定义好的函数（已导入）。                          #
        ############################################################################

        # 第一层：conv - relu - 2x2 max pool
        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)

        # 第二层：affine - relu
        out2, cache2 = affine_relu_forward(out1, W2, b2)

        # 第三层：affine（输出层）
        scores, cache3 = affine_forward(out2, W3, b3)

        ############################################################################
        #                             END OF YOUR CODE (代码结束)                  #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: 实现三层卷积神经网络的反向传播，将损失和梯度存储在 loss 和 grads   #
        # 变量中。使用 softmax 计算数据损失，并确保 grads[k] 保存 self.params[k]  #
        # 的梯度。别忘了加上 L2 正则化！                                           #
        #                                                                          #
        # 注意：为了确保你的实现与我们的一致，并能通过自动化测试，请确保你的 L2     #
        # 正则化包含一个 0.5 的系数，以简化梯度的表达式。                           #
        ############################################################################

        # 计算 softmax 数据损失
        loss, dscores = softmax_loss(scores, y)

        # 加上 L2 正则化损失（包含 0.5 系数）
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))

        # 第三层反向传播：affine（输出层）
        dout2, dW3, db3 = affine_backward(dscores, cache3)
        grads['W3'] = dW3 + self.reg * W3
        grads['b3'] = db3

        # 第二层反向传播：affine - relu
        dout1, dW2, db2 = affine_relu_backward(dout2, cache2)
        grads['W2'] = dW2 + self.reg * W2
        grads['b2'] = db2

        # 第一层反向传播：conv - relu - pool
        dX, dW1, db1 = conv_relu_pool_backward(dout1, cache1)
        grads['W1'] = dW1 + self.reg * W1
        grads['b1'] = db1

        ############################################################################
        #                             END OF YOUR CODE (代码结束)                  #
        ############################################################################

        return loss, grads
