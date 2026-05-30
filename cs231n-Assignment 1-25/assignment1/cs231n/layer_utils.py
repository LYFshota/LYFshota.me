from .layers import *


def affine_relu_forward(x, w, b):
    """
    便捷层，执行仿射变换后接 ReLU

    输入:
    - x: 仿射层的输入
    - w, b: 仿射层的权重

    返回一个元组:
    - out: ReLU 的输出
    - cache: 传递给反向传播的对象
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    仿射-ReLU 便捷层的反向传播
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

