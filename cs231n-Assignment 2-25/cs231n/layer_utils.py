from .layers import *
from .fast_layers import *


def affine_relu_forward(x, w, b):
    """
    一个便捷层，执行仿射变换后跟一个 ReLU 激活

    输入:
    - x: 仿射层的输入
    - w, b: 仿射层的权重

    返回元组:
    - out: ReLU 的输出
    - cache: 传给反向传播的对象
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """
    affine-relu 便捷层的反向传播
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def conv_relu_forward(x, w, b, conv_param):
    """
    一个便捷层，执行卷积后跟一个 ReLU 激活。

    输入:
    - x: 卷积层的输入
    - w, b, conv_param: 卷积层的权重和参数

    返回元组:
    - out: ReLU 的输出
    - cache: 传给反向传播的对象
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    conv-relu 便捷层的反向传播。
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    一个便捷层，执行卷积、ReLU 和池化。

    输入:
    - x: 卷积层的输入
    - w, b, conv_param: 卷积层的权重和参数
    - pool_param: 池化层的参数

    返回元组:
    - out: 池化层的输出
    - cache: 传给反向传播的对象
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    conv-relu-pool 便捷层的反向传播
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    高级便捷层：融合了 仿射(Affine) -> 批量归一化(BatchNorm) -> ReLU
    """
    # 1. 过全连接层
    a, fc_cache = affine_forward(x, w, b)
    # 2. 紧接着在 ReLU 前面插入批量归一化层
    an, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    # 3. 最后过 ReLU 非线性激活
    out, relu_cache = relu_forward(an)
    
    # 将三个层的小账本打包装进总 cache
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

def affine_bn_relu_backward(dout, cache):
    """
    高级便捷层对应的反向传播：严格镜像倒推
    """
    fc_cache, bn_cache, relu_cache = cache
    
    # 1. 倒序第一步：反推 ReLU
    dan = relu_backward(dout, relu_cache)
    # 2. 倒序第二步：反推 批量归一化 (这里拿到了 dgamma 和 dbeta)
    da, dgamma, dbeta = batchnorm_backward_alt(dan, bn_cache)
    # 3. 倒序第三步：反推 全连接 (拿到输入梯度 dx 以及参数梯度 dw, db)
    dx, dw, db = affine_backward(da, fc_cache)
    
    return dx, dw, db, dgamma, dbeta