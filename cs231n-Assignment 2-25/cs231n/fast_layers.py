from __future__ import print_function
import numpy as np

try:
    from .im2col_cython import col2im_cython, im2col_cython
    from .im2col_cython import col2im_6d_cython
except ImportError:
    pass
    # print("""=========== 如果你没有在做 ConvolutionalNetworks.ipynb，可以安全地忽略下面的消息 ===========""")
    # print("\t在本次作业的一部分中，你将需要编译一个 Cython 扩展。")
    # print("\t完成此操作的说明将在下面 notebook 的某个部分中给出。")

from .im2col import *


def conv_forward_im2col(x, w, b, conv_param):
    """
    基于 im2col 和 col2im 的卷积层前向传播的快速实现。
    """
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param["stride"], conv_param["pad"]

    # 检查维度
    assert (W + 2 * pad - filter_width) % stride == 0, "width does not work"
    assert (H + 2 * pad - filter_height) % stride == 0, "height does not work"

    # 创建输出
    out_height = (H + 2 * pad - filter_height) // stride + 1
    out_width = (W + 2 * pad - filter_width) // stride + 1
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    # x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache


def conv_forward_strides(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param["stride"], conv_param["pad"]

    # 检查维度
    # assert (W + 2 * pad - WW) % stride == 0, 'width does not work'
    # assert (H + 2 * pad - HH) % stride == 0, 'height does not work'

    # 对输入进行填充
    p = pad
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

    # 计算输出维度
    H += 2 * pad
    W += 2 * pad
    out_h = (H - HH) // stride + 1
    out_w = (W - WW) // stride + 1

    # 通过选择巧妙的步幅（strides）来执行 im2col 操作
    shape = (C, HH, WW, N, out_h, out_w)
    strides = (H * W, W, 1, C * H * W, stride * W, stride)
    strides = x.itemsize * np.array(strides)
    x_stride = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
    x_cols = np.ascontiguousarray(x_stride)
    x_cols.shape = (C * HH * WW, N * out_h * out_w)

    # 现在我们所有的卷积都变成了一个大的矩阵乘法
    res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)

    # 重塑输出的形状
    res.shape = (F, N, out_h, out_w)
    out = res.transpose(1, 0, 2, 3)

    # 友好一点，返回一个连续的数组。
    # 旧版本的 conv_forward_fast 没有这样做，因此为了公平对比，我们也不做此处理。
    out = np.ascontiguousarray(out)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache


def conv_backward_strides(dout, cache):
    x, w, b, conv_param, x_cols = cache
    stride, pad = conv_param["stride"], conv_param["pad"]

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, out_h, out_w = dout.shape

    db = np.sum(dout, axis=(0, 2, 3))

    dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    dx_cols = w.reshape(F, -1).T.dot(dout_reshaped)
    dx_cols.shape = (C, HH, WW, N, out_h, out_w)
    dx = col2im_6d_cython(dx_cols, N, C, H, W, HH, WW, pad, stride)

    return dx, dw, db


def conv_backward_im2col(dout, cache):
    """
    基于 im2col 和 col2im 的卷积层反向传播的快速实现。
    """
    x, w, b, conv_param, x_cols = cache
    stride, pad = conv_param["stride"], conv_param["pad"]

    db = np.sum(dout, axis=(0, 2, 3))

    num_filters, _, filter_height, filter_width = w.shape
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)
    # dx = col2im_indices(dx_cols, x.shape, filter_height, filter_width, pad, stride)
    dx = col2im_cython(
        dx_cols,
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.shape[3],
        filter_height,
        filter_width,
        pad,
        stride,
    )

    return dx, dw, db


conv_forward_fast = conv_forward_strides
conv_backward_fast = conv_backward_strides


def max_pool_forward_fast(x, pool_param):
    """
    最大池化层前向传播的快速实现。

    该方法在重塑（reshape）方法和 im2col 方法之间进行选择。如果池化区域
    是正方形并且能够无重叠地平铺（tile）输入图像，那么我们可以使用
    非常快速的重塑方法。否则，我们将回退到 im2col 方法，该方法并不比
    朴素（naive）方法快多少。
    """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]

    same_size = pool_height == pool_width == stride
    tiles = H % pool_height == 0 and W % pool_width == 0
    if same_size and tiles:
        out, reshape_cache = max_pool_forward_reshape(x, pool_param)
        cache = ("reshape", reshape_cache)
    else:
        out, im2col_cache = max_pool_forward_im2col(x, pool_param)
        cache = ("im2col", im2col_cache)
    return out, cache


def max_pool_backward_fast(dout, cache):
    """
    最大池化层反向传播的快速实现。

    该方法根据生成缓存（cache）时所使用的方法，在重塑（reshape）方法和
    im2col 方法之间进行切换。
    """
    method, real_cache = cache
    if method == "reshape":
        return max_pool_backward_reshape(dout, real_cache)
    elif method == "im2col":
        return max_pool_backward_im2col(dout, real_cache)
    else:
        raise ValueError('Unrecognized method "%s"' % method)


def max_pool_forward_reshape(x, pool_param):
    """
    使用巧妙的形状重塑（reshaping）实现最大池化层前向传播的快速版本。

    该方法仅适用于能够无重叠平铺输入的正方形池化区域。
    """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]
    assert pool_height == pool_width == stride, "Invalid pool params"
    assert H % pool_height == 0
    assert W % pool_height == 0
    x_reshaped = x.reshape(
        N, C, H // pool_height, pool_height, W // pool_width, pool_width
    )
    out = x_reshaped.max(axis=3).max(axis=4)

    cache = (x, x_reshaped, out)
    return out, cache


def max_pool_backward_reshape(dout, cache):
    """
    使用巧妙的广播（broadcasting）和形状重塑（reshaping）实现最大池化层
    反向传播的快速版本。

    该方法仅在完成了使用 max_pool_forward_reshape 计算的前向传播时才可用。

    注意：如果存在多个最大值（argmax），此方法将把梯度分配给输入中的所有最大值
    元素，而不是仅选择其中一个。在这种情况下，计算出的梯度实际上是不正确的。
    然而，这在实际中极少发生，因此影响不大。一种可能的解决方案是将上游梯度
    平分给所有最大值元素，这会产生一个有效的次梯度（subgradient）。
    你可以通过取消注释下面的行来实现这一点；但这样做会导致显著的性能损失
    （变慢约 40%），且在实际中不太重要，因此我们不进行此处理。
    """
    x, x_reshaped, out = cache

    dx_reshaped = np.zeros_like(x_reshaped)
    out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
    mask = x_reshaped == out_newaxis
    dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
    dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
    dx_reshaped[mask] = dout_broadcast[mask]
    dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
    dx = dx_reshaped.reshape(x.shape)

    return dx


def max_pool_forward_im2col(x, pool_param):
    """
    基于 im2col 实现的最大池化前向传播。

    该方法并不比朴素版本快多少，因此应尽可能避免使用。
    """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]

    assert (H - pool_height) % stride == 0, "Invalid height"
    assert (W - pool_width) % stride == 0, "Invalid width"

    out_height = (H - pool_height) // stride + 1
    out_width = (W - pool_width) // stride + 1

    x_split = x.reshape(N * C, 1, H, W)
    x_cols = im2col(x_split, pool_height, pool_width, padding=0, stride=stride)
    x_cols_argmax = np.argmax(x_cols, axis=0)
    x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
    out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)

    cache = (x, x_cols, x_cols_argmax, pool_param)
    return out, cache


def max_pool_backward_im2col(dout, cache):
    """
    基于 im2col 实现的最大池化反向传播。

    该方法并不比朴素版本快多少，因此应尽可能避免使用。
    """
    x, x_cols, x_cols_argmax, pool_param = cache
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]

    dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
    dx_cols = np.zeros_like(x_cols)
    dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped
    dx = col2im_indices(
        dx_cols, (N * C, 1, H, W), pool_height, pool_width, padding=0, stride=stride
    )
    dx = dx.reshape(x.shape)

    return dx
