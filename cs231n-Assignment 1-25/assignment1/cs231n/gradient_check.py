from __future__ import print_function
from builtins import range
from past.builtins import xrange

import numpy as np
from random import randrange


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    一个简单的数值梯度实现，计算 f 在 x 处的梯度
    - f 应该是一个只接受一个参数的函数
    - x 是计算梯度的点（numpy 数组）
    """

    fx = f(x)  # 在原始点计算函数值
    grad = np.zeros_like(x)
    # 遍历 x 中的所有索引
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:

        # 在 x+h 处计算函数
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # 增加 h
        fxph = f(x)  # 计算 f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # 计算 f(x - h)
        x[ix] = oldval  # 恢复

        # 使用中心差分公式计算偏导数
        grad[ix] = (fxph - fxmh) / (2 * h)  # 斜率
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # 步进到下一个维度

    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    为一个接受 numpy 数组并返回 numpy 数组的函数计算数值梯度。
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
    """
    计算操作于输入和输出 blob 的函数的数值梯度。

    我们要假设 f 接受几个输入 blob 作为参数，后面跟着一个输出将被写入的 blob。
    例如，f 可能像这样调用：

    f(x, w, out)

    其中 x 和 w 是输入 blob，f 的结果将被写入 out。

    输入：
    - f: 函数
    - inputs: 输入 blob 的元组
    - output: 输出 blob
    - h: 步长
    """
    numeric_diffs = []
    for input_blob in inputs:
        diff = np.zeros_like(input_blob.diffs)
        it = np.nditer(input_blob.vals, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = input_blob.vals[idx]

            input_blob.vals[idx] = orig + h
            f(*(inputs + (output,)))
            pos = np.copy(output.vals)
            input_blob.vals[idx] = orig - h
            f(*(inputs + (output,)))
            neg = np.copy(output.vals)
            input_blob.vals[idx] = orig

            diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)

            it.iternext()
        numeric_diffs.append(diff)
    return numeric_diffs


def eval_numerical_gradient_net(net, inputs, output, h=1e-5):
    return eval_numerical_gradient_blobs(
        lambda *args: net.forward(), inputs, output, h=h
    )


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    随机抽样几个元素并在这些维度上仅返回数值梯度。
    """

    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h  # 增加 h
        fxph = f(x)  # 计算 f(x + h)
        x[ix] = oldval - h  # 减少 h
        fxmh = f(x)  # 计算 f(x - h)
        x[ix] = oldval  # 重置

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (
            abs(grad_numerical) + abs(grad_analytic)
        )
        print(
            "数值梯度: %f 解析梯度: %f, 相对误差: %e"
            % (grad_numerical, grad_analytic, rel_error)
        )
