from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax损失函数，朴素实现（带循环）。

    输入的维度为D，有C个类别，我们对N个样本的小批量进行操作。

    输入：
    - W: 一个形状为(D, C)的numpy数组，包含权重。
    - X: 一个形状为(N, D)的numpy数组，包含一个小批量的数据。
    - y: 一个形状为(N,)的numpy数组，包含训练标签；y[i] = c表示X[i]的标签为c，其中0 <= c < C。
    - reg: (float) 正则化强度

    返回一个元组：
    - 损失为一个浮点数
    - 相对于权重W的梯度；一个与W形状相同的数组
    """
    # 初始化损失和梯度为零。
    loss = 0.0
    dW = np.zeros_like(W)

    # 计算损失和梯度
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # 以数值稳定的方式计算概率
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # 归一化

        # 计算损失
        loss -= np.log(p[y[i]])

        # 计算梯度
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += (p[j] - 1) * X[i]
            else:
                dW[:, j] += p[j] * X[i]

    # 归一化损失和梯度
    loss /= num_train
    dW /= num_train

    # 正则化
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax损失函数，向量化版本。

    输入和输出与softmax_loss_naive相同。
    """
    # 初始化损失和梯度为零。
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]

    # 1. 计算分数 (N, C)
    scores = X.dot(W)
    
    # 2. 数值稳定性处理：每行减去该行的最大值
    scores -= np.max(scores, axis=1, keepdims=True)
    
    # 3. 计算 Softmax 概率
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # (N, C)

    # 4. 计算损失
    # 选取正确类别的概率
    correct_class_probs = probs[np.arange(num_train), y]
    loss = np.sum(-np.log(correct_class_probs))
    loss /= num_train
    loss += reg * np.sum(W * W) # 正则化

    # 5. 计算梯度
    # 根据推导，dL/ds = p - 1 (正确类别) 或 p (错误类别)
    dscores = probs.copy()
    dscores[np.arange(num_train), y] -= 1 # 只有正确类别的梯度项需要减1
    
    # dL/dW = X^T * dL/ds
    dW = X.T.dot(dscores)
    dW /= num_train
    dW += 2 * reg * W # 正则化梯度

    return loss, dW
