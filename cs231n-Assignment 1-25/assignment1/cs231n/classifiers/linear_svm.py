from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    结构化 SVM 损失函数，朴素实现（带循环）。

    输入维度为 D，有 C 个类别，我们在 N 个样本的小批量上进行操作。

    输入:
    - W: 一个形状为 (D, C) 的 numpy 数组，包含权重。
    - X: 一个形状为 (N, D) 的 numpy 数组，包含一个小批量的数据。
    - y: 一个形状为 (N,) 的 numpy 数组，包含训练标签；y[i] = c 表示
      X[i] 的标签是 c，其中 0 <= c < C。
    - reg: (float) 正则化强度

    返回一个元组：
    - loss：一个浮点数，表示损失
    - dW：关于权重 W 的梯度；形状与 W 相同的数组
    """
    dW = np.zeros(W.shape)  # 将梯度初始化为零

    # 计算损失和梯度
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        
        # 记录大于 0 的 margin 数量，用于计算正确类别的梯度
        diff_count = 0
        
        for j in range(num_classes):
            if j == y[i]:
                continue
            
            margin = scores[j] - correct_class_score + 1  
            if margin > 0:
                loss += margin
                diff_count += 1
                
                # 计算错误类别的梯度
                # dL/dw_j = x_i
                dW[:, j] += X[i]
                
        # 计算正确类别的梯度
        # dL/dw_{y_i} = - (margin > 0 的类别数量) * x_i
        dW[:, y[i]] -= diff_count * X[i]

    # 现在损失是所有训练样本的总和，但我们要的是平均值
    # 所以我们要除以 num_train。
    loss /= num_train
    dW /= num_train # 梯度也要取平均

    # 将正则化添加到损失中。
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W # 添加正则化梯度

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    结构化 SVM 损失函数，向量化实现。

    输入和输出与 svm_loss_naive 相同。
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # 将梯度初始化为零

    #############################################################################
    # TODO:                                                                     #
    # 实现结构化 SVM 损失的向量化版本，并将结果存储在 loss 中。                   #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = X.dot(W)  # (N, C)
    
    # 计算 margins (N, C)
    scores = X.dot(W)
    correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)
    
    # broadcasting
    margins = np.maximum(0, scores - correct_class_scores + 1)
    
    # 正确类别的 margin 应该为 0，因为在上面的公式中，当 j=y_i 时，
    # score[y_i] - score[y_i] + 1 = 1 > 0，这会被计入 loss，
    # 但实际上根据公式 j != y_i，我们不应该计算正确类别。
    margins[np.arange(num_train), y] = 0
    
    loss = np.sum(margins) / num_train
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # 实现结构化 SVM 损失梯度的向量化版本，并将结果存储在 dW 中。                 #
    #                                                                           #
    # 提示：与其从头计算梯度，不如复用您在计算损失时使用的一些中间值更加容易。        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 计算梯度
    # 根据损失函数的推导：
    # 对于 margin > 0 的错误类别 j，导数是 X[i]
    # 对于 margin > 0 的正确类别 y[i]，导数是 - (diff_count) * X[i]
    
    # 1. 构造一个 binary 矩阵 (N, C)，其中的元素指示是否贡献梯度
    binary = margins
    binary[margins > 0] = 1 
    
    # 2. 计算每一行的 margin > 0 数量 (即 diff_count)
    row_sum = np.sum(binary, axis=1) # (N,)

    # 3. 将正确类别的 binary 值设为 -row_sum
    binary[np.arange(num_train), y] = -row_sum

    # 4. 计算 dW = X^T * binary
    # (D, N) * (N, C) = (D, C)
    dW = X.T.dot(binary)

    # 5. 平均梯度和添加正则化梯度
    dW /= num_train
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
