from __future__ import print_function

import os
from builtins import range
from builtins import object
import numpy as np
from ..classifiers.linear_svm import *
from ..classifiers.softmax import *
from past.builtins import xrange


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=True,
    ):
        """
        使用随机梯度下降训练此线性分类器。

        输入：
        - X: 一个形状为(N, D)的numpy数组，包含训练数据；有N个训练样本，每个样本的维度为D。
        - y: 一个形状为(N,)的numpy数组，包含训练标签；y[i] = c表示X[i]的标签为0 <= c < C，其中C是类别数。
        - learning_rate: (float) 优化的学习率。
        - reg: (float) 正则化强度。
        - num_iters: (integer) 优化时的迭代步数。
        - batch_size: (integer) 每步使用的训练样本数量。
        - verbose: (boolean) 如果为True，则在优化过程中打印进度。

        输出：
        一个包含每次训练迭代的损失函数值的列表。
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # 运行随机梯度下降以优化W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # 从训练数据中采样batch_size个元素及其对应的标签，用于本轮梯度下降。          #
            # 将数据存储在X_batch中，对应的标签存储在y_batch中；采样后X_batch应具有形状   #
            # (batch_size, dim)，y_batch应具有形状(batch_size,)。                     #
            # 提示：使用np.random.choice生成索引。带替换采样比不带替换采样更快。          #
            #########################################################################
            # 随机选择索引
            indices = np.random.choice(num_train, batch_size, replace=True)
            # 根据索引构建 batch
            X_batch = X[indices]
            y_batch = y[indices]
            # 评估损失和梯度
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # 执行参数更新
            #########################################################################
            # TODO:                                                                 #
            # 使用梯度和学习率更新权重。                                             #
            #########################################################################
            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        使用此线性分类器的训练权重预测数据点的标签。

        输入：
        - X: 一个形状为(N, D)的numpy数组，包含训练数据；有N个训练样本，每个样本的维度为D。

        返回：
        - y_pred: 数据X的预测标签。y_pred是一个长度为N的一维数组，每个元素是一个整数，表示预测的类别。
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)

        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        计算损失函数及其导数。
        子类将覆盖此方法。

        输入：
        - X_batch: 一个形状为(N, D)的numpy数组，包含一个N个数据点的小批量；每个点的维度为D。
        - y_batch: 一个形状为(N,)的numpy数组，包含小批量的标签。
        - reg: (float) 正则化强度。

        返回：一个包含以下内容的元组：
        - 损失为一个浮点数
        - 相对于self.W的梯度；一个与W形状相同的数组
        """
        pass

    def save(self, fname):
        """保存模型参数。"""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = {"W": self.W}
        np.save(fpath, params)
        print(fname, "saved.")

    def load(self, fname):
        """加载模型参数。"""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.W = params["W"]
            print(fname, "loaded.")
            return True


class LinearSVM(LinearClassifier):
    """使用多类SVM损失函数的子类"""

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """使用Softmax + 交叉熵损失函数的子类"""

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
