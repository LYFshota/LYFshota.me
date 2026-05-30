from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """一个使用L2距离的k近邻分类器"""

    def __init__(self):
        pass

    def train(self, X, y):
        """
        训练分类器。对于k近邻，这只是记住训练数据。

        输入：
        - X: 一个形状为(num_train, D)的numpy数组，包含训练数据，由num_train个样本组成，每个样本的维度为D。
        - y: 一个形状为(N,)的numpy数组，包含训练标签，其中y[i]是X[i]的标签。
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        使用此分类器预测测试数据的标签。

        输入：
        - X: 一个形状为(num_test, D)的numpy数组，包含测试数据，由num_test个样本组成，每个样本的维度为D。
        - k: 投票预测标签的最近邻数量。
        - num_loops: 决定使用哪种实现来计算训练点和测试点之间的距离。

        返回：
        - y: 一个形状为(num_test,)的numpy数组，包含测试数据的预测标签，其中y[i]是测试点X[i]的预测标签。
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        使用嵌套循环计算X中每个测试点与self.X_train中每个训练点之间的距离。

        输入：
        - X: 一个形状为(num_test, D)的numpy数组，包含测试数据。

        返回：
        - dists: 一个形状为(num_test, num_train)的numpy数组，其中dists[i, j]是第i个测试点与第j个训练点之间的欧几里得距离。
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################

                dists[i, j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))

        return dists

    def compute_distances_one_loop(self, X):
        """
        使用单个循环计算X中每个测试点与self.X_train中每个训练点之间的距离。

        输入/输出：与compute_distances_two_loops相同
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):  
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # 利用广播机制 (Broadcasting) 计算第 i 个测试点与所有训练点的差
            # X[i] 形状是 (D,), self.X_train 形状是 (num_train, D)
            # X[i] - self.X_train 会自动把 X[i] 扩充成 (num_train, D) 进行减法
            # 然后对每一行（每个训练样本）计算平方和再开根号
            # axis=1 表示沿着特征维度求和
            dists[i, :] = np.sqrt(np.sum((self.X_train - X[i]) ** 2, axis=1))

        return dists

    def compute_distances_no_loops(self, X):
        """
        不使用显式循环计算X中所有测试点与self.X_train中所有训练点之间的距离。

        输入/输出：与compute_distances_two_loops相同
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################

        test_sum_squares = np.sum(X ** 2, axis=1, keepdims=True)
    
        train_sum_squares = np.sum(self.X_train ** 2, axis=1)

        two_XY = 2 * np.dot(X, self.X_train.T)
        

        dists = np.sqrt(test_sum_squares + train_sum_squares - two_XY)

        return dists

    def predict_labels(self, dists, k=1):
        """
        给定测试点和训练点之间的距离矩阵，为每个测试点预测一个标签。

        输入：
        - dists: 一个形状为(num_test, num_train)的numpy数组，其中dists[i, j]给出第i个测试点与第j个训练点之间的距离。

        返回：
        - y: 一个形状为(num_test,)的numpy数组，包含测试数据的预测标签，其中y[i]是测试点X[i]的预测标签。
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []

            # 1. 找到第 i 个测试样本与所有训练样本的距离，并按从小到大排序，返回排序后的索引
            sorted_indices = np.argsort(dists[i])
            
            # 2. 取前 k 个最小距离对应的训练样本的索引
            closest_k_indices = sorted_indices[:k]
            
            # 3. 根据索引从 self.y_train 中取出对应的标签
            closest_y = self.y_train[closest_k_indices]

            # 使用 np.bincount 统计每个标签出现的次数 (类似于投票)
            # 然后用 np.argmax 找出出现次数最多的那个标签
            y_pred[i] = np.argmax(np.bincount(closest_y))


        return y_pred
