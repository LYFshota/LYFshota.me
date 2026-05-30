from __future__ import print_function, division
from future import standard_library

standard_library.install_aliases()
from builtins import range
from builtins import object
import os
import pickle as pickle

import numpy as np

from cs231n import optim


class Solver(object):
    """
    Solver 封装了训练分类模型所需的所有逻辑。
    Solver 使用 optim.py 中定义的多种更新规则执行随机梯度下降。

    Solver 接受训练和验证数据及其标签，以便定期在训练和验证数据上检查分类准确率，从而监控过拟合情况。

    要训练一个模型，您首先需要构建一个 Solver 实例，
    将模型、数据集和各种选项（学习率、批量大小等）传递给构造函数。
    然后调用 train() 方法运行优化过程并训练模型。

    train() 方法返回后，model.params 将包含在训练过程中在验证集上表现最好的参数。
    此外，实例变量 solver.loss_history 将包含训练期间遇到的所有损失值的列表，
    实例变量 solver.train_acc_history 和 solver.val_acc_history 将包含每个 epoch 在训练和验证集上的准确率列表。

    示例用法可能如下所示：

    data = {
      'X_train': # 训练数据
      'y_train': # 训练标签
      'X_val': # 验证数据
      'y_val': # 验证标签
    }
    model = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-4,
                    },
                    lr_decay=0.95,
                    num_epochs=5, batch_size=200,
                    print_every=100)
    solver.train()


    Solver 作用于一个必须符合以下 API 的模型对象：

    - model.params 必须是一个字典，将字符串参数名称映射到包含参数值的 numpy 数组。

    - model.loss(X, y) 必须是一个函数，用于计算训练时的损失和梯度，
      以及测试时的分类分数，具有以下输入和输出：

      输入：
      - X: 形状为 (N, d_1, ..., d_k) 的输入数据小批量数组
      - y: 形状为 (N,) 的标签数组，给出 X 的标签，其中 y[i] 是 X[i] 的标签。

      返回：
      如果 y 为 None，则运行测试时前向传播并返回：
      - scores: 形状为 (N, C) 的数组，给出 X 的分类分数，
        其中 scores[i, c] 给出 X[i] 的类别 c 的分数。

      如果 y 不为 None，则运行训练时前向和后向传播并返回一个元组：
      - loss: 给出损失的标量
      - grads: 与 self.params 具有相同键的字典，
        将参数名称映射到损失相对于这些参数的梯度。
    """

    def __init__(self, model, data, **kwargs):
        """
        构造一个新的 Solver 实例。

        必需参数：
        - model: 符合上述 API 描述的模型对象
        - data: 包含以下内容的训练和验证数据字典：
          'X_train': 形状为 (N_train, d_1, ..., d_k) 的训练图像数组
          'X_val': 形状为 (N_val, d_1, ..., d_k) 的验证图像数组
          'y_train': 形状为 (N_train,) 的训练图像标签数组
          'y_val': 形状为 (N_val,) 的验证图像标签数组

        可选参数：
        - update_rule: 指定 optim.py 中使用的更新规则名称的字符串。
          默认为 'sgd'。
        - optim_config: 包含超参数的字典，这些超参数将被传递给选定的更新规则。
          每个更新规则需要不同的超参数（参见 optim.py），
          但所有更新规则都需要 'learning_rate' 参数，因此该参数应该始终存在。
        - lr_decay: 学习率衰减的标量；每个 epoch 后学习率将乘以该值。
        - batch_size: 训练期间用于计算损失和梯度的小批量大小。
        - num_epochs: 训练期间运行的 epoch 数量。
        - print_every: 整数；每隔 print_every 次迭代打印一次训练损失。
        - verbose: 布尔值；如果设置为 false，则训练期间不会打印任何输出。
        - num_train_samples: 用于检查训练准确率的训练样本数；默认值为 1000；
          设置为 None 以使用整个训练集。
        - num_val_samples: 用于检查验证准确率的验证样本数；默认值为 None，
          使用整个验证集。
        - checkpoint_name: 如果不为 None，则每个 epoch 保存一次模型检查点。
        """
        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]

        # 解包关键字参数
        self.update_rule = kwargs.pop("update_rule", "sgd")
        self.optim_config = kwargs.pop("optim_config", {})
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.num_train_samples = kwargs.pop("num_train_samples", 1000)
        self.num_val_samples = kwargs.pop("num_val_samples", None)

        self.checkpoint_name = kwargs.pop("checkpoint_name", None)
        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)

        # 如果有额外的关键字参数则抛出错误
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        # 确保更新规则存在，然后用实际函数替换字符串名称
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self):
        """
        设置一些用于优化的簿记变量。不要手动调用此方法。
        """
        # 设置一些簿记变量
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # 为每个参数创建 optim_config 的深拷贝
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        """
        执行单次梯度更新。这由 train() 显式调用，不应手动调用。
        """
        # 生成一个小批量训练数据
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # 计算损失和梯度
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # 执行参数更新
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        checkpoint = {
            "model": self.model,
            "update_rule": self.update_rule,
            "lr_decay": self.lr_decay,
            "optim_config": self.optim_config,
            "batch_size": self.batch_size,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "epoch": self.epoch,
            "loss_history": self.loss_history,
            "train_acc_history": self.train_acc_history,
            "val_acc_history": self.val_acc_history,
        }
        filename = "%s_epoch_%d.pkl" % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        在提供的数据上检查模型的准确率。

        输入：
        - X: 数据数组，形状为 (N, d_1, ..., d_k)
        - y: 标签数组，形状为 (N,)
        - num_samples: 如果不为 None，则对数据进行采样，仅在 num_samples 个数据点上测试模型。
        - batch_size: 将 X 和 y 分成该大小的批次，以避免占用太多内存。

        返回：
        - acc: 标量，给出模型正确分类的实例比例。
        """

        # 可能对数据进行采样
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # 分批计算预测
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def train(self):
        """
        运行优化以训练模型。
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # 可能打印训练损失
            if self.verbose and t % self.print_every == 0:
                print(
                    "(Iteration %d / %d) loss: %f"
                    % (t + 1, num_iterations, self.loss_history[-1])
                )

            # 在每个 epoch 结束时，增加 epoch 计数器并衰减学习率。
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]["learning_rate"] *= self.lr_decay

            # 在第一次迭代、最后一次迭代以及每个 epoch 结束时检查训练和验证准确率。
            first_it = t == 0
            last_it = t == num_iterations - 1
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(
                    self.X_train, self.y_train, num_samples=self.num_train_samples
                )
                val_acc = self.check_accuracy(
                    self.X_val, self.y_val, num_samples=self.num_val_samples
                )
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()

                if self.verbose:
                    print(
                        "(Epoch %d / %d) train acc: %f; val_acc: %f"
                        % (self.epoch, self.num_epochs, train_acc, val_acc)
                    )

                # 记录最佳验证集准确率
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # 在训练结束时，将最佳参数换入模型
        self.model.params = self.best_params
