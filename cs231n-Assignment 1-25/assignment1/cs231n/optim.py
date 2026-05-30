import numpy as np

"""
本文件实现了常用于训练神经网络的各种一阶更新规则。
每个更新规则接受当前的权重以及损失函数对这些权重的梯度，
并产生下一组权重。所有的更新规则都具有相同的接口：

def update(w, dw, config=None):

输入:
  - w: 一个 numpy 数组，包含当前的权重。
  - dw: 一个与 w 形状相同的 numpy 数组，包含损失对 w 的梯度。
  - config: 一个包含超参数值（如学习率、动量等）的字典。
    如果更新规则需要在多次迭代中缓存数值，那么 config 也会保存这些缓存值。

返回:
  - next_w: 更新后的下一个坐标点。
  - config: 传递给更新规则下一次迭代的 config 字典。

注意：对于大多数更新规则，默认的学习率可能表现不好；
但是其他超参数的默认值应该能够在各种不同的问题上表现良好。

为了提高效率，更新规则可以执行就地更新，也就是改变 w 的值
并让 next_w 等于 w。
"""


def sgd(w, dw, config=None):
    """
    执行普通的随机梯度下降。

    config 格式:
    - learning_rate: 标量学习率。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    执行带动量的随机梯度下降。

    config 格式:
    - learning_rate: 标量学习率。
    - momentum: 在 0 到 1 之间的标量，表示动量值。
      将 momentum 设为 0 就退化为了普通的 sgd。
    - velocity: 一个与 w 和 dw 形状相同的 numpy 数组，用于存储
      梯度的滑动平均值。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: 实现动量更新公式。将更新后的值存储在 next_w 变量中。            #
    # 你也应该使用并更新 velocity(速度) v。                                   #
    ###########################################################################
    v = config["momentum"] * v - config["learning_rate"] * dw
    next_w = w + v
    ###########################################################################
    #                              代码结束                                   #
    ###########################################################################
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    使用 RMSProp 更新规则。该规则利用梯度平方值的滑动平均值
    来设置自适应的、每个参数不同的学习率。

    config 格式:
    - learning_rate: 标量学习率。
    - decay_rate: 在 0 到 1 之间的标量，表示梯度平方缓存的衰减率。
    - epsilon: 用于平滑的小标量，避免除以零。
    - cache: 梯度二阶矩的滑动平均值。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: 实现 RMSprop 更新公式，将 w 的下一个值存储在 next_w 变量中。    #
    # 不要忘记更新存储在 config['cache'] 中的 cache 值。                      #
    ###########################################################################
    config["cache"] = config["decay_rate"] * config["cache"] + (1 - config["decay_rate"]) * (dw ** 2)
    next_w = w - config["learning_rate"] * dw / (np.sqrt(config["cache"]) + config["epsilon"])
    ###########################################################################
    #                              代码结束                                   #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    使用 Adam 更新规则，该规则结合了梯度及其平方的滑动平均值，
    以及一个偏差修正项。

    config 格式:
    - learning_rate: 标量学习率。
    - beta1: 梯度一阶矩滑动平均的衰减率。
    - beta2: 梯度二阶矩滑动平均的衰减率。
    - epsilon: 用于平滑的小标量，避免除以零。
    - m: 梯度的滑动平均值。
    - v: 梯度平方的滑动平均值。
    - t: 迭代次数。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    ###########################################################################
    # TODO: 实现 Adam 更新公式，将 w 的下一个值存储在 next_w 变量中。         #
    # 不要忘记更新存储在 config 中的 m, v 和 t 变量。                         #
    #                                                                         #
    # 注意：为了与参考输出一致，请在将 t 用于任何计算_之前_先修改它。         #
    ###########################################################################
    config["t"] += 1
    t = config["t"]
    
    m = config["m"]
    v = config["v"]
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    eps = config["epsilon"]
    lr = config["learning_rate"]
    

    m = beta1 * m + (1 - beta1) * dw

    v = beta2 * v + (1 - beta2) * (dw ** 2)

    mt = m / (1 - beta1 ** t)
    vt = v / (1 - beta2 ** t)

    next_w = w - lr * mt / (np.sqrt(vt) + eps)
    
    config["m"] = m
    config["v"] = v
    ###########################################################################
    #                              代码结束                                   #
    ###########################################################################

    return next_w, config
