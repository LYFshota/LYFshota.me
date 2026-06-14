from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """计算仿射（全连接）层的前向传播。

    输入 x 的形状为 (N, d_1, ..., d_k)，包含 N 个样本的微批次（minibatch），
    其中每个样本 x[i] 的形状为 (d_1, ..., d_k)。我们将
    把每个输入重塑为维度 D = d_1 * ... * d_k 的向量，
    然后将其变换为维度 M 的输出向量。

    输入：
    - x: 包含输入数据的 numpy 数组，形状为 (N, d_1, ..., d_k)
    - w: 权重的 numpy 数组，形状为 (D, M)
    - b: 偏置的 numpy 数组，形状为 (M,)

    返回一个元组，包含：
    - out: 输出，形状为 (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: 从作业 1 中复制你的解。                                     #
    ###########################################################################
        # 1. 展平输入: 将 (N, d1, ..., dk) 重塑为 (N, D)
    N = x.shape[0]
    x_row = x.reshape(N, -1)
    
    # 2. 线性变换: Out = XW + b
    out = x_row.dot(w) + b

    ###########################################################################
    #                          END OF YOUR CODE (代码结束)                    #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """计算仿射（全连接）层的反向传播。

    输入：
    - dout: 上游导数，形状为 (N, M)
    - cache: 包含以下内容的元组：
      - x: 输入数据，形状为 (N, d_1, ... d_k)
      - w: 权重，形状为 (D, M)
      - b: 偏置，形状为 (M,)

    返回一个元组，包含：
    - dx: 相对于 x 的梯度，形状为 (N, d1, ..., d_k)
    - dw: 相对于 w 的梯度，形状为 (D, M)
    - db: 相对于 b 的梯度，形状为 (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: 从作业 1 中复制你的解。                                     #
    ###########################################################################
    
    N = x.shape[0]
    # 将输入展平，这与前向传播中的操作对应
    x_reshaped = x.reshape(N, -1)  # 形状变为 (N, D)

    # 计算关于 x 的梯度 dx
    # dout (N, M) dot w.T (M, D) -> (N, D)
    dx = dout.dot(w.T) 
    # 将 dx 恢复为原始输入的形状 (N, d1, ..., dk)
    dx = dx.reshape(x.shape)

    # 计算关于 w 的梯度 dw
    # x_reshaped.T (D, N) dot dout (N, M) -> (D, M)
    dw = x_reshaped.T.dot(dout)

    # 计算关于 b 的梯度 db
    # 对 dout 在 batch 维度 (axis=0) 求和
    db = np.sum(dout, axis=0)

    return dx, dw, db


def relu_forward(x):
    """计算修正线性单元（ReLU）层的前向传播。

    输入：
    - x: 输入，任意形状

    返回一个元组，包含：
    - out: 输出，与 x 形状相同
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: 从作业 1 中复制你的解。                                     #
    ###########################################################################

    out = np.maximum(0, x)

    ###########################################################################
    #                          END OF YOUR CODE (代码结束)                    #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """计算修正线性单元（ReLU）层的反向传播。

    输入：
    - dout: 上游导数，任意形状
    - cache: 输入 x，与 dout 形状相同

    返回：
    - dx: 相对于输入 x 的梯度
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: 从作业 1 中复制你的解。                                     #
    ###########################################################################
    dx = np.array(dout, copy=True) 

    dx[x <= 0] = 0
    ###########################################################################
    #                          END OF YOUR CODE (代码结束)                    #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """计算 softmax 分类的损失和梯度。

    输入：
    - x: 输入数据，形状为 (N, C)，其中 x[i, j] 是第 i 个输入在第 j 个类的得分。
    - y: 标签向量，形状为 (N,)，其中 y[i] 是 x[i] 的标签，且 0 <= y[i] < C

    返回一个元组，包含：
    - loss: 给出损失的标量
    - dx: 损失相对于 x 的梯度
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: 从作业 1 中复制你的解。                                     #
    ###########################################################################
    
    # 1. 计算 Softmax 概率
    # 为了数值稳定性，减去每行的最大值
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_logits)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # 2. 计算 Loss
    # 提取正确类别的概率
    N = x.shape[0]
    correct_logprobs = -np.log(probs[range(N), y])
    loss = np.sum(correct_logprobs) / N
    
    # 3. 计算梯度 dx
    dx = probs.copy()
    dx[range(N), y] -= 1
    dx /= N
    ###########################################################################
    #                          END OF YOUR CODE (代码结束)                    #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """批量归一化（Batch Normalization）的前向传播。

    在训练期间，从微批次统计数据中计算样本均值和（未校正的）样本方差，
    并将其用于归一化传入的数据。
    在训练期间，我们还会保持每个特征的均值和方差呈指数衰减的运行平均值，
    并且在测试时使用这些平均值来归一化数据。

    在每个时间步，我们使用基于动量（momentum）参数的指数衰减
    来更新均值和方差的运行平均值：

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    请注意，批量归一化论文建议了一种不同的测试时行为：
    他们使用大量训练图像而不是使用运行平均值来计算每个特征的样本均值和方差。
    在这个实现中，我们选择使用运行平均值，因为它们不需要额外的估计步骤；
    torch7 的批量归一化实现也使用了运行平均值。

    输入：
    - x: 形状为 (N, D) 的数据
    - gamma: 形状为 (D,) 的缩放参数
    - beta: 形状为 (D,) 的平移参数
    - bn_param: 具有以下键的字典：
      - mode: 'train' 或 'test'；必需
      - eps: 用于数值稳定性的常数
      - momentum: 用于运行均值/方差的常数。
      - running_mean: 形状为 (D,) 的数组，给出特征的运行均值
      - running_var: 形状为 (D,) 的数组，给出特征的运行方差

    返回一个元组，包含：
    - out: 形状为 (N, D) 的输出
    - cache: 反向传播所需的元组
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: 实现批量归一化的训练时前向传播。                            #
        # 使用微批次统计数据计算均值和方差，使用这些统计数据来归一化传入数据，#
        # 并使用 gamma 和 beta 对归一化数据进行缩放和平移。                 #
        #                                                                     #
        # 你应该将输出存储在变量 out 中。你需要用于反向传播的任何中间变量     #
        # 都应存储在 cache 变量中。                                         #
        #                                                                     #
        # 你还应该将计算出的样本均值和方差与 momentum（动量）变量一起使用，   #
        # 以更新运行均值和运行方差，将结果存储在 running_mean 和            #
        # running_var 变量中。                                              #
        #                                                                     #
        # 请注意，虽然你应该跟踪运行方差，但实际上你应该基于                  #
        # 标准差（方差的平方根）对数据进行归一化！                          #
        # 参考原论文 (https://arxiv.org/abs/1502.03167) 可能会有帮助。      #
        #######################################################################
        
        # 1. 计算样本均值和方差
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        
        # 2. 归一化传入数据
        std = np.sqrt(sample_var + eps)
        x_centered = x - sample_mean
        x_normalized = x_centered / std
        
        # 3. 缩放和平移
        out = gamma * x_normalized + beta
        
        # 4. 更新运行均值和方差
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        # 5. 存储缓存以供反向传播使用
        cache = (x, gamma, x_normalized, std, x_centered, eps)
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: 实现批量归一化的测试时前向传播。                            #
        # 使用运行均值和方差来归一化传入的数据，                            #
        # 然后使用 gamma 和 beta 对归一化后的数据进行缩放和平移。           #
        # 将结果存储在 out 变量中。                                         #
        #######################################################################
        
        # 使用测试阶段的全局运行均值和方差归一化数据
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
        
        # 缩放和平移
        out = gamma * x_normalized + beta
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """批量归一化的反向传播。

    对于此实现，你应该在纸上写出批量归一化的计算图，
    并通过中间节点向后传播梯度。

    输入：
    - dout: 上游导数，形状为 (N, D)
    - cache: 来自 batchnorm_forward 的中间变量。

    返回一个元组，包含：
    - dx: 相对于输入 x 的梯度，形状为 (N, D)
    - dgamma: 相对于缩放参数 gamma 的梯度，形状为 (D,)
    - dbeta: 相对于平移参数 beta 的梯度，形状为 (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: 实现批量归一化的反向传播。将结果存储在 dx, dgamma 
    # 和 dbeta 变量中。                                                       #
    # 参考原版研究论文 (https://arxiv.org/abs/1502.03167)               #
    # 可能会大有帮助。                                                        #
    ###########################################################################
    
    # 提取前向传播保存的缓存
    x, gamma, x_normalized, std, x_centered, eps = cache
    N, D = dout.shape
    
    # 1. 计算对 beta 和 gamma 的梯度 (基于 out = gamma * x_normalized + beta)
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normalized, axis=0)
    
    # 2. 计算对 x_normalized 的梯度
    dx_normalized = dout * gamma
    
    # 3. 计算对样本方差 sample_var 的梯度
    dvar = np.sum(dx_normalized * x_centered * -0.5 * (std ** -3), axis=0)
    
    # 4. 计算对样本均值 sample_mean 的梯度
    dmean = np.sum(dx_normalized * -1.0 / std, axis=0) + dvar * np.sum(-2.0 * x_centered, axis=0) / N
    
    # 5. 计算对原始输入数据 x 的梯度
    dx = (dx_normalized / std) + (dvar * 2.0 * x_centered / N) + (dmean / N)

    ###########################################################################
    #                          END OF YOUR CODE (代码结束)                    #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """批量归一化的一种替代的反向传播。

    对于此实现，你应该在纸上计算出批量归一化反向传播的导数并尽可能简化。
    你应该能够得出反向传播的简单表达式。
    请参阅 jupyter notebook 以获取更多提示。

    注意：此实现应预期接收与 batchnorm_backward 相同的 cache 变量，
    但可能不会使用 cache 中的所有值。

    输入 / 输出：与 batchnorm_backward 相同
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: 实现批量归一化的反向传播。将结果存储在 dx, dgamma 
    # 和 dbeta 变量中。                                                       #
    #                                                                         #
    # 在计算出相对于中心化输入的梯度后，你应该可以用一条语句计算出          #
    # 相对于输入的梯度；我们的实现只有一行且不超过 80 个字符。              #
    ###########################################################################
    
    # 从缓存中提取前向传播保存的变量
    x, gamma, x_normalized, std, x_centered, eps = cache
    N = dout.shape[0]
    
    # 计算 beta 和 gamma 的梯度 (这一步与原来相同)
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normalized, axis=0)
    
    # 替代反向传播：使用化简后的一步到位公式计算 dx
    dx = (gamma / (N * std)) * (N * dout - dbeta - x_normalized * dgamma)

    ###########################################################################
    #                          END OF YOUR CODE (代码结束)                    #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """层归一化（Layer Normalization）的前向传播。

    在训练和测试期间，传入的数据在每个数据点上进行归一化，
    然后按与批量归一化相同的 gamma 和 beta 参数进行缩放。

    请注意，与批量归一化相比，层归一化在训练和测试时的行为是相同的，
    而且我们不需要跟踪任何形式的运行平均值。

    输入：
    - x: 形状为 (N, D) 的数据
    - gamma: 形状为 (D,) 的缩放参数
    - beta: 形状为 (D,) 的平移参数
    - ln_param: 具有以下键的字典：
        - eps: 用于数值稳定性的常数

    返回一个元组，包含：
    - out: 形状为 (N, D) 的输出
    - cache: 反向传播所需的元组
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: 实现层归一化的训练时前向传播。                              #
    # 归一化输入数据，并使用 gamma 和 beta 缩放和平移归一化后的数据。         #
    # 提示：这可以通过稍微修改你的批量归一化的训练时实现，                      #
    # 并插入一两行位置恰当的代码来完成。特别是，你能否想到你可以执行的          #
    # 任何矩阵变换，使得你可以把 batch norm 代码复制过来并几乎保持不变？        #
    ###########################################################################
    
    # 1. 计算每个样本的所有特征的均值和方差（沿特征维度 axis=1）
    # 使用 keepdims=True 保持形状为 (N, 1)，以支持后续的自动广播
    mu = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    
    # 2. 归一化传入的数据
    std = np.sqrt(var + eps)
    x_centered = x - mu
    x_norm = x_centered / std
    
    # 3. 缩放和平移（gamma和beta的形状是 (D,)，这里也会被广播）
    out = gamma * x_norm + beta
    
    # 4. 存储反向传播缓存
    cache = (x_norm, gamma, std, x_centered, eps)

    ###########################################################################
    #                          END OF YOUR CODE (代码结束)                    #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """层归一化的反向传播。

    对于这个实现，你可以很大程度上依赖你已经为批量归一化完成的工作。

    输入：
    - dout: 上游导数，形状为 (N, D)
    - cache: 来自 layernorm_forward 的中间变量。

    返回一个元组，包含：
    - dx: 相对于输入 x 的梯度，形状为 (N, D)
    - dgamma: 相对于缩放参数 gamma 的梯度，形状为 (D,)
    - dbeta: 相对于平移参数 beta 的梯度，形状为 (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: 实现层归一化的反向传播。                                          #
    #                                                                         #
    # 提示：这可以通过稍微修改你的批量归一化的训练时实现来完成。                #
    # 给前向传播的提示在这里仍然适用！                                          #
    ###########################################################################
    
    # 提取缓存
    x_norm, gamma, std, x_centered, eps = cache
    N, D = dout.shape
    
    # 1. 计算对 beta 和 gamma 的梯度
    # 这两个参数仍然是针对特征维度的，所以在各个样本 (N 维度, axis=0) 上求和
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    
    # 2. 计算对归一化数据 x_norm 的梯度
    dx_norm = dout * gamma
    
    # 3. 计算下方公式的中间变量：dvar 和 dmean
    # 注意！层归一化在特征维度上进行，因此求导时我们在 axis=1 上求和。分母 N 也被替换为 D。
    dvar = np.sum(dx_norm * x_centered * -0.5 * (std ** -3), axis=1, keepdims=True)
    dmean = np.sum(dx_norm * -1.0 / std, axis=1, keepdims=True) + dvar * np.sum(-2.0 * x_centered, axis=1, keepdims=True) / D
    
    # 4. 计算对输入 x 的梯度
    dx = (dx_norm / std) + (dvar * 2.0 * x_centered / D) + (dmean / D)

    ###########################################################################
    #                          END OF YOUR CODE (代码结束)                    #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """倒置随机失活（Inverted Dropout）的前向传播。

    注意，这与原版 dropout 有所不同。
    这里，p 是保留神经元输出的概率，而不是丢弃神经元输出的概率。
    有关更多详情，请参阅 http://cs231n.github.io/neural-networks-2/#reg 。

    输入：
    - x: 输入数据，任意形状
    - dropout_param: 具有以下键的字典：
      - p: Dropout参数。我们以概率 p 保留每个神经元输出。
      - mode: 'test' 或 'train'。如果模式为 train，则执行 dropout；
        如果模式为 test，则仅返回输入。
      - seed: 随机数生成器的种子。传递 seed 使函数具有确定性，
        这对于梯度检查是必需的，但在真实网络中不需要。

    输出：
    - out: 与 x 形状相同的数组。
    - cache: 元组 (dropout_param, mask)。在训练模式中，
      mask 是用于乘以输入的 dropout 掩码；在测试模式下，mask 为 None。
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: 实现倒置 dropout 训练阶段的前向传播。                         #
        # 将 dropout 掩码存储在 mask 变量中。                                 #
        #######################################################################
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: 实现倒置 dropout 测试阶段的前向传播。                       #
        #######################################################################
        pass
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """倒置随机失活的反向传播。

    输入：
    - dout: 上游导数，任意形状
    - cache: 来自 dropout_forward 的 (dropout_param, mask)。
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: 实现倒置 dropout 训练阶段的反向传播                         #
        #######################################################################
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """卷积层前向传播的朴素实现。

    输入由 N 个数据点组成，每个数据点有 C 个通道，高度 H 和宽度 W。
    我们用 F 个不同的滤波器去卷积每个输入，其中每个滤波器涵盖了所有 C 个通道，
    并且拥有高度 HH 和宽度 WW。

    输入：
    - x: 输入数据，形状为 (N, C, H, W)
    - w: 滤波器权重，形状为 (F, C, HH, WW)
    - b: 偏置，形状为 (F,)
    - conv_param: 具有以下键的字典：
      - 'stride': 水平和垂直方向相邻感受野之间的像素数（步幅）。
      - 'pad': 将用于对输入进行零填充的零的行/列数。 

    在填充时，应在输入的高度和宽度轴上对称（即两边等量）
    填充 'pad' 个零。注意不要直接修改原始输入 x。

    返回一个元组，包含：
    - out: 输出数据，形状为 (N, F, H', W')，其中 H' 和 W' 由下式给出
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: 实现卷积的前向传播。                                              #
    # 提示：你可以使用 np.pad 函数进行填充操作。                              #
    ###########################################################################
    
    # 1. 获取输入尺寸和卷积参数
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    # 2. 计算输出数据的尺寸
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)

    # 3. 对输入数据 x 进行零填充 (padding)
    # np.pad 的第二个参数是要填充的轴：((N的填充), (C的填充), (H的填充), (W的填充))
    # 我们只对 H 和 W 维度进行填充，N 和 C 维度前后都填充 0。
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    # 4. 初始化输出矩阵
    out = np.zeros((N, F, H_out, W_out))

    # 5. 朴素的多重循环实现卷积操作
    for n in range(N):             # 遍历样本
        for f in range(F):         # 遍历滤波器
            for i in range(H_out):     # 遍历输出的高度
                for j in range(W_out): # 遍历输出的宽度
                    # 确定当前卷积窗口在填充后输入数据中的边界
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW
                    
                    # 提取当前卷积窗口的数据
                    x_window = x_pad[n, :, h_start:h_end, w_start:w_end]
                    
                    # 进行逐元素乘法、求和并加上偏置
                    out[n, f, i, j] = np.sum(x_window * w[f, :, :, :]) + b[f]

    ###########################################################################
    #                          END OF YOUR CODE (代码结束)                    #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """卷积层反向传播的朴素实现。

    输入：
    - dout: 上游导数。
    - cache: 和 conv_forward_naive 返回的值一样的 (x, w, b, conv_param) 元组。

    返回一个元组，包含：
    - dx: 相对于 x 的梯度
    - dw: 相对于 w 的梯度
    - db: 相对于 b 的梯度
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: 实现卷积过程的反向传播。                                          #
    ###########################################################################
    
    # 1. 从 cache 中提取前向传播保存的变量
    x, w, b, conv_param = cache
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    # 取出各项的尺寸
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape
    
    # 2. 初始化梯度变量
    dx_pad = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    # 对输入 x 进行和前向传播相同的填充，用于切片以计算 dw
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    
    # 3. 计算对 b 的梯度 db (在空间和批次维度上直接对 dout 求和)
    for f in range(F):
        db[f] = np.sum(dout[:, f, :, :])
        
    # 4. 朴素多重循环计算 dw 和 dx_pad
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    # 确定感受野窗口的边界
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW
                    
                    # 累加对权重 w 的梯度：上游梯度 dout 乘以对应窗口的输入数据
                    dw[f] += x_pad[n, :, h_start:h_end, w_start:w_end] * dout[n, f, i, j]
                    
                    # 累加对输入 x_pad 的梯度：上游梯度 dout 乘以对应的权重参数
                    dx_pad[n, :, h_start:h_end, w_start:w_end] += w[f] * dout[n, f, i, j]
                    
    # 5. 去除 dx_pad 周围的填充部分，还原为和 x 一致的尺寸，得到 dx
    dx = dx_pad[:, :, pad:pad+H, pad:pad+W]

    ###########################################################################
    #                          END OF YOUR CODE (代码结束)                    #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """最大池化层前向传播的朴素实现。

    输入：
    - x: 输入数据，形状为 (N, C, H, W)
    - pool_param: 具有以下键的字典：
      - 'pool_height': 每个池化区域的高度
      - 'pool_width': 每个池化区域的宽度
      - 'stride': 相邻池化区域之间的距离（步幅）

    这里不需要考虑填充（padding），你可以假定：
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    返回一个元组，包含：
    - out: 输出数据，形状为 (N, C, H', W')，其中 H' 和 W' 为：
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: 实现最大池化层的前向传播。                                        #
    ###########################################################################
    
    # 1. 提取输入尺寸和池化参数
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    # 2. 计算输出数据尺寸
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)
    
    # 3. 初始化输出数组
    out = np.zeros((N, C, H_out, W_out))
    
    # 4. 朴素的多重循环实现最大池化
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    # 计算当前池化窗口的位置
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width
                    
                    # 提取池化窗口并在局部区域内求取最大值
                    x_pool = x[n, c, h_start:h_end, w_start:w_end]
                    out[n, c, i, j] = np.max(x_pool)

    ###########################################################################
    #                          END OF YOUR CODE (代码结束)                    #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """最大池化层反向传播的朴素实现。

    输入：
    - dout: 上游导数
    - cache: 包含前向传播中存储的 (x, pool_param) 的元组

    返回：
    - dx: 相对于输入 x 的梯度
    """
    dx = None
    ###########################################################################
    # TODO: 实现最大池化层的反向传播。                                        #
    ###########################################################################
    
    # 1. 提取前向传播时的参数
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    _, _, H_out, W_out = dout.shape
    
    # 2. 初始化梯度 dx 为全零
    dx = np.zeros_like(x)
    
    # 3. 朴素的多重循环计算池化层梯度
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    # 确定感受野的边界
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width
                    
                    # 取出原来的对应窗口数据
                    x_pool = x[n, c, h_start:h_end, w_start:w_end]
                    
                    # 创建一个 mask（掩码），其中等于该窗口最大值的元素记为 True(1)，其余为 False(0)
                    mask = (x_pool == np.max(x_pool))
                    
                    # 传播上游来的梯度：仅仅传递给予窗口中最大的那个元素
                    #（注意由于窗口间可能发生重叠，这里需要使用 += 进行累积梯度）
                    dx[n, c, h_start:h_end, w_start:w_end] += mask * dout[n, c, i, j]

    ###########################################################################
    #                          END OF YOUR CODE (代码结束)                    #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """空间批量归一化（Spatial Batch Normalization）的前向传播。

    输入：
    - x: 输入数据，形状为 (N, C, H, W)
    - gamma: 缩放参数，形状为 (C,)
    - beta: 平移参数，形状为 (C,)
    - bn_param: 具有以下键的字典：
      - mode: 'train'（训练模式）或 'test'（测试模式）；必填
      - eps: 用于数值稳定性的常数
      - momentum: 用于运行均值/方差的常数。动量指示在运行均值/运行方差的计算中保留多少旧的信息。对于该值，通常 0.9 是一个较好的默认选择。
      - running_mean: 形状为 (C,) 的数组，给出特征的运行均值
      - running_var: 形状为 (C,) 的数组，给出特征的运行方差

    返回一个元组，包含：
    - out: 输出数据，形状为 (N, C, H, W)
    - cache: 用于反向传播的中间变量缓存
    """
    ###########################################################################
    # TODO: 实现空间批量归一化的前向传播。                                    #
    #                                                                         #
    # 提示：你可以通过重塑输入数据（reshape），利用之前的批量归一化前向传播   #
    # 的标准实现来完成此操作。你的实现应当非常简洁：不要超过大约5行代码。     #
    ###########################################################################
    N, C, H, W = x.shape
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)
    out_flat, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    out = out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                          END OF YOUR CODE (代码结束)                    #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """空间批量归一化的反向传播。

    输入：
    - dout: 上游导数，形状为 (N, C, H, W)
    - cache: 来自 spatial_batchnorm_forward 的中间变量缓存。

    返回一个元组，包含：
    - dx: 相对于输入数据的梯度，形状为 (N, C, H, W)
    - dgamma: 相对于缩放参数 gamma 的梯度，形状为 (C,)
    - dbeta: 相对于平移参数 beta 的梯度，形状为 (C,)
    """
    ###########################################################################
    # TODO: 实现空间批量归一化的反向传播。                                    #
    #                                                                         #
    # 提示：与前向传播类似，你可以通过重塑输入数据（reshape），利用之前的批量 #
    # 归一化反向传播的标准实现来完成此操作。你的实现应当非常简洁：大约只需5行 #
    # 代码左右。                                                              #
    ###########################################################################
    N, C, H, W = dout.shape
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx_flat, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
    dx = dx_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                          END OF YOUR CODE (代码结束)                    #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """空间组归一化（Spatial Group Normalization）的前向传播。
    
    与层归一化不同，组归一化将数据中的每个条目分成 G 个连续的部分，
    然后独立地对它们进行归一化。接着，以与批量归一化和层归一化
    相同的方式，对数据应用逐特征的平移和缩放。

    输入：
    - x: 输入数据，形状为 (N, C, H, W)
    - gamma: 缩放参数，形状为 (1, C, 1, 1)
    - beta: 平移参数，形状为 (1, C, 1, 1)
    - G: 要分成组的整数数量，应该是 C 的约数
    - gn_param: 具有以下键的字典：
      - eps: 用于数值稳定性的常数

    返回一个元组，包含：
    - out: 输出数据，形状为 (N, C, H, W)
    - cache: 用于反向传播的中间变量缓存
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: 实现空间组归一化的前向传播。                                      #
    # 这一部分的实现与你之前完成的层归一化（layer normalization）非常相似。   #
    # 特别是，你应该思考如何通过重塑（reshape）张量的形状，使得大部分代码与   #
    # 批量归一化（batch normalization）和层归一化的前向传播代码保持一致！     #
    ###########################################################################
    N, C, H, W = x.shape
    x_reshaped = x.reshape(N * G, C // G * H * W)
    mean = np.mean(x_reshaped, axis=1, keepdims=True)
    var = np.var(x_reshaped, axis=1, keepdims=True)
    
    x_norm_reshaped = (x_reshaped - mean) / np.sqrt(var + eps)
    x_norm = x_norm_reshaped.reshape(N, C, H, W)
    
    out = x_norm * gamma + beta
    cache = (x, x_norm_reshaped, mean, var, eps, gamma, beta, G)
    ###########################################################################
    #                          END OF YOUR CODE (代码结束)                    #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """空间组归一化的反向传播。

    输入：
    - dout: 上游导数，形状为 (N, C, H, W)
    - cache: 来自 spatial_groupnorm_forward 的中间变量缓存。

    返回一个元组，包含：
    - dx: 相对于输入数据的梯度，形状为 (N, C, H, W)
    - dgamma: 相对于缩放参数 gamma 的梯度，形状为 (1, C, 1, 1)
    - dbeta: 相对于平移参数 beta 的梯度，形状为 (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: 实现空间组归一化的反向传播。                                      #
    # 这与层归一化（layer norm）的实现将非常相似。                            #
    ###########################################################################
    N, C, H, W = dout.shape
    x, x_norm_reshaped, mean, var, eps, gamma, beta, G = cache
    
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dgamma = np.sum(dout * x_norm_reshaped.reshape(N, C, H, W), axis=(0, 2, 3), keepdims=True)
    
    dx_norm = dout * gamma
    dx_norm_reshaped = dx_norm.reshape(N * G, C // G * H * W)
    
    M = C // G * H * W
    dx_reshaped = (1.0 / M) / np.sqrt(var + eps) * (
        M * dx_norm_reshaped 
        - np.sum(dx_norm_reshaped, axis=1, keepdims=True) 
        - x_norm_reshaped * np.sum(dx_norm_reshaped * x_norm_reshaped, axis=1, keepdims=True)
    )
    
    dx = dx_reshaped.reshape(N, C, H, W)
    ###########################################################################
    #                          END OF YOUR CODE (代码结束)                    #
    ###########################################################################
    return dx, dgamma, dbeta
