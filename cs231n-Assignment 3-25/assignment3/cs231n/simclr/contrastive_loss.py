import torch
import numpy as np


def sim(z_i, z_j):
    """两个向量之间的归一化点积。

    输入：
    - z_i: 1xD 张量。
    - z_j: 1xD 张量。
    
    返回：
    - 一个标量值，表示 z_i 和 z_j 之间的归一化点积。
    """
    norm_dot_product = None
    ##############################################################################
    # TODO: 在此开始你的代码。                                                   #
    #                                                                            #
    # 提示: torch.linalg.norm 可能会有帮助。                                     #
    ##############################################################################
    norm_i = torch.linalg.norm(z_i)
    norm_j = torch.linalg.norm(z_j)
    norm_dot_product = torch.sum(z_i * z_j) / (norm_i * norm_j)
    ##############################################################################
    #                               你的代码结束                                 #
    ##############################################################################
    
    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """计算一个批次上的对比损失 L（朴素循环版本）。
    
    输入：
    - out_left: NxD 张量；SimCLR 模型左分支投影头 g() 的输出。
    - out_right: NxD 张量；SimCLR 模型右分支投影头 g() 的输出。
    每一行是批次中一个增强样本的 z 向量。out_left 和 out_right 中相同的行形成一个正样本对。 
    换句话说，对于所有 k=0...N-1，(out_left[k], out_right[k]) 形成一个正样本对。
    - tau: 标量值，决定指数增长速度的温度参数。
    
    返回：
    - 一个标量值；批次中所有正样本对的总损失。请参阅 notebook 获取定义。
    """
    N = out_left.shape[0]  # 训练样本的总数
    
     # 将 out_left 和 out_right 拼接成一个 2*N x D 的张量。
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # 遍历每个正样本对 (k, k+N)
        z_k, z_k_N = out[k], out[k+N]
        
        ##############################################################################
        numerator_k = torch.exp(sim(z_k, z_k_N) / tau)
        denominator_k = 0
        for a in range(2 * N):
            if a != k:
                denominator_k += torch.exp(sim(z_k, out[a]) / tau)
        l_k = -torch.log(numerator_k / denominator_k)

        numerator_k_N = torch.exp(sim(z_k_N, z_k) / tau)
        denominator_k_N = 0
        for a in range(2 * N):
            if a != k + N:
                denominator_k_N += torch.exp(sim(z_k_N, out[a]) / tau)
        l_k_N = -torch.log(numerator_k_N / denominator_k_N)

        total_loss += l_k + l_k_N
        ##############################################################################
        #                               你的代码结束                                 #
        ##############################################################################
    
    # 最后，我们需要将总损失除以 2N，即批次中的样本数。
    total_loss = total_loss / (2*N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """正样本对之间的归一化点积。

    输入：
    - out_left: NxD 张量；SimCLR 模型左分支投影头 g() 的输出。
    - out_right: NxD 张量；SimCLR 模型右分支投影头 g() 的输出。
    每一行是批次中一个增强样本的 z 向量。
    out_left 和 out_right 中相同的行形成一个正样本对。
    
    返回：
    - 一个 Nx1 张量；每一行 k 是 out_left[k] 和 out_right[k] 之间的归一化点积。
    """
    pos_pairs = None
    
    ##############################################################################
    # TODO: 在此开始你的代码。                                                   #
    #                                                                            #
    # 提示: torch.linalg.norm 可能会有帮助。                                     #
    ##############################################################################
    numerator = torch.sum(out_left * out_right, dim=1)
    denominator = torch.linalg.norm(out_left, dim=1) * torch.linalg.norm(out_right, dim=1)
    pos_pairs = (numerator / denominator).view(-1, 1)
    ##############################################################################
    #                               你的代码结束                                 #
    ##############################################################################
    return pos_pairs


def compute_sim_matrix(out):
    """计算批次中所有增强样本对之间的归一化点积的 2N x 2N 矩阵。

    输入：
    - out: 2N x D 张量；每一行是一个增强样本的 z 向量（投影头的输出）。
    批次中总共有 2N 个增强样本。
    
    返回：
    - sim_matrix: 2N x 2N 张量；矩阵中的每个元素 i, j 是 out[i] 和 out[j] 之间的归一化点积。
    """
    sim_matrix = None
    
    ##############################################################################
    # TODO: 在此开始你的代码。                                                   #
    ##############################################################################
    out_norm = torch.linalg.norm(out, dim=1, keepdim=True)
    out_normalized = out / out_norm
    sim_matrix = torch.matmul(out_normalized, out_normalized.T)
    ##############################################################################
    #                               你的代码结束                                 #
    ##############################################################################
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):
    """计算一个批次上的对比损失 L（向量化版本）。不允许使用循环。
    
    输入和输出与 simclr_loss_naive 相同。
    """
    N = out_left.shape[0]
    
    # 将 out_left 和 out_right 拼接成一个 2*N x D 的张量。
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # 计算批次中所有增强样本对之间的相似度矩阵。
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    ##############################################################################
    # TODO: 在此开始你的代码。请遵循提示。                                       #
    ##############################################################################
    
    # 第 1 步：使用 sim_matrix 计算所有增强样本的分母值。
    # 提示：计算 e^{sim / tau} 并存储到 exponential 中，其形状应为 2N x 2N。
    exponential = torch.exp(sim_matrix / tau)
    
    # 这个二进制掩码将 k=i 的项置零。
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()
    
    # 我们应用该二进制掩码。
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
    
    # 提示：计算所有增强样本的分母值。这应该是一个 2N x 1 的向量。
    denom = torch.sum(exponential, dim=1, keepdim=True)

    # 第 2 步：计算正样本对之间的相似度。
    # 你可以用两种方法实现：
    # 选项 1：从 sim_matrix 中提取相应的索引。
    # 选项 2：使用 sim_positive_pairs()。
    sim_pos = sim_positive_pairs(out_left, out_right)
    pos_similarities = torch.cat([sim_pos, sim_pos], dim=0)
    
    # 第 3 步：计算所有增强样本的分子值。
    numerator = torch.exp(pos_similarities / tau)
    
    # 第 4 步：现在你有了所有增强样本的分子和分母，计算总损失。
    loss = torch.mean(-torch.log(numerator / denom))
    
    ##############################################################################
    #                               你的代码结束                                 #
    ##############################################################################
    
    return loss


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))