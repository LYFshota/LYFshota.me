import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
from .contrastive_loss import *

def train(model, data_loader, train_optimizer, epoch, epochs, batch_size=32, temperature=0.5, device='cuda'):
    """训练定义在 ./model.py 中的模型，进行一个 epoch。
    
    输入:
    - model: 在 ./model.py 中定义的 Model 类对象。
    - data_loader: torch.utils.data.DataLoader 对象；用于加载训练数据。你可以假设加载的数据已经过数据增强。
    - train_optimizer: torch.optim.Optimizer 对象；将优化器应用于训练。
    - epoch: integer；当前的 epoch 编号。
    - epochs: integer；总 epoch 数量。
    - batch_size: 每个 batch 的训练样本数量。
    - temperature: float；在 simclr_loss_vectorized 中使用的温度 (tau) 参数。
    - device: 用于定义 torch 张量的设备名称。

    返回:
    - 平均损失 (average loss)。
    """
    model.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for data_pair in train_bar:
        x_i, x_j, target = data_pair
        x_i, x_j = x_i.to(device), x_j.to(device)
        
        out_left, out_right, loss = None, None, None
        ##############################################################################
        # TODO: 在此开始你的代码。                                                   #
        #                                                                            #
        # 查看 model.py 文件以了解模型的输入和输出。                                 #
        # 将 x_i 和 x_j 输入模型，以获取 out_left 和 out_right。                     #
        # 然后使用 simclr_loss_vectorized 计算损失 (loss)。                          #
        ##############################################################################
        _, out_left = model(x_i)
        _, out_right = model(x_j)
        loss = simclr_loss_vectorized(out_left, out_right, temperature, device)
        ##############################################################################
        #                                代码结束                                    #
        ##############################################################################
        
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


def train_val(model, data_loader, train_optimizer, epoch, epochs, device='cuda'):
    is_train = train_optimizer is not None
    model.train() if is_train else model.eval()
    loss_criterion = torch.nn.CrossEntropyLoss()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


def test(model, memory_data_loader, test_data_loader, epoch, epochs, c, temperature=0.5, k=200, device='cuda'):
    model.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # 生成特征库 (feature bank)
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = model(data.to(device))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # 循环遍历测试数据，通过加权 KNN 搜索来预测标签
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.to(device), target.to(device)
            feature, out = model(data)

            total_num += data.size(0)
            # 计算每个特征向量与特征库之间的余弦相似度 ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # 统计每个类别的数量
            one_hot_label = torch.zeros(data.size(0) * k, c, device=device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # 加权得分 ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100
