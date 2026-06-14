
import torch
import torch.nn as nn
import numpy as np
import clip
from PIL import Image
import tensorflow_datasets as tfds
from torchvision import transforms as T
import cv2
from tqdm.auto import tqdm


def get_similarity_no_loop(text_features, image_features):
    """
    计算文本和图像特征向量之间的成对余弦相似度 (pairwise cosine similarity)。

    Args:
        text_features (torch.Tensor): 形状为 (N, D) 的张量。
        image_features (torch.Tensor): 形状为 (M, D) 的张量。

    Returns:
        torch.Tensor: 形状为 (N, M) 的相似度矩阵，其中每个元素 (i, j)
        是 text_features[i] 和 image_features[j] 之间的余弦相似度。
    """
    similarity = None
    ############################################################################
    # TODO: 计算余弦相似度。不要使用 for 循环。                                  #
    ############################################################################
    # 1. 沿着最后一个维度计算特征的 L2 范数并进行归一化
    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # 2. 计算点积 (矩阵乘法)，结果形状为 (N, M)
    similarity = torch.matmul(text_features_norm, image_features_norm.t())
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return similarity


@torch.no_grad()
def clip_zero_shot_classifier(clip_model, clip_preprocess, images,
                              class_texts, device):
    """使用 CLIP 模型执行零样本图像分类。

    Args:
        clip_model (torch.nn.Module): 用于对图像和文本进行编码的预训练 CLIP 模型。
        clip_preprocess (Callable): 在编码之前应用于每张图像的预处理函数。
        images (List[np.ndarray]): 输入图像列表，格式为 NumPy 数组 (H x W x C) uint8。
        class_texts (List[str]): 用于零样本分类的类标签字符串列表。
        device (torch.device): 执行计算的设备。在将 text_tokens 传递给
            clip_model 之前，请先将其传递给此设备。

    Returns:
        List[str]: 每张图像的预测类标签，从给定的 class_texts 中选择。
    """
    
    pred_classes = []

    ############################################################################
    # TODO: 找到图像的分类标签。                                                 #
    ############################################################################
    # 1. 编码文本
    text_tokens = clip.tokenize(class_texts).to(device)
    text_features = clip_model.encode_text(text_tokens)

    # 2. 预处理并编码图像
    processed_images = [clip_preprocess(Image.fromarray(img)).unsqueeze(0) for img in images]
    images_tensor = torch.cat(processed_images, dim=0).to(device)
    image_features = clip_model.encode_image(images_tensor)

    # 3. 计算文本和图像特征之间的相似度矩阵 (形状为: 类别数 N x 图像数 M)
    similarity = get_similarity_no_loop(text_features, image_features)

    # 4. 对每张图像 (沿着维度 0，即文本类别维度) 找到相似度得分最高的那个类别的索引
    best_indices = similarity.argmax(dim=0)

    # 5. 将索引转换回对应的文字标签
    for idx in best_indices:
        pred_classes.append(class_texts[idx.item()])
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return pred_classes
  

class CLIPImageRetriever:
    """
    一个使用 CLIP 的简单图像检索系统。
    """
    
    @torch.no_grad()
    def __init__(self, clip_model, clip_preprocess, images, device):
        """
        Args:
          clip_model (torch.nn.Module): 预训练的 CLIP 模型。
          clip_preprocess (Callable): 用于预处理图像的函数。
          images (List[np.ndarray]): 图像列表，格式为 NumPy 数组 (H x W x C)。
          device (torch.device): 用于执行模型的设备。
        """
        ############################################################################
        # TODO: 存储所有必要的对象变量以在 retrieve 方法中使用。                      #
        # 请注意，您应该在这里一次性处理所有图像，避免在每次文本查询时重复计算。        #
        # 为了实现最优的计算效率，您最终可能不会使用上面的 similarity 函数。            #
        ############################################################################
        self.clip_model = clip_model
        self.device = device
        
        # 1. 一次性处理并编码所有图像
        processed_images = [clip_preprocess(Image.fromarray(img)).unsqueeze(0) for img in images]
        images_tensor = torch.cat(processed_images, dim=0).to(device)
        image_features = clip_model.encode_image(images_tensor)
        
        # 2. 为了实现“最优的计算效率”，我们在这里直接把图像特征“预先归一化”并存下来。
        # 这样在之后每次检索时，就不需要重新计算图像的模长了。
        self.image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    
    @torch.no_grad()
    def retrieve(self, query: str, k: int = 2):
        """
        检索与输入文本最相似的前 k 张图像的索引。
        您可能会发现 torch.Tensor.topk 方法很有用。

        Args:
            query (str): 文本查询。
            k (int): 返回前 k 张图像。

        Returns:
            List[int]: 前 k 张最相似图像的索引。
        """
        top_indices = []
        ############################################################################
        # TODO: 检索前 k 张图像的索引。                                              #
        ############################################################################
        # 1. 编码查询文本并进行归一化
        text_tokens = clip.tokenize([query]).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 2. 计算余弦相似度: (1, D) x (D, M) -> (1, M)
        similarity = torch.matmul(text_features_norm, self.image_features_norm.t())
        
        # 3. 使用 topk 找到得分最高的前 k 个索引
        _, indices = similarity[0].topk(k)
        top_indices = indices.tolist()
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return top_indices

  
class DavisDataset:
    def __init__(self):
        self.davis = tfds.load('davis/480p', split='validation', as_supervised=False)
        self.img_tsfm = T.Compose([
            T.Resize((480, 480)), T.ToTensor(),
            T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
        ])
        
      
    def get_sample(self, index):
        assert index < len(self.davis)
        ds_iter = iter(tfds.as_numpy(self.davis))
        for i in range(index+1):
            video = next(ds_iter)
        frames, masks = video['video']['frames'], video['video']['segmentations']
        print(f"video {video['metadata']['video_name'].decode()}  {len(frames)} frames")
        return frames, masks
    
    def process_frames(self, frames, dino_model, device):
        res = []
        for f in frames:
            f = self.img_tsfm(Image.fromarray(f))[None].to(device)
            with torch.no_grad():
              tok = dino_model.get_intermediate_layers(f, n=1)[0]
            res.append(tok[0, 1:])

        res = torch.stack(res)
        return res
    
    def process_masks(self, masks, device):
        res = []
        for m in masks:
            m = cv2.resize(m, (60,60), cv2.INTER_NEAREST)
            res.append(torch.from_numpy(m).long().flatten(-2, -1))
        res = torch.stack(res).to(device)
        return res
    
    def mask_frame_overlay(self, processed_mask, frame):
        H, W = frame.shape[:2]
        mask = processed_mask.detach().cpu().numpy()
        mask = mask.reshape((60, 60))
        mask = cv2.resize(
            mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        overlay = create_segmentation_overlay(mask, frame.copy())
        return overlay
        


def create_segmentation_overlay(segmentation_mask, image, alpha=0.5):
    """
    在 RGB 图像上生成彩色的分割叠加图 (segmentation overlay)。

    Parameters:
        segmentation_mask (np.ndarray): 形状为 (H, W) 的 2D 数组，包含类别索引。
        image (np.ndarray): 形状为 (H, W, 3) 的 3D 数组，即 RGB 图像。
        alpha (float): 叠加图的透明度因子 (0 = 仅显示图像，1 = 仅显示掩码)。

    Returns:
        np.ndarray: 带分割叠加图的图像 (形状: (H, W, 3), dtype: uint8)。
    """
    assert segmentation_mask.shape[:2] == image.shape[:2], "分割掩码和图像尺寸不匹配"
    assert image.dtype == np.uint8, "图像必须是 uint8 类型"

    # 使用固定的颜色映射为每个类生成确定性颜色
    def generate_colormap(n):
        np.random.seed(42)  # 为了确保每次生成的颜色相同
        colormap = np.random.randint(0, 256, size=(n, 3), dtype=np.uint8)
        return colormap

    colormap = generate_colormap(10)

    # 为分割掩码创建一个彩色图像
    seg_color = colormap[segmentation_mask]  # 形状: (H, W, 3)

    # 与原始图像混合
    overlay = cv2.addWeighted(image, 1 - alpha, seg_color, alpha, 0)

    return overlay


def compute_iou(pred, gt, num_classes):
    """计算平均交并比 (mean Intersection over Union, mIoU)。"""
    iou = 0
    for ci in range(num_classes):
        p = pred == ci
        g = gt == ci
        iou += (p & g).sum() / ((p | g).sum() + 1e-8)
    return iou / num_classes


class DINOSegmentation:
    def __init__(self, device, num_classes: int, inp_dim : int = 384):
        """
        初始化 DINOSegmentation 模型。

        这里定义了一个简单的神经网络，旨在将 DINO 特征向量分类到各个分割类别中。
        它包含了模型初始化、优化器和损失函数的设置。

        Args:
            device (torch.device): 运行模型的设备 (CPU 或 CUDA)。
            num_classes (int): 分割类别的数量。
            inp_dim (int, optional): 输入的 DINO 特征的维度。
        """

        ############################################################################
        # TODO: 定义一个非常轻量级的 pytorch 模型、优化器和损失函数，                #
        # 用来训练将每个 DINO 特征向量分类到一个分割类别中。                         #
        # 它可以是一个线性层或两层神经网络。                                         #
        ############################################################################
        self.device = device
        self.model = nn.Linear(inp_dim, num_classes).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def train(self, X_train, Y_train, num_iters=500):
        """使用提供的训练数据训练分割模型。

        Args:
            X_train (torch.Tensor): 形状为 (N, D) 的输入特征向量。
            Y_train (torch.Tensor): 形状为 (N,) 的真实标签 (Ground truth labels)。
            num_iters (int, optional): 优化步数。
        """
        ############################################################################
        # TODO: 训练您的模型 `num_iters` 步。                                        #
        ############################################################################
        self.model.train()
        X_train = X_train.to(self.device)
        Y_train = Y_train.to(self.device)
        
        for _ in range(num_iters):
            self.optimizer.zero_grad()
            logits = self.model(X_train)
            loss = self.criterion(logits, Y_train)
            loss.backward()
            self.optimizer.step()
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    
    @torch.no_grad()
    def inference(self, X_test):
        """对给定的测试 DINO 特征向量执行推断 (inference)。

        Args:
            X_test (torch.Tensor): 形状为 (N, D) 的输入特征向量。

        Returns:
            形状为 (N,) 的 torch.Tensor: 预测的类别索引。
        """
        pred_classes = None
        ############################################################################
        # TODO: 预测测试集上的类别 (原英文注释有误写成了 Train num_iters 步)。         #
        ############################################################################
        self.model.eval()
        X_test = X_test.to(self.device)
        logits = self.model(X_test)
        pred_classes = logits.argmax(dim=-1)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return pred_classes