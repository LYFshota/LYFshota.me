import os

file_path = r"d:\Python\west2AI\LYFshota.me\cs231n-Assignment 3-25\assignment3\cs231n\clip_dino.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

replacements = [
    (
'''    """
    Computes the pairwise cosine similarity between text and image feature vectors.

    Args:
        text_features (torch.Tensor): A tensor of shape (N, D).
        image_features (torch.Tensor): A tensor of shape (M, D).

    Returns:
        torch.Tensor: A similarity matrix of shape (N, M), where each entry (i, j)
        is the cosine similarity between text_features[i] and image_features[j].
    """''',
'''    """
    计算文本和图像特征向量之间的成对余弦相似度 (pairwise cosine similarity)。

    Args:
        text_features (torch.Tensor): 形状为 (N, D) 的张量。
        image_features (torch.Tensor): 形状为 (M, D) 的张量。

    Returns:
        torch.Tensor: 形状为 (N, M) 的相似度矩阵，其中每个元素 (i, j)
        是 text_features[i] 和 image_features[j] 之间的余弦相似度。
    """'''
    ),
    (
'''    ############################################################################
    # TODO: Compute the cosine similarity. Do NOT use for loops.               #
    ############################################################################''',
'''    ############################################################################
    # TODO: 计算余弦相似度。不要使用 for 循环。                                  #
    ############################################################################'''
    ),
    (
'''    """Performs zero-shot image classification using a CLIP model.

    Args:
        clip_model (torch.nn.Module): The pre-trained CLIP model for encoding
            images and text.
        clip_preprocess (Callable): A preprocessing function to apply to each
            image before encoding.
        images (List[np.ndarray]): A list of input images as NumPy arrays
            (H x W x C) uint8.
        class_texts (List[str]): A list of class label strings for zero-shot
            classification.
        device (torch.device): The device on which computation should be
            performed. Pass text_tokens to this device before passing it to
            clip_model.

    Returns:
        List[str]: Predicted class label for each image, selected from the
            given class_texts.
    """''',
'''    """使用 CLIP 模型执行零样本图像分类。

    Args:
        clip_model (torch.nn.Module): 用于对图像和文本进行编码的预训练 CLIP 模型。
        clip_preprocess (Callable): 在编码之前应用于每张图像的预处理函数。
        images (List[np.ndarray]): 输入图像列表，格式为 NumPy 数组 (H x W x C) uint8。
        class_texts (List[str]): 用于零样本分类的类标签字符串列表。
        device (torch.device): 执行计算的设备。在将 text_tokens 传递给
            clip_model 之前，请先将其传递给此设备。

    Returns:
        List[str]: 每张图像的预测类标签，从给定的 class_texts 中选择。
    """'''
    ),
    (
'''    ############################################################################
    # TODO: Find the class labels for images.                                  #
    ############################################################################''',
'''    ############################################################################
    # TODO: 找到图像的分类标签。                                                 #
    ############################################################################'''
    ),
    (
'''    """
    A simple image retrieval system using CLIP.
    """''',
'''    """
    一个使用 CLIP 的简单图像检索系统。
    """'''
    ),
    (
'''        """
        Args:
          clip_model (torch.nn.Module): The pre-trained CLIP model.
          clip_preprocess (Callable): Function to preprocess images.
          images (List[np.ndarray]): List of images as NumPy arrays (H x W x C).
          device (torch.device): The device for model execution.
        """''',
'''        """
        Args:
          clip_model (torch.nn.Module): 预训练的 CLIP 模型。
          clip_preprocess (Callable): 用于预处理图像的函数。
          images (List[np.ndarray]): 图像列表，格式为 NumPy 数组 (H x W x C)。
          device (torch.device): 用于执行模型的设备。
        """'''
    ),
    (
'''        ############################################################################
        # TODO: Store all necessary object variables to use in retrieve method.    #
        # Note that you should process all images at once here and avoid repeated  #
        # computation for each text query. You may end up NOT using the above      #
        # similarity function for most compute-optimal implementation.#
        ############################################################################''',
'''        ############################################################################
        # TODO: 存储所有必要的对象变量以在 retrieve 方法中使用。                      #
        # 请注意，您应该在这里一次性处理所有图像，避免在每次文本查询时重复计算。        #
        # 为了实现最优的计算效率，您最终可能不会使用上面的 similarity 函数。            #
        ############################################################################'''
    ),
    (
'''        """
        Retrieves the indices of the top-k images most similar to the input text.
        You may find torch.Tensor.topk method useful.

        Args:
            query (str): The text query.
            k (int): Return top k images.

        Returns:
            List[int]: Indices of the top-k most similar images.
        """''',
'''        """
        检索与输入文本最相似的前 k 张图像的索引。
        您可能会发现 torch.Tensor.topk 方法很有用。

        Args:
            query (str): 文本查询。
            k (int): 返回前 k 张图像。

        Returns:
            List[int]: 前 k 张最相似图像的索引。
        """'''
    ),
    (
'''        ############################################################################
        # TODO: Retrieve the indices of top-k images.                              #
        ############################################################################''',
'''        ############################################################################
        # TODO: 检索前 k 张图像的索引。                                              #
        ############################################################################'''
    ),
    (
'''    """
    Generate a colored segmentation overlay on top of an RGB image.

    Parameters:
        segmentation_mask (np.ndarray): 2D array of shape (H, W), with class indices.
        image (np.ndarray): 3D array of shape (H, W, 3), RGB image.
        alpha (float): Transparency factor for overlay (0 = only image, 1 = only mask).

    Returns:
        np.ndarray: Image with segmentation overlay (shape: (H, W, 3), dtype: uint8).
    """''',
'''    """
    在 RGB 图像上生成彩色的分割叠加图 (segmentation overlay)。

    Parameters:
        segmentation_mask (np.ndarray): 形状为 (H, W) 的 2D 数组，包含类别索引。
        image (np.ndarray): 形状为 (H, W, 3) 的 3D 数组，即 RGB 图像。
        alpha (float): 叠加图的透明度因子 (0 = 仅显示图像，1 = 仅显示掩码)。

    Returns:
        np.ndarray: 带分割叠加图的图像 (形状: (H, W, 3), dtype: uint8)。
    """'''
    ),
    (
'''    assert segmentation_mask.shape[:2] == image.shape[:2], "Segmentation and image size mismatch"
    assert image.dtype == np.uint8, "Image must be of type uint8"

    # Generate deterministic colors for each class using a fixed colormap
    def generate_colormap(n):
        np.random.seed(42)  # For determinism
        colormap = np.random.randint(0, 256, size=(n, 3), dtype=np.uint8)
        return colormap

    colormap = generate_colormap(10)

    # Create a color image for the segmentation mask
    seg_color = colormap[segmentation_mask]  # shape: (H, W, 3)

    # Blend with original image''',
'''    assert segmentation_mask.shape[:2] == image.shape[:2], "分割掩码和图像尺寸不匹配"
    assert image.dtype == np.uint8, "图像必须是 uint8 类型"

    # 使用固定的颜色映射为每个类生成确定性颜色
    def generate_colormap(n):
        np.random.seed(42)  # 为了确保每次生成的颜色相同
        colormap = np.random.randint(0, 256, size=(n, 3), dtype=np.uint8)
        return colormap

    colormap = generate_colormap(10)

    # 为分割掩码创建一个彩色图像
    seg_color = colormap[segmentation_mask]  # 形状: (H, W, 3)

    # 与原始图像混合'''
    ),
    (
'''    """Compute the mean Intersection over Union (IoU)."""''',
'''    """计算平均交并比 (mean Intersection over Union, mIoU)。"""'''
    ),
    (
'''        """
        Initialize the DINOSegmentation model.

        This defines a simple neural network designed to  classify DINO feature
        vectors into segmentation classes. It includes model initialization,
        optimizer, and loss function setup.

        Args:
            device (torch.device): Device to run the model on (CPU or CUDA).
            num_classes (int): Number of segmentation classes.
            inp_dim (int, optional): Dimensionality of the input DINO features.
        """''',
'''        """
        初始化 DINOSegmentation 模型。

        这里定义了一个简单的神经网络，旨在将 DINO 特征向量分类到各个分割类别中。
        它包含了模型初始化、优化器和损失函数的设置。

        Args:
            device (torch.device): 运行模型的设备 (CPU 或 CUDA)。
            num_classes (int): 分割类别的数量。
            inp_dim (int, optional): 输入的 DINO 特征的维度。
        """'''
    ),
    (
'''        ############################################################################
        # TODO: Define a very lightweight pytorch model, optimizer, and loss       #
        # function to train classify each DINO feature vector into a seg. class.   #
        # It can be a linear layer or two layer neural network.                    #
        ############################################################################''',
'''        ############################################################################
        # TODO: 定义一个非常轻量级的 pytorch 模型、优化器和损失函数，                #
        # 用来训练将每个 DINO 特征向量分类到一个分割类别中。                         #
        # 它可以是一个线性层或两层神经网络。                                         #
        ############################################################################'''
    ),
    (
'''        """Train the segmentation model using the provided training data.

        Args:
            X_train (torch.Tensor): Input feature vectors of shape (N, D).
            Y_train (torch.Tensor): Ground truth labels of shape (N,).
            num_iters (int, optional): Number of optimization steps.
        """''',
'''        """使用提供的训练数据训练分割模型。

        Args:
            X_train (torch.Tensor): 形状为 (N, D) 的输入特征向量。
            Y_train (torch.Tensor): 形状为 (N,) 的真实标签 (Ground truth labels)。
            num_iters (int, optional): 优化步数。
        """'''
    ),
    (
'''        ############################################################################
        # TODO: Train your model for `num_iters` steps.                            #
        ############################################################################''',
'''        ############################################################################
        # TODO: 训练您的模型 `num_iters` 步。                                        #
        ############################################################################'''
    ),
    (
'''        """Perform inference on the given test DINO feature vectors.

        Args:
            X_test (torch.Tensor): Input feature vectors of shape (N, D).

        Returns:
            torch.Tensor of shape (N,): Predicted class indices.
        """''',
'''        """对给定的测试 DINO 特征向量执行推断 (inference)。

        Args:
            X_test (torch.Tensor): 形状为 (N, D) 的输入特征向量。

        Returns:
            形状为 (N,) 的 torch.Tensor: 预测的类别索引。
        """'''
    ),
    (
'''        ############################################################################
        # TODO: Train your model for `num_iters` steps.                            #
        ############################################################################''',
'''        ############################################################################
        # TODO: 执行推断以获得预测结果 (原英文注释有误写成了 Train)。                  #
        ############################################################################'''
    )
]

for orig, target in replacements:
    content = content.replace(orig, target)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Translation applied.")
