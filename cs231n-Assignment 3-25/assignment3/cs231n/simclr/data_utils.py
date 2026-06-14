from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import random
import torch

def compute_train_transform(seed=123456):
    """
    This function returns a composition of data augmentations to a single training image.
    Complete the following lines. Hint: look at available functions in torchvision.transforms
    """
    random.seed(seed)
    torch.random.manual_seed(seed)
    
    # Transformation that applies color jitter with brightness=0.4, contrast=0.4, saturation=0.4, and hue=0.1
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
    
    train_transform = transforms.Compose([
        ##############################################################################
        # TODO: 在此开始你的代码。                                                   #
        #                                                                            #
        # 提示: 查看 torchvision.transforms 中定义的变换函数                         #
        # 第一个操作已作为示例为你填写。
        ##############################################################################
        # 步骤 1: 随机裁剪并缩放到 32x32。
        transforms.RandomResizedCrop(32),
        # 步骤 2: 以 0.5 的概率水平翻转图像
        transforms.RandomHorizontalFlip(p=0.5),
        # 步骤 3: 以 0.8 的概率应用颜色抖动 (你可以使用上面定义的 "color_jitter")。
        transforms.RandomApply([color_jitter], p=0.8),
        # 步骤 4: 以 0.2 的概率将图像转换为灰度图
        transforms.RandomGrayscale(p=0.2),
        ##############################################################################
        #                                代码结束                                    #
        ##############################################################################
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return train_transform
    
def compute_test_transform():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return test_transform


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        x_i = None
        x_j = None

        if self.transform is not None:
            ##############################################################################
            # TODO: 在此开始你的代码。                                                   #
            #                                                                            #
            # 对图像应用 self.transform，以生成论文中的 x_i 和 x_j                       #
            ##############################################################################
            x_i = self.transform(img)
            x_j = self.transform(img)
            ##############################################################################
            #                                代码结束                                    #
            ##############################################################################

        if self.target_transform is not None:
            target = self.target_transform(target)

        return x_i, x_j, target