import json
import sys
import re

def translate(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # 增加更多翻译规则
    subs = {
        "**Generative AI Use**": "**生成式AI的使用**",
        "For the purposes of the assignments, the use of generative AI is subject to the same policies regarding collaboration.": "就本次作业而言，生成式AI的使用同样受关于协作的政策约束。",
        "Just as with other collaborators, each student must write down the solutions independently of the output of the interaction and the submission should include a note denoting the nature of the collaboration.": "与其他合作者一样，每个学生必须独立于交互的输出写下解答，并且提交中应包含说明协作性质的附注。",
        "The use of generative AI tools to substantially complete sections of the assignments is not in line with the spirit of the assignments, and would be a violation of the [Honor Code]": "使用生成式AI工具实质性地完成作业的大部分内容不符合本次作业的精神，将构成对[荣誉守则]的违背",
        "# Convolutional Networks": "# 卷积网络",
        "So far we have worked with deep fully connected networks": "到目前为止，我们已经使用了深度全连接网络",
        "First you will implement several layer types that are used in convolutional networks.": "首先，你将实现卷积网络中使用的几种层类型。",
        "# Convolution: Naive Forward Pass": "# 卷积：朴素前向传播",
        "The core of a convolutional network is the convolution operation. In the file `cs231n/layers.py`, implement the forward pass for the convolution layer in the function `conv_forward_naive`.": "卷积网络的核心是卷积操作。在 `cs231n/layers.py` 文件中，在 `conv_forward_naive` 函数里实现卷积层的前向传播。",
        "## Aside: Image Processing via Convolutions": "## 旁注：通过卷积进行图像处理",
        "# Convolution: Naive Backward Pass": "# 卷积：朴素反向传播",
        "# This mounts your Google Drive to the Colab VM.": "# 将您的 Google Drive 挂载到 Colab 虚拟机。",
        "# TODO: Enter the foldername in your Drive where you have saved the unzipped": "# TODO：输入您在 Drive 中保存解压后的",
        "# assignment folder, e.g. 'cs231n/assignments/assignment2/'": "# 作业文件夹的名称，例如 'cs231n/assignments/assignment2/'",
        "assert FOLDERNAME is not None, \"[!] Enter the foldername.\"": "assert FOLDERNAME is not None, \"[!] 请输入文件夹名称。\"",
        "## Answer:\n[FILL THIS IN]": "## 回答：\n[在此填写]",
        "## Answer:\n\n[FILL THIS IN]": "## 回答：\n\n[在此填写]",
        "## Answer": "## 回答",
        "# PyTorch": "# PyTorch",
        "# RNN Captioning": "# RNN 图像描述 (RNN Captioning)",
        "## Inline Question 1:": "## 内联问题 1 (Inline Question 1):",
        "## Inline Question 2:": "## 内联问题 2 (Inline Question 2):",
        "## Inline Question 3:": "## 内联问题 3 (Inline Question 3):",
        "## Inline Question 4:": "## 内联问题 4 (Inline Question 4):",
        "# Setup cell.": "# 设置单元格。",
        "# Max-Pooling: Naive Forward Pass": "# 最大池化：朴素前向传播",
        "# Max-Pooling: Naive Backward Pass": "# 最大池化：朴素反向传播",
    }
    
    for cell in nb['cells']:
        for i, line in enumerate(cell['source']):
            for k, v in subs.items():
                cell['source'][i] = cell['source'][i].replace(k, v)
                
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

try:
    translate("d:/Python/west2AI/collection-ai/work4-6/LYFshota/work4/cs231n-Assignment 2-25/ConvolutionalNetworks.ipynb")
    translate("d:/Python/west2AI/collection-ai/work4-6/LYFshota/work4/cs231n-Assignment 2-25/PyTorch.ipynb")
    translate("d:/Python/west2AI/collection-ai/work4-6/LYFshota/work4/cs231n-Assignment 2-25/RNN_Captioning_pytorch.ipynb")
    print("Done")
except Exception as e:
    print(e)