import json
import os

file_path = r"D:\Python\west2AI\LYFshota.me\cs231n-Assignment 3-25\assignment3\CLIP_DINO.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

replacements = {
    "**Generative AI Use**: For the purposes of the assignments, the use of generative AI is subject to the same policies regarding collaboration. Just as with other collaborators, each student must write down the solutions independently of the output of the interaction and the submission should include a note denoting the nature of the collaboration. The use of generative AI tools to substantially complete sections of the assignments is not in line with the spirit of the assignments, and would be a violation of the [Honor Code](https://communitystandards.stanford.edu/policies-and-guidance/honor-code).\n": 
    "**生成式AI的使用**：对于本作业，生成式AI的使用需遵守有关合作的相同政策。就像与其他合作者一样，每个学生必须独立于交互输出写下解决方案，并且提交的材料应包含说明合作性质的注释。使用生成式AI工具来大幅完成作业的各个部分不符合本作业的精神，并将违反[荣誉代码(Honor Code)](https://communitystandards.stanford.edu/policies-and-guidance/honor-code)。\n",

    "# This mounts your Google Drive to the Colab VM.\n": 
    "# 这会将您的 Google Drive 挂载到 Colab 虚拟机。\n",

    "# TODO: Enter the foldername in your Drive where you have saved the unzipped\n": 
    "# TODO: 输入您保存在 Drive 中解压后的作业文件夹名称，\n",

    "# assignment folder, e.g. 'cs231n/assignments/assignment3/'\n": 
    "# 例如：'cs231n/assignments/assignment3/'\n",

    "# Now that we've mounted your Drive, this ensures that\n": 
    "# 既然我们已经挂载了您的 Drive，这能确保\n",

    "# the Python interpreter of the Colab VM can load\n": 
    "# Colab 虚拟机的 Python 解释器可以从中加载\n",

    "# python files from within it.\n": 
    "# python 文件。\n",

    "# This downloads the COCO dataset to your Drive if it doesn't already exist\n": 
    "# 这会将 COCO 数据集下载到您的 Drive（如果它还不存在的话）\n",

    "# (you should already have this dataset from a previous notebook!)\n": 
    "# （您应该已经在之前的 notebook 中获得了此数据集！）\n",

    "# Uncomment the following if you don't have it.\n": 
    "# 如果您没有，请取消注释以下内容。\n",

    "# Some useful python libraries\n": 
    "# 一些有用的 python 库\n",

    "# State-of-the-Art Pretrained Image Models\n": 
    "# 最先进的预训练图像模型 (State-of-the-Art Pretrained Image Models)\n",

    "In the previous exercise, you learned about [SimCLR](https://arxiv.org/abs/2002.05709) and how contrastive self-supervised learning can be used to learn meaningful image representations. In this notebook, we will explore two more recent models that also aim to learn high-quality visual representations and have demonstrated strong and robust performance on a variety of downstream tasks.\n": 
    "在之前的练习中，您了解了 [SimCLR](https://arxiv.org/abs/2002.05709) 以及如何使用对比自监督学习来学习有意义的图像表示。在本 notebook 中，我们将探索另外两个更近期的模型，它们也旨在学习高质量的视觉表示，并在各种下游任务中展示了强大且稳健的性能。\n",

    "First, we will examine the [CLIP](https://github.com/openai/CLIP) model. Like SimCLR, CLIP uses a contrastive learning objective, but instead of contrasting two augmented views of the same image, it contrasts two different modalities: text and image. To train CLIP, OpenAI collected a large dataset of ~400M image-text pairs from the internet, including sources like Wikipedia and image alt text. The resulting model learns rich, high-level image features and has achieved impressive zero-shot performance on many vision benchmarks.\n": 
    "首先，我们将研究 [CLIP](https://github.com/openai/CLIP) 模型。与 SimCLR 一样，CLIP 使用对比学习目标，但它不是对比同一图像的两个增强视图，而是对比两种不同的模态：文本和图像。为了训练 CLIP，OpenAI 从互联网收集了约 4 亿对图像-文本对的大型数据集，包括维基百科和图像 alt 文本等来源。生成的模型学习到了丰富、高层次的图像特征，并在许多视觉基准测试中取得了令人印象深刻的零样本 (zero-shot) 性能。\n",

    "Next, we will explore [DINO](https://github.com/facebookresearch/dino), a self-supervised learning method for vision tasks that applies contrastive learning in a self-distillation framework with multi-crop augmentation strategy. The authors showed that the features learned by DINO ViTs are fine-grained and semantically rich with explicit information about the semantic segmentation of the image.\n": 
    "接下来，我们将探索 [DINO](https://github.com/facebookresearch/dino)，这是一种用于视觉任务的自监督学习方法，它在带有多种裁剪增强策略 (multi-crop augmentation) 的自蒸馏 (self-distillation) 框架中应用了对比学习。作者表明，DINO ViTs 学习到的特征是细粒度的，语义丰富，且包含有关图像语义分割的显式信息。\n",

    "As explained above, CLIP's training objective incorporates both text and images, building upon the principles of contrastive learning. Consider this quote from the SimCLR notebook:\n": 
    "如上所述，CLIP 的训练目标结合了文本和图像，建立在对比学习的原理之上。回想一下 SimCLR notebook 中的这句话：\n",

    ">The goal of the contrastive loss is to maximize agreement between the final vectors **$z_i = g(h_i)$** and **$z_j = g(h_j)$**.\n": 
    ">对比损失 (contrastive loss) 的目标是最大化最终向量 **$z_i = g(h_i)$** 和 **$z_j = g(h_j)$** 之间的一致性。\n",

    "Similarly, CLIP is trained to maximize agreement between two vectors. However, because these vectors come from different modalities, CLIP uses two separate encoders: a transformer-based Text Encoder and a Vision Transformer (ViT)-based Image Encoder. Note that some smaller, more efficient versions of CLIP use a ResNet as the Image Encoder instead of a ViT.\n": 
    "同样，CLIP 被训练为最大化两个向量之间的一致性。但是，因为这些向量来自不同的模态，CLIP 使用了两个独立的编码器：一个基于 Transformer 的文本编码器 (Text Encoder) 和一个基于视觉 Transformer (ViT) 的图像编码器 (Image Encoder)。请注意，CLIP 的一些更小、更高效的版本使用 ResNet 作为图像编码器，而不是 ViT。\n",

    "Run the cell below to visualize the training and inference pipeline of CLIP.\n": 
    "运行下面的单元格以可视化 CLIP 的训练和推理管道。\n",

    "During the pretraining phase, each batch consists of multiple images along with their corresponding captions. Each image is independently processed by an Image Encoder—typically a visual model like a Vision Transformer (ViT) or a Convolutional Neural Network (ConvNet)—which produces an image embedding $I_n$. Likewise, each caption is independently processed by a Text Encoder to generate a corresponding text embedding $T_n$. Next, we compute the pairwise similarities between all image-text combinations, meaning each image is compared with every caption, and vice versa. The training objective is to maximize the similarity scores along the diagonal of the resulting similarity matrix -- that is, the scores for the matching image-caption pairs $(I_n, T_n)$.  Through backpropagation, the model learns to assign higher similarity scores to true matches than to mismatched pairs.\n": 
    "在预训练阶段，每个批次包含多张图像及其对应的标题。每张图像都由图像编码器（通常是如视觉 Transformer (ViT) 或卷积神经网络 (ConvNet) 这样的视觉模型）独立处理，生成图像嵌入 (image embedding) $I_n$。同样地，每个标题也由文本编码器独立处理，生成对应的文本嵌入 $T_n$。接下来，我们计算所有图像-文本组合的成对相似度 (pairwise similarities)，这意味着每张图像都与每个标题进行比较，反之亦然。训练目标是最大化结果相似度矩阵对角线上的相似度得分——即匹配的图像-标题对 $(I_n, T_n)$ 的得分。通过反向传播，模型学会为真正的匹配分配比不匹配对更高的相似度得分。\n",

    "Through this setup, CLIP effectively learns to represent images and texts in a shared latent space. In this space, semantic concepts are encoded in a modality-independent way, enabling meaningful cross-modal comparisons between visual and textual inputs.\n": 
    "通过这种设置，CLIP 有效地学会了在共享的潜在空间 (latent space) 中表示图像和文本。在这个空间中，语义概念以独立于模态的方式被编码，使得在视觉和文本输入之间进行有意义的跨模态比较成为可能。\n",

    "**Inline Question 1** -\n": 
    "**内联问题 1 (Inline Question 1)** -\n",

    "Why does CLIP's learning depend on the batch size? If the batch size is fixed, what strategy can we use to learn rich image features?\n": 
    "为什么 CLIP 的学习依赖于批次大小 (batch size)？如果批次大小是固定的，我们可以使用什么策略来学习丰富的图像特征？\n",

    "$\\color{blue}{\\textit Your Answer:}$\n": 
    "$\\color{blue}{\\textit 你的答案 (Your Answer):}$\n",

    "# Loading COCO dataset\n": 
    "# 加载 COCO 数据集\n",

    "We'll use the same captioning dataset you used to train your RNN captioning model, but instead of generating the captions lets see if we can match each image to the correct caption.": 
    "我们将使用您训练 RNN 图像描述 (captioning) 模型时用到的同一个数据集，但这次我们不生成描述，而是看看我们能否将每张图像与正确的描述匹配起来。",

    "# Returns relative error.\"\"\"\n": 
    "# 返回相对误差。\"\"\"\n",

    "# Load COCO data from disk into a dictionary.\n": 
    "# 将磁盘上的 COCO 数据加载到字典中。\n",

    "# this is the same dataset you used for the RNN captioning notebook :)\n": 
    "# 这是您在 RNN 图像描述 notebook 中用过的相同数据集 :)\n",

    "# Print out all the keys and values from the data dictionary.\n": 
    "# 打印数据字典中的所有键和值。\n",

    "# we're just using the loaded captions from COCO, so we need to decode them and get rid of the special tokens.\n": 
    "# 我们只使用从 COCO 加载的标题，因此我们需要对它们进行解码并去掉特殊标记。\n",

    "# lets get 10 examples\n": 
    "# 让我们获取 10 个示例\n",

    "# the images the captions refer to\n": 
    "# 标题指向的图像\n",

    "# Running the CLIP Model\n": 
    "# 运行 CLIP 模型\n",

    "First we'll use the pretrained CLIP model to extract features from the texts and images separetely.": 
    "首先，我们将使用预训练的 CLIP 模型分别从文本和图像中提取特征。",

    "# You can check the model layers by printing the model.\n": 
    "# 您可以通过打印模型来检查模型层。\n",

    "# CLIP's model code is available at https://github.com/openai/CLIP/tree/main/clip\n": 
    "# CLIP 的模型代码可在 https://github.com/openai/CLIP/tree/main/clip 获得\n",

    "# First, we encode the captions into vectors in the shared embedding space.\n": 
    "# 首先，我们将标题编码为共享嵌入空间中的向量。\n",

    "# Since we're using a Transformer as the text encoder, we need to tokenize the text first.\n": 
    "# 因为我们使用 Transformer 作为文本编码器，所以我们需要先对文本进行分词 (tokenize)。\n",

    "# Sanity check, print the shape\n": 
    "# 健全性检查，打印形状\n",

    "# sanity check, print the shape\n": 
    "# 健全性检查，打印形状\n",

    "# Note: For your implementations, use the above clip_model.encode_text() function. Avoid using clip_model.forward().\n": 
    "# 注意：在您的实现中，请使用上面的 clip_model.encode_text() 函数。避免使用 clip_model.forward()。\n",

    "# Then, we encode the images into the same embedding space.\n": 
    "# 然后，我们将图像编码到相同的嵌入空间中。\n",

    "# Note: For your implementations, use the above clip_model.encode_image() function. Avoid using clip_model.forward().\n": 
    "# 注意：在您的实现中，请使用上面的 clip_model.encode_image() 函数。避免使用 clip_model.forward()。\n",

    "Open `cs231n/clip_dino.py` and implement `get_similarity_no_loop` to compute similarity scores between text features and image features. Test your implementation below, you should see relative errors less than 1e-5.": 
    "打开 `cs231n/clip_dino.py` 并实现 `get_similarity_no_loop` 以计算文本特征和图像特征之间的相似度得分。在下面测试您的实现，您应该看到相对误差小于 1e-5。",

    "# Let's visualize the similarities between our batch of images and their captions.\n": 
    "# 让我们可视化批次中的图像和它们的标题之间的相似度。\n",

    "plt.title(\"Cosine similarity between text and image features\", size=20)\n": 
    "plt.title(\"Cosine similarity between text and image features\", size=20)\n",

    "# Zero Shot Classifier\n": 
    "# 零样本分类器 (Zero Shot Classifier)\n",

    "You will be able to see a high similarity between matching image-caption pairs above. We can leverage this property to design an image classifier that doesn't require any labeled data (i.e., a zero-shot classifier). Each class can be represented using an appropriate natural language description, and any input image will be classified into the class whose description has the highest similarity with the image in CLIP's embedding space.": 
    "在上面您可以看到匹配的图像-标题对之间存在很高的相似度。我们可以利用这个属性来设计一个不需要任何标记数据（即零样本分类器）的图像分类器。每个类可以使用适当的自然语言描述来表示，任何输入图像都将被分类到在 CLIP 的嵌入空间中与该图像相似度最高的描述所在的类中。",

    "Implement `clip_zero_shot_classifier` in `cs231n/clip_dino.py` and test it below. You should be able to see the following predictions:\n": 
    "在 `cs231n/clip_dino.py` 中实现 `clip_zero_shot_classifier` 并在下方测试。您应该能看到以下预测：\n",

    "Run the cell below to visualize the predictions. As you can see, CLIP offers a straightforward way to perform reasonable zero-shot classification across any class taxonomy.\n": 
    "运行下面的单元格以可视化预测。正如您所见，CLIP 提供了一种简单直接的方法，能够在任何类别分类法 (taxonomy) 中执行合理的零样本分类。\n",

    "CLIP was the first model to outperform standard supervised training on ImageNet classification without using any ImageNet images or labels (The original CLIP paper has many such interesting experiments and analysis).\n": 
    "CLIP 是第一个在不使用任何 ImageNet 图像或标签的情况下，在 ImageNet 分类上超越标准监督训练的模型（CLIP 原论文包含许多此类有趣的实验和分析）。\n",

    "# Visualize the zero shot predictions\n": 
    "# 可视化零样本预测结果\n",

    "# Image Retrieval using CLIP\n": 
    "# 使用 CLIP 进行图像检索 (Image Retrieval)\n",

    "Just as we used CLIP to retrieve the matching class name for each image, we can also use it to retrieve matching images from text inputs (semantic image retrieval). Implement the `CLIPImageRetriever` in `cs231n/clip_dino.py` and test it by running the two cells below. The expected top 2 outputs for each query are provided in the comments.": 
    "正如我们使用 CLIP 检索每张图像对应的类别名称一样，我们也可以使用它通过文本输入来检索匹配的图像（语义图像检索）。在 `cs231n/clip_dino.py` 中实现 `CLIPImageRetriever`，并通过运行下面的两个单元格来进行测试。注释中提供了每个查询的预期前两个输出。",

    "**Inline Question 2** -\n": 
    "**内联问题 2 (Inline Question 2)** -\n",

    "CLIP learns to align image and text representations in a shared latent space using a contrastive loss. How would you extend this idea to more than two modalities?\n": 
    "CLIP 学习使用对比损失在共享的潜在空间中对齐图像和文本表示。您将如何将此想法扩展到两种以上的模态？\n",

    "As mentioned earlier, models trained with vanilla contrastive learning methods such as SimCLR and CLIP require very large batch sizes. This makes them computationally expensive and limits their accessibility. Subsequent works, like [BYOL](https://arxiv.org/abs/2006.07733), propose an alternative approach that avoids the need for numerous negative samples by using a student-teacher framework. This method performs surprisingly well and was later adopted by [DINO](https://arxiv.org/abs/2104.14294) .\n": 
    "如前所述，使用诸如 SimCLR 和 CLIP 等普通的对比学习方法训练的模型需要非常大的批次大小 (batch sizes)。这使得它们在计算上非常昂贵并限制了它们的可访问性。随后的研究，如 [BYOL](https://arxiv.org/abs/2006.07733)，提出了一种替代方法，通过使用学生-教师框架 (student-teacher framework) 避免了对大量负样本 (negative samples) 的需求。这种方法表现出奇地好，后来被 [DINO](https://arxiv.org/abs/2104.14294) 所采用。\n",

    "Similar to SimCLR, DINO is trained to maximize the agreement between two vectors derived from different views of the same image. However, unlike SimCLR, DINO uses two separate encoders which are trained differently. The student network is updated via backpropagation to match the outputs of the teacher network. The teacher network is not updated via backpropagation; instead, its weights are updated using an exponential moving average (EMA) of the student's weights. This means that the teacher model evolves more slowly and provides a stable target for the student to learn from.\n": 
    "与 SimCLR 类似，DINO 经过训练可最大化从同一图像的不同视图派生的两个向量之间的一致性。但是，与 SimCLR 不同的是，DINO 使用两个经过不同方式训练的独立编码器。学生网络通过反向传播进行更新以匹配教师网络的输出。教师网络不通过反向传播进行更新；相反，它的权重使用学生权重的指数移动平均 (EMA) 来更新。这意味着教师模型的演化速度更慢，为学生的学习提供了一个稳定的目标。\n",

    "Run the cell below to visualize the DINO training pipeline.": 
    "运行下面的单元格以可视化 DINO 训练管道。",

    "# first let's get rid of the CLIP model that's currently using memory\n": 
    "# 首先让我们移除当前正在使用内存的 CLIP 模型\n",

    "# Uncomment the following if you are using GPU runtime\n": 
    "# 如果您使用的是 GPU 运行时，请取消注释以下内容\n",

    "# Load smallest dino model. ViT-S/8. Here ViT-S has ~22M parameters and\n": 
    "# 加载最小的 dino 模型：ViT-S/8。这里 ViT-S 有大约 22M 参数，\n",

    "# works on 8x8 patches.\n": 
    "# 并且在 8x8 的图块 (patches) 上运行。\n",

    "# the image we will be playing around with\n": 
    "# 我们将用来实验的图像\n",

    "# DINO Attention Maps\n": 
    "# DINO 注意力图 (Attention Maps)\n",

    "Since the loaded DINO checkpoint is based on the ViT architecture, we can visualize what each attention head is focusing on. The code below generates heatmaps showing which patches of the original image the [CLS] token attends to across the various heads in the final layer. Although this model was trained using a self-supervised objective without any explicit instruction to recognize \"structure\" in images, still...\n": 
    "由于加载的 DINO 检查点基于 ViT 架构，我们可以可视化每个注意力头 (attention head) 关注的内容。下面的代码生成热力图，显示了最后一层中各个头的 [CLS] 标记 (token) 关注了原始图像的哪些图块。尽管该模型是使用自监督目标进行训练的，没有明确指示其识别图像中的“结构”，但是...\n",

    "Do you notice any patterns?": 
    "您注意到任何模式了吗？",

    "# Preprocess\n": 
    "# 预处理\n",

    "# Extract attention\n": 
    "# 提取注意力\n",

    "# Plot attention heads\n": 
    "# 绘制注意力头\n",

    "# Extract patch token features and discard [CLS] token.\n": 
    "# 提取图块标记 (patch token) 特征并丢弃 [CLS] 标记。\n",

    "**Inline Question 3**\n": 
    "**内联问题 3 (Inline Question 3)**\n",

    "How do we get the tensor shapes printed above? Explain your answer.\n": 
    "我们是如何得到上面打印的张量形状的？请解释您的答案。\n",

    "# DINO Features\n": 
    "# DINO 特征 (Features)\n",

    "To understand what the model is encoding in each patch, we can visualize the contents of each patch token. Since these embeddings are high-dimensional and difficult to interpret directly, we'll use PCA to identify the directions of highest variance in the feature space.\n": 
    "为了理解模型在每个图块中编码了什么，我们可以可视化每个图块标记 (patch token) 的内容。由于这些嵌入是高维的且难以直接解释，我们将使用主成分分析 (PCA) 来识别特征空间中方差最大的方向。\n",

    "In the next cell, we visualize the three principal directions of variance in the feature space. This reveals the dominant structure that the patch embeddings are capturing.": 
    "在下一个单元格中，我们可视化特征空间中方差的三个主方向。这揭示了图块嵌入所捕获的主导结构。",

    "# Normalize PCA components to [0, 1] for RGB display\n": 
    "# 将 PCA 组件归一化到 [0, 1] 以便 RGB 显示\n",

    "# Reshape to image grid (60x60, 3)\n": 
    "# 重塑为图像网格 (60x60, 3)\n",

    "# Show as image\n": 
    "# 显示为图像\n",

    "**Inline Question 4** -\n": 
    "**内联问题 4 (Inline Question 4)** -\n",

    "What kind of structure do you see in the visualization above? What does it imply when a region consistently appears in a specific color? What does it mean when two regions have distinctly different color? Remember that PCA reveals the directions of highest variance in the feature space across all patches. A patch's color reflects its distinct feature content.\n": 
    "在上面的可视化中，您看到了什么样的结构？当一个区域始终以特定颜色出现时，这意味着什么？当两个区域具有明显不同的颜色时，这意味着什么？请记住，PCA 揭示了跨所有图块的特征空间中方差最大的方向。图块的颜色反映了其独特的特征内容。\n",

    "# A Simple Segmentation Model over DINO Features\n": 
    "# 基于 DINO 特征的简单分割模型 (A Simple Segmentation Model over DINO Features)\n",

    "In the previous section, we saw that DINO features can provide surprisingly good segmentation cues. Now, let's put that idea to the test by training a simple segmentation model on the [DAVIS dataset](https://davischallenge.org). The DAVIS dataset (Densely Annotated VIdeo Segmentation) was created for video object segmentation tasks. It provides frame-by-frame, pixel-level annotations of objects within videos. For this experiment, we'll train our model using the annotations from just a single frame of a video and see how well it performs on the remaining frames of the same.\n": 
    "在前一节中，我们看到 DINO 特征可以提供出奇好的分割线索。现在，让我们在 [DAVIS 数据集](https://davischallenge.org) 上训练一个简单的分割模型来检验这个想法。DAVIS 数据集 (密集注释视频分割) 是为视频对象分割任务而创建的。它提供了视频中对象的逐帧、像素级注释。在这个实验中，我们将仅使用视频单帧的注释来训练我们的模型，并观察它在同一个视频剩余帧上的表现如何。\n",

    "Our model will be intentionally minimal: we'll extract DINO features per patch and train a lightweight per-patch classifier using only the patches from that one annotated frame. Typically, you would train on the full dataset and evaluate on a separate validation set containing different videos. But here, we will test the one-shot capabilities of DINO features.\n": 
    "我们的模型将是有意保持在最简化的水平：我们将提取每个图块的 DINO 特征，并仅使用来自那一帧带注释的图块训练一个轻量级的逐图块分类器 (per-patch classifier)。通常，您会在完整数据集上进行训练，并在包含不同视频的独立验证集上进行评估。但在这里，我们将测试 DINO 特征的一次性 (one-shot) 能力。\n",

    "# A helper class to work with DAVIS dataset.\n": 
    "# 用于处理 DAVIS 数据集的辅助类。\n",

    "# It may take ~5 minutes on the first run of this cell to download the dataset.\n": 
    "# 第一次运行此单元格时可能需要大约 5 分钟来下载数据集。\n",

    "# Get a specific test video. Do NOT change this for submission.\n": 
    "# 获取特定的测试视频。提交作业时请勿更改此项。\n",

    "# Get DINO patch features and corresponding class labels for a middle frame\n": 
    "# 获取中间帧的 DINO 图块特征和相应的类标签\n",

    "Complete the implementation of the `DINOSegmentation` class in `cs231n/clip_dino.py`, and test it by running the two cells below. You should achieve a mean IoU greater than 0.45 on the first test frame and greater than 0.50 on the last test frame. To prevent overfitting on the training patch features, consider designing a very lightweight model (e.g., a linear layer or a 2-layer MLP) and applying appropriate weight decay.\n": 
    "完成 `cs231n/clip_dino.py` 中 `DINOSegmentation` 类的实现，并通过运行下面的两个单元格进行测试。在第一个测试帧上，您应达到大于 0.45 的平均 IoU，在最后一个测试帧上应达到大于 0.50。为防止在训练图块特征上过拟合，请考虑设计一个非常轻量级的模型（例如，线性层或 2 层 MLP）并应用适当的权重衰减 (weight decay)。\n",

    "You may use GPU runtime to speed up training and evaluation. Make sure to rerun the entire notebook if you change runtime type.": 
    "您可以使用 GPU 运行时来加快训练和评估的速度。如果更改了运行时类型，请确保重新运行整个 notebook。",

    "# Test on first, middle, and last frame\n": 
    "# 在第一帧、中间帧和最后一帧进行测试\n",

    "print(f\"Mean IoU on first test frames: {ious[0]:.3f}\")  # should be >0.45\n": 
    "print(f\"Mean IoU on first test frames: {ious[0]:.3f}\")  # 应该 >0.45\n",

    "print(f\"Mean IoU on last test frames: {ious[2]:.3f}\")  # should be >0.50\n": 
    "print(f\"Mean IoU on last test frames: {ious[2]:.3f}\")  # 应该 >0.50\n",

    "Now let's visualize the results. Run the two cells below to display the ground truth and predicted segmentation masks for the first, middle, and last frames. Note that the middle frame is part of the training set, while the other frames are unseen.": 
    "现在让我们可视化结果。运行下面的两个单元格，以显示第一帧、中间帧和最后一帧的真实值 (ground truth) 和预测的分割掩码 (segmentation masks)。请注意，中间帧是训练集的一部分，而其他帧是未见过的。",

    "Now run the following three cells to evaluate and visualize the entire video. You should achieve a mean IoU greater than 0.55. The saved visualization video may take some time to process in Google Drive, but you can download it to your computer and view it locally.\n": 
    "现在运行以下三个单元格来评估和可视化整个视频。您应该达到大于 0.55 的平均 IoU。保存的可视化视频在 Google Drive 中可能需要一些时间来处理，但您可以将其下载到计算机并在本地查看。\n",

    "# Run on all frames\n": 
    "# 在所有帧上运行\n",

    "print(f\"Mean IoU on all frames: {sum(ious) / len(ious):.3f}\")  # should be >0.55\n": 
    "print(f\"Mean IoU on all frames: {sum(ious) / len(ious):.3f}\")  # 应该 >0.55\n",

    "# It might take a while to process in google drive but you can just download it and watch on your computer\n": 
    "# 在 google drive 中可能需要一段时间来处理，但您可以直接下载并在电脑上观看\n",

    "**Inline Question 5** -\n": 
    "**内联问题 5 (Inline Question 5)** -\n",

    "If you train a segmentation model on CLIP ViT's patch features, do you expect it to perform better or worse than DINO? Why should that be the case?\n": 
    "如果您在 CLIP ViT 的图块特征上训练分割模型，您认为它的性能会比 DINO 更好还是更差？为什么会这样？\n"
}

def translate_item(item):
    if isinstance(item, list):
        return [translate_item(x) for x in item]
    elif isinstance(item, str):
        for eng, chn in replacements.items():
            item = item.replace(eng, chn)
        return item
    else:
        return item

for cell in data.get('cells', []):
    if cell['cell_type'] == 'markdown':
        cell['source'] = translate_item(cell['source'])
    elif cell['cell_type'] == 'code':
        cell['source'] = translate_item(cell['source'])

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1, ensure_ascii=False)

print("Translation completed successfully!")
