## 遇到的挑战和解决方案

### 1.URL无法链接
在测试transformer的时候，要调取使用COCO数据集中对应图片和文字数据，但是发现原本的代码如果读取数据的时候有些URL无法访问就会报错，后面添加遇到URL无法访问就跳过或者换一个URL

### 2.静态检查报错
在Unet模型代码里面对padding进行静态检查的时候，会有红波浪线报错：Argument object | Unknown is not assignable to parameter out_channels，然后发现是提供的函数 default()没有写Python的类型注解，导致静态检查器把它当成了Unknown类型，实际跑起来没什么问题，只是看起来有点强迫症（），后面加个# type: ignore就好了。

### 3.本地运行路径报错
跑 DDPM 的时候，因为我是用的本地环境，没有挂载Google Drive，遇到代码里如Image(f'/content/drive/My Drive/{FOLDERNAME}/unet.png')这种地方就会直接报变量未定义的错。直接删掉Colab的路径前缀，改成相对路径 Image('unet.png') 就好了。

### 4.张量维度重塑报错
在写多头注意力机制的时候，矩阵转置完直接用.view()想恢复形状，结果抛出is not contiguous。查了一下才知道 PyTorch 里转置操作其实没有在内存里真正移动数据，导致内存不连续，这时候直接调view会报错。后来在.view()前面加个.contiguous()重新整理下内存，或者直接用 .reshape()就解决了。