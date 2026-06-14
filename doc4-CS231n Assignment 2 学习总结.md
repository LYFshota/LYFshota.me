## 遇到的挑战和解决方案

### 1. PyTorch与GPU环境配置问题
在第一次装pytorch的时候装成CPU版本了，发现调用不了GPU，后来重新下一个GPU版本的。

### 2. Cython 编译环境报错
做CNN那个Notebook时，为了加速卷积用了Cython写的im2col。刚开始按照提示pip install cython，装完后还是不行，一查发现是缺少C语言的编译工具链，还得装个Microsoft C++ Build Tools。

### 3. 显存溢出
在运行RNN的时候batch size在调试的时候调大了，导致爆显存了。后来就是batch size调小一点，迭代多一点。


## 为什么需要 PyTorch 这样的框架？
- 可以自动求导，不用手动求梯度；
- 可以调用GPU加速计算；
- 封装了很多常用的模型层和优化器，方便使用。