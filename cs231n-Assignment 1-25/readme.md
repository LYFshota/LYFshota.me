# cs231n-25 Windows运行环境搭建指南

## 前言

本仓库适合带GPU的(windows)用户使用，旨在帮助大家快速搭建起适合完成CS231n课程作业的Python环境。
如果你的电脑不带GPU，建议使用Google Colab完成作业。

本仓库适合实战虚拟环境的搭建和管理，推荐使用`uv`工具来管理虚拟环境。如果你不想使用`uv`，也可以使用Python自带的`venv`模块来创建虚拟环境。
当然，社区内也有很多其他的虚拟环境管理工具，比如`conda`，`virtualenv`等，大家可以根据自己的喜好选择使用。

## 如何配置每个作业的环境

### 使用 uv

1. 先前往[uv 官网](https://uv.doczh.com/getting-started/installation/)下载并安装 uv
2. `git clone https://github.com/weijianxian/cs231n-25.git` 克隆代码库到本地
3. `cd cs231n-25`
4. `uv sync`
5. `code .` (使用 VS Code 打开项目)
6. 在 VS Code 终端中选中 `cs231n-25` 环境,在终端中`.venv\Scripts\activate` 激活虚拟环境

### 使用venv

1. `git clone https://github.com/weijianxian/cs231n-25.git` 克隆代码库到本地
2. `cd cs231n-25`
3. `python -m venv .venv --name cs231n-25` 创建虚拟环境
4. `.venv\Scripts\activate` 激活虚拟环境
5. `pip install -r requirments.txt` 安装依赖
6. `code .` (使用 VS Code 打开项目)
7. 在 VS Code 中选中 `cs231n-25` 环境

## CHANGE LOG

1. 不再挂载到Google Drive，改为自行下载数据集
2. 添加如何运行每个作业的说明
3. 删除提交作业相关内容
4. 使用gpt-4o翻译部分代码注释为中文
