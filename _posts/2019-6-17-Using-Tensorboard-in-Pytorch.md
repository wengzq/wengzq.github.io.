---
layout: post
title:  "Using Tensorboard in Pytroch"
date:   2019-06-17 15:53:36 +0530
categories: Python
---

# 在Pytorch中使用Tensorboard

## 1.配置Pytorch中的tensorboard环境

首先附上tensorboardX的Github地址：[TensorboardX](https://github.com/lanpa/tensorboardX)

**所需环境：anacoda 3, with Pytorch 1.0.0/torchvision0.2.2/tensorboard 1.13.0**

在满足上述需求后可以进行tensorboardX的配置了，安装方法很简单，首先要在Pytorch的envs下安装tensorboad，并且版本需要满足上述要求。安装完毕后再直接安装tensorboardX即可。

```python
pip install tensorboard
pip install tensorboardX
```

在安装完上述两个包以后，使用conda list查看是否安装上，并且查看版本是否正确。

可以进入上面的TensorboardX project中，使用python example/deno.py进行测试。运行代码后，会生成该代码的计算图，然后如同在TF中使用tensorboard一样，使用`tensorboard --logdir 'project path'`语句（在cmd或者anacoda中）让tensorboard生成本地的连接即可。不过要运行这一行命令需要当前环境中有Tensorflow，因此在这个环境下运行下面语句，安装TF(CPU版本即可)。

```python
pip install tensorflow
```

## 2.采坑大户

### 2.1 Num.1坑

一开始使用的tensorflow的whl貌似缺少某个模块（损坏了），这个时候Terminal报错显示tensorflow缺少必要的模组。不过这个是可以检测的，在当前环境使用Python命令行，输入：

```python
import tensorflow
```

0.0/然后，缺少模块安排。

**解决办法：**把这个坏掉的tensorflow安排掉，删除以后，安装完整的tensorflow就可以解决了。

### 2.2 Num.2坑

tensorboard正确运行后，生成的本地连接是http://MSI.6006，结果Chorme无法访问。网页错误。

**解决办法**：在命令行对Host进行限定：

```python
tensorboard --logdir '' --host = 127.0.0.1
```

然后现在生成的Local地址就可以访问了~