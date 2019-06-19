---
layout: post
title:  "Using Tensorboard in Pytroch（二）"
date:   2019-06-19 15:53:36 +0530
categories: Python
---

## TensorboardX使用笔记

## 1.如何导入tensorboardX包

使用如下语句导入：

```python
from tensorboardX import SummaryWriter
#SummaryWriter encapsulates everything
writer = SummaryWriter('runs/exp-1')
#creates writer object. The log will be saved in 'runs/exp-1'
writer2 = SummaryWriter()
#creates writer2 object with auto generated file name, the dir will be something like 'runs/Aug20-17-20-33'
writer3 = SummaryWriter(comment='3x learning rate')
#creates writer3 object with auto generated file name, the comment will be appended to the filename. The dir will be something like 'runs/Aug20-17-20-33-3xlearning rate'
```

在记录任何东西之前需要先使用SummaryWriter来拟定写入路径等。注意，每次你重新运行程序就需要把名字改掉，因为每次re-run都会重新生成文件。

## 2.使用笔记

```python
# 模式：
add_something(tag name, object, iteration number)
```

### 2.1 Add scalar

**[注意事项：若向其传Pytorch Tensor的话，这个函数会报错。可以使用x.item() if x is a torch scalar tensor来获得标量数值。]**

```python
# 标准情况下示例：
writer.add_scalar('myscalar', value, iteration)
# 使用实例：(多个得时候，可以使用字典来保存object)注意，多个得时候add后缀事scalars
for n_iter in range(100):
	writer.add_scalar('data/scalar1', dummy_s1[0], n_iter)
    ...
    writer.add_scalars('data/scalar_group', {'xsinx': n_iter * np.sin(n_iter),
                                             'xcosx': n_iter * np.cos(n_iter),
                                             'arctanx': np.arctan(n_iter)}, n_iter)
```

### 2.2 Add image

**[注：记得对图片进行Normalize]**

由于图片往往是由三个维度得tensor构成的，最简单的办法就是一次性把这三个维度都存进去。这意味着按照上面的说法，需要作为一个具有[s, h, w]的3-dim tensor传入。这三个维度分别对应图像的R,G,B颜色通道，在图片进行运算后，就可以进行如下操作：

```python
# 标准示例：
		writer.add_image('imresult', x, iteration)
# 如果你有一个批量的图片需要显示，就需要使用torchvision.utils的make_grid函数来对图片进行预排# 序处理，然后再将图片传递给add_image()。（PS：make_grid输入一个4-dim tensor返回一个 # 3-dim的tensor）
# 实例：
        x = vutils.make_grid(dummy_img, normalize=True, scale_each=True)
        writer.add_image('Image', x, n_iter)
```

### 2.3 Add histogram

保存一个直方图是非常的消耗资源的，不论是计算时间还是存储时间。因此如果在程序执行过程中发现训练非常缓慢，可以优先检查是不是这个package在作怪。

```python
# 示例：
writer.add_histogram('hist', array, iteration)
# 实例
 for name, param in resnet18.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)
```

### 2.4 Add figure

载入matplotlib中的数据需要使用add_figure函数，输入的数据需要是matplotlib.pyplot.figure（或者这个类型的列表）。

```python
# 示例：（需要使用matplotlib包）
add_figure(tag, figure, global_step=None, close=True, walltime=None)
```

### 2.5 Add graph

加入一个计算图来可视化模型意味着需要Model（m）以及input(t)，t可以是一个tensor或者是存储tensor的一个列表，**如果出错的话请先验证m(t)是否出错。**

```python
# 示例：
add_graph(model, input_to_model=None, verbose=False, **kwargs)
# 实例：
class LinearInLinear(nn.Module):
    def __init__(self):
        super(LinearInLinear, self).__init__()
        self.l = nn.Linear(3, 5)

    def forward(self, x):
        return self.l(x)

with SummaryWriter(comment='LinearInLinear') as w:
    w.add_graph(LinearInLinear(), dummy_input, True)
```

0.0/后续用到的会再添加。