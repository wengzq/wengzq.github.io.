---
layout: post
title:  "The notebook of Pytorch（一）"
date:   2019-06-16 15:53:36 +0530
categories: Python
---

# Pytorch从零到无笔记

## 1.使用torch.nn来让网络变得简单可用

使用nn.Module与nn.Parameter模块来构造基本框架：

```python
from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias
 #instance the module
model = Mnist_Logistic()
```

当然上述的过程还是不够简便，**使用nn.Linear()**让nn.Parameter()对象更简单：

```python
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)
```

**直接使用optim对象来帮助我们进行优化**

只需要简单两步：

```python
import torch.optim as opt
opt.step() #优化方法载入，然后自动前进计算
opt.zero_grad() #在计算下一个Batch之前需要call它来清空归零上一次的grad
```

使用**Dataset**来简化数据处理

Dataset对象中包含有诸如`__len__`，`__getitem__`等方法，该对象中的TensorDataset帮助我们将变量变得更易迭代和使用，在没有使用它之前我们读入数据可能会采用如下方式：

```python
xb = x_train[start_i:end_i]
yb = y_train[start_i:end_i]
```

现在：

```python
train_ds = TensorDataset(x_train, y_train)
#使得数据更容易迭代，更容易进行切片处理
xb,yb = train_ds[i*bs : i*bs+bs]
```

**这还远远不够，在训练过程中，我们常常把训练数据打包成为一个个minibatch来进行训练，来增加收敛速度并且防止某些opt方法梯度消失。使用Dataloader可以将训练数据按照一个个的Batch进行打包：**

```python
from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)
#现在在做循环的时候就不需要在对数据手动切片了
model, opt = get_model()

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
```

## 2.添加验证集

**训练过程中需要添加验证集来保证：1.训练没有过拟合；2.添加交叉验证来调参。**

同样可以使用上述介绍的TensorDataset与DataLoader来进行处理，顺带一提的是，由于validation过程相当于一个test过程，所以数据打乱对他来说并没有很大的意义。

```python
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
#DataLoader中的shuffle参数可以将数据的顺序打乱
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
#两个东西共用很多，所以可以定义成一个方法调用
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```

当然，验证集得到的结果同样不需要进行反向传播，这意味着我们不需要保留它的梯度。

```python
model, opt = get_model()

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))
```

## 3.Create ur own CNN

**一整个训练可以分成如下步骤：1.数据载入；2.定义网络结构与前向传播过程；3.设置loss与optim；4.训练**

可以使用torch.nn模块中的nn.Conv2d()来一步一个脚印的搭建网络（这意味着需要一步一步的写前向传播过程）

```python
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

lr = 0.1
```

**Pytorch提供了一个更加handy的方法，即序列nn.Sequenial()**

由于Pytorch没有view() layer来处理Tensor的shape，所以可以先定义一个Lambda对象来处理该问题：

```python
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(-1, 1, 28, 28)
```

这样就可以开始搭建Model了：

```python
model = nn.Sequential(
    Lambda(preprocess), #使用preprocess对tensor进行处理
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),#把x处理成一维的tensor了
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```

## 4.使用GPU

在使用GPU之前，需要先**创建设备对象**：

```python
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
```

然后只需要调用.to(dev)就可以在tensor,model等等上使用gpu了：

```python
#instance 1
def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)

#instance 2
model.to(dev)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
```



## 5.保存模型与加载模型

Pytorch的模型保存主要包括三种模式：**保存state_dict,whole model,checkpoint**

一般使用第一种(fast, easy)，下面是例子：

```python
torch.save(model.state_dict(), PATH)
```

**这里的PATH是file本身而不是指向某个已经存在的文件夹。否则会报错。**

state_dict()主要保存了网络参数信息（包括权重）以及优化器信息。在恢复模型的时候，需要先实例化模型，然后通过load_state_dict(torch.load())来进行恢复。同时使用model.eval()来恢复dp或者bn的信息。

第二种办法示例：

```python
torch.save(model, PATH)
#载入
model = torch.load(PATH)
model.eval()
```

对于最后一种checkpoint的方法，可以自定义想要保存的项目（dict）

```python
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)

#重载模型
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()
```

**Pytorch支持在CPU/GPU端存储，并在GPU/CPU端重载模型：**

这里只记录在CPU上保存在GPU上重载：

```python
torch.save(model.state_dict(), PATH)

#重载在GPU上
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model
```

