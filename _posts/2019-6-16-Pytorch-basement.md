---
layout: post
title:  "The notebook of Pytorch"
date:   2019-06-16 11:50:36 +0530
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

访问[Pytorch][pytorch] 

[pytorch]: https://pytorch.org/docs/stable/search.html?q=optim.SGD&check_keywords=yes&area=default