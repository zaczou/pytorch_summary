# pytorch_summary
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoadere


x = torch.empty(5, 3)
x = torch.zeros(5, 3, dtype=torch.float)
x = torch.rand(5, 3)

x = torch.Tensor(3, 4)  ## 是有值的
y = torch.tensor(3, 4)  ## 这行是错误的，因为tensor需要确切的数据值

Tensor主要是创建多维矩阵的，标量从某种意义上，不算矩阵。所以Tensor可以通过赋值多维矩阵的方式创建，但是无法指定标量值进行创建。
同时，Tensor可以指定多维矩阵形状的大小，并且默认的数据类型是FloatTensor。
如果想创建单个值，采用[5.6] 这种形式，指定一行一列的矩阵。
x = torch.Tensor(3, 4)
torch.nn.init.xavier_normal(x)
x = torch.Tensor(5.6) ## 错误

tensor主要是根据确定的数据值进行创建，无法直接指定形状大小，需要根据数据的大小进行创建。但同时，tensor没有赋值数据值是矩阵的限制，可以直接使用
x = tensor(5.6) #标量
x = torch.tensor([5.5, 3])  ## 确定数据赋值，np.array([5.5, 3])

x.size()  ## x.shape
x.view(-1, 8)  ## x.reshape()

y = x.numpy()
xx = torch.from_numpy(x)

torch.cuda.is_available()
device = torch.device("cuda")
x = x.to(device)

class Model(nn.Module):
    def __init__(self, input_size, output_size):
	    super(Model, self).__init__()
		
		
	def forward(self, x)
		
		
nn.Embedding.from_pretrained(word_mat, freeze=False)
		
		
		
		
model = Model(12, 20)
print(model)
criterion = nn.CrossEntropyLoss
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
RMSprop

optimizer.zero_grad()

out = model(data)
loss = criterion(out, labels)
loss.backward()
optimizer.step()
total_loss += loss.item()
```
