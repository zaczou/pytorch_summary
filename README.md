# pytorch_summary
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoadere


x = torch.empty(5, 3)
x = torch.zeros(5, 3, dtype=torch.float)
x = torch.tensor([5.5, 3])  ## np.array([5.5, 3])
x = torch.rand(5, 3)


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
