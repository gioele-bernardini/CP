import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, input_size, hidden_sizes, output_size):
    super(MLP, self).__init__()
    layers = []
    in_size = input_size
    for h_size in hidden_sizes:
      layers.append(nn.Linear(in_size, h_size))
      layers.append(nn.ReLU())
      in_size = h_size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(nn.Sigmoid())
    self.net = nn.Sequential(*layers)
    
  def forward(self, x):
    return self.net(x)
