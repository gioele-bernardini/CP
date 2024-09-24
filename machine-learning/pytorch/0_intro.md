# Introduction to Tensors

**Tensors are the basic building block of all
machine learning and deep learning**

## Some commands

import torch

// tensor
[3, 224, 224]
which could mean [colur_channel, heihgt, width]
tensore di rango 3 questo sarebbe!

// Scalar
scalar = torch.tensor(7)
scalar.ndim // printa il rango del tensore
scalar.item() ritorna il valore

posso immaginare il tensore come una classe con attributi
ndim, e item() e i vari metodi :)


## Gestione del random

random_tensor = torch.rand(size=(3, 4))
random_tensor, random_tensor.dtype
(dtype ritorna il torch.float32 cioe' il tipo di dato)

zeros = torch.zeros(size=(3, 4))
ones = torch.ones(size=(3, 4))

## Operazioni tra tensori

tensor + 10
tensor * 10
tensor - 10
element-wise moltiplication [1*1, 2*2, 3*3] = [1, 4, 9]
matrix-moltipliation [1*1 + 2*2 + 3*3] = [14]
