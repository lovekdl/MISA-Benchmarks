import torch


try:
    x = torch.norm(torch.tensor([1, 2]))
    print(x)
except RuntimeError :
    print("ok")