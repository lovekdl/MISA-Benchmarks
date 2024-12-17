import torch
x = torch.tensor(1.412, dtype=torch.float8_e4m3fn)
print(x)
print(x.to(torch.float32))