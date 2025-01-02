import torch
import time
st = time.time()
A = torch.randn(4096, 16384, device="cuda")  # 将矩阵放在 GPU 上
norm_2 = torch.linalg.norm(A, ord=2)
print(norm_2)
end = time.time()
duration = end - st
print(duration)

st = time.time()
w = torch.norm(A)
print(w)
end = time.time()

print(end - st)
