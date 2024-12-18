import torch
def optimizer_state_scale( x) :
    x_mx_val = torch.max(torch.abs(x)).to(dtype=torch.float32)
    scale_factor = ((2**7) - 1) / x_mx_val
    x_quantized = torch.round(x * scale_factor).to(dtype=torch.int8)
    return x_quantized, scale_factor

def optimizer_state_unscale( x, scale_factor) :
    x = (x.to(dtype=torch.float32) / scale_factor).to(dtype=torch.float32)
    return x
# 示例
tensor = torch.randn(4096, 4096) /1000  # 创建一个示例tensor，范围可能较广
print("Original tensor:")
print(tensor)

# 缩放到int8
scaled_tensor, scale = optimizer_state_scale(tensor)
print("Scaled tensor (int8):")
print(scaled_tensor)

# 反向缩放
unscaled_tensor = optimizer_state_unscale(scaled_tensor, scale)
print("Unscaled tensor:")
print(unscaled_tensor)
# print(tensor)

print((tensor - unscaled_tensor) )
print((tensor - unscaled_tensor) / tensor)

print(torch.mean(torch.abs((tensor - unscaled_tensor) / (tensor+(1e-14)))))