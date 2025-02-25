import torch
import numpy as np

def create_rectangular_mask(h, w, start, end):
    mask = np.zeros((h, w), dtype=bool)
    mask[start:end, start:end] = True
    return mask

def extract_rectangular_annular_regions(input_tensor):
    # 确保输入张量的形状为[1, 1, 200, 200]
    assert input_tensor.shape == (1, 1, 10, 10), "Input tensor shape must be [1, 1, 200, 200]"

    h, w = input_tensor.shape[2], input_tensor.shape[3]
    center = (w // 2, h // 2)

    # 创建矩形掩码
    mask_80 = create_rectangular_mask(h, w, center[0] - 4, center[0] + 4)
    mask_140 = create_rectangular_mask(h, w, center[0] - 6, center[0] + 6)
    mask_200 = create_rectangular_mask(h, w, center[0] - 10, center[0] + 10)

    # 转换掩码为张量
    mask_80 = torch.from_numpy(mask_80).unsqueeze(0).unsqueeze(0).float()
    mask_140 = torch.from_numpy(mask_140).unsqueeze(0).unsqueeze(0).float()
    mask_200 = torch.from_numpy(mask_200).unsqueeze(0).unsqueeze(0).float()

    # 提取80x80的矩形区域
    region_80 = input_tensor * mask_80
    print(input_tensor)
    print(region_80)
    # 提取80-140的环状区域
    mask_80_140 = mask_140 - mask_80
    print(mask_80_140)
    region_80_140 = input_tensor * mask_80_140

    # 提取140-200的环状区域
    mask_140_200 = mask_200 - mask_140
    region_140_200 = input_tensor * mask_140_200

    return region_80, region_80_140, region_140_200

# 示例输入张量
input_tensor = torch.randn(1, 1, 10, 10)

# 提取各个区域
region_80, region_80_140, region_140_200 = extract_rectangular_annular_regions(input_tensor)

print(f"80x80 region shape: {region_80.shape}")
print(f"80-140 annular region shape: {region_80_140.shape}")
print(f"140-200 annular region shape: {region_140_200.shape}")

# 可视化各个区域
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(region_80.squeeze().numpy(), cmap='gray')
axs[0].set_title('80x80 Rectangular Region')
axs[0].axis('off')

axs[1].imshow(region_80_140.squeeze().numpy(), cmap='gray')
axs[1].set_title('80-140 Rectangular Annular Region')
axs[1].axis('off')

axs[2].imshow(region_140_200.squeeze().numpy(), cmap='gray')
axs[2].set_title('140-200 Rectangular Annular Region')
axs[2].axis('off')

plt.show()

