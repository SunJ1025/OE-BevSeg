import torch.nn as nn
import os
import torch
import os
from torchvision.transforms import ToTensor
import cv2
import numpy as np


class ConvVAE(nn.Module):
    # [64, 128, 256, 512, 1024]  [16, 32, 64, 128] [32, 64, 128, 256]
    def __init__(self, hiddens=[64, 128, 256, 512, 1024], latent_dim=128) -> None:
        super().__init__()

        prev_channels = 3
        modules = []
        img_length = 256
        for cur_channels in hiddens:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels,
                              cur_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1), nn.BatchNorm2d(cur_channels),
                    nn.ReLU()))
            prev_channels = cur_channels
            img_length //= 2
        self.encoder = nn.Sequential(*modules)

        self.mean_linear = nn.Linear(prev_channels * img_length * img_length,
                                     latent_dim)
        self.var_linear = nn.Linear(prev_channels * img_length * img_length,
                                    latent_dim)
        self.latent_dim = latent_dim

        modules = []
        self.decoder_projection = nn.Linear(
            latent_dim, prev_channels * img_length * img_length)
        self.decoder_input_chw = (prev_channels, img_length, img_length)
        for i in range(len(hiddens) - 1, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddens[i],
                                       hiddens[i - 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hiddens[i - 1]), nn.ReLU()))
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hiddens[0],
                                   hiddens[0],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hiddens[0]),
                nn.ReLU(),
                nn.Conv2d(hiddens[0], 3, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            ))
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded_0 = self.encoder[0](x)
        encoded_1 = self.encoder[1](encoded_0)
        encoded_2 = self.encoder[2](encoded_1)
        encoded_3 = self.encoder[3](encoded_2)
        encoded_4 = self.encoder[4](encoded_3)

        encoded = torch.flatten(encoded, 1)

        mean = self.mean_linear(encoded)
        logvar = self.var_linear(encoded)
        eps = torch.randn_like(logvar)
        std = torch.exp(logvar / 2)
        z = eps * std + mean

        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))

        decoded_0 = self.decoder[0](x)
        decoded_1 = self.decoder[1](decoded_0)
        decoded_2 = self.decoder[2](decoded_1)
        decoded_3 = self.decoder[3](decoded_2)
        decoded_4 = self.decoder[4](decoded_3)

        return decoded_4, mean, logvar  # encoded_3

    def sample(self, device='cuda'):
        z = torch.randn(1, self.latent_dim).to(device)
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        decoded = self.decoder(x)
        return decoded


device = 'cuda:0'
model = ConvVAE()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to('cpu')

# 选择权重 注意选择权重时看是哪一个类别的权重
# model.load_state_dict(torch.load('/home/sun/store/UAV/uav-stallite-cross-view/train_model_1/best_model_70'))
model.eval()

# 01-18 19-36 37-54
image_path = '/home/oem/桌面/code_bev_seg/simple_bev-main/vis/sample_vis_055/cam0_rgb_019.png'
image = cv2.imread(image_path)  # 用cv2加载原始图像
image = cv2.resize(image, (256, 256))
transform = ToTensor()
tensor_image = transform(image).unsqueeze(0)

outputs = model(tensor_image)[0]
outputs = outputs.squeeze(0).detach().numpy()  #
print(outputs.shape)

times = outputs.shape[0]
final_out = np.ones_like(outputs)
for i in range(times - 1):
    final_out = outputs[i, :, :] + outputs[i + 1, :, :]
outputs = outputs / times
outputs = final_out
# outputs = (outputs[0,:,:] + outputs[1,:,:] + outputs[2,:,:])/3

v_min = outputs.min()
v_max = outputs.max()
outputs = (outputs - v_min) / max((v_max - v_min), 1e-10)

heatmap = cv2.resize(outputs, (256, 256)) * 255
heatmap_1 = heatmap.astype(np.uint8)

heatmap = cv2.applyColorMap(heatmap_1, cv2.COLORMAP_JET)

superimposed_img = heatmap * 0.5 + image * 0.5

# high midd low 
heat_out = 'heat_out/'
if not os.path.exists(f"{heat_out}"):
    os.mkdir(f"./{heat_out}")
    print("yes")
cv2.imwrite(f"./{heat_out}/test.jpg", superimposed_img)
cv2.imwrite(f"./{heat_out}/test_1.jpg", heatmap_1)
cv2.imwrite(f"./{heat_out}/test_2.jpg", heatmap)

# 保存原始图像
os.system('cp %s %s/query.jpg' % (image_path, f"./{heat_out}"))
