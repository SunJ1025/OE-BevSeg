import numpy as np
import os
import imageio.v2 as imageio
import cv2


vid_id = 17
model_name = "vov-mamba-bi-deform-lidar"  # vov-mamba-bi-deform-radar sample-rgb
folder_name = "vis_output/vis-%s/sample_vis_%03d" % (model_name, vid_id)
print('folder_name', folder_name)
savefolder_name = "vis_output/collage_%s-2/sample_vis_%03d" % (model_name, vid_id)
os.makedirs(savefolder_name, exist_ok=True)
T = 40
seg_h, seg_w = 400, 400
rgb_h, rgb_w = 200, 360

# 定义边框的大小和颜色
top, bottom, left, right = 1, 1, 1, 1
border_color = [0, 0, 0]  # 黑色边框


for t in range(T):
    print(t)
    seg_g = imageio.imread(os.path.join(folder_name, "seg_gt_%03d.png" % t))
    seg_g = cv2.resize(seg_g.astype(np.uint8), (seg_w, seg_h))
    seg_g = np.flip(seg_g, 0)
    seg_g_sig = cv2.copyMakeBorder(seg_g, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color)

    seg_e = imageio.imread(os.path.join(folder_name, "seg_et_%03d.png" % t))
    seg_e = cv2.resize(seg_e.astype(np.uint8), (seg_w, seg_h))
    seg_e = np.flip(seg_e, 0)
    seg_e_sig = cv2.copyMakeBorder(seg_e, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color)

    cam0 = imageio.imread(os.path.join(folder_name, "cam0_rgb_%03d.png" % t))
    cam0 = cv2.resize(cam0.astype(np.uint8), (rgb_w, rgb_h))

    cam1 = imageio.imread(os.path.join(folder_name, "cam1_rgb_%03d.png" % t))
    cam1 = cv2.resize(cam1.astype(np.uint8), (rgb_w, rgb_h))

    cam2 = imageio.imread(os.path.join(folder_name, "cam2_rgb_%03d.png" % t))
    cam2 = cv2.resize(cam2.astype(np.uint8), (rgb_w, rgb_h))

    cam3 = imageio.imread(os.path.join(folder_name, "cam3_rgb_%03d.png" % t))
    cam3 = cv2.resize(cam3.astype(np.uint8), (rgb_w, rgb_h))

    cam4 = imageio.imread(os.path.join(folder_name, "cam4_rgb_%03d.png" % t))
    cam4 = cv2.resize(cam4.astype(np.uint8), (rgb_w, rgb_h))

    cam5 = imageio.imread(os.path.join(folder_name, "cam5_rgb_%03d.png" % t))
    cam5 = cv2.resize(cam5.astype(np.uint8), (rgb_w, rgb_h))

    # radar = imageio.imread(os.path.join(folder_name, "radar_%03d.png" % t))
    # radar = cv2.resize(radar.astype(np.uint8), (seg_w, seg_h))

    # collect into collage

    collage_t = np.zeros((seg_h, rgb_w * 3 + seg_w * 2, 3))  # radar seg_w *3
    collage_t[:rgb_h, :rgb_w] = cam1
    collage_t[:rgb_h, rgb_w:rgb_w*2] = cam0
    collage_t[:rgb_h, rgb_w*2:rgb_w * 3] = cam2
    collage_t[rgb_h:rgb_h*2, :rgb_w] = cam5
    collage_t[rgb_h:rgb_h*2, rgb_w:rgb_w*2] = cam4
    collage_t[rgb_h:rgb_h*2, rgb_w*2:rgb_w * 3] = cam3
    # collage_t[:seg_h, rgb_w*3:rgb_w*3+seg_w] = radar
    # collage_t[:seg_h, rgb_w*3+seg_w:rgb_w*3 + seg_w*2] = seg_e
    # collage_t[:seg_h, rgb_w * 3 + seg_w*2:rgb_w * 3 + seg_w * 3] = seg_g

    collage_t[:seg_h, rgb_w * 3 :rgb_w * 3 + seg_w ] = seg_e
    collage_t[:seg_h, rgb_w * 3 + seg_w :rgb_w * 3 + seg_w * 2] = seg_g


    # collage_t = np.zeros((seg_h + rgb_h*2, seg_w * 2, 3))
    # collage_t[:seg_h, :seg_w] = seg_e
    # collage_t[:seg_h, seg_w:] = seg_g
    # collage_t[seg_h:seg_h+rgb_h, :rgb_w] = cam1
    # collage_t[seg_h:seg_h+rgb_h, rgb_w:rgb_w*2] = cam0
    # collage_t[seg_h:seg_h+rgb_h, rgb_w*2:] = cam2
    # collage_t[seg_h+rgb_h:seg_h+rgb_h*2, :rgb_w] = cam5
    # collage_t[seg_h+rgb_h:seg_h+rgb_h*2, rgb_w:rgb_w*2] = cam4
    # collage_t[seg_h+rgb_h:seg_h+rgb_h*2, rgb_w*2:] = cam3

    collage_t_name = os.path.join(savefolder_name, "collage_%03d.png" % t)
    collage_t_name_e = os.path.join(savefolder_name, "collage_%03d_e.png" % t)
    collage_t_name_g = os.path.join(savefolder_name, "collage_%03d_g.png" % t)
    print(collage_t_name)
    imageio.imsave(collage_t_name, collage_t.astype(np.uint8))
    # imageio.imsave(collage_t_name_e, seg_e_sig.astype(np.uint8))
    # imageio.imsave(collage_t_name_g, seg_g_sig.astype(np.uint8))
