import os
import time
import argparse
import numpy as np
import saverloader
from fire import Fire
from nets.segnet import Segnet
from nets.segnet_m import Segnet as Segnet_m
from nets.segnet_m_cen import Segnet as Segnet_m_cen
from nets.segnet_m_cen_deform import Segnet as Segnet_m_cen_deform
import utils.misc
import utils.improc
import utils.vox
import random
import nuscenesdataset
import torch
import torch.multiprocessing
import cv2

torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F

import matplotlib

matplotlib.use('Agg')
from nuscenesdataset import get_nusc_maps, fetch_nusc_map2, add_ego2
import matplotlib.pyplot as plt
import imageio
import io

random.seed(125)
np.random.seed(125)

# the scene centroid is defined w.r.t. a reference camera
# which is usually random
scene_centroid_x = 0.0
scene_centroid_y = 1.0
scene_centroid_z = 0.0

scene_centroid_py = np.array([scene_centroid_x,
                              scene_centroid_y,
                              scene_centroid_z]).reshape([1, 3])
scene_centroid = torch.from_numpy(scene_centroid_py).float()

XMIN, XMAX = -50, 50
ZMIN, ZMAX = -50, 50
YMIN, YMAX = -5, 5
bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

Z, Y, X = 200, 8, 200


def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype='float32') / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')


def reshape_transform(tensor, height=7, width=7):
    # result = tensor.reshape(tensor.size(0),
    #                         height, width, tensor.size(2))
    # Bring the channels to the first dimension,like in CNNs.
    result = tensor.transpose(2, 3).transpose(1, 2)
    return result


def run_model(loader, index, model, d, img_dir, device='cuda:0', sw=None):
    imgs_all, rots_all, trans_all, intrins_all, pts0_all, extra0_all, pts_all, extra_all, lrtlist_velo_all, vislist_all, tidlist_all, scorelist_all, seg_bev_g_all, valid_bev_g_all, center_bev_g_all, offset_bev_g_all, radar_data_all, egopose_all = d

    T = imgs_all.shape[1]  # [1, 40, 6, 3, 448, 800]

    nusc_maps = get_nusc_maps(loader.dataset.data_root)
    scene2map = {}
    for rec in loader.dataset.nusc.scene:
        log = loader.dataset.nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']
    dx = loader.dataset.dx[:2]
    bx = loader.dataset.bx[:2]

    for t in range(T):
        # eliminate the time dimension
        imgs = imgs_all[:, t]
        rots = rots_all[:, t]
        trans = trans_all[:, t]
        intrins = intrins_all[:, t]
        pts0 = pts0_all[:, t]
        extra0 = extra0_all[:, t]
        pts = pts_all[:, t]
        extra = extra_all[:, t]
        lrtlist_velo = lrtlist_velo_all[:, t]
        vislist = vislist_all[:, t]
        tidlist = tidlist_all[:, t]
        scorelist = scorelist_all[:, t]
        seg_bev_g = seg_bev_g_all[:, t]
        valid_bev_g = valid_bev_g_all[:, t]
        center_bev_g = center_bev_g_all[:, t]
        offset_bev_g = offset_bev_g_all[:, t]
        radar_data = radar_data_all[:, t]
        egopose = egopose_all[:, t]

        origin_T_velo0t = egopose.to(device)  # B,T,4,4
        lrtlist_velo = lrtlist_velo.to(device)
        scorelist = scorelist.to(device)

        rgb_camXs = imgs.float().to(device)
        rgb_camXs = rgb_camXs - 0.5  # go to -0.5, 0.5

        seg_bev_g = seg_bev_g.to(device)
        valid_bev_g = valid_bev_g.to(device)
        center_bev_g = center_bev_g.to(device)
        offset_bev_g = offset_bev_g.to(device)

        xyz_velo0 = pts.to(device).permute(0, 2, 1)
        rad_data = radar_data.to(device).permute(0, 2, 1)  # B, R, 19
        xyz_rad = rad_data[:, :, :3]
        meta_rad = rad_data[:, :, 3:]

        B, S, C, H, W = rgb_camXs.shape
        B, V, D = xyz_velo0.shape

        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)

        mag = torch.norm(xyz_velo0, dim=2)
        xyz_velo0 = xyz_velo0[:, mag[0] > 1]
        xyz_velo0_bak = xyz_velo0.clone()

        intrins_ = __p(intrins)
        pix_T_cams_ = utils.geom.merge_intrinsics(*utils.geom.split_intrinsics(intrins_)).to(device)
        pix_T_cams = __u(pix_T_cams_)

        velo_T_cams = utils.geom.merge_rtlist(rots, trans).to(device)
        cams_T_velo = __u(utils.geom.safe_inverse(__p(velo_T_cams)))

        cam0_T_camXs = utils.geom.get_camM_T_camXs(velo_T_cams, ind=0)
        camXs_T_cam0 = __u(utils.geom.safe_inverse(__p(cam0_T_camXs)))
        cam0_T_camXs_ = __p(cam0_T_camXs)
        camXs_T_cam0_ = __p(camXs_T_cam0)

        xyz_cam0 = utils.geom.apply_4x4(cams_T_velo[:, 0], xyz_velo0)
        rad_xyz_cam0 = utils.geom.apply_4x4(cams_T_velo[:, 0], xyz_rad)

        lrtlist_cam0 = utils.geom.apply_4x4_to_lrtlist(cams_T_velo[:, 0], lrtlist_velo)

        vox_util = utils.vox.Vox_util(
            Z, Y, X,
            scene_centroid=scene_centroid.to(device),
            bounds=bounds,
            assert_cube=False)

        V = xyz_velo0.shape[1]

        occ_mem0 = vox_util.voxelize_xyz(xyz_cam0, Z, Y, X, assert_cube=False)
        rad_occ_mem0 = vox_util.voxelize_xyz(rad_xyz_cam0, Z, Y, X, assert_cube=False)
        metarad_occ_mem0 = vox_util.voxelize_xyz_and_feats(rad_xyz_cam0, meta_rad, Z, Y, X, assert_cube=False)

        if not (model.module.use_radar or model.module.use_lidar):
            in_occ_mem0 = None
        elif model.module.use_lidar:
            assert (model.module.use_radar == False)  # either lidar or radar, not both
            assert (model.module.use_metaradar == False)  # either lidar or radar, not both
            in_occ_mem0 = occ_mem0
        elif model.module.use_radar and model.module.use_metaradar:
            in_occ_mem0 = metarad_occ_mem0
        elif model.module.use_radar:
            in_occ_mem0 = rad_occ_mem0
        elif model.module.use_metaradar:
            assert (False)  # cannot use_metaradar without use_radar

        cam0_T_camXs = cam0_T_camXs

        lrtlist_cam0_g = lrtlist_cam0

        model.eval()

        raw_e, feat_bev_e, seg_bev_e, center_bev_e, offset_bev_e = model(
            rgb_camXs=rgb_camXs,
            pix_T_cams=pix_T_cams,
            cam0_T_camXs=cam0_T_camXs,
            # vox_util=vox_util,
            rad_occ_mem0=in_occ_mem0)  # feat_bev_e [1, 128, 200, 200]

        # ----------------------------------- add grad-cam ------------------------------- #
        #
        # from grad_cam.pytorch_grad_cam import GradCAM, \
        #     ScoreCAM, \
        #     GradCAMPlusPlus, \
        #     AblationCAM, \
        #     XGradCAM, \
        #     EigenCAM, \
        #     EigenGradCAM, \
        #     LayerCAM, \
        #     FullGrad
        # from grad_cam.pytorch_grad_cam.utils.image import show_cam_on_image
        #
        # methods = \
        #     {"gradcam": GradCAM,
        #      "scorecam": ScoreCAM,
        #      "gradcam++": GradCAMPlusPlus,
        #      "ablationcam": AblationCAM,
        #      "xgradcam": XGradCAM,
        #      "eigencam": EigenCAM,
        #      "eigengradcam": EigenGradCAM,
        #      "layercam": LayerCAM,
        #      "fullgrad": FullGrad}
        #
        # normalized_masks = torch.nn.functional.softmax(seg_bev_e, dim=1).cpu()
        # sem_classes = [
        #     '__background__', 'car'
        # ]
        # sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
        #
        # car_category = sem_class_to_idx["__background__"]  # 1
        # car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()  # (512,512)
        #
        #
        #
        # car_mask_uint8 = 255 * np.uint8(car_mask == car_category)
        # car_mask_float = np.float32(car_mask == car_category)
        # # both_images = np.hstack((seg_g_t_onmap_use, np.repeat(car_mask_float[:, :, None], 3, axis=-1)))
        # # cv2.imwrite(f"seg-cam.jpg", both_images)
        #
        # class SemanticSegmentationTarget:
        #     def __init__(self, category, mask):
        #         self.category = category
        #         self.mask = torch.from_numpy(mask)
        #         if torch.cuda.is_available():
        #             self.mask = self.mask.cuda()
        #
        #     def __call__(self, model_output):
        #         return (model_output[self.category, :, :] * self.mask).sum()
        #
        # targets = [SemanticSegmentationTarget(car_category, car_mask_float)]
        #
        #
        #
        # # import pdb; pdb.set_trace()
        # target_layers = model.module.encoder.layer3
        # cam = methods['gradcam'](model=model,
        #                          target_layers=target_layers,
        #                          use_cuda=True,
        #                          reshape_transform=reshape_transform,
        #                          pix_T_cams=pix_T_cams,
        #                          cam0_T_camXs=cam0_T_camXs,
        #                          rad_occ_mem0=in_occ_mem0)  # gradcam/gradcam++/scorecam/xgradcam/ablationcam
        # cam.batch_size = 32
        # # [1, 6, 3, 448, 800]
        # grayscale_cam = cam(input_tensor=rgb_camXs.requires_grad_(),
        #                     targets=targets,
        #                     eigen_smooth=False,
        #                     aug_smooth=False, )
        #
        # # Here grayscale_cam has only one image in the batch
        # grayscale_cam = grayscale_cam[0, :]
        # grayscale_cam = cv2.resize(grayscale_cam,(200,200))
        #
        # n_cam = rgb_camXs.shape[1]
        # folder_name= os.path.join('./grad-vis', "sample_vis_%03d" % index)
        # os.makedirs(folder_name, exist_ok=True)
        # # for cam_id in range(n_cam):
        # #     camX_t_vis = utils.improc.back2color(rgb_camXs[0, cam_id:cam_id + 1]).cpu().numpy()[0].transpose(1, 2,
        # #                                                                                                      0)  # (448, 800, 3)
        # #     camX_t_vis = np.array(camX_t_vis, dtype=np.float32) / 255.0
        # #     cam_image = show_cam_on_image(camX_t_vis, grayscale_cam)
        # #
        # #     cv2.imwrite(f'{folder_name}/gradcam_cam_{t}.jpg', cam_image)
        # ----------------------------------- add grad-cam ------------------------------- #

        # visualize ground truth
        rec = loader.dataset.ixes[loader.dataset.indices[index][t]]
        car_from_current = np.eye(4)
        car_from_current[:3, :3] = rots[0, 0].cpu().numpy()
        car_from_current[:3, 3] = np.transpose(trans[0, 0].numpy())

        poly_names, line_names, lmap = fetch_nusc_map2(rec, nusc_maps, loader.dataset.nusc, scene2map, car_from_current)

        plt.close('all')
        fig = plt.figure(figsize=(4, 4), frameon=False)
        ax = fig.gca()
        ax.axis('off')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.axis('off')
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
        fig.axes[1].get_xaxis().set_visible(False)
        fig.axes[1].get_yaxis().set_visible(False)
        plt.axis('off')
        line_names = ['road_divider', 'lane_divider']
        for name in poly_names:
            for la in lmap[name]:
                pts = (la - bx) / dx
                plt.fill(pts[:, 1], pts[:, 0], c=(1.00, 0.50, 0.31), alpha=0.2)
        for la in lmap['road_divider']:
            pts = (la - bx) / dx
            plt.plot(pts[:, 1], pts[:, 0], c=(0.0, 0.0, 1.0), alpha=0.5)
        for la in lmap['lane_divider']:
            pts = (la - bx) / dx
            plt.plot(pts[:, 1], pts[:, 0], c=(159. / 255., 0.0, 1.0), alpha=0.5)
        plt.xlim((200, 0))
        plt.ylim((0, 200))
        add_ego2(bx, dx)

        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw')
        io_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        io_buf.close()

        img_arr = rgba2rgb(img_arr)
        img_arr = np.rot90(img_arr, 1)
        img_arr = np.flip(img_arr, axis=1)
        map_vis = torch.from_numpy(img_arr.astype(float) / 255.0 - 0.5)  # H, W, 3
        map_vis = map_vis.unsqueeze(0).permute(0, 3, 1, 2).float().to(rgb_camXs.device)  # 1, 3, H, W

        _, _, mH, mW = map_vis.shape

        blue_img = torch.zeros_like(map_vis).to(map_vis.device)
        blue_img[:, [0, 1]] = -0.5
        blue_img[:, 2] = 0.5
        seg_g_t = F.interpolate(seg_bev_g, (mH, mW))
        seg_g_t_onmap = map_vis * (1 - seg_g_t) + blue_img * seg_g_t

        seg_e_t = torch.sigmoid(F.interpolate(seg_bev_e, (mH, mW)))  # [1, 1, 400, 400]

        # ----------------------------------- vis heatmap ------------------------------- #
        # outputs = feat_bev_e.to('cpu').squeeze(0).detach().numpy()  # (128, 200, 200)
        # outputs_2c = raw_e.to('cpu').squeeze(0).detach().numpy()

        # -------------------------------- 选择要可视化的特征 ------------------------------ #
        outputs = raw_e.to('cpu').squeeze(0).detach().numpy()  # (128, 200, 200)

        # ---------------- 使用center需要把这一部分注释掉 ----------------------- #
        # times = outputs.shape[0]
        # final_out = np.ones_like(outputs)
        # for i in range(times - 1):
        #     final_out = outputs[i, :, :] + outputs[i + 1, :, :]
        # outputs = final_out
        # ---------------- 使用center需要把这一部分注释掉 ----------------------- #
        # outputs_2c = outputs_2c.squeeze(0)  # seg 和 center 用
        outputs = outputs.squeeze(0)

        v_min = outputs.min()
        v_max = outputs.max()
        outputs = (outputs - v_min) / max((v_max - v_min), 1e-10)
        heatmap = cv2.resize(outputs, (200, 200)) * 255
        heatmap_1 = heatmap.astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap_1, cv2.COLORMAP_JET)  #

        # v_min_2c = outputs_2c.min()
        # v_max_2c = outputs_2c.max()
        # outputs_2c = (outputs_2c - v_min_2c) / max((v_max_2c - v_min_2c), 1e-10)
        # heatmap_2c = cv2.resize(outputs_2c, (200, 200)) * 255
        # heatmap_1_2c = heatmap_2c.astype(np.uint8)
        # heatmap_2c = cv2.applyColorMap(heatmap_1_2c, cv2.COLORMAP_JET)  #

        # -------------------------- 保存 heat map 的路径 ------------------------- #
        heat_out = f'vis_output/heat_vov_mamba_bi_cen_deform-lidar/{index}'
        os.makedirs(heat_out, exist_ok=True)
        # -------------------------- 保存 heat map 的路径 ------------------------- #

        n_cam = rgb_camXs.shape[1]  # [1, 6, 3, 448, 800]
        # for cam_id in range(n_cam):
        #     camX_t_vis = rgb_camXs[0, cam_id:cam_id + 1].cpu().numpy()[0].transpose(1, 2, 0)  # (448, 800, 3)
        # superimposed_img = heatmap * 0.5 + camX_t_vis * 0.5
        # cv2.imwrite(f"./{heat_out}/fuse_{cam_id}.jpg", superimposed_img)
        seg_g_t_onmap_use = utils.improc.back2color(seg_g_t_onmap).cpu().numpy()[0].transpose(1, 2, 0)
        seg_g_t_onmap_use = cv2.resize(seg_g_t_onmap_use, (200, 200))

        superimposed_img = seg_g_t_onmap_use * 0.5 + heatmap * 0.5
        cv2.imwrite(f"./{heat_out}/fuse_{t}.jpg", superimposed_img)

        # superimposed_img_2c = seg_g_t_onmap_use * 0.5 + heatmap_2c * 0.5
        # cv2.imwrite(f"./{heat_out}/fuse_{t}_cen_deform.jpg", superimposed_img_2c)
        # imageio.imwrite(f"./{heat_out}/fuse_{t}.png", superimposed_img.astype(np.uint8))

        # ------------------
        # camX_t_vis = np.array(seg_g_t_onmap_use, dtype=np.float32) / 255.0
        # cam_image = show_cam_on_image(camX_t_vis, grayscale_cam)
        # cv2.imwrite(f'{folder_name}/gradcam_cam_{t}.jpg', cam_image)

        if not os.path.exists(f"{heat_out}"):
            os.mkdir(f"./{heat_out}")
            print("yes")

        cv2.imwrite(f"./{heat_out}/heat_1-{t}.jpg", heatmap_1)
        # cv2.imwrite(f"./{heat_out}/heat_1-{t}_cen.jpg", heatmap_1_2c)
        # cv2.imwrite(f"./{heat_out}/heat_2-{t}.jpg", heatmap)
        # ----------------------------------- vis heatmap ------------------------------- #

        seg_e_t_onmap = map_vis * (1 - seg_e_t) + blue_img * seg_e_t
        # save to folder
        folder_name = os.path.join(img_dir, "sample_vis_%03d" % index)
        os.makedirs(folder_name, exist_ok=True)

        seg_g_t_vis = utils.improc.back2color(seg_g_t_onmap).cpu().numpy()[0].transpose(1, 2, 0)
        seg_g_t_vis_name = os.path.join(folder_name, "seg_gt_%03d.png" % t)
        imageio.imwrite(seg_g_t_vis_name, seg_g_t_vis)

        seg_e_t_vis = utils.improc.back2color(seg_e_t_onmap).cpu().numpy()[0].transpose(1, 2, 0)
        seg_e_t_vis_name = os.path.join(folder_name, "seg_et_%03d.png" % t)
        imageio.imwrite(seg_e_t_vis_name, seg_e_t_vis)

        n_cam = rgb_camXs.shape[1]
        for cam_id in range(n_cam):
            camX_t_vis = utils.improc.back2color(rgb_camXs[0, cam_id:cam_id + 1]).cpu().numpy()[0].transpose(1, 2,
                                                                                                             0)  # (448, 800, 3)
            camX_t_vis_name = os.path.join(folder_name, "cam" + str(cam_id) + "_rgb_%03d.png" % t)
            imageio.imwrite(camX_t_vis_name, camX_t_vis)

        if model.module.use_radar:
            radar_t_vis = torch.sum(rad_occ_mem0[0], 2).clamp(0, 1)  # (1, 200, 200)
            radar_t_vis = utils.improc.back2color(radar_t_vis.repeat(3, 1, 1) - 0.5).cpu().numpy().transpose(1, 2, 0)
            radar_t_vis_name = os.path.join(folder_name, "radar_%03d.png" % t)
            imageio.imwrite(radar_t_vis_name, radar_t_vis)

            lidar_t_vis = torch.sum(occ_mem0[0], 2).clamp(0, 1)  # (1, 200, 200)
            lidar_t_vis = utils.improc.back2color(lidar_t_vis.repeat(3, 1, 1) - 0.5).cpu().numpy().transpose(1, 2, 0)
            lidar_t_vis_name = os.path.join(folder_name, "lidar_%03d.png" % t)
            imageio.imwrite(lidar_t_vis_name, lidar_t_vis)


def main(
        exp_name='debug',
        # eval
        max_iters=100000,
        log_freq=100,
        dset='trainval',
        batch_size=1,  # batch size = 1 only
        timesteps=40,  # a sequence is typically 40 frames (20s * 2fps)
        nworkers=0,
        # data/log/save/load directories
        data_dir='/home/oem/nuscenes',
        log_dir='logs_nuscenes_bevseg',
        img_dir='vis_output/vis-vov-mamba-bi-deform-lidar',
        ckpt_dir='checkpoints/',
        keep_latest=1,
        init_dir='/home/oem/桌面/code_bev_seg/simple_bev-main/checkpoints/vov_mamba_bi_center_deform_lidar',
        # vov_mamba_center 8x5_5e-4_rgb12_simple
        ignore_load=None,
        # data
        res_scale=2,
        ncams=6,
        nsweeps=3,
        # model
        encoder_type='vov99',  # vov99 res101
        use_radar=False,        # False
        use_radar_filters=False,
        use_lidar=True,
        use_metaradar=False,  # True
        do_rgbcompress=True,
        # cuda
        device_ids=[0],  # 1 device only for now
):
    B = batch_size
    assert (B % len(device_ids) == 0)  # batch size must be divisible by number of gpus
    device = 'cuda:%d' % device_ids[0]

    # autogen a name
    model_name = "%d" % B
    model_name += "t%d" % timesteps
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)

    # set up loggingg
    os.makedirs(img_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(log_dir, model_name), max_queue=10, flush_secs=60)

    # set up dataloaders
    final_dim = (int(224 * res_scale), int(400 * res_scale))
    print('resolution:', final_dim)

    resize_lim = [1.0, 1.0]
    crop_offset = 0

    data_aug_conf = {
        'crop_offset': crop_offset,
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'H': 900, 'W': 1600,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'ncams': ncams,
    }

    _, dataloader = nuscenesdataset.compile_data(
        dset,
        data_dir,
        data_aug_conf=data_aug_conf,
        centroid=scene_centroid_py,
        bounds=bounds,
        res_3d=(Z, Y, X),
        bsz=B,
        nworkers=nworkers,
        shuffle=False,
        use_radar_filters=use_radar_filters,
        seqlen=timesteps,  # we do not load a temporal sequence here, but that can work with this dataloader
        nsweeps=nsweeps,
        do_shuffle_cams=False,
        get_tids=True,
    )
    dataloader.dataset.data_root = os.path.join(data_dir, dset)
    iterloader = iter(dataloader)

    # ----------------------------- set up model & seg loss ------------------------------ #
    # model = Segnet(Z, Y, X, use_radar=use_radar, use_lidar=use_lidar, use_metaradar=use_metaradar,
    #                do_rgbcompress=do_rgbcompress, encoder_type=encoder_type, rand_flip=False)

    # model = Segnet_m(Z, Y, X, use_radar=use_radar, use_lidar=use_lidar, use_metaradar=use_metaradar,
    #                do_rgbcompress=do_rgbcompress, encoder_type=encoder_type, rand_flip=False)

    # model = Segnet_m_cen(Z, Y, X, use_radar=use_radar, use_lidar=use_lidar, use_metaradar=use_metaradar,
    #                  do_rgbcompress=do_rgbcompress, encoder_type=encoder_type)

    model = Segnet_m_cen_deform(Z, Y, X, use_radar=use_radar, use_lidar=use_lidar, use_metaradar=use_metaradar,
                     do_rgbcompress=do_rgbcompress, encoder_type=encoder_type, rand_flip=False)
    # ----------------------------- set up model & seg loss ------------------------------ #
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    parameters = list(model.parameters())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total_params', total_params)

    # load checkpoint
    global_step = 0
    if init_dir:
        _ = saverloader.load(init_dir, model.module, ignore_load=ignore_load)
        global_step = 0
    requires_grad(parameters, False)
    model.eval()

    while global_step < max_iters:
        global_step += 1

        read_start_time = time.time()

        sw = utils.improc.Summ_writer(
            writer=writer,
            global_step=global_step,
            log_freq=log_freq,
            fps=2,
            scalar_freq=int(log_freq / 2),
            just_gif=True)

        try:
            sample = next(iterloader)
        except:
            break

        read_time = time.time() - read_start_time
        iter_start_time = time.time()

        # run training iteration
        run_model(dataloader, global_step - 1, model, sample, img_dir, device, sw)

        iter_time = time.time() - iter_start_time

        print('%s; step %06d/%d; rtime %.2f; itime %.2f' % (
            model_name, global_step, max_iters, read_time, iter_time))

    writer.close()


if __name__ == '__main__':
    Fire(main)
