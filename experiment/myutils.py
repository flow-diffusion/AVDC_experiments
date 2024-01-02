from argparse import ArgumentParser
import sys
sys.path.append('core')
import numpy as np
import torch
from PIL import Image


import imageio as imageio
from unimatch.unimatch import UniMatch
import imageio
from torchvision.utils import draw_bounding_boxes
from utils.flow_viz import flow_to_image
import torch.nn.functional as F
from rigid_transform import *
import time 
from torch import nn
import random

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

class dummy_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = Unet()

def get_flow_model():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')
    parser.add_argument('--feature_channels', type=int, default=128)
    parser.add_argument('--num_scales', type=int, default=2)
    parser.add_argument('--upsample_factor', type=int, default=4)
    parser.add_argument('--num_head', type=int, default=1)
    parser.add_argument('--ffn_dim_expansion', type=int, default=4)
    parser.add_argument('--num_transformer_layers', type=int, default=6)
    parser.add_argument('--reg_refine', type=bool, default=True)
    parser.add_argument('--task', type=str, default='flow')
    args = parser.parse_args(args=[])
    DEVICE = 'cuda:0'

    model = UniMatch(feature_channels=args.feature_channels,
                        num_scales=args.num_scales,
                        upsample_factor=args.upsample_factor,
                        num_head=args.num_head,
                        ffn_dim_expansion=args.ffn_dim_expansion,
                        num_transformer_layers=args.num_transformer_layers,
                        reg_refine=args.reg_refine,
                        task=args.task).to(DEVICE)

    checkpoint = torch.load(args.model, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])

    model.to(DEVICE)
    model.eval()
    model._requires_grad = False
    return model

### predict per frame flow   
def pred_flow_frame(model, frames, stride=1, device='cuda:0'):
    DEVICE = device 
    model = model.to(DEVICE)
    frames = torch.from_numpy(frames).float()
    images1 = frames[:-1]
    images2 = frames[1:]
    flows = []
    flows_b = []
    # print("starting prediction")
    # t0 = time.time()
    for image1, image2 in zip(images1, images2):
        image1, image2 = image1.unsqueeze(0).to(DEVICE), image2.unsqueeze(0).to(DEVICE)
    
        # nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
        #                     int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]
        ### dumb upsampling to (480, 640)
        nearest_size = [480, 640]
        inference_size = nearest_size
        ori_size = image1.shape[-2:]
        
        # print("inference_size", inference_size)
        # resize before inference
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                align_corners=True)
            image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                align_corners=True)
        with torch.no_grad():
            results_dict = model(image1, image2,
                attn_type='swin',
                attn_splits_list=[2, 8],
                corr_radius_list=[-1, 4],
                prop_radius_list=[-1, 1],
                num_reg_refine=6,
                task='flow',
                pred_bidir_flow=True,
            )
        
        flow_pr = results_dict['flow_preds'][-1]
        
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                        align_corners=True)
                flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
                flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]
                
        flows += [flow_pr[0:1].permute(0, 2, 3, 1).cpu()]
        flows_b += [flow_pr[1:2].permute(0, 2, 3, 1).cpu()]
        
    flows = torch.cat(flows, dim=0)
    flows_b = torch.cat(flows_b, dim=0)
    # print(flows.shape)
    # print("end prediction")
    # print(time.time() - t0)
    
    flows = flows.numpy()
    flows_b = flows_b.numpy()
    colors = [flow_to_image(flow) for flow in flows]
    
    return images1, images2, colors, flows, flows_b

def get_bbox_keypoints(img_size, label, r=4):
    w, h = img_size[2], img_size[1]
    x_mult, y_mult = w/100, h/100
    x0, y0, x1, y1 = label["x"]*x_mult, label["y"]*y_mult, (label["x"]+label["width"])*x_mult, (label["y"]+label["height"])*y_mult
    x_stride, y_stride = (x1-x0)/r, (y1-y0)/r
    kps = []
    for i in range(r):
        for j in range(r):
            x = x0 + x_stride * (i+0.5)
            y = y0 + y_stride * (j+0.5)
            kps.append((x, y))
    return kps, ((x0+x1)/2, (y0+y1)/2)

def sample_with_binear(fmap, kp):
    max_x, max_y = fmap.shape[1]-1, fmap.shape[0]-1
    x0, y0 = int(kp[0]), int(kp[1])
    x1, y1 = x0+1, y0+1
    x, y = kp[0]-x0, kp[1]-y0
    fmap_x0y0 = fmap[y0, x0]
    fmap_x1y0 = fmap[y0, x1]
    fmap_x0y1 = fmap[y1, x0]
    fmap_x1y1 = fmap[y1, x1]
    fmap_y0 = fmap_x0y0 * (1-x) + fmap_x1y0 * x
    fmap_y1 = fmap_x0y1 * (1-x) + fmap_x1y1 * x
    feature = fmap_y0 * (1-y) + fmap_y1 * y
    return feature

def warp_kp_with_bilinear(flow, kp):
    max_x, max_y = flow.shape[1]-1, flow.shape[0]-1
    x0, y0 = int(kp[0]), int(kp[1])
    x1, y1 = x0+1, y0+1
    x, y = kp[0]-x0, kp[1]-y0
    flow_x0y0 = flow[y0, x0]
    flow_x1y0 = flow[y0, x1]
    flow_x0y1 = flow[y1, x0]
    flow_x1y1 = flow[y1, x1]
    flow_y0 = flow_x0y0 * (1-x) + flow_x1y0 * x
    flow_y1 = flow_x0y1 * (1-x) + flow_x1y1 * x
    flow = flow_y0 * (1-y) + flow_y1 * y
    new_kp = (np.clip(kp[0]+flow[0], 0, max_x-1), np.clip(kp[1]+flow[1], 0, max_y-1))
    return new_kp

def sample_with_binear_v2(fmap, kp):
    max_x, max_y = fmap.shape[1]-1, fmap.shape[0]-1
    x0, y0 = max(0, int(kp[0])), max(0, int(kp[1]))
    x1, y1 = min(max_x, x0+1), min(max_y, y0+1)
    x, y = max(0, kp[0]-x0), max(0, kp[1]-y0)
    fmap_x0y0 = fmap[y0, x0]
    fmap_x1y0 = fmap[y0, x1]
    fmap_x0y1 = fmap[y1, x0]
    fmap_x1y1 = fmap[y1, x1]
    fmap_y0 = fmap_x0y0 * (1-x) + fmap_x1y0 * x
    fmap_y1 = fmap_x0y1 * (1-x) + fmap_x1y1 * x
    feature = fmap_y0 * (1-y) + fmap_y1 * y
    return feature

def warp_kp_with_bilinear_v2(flow, kp):
    max_x, max_y = flow.shape[1]-1, flow.shape[0]-1
    x0, y0 = max(0, int(kp[0])), max(0, int(kp[1]))
    x1, y1 = min(max_x, x0+1), min(max_y, y0+1)
    x, y = max(0, kp[0]-x0), max(0, kp[1]-y0)
    flow_x0y0 = flow[y0, x0]
    flow_x1y0 = flow[y0, x1]
    flow_x0y1 = flow[y1, x0]
    flow_x1y1 = flow[y1, x1]
    flow_y0 = flow_x0y0 * (1-x) + flow_x1y0 * x
    flow_y1 = flow_x0y1 * (1-x) + flow_x1y1 * x
    flow = flow_y0 * (1-y) + flow_y1 * y
    new_kp = kp[0]+flow[0], kp[1]+flow[1]
    return new_kp    

def warp_points(flow, points):
    warped_points = []
    for kp in points:
        warped_points.append(warp_kp_with_bilinear(flow, kp))
    return np.array(warped_points)

def warp_points_v2(flow, points):
    warped_points = []
    for kp in points:
        warped_points.append(warp_kp_with_bilinear_v2(flow, kp))
    return np.array(warped_points)

def draw_bbox(img, label):
    ### get image size
    img = (img*255).type(torch.uint8)
    w, h = img.shape[2], img.shape[1]
    x_mult = w/100
    y_mult = h/100
    x0, y0, x1, y1 = label["x"]*x_mult, label["y"]*y_mult, (label["x"]+label["width"])*x_mult, (label["y"]+label["height"])*y_mult

    img = draw_bounding_boxes(img, torch.tensor([[x0, y0, x1, y1]]), width=2, colors='red')
    return img.type(torch.float32) / 255

def get_tetrahedron(center, r=0.3):
    points = [
        center,
        (center[0], center[1]+r, center[2]),
        (center[0]+(1/2)**0.5*r, center[1]-0.5*r, center[2]+0.5*r),
        (center[0]-(1/2)**0.5*r, center[1]-0.5*r, center[2]+0.5*r),
        (center[0], center[1]-0.5*r, center[2]-(3/4)**0.5*r),
    ]
    return np.array(points)

def to_3d(points, depth, cmat):
    points = points.reshape(-1, 2)
    depths = np.array([[sample_with_binear(depth, kp)] for kp in points])
    # depths = np.array([[depth[int(p[1]), int(p[0])]] for p in points])
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1) * depths
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    cmat = np.concatenate([cmat, np.array([[0, 0, 0, 1]])], axis=0)
    points = np.dot(np.linalg.inv(cmat), points.T).T
    points = points[:, :3]
    return points

def to_3d_v2(points, depth, cmat):
    points = points.reshape(-1, 2)
    depths = np.array([[sample_with_binear_v2(depth, kp)] for kp in points])
    # depths = np.array([[depth[int(p[1]), int(p[0])]] for p in points])
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1) * depths
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    cmat = np.concatenate([cmat, np.array([[0, 0, 0, 1]])], axis=0)
    points = np.dot(np.linalg.inv(cmat), points.T).T
    points = points[:, :3]
    return points

def to_3d_uvd(points, depth, cmat):
    points = points.reshape(-1, 2)
    depths = np.array(depth)
    # depths = np.array([[depth[int(p[1]), int(p[0])]] for p in points])
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1) * depths
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    cmat = np.concatenate([cmat, np.array([[0, 0, 0, 1]])], axis=0)
    points = np.dot(np.linalg.inv(cmat), points.T).T
    points = points[:, :3]
    return points

def to_2d(points, cmat):
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = np.dot(cmat, points.T).T
    points = points[:, 0:2] / points[:, 2:3]
    return points

def sample_n_frames(frames, n):
    new_vid_ind = [int(i*len(frames)/(n-1)) for i in range(n-1)] + [len(frames)-1]
    return np.array([frames[i] for i in new_vid_ind])

def sample_from_mask(mask, num_samples=100):
    on = np.array(mask.nonzero())[::-1].T.astype(np.float64)
    if len(on) == 0:
        on = np.array((mask==0).nonzero())[::-1].T.astype(np.float64)
    sample_ind = np.random.choice(len(on), num_samples, replace=True)
    ### add +-0.5 uniform noises to the samples
    samples = on[sample_ind]
    samples += np.random.uniform(-0.5, 0.5, samples.shape)
    return samples

def get_grasp(samples, depth, cmat, r=5):
    def loss(i):
        return np.linalg.norm(samples - samples[i], axis=1).sum()
    grasp_2d = samples[np.argmin([loss(i) for i in range(len(samples))])]
    neighbor_threshold = r
    neighbors = samples[np.linalg.norm(samples - grasp_2d, axis=1) < neighbor_threshold]
    neighbors_d = np.array([[sample_with_binear(depth, kp)] for kp in neighbors])
    d = np.median(neighbors_d)
    # print(d)
    # print(grasp_2d)
    return to_3d_uvd(grasp_2d, [d], cmat)

def get_transforms(seg, depth, cmat, flows=[], ransac_tries=100, ransac_threshold=0.5, rgd_tfm_tries=50, rgd_tfm_threshold=1e-3):
    transformss = []
    center_2ds = []
    sampless = []
    samples_2d = sample_from_mask(seg, 500)
    sampless.append(samples_2d)
    samples_3d = to_3d(samples_2d, depth, cmat)
    grasp = get_grasp(samples_2d, depth, cmat)
    # print(grasp.shape)
    # print(samples_3d.shape)
    
    points1_uv = samples_2d
    points1 = samples_3d
    center = grasp
    for i in range(len(flows)):
        flow = flows[i]
        center_uv = to_2d(center, cmat)[0]
        center_2ds.append(center_uv)
        points2_uv = warp_points(flow, points1_uv)
        t0 = time.time()
        _, inliners = ransac(points1_uv, center_uv, points2_uv, ransac_tries, ransac_threshold)
        t1 = time.time()
        # print("inliners:", len(inliners))
        points1_uv = np.array(points1_uv)[inliners]
        points2_uv = np.array(points2_uv)[inliners]
        points1 = np.array(points1)[inliners]
        sampless.append(points2_uv)
        
        solution, mat = solve_3d_rigid_tfm(points1, points2_uv, cmat, rgd_tfm_tries, rgd_tfm_threshold)
        t2 = time.time()

        # print("ransac time:", t1-t0)
        # print("solve time:", t2-t1)
        # print("transform parameters:", solution.x)
        # print("loss:", solution.fun)
        T = get_transformation_matrix(*solution.x)
        
        points1_ext = np.concatenate([points1, np.ones((len(points1), 1))], axis=1)
        points1 = (T @ points1_ext.T).T[:, :3]
        center = (T @ np.concatenate([center, np.ones((1, 1))], axis=1).T).T[:, :3]
        # print("center:", center)
        points1_uv = to_2d(points1, cmat)
        
        transformss.append(solution.x)
    
    return grasp, np.array(transformss), np.array(center_2ds), sampless


def get_inbound_kp_idxs(kps, size):
    h, w = size
    shrink = 4
    return np.array([i for i, kp in enumerate(kps) if kp[0] >= shrink and kp[0] < w-shrink and kp[1] >= shrink and kp[1] < h-shrink])
  
import imageio
def get_transforms_nav(depth, cmat, flows=[], moving_threshold=1.0, rgd_tfm_tries=30, rgd_tfm_threshold=1e-3):
    transformss = []
    sampless = []
    num_samples = 1000
    seg = (np.linalg.norm(flows[0], axis=2) > moving_threshold)
    # imageio.imsave("seg.png", seg.astype(np.uint8)*255)
    samples_2d = sample_from_mask(seg, num_samples)
    sampless.append(samples_2d)
    samples_3d = to_3d_v2(samples_2d, depth, cmat)

    # print(grasp.shape)
    # print(samples_3d.shape)
    
    points1_uv = samples_2d
    points1 = samples_3d
    for i in range(len(flows)):
        flow = flows[i]
        points2_uv = warp_points_v2(flow, points1_uv)
        inliners = get_inbound_kp_idxs(points2_uv, depth.shape[:2])
        # print("inliners:", len(inliners))
        if len(inliners) < num_samples // 10:
            return np.array(transformss)
        points1_uv = np.array(points1_uv)[inliners]
        points2_uv = np.array(points2_uv)[inliners]
        points1 = np.array(points1)[inliners]
        sampless.append(points2_uv)
        
        solution, mat = solve_3d_rigid_tfm(points1, points2_uv, cmat, rgd_tfm_tries, rgd_tfm_threshold)
        # print(solution.fun)
        t2 = time.time()

        # print("ransac time:", t1-t0)
        # print("solve time:", t2-t1)
        # print("transform parameters:", solution.x)
        # print("loss:", solution.fun)
        T = get_transformation_matrix(*solution.x)
        
        points1_ext = np.concatenate([points1, np.ones((len(points1), 1))], axis=1)
        points1 = (T @ points1_ext.T).T[:, :3]
        # print("center:", center)
        points1_uv = to_2d(points1, cmat)
        inliners = get_inbound_kp_idxs(points1_uv, depth.shape[:2])
        if len(inliners) < num_samples // 10:
            return np.array(transformss)
        points1_uv = np.array(points1_uv)[inliners]
        points1 = np.array(points1)[inliners]
        
        transformss.append(solution.x)
    
    return np.array(transformss)

def transforms2actions(transforms, verbose=False):
    actions = []
    for transform in transforms:
        T = get_transformation_matrix(*transform)
        subgoal = np.matmul(T, np.array([0, 0, 1, 1]))[:3]
        if verbose:
            print("subgoal:", subgoal)

        if np.allclose(subgoal, np.array([0, 0, 1]), atol=1e-3):
            actions.append("Done")
            return actions
        elif subgoal[0] > 0.20:
            actions.append("RotateLeft")
        elif subgoal[0] < -0.20:
            actions.append("RotateRight")
        else:
            actions.append("MoveAhead")
    return actions