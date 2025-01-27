#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import DeformModel
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image


def feature_to_rgb(features):
    # Input features shape: (16, H, W)
    features = features / (torch.norm(features, dim=-1, keepdim=True) + 1e-9)
    
    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')

    return rgb_array

def show_masks(segmasks):
    img = np.zeros((segmasks.shape[0], segmasks.shape[1], 3))

    for mask_id in np.unique(segmasks):
        if mask_id:
            img[segmasks == mask_id] = 255 * np.random.random(3)
    
    return img.astype('uint8')

def get_feature(x, y, view, gaussians, pipeline, background, scaling_modifier, override_color, d_xyz, d_rotation, d_scaling, patch=None):
    with torch.no_grad():
        render_feature_dino_pkg = render(view, gaussians, pipeline, background, d_xyz, d_scaling, d_rotation, )
        image_feature_dino = render_feature_dino_pkg["render_feature_map"]
    if patch is None:
        return image_feature_dino[:, y, x]
    else:
        a = image_feature_dino[:, y:y+patch[1], x:x+patch[0]]
        return a.mean(dim=(1,2))
    
def calculate_selection_score(features, query_feature, score_threshold=0.7):
    features /= features.norm(dim=-1, keepdim=True)
    query_feature /= query_feature.norm(dim=-1, keepdim=True)
    scores = features.half() @ query_feature.half()
    scores = scores[:, 0]
    mask = (scores >= score_threshold).float()
    return mask

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, deform, load2gpu_on_the_fly):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gt_segmasks = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_seg")
    render_features_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_features")
    segment_objects_path = os.path.join(model_path, name, "ours_{}".format(iteration), "segment_objects")
    pred_masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "pred_masks")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gt_segmasks, exist_ok=True)
    makedirs(render_features_path, exist_ok=True)
    makedirs(segment_objects_path, exist_ok=True)
    makedirs(pred_masks_path, exist_ok=True)
    to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
    
    points = [eval(point) for point in args.points] if args.points is not None else None
    # thetas = [eval(theta) for theta in args.thetas] if args.thetas is not None else None
    # thetas = [0.7 for theta in args.thetas] if args.thetas is not None else None
    
    if points is not None:
        thetas = [0.7 for i in range(len(points))]
        print(thetas)
    selected_indices = None
    if points is not None:
        selected_indices = []
        view = views[0]
        if load2gpu_on_the_fly:
            view.load2device()
        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        # color = [[0,10,0],[10,0,0],[0,0,10],[10,10,0],[0,10,10],[10,0,10],[10,10,10]]
        semantic_features = gaussians.get_seg_features
        for i in range(len(points)):
            # print(points)
            query_feature = get_feature(points[i][0], points[i][1], view, gaussians, pipeline, background, 1.0,
                                         semantic_features[:,0,:], d_xyz, d_rotation, d_scaling, patch = (5,5))
            mask = calculate_selection_score(semantic_features, query_feature, score_threshold = thetas[i])
            indices_above_threshold = np.where(mask.cpu().numpy() >= thetas[i])[0]
            # if selected_indices is None:
            #     selected_indices = indices_above_threshold
            # else:
            #     selected_indices
            selected_indices.append(indices_above_threshold)
        selected_indices = np.concatenate(selected_indices, axis=0)
        print(selected_indices.shape)
        selected_indices = np.unique(selected_indices)
        print(selected_indices.shape)
        print(selected_indices)
        if load2gpu_on_the_fly:
            view.load2device(data_device='cpu')
        
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        rendering_pkg = render(view, gaussians, pipeline, background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling)
        rendering = rendering_pkg["render"]
        rendering_features = rendering_pkg["render_feature_map"]
        
        gt = view.original_image[0:3, :, :]
        gt_seg = view.masks

        # Visualize rendered feature map (by applying PCA)
        rendering_features_rgb = feature_to_rgb(rendering_features)

        # Show all the masks on the image
        gt_seg_rgb = show_masks(gt_seg.cpu().numpy().astype(np.uint8))
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(rendering_features_rgb).save(os.path.join(render_features_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(gt_seg_rgb).save(os.path.join(gt_segmasks, '{0:05d}'.format(idx) + ".png"))
        if selected_indices is not None:
            # print("hi")
            # dummy_indices = np.arange(xyz.shape[0])
            mask = np.zeros(xyz.shape[0])
            # print(mask)
            mask[selected_indices] = 1
            mask = mask.astype('bool')
            # print(mask.sum())
            buffer_image = render(view, gaussians, pipeline, background, d_xyz, d_scaling, d_rotation, indices=mask, override_color=torch.ones(xyz.shape[0], 3).cuda().float())['render']
            buffer_image[buffer_image < 0.5] = 0
            buffer_image[buffer_image != 0] = 1
            inlier_mask = buffer_image.mean(axis=0).bool()
            
            torchvision.utils.save_image(buffer_image.cpu(), os.path.join(pred_masks_path, '{0:05d}'.format(idx) + ".png"))
            
            buffer_image = render(view, gaussians, pipeline, background, d_xyz, d_scaling, d_rotation, indices=mask)['render']
            buffer_image[:, ~inlier_mask] = 0
            
            
            # import matplotlib.pyplot as plt
            # plt.imshow(to8b(buffer_image).transpose(1,2,0))
            # plt.show()
            
            
            torchvision.utils.save_image(buffer_image.cpu(), os.path.join(segment_objects_path, '{0:05d}'.format(idx) + ".png"))
        if load2gpu_on_the_fly:
            view.load2device(data_device='cpu')

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        deform = DeformModel()
        deform.load_weights(dataset.model_path, iteration=iteration)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, deform, dataset.load2gpu_on_the_fly)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, deform, dataset.load2gpu_on_the_fly)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--points', nargs='+', default=None)
    parser.add_argument('--thetas', nargs='+', default=None)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)