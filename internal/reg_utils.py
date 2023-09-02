from cgi import test
import math
import functools
from operator import mod
from pkgutil import walk_packages
from typing import Optional
from absl import flags
from functools import partial

import jax
from jax import random
import jax.numpy as jnp

from internal import math
from internal import utils
from internal import models
from internal import reg_models

FLAGS = flags.FLAGS

@partial(jax.jit, static_argnums = [0, 1, 2, 3, 4])
def reg_train_step(render_pfn, reg_model, data_type, orig_size, reg_type, rng, state, params, batch, alpha, step): 
    """
    batch[pixels] : GT image of seen image
    batch[rays] : Rays from transformed pose
    batch[seen_pose] : GT pose of seen image
    """
    rng = random.split(rng)
    key_0, key_1 = rng

    if data_type == "blender":
        img_H = 800
        img_W = 800
    elif data_type == "llff":
        img_H = 378
        img_W = 504
    elif data_type == "dtu":
        img_H = 300
        img_W = 400
    
    ray_H = 60
    ray_W = 60
    
    orig_ray_H = orig_size
    orig_ray_W = orig_size

    downsample = orig_ray_H // ray_H
    
    rays = utils.namedtuple_map(lambda r: r.reshape(ray_H,ray_W,-1), batch["reduced_rays"])
    orig_rays = utils.namedtuple_map(lambda r: r.reshape(orig_ray_H,orig_ray_W,-1), 
                                        batch["orig_rays"])

    # Receiving multiple images
    gt_img, gt_pose, gt_pose_inv, camtopix = [], [], [], []
    for k in range(batch["pixels"].shape[0]):
        gt_img.append(batch["pixels"][k].reshape(img_H,img_W,3))
        gt_pose.append(batch["seen_pose"][k].reshape(4,4))
        gt_pose_inv.append(batch["seen_pose_inv"][k].reshape(4,4))

        if data_type == "dtu":
            camtopix.append(batch["camtopix"].reshape(3,3))
        else:
            camtopix = None

    unseen_gt = None
    def warp_loss_fn(variables):
        # Loss initialization
        loss = 0.0
        total_reg_loss = 0.0
        w_rgb_loss = 0.0
        # fin_reg_loss = 0.0
        fin_rgb_loss = 0.0
        lr_threshold =  2 * jnp.exp(- step / FLAGS.occ_decay_param)

        # Inverse warping
        pred_color, pred_depth, pred_acc, unmasked_img, mask_img = models.render_image(
            functools.partial(render_pfn, variables),
            rays,
            key_0,
            alpha,
            orig_ray_H,
            orig_rays,
            chunk=FLAGS.chunk,
            match_render=True,
            gt_view=True,  # False
            gt_img=gt_img,
            gt_pose=gt_pose,
            gt_pose_inv=gt_pose_inv,
            camtopix=camtopix,
            eval_mode = False,
            reshape = True,
            lr_threshold=lr_threshold,
            data_type = data_type
        )

        length = unmasked_img[0].shape[0]
        prcp_img = unmasked_img
        d_rgb = []
        per_layer_difmaps = [[] for k in range(FLAGS.layer_depth)]
        
        pred_differences = [[pred_color, wrp_img] for wrp_img in prcp_img]
        warp_differences = [] # under construction
        
        for num, (img_p, img_w) in enumerate(pred_differences):
            # Difference map lists initialization
            mask_img = jax.lax.stop_gradient(mask_img)

            if FLAGS.use_reg_models:
                mask = mask_img[num] if num < len(mask_img) else None
                img_w_grad = img_w
                img_w = jax.lax.stop_gradient(img_w) 
                total_reg_loss, L1_dif_maps = prcp_difference(reg_model, params, 
                                                                img_w, 
                                                                img_p, 
                                                                mask, 
                                                                data_type,
                                                                layer_depth=FLAGS.layer_depth,
                                                                )
                if num == 0:
                    fin_reg_loss = total_reg_loss

                    if FLAGS.use_sem_loss:   
                        param_decay = jnp.exp(-step/FLAGS.decay_param)
                        decayed_reg_loss = param_decay * total_reg_loss
                        loss += decayed_reg_loss
    
        if FLAGS.use_photo_calculation:
            num = 0

            w_rgb_param = FLAGS.photo_weight 
            img_w = jax.lax.stop_gradient(unmasked_img[0]) 
                            
            p_pred_color = jax.image.resize(pred_color, (length, length, 3), 'bicubic')
            rgb_mask = jax.image.resize(mask_img[num][...,None], (length, length, 1), 'nearest')

            rgb_diff_map = l1_difference_map(p_pred_color, img_w, rgb_mask.reshape(length, length))
            d_rgb.append(rgb_diff_map)

            if FLAGS.photo_loss_type == "mse":
                w_rgb_loss = (rgb_mask * (p_pred_color - img_w)**2).sum() / (rgb_mask.sum() + 1e-9)     
            
            elif FLAGS.photo_loss_type == "ssim":
                ssim_pred = mask_img[num][...,None] * pred_color
                ssim_warp = mask_img[num][...,None] * img_w
                w_rgb_loss = math.compute_ssim(ssim_pred, ssim_warp)
            
            fin_rgb_loss = w_rgb_loss
            if FLAGS.use_photo_loss:
                loss += w_rgb_param * fin_rgb_loss

        wa = jax.image.resize(unmasked_img[0], (224,224,3), 'bilinear')
        pr = jax.image.resize(pred_color, (224,224,3), 'bilinear')
        return loss, (wa, pr, d_rgb, fin_reg_loss, fin_rgb_loss, pred_depth, unmasked_img, decayed_reg_loss)

    (loss , outputs), grad = jax.value_and_grad(warp_loss_fn, has_aux = True)(jax.device_get(jax.tree_map(lambda x:x[0], state)).optimizer.target)
    # First value is loss

    def tree_norm(tree):
        return jnp.sqrt(
            jax.tree_util.tree_reduce(
                lambda x, y: x + jnp.sum(y**2), tree, initializer=0))
    return grad, outputs, rng


def prcp_difference(reg_model, params, image_q, image_k, mask, data_type="blender", layer_depth=3):
    ## gradient stop must be given before image input
    ## pred == query!!!!

    perceptual_loss = 0.0

    L1_dif_maps = []
    gt_key_maps = []
    gt_query_maps = []
   
    q_resized = jax.image.resize(image_q[None,...], (1, 224, 224, 3), 'bicubic')
    k_resized = jax.image.resize(image_k[None,...], (1, 224, 224, 3), 'bicubic')
            
    _, feat_q = reg_model.apply(params, q_resized, train = False)
    _, feat_k = reg_model.apply(params, k_resized, train = False)

    start=0
    end=2
    for feat_q, feat_k in zip(feat_q[start:end], feat_k[start:end]):
        reg_param = FLAGS.prcp_weight
        reg_param = 2.3 * reg_param
        reg_loss = L1Loss(feat_q, feat_k, mask)
        perceptual_loss += reg_loss

        # Calculate perceptual difference maps between pred and warped images
        l1_dif_map = l1_difference_map(feat_q, feat_k)
        L1_dif_maps.append(l1_dif_map)

    loss = reg_param * perceptual_loss

    return loss, L1_dif_maps


def L1Loss(feat_q,feat_k,occ_mask = None):
    
    batch_size, S, S, dim = feat_q.shape

    if FLAGS.no_occlusion:
        occ_mask = None

    if occ_mask != None:
        D, D = occ_mask.shape
        resized_mask = jnp.round(jax.image.resize(occ_mask, (S, S), 'nearest')).astype(int)
        num_live_patch = jnp.sum(resized_mask)
    
    else:
        resized_mask = jnp.ones((S,S))
        num_live_patch = S * S
    
    loss = jnp.sum(jnp.abs(feat_q - feat_k) * resized_mask[...,None]) / (num_live_patch * dim + 1e-9)

    return Loss


def l1_difference_map(map_a, map_b, occ_mask = None):
    """
    Dimensions: (H, W, C)
    Occlusion mask: (H, W)
    """
    if occ_mask != None:
        l1_maps = jnp.average(jnp.abs(map_a - map_b) * occ_mask[...,None], axis=-1)
    
    else:
        l1_maps = jnp.average(jnp.abs(map_a - map_b), axis=-1)
    
    total_max = jnp.max(l1_maps)
    # l1_dif_map = l1_maps / total_max
    l1_dif_map = l1_maps / 7.0

    return l1_dif_map


def init_reg(dtype: str, model_name: Optional[str], rng):
    """
        initialize jax-based model
        Args:
            dtype: string, data type 
            model_name: string, model's name
        Return
            model
    """
    if dtype == 'float16':
        dtype = jnp.float16
    elif dtype == 'float32':
        dtype = jnp.float32
    else:
        raise ValueError
    
    fake_img = jnp.ones((1,224,224,3))

    if model_name == "vgg16":
        model = reg_models.VGG16(output="logits", pretrained="imagenet")
        params = model.init(rng, fake_img)

    elif model_name == "vgg19":
        model = reg_models.VGG19(output="logits", pretrained="imagenet")
        params = model.init(rng, fake_img)
    
    elif model_name == "resnet101":
        model = reg_models.ResNet101(output="logits", pretrained="imagenet")
        params = model.init(rng, fake_img)
    
    elif model_name == "resnet18":
        model = reg_models.ResNet18(output="logits", pretrained="imagenet")
        params = model.init(rng, fake_img)

    elif model_name is None:
        model = reg_models.ResNet101(output="logits", pretrained="imagenet")
        params = model.init(rng, fake_img)
    
    return model, params
