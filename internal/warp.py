from calendar import c
from distutils.log import WARN
import enum
from hashlib import new
from optparse import Values
from pickle import BINSTRING
from re import S
from sre_parse import FLAGS
from ssl import cert_time_to_seconds

from absl import flags
from jax import lax
from jax import jit
from jax import random

import scipy
import jax.scipy as jsp
import jax.numpy as jnp

from internal import math
from internal import utils

FLAGS = flags.FLAGS

def inverse_warp(src_coord_raw, proj_depths, source_rgb, acc, lr_mask, length, data_type, lr_cons=False, eval_mode=False, camtopix=None):

    key = random.PRNGKey(100)

    if data_type == "blender":
        focal = 1111.1110311937682
        H = 800
        W = 800
        size = H * W
        ratio = 1    
    elif data_type == "llff":
        focal = 832.46746 / 2
        H = 378
        W = 504
        size = H * W
        ratio = 1    
    elif data_type == "dtu":
        H = 300
        W = 400
        size = H * W

     # May change, but not while using patchwise

    if not eval_mode:
        patch_H = length
        patch_W = length
        patch_size = length * length

    if data_type == "blender" or data_type == "llff":
        # Turn raw coord into pixel coord
        pixel_x_float = (focal * src_coord_raw[:,0] + W * 0.5 - 0.5) / ratio
        pixel_y_float = (- focal * src_coord_raw[:,1] + H * 0.5 - 0.5) / ratio


    x_mask_zero = pixel_x_float > 0
    x_mask_top = pixel_x_float < W - 1
    y_mask_zero = pixel_y_float > 0
    y_mask_top = pixel_y_float < H - 1
    
    clipped_mask = x_mask_zero * x_mask_top * y_mask_zero * y_mask_top * jnp.round(acc)
    final_mask = clipped_mask

    fin_x = pixel_x_float * final_mask
    fin_y = pixel_y_float * final_mask

    final_projection = jnp.stack((fin_y, fin_x), axis=-1)

    warped_image = bilinear_sampler(source_rgb, final_projection)

    if eval_mode:
        warped_image = warped_image.reshape(H,W,3)
        mask_image = final_mask.reshape(H,W)

        if lr_cons:
            lr_mask_image = lr_mask.reshape(H,W)

    else:
        warped_image = warped_image.reshape(patch_H,patch_W,3)
        mask_image = final_mask.reshape(patch_H,patch_W)

        if lr_cons:
            lr_mask_image = lr_mask.reshape(patch_H,patch_W)

    # middle_image = mask_image[...,None] * warped_image
    middle_image = warped_image

    if lr_cons:
        warped_image = lr_mask_image[...,None] * middle_image
        mask_image = lr_mask_image * mask_image

    return warped_image, middle_image, mask_image 


def bilinear_sampler(source_img, sampling_points):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    # """

    H = source_img.shape[0]
    W = source_img.shape[1]

    x = sampling_points[:,1]
    y = sampling_points[:,0]

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = jnp.floor(x).astype(int)
    x1 = x0 + 1
    y0 = jnp.floor(y).astype(int)
    y1 = y0 + 1

    # get pixel value at corner coords
    Ia = source_img[y0, x0]
    Ib = source_img[y1, x0]
    Ic = source_img[y0, x1]
    Id = source_img[y1, x1]

    # recast as float for delta calculation
    x0 = x0.astype(float)
    x1 = x1.astype(float)
    y0 = y0.astype(float)   
    y1 = y1.astype(float)    

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # # add dimension for addition
    wa = wa[...,None]
    wb = wb[...,None]
    wc = wc[...,None]
    wd = wd[...,None]

    # compute output
    stacked = jnp.stack((wa*Ia, wb*Ib, wc*Ic, wd*Id), axis=0)
    out = jnp.sum(stacked, axis=0)

    return out
