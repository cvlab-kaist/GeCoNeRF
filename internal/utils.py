# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Utility functions."""
import collections
import os
from os import path
from re import S
from absl import flags
import dataclasses
import flax             
import gin              
import jax
import jax.numpy as jnp
import numpy as np
import random as rand

from PIL import Image
from tqdm import tqdm
import requests
import os
import tempfile

from internal import math

gin.add_config_file_search_path('../')


gin.config.external_configurable(flax.linen.relu, module='flax.linen')
gin.config.external_configurable(flax.linen.sigmoid, module='flax.linen')
gin.config.external_configurable(flax.linen.softplus, module='flax.linen')


@flax.struct.dataclass
class TrainState:
  optimizer: flax.optim.Optimizer


@flax.struct.dataclass
class Stats:
  loss: float
  losses: float
  m_loss: float
  weight_l2: float
  psnr: float
  psnrs: float
  grad_norm: float
  grad_abs_max: float
  grad_norm_clipped: float


Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'pose', 'radii', 'lossmult', 'near', 'far'))


# TODO(barron): Do a default.gin thing
@gin.configurable()
@dataclasses.dataclass
class Config:
  """Configuration flags for everything."""
  dataset_loader: str = 'llff'  # The type of dataset loader to use.
  batching: str = 'single_image'  # Batch composition, [single_image, all_images].
  batch_size: int = 1024  # The number of rays/pixels in each batch.
  factor: int = 0  # The downsample factor of images, 0 for no downsampling.
  spherify: bool = False  # Set to True for spherical 360 scenes.
  render_path: bool = False  # If True, render a path. Used only by LLFF.
  llffhold: int = 8  # Use every Nth image for the test set. Used only by LLFF.
  lr_init: float = 5e-4  # The initial learning rate.
  lr_final: float = 5e-6  # The final learning rate.
  lr_delay_steps: int = 2500  # The number of "warmup" learning steps.
  lr_delay_mult: float = 0.01  # How much sever the "warmup" should be.
  grad_max_norm: float = 0.  # Gradient clipping magnitude, disabled if == 0.
  grad_max_val: float = 0.  # Gradient clipping value, disabled if == 0.
  max_steps: int = 85000  # The number of optimization steps.
  save_every: int = 5001  # The number of steps to save a checkpoint.
  print_every: int = 200  # The number of steps between reports to tensorboard.
  gc_every: int = 4000  # The number of steps between garbage collections.
  test_render_interval: int = 1  # The interval between images saved to disk.
  disable_multiscale_loss: bool = False  # If True, disable multiscale loss.
  randomized: bool = True  # Use randomized stratified sampling.
  near: float = 2.  # Near plane distance.
  far: float = 6.  # Far plane distance.
  coarse_loss_mult: float = 0.1  # How much to downweight the coarse loss(es).
  weight_decay_mult: float = 0.  # The multiplier on weight decay.
  white_bkgd: bool = True  # If True, use white as the background (black o.w.).
  patch_size: int = 120
  clip_output_dtype: str = "float16"
  reg_output_dtype: str = "float16"
  use_clip_loss: bool = True
  dtu_max_images: int = 49  # Whether to restrict the max number of images.
  dtu_light_cond: int = 3  # Light condition. Used only by DTU.
  anneal_nearfar: bool = True  # Whether to anneal near/far planes.
  anneal_nearfar_steps: int = 256  # Steps for near/far annealing.
  anneal_nearfar_perc: float = 0.5  # Percentage for near/far annealing.
  anneal_mid_perc: float = 0.5  # Perc for near/far mid point.
  ####################### new configs


def define_common_flags():
  # Define the flags used by both train.py and eval.py
  flags.DEFINE_multi_string('gin_file', None,
                            'List of paths to the config files.')
  flags.DEFINE_multi_string(
      'gin_param', None, 'Newline separated list of Gin parameter bindings.')
  flags.DEFINE_string('train_dir', None, 'where to store ckpts and logs')
  flags.DEFINE_string('data_dir', None, 'input data directory.')
  flags.DEFINE_integer(
      'chunk', 4000,
      'the size of chunks for evaluation inferences, set to the value that'
      'fits your GPU/TPU memory.')
  flags.DEFINE_string("input_dataset", "lego", "input dataset")

  # Image saving...!!
  flags.DEFINE_integer("img_save_every", 100,
                       "training image saving interval (for tensorboard size...)")

  # Sparse Training Flags
  flags.DEFINE_bool("sparse_viewpoint", True,
                    "Whether to use sparse viewpoint or not")
  flags.DEFINE_bool("single_image_gt", True,
                    "Only one GT image or all (N)?")    
  flags.DEFINE_bool("random_pose", False,
                    "Completely random pose?")
  flags.DEFINE_string("pose_random_type", "random_divergence",
                    "random_[pose / divergence], set_divergence")
  ######## if gt_unseen is used ###########
  flags.DEFINE_integer("num_nearby_views", 5,
                      "number of closest unseen views extracted from GT")
  #########################################
  flags.DEFINE_bool("lr_consistency", True,
                    "Use LR consistency for occlusion?")
  flags.DEFINE_float("lr_threshold", 0.01,
                    "Use LR consistency for occlusion?")
  flags.DEFINE_integer("alternating_num", 5,
                    "alternating number between ob and kview")
  flags.DEFINE_integer("decay_param", 20000,
                    "parameter for loss decay")
  flags.DEFINE_integer("occ_decay_param", 25000,
                    "parameter for lr consistency parameter decay")

  flags.DEFINE_bool("window_annealing", True,
                    "NerFies window annealing")
  flags.DEFINE_integer("window_parameter", 20000,
                    "NerFies window annealing parameter N")
  flags.DEFINE_bool("no_occlusion", False,
                    "NerFies window annealing parameter N")

  flags.DEFINE_integer("num_input_viewpoint", 3,
                       "Number of input viewpoints")
  flags.DEFINE_integer("early_stop", 150000,
                       "Early stop parameter")
                       

  # Input Images Selection
  flags.DEFINE_list("input_images", [4, 16, 99],
                    "The list of input image given (0~100)")
  # flags.DEFINE_list("input_images", [60, 63, 51],
  #                   "The list of input image given (0~100)")
  flags.DEFINE_list("input_images_test", [7, 128, 66],
                    "The list of test input images given (0~???)")
  flags.DEFINE_list("input_images_llff", [0, 17, -1],
                    "The list of input image given (0~100)")
  flags.DEFINE_list("input_images_dtu", [0, 1, 2],
                    "The list of input image given (0~100)")
  flags.DEFINE_list("input_images_test_llff", [0,1,3],
                    "The list of test input images given (0~???)")
  flags.DEFINE_list("input_images_test_dtu", [0, 1, 2],
                    "The list of input image given (0~100)")

  # General regularization loss rules 
  flags.DEFINE_bool("regularization_on", True,
                    "Activate regularization")
  flags.DEFINE_bool("use_reg_models", True,
                    "whether use regularization models or not")
  flags.DEFINE_string("reg_model", "vgg19",
                      "vgg16 / vgg19 / resnet18 / resnet101 (which model to use?)")
  flags.DEFINE_string("reg_output_dtype", "float16",
                      "float32/ float16 (float16 for memory saving)")
  flags.DEFINE_integer("reg_loss_every", 2,
                        "no. of steps to take before performing semantic loss evaluation")
  flags.DEFINE_integer("stop_reg_step", 10000000,
                        "When to stop the regularization loss")
  flags.DEFINE_integer("start_reg_step", 500,
                        "When to start the regularization loss")
                        
  # Semantic regularization loss configs
  flags.DEFINE_bool("use_sem_loss", True,
                    "whether use regularization loss or not")
  flags.DEFINE_integer("layer_depth", 3,
                       "Depth of layer used")  
  flags.DEFINE_bool("model_input_unmasked", True,
                      "Is the input to the networks unmasked?")
  flags.DEFINE_float("nce_weight", 0.005,
                      "Weight hyperparameter for PatchNCE loss")       
  flags.DEFINE_float("prcp_weight", 0.005,
                      "Weight hyperparameter for perceptual loss") 

  # Photometric regularization loss configs
  flags.DEFINE_bool("use_photo_calculation", False,
                    "whether use photometric loss or not")
  flags.DEFINE_bool("use_photo_loss", False,
                    "whether use photometric loss or not")
  flags.DEFINE_string("photo_loss_type", "mse",
                    " mse / ssim  ((Which type of photometric loss to use")
  flags.DEFINE_float("photo_weight", 0.15,
                      "Weight hyperparameter for PatchNCE loss")

  # Patch Configurations
  flags.DEFINE_integer("downsample", 2,
                        "Patch downsampling ratio for regularization")
  flags.DEFINE_integer("patch_size", 120,
                    "Side size of patch")    

  # Small Image Configurations
  flags.DEFINE_integer("img_downsample", 2,
                        "Downsampling ratio for regularization")
  flags.DEFINE_integer("small_img_side", 160,
                    "Side size of the whole image")       

  # CLIP part Flags
  flags.DEFINE_bool("use_clip_loss", False,
                    "whether use semantic loss or not")
  flags.DEFINE_string("clip_model_name", "openai/clip-vit-base-patch32", 
                      "model type for CLIP")
  flags.DEFINE_string("clip_output_dtype", "float16",
                      "float32/ float16 (float16 for memory saving)")
  flags.DEFINE_integer("sc_loss_every", 30,
                       "no. of steps to take before performing semantic loss evaluation")
  flags.DEFINE_float("sc_loss_mult", 1e-2,
                     "weighting for semantic loss from CLIP")
  flags.DEFINE_integer("random_ray_size", 200,
                       "H and W of random rays size")
  flags.DEFINE_integer("random_ray_downsample", 4,
                       "the downsample factor of random rays, the random rays shape will be random_ray_size//random_ray_downsample")
  flags.DEFINE_integer("stop_sc_loss", 10000000,
                       "When to stop semantic loss")    



def load_config():
  gin.parse_config_files_and_bindings(flags.FLAGS.gin_file,
                                      flags.FLAGS.gin_param)
  return Config()


def open_file(pth, mode='r'):
  return open(pth, mode=mode)


def file_exists(pth):
  return path.exists(pth)


def listdir(pth):
  return os.listdir(pth)


def isdir(pth):
  return path.isdir(pth)


def makedirs(pth):
  os.makedirs(pth)


def namedtuple_map(fn, tup):
  """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
  return type(tup)(*map(fn, tup))


def shard(xs):
  """Split data into shards for multiple devices along the first dimension."""
  return jax.tree_map(
      lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), xs)


def to_device(xs):
  """Transfer data to devices (GPU/TPU)."""
  return jax.tree_map(jnp.array, xs)


def unshard(x, padding=0):
  """Collect the sharded tensor to the shape before sharding."""

  y = x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))
  
  if padding > 0:
    y = y[:-padding]
  return y


def save_img_uint8(img, pth):
  """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
  with open_file(pth, 'wb') as f:
    Image.fromarray(
        (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(jnp.uint8)).save(
            f, 'PNG')


def save_img_float32(depthmap, pth):
  """Save an image (probably a depthmap) to disk as a float32 TIFF."""
  with open_file(pth, 'wb') as f:
    Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(f, 'TIFF')


def camtoworld_matrix_to_rays(camtoworld, downsample_by, dataset = "blender"):
    """ render one instance of rays given a camera to world matrix (4, 4) """
    pixel_center = 0.

    if dataset == "blender":
      w, h = 800, 800
      focal, downsample = 1111.1110311937682, downsample_by # check!!!!!!!
    
    elif dataset == 'llff':
      w, h = 504, 378
      focal, downsample = 832.46746/2 , downsample_by

    elif dataset == "dtu":
      w, h = 400, 300
      focal, downsample = 723.08258, downsample_by

    x, y = jnp.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        jnp.arange(0, w, downsample, dtype=jnp.float32) + pixel_center,  # X-Axis (columns)
        jnp.arange(0, h, downsample, dtype=jnp.float32) + pixel_center,  # Y-Axis (rows)
        indexing="xy")

    camera_dirs = jnp.stack([(x - w * 0.5) / focal,
                            -(y - h * 0.5) / focal,
                            -jnp.ones_like(x)],
                           axis=-1)   
    directions = (camera_dirs[..., None, :] * camtoworld[None, None, :3, :3]).sum(axis=-1)
    origins = jnp.broadcast_to(camtoworld[None, None, :3, -1], directions.shape)
    viewdirs = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
    
    camtoworld_flat = camtoworld.reshape(-1,16)[0]

    pose = jnp.broadcast_to(camtoworld_flat[None, None, :],
                          (directions.shape[0], directions.shape[1], 16))
     
    # Distance from each unit-norm direction vector to its x-axis neighbor.

    if dataset == "blender" or dataset == "dtu":
      dx = jnp.sqrt(
          jnp.sum((directions[:-1, :, :] - directions[1:, :, :])**2, -1))
      dx = jnp.concatenate([dx, dx[:, -2:-1, :]], 0)

      radii = dx[..., None] * 2 / jnp.sqrt(12)

    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.

    if dataset == "llff":
      origins, directions = math.convert_to_ndc(origins, directions, focal, w, h)

      mat = origins

      # Distance from each unit-norm direction vector to its x-axis neighbor.
      dx = np.sqrt(np.sum((mat[:-1, :, :] - mat[1:, :, :])**2, -1))
      dx = np.concatenate([dx, dx[-2:-1, :]], 0)

      dy = np.sqrt(np.sum((mat[:, :-1, :] - mat[:, 1:, :])**2, -1))
      dy = np.concatenate([dy, dy[:, -2:-1]], 1)
      # Cut the distance in half, and then round it out so that it's
      # halfway between inscribed by / circumscribed about the pixel.
      radii = (0.5 * (dx + dy))[..., None] * 2 / np.sqrt(12)

    # import pdb; pdb.set_trace()

    ones = jnp.ones_like(origins[..., :1])

    return Rays(origins=origins,
                directions=directions,
                viewdirs=viewdirs,
                pose=pose,
                radii=radii,
                lossmult=ones,
                near=ones * 2.,
                far=ones * 6.)


def preprocess_for_reg(image, new_h, new_w):
    """
        jax-based preprocessing for reg
        Args:
            image [B, 3, H, W]: batch image
        Return
            image [B, 3, 224, 224]: pre-processed image for CLIP
    """

    B, H, W, D = image.shape

    # mean = jnp.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 1, 1, 3)
    # std = jnp.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 1, 1, 3)
    image = jax.image.resize(image, (B, new_h, new_w, D), 'bicubic')  # assume that images have rectangle shape.
    # image = (image - mean.astype(image.dtype)) / std.astype(image.dtype)

    return image



def download(ckpt_dir, url):
    name = url[url.rfind('/') + 1 : url.rfind('?')]
    if ckpt_dir is None:
        ckpt_dir = tempfile.gettempdir()
    ckpt_dir = os.path.join(ckpt_dir, 'flaxmodels')
    ckpt_file = os.path.join(ckpt_dir, name)
    if not os.path.exists(ckpt_file):
        print(f'Downloading: \"{url[:url.rfind("?")]}\" to {ckpt_file}')
        if not os.path.exists(ckpt_dir): 
            os.makedirs(ckpt_dir)

        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        
        # first create temp file, in case the download fails
        ckpt_file_temp = os.path.join(ckpt_dir, name + '.temp')
        with open(ckpt_file_temp, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print('An error occured while downloading, please try again.')
            if os.path.exists(ckpt_file_temp):
                os.remove(ckpt_file_temp)
        else:
            # if download was successful, rename the temp file
            os.rename(ckpt_file_temp, ckpt_file)
    return ckpt_file