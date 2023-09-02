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
"""Different datasets implementation plus a general port for all the datasets."""
from json.encoder import py_encode_basestring
from ast import Num
import json
import os
from os import path
import queue
from re import S
from this import d
import threading
import cv2
import jax
import numpy as np
import os, time
import random as rand

from absl import flags

from jax import random
from PIL import Image
from internal import utils
from internal import pose

FLAGS = flags.FLAGS

def get_dataset(split, train_dir, config):
  return dataset_dict[config.dataset_loader](split, train_dir, config)


def downsample(img, factor, patch_size=-1, mode=cv2.INTER_AREA):
  """Area downsample img (factor must evenly divide img height and width)."""
  sh = img.shape
  max_fn = lambda x: max(x, patch_size)
  out_shape = (max_fn(sh[1] // factor), max_fn(sh[0] // factor))
  img = cv2.resize(img, out_shape, mode)
  return img


def recenter_poses(poses):
  """Recenter poses around the origin."""
  cam2world = poses_avg(poses)
  poses = np.linalg.inv(pad_poses(cam2world)) @ pad_poses(poses)
  return unpad_poses(poses)


def convert_to_ndc(origins, directions, focal, w, h, near=1.):
  """Convert a set of rays to NDC coordinates."""
  # Shift ray origins to near plane
  t = -(near + origins[..., 2]) / directions[..., 2]

  origins = origins + t[..., None] * directions

  dx, dy, dz = tuple(np.moveaxis(directions, -1, 0))
  ox, oy, oz = tuple(np.moveaxis(origins, -1, 0))

  # Projection
  o0 = -((2 * focal) / w) * (ox / oz)
  o1 = -((2 * focal) / h) * (oy / oz)
  o2 = 1 + 2 * near / oz

  d0 = -((2 * focal) / w) * (dx / dz - ox / oz)
  d1 = -((2 * focal) / h) * (dy / dz - oy / oz)
  d2 = -2 * near / oz

  origins = np.stack([o0, o1, o2], -1)
  directions = np.stack([d0, d1, d2], -1)

  ############### maths로 옮기기

  return origins, directions


def generate_spiral_path(poses, bounds, n_frames=120, n_rots=2, zrate=.5):
  """Calculates a forward facing spiral path for rendering."""
  # Find a reasonable 'focus depth' for this dataset as a weighted average
  # of near and far bounds in disparity space.
  close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
  dt = .75
  focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

  # Get radii for spiral path using 90th percentile of camera positions.
  positions = poses[:, :3, 3]
  radii = np.percentile(np.abs(positions), 90, 0)
  radii = np.concatenate([radii, [1.]])


class Dataset(threading.Thread):
  """Dataset Base Class."""

  def __init__(self, split, data_dir, config):
    super(Dataset, self).__init__()
    self.queue = queue.Queue(3)  # Set prefetch buffer to 3 batches.
    self.daemon = True
    self.split = split
    self.data_dir = data_dir
    self.near = config.near
    self.far = config.far
    if split == 'train':
      self._train_init(config)
    elif split == 'test':
      self._test_init(config)
    else:
      raise ValueError(
          'the split argument should be either \'train\' or \'test\', set'
          'to {} here.'.format(split))
    self.batch_size = config.batch_size // jax.process_count()
    self.batching = config.batching
    self.render_path = config.render_path
    self.data_type = config.dataset_loader
    self.start()

  def __iter__(self):
    return self

  def __next__(self):
    """Get the next training batch or test example.

    Returns:
      batch: dict, has 'pixels' and 'rays'.
    """
    x = self.queue.get()
    if self.split == 'train':
      return utils.shard(x)
    else:
      return utils.to_device(x)

  def peek(self):
    """Peek at the next training batch or test example without dequeuing it.

    Returns:
      batch: dict, has 'pixels' and 'rays'.
    """
    x = self.queue.queue[0].copy()  # Make a copy of the front of the queue.
    if self.split == 'train':
      return utils.shard(x)
    else:
      return utils.to_device(x)

  def run(self):
    if self.split == 'train':
      next_func = self._next_train
    else:
      next_func = self._next_test
    while True:
      self.queue.put(next_func())

  @property
  def size(self):
    return self.n_examples

  def _train_init(self, config):
    """Initialize training."""
    self._load_renderings(config)
    self._generate_rays()

    self.shaped_images = self.images
    self.shaped_rays = self.rays

    if config.batching == 'all_images':
      # flatten the ray and image dimension together.
      self.images = self.images.reshape([-1, 3])
      self.rays = utils.namedtuple_map(lambda r: r.reshape([-1, r.shape[-1]]),
                                       self.rays)

    elif config.batching == 'single_image':
      self.images = self.images.reshape([-1, self.resolution, 3])
      self.rays = utils.namedtuple_map(
          lambda r: r.reshape([-1, self.resolution, r.shape[-1]]), self.rays)

    else:
      raise NotImplementedError(
          f'{config.batching} batching strategy is not implemented.')
    
    self.origins_list = self.rays.origins[:,0]

    self.num_input_view = FLAGS.num_input_viewpoint

    self.image_dict = {
      "chair": [66, 73, 59],
      "drums": [73, 14, 91],
      "ficus": [65, 73, 99],
      "hotdog": [3, 75, 76],
      "lego": [4, 16, 99],
      "materials": [2, 34, 47],
      "mic": [58, 99, 3],
      "ship": [2, 67, 89]
    }

  def _test_init(self, config):
    self._load_renderings(config)
    self._generate_rays()
    self.it = 0

  def _next_train(self):
    """Sample next training batch."""
    
    if self.data_type == "blender":
      image_list = FLAGS.input_images
    elif self.data_type == 'llff':
      image_list = FLAGS.input_images_llff

    if self.batching == 'all_images':
      ray_indices = np.random.randint(0, self.rays[0].shape[0],
                                      (self.batch_size,))
      batch_pixels = self.images[ray_indices]
      batch_rays = utils.namedtuple_map(lambda r: r[ray_indices], self.rays)
    elif self.batching == 'single_image':
      if FLAGS.sparse_viewpoint:
        ind = np.random.randint(0, self.num_input_view, ())
        # image_list = self.image_dict[FLAGS.input_dataset][:self.num_input_view]
        image_index = image_list[ind]

      else:
        image_index = np.random.randint(0, self.n_examples, ()) # Random Images

      ray_indices = np.random.randint(0, self.rays[0][0].shape[0], # Rays origins
                                      (self.batch_size,))
      batch_pixels = self.images[image_index][ray_indices]
      batch_rays = utils.namedtuple_map(lambda r: r[image_index][ray_indices],
                                        self.rays)
    else:
      raise NotImplementedError(
          f'{self.batching} batching strategy is not implemented.')
      
    return {'pixels': batch_pixels, 'rays': batch_rays}
  
  def _next_test(self):
    """Sample next test example."""
    idx = self.it
    self.it = (self.it + 1) % self.n_examples

    if self.render_path:
      return {'rays': utils.namedtuple_map(lambda r: r[idx], self.render_rays)}
    else:
      if self.data_type == "blender":
        image_list = FLAGS.input_images_test
      elif self.data_type == 'llff':
        image_list = FLAGS.input_images_test_llff
      test_idx = np.random.randint(0, self.images.shape[0])
      
      return {
          'src_pixels': [self.images[idx] for idx in image_list],
          'src_rays': [utils.namedtuple_map(lambda r: r[idx], self.rays) for idx in image_list],
          'tgt_pixels': self.images[test_idx],
          'tgt_rays': utils.namedtuple_map(lambda r: r[test_idx], self.rays),
      }

  def next_reg(self, rng, step, reg_type):
    """Sample next training batch."""
    data_type = self.data_type 
    # rng, key = random.split(rng)
    
    ## Training gt image selection
    if data_type == "blender":
      image_list = self.image_dict[FLAGS.input_dataset][:self.num_input_view]
    elif data_type == 'llff':
      image_list = FLAGS.input_images_llff

    # Applied for single image only
    n = len(image_list)
    gt_num = n
    ind = np.random.randint(0, n)
    image_index = image_list[ind]

    other_gt_inds = [image_list[ind - i] for i in range(1,n)]
    other_gt_origin = [self.rays.origins[gt,0] for gt in other_gt_inds]
    other_gt_origin = np.stack(other_gt_origin, axis=0)

    gt_image = self.shaped_images[image_index][None,...]
    x_size = gt_image.shape[2]
    y_size = gt_image.shape[1]

    # GT pose generator
    if FLAGS.single_image_gt:
      batch_pixels = gt_image.reshape(1,-1,3)
      seen_pose = self.rays.pose[image_index,0][None,...]
      gt_pose = seen_pose[0].reshape(4,4)

      closest_origin = self.rays.origins[image_index, 0]

      # Inverse GT pose generator
      gt_pose_inv = np.linalg.inv(gt_pose)
      seen_pose_inv = gt_pose_inv.reshape(1,16)

    else:
      batch_pixels_list, seen_pose_list, seen_pose_inv_list, unseen_gt_list = [], [], [], []

      rand.shuffle(image_list)
      all_gt_images = [self.shaped_images[k][None,...] for k in image_list]

      for num, (gt_image, image_index) in enumerate(zip(all_gt_images, image_list)):
        batch_pixels = gt_image.reshape(1,-1,3)

        seen_pose = self.rays.pose[image_index,0][None,...]
        gt_pose = seen_pose[0].reshape(4,4)

        seen_origin = self.rays.origins[image_index, 0]

        if num == 0:
          closest_origin = seen_origin

        # Inverse GT pose generator
        gt_pose_inv = np.linalg.inv(gt_pose)
        seen_pose_inv = gt_pose_inv.reshape(1,16)

        batch_pixels_list.append(batch_pixels)
        seen_pose_list.append(seen_pose)
        seen_pose_inv_list.append(seen_pose_inv)
      
      batch_pixels = np.concatenate(batch_pixels_list, axis=0)
      seen_pose = np.concatenate(seen_pose_list, axis=0)
      seen_pose_inv = np.concatenate(seen_pose_inv_list, axis=0)

    ############################
    if data_type == "blender":
      # div_angle = np.log(step/100 + 1) / np.log(3)
      div_angle = 3 + (step/120000 * 6)
      # div_angle = (step/120000) * 8

      new_pose = pose.pose_generator(FLAGS.pose_random_type, gt_pose, data_type, divergence_angle = div_angle)
      raw_rays = utils.camtoworld_matrix_to_rays(new_pose, downsample_by = 1, dataset = data_type)
    else:
      num = FLAGS.num_nearby_views
      k = np.random.randint(0,num,())

      # The closer, the larger!
      pose_similarity = np.sum(closest_origin[None,...] * self.origins_list, axis=-1) / (np.linalg.norm(closest_origin, axis=-1)[None,...] * np.linalg.norm(self.origins_list, axis=-1))
      unseen_idx_list = np.argsort(-pose_similarity)[1: num + 1]
      eval_index = unseen_idx_list[k]
      ev_gt_image = self.shaped_images[eval_index][None,...]

      # Random pose extractor (from GT poses)
      new_pose = self.rays.pose[eval_index,0][None, ...]
      new_pose = new_pose[0].reshape(4,4)
      ################################### 

      raw_rays = utils.namedtuple_map(lambda x: x[eval_index], self.shaped_rays)

    patch_size = FLAGS.patch_size
    d = FLAGS.patch_size//2
    downsample = FLAGS.downsample

    # randomly crop the rays and downsample it.
    if data_type == "blender":
      cx = np.random.randint(200, 600)
      cy = np.random.randint(200, 600)

    elif data_type == "llff":
      cx = np.random.randint(120, 384)
      cy = np.random.randint(120, 268)

    size = ( patch_size // downsample) ** 2

    orig_rays = utils.namedtuple_map(lambda x: x[cy-d:cy+d,cx-d:cx+d], raw_rays)
    orig_rays = utils.namedtuple_map(lambda r: r.reshape(patch_size**2,-1)[None,...], orig_rays)
    
    reduced_rays = utils.namedtuple_map(lambda x: x[cy-d:cy+d:downsample,cx-d:cx+d:downsample], raw_rays)
    reduced_rays = utils.namedtuple_map(lambda r: r.reshape(size,-1)[None,...], reduced_rays)


    if data_type == "blender":
      batch_dict = {'orig_rays': orig_rays ,'reduced_rays': reduced_rays, 'pixels': batch_pixels, 
                    'seen_pose': seen_pose, 'seen_pose_inv': seen_pose_inv}
    else:
      real_unseen_gt = ev_gt_image[:,cy-d:cy+d,cx-d:cx+d]
      real_unseen_gt = real_unseen_gt.reshape(1,-1,3)

      batch_dict = {'orig_rays': orig_rays ,'reduced_rays': reduced_rays, 'pixels': batch_pixels, 
                    'seen_pose': seen_pose, 'seen_pose_inv': seen_pose_inv, 'unseen_gt': real_unseen_gt}

    return batch_dict

  # TODO(bydeng): Swap this function with a more flexible camera model.
  def _generate_rays(self):
    """Generating rays for all images."""
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(self.w, dtype=np.float32),  # X-Axis (columns)
        np.arange(self.h, dtype=np.float32),  # Y-Axis (rows)
        indexing='xy')
    camera_dirs = np.stack(
        [(x - self.w * 0.5 + 0.5) / self.focal,
         -(y - self.h * 0.5 + 0.5) / self.focal, -np.ones_like(x)],
        axis=-1)

    """
    Camera coordinate that goes like this,
    with camera origin at (0,0,0)

    (-400, 400) ------------------   (400,400)
          :                               :
          :      plane at z = -1          :
          :                               :
    (-400,-400) ------------------   (400,-400)

    (48, 756, 1008, 3)
    (48, 378, 504, 3)

    """

    # import pdb; pdb.set_trace()

    directions = ((camera_dirs[None, ..., None, :] *
                   self.camtoworlds[:, None, None, :3, :3]).sum(axis=-1))
        # Apply rotation matrix to each pixelwise directions
        # Because of sum(axis=-1), this is effectively rotmat calculation
        # self.camtoworlds[:, None, None, :3, :3]) ==> rotmat
    
    origins = np.broadcast_to(self.camtoworlds[:, None, None, :3, -1],
                              directions.shape)
        # Origin movement to a point on sphere with
        # Blender dataset) Radius 4.031128 (sqrt of 16.25)
        # LLFF dataset) Not a sphere!! Let's see some other code....

    if self.camtoworlds.shape[1] == 3:
      poses_raw = np.concatenate((self.camtoworlds,np.broadcast_to([0.,0.,0.,1.],(self.camtoworlds.shape[0],1,self.camtoworlds.shape[2]))), axis=-2)
      camtoworlds = poses_raw.reshape(-1,16)
      
    else:
      camtoworlds = self.camtoworlds.reshape(-1, 16)

    pose = np.broadcast_to(camtoworlds[:, None, None, :],
                          (directions.shape[0], directions.shape[1], directions.shape[2], 16))
                    
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    
    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = np.sqrt(
        np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :])**2, -1))
    dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.

    radii = dx[..., None] * 2 / np.sqrt(12)

    # import pdb; pdb.set_trace()

    ones = np.ones_like(origins[..., :1])

    self.rays = utils.Rays(
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        pose=pose,
        radii=radii,
        lossmult=ones,
        near=ones * self.near,
        far=ones * self.far)


class Blender(Dataset):
  """Blender Dataset."""

  def _load_renderings(self, config):
    """Load images from disk."""
    if config.render_path:
      raise ValueError('render_path cannot be used for the blender dataset.')
    with utils.open_file(
        path.join(self.data_dir, 'transforms_{}.json'.format(self.split)),
        'r') as fp:
      meta = json.load(fp)

    images = []
    cams = []

    for i in range(len(meta['frames'])):
      frame = meta['frames'][i]
      fname = os.path.join(self.data_dir, frame['file_path'] + '.png')
      with utils.open_file(fname, 'rb') as imgin:
        image = np.array(Image.open(imgin), dtype=np.float32) / 255.
        if config.factor == 2:
          [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
          image = cv2.resize(
              image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
        elif config.factor > 0:
          raise ValueError('Blender dataset only supports factor=0 or 2, {} '
                           'set.'.format(config.factor))
                           
      cams.append(np.array(frame['transform_matrix'], dtype=np.float32))
      images.append(image)

    self.images = np.stack(images, axis=0)
    if config.white_bkgd:
      self.images = (
          self.images[..., :3] * self.images[..., -1:] +
          (1. - self.images[..., -1:]))
    else:
      self.images = self.images[..., :3]
    self.h, self.w = self.images.shape[1:3]
    self.resolution = self.h * self.w
    self.camtoworlds = np.stack(cams, axis=0)
    camera_angle_x = float(meta['camera_angle_x'])
    self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
    self.n_examples = self.images.shape[0]
    self.dtype = config.clip_output_dtype
    

class LLFF(Dataset):
  """LLFF Dataset."""

  def _load_renderings(self, config):
    """Load images from disk."""
    # Load images.
    imgdir_suffix = ''
    if config.factor > 0:
      imgdir_suffix = '_{}'.format(config.factor)
      factor = config.factor
    else:
      factor = 1
    imgdir = path.join(self.data_dir, 'images' + imgdir_suffix)
    if not utils.file_exists(imgdir):
      raise ValueError('Image folder {} does not exist.'.format(imgdir))
    imgfiles = [
        path.join(imgdir, f)
        for f in sorted(utils.listdir(imgdir))
        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
    ]
    images = []
    for imgfile in imgfiles:
      with utils.open_file(imgfile, 'rb') as imgin:
        image = np.array(Image.open(imgin), dtype=np.float32) / 255.
        images.append(image)
    images = np.stack(images, axis=-1)

    # Load poses and bds.
    with utils.open_file(path.join(self.data_dir, 'poses_bounds.npy'),
                         'rb') as fp:
      poses_arr = np.load(fp)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])
    if poses.shape[-1] != images.shape[-1]:
      raise RuntimeError('Mismatch between imgs {} and poses {}'.format(
          images.shape[-1], poses.shape[-1]))

    # Update poses according to downsampling.
    poses[:2, 4, :] = np.array(images.shape[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    # Correct rotation matrix ordering and move variable dim to axis 0.
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(images, -1, 0)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale according to a default bd factor.
    scale = 1. / (bds.min() * .75)
    poses[:, :3, 3] *= scale
    bds *= scale

    # Recenter poses.
    poses = self._recenter_poses(poses)

    # Generate a spiral/spherical ray path for rendering videos.
    if config.spherify:
      poses = self._generate_spherical_poses(poses, bds)
      self.spherify = True
    else:
      self.spherify = False
    if not config.spherify and self.split == 'test':
      self._generate_spiral_poses(poses, bds)

    # Select the split.
    i_test = np.arange(images.shape[0])[::config.llffhold]
    i_train = np.array(
        [i for i in np.arange(int(images.shape[0])) if i not in i_test])
    if self.split == 'train':
      indices = i_train
    else:
      indices = i_test
    images = images[indices]
    poses = poses[indices]

    self.images = images
    self.camtoworlds = poses[:, :3, :4]
    
    self.focal = poses[0, -1, -1]
    self.h, self.w = images.shape[1:3]
    self.resolution = self.h * self.w
    if config.render_path:
      self.n_examples = self.render_poses.shape[0]
    else:
      self.n_examples = images.shape[0]

  def _generate_rays(self):
    """Generate normalized device coordinate rays for llff."""
    if self.split == 'test':
      n_render_poses = self.render_poses.shape[0]
      self.camtoworlds = np.concatenate([self.render_poses, self.camtoworlds],
                                        axis=0)

    super()._generate_rays()

    if not self.spherify:
      
      ndc_origins, ndc_directions = convert_to_ndc(self.rays.origins,
                                                   self.rays.directions,
                                                   self.focal, self.w, self.h)

      mat = ndc_origins
      # Distance from each unit-norm direction vector to its x-axis neighbor.
      dx = np.sqrt(np.sum((mat[:, :-1, :, :] - mat[:, 1:, :, :])**2, -1))
      dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)

      dy = np.sqrt(np.sum((mat[:, :, :-1, :] - mat[:, :, 1:, :])**2, -1))
      dy = np.concatenate([dy, dy[:, :, -2:-1]], 2)
      # Cut the distance in half, and then round it out so that it's
      # halfway between inscribed by / circumscribed about the pixel.
      radii = (0.5 * (dx + dy))[..., None] * 2 / np.sqrt(12)

      ones = np.ones_like(ndc_origins[..., :1])
      self.rays = utils.Rays(
          origins=ndc_origins,
          directions=ndc_directions,
          viewdirs=self.rays.directions,
          pose=self.rays.pose,
          radii=radii,
          lossmult=ones,
          near=ones * self.near,
          far=ones * self.far)
        
      self.shaped_rays = self.rays
      
    # Split poses from the dataset and generated poses
    if self.split == 'test':
      self.camtoworlds = self.camtoworlds[n_render_poses:]
      split = [np.split(r, [n_render_poses], 0) for r in self.rays]
      split0, split1 = zip(*split)
      self.render_rays = utils.Rays(*split0)
      self.rays = utils.Rays(*split1)

  def _recenter_poses(self, poses):
    """Recenter poses according to the original NeRF code."""
    poses_ = poses.copy()
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    
    c2w = self._poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses

  def _poses_avg(self, poses):
    """Average poses according to the original NeRF code."""
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = self._normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([self._viewmatrix(vec2, up, center), hwf], 1)
    return c2w

  def _viewmatrix(self, z, up, pos):
    """Construct lookat view matrix."""
    vec2 = self._normalize(z)
    vec1_avg = up
    vec0 = self._normalize(np.cross(vec1_avg, vec2))
    vec1 = self._normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

  def _normalize(self, x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)

  def _generate_spiral_poses(self, poses, bds):
    """Generate a spiral path for rendering."""
    c2w = self._poses_avg(poses)
    # Get average pose.
    up = self._normalize(poses[:, :3, 1].sum(0))
    # Find a reasonable 'focus depth' for this dataset.
    close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
    dt = .75
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = mean_dz
    # Get radii for spiral path.
    tt = poses[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    n_views = 120
    n_rots = 2
    # Generate poses for spiral path.
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w_path[:, 4:5]
    zrate = .5
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_views + 1)[:-1]:
      c = np.dot(c2w[:3, :4], (np.array(
          [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads))
      z = self._normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
      render_poses.append(np.concatenate([self._viewmatrix(z, up, c), hwf], 1))
    self.render_poses = np.array(render_poses).astype(np.float32)[:, :3, :4]

  def _generate_spherical_poses(self, poses, bds):
    """Generate a 360 degree spherical path for rendering."""
    # pylint: disable=g-long-lambda
    p34_to_44 = lambda p: np.concatenate([
        p,
        np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])
    ], 1)
    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
      a_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
      b_i = -a_i @ rays_o
      pt_mindist = np.squeeze(-np.linalg.inv(
          (np.transpose(a_i, [0, 2, 1]) @ a_i).mean(0)) @ (b_i).mean(0))
      return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)
    vec0 = self._normalize(up)
    vec1 = self._normalize(np.cross([.1, .2, .3], vec0))
    vec2 = self._normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)
    poses_reset = (
        np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4]))
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc
    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2 - zh**2)
    new_poses = []

    for th in np.linspace(0., 2. * np.pi, 120):
      camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
      up = np.array([0, 0, -1.])
      vec2 = self._normalize(camorigin)
      vec0 = self._normalize(np.cross(vec2, up))
      vec1 = self._normalize(np.cross(vec2, vec0))
      pos = camorigin
      p = np.stack([vec0, vec1, vec2, pos], 1)
      new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    new_poses = np.concatenate([
        new_poses,
        np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)
    ], -1)
    poses_reset = np.concatenate([
        poses_reset[:, :3, :4],
        np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)
    ], -1)
    if self.split == 'test':
      self.render_poses = new_poses[:, :3, :4]
    return poses_reset


dataset_dict = {
    'blender': Blender,
    'llff': LLFF,
}
