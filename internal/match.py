
from calendar import c
import enum
from hashlib import new
from ssl import cert_time_to_seconds
from tkinter import N
from jax import lax
from jax import jit
from jax import random
import jax.numpy as jnp
import numpy as np

from internal import math
from internal import utils


def reprojector(source_rays, source_rgb, source_distance, source_pose, 
                target_pose, target_pose_inv, 
                key, data_type, n_sampled_pts = 1024, full=True, eval=False):
    # Only supports single pose per ray batch
    """
    When Inverse
    Source: Unseen view
    Target: GT view
    
    """ 
    rays = source_rays
    pts_locations = source_rays.directions * source_distance[..., None] + source_rays.origins

    # When needed, sample N points

    if full:
        pts_sampled = pts_locations

    elif n_sampled_pts < pts_locations.shape[0]:
        sampled_ind = random.choice(key, 1024, shape=(n_sampled_pts,), replace=False)
        pts_sampled = pts_locations[sampled_ind, :]
    
    else:
        pts_sampled = pts_locations
  
    if data_type == "blender" or data_type == "dtu":
        target_origin = target_pose[:3,-1]        
        target_origin_broad = jnp.broadcast_to(target_origin, pts_sampled.shape[-2:])

        if data_type == "blender":
            target_center_viewdir = (- target_pose[:3,2])
        elif data_type == "dtu":
            target_center_viewdir = (target_pose[:3,2])
    
        # Raymaker ########################### From Seen to Unseen
        pts_to_tgt_origin = pts_sampled - target_origin[None, :]
        
        dist_to_tgt_origin = jnp.linalg.norm(pts_to_tgt_origin, axis=-1, keepdims=True)
        target_viewdirs = pts_to_tgt_origin / dist_to_tgt_origin
        new_per_view_lengths = (target_viewdirs * target_center_viewdir[None, :]).sum(axis = -1)
        target_directions = target_viewdirs / new_per_view_lengths[..., None]
        
        # Reprojector: Given target view, where do the points fall? ###############
        worldtocamera = target_pose_inv
        target_cameradir = (target_directions[...,None,:] * worldtocamera[..., None, :3, :3]).sum(-1)
        target_projection = target_cameradir[...,:2]

        # Reshapes & rays object making
        if eval:         
            pose = jnp.broadcast_to(target_pose.reshape(-1,16), [pts_sampled.shape[1], 16])
        else: 
            pose = jnp.broadcast_to(target_pose.reshape(-1,16), [pts_sampled.shape[0], 16])

        ones = jnp.ones_like(target_origin_broad[...,:1])
        radii = rays.radii[:pts_sampled.shape[0],...]
        near = rays.near[:pts_sampled.shape[0],...]
        far = rays.far[:pts_sampled.shape[0],...]

        target_rays = utils.Rays(
            origins = target_origin_broad.reshape(rays.origins.shape),
            directions=target_directions.reshape(rays.directions.shape),
            viewdirs= target_viewdirs.reshape(rays.viewdirs.shape),
            pose = pose.reshape(rays.pose.shape),
            radii=radii.reshape(rays.radii.shape),
            lossmult = ones.reshape(rays.lossmult.shape),
            near = near.reshape(rays.near.shape),
            far = far.reshape(rays.far.shape)
        )
    

    elif data_type=="llff":
        pts_sampled = math.convert_ndc_to_raw(504, 378, 416.23373, 1.0, pts_sampled)

        # translate NDC space to raw coordinates space
        target_origin = target_pose[:3,-1]        
        target_origin_broad = jnp.broadcast_to(target_origin, pts_sampled.shape[-2:])
        target_center_viewdir = (- target_pose[:3,2])
        
        # Raymaker ########################### From Seen to Unseen
        pts_to_tgt_origin = pts_sampled - target_origin[None, :]
        dist_to_tgt_origin = jnp.linalg.norm(pts_to_tgt_origin, axis=-1, keepdims=True)
        target_viewdirs = pts_to_tgt_origin / dist_to_tgt_origin
        new_per_view_lengths = (target_viewdirs * target_center_viewdir[None, :]).sum(axis = -1)
        target_directions = target_viewdirs / new_per_view_lengths[..., None]
        
        # Reprojector: Given target view, where do the points fall? ###############
        worldtocamera = target_pose_inv
        target_cameradir = (target_directions[...,None,:] * worldtocamera[..., None, :3, :3]).sum(-1)
        target_projection = target_cameradir[...,:2]

        # Reshapes & rays object making
        if eval:         
            pose = jnp.broadcast_to(target_pose.reshape(-1,16), [pts_sampled.shape[1], 16])
        else: 
            pose = jnp.broadcast_to(target_pose.reshape(-1,16), [pts_sampled.shape[0], 16])
        
        origins = target_origin_broad.reshape(rays.origins.shape)
        directions = target_directions.reshape(rays.directions.shape)
        
        ndc_origins, ndc_directions = math.convert_to_ndc(origins, directions, 416.23373, 504, 378)

        ones = jnp.ones_like(target_origin_broad[...,:1])
        radii = rays.radii[:pts_sampled.shape[0],...]
        near = rays.near[:pts_sampled.shape[0],...]
        far = rays.far[:pts_sampled.shape[0],...]

        target_rays = utils.Rays(
            origins = ndc_origins,
            directions = ndc_directions,
            viewdirs= target_viewdirs.reshape(rays.viewdirs.shape),
            pose = pose.reshape(rays.pose.shape),
            radii=radii.reshape(rays.radii.shape),
            lossmult = ones.reshape(rays.lossmult.shape),
            near = near.reshape(rays.near.shape),
            far = far.reshape(rays.far.shape)
        )

    return target_projection, dist_to_tgt_origin, target_rays


def rotate_rays(x, y, z, rays):
  origins = rays.origins.reshape([-1,3])
  directions = rays.directions.reshape([-1,3])

  rotation_matrix = rotmat_generator(x, y, z)

  origin = ((origins[0, :] * rotation_matrix[:,:]).sum(axis=-1))

  directions = ((directions[:, None, :] * rotation_matrix[None,:,:]).sum(axis= -1))

  return origin, directions


def rotmat_generator(x, y, z):
  x = jnp.radians(x)
  y = jnp.radians(y)
  z = jnp.radians(z)

  rot_x = jnp.array([[1, 0, 0],
                     [0, jnp.cos(x), -jnp.sin(x)],
                     [0, jnp.sin(x), jnp.cos(x)]])
  rot_y = jnp.array([[jnp.cos(y), 0, jnp.sin(y)],
                     [0, 1, 0],
                     [-jnp.sin(y), 0, jnp.cos(y)]])
  rot_z = jnp.array([[jnp.cos(z), -jnp.sin(z), 0],
                     [jnp.sin(z), jnp.cos(z), 0],
                     [0, 0, 1]])

  rotmat = jnp.matmul(rot_z, jnp.matmul(rot_y, rot_x))

  return rotmat


def rotmat_from_vectors(vec1, vec2):
  """ Find the rotation matrix that aligns vec1 to vec2
  :param vec1: A 3d "source" vector
  :param vec2: A 3d "destination" vector
  :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
  """

  # Change this equation for tensors....
  a, b = (vec1 / jnp.linalg.norm(vec1)).reshape(3), (vec2 / jnp.linalg.norm(vec2)).reshape(3)
  v = jnp.cross(a, b)
  c = jnp.dot(a, b)
  s = jnp.linalg.norm(v)

  kmat = jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
  rotation_matrix = jnp.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
  return rotation_matrix