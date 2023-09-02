
import enum
from hashlib import new
from ssl import cert_time_to_seconds

import jax
import gin
import jax.numpy as jnp
import numpy as np
from jax import lax
from flax import linen as nn
from jax import random
from scipy.spatial.transform import Rotation as R


from internal import math
from internal import utils
from internal import datasets


def pose_generator(pose_config, input_pose, data_type, divergence_angle = 0, theta=10, phi=7):        
    '''
    sample random poses in the upper hemisphere.
    Args:
        rng: jnp.ndarray, random number generator.
        bds: boundaries of radious
    Return:
        Translation-rotation matrix according to a random pose:
    '''

    if data_type == 'blender':
        radius = np.sqrt(16.25)
    elif data_type == 'dtu':
        radius = np.linalg.norm(input_pose[:,3])

    if pose_config == 'random_pose':

        # Completely New Pose Generation
        theta = np.random.uniform(-np.pi,np.pi)
        phi = np.random.uniform(0, np.pi/2)
        
        new_pose = pose_spherical(radius, theta, phi)

    elif pose_config == 'random_divergence':

        # Random pose between interval
        theta = np.random.uniform(-divergence_angle/2, divergence_angle/2)
        phi = np.random.uniform(-divergence_angle/2, divergence_angle/2)

        # Original Angle Decomposition
        divergence_angle = np.array([np.radians(phi),np.radians(theta), 0])
        input_pose_camera = np.array([[-1, 0, 0, 0], [0, 0, 1, 0],  [0, 1, 0, 0], [0, 0, 0, 1]]) @ input_pose

        rotation_angles = rotmat_to_euler(input_pose_camera)
        
        # Angle Jitter
        new_angles = divergence_angle + ( rotation_angles)

        # New Pose Generation
        new_pose = pose_spherical(radius, new_angles[1], new_angles[0])
    
    elif pose_config == 'set_divergence':
        # Pose Set (Phi, Theta)

        # Original Angle Decomposition
        divergence_angle = jnp.array([jnp.radians(phi),jnp.radians(theta), 0])
        input_pose_camera = jnp.array([[-1, 0, 0, 0], [0, 0, 1, 0],  [0, 1, 0, 0], [0, 0, 0, 1]]) @ input_pose
        rotation_angles = rotmat_to_euler(input_pose_camera)

        # print("Input Pose: " + str(rotation_angles))

        # Angle Jitter
        new_angles = divergence_angle + ( rotation_angles)

        # print("New Angles : " + str(new_angles))

        # New Pose Generation
        new_pose = pose_spherical(radius, new_angles[1], new_angles[0])

        new_pose_for_check = jnp.array([[-1, 0, 0, 0], [0, 0, 1, 0],  [0, 1, 0, 0], [0, 0, 0, 1]]) @ new_pose
        rotation_angles = rotmat_to_euler(new_pose_for_check[:3,:3])

        # print("New Pose: " + str(rotation_angles))

    return new_pose


def trans_t(t):
    return jnp.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]], dtype=jnp.float32)


def rot_phi(phi):
    return jnp.array([
        [1, 0, 0, 0],
        [0, jnp.cos(phi), -jnp.sin(phi), 0],
        [0, jnp.sin(phi), jnp.cos(phi), 0],
        [0, 0, 0, 1]], dtype=jnp.float32)


def rot_theta(th):
    return jnp.array([
        [jnp.cos(th), 0, jnp.sin(th), 0],
        [0, 1, 0, 0],
        [-jnp.sin(th), 0, jnp.cos(th), 0],
        [0, 0, 0, 1]], dtype=jnp.float32)


def pose_spherical(radius, theta, phi):
    '''
        The codes for generating random pose is based on https://github.com/yenchenlin/nerf-pytorch/blob/ec26d1c17d9ba2a897bc2ab254a0e15fce0d83b8/load_LINEMOD.py.
        and modified to make each coordinate indicates below.
               theta
                 ^
        - phi <= o => phi 
                /
            radius 
        Args:
            components of spharical coordinates.
    '''

    c2w = trans_t(radius)
    c2w = rot_phi(phi) @ c2w
    c2w = rot_theta(-theta) @ c2w
    c2w = jnp.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w

    return c2w

def rotmat_to_euler(R) :

    sy = jnp.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    x = jnp.arctan2(-R[1,2], R[1,1])
    y = jnp.arctan2(-R[2,0], sy)
    z = 0

    return jnp.array([x, y, z])


