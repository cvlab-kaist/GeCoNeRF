# GeCo-NeRF

This code is the official implementation of the paper [GeCoNeRF: Few-shot Neural Radiance Fields via Geometric Consistency](https://arxiv.org/abs/2301.10941). This implementation is written in [JAX](https://github.com/google/jax).

## Abstract
We present a novel framework to regularize Neural Radiance Field (NeRF) in a few-shot setting with a geometry-aware consistency regularization. The proposed approach leverages a rendered depth map at unobserved viewpoint to warp sparse input images to the unobserved viewpoint and impose them as pseudo ground truths to facilitate learning of NeRF. By encouraging such geometry-aware consistency at a feature-level instead of using pixel-level reconstruction loss, we regularize the NeRF at semantic and structural levels while allowing for modeling view dependent radiance to account for color variations across viewpoints. We also propose an effective method to filter out erroneous warped solutions, along with training strategies to stabilize training during optimization. We show that our model achieves competitive results compared to state-of-the-art few-shot NeRF models. 

## Installation

We recommend using [Anaconda](https://www.anaconda.com/products/individual) to set
up the environment. Run the following commands:

```
# Clone the repo
git clone https://github.com/KU-CVLAB/GeCoNeRF.git; cd geconerf
conda create --nam e geconerf python=3.6.13; conda activate geconerf
conda install pip; pip install --upgrade pip
pip install --upgrade jax jaxlib==0.1.68+cuda110 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```

**Note** : As of 2024.03.04, we find that the codebase currently uploaded on Github contains an error, with its results on LLFF falling short its original performance stated in our paper due to its currently erroneous, downgraded performance. We intend to fix this error as soon as possible, and will update our codebase in a short notice.

## Data

Then, you'll need to download the datasets
from the [NeRF official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
Please download and unzip `nerf_synthetic.zip` and `nerf_llff_data.zip`.


## Running

Example scripts for training GeCoNeRF on individual scenes from datasets used in the paper can be found in `scripts/`. 
You'll need to change the paths to point to wherever the datasets are located.
To train a GeCoNeRF on the example from `Blender dataset` :
```
scripts/train_blender.py
```
To train a GeCoNeRF on the example from `LLFF dataset` :
```
scripts/train_llff.py
```


## Citation
If you use this software package, please cite our paper:

```
@article{kwak2023geconerf,
  title={GeCoNeRF: Few-shot Neural Radiance Fields via Geometric Consistency},
  author={Kwak, Minseop and Song, Jiuhn and Kim, Seungryong},
  journal={arXiv preprint arXiv:2301.10941},
  year={2023}
}
```

## Acknowledgements
This code heavily borrows from [mip-NeRF](https://github.com/google/mipnerf).


