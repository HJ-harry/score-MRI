# Score-POCS for accelerated MRI

This project contains the implementation of ```score-POCS```, introduced in this [paper](https://arxiv.org/abs/2110.05243).

## Brief explanation of the inference procedure

Running ```sampling.get_pc_fouriercs_fast``` is equivalent to solving Algorithm 2
in the [paper](https://arxiv.org/abs/2110.05243). It iteratively applied ```N```
number of **precitor-corrector** sampling with data consistency projection steps in-between.
Hence, the reconstruction starts from random noise, gradually updated closer and closer to
a clean reconstructed image.

## Installation

```bash
source install.sh
```
Above installation script will handle downloading model weights, and installing dependencies.
Alternatively, you can download the model weights [here](?), and place it as ```weights/checkpoint_95.pth```.

## Project structure

```bash
├── configs
│   ├── default_lsun_configs.py
│   │   └── default_lsun_configs.cpython-38.pyc
│   └── ve
│       ├── fastmri_knee_320_ncsnpp_continuous.py
├── fastmri_utils.py
├── utils.py
├── models
│   ├── ...
├── op
│   ├── ...
├── samples
│   ├── ...
├── sampling.py
├── sde_lib.py
└── inference_real.py
```

1. ```configs```: contains the hyper-parameters for defining neural nets, sampling procedure, and so on.
Ordered in the form of ```ml_collections```.
2. ```fastmri_utils.py, utils.py```: ```utils.py``` contains helper functions used in pre/post-processing of data. It also
wraps ```fastmri_utils.py```, which contains helper functions related to Fourier transforms required in MRI reconstructions.
3. ```models```: This directory contains files that are required for defining the ```ncsnpp``` model, which is a
heavy U-Net architecture with several modifications including transformer atention, Fourier features, and anti-aliasing down/up-sampling.
4. ```ops```: This directory contains CUDA kernels that are used in ```ncsnpp```.
5. ```samples```: contains sample MR images to test the code.
6. ```sampling.py```: Contains Algorithm 1 of the paper. Workhorse for reconstruction.
7. ```sde_lib.py```: Defines VE-SDE of eq. (3),(4), and (6).
8. ```inference.py```: main script for inference.

## Inference

### Retrospective inference

Default mode for inference is retrospective mode. In this mode, the user needs to prepare a single image from fully-sampled k-space.
In order to specify the mask to use for under-sampling, control the following: ```--mask_type, --acc_factor, --center_fraction```.
The ```mask_type``` argument will be one of ```'gaussian1d`, 'uniform1d', 'gaussian2d' ```. For example, one can run the below command.

```python
python inference.py --task 'retrospective' \
                    --data '001' \
                    --mask_type 'gaussian1d' \
                    --acc_factor 4 \
                    --center_fraction 0.08 \
                    --N 500
```

### Prospective inference

You can also perform prospective inference, given that you have matching pairs of aliased image from under-sampled k-space, and the corresponding mask.
We expect the matching filnames be ```{filename}.npy, {filename}_mask.npy```. In this case, you can run, for example, the following:

```python
python inference.py --task 'prospective' \
                    --data '001' \
                    --N 500
```


