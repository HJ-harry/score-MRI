import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from fastmri_utils import fft2c_new, ifft2c_new
from statistics import mean, stdev
from sigpy.mri import poisson


"""
Helper functions for new types of inverse problems
"""

def fft2(x):
  """ FFT with shifting DC to the center of the image"""
  return torch.fft.fftshift(torch.fft.fft2(x), dim=[-1, -2])


def ifft2(x):
  """ IFFT with shifting DC to the corner of the image prior to transform"""
  return torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-1, -2]))


def fft2_m(x):
  """ FFT for multi-coil """
  return torch.view_as_complex(fft2c_new(torch.view_as_real(x)))


def ifft2_m(x):
  """ IFFT for multi-coil """
  return torch.view_as_complex(ifft2c_new(torch.view_as_real(x)))


def crop_center(img, cropx, cropy):
  c, y, x = img.shape
  startx = x // 2 - (cropx // 2)
  starty = y // 2 - (cropy // 2)
  return img[:, starty:starty + cropy, startx:startx + cropx]


def normalize(img):
  """ Normalize img in arbitrary range to [0, 1] """
  img -= torch.min(img)
  img /= torch.max(img)
  return img


def normalize_np(img):
  """ Normalize img in arbitrary range to [0, 1] """
  img -= np.min(img)
  img /= np.max(img)
  return img


def normalize_complex(img):
  """ normalizes the magnitude of complex-valued image to range [0, 1] """
  abs_img = normalize(torch.abs(img))
  ang_img = normalize(torch.angle(img))
  return abs_img * torch.exp(1j * ang_img)


class lambda_schedule:
  def __init__(self, total=2000):
    self.total = total

  def get_current_lambda(self, i):
    pass


class lambda_schedule_linear(lambda_schedule):
  def __init__(self, start_lamb=1.0, end_lamb=0.0):
    super().__init__()
    self.start_lamb = start_lamb
    self.end_lamb = end_lamb

  def get_current_lambda(self, i):
    return self.start_lamb + (self.end_lamb - self.start_lamb) * (i / self.total)


class lambda_schedule_const(lambda_schedule):
  def __init__(self, lamb=1.0):
    super().__init__()
    self.lamb = lamb

  def get_current_lambda(self, i):
    return self.lamb


def clear(x):
  return x.detach().cpu().squeeze().numpy()

def clear_color(x):
  x = x.detach().cpu().squeeze().numpy()
  return np.transpose(x, (1, 2, 0))


def get_mask(img, size, batch_size, type='gaussian2d', acc_factor=8, center_fraction=0.04, fix=False):
  mux_in = size ** 2
  if type.endswith('2d'):
    Nsamp = mux_in // acc_factor
  elif type.endswith('1d'):
    Nsamp = size // acc_factor
  if type == 'gaussian2d':
    mask = torch.zeros_like(img)
    cov_factor = size * (1.5 / 128)
    mean = [size // 2, size // 2]
    cov = [[size * cov_factor, 0], [0, size * cov_factor]]
    if fix:
      samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
      int_samples = samples.astype(int)
      int_samples = np.clip(int_samples, 0, size - 1)
      mask[..., int_samples[:, 0], int_samples[:, 1]] = 1
    else:
      for i in range(batch_size):
        # sample different masks for batch
        samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
        int_samples = samples.astype(int)
        int_samples = np.clip(int_samples, 0, size - 1)
        mask[i, :, int_samples[:, 0], int_samples[:, 1]] = 1
  elif type == 'uniformrandom2d':
    mask = torch.zeros_like(img)
    if fix:
      mask_vec = torch.zeros([1, size * size])
      samples = np.random.choice(size * size, int(Nsamp))
      mask_vec[:, samples] = 1
      mask_b = mask_vec.view(size, size)
      mask[:, ...] = mask_b
    else:
      for i in range(batch_size):
        # sample different masks for batch
        mask_vec = torch.zeros([1, size * size])
        samples = np.random.choice(size * size, int(Nsamp))
        mask_vec[:, samples] = 1
        mask_b = mask_vec.view(size, size)
        mask[i, ...] = mask_b
  elif type == 'gaussian1d':
    mask = torch.zeros_like(img)
    mean = size // 2
    std = size * (15.0 / 128)
    Nsamp_center = int(size * center_fraction)
    if fix:
      samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp * 1.2))
      int_samples = samples.astype(int)
      int_samples = np.clip(int_samples, 0, size - 1)
      mask[... , int_samples] = 1
      c_from = size // 2 - Nsamp_center // 2
      mask[... , c_from:c_from + Nsamp_center] = 1
    else:
      for i in range(batch_size):
        samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp*1.2))
        int_samples = samples.astype(int)
        int_samples = np.clip(int_samples, 0, size - 1)
        mask[i, :, :, int_samples] = 1
        c_from = size // 2 - Nsamp_center // 2
        mask[i, :, :, c_from:c_from + Nsamp_center] = 1
  elif type == 'uniform1d':
    mask = torch.zeros_like(img)
    if fix:
      Nsamp_center = int(size * center_fraction)
      samples = np.random.choice(size, int(Nsamp - Nsamp_center))
      mask[..., samples] = 1
      # ACS region
      c_from = size // 2 - Nsamp_center // 2
      mask[..., c_from:c_from + Nsamp_center] = 1
    else:
      for i in range(batch_size):
        Nsamp_center = int(size * center_fraction)
        samples = np.random.choice(size, int(Nsamp - Nsamp_center))
        mask[i, :, :, samples] = 1
        # ACS region
        c_from = size // 2 - Nsamp_center // 2
        mask[i, :, :, c_from:c_from+Nsamp_center] = 1
  elif type == 'poisson':
    mask = poisson((size, size), accel=acc_factor)
    mask = torch.from_numpy(mask)
  else:
    NotImplementedError(f'Mask type {type} is currently not supported.')

  return mask


def kspace_to_nchw(tensor):
    """
    Convert torch tensor in (Slice, Coil, Height, Width, Complex) 5D format to
    (N, C, H, W) 4D format for processing by 2D CNNs.

    Complex indicates (real, imag) as 2 channels, the complex data format for Pytorch.

    C is the coils interleaved with real and imaginary values as separate channels.
    C is therefore always 2 * Coil.

    Singlecoil data is assumed to be in the 5D format with Coil = 1

    Args:
        tensor (torch.Tensor): Input data in 5D kspace tensor format.
    Returns:
        tensor (torch.Tensor): tensor in 4D NCHW format to be fed into a CNN.
    """
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dim() == 5
    s = tensor.shape
    assert s[-1] == 2
    tensor = tensor.permute(dims=(0, 1, 4, 2, 3)).reshape(shape=(s[0], 2 * s[1], s[2], s[3]))
    return tensor


def nchw_to_kspace(tensor):
  """
  Convert a torch tensor in (N, C, H, W) format to the (Slice, Coil, Height, Width, Complex) format.

  This function assumes that the real and imaginary values of a coil are always adjacent to one another in C.
  If the coil dimension is not divisible by 2, the function assumes that the input data is 'real' data,
  and thus pads the imaginary dimension as 0.
  """
  assert isinstance(tensor, torch.Tensor)
  assert tensor.dim() == 4
  s = tensor.shape
  if tensor.shape[1] == 1:
    imag_tensor = torch.zeros(s, device=tensor.device)
    tensor = torch.cat((tensor, imag_tensor), dim=1)
    s = tensor.shape
  tensor = tensor.view(size=(s[0], s[1] // 2, 2, s[2], s[3])).permute(dims=(0, 1, 3, 4, 2))
  return tensor


def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.
    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform
    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim))


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def restore_checkpoint(ckpt_dir, state, device, skip_sigma=False):
  loaded_state = torch.load(ckpt_dir, map_location=device)
  loaded_model_state = loaded_state['model']
  if skip_sigma:
    loaded_model_state.pop('module.sigmas')

  state['model'].load_state_dict(loaded_model_state, strict=False)
  state['ema'].load_state_dict(loaded_state['ema'])
  state['step'] = loaded_state['step']
  print(f'loaded checkpoint dir from {ckpt_dir}')
  return state