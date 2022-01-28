# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools
import time

import torch
import numpy as np
import abc

import matplotlib.pyplot as plt
import functools
from utils import fft2, ifft2, clear, fft2_m, ifft2_m, root_sum_of_squares
from tqdm import tqdm
from models import utils as mutils

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_corrector(name):
  return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config.sampling.method
  predictor = get_predictor(config.sampling.predictor.lower())
  corrector = get_corrector(config.sampling.corrector.lower())
  sampling_fn = get_pc_sampler(sde=sde,
                               shape=shape,
                               predictor=predictor,
                               corrector=corrector,
                               inverse_scaler=inverse_scaler,
                               snr=config.sampling.snr,
                               n_steps=config.sampling.n_steps_each,
                               probability_flow=config.sampling.probability_flow,
                               continuous=config.training.continuous,
                               denoise=config.sampling.noise_removal,
                               eps=eps,
                               device=config.device)
  return sampling_fn


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    f, G = self.rsde.discretize(x, t)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[:, None, None, None] * z
    return x, x_mean


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    alpha = torch.ones_like(t)

    for i in range(n_steps):
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    return x, x_mean


def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  predictor_obj = predictor(sde, score_fn, probability_flow)
  return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper tha configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  corrector_obj = corrector(sde, score_fn, snr, n_steps)
  fn = corrector_obj.update_fn(x, t)
  return fn


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def pc_sampler(model):
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      x = sde.prior_sampling(shape).to(device)
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

      time_corrector_tot = 0
      time_predictor_tot = 0
      for i in range(sde.N):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        tic_corrector = time.time()
        x, x_mean = corrector_update_fn(x, vec_t, model=model)
        time_corrector_tot += time.time() - tic_corrector
        tic_predictor = time.time()
        x, x_mean = predictor_update_fn(x, vec_t, model=model)
        time_predictor_tot += time.time() - tic_predictor
      print(f'Average time for corrector step: {time_corrector_tot / sde.N} sec.')
      print(f'Average time for predictor step: {time_predictor_tot / sde.N} sec.')

      return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

  return pc_sampler


def get_pc_fouriercs_fast(sde, predictor, corrector, inverse_scaler, snr,
                          n_steps=1, probability_flow=False, continuous=False,
                          denoise=True, eps=1e-5, save_progress=False, save_root=None):
  """Create a PC sampler for solving compressed sensing problems as in MRI reconstruction.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
    corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for the corrector.
    n_steps: An integer. The number of corrector steps per update of the corrector.
    continuous: `True` indicates that the score-based model was trained with continuous time.
    denoise: If `True`, add one-step denoising to final samples.
    eps: A `float` number. The reverse-time SDE/ODE is integrated to `eps` for numerical stability.

  Returns:
    A CS solver function.
  """
  # Define predictor & corrector
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def data_fidelity(mask, x, Fy):
      """
      Data fidelity operation for Fourier CS
      x: Current aliased img
      Fy: k-space measurement data (masked)
      """
      x = torch.real(ifft2(fft2(x) * (1. - mask) + Fy))
      x_mean = torch.real(ifft2(fft2(x) * (1. - mask) + Fy))
      return x, x_mean

  def get_fouriercs_update_fn(update_fn):
    """Modify the update function of predictor & corrector to incorporate data information."""

    def fouriercs_update_fn(model, data, mask, x, t, Fy=None):
      with torch.no_grad():
        vec_t = torch.ones(data.shape[0], device=data.device) * t
        x, x_mean = update_fn(x, vec_t, model=model)
        x, x_mean = data_fidelity(mask, x, Fy)
        return x, x_mean

    return fouriercs_update_fn

  projector_fouriercs_update_fn = get_fouriercs_update_fn(predictor_update_fn)
  corrector_fouriercs_update_fn = get_fouriercs_update_fn(corrector_update_fn)

  def pc_fouriercs(model, data, mask, Fy=None):
    with torch.no_grad():
      # Initial sample
      x = torch.real(ifft2(Fy + fft2(sde.prior_sampling(data.shape).to(data.device)) * (1. - mask)))
      timesteps = torch.linspace(sde.T, eps, sde.N)
      for i in tqdm(range(sde.N), total=sde.N):
        t = timesteps[i]
        x, x_mean = corrector_fouriercs_update_fn(model, data, mask, x, t, Fy=Fy)
        x, x_mean = projector_fouriercs_update_fn(model, data, mask, x, t, Fy=Fy)
        if save_progress and i >= 300 and i % 100 == 0:
          plt.imsave(save_root / f'step{i}.png', clear(x_mean), cmap='gray')

      return inverse_scaler(x_mean if denoise else x)

  return pc_fouriercs


def get_pc_fouriercs_RI(sde, predictor, corrector, inverse_scaler, snr,
                        n_steps=1, probability_flow=False, continuous=False,
                        denoise=True, eps=1e-5):
  # Define predictor & corrector
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def data_fidelity(mask, x, x_mean, Fy):
      x = ifft2(fft2(x) * (1. - mask) + Fy)
      x_mean = ifft2(fft2(x_mean) * (1. - mask) + Fy)
      return x, x_mean

  def get_fouriercs_update_fn(update_fn):
    def fouriercs_update_fn(model, data, mask, x, t, Fy=None):
      with torch.no_grad():
        vec_t = torch.ones(data.shape[0], device=data.device) * t
        # split real / imag part
        x_real = torch.real(x)
        x_imag = torch.imag(x)

        # perform update step with real / imag part seperately
        x_real, x_real_mean = update_fn(x_real, vec_t, model=model)
        x_imag, x_imag_mean = update_fn(x_imag, vec_t, model=model)

        # merge real / imag values to form complex image
        x = x_real + 1j * x_imag
        x_mean = x_real_mean + 1j * x_imag_mean
        x, x_mean = data_fidelity(mask, x, x_mean, Fy)
        return x, x_mean

    return fouriercs_update_fn

  projector_fouriercs_update_fn = get_fouriercs_update_fn(predictor_update_fn)
  corrector_fouriercs_update_fn = get_fouriercs_update_fn(corrector_update_fn)

  def pc_fouriercs(model, data, mask, Fy=None):
    with torch.no_grad():
      # Initial sample (complex-valued)
      x = ifft2(Fy + fft2(sde.prior_sampling(data.shape).to(data.device)) * (1. - mask))
      timesteps = torch.linspace(sde.T, eps, sde.N)
      for i in tqdm(range(sde.N)):
        t = timesteps[i]
        x, x_mean = corrector_fouriercs_update_fn(model, data, mask, x, t, Fy=Fy)
        x, x_mean = projector_fouriercs_update_fn(model, data, mask, x, t, Fy=Fy)

      return inverse_scaler(x_mean if denoise else x)

  return pc_fouriercs


def get_pc_fouriercs_RI_PI_SSOS(sde, predictor, corrector, inverse_scaler, snr,
                                n_steps=1, probability_flow=False, continuous=False,
                                denoise=True, eps=1e-5, mask=None,
                                save_progress=False, save_root=None):
  # Define predictor & corrector
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  # functions to impose data fidelity 1/2\|Ax - y\|^2
  def data_fidelity(mask, x, x_mean, y):
      x = ifft2_m(fft2_m(x) * (1. - mask) + y)
      x_mean = ifft2_m(fft2_m(x_mean) * (1. - mask) + y)
      return x, x_mean

  def get_coil_update_fn(update_fn):
    def fouriercs_update_fn(model, data, x, t, y=None):
      with torch.no_grad():
        vec_t = torch.ones(data.shape[0], device=data.device) * t
        # split real / imag part
        x_real = torch.real(x)
        x_imag = torch.imag(x)

        # perform update step with real / imag part seperately
        x_real, x_real_mean = update_fn(x_real, vec_t, model=model)
        x_imag, x_imag_mean = update_fn(x_imag, vec_t, model=model)

        # merge real / imag values to form complex image
        x = x_real + 1j * x_imag
        x_mean = x_real_mean + 1j * x_imag_mean

        # coil mask
        mask_c = mask[0, 0, :, :].squeeze()
        x, x_mean = data_fidelity(mask_c, x, x_mean, y)
        return x, x_mean

    return fouriercs_update_fn

  predictor_coil_update_fn = get_coil_update_fn(predictor_update_fn)
  corrector_coil_update_fn = get_coil_update_fn(corrector_update_fn)

  def pc_fouriercs(model, data, y=None):
    with torch.no_grad():
      # Initial sample: [1, 15, 320, 320] (dtype: torch.complex64)
      x_r = sde.prior_sampling(data.shape).to(data.device)
      x_i = sde.prior_sampling(data.shape).to(data.device)
      x = torch.complex(x_r, x_i)
      x_mean = x.clone().detach()

      timesteps = torch.linspace(sde.T, eps, sde.N)

      # number of iterations of PC sampler
      for i in tqdm(range(sde.N)):
        # coil x_c update
        for c in range(15):
          t = timesteps[i]
          # slicing the dimension with c:c+1 ("one-element slice") preserves dimension
          x_c = x[:, c:c+1, :, :]
          y_c = y[:, c:c+1, :, :]
          x_c, x_c_mean = predictor_coil_update_fn(model, data, x_c, t, y=y_c)
          x_c, x_c_mean = corrector_coil_update_fn(model, data, x_c, t, y=y_c)

          # Assign coil dates to the global x, x_mean
          x[:, c, :, :] = x_c
          x_mean[:, c, :, :] = x_c_mean
        if save_progress:
          if i % 100 == 0:
            for c in range(15):
              x_c = clear(x[:, c:c+1, :, :])
              plt.imsave(save_root / 'recon' / f'coil{c}' / f'after{i}.png', np.abs(x_c), cmap='gray')
            x_rss = clear(root_sum_of_squares(torch.abs(x), dim=1).squeeze())
            plt.imsave(save_root / 'recon' / f'after{i}.png', x_rss, cmap='gray')

      return inverse_scaler(x_mean if denoise else x)

  return pc_fouriercs


def get_pc_fouriercs_RI_coil_SENSE(sde, predictor, corrector, inverse_scaler, snr,
                                   n_steps=1, lamb_schedule=None, probability_flow=False, continuous=False,
                                   denoise=True, eps=1e-5, sens=None, mask=None, m_steps=10,
                                   save_progress=False, save_root=None):
  '''Every once in a while during separate coil reconstruction,
  apply SENSE data consistency and incorporate information.
  (Args)
    (sens): sensitivity maps
    (m_steps): frequency in which SENSE operation is incorporated
  '''
  # Define predictor & corrector
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  # functions to impose data fidelity 1/2\|Ax - y\|^2
  def data_fidelity(mask, x, x_mean, y):
      x = ifft2_m(fft2_m(x) * (1. - mask) + y)
      x_mean = ifft2_m(fft2_m(x_mean) * (1. - mask) + y)
      return x, x_mean

  def A(x, sens=sens, mask=mask):
      return mask * fft2_m(sens * x)

  def A_H(x, sens=sens, mask=mask):  # Hermitian transpose
      return torch.sum(torch.conj(sens) * ifft2_m(x * mask), dim=1).unsqueeze(dim=1)

  def kaczmarz(x, x_mean, y, lamb=1.0):
      x = x + lamb * A_H(y - A(x))
      x_mean = x_mean + lamb * A_H(y - A(x_mean))
      return x, x_mean

  def get_coil_update_fn(update_fn):
    def fouriercs_update_fn(model, data, x, t, y=None):
      with torch.no_grad():
        vec_t = torch.ones(data.shape[0], device=data.device) * t
        # split real / imag part
        x_real = torch.real(x)
        x_imag = torch.imag(x)

        # perform update step with real / imag part seperately
        x_real, x_real_mean = update_fn(x_real, vec_t, model=model)
        x_imag, x_imag_mean = update_fn(x_imag, vec_t, model=model)

        # merge real / imag values to form complex image
        x = x_real + 1j * x_imag
        x_mean = x_real_mean + 1j * x_imag_mean

        # coil mask
        mask_c = mask[0, 0, :, :].squeeze()
        x, x_mean = data_fidelity(mask_c, x, x_mean, y)
        return x, x_mean

    return fouriercs_update_fn

  predictor_coil_update_fn = get_coil_update_fn(predictor_update_fn)
  corrector_coil_update_fn = get_coil_update_fn(corrector_update_fn)

  def pc_fouriercs(model, data, y=None):
    with torch.no_grad():
      # Initial sample: [1, 15, 320, 320] (dtype: torch.complex64)
      x_r = sde.prior_sampling(data.shape).to(data.device)
      x_i = sde.prior_sampling(data.shape).to(data.device)
      x = torch.complex(x_r, x_i)
      x_mean = x.clone().detach()

      timesteps = torch.linspace(sde.T, eps, sde.N)

      # number of iterations of PC sampler
      for i in tqdm(range(sde.N)):
        # coil x_c update
        for c in range(15):
          t = timesteps[i]

          # slicing the dimension with c:c+1 ("one-element slice") preserves dimension
          x_c = x[:, c:c+1, :, :]
          y_c = y[:, c:c+1, :, :]
          x_c, x_c_mean = predictor_coil_update_fn(model, data, x_c, t, y=y_c)
          x_c, x_c_mean = corrector_coil_update_fn(model, data, x_c, t, y=y_c)

          # Assign coil dates to the global x, x_mean
          x[:, c, :, :] = x_c
          x_mean[:, c, :, :] = x_c_mean

        # global x update
        if i % m_steps == 0:
          lamb = lamb_schedule.get_current_lambda(i)
          x, x_mean = kaczmarz(x, x_mean, y, lamb=lamb)
        if save_progress:
          if i % 100 == 0:
            for c in range(15):
              x_c = clear(x[:, c:c+1, :, :])
              plt.imsave(save_root / 'recon' / f'coil{c}' / f'after{i}.png', np.abs(x_c), cmap='gray')
            x_rss = clear(root_sum_of_squares(torch.abs(x), dim=1).squeeze())
            plt.imsave(save_root / 'recon' / f'after{i}.png', x_rss, cmap='gray')

      return inverse_scaler(x_mean if denoise else x)

  return pc_fouriercs