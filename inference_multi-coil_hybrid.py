from pathlib import Path
from models import utils as mutils
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      get_pc_fouriercs_RI_coil_SENSE)
from models import ncsnpp
import time
from utils import fft2_m, ifft2_m, get_mask, get_data_scaler, get_data_inverse_scaler, restore_checkpoint, \
    normalize_complex, root_sum_of_squares, lambda_schedule_const, lambda_schedule_linear
import torch
import torch.nn as nn
import numpy as np
from models.ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import importlib
import argparse
import sigpy.mri as mr


def main():
    ###############################################
    # 1. Configurations
    ###############################################

    # args
    args = create_argparser().parse_args()
    N = args.N
    m = args.m
    fname = args.data
    filename = f'./samples/multi-coil/{fname}.npy'

    print('initaializing...')
    configs = importlib.import_module(f"configs.ve.fastmri_knee_320_ncsnpp_continuous")
    config = configs.get_config()
    img_size = config.data.image_size
    batch_size = 1

    schedule = 'linear'
    start_lamb = 1.0
    end_lamb = 0.2
    m_steps = 50

    if schedule == 'const':
        lamb_schedule = lambda_schedule_const(lamb=start_lamb)
    elif schedule == 'linear':
        lamb_schedule = lambda_schedule_linear(start_lamb=start_lamb, end_lamb=end_lamb)
    else:
        NotImplementedError(f"Given schedule {schedule} not implemented yet!")

    # Read data
    img = normalize_complex(torch.from_numpy(np.load(filename).astype(np.complex64)))
    img = img.view(1, 15, 320, 320)
    img = img.to(config.device)

    mask = get_mask(img, img_size, batch_size,
                    type=args.mask_type,
                    acc_factor=args.acc_factor,
                    center_fraction=args.center_fraction)

    ckpt_filename = f"./weights/checkpoint_95.pth"
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=N)

    config.training.batch_size = batch_size
    predictor = ReverseDiffusionPredictor
    corrector = LangevinCorrector
    probability_flow = False
    snr = 0.16

    # sigmas = mutils.get_sigmas(config)
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # create model and load checkpoint
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, model=score_model, ema=ema)
    state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True)
    ema.copy_to(score_model.parameters())

    # Specify save directory for saving generated samples
    save_root = Path(f'./results/multi-coil/hybrid')
    save_root.mkdir(parents=True, exist_ok=True)

    irl_types = ['input', 'recon', 'recon_progress', 'label']
    for t in irl_types:
        save_root_f = save_root / t
        save_root_f.mkdir(parents=True, exist_ok=True)

    ###############################################
    # 2. Inference
    ###############################################
    mps_dir = save_root / f'sens.npy'
    # fft
    kspace = fft2_m(img)

    # undersampling
    under_kspace = kspace * mask
    under_img = ifft2_m(under_kspace)

    # ESPiRiT
    if mps_dir.exists():
        mps = np.load(str(mps_dir))
    else:
        mps = mr.app.EspiritCalib(kspace.cpu().detach().squeeze().numpy()).run()
        np.save(str(save_root / f'sens.npy'), mps)
    mps = torch.from_numpy(mps).view(1, 15, 320, 320).to(kspace.device)

    pc_fouriercs = get_pc_fouriercs_RI_coil_SENSE(sde,
                                                  predictor, corrector,
                                                  inverse_scaler,
                                                  snr=snr,
                                                  n_steps=m,
                                                  m_steps=50,
                                                  mask=mask,
                                                  sens=mps,
                                                  lamb_schedule=lamb_schedule,
                                                  probability_flow=probability_flow,
                                                  continuous=config.training.continuous,
                                                  denoise=True)

    print(f'Beginning inference')
    tic = time.time()
    x = pc_fouriercs(score_model, scaler(under_img), y=under_kspace)
    toc = time.time() - tic
    print(f'Time took for recon: {toc} secs.')

    ###############################################
    # 3. Saving recon
    ###############################################
    under_img = root_sum_of_squares(under_img, dim=1)
    label = root_sum_of_squares(img, dim=1)
    input = under_img.squeeze().cpu().detach().numpy()
    label = label.squeeze().cpu().detach().numpy()
    mask_sv = mask[0, 0, :, :].squeeze().cpu().detach().numpy()

    np.save(str(save_root / 'input' / fname) + '.npy', input)
    np.save(str(save_root / 'input' / (fname + '_mask')) + '.npy', mask_sv)
    np.save(str(save_root / 'label' / fname) + '.npy', label)
    plt.imsave(str(save_root / 'input' / fname) + '.png', np.abs(input), cmap='gray')
    plt.imsave(str(save_root / 'label' / fname) + '.png', np.abs(label), cmap='gray')

    x = root_sum_of_squares(x, dim=1)
    recon = x.squeeze().cpu().detach().numpy()
    np.save(str(save_root / 'recon' / fname) + '.npy', recon)
    plt.imsave(str(save_root / 'recon' / fname) + '.png', np.abs(recon), cmap='gray')


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='which data to use for reconstruction', required=True)
    parser.add_argument('--mask_type', type=str, help='which mask to use for retrospective undersampling.'
                                                      '(NOTE) only used for retrospective model!', default='gaussian1d',
                        choices=['gaussian1d', 'uniform1d', 'gaussian2d'])
    parser.add_argument('--acc_factor', type=int, help='Acceleration factor for Fourier undersampling.'
                                                       '(NOTE) only used for retrospective model!', default=4)
    parser.add_argument('--center_fraction', type=float, help='Fraction of ACS region to keep.'
                                                       '(NOTE) only used for retrospective model!', default=0.08)
    parser.add_argument('--save_dir', default='./results')
    parser.add_argument('--N', type=int, help='Number of iterations for score-POCS sampling', default=500)
    parser.add_argument('--m', type=int, help='Number of corrector step per single predictor step.'
                                              'It is advised not to change this default value.', default=1)
    return parser


if __name__ == "__main__":
    main()