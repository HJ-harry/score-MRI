from pathlib import Path
from models import utils as mutils
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      get_pc_fouriercs_fast)
from models import ncsnpp
import time
from utils import fft2, ifft2, get_mask, get_data_scaler, get_data_inverse_scaler, restore_checkpoint
import torch
import torch.nn as nn
import numpy as np
from models.ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import importlib
import argparse


def main():
    ###############################################
    # 1. Configurations
    ###############################################

    # args
    args = create_argparser().parse_args()
    N = args.N
    m = args.m
    # fname = '001'
    fname = args.data
    filename = f'./samples/real/{args.task}/{fname}.npy'
    mask_filename = f'./samples/real/prospective/{fname}_mask.npy'

    print('initaializing...')
    configs = importlib.import_module(f"configs.ve.fastmri_knee_320_ncsnpp_continuous")
    config = configs.get_config()
    img_size = config.data.image_size
    batch_size = 1

    # Read data
    img = torch.from_numpy(np.load(filename))
    img = img.view(1, 1, 320, 320)
    img = img.to(config.device)
    if args.task == 'retrospective':
        # generate mask
        mask = get_mask(img, img_size, batch_size,
                        type=args.mask_type,
                        acc_factor=args.acc_factor,
                        center_fraction=args.center_fraction)
    elif args.task == 'prospective':
        mask = torch.from_numpy(np.load(mask_filename))
        mask = mask.view(1, 1, 320, 320)

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
    save_root = Path(f'./results/real')
    save_root.mkdir(parents=True, exist_ok=True)

    irl_types = ['input', 'recon', 'recon_progress', 'label']
    for t in irl_types:
        save_root_f = save_root / t
        save_root_f.mkdir(parents=True, exist_ok=True)

    ###############################################
    # 2. Inference
    ###############################################

    pc_fouriercs = get_pc_fouriercs_fast(sde,
                                         predictor, corrector,
                                         inverse_scaler,
                                         snr=snr,
                                         n_steps=m,
                                         probability_flow=probability_flow,
                                         continuous=config.training.continuous,
                                         denoise=True,
                                         save_progress=True,
                                         save_root=save_root / 'recon_progress')
    # fft
    kspace = fft2(img)

    # undersampling
    under_kspace = kspace * mask
    under_img = torch.real(ifft2(under_kspace))

    print(f'Beginning inference')
    tic = time.time()
    x = pc_fouriercs(score_model, scaler(under_img), mask, Fy=under_kspace)
    toc = time.time() - tic
    print(f'Time took for recon: {toc} secs.')

    ###############################################
    # 3. Saving recon
    ###############################################

    if args.task == 'retrospective':
        # save input and label only if this is retrospective recon.
        input = under_img.squeeze().cpu().detach().numpy()
        label = img.squeeze().cpu().detach().numpy()
        mask_sv = mask.squeeze().cpu().detach().numpy()

        np.save(str(save_root / 'input' / fname) + '.npy', input)
        np.save(str(save_root / 'input' / (fname + '_mask')) + '.npy', mask_sv)
        np.save(str(save_root / 'label' / fname) + '.npy', label)
        plt.imsave(str(save_root / 'input' / fname) + '.png', input, cmap='gray')
        plt.imsave(str(save_root / 'label' / fname) + '.png', label, cmap='gray')

    recon = x.squeeze().cpu().detach().numpy()
    np.save(str(save_root / 'recon' / fname) + '.npy', recon)
    plt.imsave(str(save_root / 'recon' / fname) + '.png', recon, cmap='gray')


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['retrospective', 'prospective'], default='retrospective',
                        type=str, help='If retrospective, under-samples the fully-sampled data with generated mask.'
                                       'If prospective, runs score-POCS with the given mask')
    parser.add_argument('--data', type=str, help='which data to use for reconstruction', required=True)
    parser.add_argument('--mask_type', type=str, help='which mask to use for retrospective undersampling.'
                                                      '(NOTE) only used for retrospective model!', default='gaussian1d',
                        choices=['gaussian1d', 'uniform1d', 'gaussian2d'])
    parser.add_argument('--acc_factor', type=int, help='Acceleration factor for Fourier undersampling.'
                                                       '(NOTE) only used for retrospective model!', default=4)
    parser.add_argument('--center_fraction', type=float, help='Fraction of ACS region to keep.'
                                                       '(NOTE) only used for retrospective model!', default=0.08)
    parser.add_argument('--save_dir', default='./results')
    parser.add_argument('--N', type=int, help='Number of iterations for score-POCS sampling', default=2000)
    parser.add_argument('--m', type=int, help='Number of corrector step per single predictor step.'
                                              'It is advised not to change this default value.', default=1)
    return parser


if __name__ == "__main__":
    main()