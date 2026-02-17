import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd

from glob import glob
from ntpath import basename
# from scipy.misc import imread
from imageio.v3 import imread
#from skimage.measure import compare_ssim
#from skimage.measure import compare_psnr
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from skimage.color import rgb2gray
from skimage.transform import resize

def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--data-path', help='Path to ground truth data', type=str)
    parser.add_argument('--output-path', help='Path to output data', type=str)
    parser.add_argument('--output-csv', help='filename of results CSV', type=str)
    parser.add_argument('--debug', default=0, help='Debug', type=int)
    args = parser.parse_args()
    return args


def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)


args = parse_args()
for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))

path_true = args.data_path
path_pred = args.output_path
path_csv = args.output_csv

psnr = []
ssim = []
mae = []
names = []
index = 1

files = list(glob(path_true + '/*.jpg')) + list(glob(path_true + '/*.png'))
for fn in sorted(files):
    name = basename(str(fn))
    names.append(name)

    img_gt = (imread(fn) / 255.0).astype(np.float32)
    img_pred = (imread(path_pred + '/' + basename(fn)) / 255.0).astype(np.float32)

    ssim_channel_axis = None
    if img_gt.ndim >= 3 :
        img_gt = rgb2gray(img_gt)
        ssim_channel_axis = None;

    if img_pred.ndim >= 3 :
        img_pred = rgb2gray(img_pred)
        ssim_channel_axis = None;

    if img_gt.shape != img_pred.shape :
        img_gt = resize(img_gt, img_pred.shape)

    if args.debug != 0:
        plt.subplot('121')
        plt.imshow(img_gt)
        plt.title('Groud truth')
        plt.subplot('122')
        plt.imshow(img_pred)
        plt.title('Output')
        plt.show()

    win_size = min (51, min(img_pred.shape))
    if win_size % 2 == 0:
        win_size = win_size - 1

    psnr.append(peak_signal_noise_ratio(img_gt, img_pred, data_range=1))
    ssim.append(structural_similarity(img_gt, img_pred, data_range=1, win_size=win_size, channel_axis = ssim_channel_axis))
    mae.append(compare_mae(img_gt, img_pred))

    if np.mod(index, 100) == 0:
        print(
            str(index) + ' images processed',
            "PSNR: %.4f" % round(np.mean(psnr), 4),
            "SSIM: %.4f" % round(np.mean(ssim), 4),
            "MAE: %.4f" % round(np.mean(mae), 4),
        )
    index = index + 1

# 計算結果をCSVファイルに出力
if path_csv != None :
    results_df = pd.DataFrame({'fname':names, 'psnr':psnr, 'ssim':ssim, 'mae':mae})
    results_df.to_csv(path_csv, index=False )

np.savez(args.output_path + '/metrics.npz', psnr=psnr, ssim=ssim, mae=mae, names=names)
print(
    "PSNR: %.4f" % round(np.mean(psnr), 4),
    "PSNR Variance: %.4f" % round(np.var(psnr), 4),
    "SSIM: %.4f" % round(np.mean(ssim), 4),
    "SSIM Variance: %.4f" % round(np.var(ssim), 4),
    "MAE: %.4f" % round(np.mean(mae), 4),
    "MAE Variance: %.4f" % round(np.var(mae), 4)
)