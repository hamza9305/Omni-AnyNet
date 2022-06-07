import matplotlib.pyplot as plt
from tifffile import tifffile
import numpy as np
import glob
import re
import os

import argparse

parser = argparse.ArgumentParser(description='Tiff Viewer')
parser.add_argument('--indir',help='input directory to display plots')


def compare_plot(list_images):
    fig, axs = plt.subplots(4, 2)
    fig.suptitle('Vertically stacked subplots')

    gt = tifffile.imread(list_images[0]).squeeze()
    stage_0 = tifffile.imread(list_images[1]).squeeze()
    stage_1 = tifffile.imread(list_images[2]).squeeze()
    stage_2 = tifffile.imread(list_images[3]).squeeze()
    stage_3 = tifffile.imread(list_images[4]).squeeze()

    axs[0, 0].imshow(gt)
    axs[0, 0].set_title('Ground Truth')

    axs[0, 1].imshow(stage_0)
    axs[0, 1].set_title('Stage 0')

    axs[1, 0].imshow(gt)
    axs[1, 0].set_title('Ground Truth')

    axs[1, 1].imshow(stage_1)
    axs[1, 1].set_title('Stage 1')

    axs[2, 0].imshow(gt)
    axs[2, 0].set_title('Ground Truth')

    axs[2, 1].imshow(stage_2)
    axs[2, 1].set_title('Stage 2')

    axs[3, 0].imshow(gt)
    axs[3, 0].set_title('Ground Truth')

    axs[3, 1].imshow(stage_3)
    axs[3, 1].set_title('Stage 3')

    plt.show()


def plot_l1_loss(list_l1_loss):

    fig, axs = plt.subplots(4, 1)
    fig.suptitle('Vertically stacked subplots')

    l1_stage_0 = tifffile.imread(list_l1_loss[0]).squeeze()
    l1_stage_1 = tifffile.imread(list_l1_loss[1]).squeeze()
    l1_stage_2 = tifffile.imread(list_l1_loss[2]).squeeze()
    l1_stage_3 = tifffile.imread(list_l1_loss[3]).squeeze()

    axs[0].imshow(l1_stage_0)
    axs[0].set_title('l1 Loss Stage 0')

    axs[1].imshow(l1_stage_1)
    axs[1].set_title('l1 Loss Stage 1')

    axs[2].imshow(l1_stage_2)
    axs[2].set_title('l1 Loss Stage 2')

    axs[3].imshow(l1_stage_3)
    axs[3].set_title('l1 Loss Stage 3')

    plt.show()

def main():
    args = parser.parse_args()


    list_images = glob.glob(args.indir + "/" '*tiff')

    list_l1_loss = []
    list_estimated_disp = []
    regex = r'\w*l1_loss'
    for item in list_images:

        basname = os.path.basename(item)
        if re.match(regex, basname):
            list_l1_loss.append(item)
        else:
            list_estimated_disp.append(item)

    list_estimated_disp = sorted(list_estimated_disp)
    list_l1_loss = sorted(list_l1_loss)

    compare_plot(list_estimated_disp)
    plot_l1_loss(list_l1_loss)

if __name__ == '__main__':
    main()
