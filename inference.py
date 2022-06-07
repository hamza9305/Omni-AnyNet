import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from dataloader import Dataloader_anynet
from dataloader import DataProcessing_AnyNet as DP
import utils.logger as logger
from utils.conv_maps import *
import torch.backends.cudnn as cudnn
import re
import models.anynet
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import map_processing.map_proc.proj_models as pm

import numpy as np
import tifffile
import random

import time

parser = argparse.ArgumentParser(description='Anynet fintune on KITTI')
parser.add_argument('--maxdisp', type=int, default=176,  help='maxium disparity')
parser.add_argument('--baseline', type=float, default=None, help='baseline [same unit as desired depth map] to convert outputs to depth maps', required=True)
parser.add_argument('--fov_deg', type=float, default=None, help='field of fiew [°] to convert outputs to depth maps', required=True)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 2, 2])
parser.add_argument('--datapath', default='Dataset/theostereo_1024', help='datapath')
parser.add_argument('--dump_disp_index', action='store_true', help='dumps disparity index maps')
parser.add_argument('--dump_disp', action='store_true', help='dumps disparity maps [radians]')
parser.add_argument('--dump_euc_dist', action='store_true', help='dumps Euclidean distance map [unit as baseline]')
parser.add_argument('--dump_z_depth', action='store_true', help='dumps z depth map [unit as baseline]')
parser.add_argument('--test_bsize', type=int, default=1, help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='inference_results', help='the path of saving checkpoints and log')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
parser.add_argument('--start_epoch_for_spn', type=int, default=121)
parser.add_argument('--stop_after', type=int, help="stops after N batches (default: take all batches)")
parser.add_argument('--params', type=str, default='checkpoint/sceneflow/sceneflow.tar', help='checkpoint / model path')
parser.add_argument('--split_file', type=str, default=None)
parser.add_argument('--inference_set', type=str, default=None, \
        help="name of the inference set (default: 'valid' if in training mode or 'test' set if evaluating once)")
parser.add_argument('--height', default=1024, help='image height')
parser.add_argument('--width', default=1024, help='image width')
parser.add_argument('--mask_lut_dir', default='masks_and_luts/', help='directory containing mask') # invalid masks for U-Net resolutions
parser.add_argument('--no_steps_stage1', type=int, default=12, help='depth of cost volume stage 1')
parser.add_argument('--no_steps_stage2', type=int, default=25, help='depth of cost volume stage 2')
parser.add_argument('--no_steps_stage3', type=int, default=51, help='depth of cost volume stage 3')
parser.add_argument('--max_disp_rad', type=float, default=0.31)
parser.add_argument('--mask_new', default='masks_and_luts/mask_full_res.pt') # invalid mask full resolution

args = parser.parse_args()
args.train_bsize = args.test_bsize # for compatibility


def save_tiff(map, path):
    map = map.detach().cpu().numpy()
    map = map.astype(np.float32)
    with tifffile.TiffWriter(path) as tiff:
        tiff.save(map, photometric='MINISBLACK', planarconfig='CONTIG', bitspersample=32, compression="ZSTD")

def assig_mask_and_lut(list):
    regex = r'mask'

    for item in list:
        basname = os.path.basename(item)

        if re.match(regex, basname):
            mask = torch.load(item,map_location="cuda:0")
        else:
            lut = torch.load(item,map_location="cuda:0")

    return lut, mask


def dump_batch_to_disk(data_type, estim, save_path, fid, set_name, stage=3):
    """
    Dump several results of the first sample of a batch to disk

    Parameters
    ----------
    data_type : str
        type of output map, e.g. disp, disp_index or z_depth, ...
    estim : list
        list of estimation tensors (size of list= #stages)
        Each esimation has the shape (B, H, W)
    gt : torch.Tensor
        the ground truth maps of shape (B, H, W)
    valid_masks : list
        list of mask indicating valid map entries (size of list= #stages)
        Each mask has the shape (B, H, W)
    save_path : str
        path to the parent directory in which the results should be saved
    epoch : int
        current epoch
    batch_idx : int
        index of the current batch
    fid : list
        list of file ids (size of list is batch size)
    set_name : str
        name of the set to which the sample belongs ("train", "test", "valid", ...)

    """

    outdir = f"{save_path}/output_results/{set_name}/"
    os.makedirs(outdir, exist_ok=True)
    assert type(estim) is torch.Tensor
    for i in range(estim.shape[0]):
        prefix = outdir + f"fid_{fid[i].item():07d}_"
        path = prefix + f"{data_type}_stage_{stage}.tiff"
        save_tiff(estim[i], path)

def main():

    global args

    torch.backends.cudnn.deterministic = True
    seed = 20220414
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    log = logger.setup_logger(args.save_path + '/inference.log')

    if args.inference_set is None:
        if args.evaluate:
            args.inference_set = 'test'
        else:
            args.inference_set = 'valid'
    
    _, _, _, test_left_img, test_right_img, test_left_disp = \
            Dataloader_anynet.dataloader(args.datapath, log, None, testing_set=args.inference_set)

    dataset = DP.myImageFloder(test_left_img, test_right_img, test_left_disp, False, with_gt=False)
    args.total_num_samples = len(dataset)
    TestImgLoader = torch.utils.data.DataLoader(dataset, 
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    dict = {"stage0": [], "stage1": [], "stage2": []}

    for path, subdirs, files in os.walk(args.mask_lut_dir):

        for name in files:
            m = re.search(".*stage(\d).*", name)
            if m is None:
                continue
            file = os.path.join(path, name)
            stage = m.group(1)
            stage = 'stage' + str(stage)
            dict[stage].append(file)

    lut1, mask1 = assig_mask_and_lut(dict['stage0'])
    lut2, mask2 = assig_mask_and_lut(dict['stage1'])
    lut3, mask3 = assig_mask_and_lut(dict['stage2'])
    stage1 = [lut1,mask1, args.no_steps_stage1]
    stage2 = [lut2,mask2, args.no_steps_stage2]
    stage3 = [lut3,mask3, args.no_steps_stage3]
    luts_masks = [stage1, stage2, stage3]

    mask_new = torch.load(args.mask_new)

    model = models.anynet.AnyNet(args, luts_masks, mask_new, device='cuda')
    model = nn.DataParallel(model).cuda()

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.params:
        if os.path.isfile(args.params):
            checkpoint = torch.load(args.params)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            log.info("=> loaded pretrained model '{}'"
                     .format(args.params))
        else:
            log.info("=> no pretrained model found at '{}'".format(args.params))
            log.info("=> Will start from scratch.")


    cudnn.benchmark = False
    start_full_time = time.time()
    
    inference(TestImgLoader, model, log, set_name=args.inference_set)

def inference(dataloader, model, log, set_name="valid"):
    global args

    stages = 3 + args.with_spn
    length_loader = len(dataloader)

    no_steps = (args.maxdisplist[0] - 1) * 16
    step_size = (args.max_disp_rad) / no_steps

    fov_rad = args.fov_deg * np.pi / 180.0
    rays = None # to be initialized by first batch

    model.eval()
    
    tic = time.time()
    for batch_idx, (imgL, imgR, fid) in enumerate(dataloader):
        if args.stop_after is not None and batch_idx >= args.stop_after:
            log.debug(f'stop after {args.stop_after} batches (batch idx is {batch_idx})')
            break

        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        
        if rays is None:
            height, width = imgL.shape[2:4]
            rays = pm.get_rays_equidist((height, width), fov_rad, fov_rad) # only square images supported currently
            rays = torch.from_numpy(rays).cuda()

        with torch.no_grad():

            outputs = model(imgL, imgR)

            outputs = [torch.squeeze(output, 1) for output in outputs]

            num_out = 4 if args.with_spn else 3
            outputs_euc_dist_map = []
            outputs_z_depth_map = []
            
            for x in range(stages):
                output = torch.squeeze(outputs[x], 1)
                cur_disp = output * step_size
                output_euc_dist_map, output_z_depth_map = disp_to_dist_and_depth(cur_disp, rays, fov=fov_rad, baseline=args.baseline, calc_z_depth=args.dump_z_depth)
                if x == stages - 1:
                    if args.dump_disp:
                        dump_batch_to_disk("disp", cur_disp, args.save_path, fid, set_name, stage=num_out-1)
                    if args.dump_disp_index:
                        dump_batch_to_disk("disp_index", outputs[x], args.save_path, fid, set_name, stage=x)
                    if args.dump_euc_dist:
                        dump_batch_to_disk("euc_dist", output_euc_dist_map, args.save_path, fid, set_name, stage=x)
                    if args.dump_z_depth:
                        dump_batch_to_disk("z_depth", output_z_depth_map, args.save_path, fid, set_name, stage=x)

        # log.info(f'batches processed [{batch_idx + 1}/{length_loader}]')
    toc = time.time()
    elapsed = toc - tic
    time_per_sample = elapsed / args.total_num_samples
    throughput = 1 / time_per_sample
    log.info(f'took: {elapsed:.02f} s for {args.total_num_samples} image pairs')
    log.info(f'time per sample: {time_per_sample} s')
    log.info(f'throughput: {throughput}')


if __name__ == '__main__':
    main()
