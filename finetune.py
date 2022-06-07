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

parser = argparse.ArgumentParser(description='Anynet fintune on KITTI')
parser.add_argument('--maxdisp', type=int, default=176,  help='maxium disparity')
parser.add_argument('--beta', type=float, default=1.0,  help='smooth l1 loss beta value')
parser.add_argument('--baseline', type=float, default=None, help='baseline [same unit as desired depth map] to convert outputs to depth maps', required=True)
parser.add_argument('--fov_deg', type=float, default=None, help='field of fiew [°] to convert outputs to depth maps', required=True)
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 2, 2])
parser.add_argument('--datatype', default='2015',help='datapath')
parser.add_argument('--datapath', default='Dataset/theostereo_1024', help='datapath')
parser.add_argument('--dump_results', action='store_true', help='dumps outputs for each plot_int iterations')
parser.add_argument('--dump_elem_in_batch', type=int, default=0, help='if dump_results activated, plot the elem in the batch with this id [-1 means all]')
parser.add_argument('--dump_error_maps', action='store_true', help='dumps all error maps')
parser.add_argument('--dump_only_fid', type=int, default=-1, help='if activated, dumps only files associated to a certain fid [ignores dump_elem_in_batch]')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=1, help='batch size for training (default: 6)')
parser.add_argument('--test_bsize', type=int, default=1, help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='train_results', help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default='train_results', help='resume path')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--print_freq', type=int, default=1, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
parser.add_argument('--start_epoch_for_spn', type=int, default=121)
parser.add_argument('--stop_after', type=int, help="stops after N batches (default: take all batches)")
parser.add_argument('--pretrained', type=str, default='checkpoint/sceneflow/sceneflow.tar', help='pretrained model path')
parser.add_argument('--split_file', type=str, default=None)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--evaluation_set', type=str, default=None, \
        help="name of the evaluation set (default: 'valid' if in training mode or 'test' set if evaluating once)")
parser.add_argument('--eval_graph_name', type=str, default=None, \
        help="name of the evaluation graph on the tensorboard (default: 'eval_<evaluation_set>', e.g., 'eval_test')")
parser.add_argument('--height', default=1024, help='image height')
parser.add_argument('--width', default=1024, help='image width')
parser.add_argument('--load_epoch', default=None, type=int, \
        help='loads a checkpoint of an epoch with a certain epoch ID (only available in combination with --resume; \
        default: last available epoch)')
parser.add_argument('--mask_lut_dir', default='masks_and_luts/', help='directory containing mask') # invalid masks for U-Net resolutions
parser.add_argument('--no_steps_stage1', type=int, default=12, help='depth of cost volume stage 1')
parser.add_argument('--no_steps_stage2', type=int, default=25, help='depth of cost volume stage 2')
parser.add_argument('--no_steps_stage3', type=int, default=51, help='depth of cost volume stage 3')
parser.add_argument('--max_disp_rad', type=float, default=0.31)
parser.add_argument('--mask_new', default='masks_and_luts/mask_full_res.pt') # invalid mask full resolution
parser.add_argument('--cosanneal', action='store_true', help="activates torch cosine annealing scheduler")
parser.add_argument('--with_tensorboard', action='store_true', help="store tensorboard results")
parser.add_argument('--plot_results', action='store_true', help="plot some intermediate results")
parser.add_argument('--plot_int', type=int, default=1000, help="plot and dump interval [#batches]")


args = parser.parse_args()

if args.datatype == '2015':
    from dataloader import KITTIloader2015 as ls
elif args.datatype == '2012':
    from dataloader import KITTIloader2012 as ls
elif args.datatype == 'other':
    from dataloader import diy_dataset as ls

def save_tiff(map, path):
    map = map.detach().cpu().numpy()
    map = map.astype(np.float32)
    with tifffile.TiffWriter(path) as tiff:
        tiff.save(map, photometric='MINISBLACK', planarconfig='CONTIG', bitspersample=32, compression="ZSTD")


def compare_plot(estimated_disparity, ground_truth_disp, left_title="output", right_title="left_image", fig_title='Vertically stacked subplots', vmin=None, vmax=None):
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(fig_title)

    estimated_disparity = estimated_disparity.clone().to('cpu').detach()
    ground_truth_disp = ground_truth_disp.clone().to('cpu').detach()

    estimated_disparity = torch.squeeze(estimated_disparity)
    ground_truth_disp = torch.squeeze(ground_truth_disp)

    if vmin is not None and vmax is not None:
        axs[0].imshow(estimated_disparity, vmin=vmin, vmax=vmax)
        axs[1].imshow(ground_truth_disp, vmin=vmin, vmax=vmax)
    else:
        axs[0].imshow(estimated_disparity)
        axs[1].imshow(ground_truth_disp)
    
    axs[0].set_title(left_title)
    axs[1].set_title(right_title)

    plt.show()


def assig_mask_and_lut(list):
    regex = r'mask'

    for item in list:
        basname = os.path.basename(item)

        if re.match(regex, basname):
            mask = torch.load(item,map_location="cuda:0")
        else:
            lut = torch.load(item,map_location="cuda:0")

    return lut, mask


def dump_result_to_disk(data_type, estim, gt, valid_masks, save_path, epoch, batch_idx, fid, set_name, dump_elem_in_batch=0, dump_only_fid=-1):
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
    dump_elem_in_batch : int
        dump the elem with this id in all batches [-1 means all]
    dump_only_fid : int
        ignores dump_elem_in_batch and dumps only files associated with a certain fid
    """

    outdir = f"{save_path}/output_results/{set_name}/{epoch:05d}/{batch_idx:05d}/"
    os.makedirs(outdir, exist_ok=True)

    eids = None
    if dump_only_fid >= 0:
        for eid, cur_fid in enumerate(fid):
            if int(cur_fid) == dump_only_fid:
                eids = [eid]
                break
        if eids is None:
            return # fid set but file no file in batch associated with that fid
    elif dump_elem_in_batch >= 0:
        eids = [dump_elem_in_batch]
    else:
        eids = range(len(fid))

    for eid in eids:
        prefix = outdir + f"epoch_{epoch:05d}_fid_{fid[eid].item():07d}_"
        fst_gt = gt[eid]
        for stage in range(len(estim)):
            fst_estim = estim[stage][eid]
            fst_valid_mask = valid_masks[stage][eid]
            path    = prefix + f"{data_type}_stage_{stage}.tiff"
            path_l1 = prefix + f"{data_type}_stage_{stage}_l1_loss.tiff"
            save_tiff(fst_estim, path)
            abs_error_map = torch.abs(fst_gt - fst_estim)
            save_tiff(abs_error_map, path_l1)

        path_gt = prefix + f"{data_type}_groundtruth.tiff"
        save_tiff(fst_gt, path_gt)

def main():

    global args

    torch.backends.cudnn.deterministic = True
    seed = 20220414
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    log = logger.setup_logger(args.save_path + '/training_new_with_TB.log')

    if args.evaluation_set is None:
        if args.evaluate:
            args.evaluation_set = 'test'
        else:
            args.evaluation_set = 'valid'

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = \
            Dataloader_anynet.dataloader(args.datapath, log, None, testing_set=args.evaluation_set)

    TrainImgLoader = torch.utils.data.DataLoader(
        DP.myImageFloder(train_left_img, train_right_img, train_left_disp, True),
        batch_size=args.train_bsize, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DP.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    if args.cosanneal:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None



    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            log.info("=> loaded pretrained model '{}'"
                     .format(args.pretrained))
        else:
            log.info("=> no pretrained model found at '{}'".format(args.pretrained))
            log.info("=> Will start from scratch.")


    if not os.path.isdir(args.resume):
        os.makedirs(args.resume)

    checkpoints = os.listdir(args.resume + '/')
    # print(checkpoints)

    if len(checkpoints) == 0:
        args.resume = False

    args.start_epoch = 0
    if args.resume:

        sorted_checkpoints = sorted(checkpoints)
        if args.load_epoch is not None:
            tarball = f'epoch_{args.load_epoch:05d}_checkpoint.tar'
            if tarball not in sorted_checkpoints:
                log.error("No such epoch {args.load_epoch:05d} to resume")
                exit(1)
            args.resume = args.resume + '/' + tarball
        else:
            args.resume = args.resume + '/' + sorted_checkpoints[-1]


        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
            if args.cosanneal and 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('Not Resume')
    cudnn.benchmark = True
    start_full_time = time.time()
    writer = SummaryWriter(args.save_path + "/tensorboard")
    if args.evaluate:
        epoch = max(0, args.start_epoch - 1)
        test_losses, bad_1_test, bad_2_test, bad_4_test, test_D1s, mae_test, rel_mae_test, rmse_test, mae_euc_dist_test, \
                rel_mae_euc_dist_test, rmse_euc_dist_test, mae_z_depth_test, rel_mae_z_depth_test, rmse_z_depth_test = \
                test(TestImgLoader, model, log, dump_results=args.dump_results, epoch=epoch, set_name=args.evaluation_set, \
                dump_error_maps=args.dump_error_maps)

        if args.with_tensorboard and args.start_epoch -1 >= 0:
            # belongs to tensorboard of current transfer learning
            graph_name = f'eval_{args.evaluation_set}' if args.eval_graph_name is None else args.eval_graph_name
            stages = 3 + args.with_spn
            for i in range(stages):
                writer.add_scalars(f'loss/stage' + str(i), {
                    graph_name: test_losses[i].avg,
                }, epoch)
                writer.add_scalars(f'3-pixel/stage' + str(i), {
                    graph_name: test_D1s[i].avg,
                }, epoch)
                writer.add_scalars(f'bad_1/stage' + str(i), {
                    graph_name: bad_1_test[i].avg,
                }, epoch)
                writer.add_scalars(f'bad_2/stage' + str(i), {
                    graph_name: bad_2_test[i].avg,
                }, epoch)
                writer.add_scalars(f'bad_4/stage' + str(i), {
                    graph_name: bad_4_test[i].avg,
                }, epoch)
                writer.add_scalars(f'mae/stage' + str(i), {
                    graph_name: mae_test[i].avg,
                }, epoch)
                writer.add_scalars(f'rel_mae/stage' + str(i), {
                    graph_name: rel_mae_test[i].avg,
                }, epoch)
                writer.add_scalars(f'rmse/stage' + str(i), {
                    graph_name: rmse_test[i].avg,
                }, epoch)
                writer.add_scalars(f'mae_euc_dist/stage' + str(i), {
                    graph_name: mae_euc_dist_test[i].avg,
                }, epoch)
                writer.add_scalars(f'rel_mae_euc_dist/stage' + str(i), {
                    graph_name: rel_mae_euc_dist_test[i].avg,
                }, epoch)
                writer.add_scalars(f'rmse_euc_dist/stage' + str(i), {
                    graph_name: rmse_euc_dist_test[i].avg,
                }, epoch)
                writer.add_scalars(f'mae_z_depth/stage' + str(i), {
                    graph_name: mae_z_depth_test[i].avg,
                }, epoch)
                writer.add_scalars(f'rel_mae_z_depth/stage' + str(i), {
                    graph_name: rel_mae_z_depth_test[i].avg,
                }, epoch)
                writer.add_scalars(f'rmse_z_depth/stage' + str(i), {
                    graph_name: rmse_z_depth_test[i].avg,
                }, epoch)
                print("write scalar for stage", i)
                writer.flush()

        return

    for epoch in range(args.start_epoch, args.epochs):
        log.info('This is {}-th epoch'.format(epoch))

        if not args.cosanneal:
            adjust_learning_rate(optimizer, epoch)


        train_losses, train_D1s = train(TrainImgLoader, model, optimizer, log, epoch)

        if epoch % 1 ==0:
            cur_learning_rate = optimizer.param_groups[0]['lr'] * 1e4
            writer.add_scalar('learning rate x 0.0001', cur_learning_rate, epoch)

            test_losses, bad_1_test, bad_2_test, bad_4_test, test_D1s, \
                    mae_test, rel_mae_test, rmse_test, \
                    mae_euc_dist_test, rel_mae_euc_dist_test, rmse_euc_dist_test, \
                    mae_z_depth_test, rel_mae_z_depth_test, rmse_z_depth_test = \
                test(TestImgLoader, model, log, dump_results=args.dump_results, epoch=epoch, set_name=args.evaluation_set)

            losses_train_after_epoch, bad_1_train_after_epoch, bad_2_train_after_epoch, bad_4_train_after_epoch, D1s_delta_3_train_after_epoch, \
                    mae_train_after_epoch, rel_mae_train_after_epoch, rmse_train_after_epoch, \
                    mae_euc_dist_train_after_epoch, rel_mae_euc_dist_train_after_epoch, rmse_euc_dist_train_after_epoch, \
                    mae_z_depth_train_after_epoch, rel_mae_z_depth_train_after_epoch, rmse_z_depth_train_after_epoch = \
                    test(TrainImgLoader, model, log, epoch=epoch, set_name="train")

            stages = 3 + args.with_spn
            if args.with_tensorboard:
                for i in range(stages):
                    writer.add_scalars(f'loss/stage' + str(i), {
                        'training': train_losses[i].avg,
                        'validation': test_losses[i].avg,
                    }, epoch)
                    writer.add_scalars(f'3-pixel/stage' + str(i), {
                        'training': train_D1s[i].avg,
                        'validation': test_D1s[i].avg,
                    }, epoch)
                    writer.add_scalars(f'bad_1/stage' + str(i), {
                        'training': bad_1_train_after_epoch[i].avg,
                        'validation': bad_1_test[i].avg,
                    }, epoch)
                    writer.add_scalars(f'bad_2/stage' + str(i), {
                        'training': bad_2_train_after_epoch[i].avg,
                        'validation': bad_2_test[i].avg,
                    }, epoch)
                    writer.add_scalars(f'bad_4/stage' + str(i), {
                        'training': bad_4_train_after_epoch[i].avg,
                        'validation': bad_4_test[i].avg,
                    }, epoch)
                    writer.add_scalars(f'mae/stage' + str(i), {
                        'training': mae_train_after_epoch[i].avg,
                        'validation': mae_test[i].avg,
                    }, epoch)
                    writer.add_scalars(f'rel_mae/stage' + str(i), {
                        'training': rel_mae_train_after_epoch[i].avg,
                        'validation': rel_mae_test[i].avg,
                    }, epoch)
                    writer.add_scalars(f'rmse/stage' + str(i), {
                        'training': rmse_train_after_epoch[i].avg,
                        'validation': rmse_test[i].avg,
                    }, epoch)
                    writer.add_scalars(f'mae_euc_dist/stage' + str(i), {
                        'training': mae_euc_dist_train_after_epoch[i].avg,
                        'validation': mae_euc_dist_test[i].avg,
                    }, epoch)
                    writer.add_scalars(f'rel_mae_euc_dist/stage' + str(i), {
                        'training': rel_mae_euc_dist_train_after_epoch[i].avg,
                        'validation': rel_mae_euc_dist_test[i].avg,
                    }, epoch)
                    writer.add_scalars(f'rmse_euc_dist/stage' + str(i), {
                        'training': rmse_euc_dist_train_after_epoch[i].avg,
                        'validation': rmse_euc_dist_test[i].avg,
                    }, epoch)
                    writer.add_scalars(f'mae_z_depth/stage' + str(i), {
                        'training': mae_z_depth_train_after_epoch[i].avg,
                        'validation': mae_z_depth_test[i].avg,
                    }, epoch)
                    writer.add_scalars(f'rel_mae_z_depth/stage' + str(i), {
                        'training': rel_mae_z_depth_train_after_epoch[i].avg,
                        'validation': rel_mae_z_depth_test[i].avg,
                    }, epoch)
                    writer.add_scalars(f'rmse_z_depth/stage' + str(i), {
                        'training': rmse_z_depth_train_after_epoch[i].avg,
                        'validation': rmse_z_depth_test[i].avg,
                    }, epoch)
                    
                    print("write scalar for stage", i)
                    writer.flush()

        if args.cosanneal:
            scheduler.step()

        filename = f"/checkpoints/epoch_{epoch:05d}_checkpoint.tar"
        savefilename = args.save_path + filename


        if scheduler is None:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, savefilename)
        else:
            torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
            }, savefilename)

        writer.flush()
    writer.close()


    # test(TestImgLoader, model, log)
    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


def train(dataloader, model, optimizer, log, epoch=0, set_name="train"):
    global args

    stages = 3 + args.with_spn
    losses = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)
    D1s = [AverageMeter() for _ in range(stages)]

    model.train()
    no_steps = (args.maxdisplist[0] - 1) * 16
    step_size = (args.max_disp_rad) / no_steps

    for batch_idx, (imgL, imgR, disp_L, fid) in enumerate(dataloader):
        if args.stop_after is not None and batch_idx >= args.stop_after:
            log.debug(f'stop after {args.stop_after} batches (batch idx is {batch_idx})')
            break

        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        optimizer.zero_grad()
        mask = disp_L > 0
        mask.detach_()

        disp_L_index = disp_L / step_size

        outputs = model(imgL, imgR)

        outputs = [torch.squeeze(output, 1) for output in outputs]
        output_masks = [output == output for output in outputs]
        output_masks = [torch.bitwise_and(output_mask, mask) for output_mask in output_masks]
        
        for x in range(stages):
            output = torch.squeeze(outputs[x], 1)
            D1s[x].update(error_estimating(output, disp_L_index, maxdisp=args.maxdisp).item())

        if (batch_idx + 1) % args.plot_int == 0:
            if args.plot_results:
                compare_plot(outputs[0][0], disp_L_index[0], left_title="network ouput stage 1 disp index", right_title="ground truth", vmin=0, vmax=40)
                compare_plot(outputs[1][0], disp_L_index[0], left_title="network ouput stage 2 disp index", right_title="ground truth", vmin=0, vmax=15)
                compare_plot(outputs[2][0], disp_L_index[0], left_title="network ouput stage 3 disp index", right_title="ground truth", vmin=0, vmax=10)
                if args.with_spn:
                    compare_plot(outputs[3][0], disp_L_index[0], left_title="network ouput stage 4 disp index", right_title="ground truth", vmin=0, vmax=10)

            if args.dump_results:
                dump_result_to_disk("disp_index", outputs, disp_L_index, output_masks, args.save_path, epoch, batch_idx, fid, set_name, dump_elem_in_batch=args.dump_elem_in_batch, dump_only_fid=args.dump_only_fid)

                outputs_disp = [[o * step_size for o in outputs[x]] for x in range(stages)]
                dump_result_to_disk("disp", outputs_disp, disp_L, output_masks, args.save_path, epoch, batch_idx, fid, set_name, dump_elem_in_batch=args.dump_elem_in_batch, dump_only_fid=args.dump_only_fid)
                

        if args.with_spn:
            if epoch >= args.start_epoch_for_spn:
                num_out = len(outputs)
            else:
                num_out = len(outputs) - 1
        else:
            num_out = len(outputs)


        # output_display = torch.squeeze(outputs[2])
        # plt.imshow(output_display[mask].to('cpu').detach())
        # plt.show()

        # error = torch.abs(outputs[2] - disp_L)
        # error = torch.squeeze(error)
        # plt.imshow(error.to('cpu').detach())
        # plt.show()

        # compare_plot(output_masks[0], mask, "output mask stage 0", "gt mask", "Valid mask comparison")

        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][output_masks[x]], disp_L_index[output_masks[x]], reduction='mean', beta=args.beta)
                for x in range(num_out)]

        sum(loss).backward()
        optimizer.step()
        # compare_plot(outputs[2], disp_L)

        for idx in range(num_out):
            losses[idx].update(loss[idx].item())

        if (batch_idx + 1) % args.print_freq == 0:
            info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(num_out)]
            info_str = '\t'.join(info_str)

            log.info('Epoch{} [{}/{}] {}'.format(
                epoch, batch_idx, length_loader, info_str))

    info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
    log.info('Average train loss = ' + info_str)

    return losses, D1s


def test(dataloader, model, log, dump_results=False, epoch=0, set_name="valid", dump_error_maps=False):
    global args

    stages = 3 + args.with_spn
    D1s = [AverageMeter() for _ in range(stages)]
    losses = [AverageMeter() for _ in range(stages)]
    bad_1 = [AverageMeter() for _ in range(stages)]
    bad_2 = [AverageMeter() for _ in range(stages)]
    bad_4 = [AverageMeter() for _ in range(stages)]
    
    mae = [AverageMeter() for _ in range(stages)]
    rel_mae = [AverageMeter() for _ in range(stages)]
    rmse = [AverageMeter() for _ in range(stages)]

    mae_euc_dist = [AverageMeter() for _ in range(stages)]
    rel_mae_euc_dist = [AverageMeter() for _ in range(stages)]
    rmse_euc_dist = [AverageMeter() for _ in range(stages)]
    
    mae_z_depth = [AverageMeter() for _ in range(stages)]
    rel_mae_z_depth = [AverageMeter() for _ in range(stages)]
    rmse_z_depth = [AverageMeter() for _ in range(stages)]
    
    length_loader = len(dataloader)

    no_steps = (args.maxdisplist[0] - 1) * 16
    step_size = (args.max_disp_rad) / no_steps

    fov_rad = args.fov_deg * np.pi / 180.0
    rays = None # to be initialized by first batch

    model.eval()

    for batch_idx, (imgL, imgR, disp_L, fid) in enumerate(dataloader):
        if args.stop_after is not None and batch_idx >= args.stop_after:
            log.debug(f'stop after {args.stop_after} batches (batch idx is {batch_idx})')
            break

        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()
        # disp_L = disp_L[:, ::2, ::2]

        disp_L_index = disp_L / step_size
        
        if rays is None:
            height, width = imgL.shape[2:4]
            rays = pm.get_rays_equidist((height, width), fov_rad, fov_rad) # only square images supported currently
            rays = torch.from_numpy(rays).cuda()

        with torch.no_grad():

            mask = disp_L > 0

            outputs = model(imgL, imgR)

            outputs = [torch.squeeze(output, 1) for output in outputs]
            output_masks = [output == output for output in outputs]
            output_masks = [torch.bitwise_and(output_mask, mask) for output_mask in output_masks]

            if args.with_spn:
                if epoch >= args.start_epoch_for_spn:
                    num_out = len(outputs)
                else:
                    num_out = len(outputs) - 1
            else:
                num_out = len(outputs)
        
            loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][output_masks[x]], disp_L_index[output_masks[x]], reduction='mean', beta=args.beta)
                for x in range(num_out)]
        
            for idx in range(num_out):
                losses[idx].update(loss[idx].item())

            estim_euc_dist_maps = []
            valid_mask_euc_dist_maps = []
            estim_z_depth_maps = []
            valid_mask_z_depth_maps = []

            gt_euc_dist_map, gt_z_depth_map = disp_to_dist_and_depth(disp_L, rays, fov=fov_rad, baseline=args.baseline)

            for x in range(stages):
                output = torch.squeeze(outputs[x], 1)
                errmap, valid_mask = error_map(output, disp_L_index, args.maxdisp)
                valid_abs_errors = errmap[valid_mask]
                rel_valid_abs_errors = valid_abs_errors / disp_L_index[valid_mask]

                output_euc_dist_map, output_z_depth_map = disp_to_dist_and_depth(output * step_size, rays, fov=fov_rad, baseline=args.baseline)
                
                errmap_euc_dist, valid_mask_euc_dist = error_map(output_euc_dist_map, gt_euc_dist_map)
                errmap_z_depth, valid_mask_z_depth = error_map(output_z_depth_map, gt_z_depth_map)

                # important: avoid evaluation of Euc. dist and z-depth if corresp. dispartity is invalid
                valid_mask_euc_dist *= valid_mask
                valid_mask_z_depth *= valid_mask

                output_euc_dist_map[torch.logical_not(valid_mask_euc_dist)] = float('nan')
                output_z_depth_map[torch.logical_not(valid_mask_z_depth)] = float('nan')

                if dump_error_maps and x == 3:
                    abs_errmap_euc_dist_2D = torch.zeros_like(errmap_euc_dist) * float('nan')
                    abs_errmap_euc_dist_2D[valid_mask_euc_dist] = errmap_euc_dist[valid_mask_euc_dist]
                    
                    rel_errmap_euc_dist_2D = torch.zeros_like(errmap_euc_dist) * float('nan')
                    rel_errmap_euc_dist_2D[valid_mask_euc_dist] = \
                            errmap_euc_dist[valid_mask_euc_dist] / gt_euc_dist_map[valid_mask_euc_dist]

                    for elem in range(imgL.shape[0]):
                        outdir = f"{args.save_path}/output_results/{set_name}/error_maps/max_disp_{args.maxdisp}/{epoch:05d}/"
                        os.makedirs(outdir, exist_ok=True)
                        fpath = outdir + f"abs_euc_dist_error_map_fid_{fid[elem].item():07d}.tiff"
                        save_tiff(abs_errmap_euc_dist_2D[elem], fpath)
                        fpath = outdir + f"rel_euc_dist_error_map_fid_{fid[elem].item():07d}.tiff"
                        save_tiff(rel_errmap_euc_dist_2D[elem], fpath)

                errmap_euc_dist = errmap_euc_dist[valid_mask_euc_dist]
                rel_errmap_euc_dist = errmap_euc_dist / gt_euc_dist_map[valid_mask_euc_dist]

                errmap_z_depth = errmap_z_depth[valid_mask_z_depth]
                rel_errmap_z_depth = errmap_z_depth / gt_z_depth_map[valid_mask_z_depth]

                estim_euc_dist_maps += [output_euc_dist_map]
                valid_mask_euc_dist_maps += [valid_mask_euc_dist]
                estim_z_depth_maps += [output_z_depth_map]
                valid_mask_z_depth_maps += [valid_mask_z_depth]

                bad_1[x].update(bad_i(valid_abs_errors, delta=1.).item())
                bad_2[x].update(bad_i(valid_abs_errors, delta=2.).item())
                bad_4[x].update(bad_i(valid_abs_errors, delta=4.).item())
                D1s[x].update(error_estimating(output, disp_L_index, maxdisp=args.maxdisp, errmap=errmap, valid_mask=valid_mask).item())
                
                mae[x].update(torch.mean(valid_abs_errors).item())
                rel_mae[x].update(torch.mean(rel_valid_abs_errors).item())
                rmse[x].update(root_mean_square_error(valid_abs_errors).item())

                mae_euc_dist[x].update(torch.mean(errmap_euc_dist).item())
                rel_mae_euc_dist[x].update(torch.mean(rel_errmap_euc_dist).item())
                rmse_euc_dist[x].update(root_mean_square_error(errmap_euc_dist).item())

                mae_z_depth[x].update(torch.mean(errmap_z_depth).item())
                rel_mae_z_depth[x].update(torch.mean(rel_errmap_z_depth).item())
                rmse_z_depth[x].update(root_mean_square_error(errmap_z_depth).item())
        
            if (batch_idx + 1) % args.plot_int == 0:
                if dump_results:
                    dump_result_to_disk("euc_dist", estim_euc_dist_maps, gt_euc_dist_map, valid_mask_euc_dist_maps, args.save_path, epoch, batch_idx, fid, set_name, dump_elem_in_batch=args.dump_elem_in_batch, dump_only_fid=args.dump_only_fid)
                    dump_result_to_disk("z_depth", estim_z_depth_maps, gt_z_depth_map, valid_mask_z_depth_maps, args.save_path, epoch, batch_idx, fid, set_name, dump_elem_in_batch=args.dump_elem_in_batch, dump_only_fid=args.dump_only_fid)
                    dump_result_to_disk("disp_index", outputs, disp_L_index, output_masks, args.save_path, epoch, batch_idx, fid, set_name, dump_elem_in_batch=args.dump_elem_in_batch, dump_only_fid=args.dump_only_fid)
                    
                    outputs_disp = [[o * step_size for o in outputs[x]] for x in range(stages)]
                    dump_result_to_disk("disp", outputs_disp, disp_L, output_masks, args.save_path, epoch, batch_idx, fid, set_name, dump_elem_in_batch=args.dump_elem_in_batch, dump_only_fid=args.dump_only_fid)

        info_str = '\t'.join(['Stage {} = {:.4f}({:.4f})'.format(x, D1s[x].val, D1s[x].avg) for x in range(stages)])

        log.info('[{}/{}] {}'.format(
            batch_idx, length_loader, info_str))

    info_str = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(stages)])
    log.info('Average test 3-Pixel Error = ' + info_str)

    return losses, bad_1, bad_2, bad_4, D1s, mae, rel_mae, rmse, mae_euc_dist, rel_mae_euc_dist, rmse_euc_dist, mae_z_depth, rel_mae_z_depth, rmse_z_depth

def error_map(disp, gt, max_gt_val=None):
    valid_mask = gt > 0
    if max_gt_val is not None:
        valid_mask = valid_mask * (gt <= max_gt_val)
    valid_mask = torch.logical_and(valid_mask, disp == disp)
    errmap = torch.abs(disp - gt)
    return errmap, valid_mask


def error_estimating(disp, gt, maxdisp=176, delta=3., epsilon=0.05, errmap=None, valid_mask=None):
    if errmap is None or valid_mask is None:
        errmap, valid_mask = error_map(disp, gt, maxdisp)
    err3 = ((errmap[valid_mask] > delta) & (errmap[valid_mask] / gt[valid_mask] > epsilon)).sum()
    return err3.float() / valid_mask.sum().float()

def bad_i(valid_abs_errors, delta=3.):
    num_err = (valid_abs_errors > delta).sum()
    numel = valid_abs_errors.numel()
    return num_err / numel

def root_mean_square_error(valid_abs_errors):
    return torch.sqrt(torch.mean(valid_abs_errors**2))

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = args.lr
    elif epoch <= 400:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
