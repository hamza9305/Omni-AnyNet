from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodules import post_3dconvs,feature_extraction_conv
import sys
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("..")

def plot_GPU(map, title,  cmap_name="gist_rainbow", rad2deg=False,vmin=None, vmax= None, xlabel = None):
    cmap = plt.get_cmap(cmap_name)
    if rad2deg:
        map = torch.clone(map.detach())
        map *= 180 / np.pi
    map = map.to('cpu').detach()
    map = map.squeeze()
    pos = plt.imshow(map, cmap)
    plt.title(title)
    plt.colorbar(pos)
    plt.clim(vmin,vmax)
    plt.xlabel(xlabel)
    plt.show()


class AnyNet(nn.Module):
    def __init__(self, args, luts_masks, mask_new, device):
        super(AnyNet, self).__init__()

        self.init_channels = args.init_channels
        self.maxdisplist = args.maxdisplist
        self.spn_init_channels = args.spn_init_channels
        self.nblocks = args.nblocks
        self.layers_3d = args.layers_3d
        self.channels_3d = args.channels_3d
        self.growth_rate = args.growth_rate
        self.with_spn = args.with_spn
        
        self.max_batch_size = max(args.train_bsize, args.test_bsize)
        # self.luts_masks = luts_masks
        self.luts_masks = []

        f_channels = [8, 4, 2]

        for i in range(len(luts_masks)):
            cur_lut, cur_mask, no_steps = luts_masks[i]
            cur_lut = torch.squeeze(cur_lut, 1)
            height = cur_lut.shape[1]
            width = cur_lut.shape[2]
            assert no_steps == cur_lut.shape[0], f"no_steps: {no_steps}, cur_lut.shape: {cur_lut.shape}"
           
            lut_batch_1 = torch.zeros(size=(1, no_steps, height, width, 3), dtype=cur_lut.dtype, device=device) # shape: [1, 12, H, W, 3]
            lut_batch_1[0, :, :, :, :2] = cur_lut
            lut_batch = lut_batch_1.repeat(self.max_batch_size, 1, 1, 1, 1) # shape: [N, 12, H, W, 3]

            valid_mask = torch.logical_not(cur_mask)
            valid_mask = valid_mask[None, None, :, :, :]
            valid_mask = valid_mask.repeat([self.max_batch_size, f_channels[i], 1, 1, 1])
            self.luts_masks.append([lut_batch, valid_mask, no_steps])


        
        temp = (self.maxdisplist[0]-1)
        self.no_steps = [temp, temp * 2, temp * 4]
        self.max_disp_rad = args.max_disp_rad
        self.step_size = [self.max_disp_rad/self.no_steps[0],self.max_disp_rad/self.no_steps[1], self.max_disp_rad/self.no_steps[2]]
        self.mask_new = mask_new



        if self.with_spn:
            try:
                # from .spn.modules.gaterecurrent2dnoind import GateRecurrent2dnoind
                from .spn_t1.modules.gaterecurrent2dnoind import GateRecurrent2dnoind
            except:
                print('Cannot load spn model')
                sys.exit()
            self.spn_layer = GateRecurrent2dnoind(True,False)
            spnC = self.spn_init_channels
            self.refine_spn = [nn.Sequential(
                nn.Conv2d(3, spnC*2, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(spnC*2, spnC*2, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(spnC*2, spnC*2, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(spnC*2, spnC*3, 3, 1, 1, bias=False),
            )]
            self.refine_spn += [nn.Conv2d(1,spnC,3,1,1,bias=False)]
            self.refine_spn += [nn.Conv2d(spnC,1,3,1,1,bias=False)]
            self.refine_spn = nn.ModuleList(self.refine_spn)
        else:
            self.refine_spn = None

        self.feature_extraction = feature_extraction_conv(self.init_channels,
                                      self.nblocks)

        self.volume_postprocess = []

        for i in range(3):
            net3d = post_3dconvs(self.layers_3d, self.channels_3d*self.growth_rate[i])
            self.volume_postprocess.append(net3d)
        self.volume_postprocess = nn.ModuleList(self.volume_postprocess)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _build_volume_2d(self, feat_l, feat_r, luts_masks, device='cuda'):

        lut_batch = luts_masks[0]
        valid_mask = luts_masks[1]

        no_steps = luts_masks[2] # 12
        height = feat_l.size(2)
        width = feat_l.size(3)
        
        batch_size = feat_l.size(0)
        f_channels = feat_l.size(1)  # shape : [1, 8, 64, 64]

        # feat_l has shape [N, 8, H, W]                
        feat_l_5D = feat_l[:, :, None, :, :] # [N, 8, 1, H, W]
        feat_r_5D = feat_r[:, :, None, :, :] # [N, 8, 1, H, W]

        feat_l_5D_rep = feat_l_5D.repeat([1, 1, no_steps, 1, 1])

        lut_batch = lut_batch[:batch_size]
        valid_mask = valid_mask[:batch_size]
        
        output_image_feat = torch.nn.functional.grid_sample(feat_r_5D, lut_batch, align_corners=False) # [N, 8, 12, H, W]
        
        cost_disp_feat = torch.ones(size=(batch_size, f_channels, no_steps, height, width), dtype=torch.float32, device=device)
        cost_disp_feat[:] = float('inf')

        cost_disp_feat[valid_mask] = torch.abs(output_image_feat[valid_mask] - feat_l_5D_rep[valid_mask]) #  [N, 8, 12, H, W]
        
        cost_volume = torch.sum(cost_disp_feat, dim=1) # [N, 12, H, W] # L1 distance between features

        return cost_volume

    def _build_volume_2d3(self, feat_l, feat_r, disp, maxdisp, luts_masks, scale, device='cuda'):
        lut_batch = luts_masks[0]
        valid_mask = luts_masks[1]
        no_steps = luts_masks[2]
        f_channels = feat_l.size(1)
        height = feat_l.size(2)
        width = feat_l.size(3)
        lower = -maxdisp
        upper = maxdisp
        tensor_length = upper-lower + 1
        no_channels = feat_l.size(1)
        batch_size = feat_l.size(0)

        # print(disp.size())
        # disp = torch.squeeze(disp)
        # print(disp.size())
        # exit()

        # disp = torch.squeeze(disp, dim = 0)
        disp_debug = disp[0, 0, :, :]
        # plot_GPU(disp_debug, title = f'DISPARITY FROM STAGE {scale - 1}')
        disp = disp.repeat([1,5,1,1])

        summand = torch.arange(lower, upper + 1,dtype=disp.dtype).to(device="cuda")
        summand = summand[:,None, None]

        index_guesses = disp + summand
        index_guesses = torch.round(index_guesses).long()
        invalid_index_guesses = index_guesses < 0
        invalid_index_guesses = torch.logical_or(invalid_index_guesses, index_guesses != index_guesses)
        invalid_index_guesses = torch.logical_or(invalid_index_guesses, index_guesses >= no_steps)
        # plot_GPU(invalid_index_guesses[0][0],title= 'Invalid index guess')
        # plot_GPU(invalid_index_guesses[0][1], title='Invalid index guess')
        # plot_GPU(invalid_index_guesses[0][2], title='Invalid index guess')
        # plot_GPU(invalid_index_guesses[0][3], title='Invalid index guess')
        # plot_GPU(invalid_index_guesses[0][4], title='Invalid index guess')

        index_guesses[invalid_index_guesses] = 0
        index_guesses_lut = torch.unsqueeze(index_guesses,4)
        index_guesses_lut = index_guesses_lut.repeat(1,1,1,1,3)

        index_guesses_mask = torch.unsqueeze(index_guesses, 1)
        index_guesses_mask = index_guesses_mask.repeat(1,no_channels,1,1,1)
        
        ## New implementation below (batch size)
        invalid_index_guesses_mask = invalid_index_guesses[:, None].repeat(1, f_channels, 1, 1, 1)

        # feat_l has shape [N, 8, H, W]                
        feat_l_5D = feat_l[:, :, None, :, :] # [N, 8, 1, H, W]
        feat_r_5D = feat_r[:, :, None, :, :] # [N, 8, 1, H, W]

        feat_l_5D_rep = feat_l_5D.repeat([1, 1, tensor_length, 1, 1])

        lut_batch = lut_batch[:batch_size]
        valid_mask = valid_mask[:batch_size]


        lut_small = torch.gather(lut_batch, 1, index_guesses_lut)
       
        mask_small = torch.gather(valid_mask, 2, index_guesses_mask)
        # mask_small = mask_small[:, None].repeat(1, f_channels, 1, 1, 1)
        
        mask_small[invalid_index_guesses_mask] = False

        
        output_image_feat = torch.nn.functional.grid_sample(feat_r_5D, lut_small, align_corners=False) # [N, 4, 5, H, W]
        
        cost_disp_feat = torch.ones(size=(batch_size, f_channels, tensor_length, height, width), dtype=torch.float32, device=device)
        cost_disp_feat[:] = float('inf')

        cost_disp_feat[mask_small] = torch.abs(output_image_feat[mask_small] - feat_l_5D_rep[mask_small]) #  [N, 4, 5, H, W]
        
        cost_volume = torch.sum(cost_disp_feat, dim=1) # [N, 5, H, W] # L1 distance between features

        #### Old implementation with batch loop below

        # plt.imshow(feat_r[0][0].to('cpu').detach())
        # plt.show()
        #cost_volume = torch.ones(batch_size, tensor_length, height, width, device="cuda") * torch.inf

        #for index in range(batch_size):
        #    lut_small = torch.gather(lut_batch[index], 0, index_guesses_lut[index])
        #    print(f"{lut_small.shape=}")
        #    mask_small = torch.gather(valid_mask[index, 0], 0, index_guesses[index])
        #    mask_small[invalid_index_guesses[index]] = False
        #    mask_small = mask_small[:, None].repeat(1, f_channels, 1, 1)
        #    print(f"{mask_small.shape=}")
        #   
        #    # repeat the features as much as steps we have (5 for [-2, -1, 0, 1, 2])
        #    print(f"{feat_l.shape=}")

        #    features_r = feat_r[index][:, None].repeat(1, tensor_length, 1, 1)
        #    features_l = feat_l[index][:, None].repeat(1, tensor_length, 1, 1)
        #    print(f"{features_l.shape=}")

        #    output_image_feat = torch.nn.functional.grid_sample(features_r, lut_small, align_corners=False)
        #    cost_disp_feat = torch.ones(size=(f_channels, no_steps, height, width), dtype=torch.float32, device=device) * float('inf') # [4, 5, H, W]
        #    cost_disp_feat[:] = float('inf')

        #    # cost_disp_feat[mask_small] = torch.abs(output_image_feat[mask_small] - feat_l_5D_rep[mask_small]) #  [4, 5, H, W]
        #    cost_disp_feat[mask_small] = torch.abs(output_image_feat[mask_small] - features_l[mask_small]) #  [4, 5, H, W]
        #    cur_cost_volume = torch.sum(cost_disp_feat, dim=0) # [5, H, W] # L1 distance between features
        #    cost_volume[index, :, :, :] = cur_cost_volume

        return cost_volume


    def forward(self, left, right):

        # left = left[:, :, ::2, ::2]
        # right = right[:, :, ::2, ::2]

        img_size = left.size()

        feats_l = self.feature_extraction(left)
        feats_r = self.feature_extraction(right)
        batch_size = feats_l[0].size(0)

        # plt.imshow(feats_l[0][0][0].to('cpu').detach())
        # plt.show()
        # plt.imshow(feats_l[0][0][1].to('cpu').detach())
        # plt.show()
        # exit()
        pred_ind = []
        pred_rad = []
        for scale in range(len(feats_l)):
            if scale > 0:
                # print(pred[scale-1].size())
                scaling_factor = feats_l[scale].size(2) / img_size[2]
                wflow = F.interpolate(pred_ind[scale - 1], (feats_l[scale].size(2), feats_l[scale].size(3)), mode='bilinear', align_corners=False) * feats_l[scale].size(2) / img_size[2]
                # wflow *= 2
                # print(wflow.size())
                # exit()
                # wflow_plot = torch.squeeze(wflow)
                # plt.imshow(wflow_plot.to('cpu').detach())
                # plt.show()
                # exit()

                cost = self._build_volume_2d3(feats_l[scale], feats_r[scale], wflow,self.maxdisplist[scale], self.luts_masks[scale], scale)
            else:
                cost = self._build_volume_2d(feats_l[scale], feats_r[scale], self.luts_masks[scale])

            # allow invalid regions to influence valid regions
            # otherwise -> infs or NaNs everywhere

            invalid_cost = cost != cost
            invalid_cost = torch.logical_or(invalid_cost, cost == torch.inf)
            invalid_cost = torch.logical_or(invalid_cost, cost == -torch.inf)
            cost[invalid_cost] = 0
            
            cost = torch.unsqueeze(cost, 1)
            cost = self.volume_postprocess[scale](cost)
            cost = cost.squeeze(1)

            cost[invalid_cost] = torch.inf
            
            mask_new_change = self.mask_new
            mask_new_change = torch.unsqueeze(mask_new_change, dim=0)
            mask_new_change = mask_new_change.repeat([batch_size, 1, 1, 1])

            if scale == 0:
                pred_low_res = disparityregression2(0, self.maxdisplist[0])(F.softmax(-cost, dim=1))
                pred_low_res_rad = pred_low_res * self.step_size[scale]
                pred_low_res_rad = F.interpolate(pred_low_res_rad, (img_size[2], img_size[3]), mode='bilinear', align_corners=False)
                pred_low_res_rad[mask_new_change] = 0
                pred_rad.append(pred_low_res_rad)

                pred_low_res = pred_low_res * img_size[2] / pred_low_res.size(2)
                disp_up = F.interpolate(pred_low_res, (img_size[2], img_size[3]), mode='bilinear', align_corners=False)
                disp_up[mask_new_change] = 0
                # plot_GPU(disp_up[0], "high resolution prediction (first stage)")
                pred_ind.append(disp_up)

            else:
                lower = -self.maxdisplist[scale]
                upper = self.maxdisplist[scale]
                pred_low_res = disparityregression2(lower, upper + 1, stride=1)(F.softmax(-cost, dim=1))
                # pred_low_res_rad = pred_low_res * self.step_size[scale]
                # pred_low_res_rad = F.interpolate(pred_low_res_rad, (img_size[2], img_size[3]), mode='bilinear', align_corners=False)
                # pred_low_res_rad[mask_new_change] = 0
                # pred_rad.append(pred_low_res_rad + pred_rad[scale -1])
                # pred_rad.append(pred_low_res_rad)

                # plot_GPU(pred_low_res, "low resolution prediction residuum")
                pred_low_res = pred_low_res * img_size[2] / pred_low_res.size(2)
                disp_up = F.interpolate(pred_low_res, (img_size[2], img_size[3]), mode='bilinear', align_corners=False)
                disp_up[mask_new_change] = 0
                final_disparity = disp_up + pred_ind[scale - 1]
                # plot_GPU(final_disparity, "high resolution prediction")
                pred_ind.append(final_disparity)

            # print(disp_up)
            # print(disp_up * 2)


            # disp_up = torch.squeeze(disp_up)
            # plt.imshow(disp_up.to('cpu').detach())
            # plt.show()

        if self.refine_spn:
            spn_out = self.refine_spn[0](nn.functional.interpolate(left, (img_size[2]//4, img_size[3]//4), mode='bilinear', align_corners=False))
            G1, G2, G3 = spn_out[:,:self.spn_init_channels,:,:], spn_out[:,self.spn_init_channels:self.spn_init_channels*2,:,:], spn_out[:,self.spn_init_channels*2:,:,:]
            sum_abs = G1.abs() + G2.abs() + G3.abs()
            G1 = torch.div(G1, sum_abs + 1e-8)
            G2 = torch.div(G2, sum_abs + 1e-8)
            G3 = torch.div(G3, sum_abs + 1e-8)
            pred_flow = nn.functional.interpolate(pred_ind[-1], (img_size[2]//4, img_size[3]//4), mode='bilinear', align_corners=False)
            inv_pred_flow = pred_flow != pred_flow
            pred_flow[inv_pred_flow] = 0
            # plot_GPU(pred_flow[0, 0], "pred flow")
            refine_flow = self.spn_layer(self.refine_spn[1](pred_flow), G1, G2, G3)
            refine_flow = self.refine_spn[2](refine_flow)
            refine_flow[inv_pred_flow] = float('nan')
            pred_ind.append(nn.functional.interpolate(refine_flow, (img_size[2] , img_size[3]), mode='bilinear', align_corners=False))

        return pred_ind

class disparityregression2(nn.Module):
    def __init__(self, start, end, stride=1):
        super(disparityregression2, self).__init__()
        self.disp = torch.arange(start*stride, end*stride, stride, device='cuda', requires_grad=False).view(1, -1, 1, 1).float()

    def forward(self, x):
        # repeating tensor: [1, #steps, 1, 1] -> [N, #steps, H, W]
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1, keepdim=True)
        return out
