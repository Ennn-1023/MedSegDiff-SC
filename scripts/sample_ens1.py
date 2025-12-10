

import argparse
import os
from ssl import OP_NO_TLSv1
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
from PIL import Image
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.custom_dataset_loader import CustomDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
from torchsummary import summary
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_test = transforms.Compose(tran_list)

        ds = ISICDataset(args, args.data_dir, transform_test, mode = 'Test')
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_test = transforms.Compose(tran_list)

        ds = BRATSDataset3D(args.data_dir,transform_test)
        args.in_ch = 5
    else:
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor()]
        transform_test = transforms.Compose(tran_list)

        ds = CustomDataset(args, args.data_dir, transform_test, mode = 'Test')
        args.in_ch = 4

    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []


    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    for _ in range(len(data)):
        b, m, path = next(data)  #should return an image from the dataloader "data"
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)     #add a noise channel$
        if args.data_name == 'ISIC':
            slice_ID=path[0].split("_")[-1].split('.')[0]
        elif args.data_name == 'BRATS':
            # slice_ID=path[0].split("_")[2] + "_" + path[0].split("_")[4]
            slice_ID=path[0].split("_")[-3] + "_" + path[0].split("slice")[-1].split('.nii')[0]
        else:
            # For other datasets (e.g., PH2, custom datasets)
            slice_ID = os.path.splitext(os.path.basename(path[0]))[0]

        logger.log("sampling...")

        # ✅ Save ground truth mask for comparison
        vutils.save_image(m, fp=os.path.join(args.out_dir, str(slice_ID)+'_gt.jpg'), nrow=1, padding=0)

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []

        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org, cal, cal_out = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                step = args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

            co = cal_out.detach().clone() if isinstance(cal_out, th.Tensor) else th.tensor(cal_out)
            if args.version == 'new':
                enslist.append(sample[:,-1,:,:])
                current_mask = sample[:,-1,:,:]  # Extract mask for individual saving
            else:
                # #正規化讓多版本輸出變亮
                # co_norm = (co - co.min()) / (co.max() - co.min() + 1e-8)
                # enslist.append(co_norm)
                # current_mask = co_norm
                enslist.append(co) 
                current_mask = co
            # 直接儲存機率圖，不做 Min-Max 拉伸
            # 如果模型輸出全黑 (數值很小)，圖片就該是黑的
            vutils.save_image(current_mask, fp=os.path.join(args.out_dir, str(slice_ID)+f'_prob{i}.jpg'), nrow=1, padding=0, normalize=False)

            # 額外儲存二值化後的結果 (Mask)，閾值設為 0.5
            # 這張圖代表模型最終判定哪裡是腫瘤
            binary_mask = (current_mask > 0.3).float()
            vutils.save_image(binary_mask, fp=os.path.join(args.out_dir, str(slice_ID)+f'_mask{i}.jpg'), nrow=1, padding=0, normalize=False)
            
            if i < 3:
                print(f'  Mask {i} - Range: [{current_mask.min():.4f}, {current_mask.max():.4f}] (Should be approx 0 to 1)')
            # --- 修正結束 ---
            # # ✅ 標準化 mask 到 [0, 1] 範圍，確保不是淡白色
            # # 方法: Min-Max 標準化
            # mask_to_save = current_mask.clone()
            # mask_min = mask_to_save.min()
            # mask_max = mask_to_save.max()

            # if mask_max > mask_min:  # 避免除零
            #     mask_to_save = (mask_to_save - mask_min) / (mask_max - mask_min)
            # else:
            #     mask_to_save = mask_to_save * 0  # 全黑

            # # Debug: 打印前 3 個樣本的統計信息
            # if i < 3:
            #     print(f'  Mask {i} - Original range: [{current_mask.min():.4f}, {current_mask.max():.4f}]')
            #     print(f'  Mask {i} - Normalized range: [{mask_to_save.min():.4f}, {mask_to_save.max():.4f}]')
            #     print(f'  Mask {i} - Mean: {mask_to_save.mean():.4f}')

            # # ✅ Always save individual masks for evaluation (normalized to [0,1])
            # vutils.save_image(mask_to_save, fp=os.path.join(args.out_dir, str(slice_ID)+f'_mask{i}.jpg'), nrow=1, padding=0, normalize=False)

            if args.debug:
                # print('sample size is',sample.size())
                # print('org size is',org.size())
                # print('cal size is',cal.size())
                if args.data_name == 'ISIC':
                    # s = th.tensor(sample)[:,-1,:,:].unsqueeze(1).repeat(1, 3, 1, 1)
                    o = th.tensor(org)[:,:-1,:,:]
                    c = th.tensor(cal).repeat(1, 3, 1, 1)
                    # co = co.repeat(1, 3, 1, 1)

                    s = sample[:,-1,:,:]
                    b,h,w = s.size()
                    ss = s.clone()
                    ss = ss.view(s.size(0), -1)
                    ss -= ss.min(1, keepdim=True)[0]
                    ss /= ss.max(1, keepdim=True)[0]
                    ss = ss.view(b, h, w)
                    ss = ss.unsqueeze(1).repeat(1, 3, 1, 1)

                    tup = (ss,o,c)
                elif args.data_name == 'BRATS':
                    s = th.tensor(sample)[:,-1,:,:].unsqueeze(1)
                    m = th.tensor(m.to(device = 'cuda:0'))[:,0,:,:].unsqueeze(1)
                    o1 = th.tensor(org)[:,0,:,:].unsqueeze(1)
                    o2 = th.tensor(org)[:,1,:,:].unsqueeze(1)
                    o3 = th.tensor(org)[:,2,:,:].unsqueeze(1)
                    o4 = th.tensor(org)[:,3,:,:].unsqueeze(1)
                    c = th.tensor(cal)

                    tup = (o1/o1.max(),o2/o2.max(),o3/o3.max(),o4/o4.max(),m,s,c,co)

                compose = th.cat(tup,0)
                vutils.save_image(compose, fp = os.path.join(args.out_dir, str(slice_ID)+'_output'+str(i)+".jpg"), nrow = 1, padding = 10)

        # # Ensemble fusion
        # ensres = staple(th.stack(enslist,dim=0)).squeeze(0)
        # 1. 堆疊所有預測
        stacked_preds = th.stack(enslist, dim=0) 

        # 2. (建議) 先轉成 0/1 Mask 再做 STAPLE，這樣最穩
        # 設定閾值，例如 0.5。如果模型普遍信心低，可暫時降到 0.3 測試
        binary_preds = (stacked_preds > 0.3).float() 

        # 3. 進行集成
        ensres = staple(binary_preds).squeeze(0)

        # # ✅ 標準化 ensemble 結果
        # ensres_min = ensres.min()
        # ensres_max = ensres.max()
        # if ensres_max > ensres_min:
        #     ensres_normalized = (ensres - ensres_min) / (ensres_max - ensres_min)
        # else:
        #     ensres_normalized = ensres * 0

        # print(f'\nEnsemble result - Original range: [{ensres.min():.4f}, {ensres.max():.4f}]')
        # print(f'Ensemble result - Normalized range: [{ensres_normalized.min():.4f}, {ensres_normalized.max():.4f}]')
        # print(f'Ensemble result - Mean: {ensres_normalized.mean():.4f}')

        # vutils.save_image(ensres_normalized, fp = os.path.join(args.out_dir, str(slice_ID)+'_output_ens'+".jpg"), nrow = 1, padding = 10, normalize=False)
        # --- 修正開始 ---
        # STAPLE 輸出通常是機率或 0/1，不需要再做 Min-Max Normalization
        # 這樣才能保留 "全黑" 的預測結果
        
        print(f'\nEnsemble result - Range: [{ensres.min():.4f}, {ensres.max():.4f}]')
        
        # 直接存檔
        vutils.save_image(ensres, fp = os.path.join(args.out_dir, str(slice_ID)+'_output_ens'+".jpg"), nrow = 1, padding = 10, normalize=False)
        # --- 修正結束 ---

def create_argparser():
    defaults = dict(
        data_name = 'BRATS',
        data_dir="../dataset/brats2020/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",         #path to pretrain model
        num_ensemble=5,      #number of samples in the ensemble
        gpu_dev = "0",
        out_dir='./results/',
        multi_gpu = None, #"0,1,2"
        debug = False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()