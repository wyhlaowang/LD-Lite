import os
import cv2
import torch
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.io.image import read_image, ImageReadMode
from model import *
from prune import *


def rgb_y(im):
    DEV = im.device
    im_ra = rearrange(im, 'c h w -> h w c').cpu().numpy()
    im_ycrcb = cv2.cvtColor(im_ra, cv2.COLOR_RGB2YCrCb)
    im_y = torch.from_numpy(im_ycrcb[:,:,0]).unsqueeze(0).to(device=DEV)
    return im_y


def to_rgb(im_3, im_1):
    DEV = im_1.device
    im_3 = rearrange(im_3, 'c h w -> h w c').cpu().numpy()
    im_1 = rearrange(im_1, 'c h w -> h w c').cpu().numpy()
    crcb = cv2.cvtColor(im_3, cv2.COLOR_RGB2YCrCb)[:,:,1:]
    ycrcb = np.concatenate((im_1, crcb), -1)
    rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return rearrange(torch.from_numpy(rgb), 'h w c -> c h w').to(device=DEV)


def test(model, DEV='cuda'): 
    path_dict = './test_imgs/'

    ir_path = os.path.join(path_dict, 'ir/')
    vis_path = os.path.join(path_dict, 'vi/')
    file_list = os.listdir(ir_path)
    save_path = './results/'
    os.makedirs(save_path, exist_ok=True)

    print(f'Testing ... ')

    with torch.no_grad():
        for i in file_list:
            vis = read_image(vis_path + i, ImageReadMode.RGB).to(device=DEV) / 255.
            ir = read_image(ir_path + i, ImageReadMode.GRAY).to(device=DEV) / 255.

            vi_1 = rgb_y(vis)
            _, H, W = vis.shape

            fu = model(vi_1.unsqueeze(0), ir.unsqueeze(0)).squeeze(0)[:,:H,:W]
            fu_3 = to_rgb(vis, fu)

            im = 255 * fu_3.clamp(0,1).cpu()
            im = Image.fromarray(rearrange(im, 'c h w -> h w c').numpy().astype('uint8'))
            im.save(os.path.join(save_path, os.path.splitext(i)[0]+'.png'), quality=100)

            print(i)


def main():   
    DEV = 'cuda'
    model = Model(load_weight=False).to(DEV)
    pruned_model = prune_model(model, 0.5)
    weight = torch.load('./weight/final.pt', map_location=DEV)
    pruned_model.load_state_dict(weight, strict=True)

    test(model, DEV)


if __name__ == "__main__":
    main()



