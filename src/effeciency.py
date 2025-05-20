import os
import torch
from thop import profile
from torchvision.io import read_image, ImageReadMode
from model import *
from prune import *


def effeciency_test(model, DEV='cuda'):
    im1 = torch.rand((1,1,224,224)).to(DEV)
    im2 = torch.rand((1,1,224,224)).to(DEV)
    flops, params = profile(model, inputs=(im1, im2, ))
    print(f'===== FLOPs:  {flops / 1e9:.4f} G')
    print(f'===== Params: {params / 1e6:.4f} M')

    # time on fixed-size images
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    test_num = 100
    with torch.no_grad():
        starter.record()
        for _ in range(test_num):
            model(im1, im2)
        ender.record()
        torch.cuda.synchronize()
        total_time = starter.elapsed_time(ender)
        print(f'===== time: {(total_time/test_num):.4f} ms')
        print(f'===== fps:  {(test_num/(total_time/1000)):.4f}')

    # time on dataset
    test_num = 100
    ir_path = './test_imgs/ir/'
    vi_path = './test_imgs/vi/'
    file_list = os.listdir(ir_path) * test_num
    total_time = 0
    with torch.no_grad():
        for i in file_list:
            print(i)
            ir = (read_image(os.path.join(ir_path, i), ImageReadMode.GRAY) / 255.).unsqueeze(0).to(DEV)
            vi = (read_image(os.path.join(vi_path, i), ImageReadMode.GRAY) / 255.).unsqueeze(0).to(DEV)
            starter.record()
            model(ir, vi)
            ender.record()
            torch.cuda.synchronize()
            single_time = starter.elapsed_time(ender)
            total_time = total_time + single_time
    print(f'===== time: {(total_time/len(file_list)):.4f} ms')
    print(f'===== fps:  {(len(file_list)/(total_time/1000)):.4f}')


def main():   
    DEV = 'cuda'
    model = Model(load_weight=False).to(DEV)
    pruned_model = prune_model(model, 0.5)
    weight = torch.load('./weight/final.pt', map_location=DEV)
    pruned_model.load_state_dict(weight, strict=True)

    effeciency_test(model, DEV)

    
if __name__ == "__main__":
    main()