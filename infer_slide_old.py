"""Script for applying a trained cyclegan or pix2pix model patchwise on all patches
of a WSI, and reconstucting a slide from them. Adapted from test.py.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

If you provide a mask_dir, it will only apply the model to the patches that are
within the mask. masks are expected to be named slide_stem + '_mask.png'.
passing 'otsu' will use tiatoolbox to make a mask (default). none will not use a mask.

A pattern can be passed in the dataroot arg to only process slides that match the pattern.
e.g /path/to/slides/*.ndpi will only process ndpi slides.

Example:
    Test a CycleGAN model:
        python infer_slide.py --dataroot path/to/slides --name saved_model_name --model cycle_gan

    The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python infer_slide.py --dataroot path/to/slides --name model_name --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.

"""
import os
import argparse
from options.test_options import TestOptions, SlideInferOptions
from models import create_model
from util import html
from pathlib import Path
import numpy as np
import pyvips as vips
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor, PatchExtractor
from tiatoolbox.wsicore import WSIReader
from tqdm import tqdm
import torchvision.transforms as transforms
from util.util import tensor2im
import torch
import matplotlib.pyplot as plt
from tiatoolbox.utils.image import imresize
import torch.nn as nn
from tiatoolbox.models.dataset import WSIPatchDataset

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

def construct_slide(slide_path, mask, patch_size=256, model=None, resolution=0, units='level', stride=None, save_path=None, back_heuristic='none', var_thresh=None, **kwargs):
    
    
    filename = Path(slide_path)
    if save_path is None:
        save_path = filename.parent

    tmp_path = save_path / (filename.stem + '_tmp_.npy')
    wsi = WSIReader.open(slide_path)
    canvas_shape = wsi.info.slide_dimensions[::-1]
    mpp = wsi.info.mpp
    back_level = 245
    out_ch = 4 if stride is not None else 3

    patch_extractor = SlidingWindowPatchExtractor(
        input_img=slide_path,
        patch_size=(patch_size, patch_size),
        stride=stride,
        input_mask=mask, #'morphological',
        min_mask_ratio=0.1, # only discard patches with very low tissue content
        within_bound=True,
        resolution=resolution,
        units=units,
    ) 

    locs = patch_extractor.coordinate_list[:, :2]

    cum_canvas = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        shape=tuple(canvas_shape) + (out_ch,),
        dtype=np.uint16 if stride is not None else np.uint8,
    )
    
    if stride is not None:
        cum_canvas[:] = 0
    else:
        cum_canvas[:] = back_level

    for i, tile in tqdm(enumerate(patch_extractor), total=len(patch_extractor)):
        # if variance of tile vals less than threshold, skip
        if var_thresh and np.var(tile) < var_thresh:
            rec = tile
        else:
            rec = model(tile)
            # if variance of processed tile vals less than threshold, skip
            if np.var(rec) < var_thresh:
                rec = tile
        # if tile is very dark, replace with background level
        if np.mean(rec) < 55:
            rec = np.ones_like(rec) * back_level
        x, y = locs[i]
        if resolution > 0 and units == 'mpp':
            x, y = int(x * resolution / mpp[0]), int(y * resolution / mpp[1])
            out_size = (np.array(rec.shape[:2]) * resolution / mpp).astype(int) + 1
            rec = imresize(rec, output_size=out_size)
        if y+rec.shape[0] > canvas_shape[0] or x+rec.shape[1] > canvas_shape[1]:
            print("patch out of bounds, cropping.")
            rec = rec[:canvas_shape[0]-y, :canvas_shape[1]-x]
            if rec.shape[0] == 0 or rec.shape[1] == 0:
                continue
        if stride is None:
            cum_canvas[y:y + rec.shape[0], x:x + rec.shape[1], :3] = rec
        else:
            # keep track of how many times each pixel has been written to
            cum_canvas[y:y + rec.shape[0], x:x + rec.shape[1], 3] += 1
            # add the new tile to the canvas
            cum_canvas[y:y + rec.shape[0], x:x + rec.shape[1], :3] += rec
    if stride is not None:
        # set pixels that havent been written to background level
        cum_canvas[cum_canvas[:,:,3] == 0, :3] = back_level
        # set pixel counts of background pixels to 1 to avoid divide by zero
        cum_canvas[cum_canvas[:,:,3] == 0, 3] = 1
        # divide by the number of times each pixel was written to
        cum_canvas[:,:,:3] = cum_canvas[:,:,:3] / cum_canvas[:,:,3:4]
        cum_canvas = cum_canvas[:,:,:3]
        
    # make a vips image and save it as a pyramidal tiff
    #height, width, bands = cum_canvas.shape
    #linear = cum_canvas.reshape(width * height * bands)
    vips_img = vips.Image.new_from_memory(
        cum_canvas[:,:,:3].astype(np.uint8).tobytes(),
        canvas_shape[1],
        canvas_shape[0],
        3,
        "uchar"
    )
    # set resolution metadata - tiffsave expects res in pixels per mm regardless of resunit
    save_path = save_path / (filename.stem + '_proc.tiff')
    vips_img.tiffsave(save_path, tile=True, pyramid=True, compression="jpeg", Q=85, bigtiff=True, xres=1000/mpp[0], yres=1000/mpp[1], resunit="cm", tile_width=512, tile_height=512)
    print(f"saved slide {filename.stem} to {save_path}")
    # close memmap and clean up
    cum_canvas._mmap.close()
    del cum_canvas
    os.remove(tmp_path)


if __name__ == '__main__':
    opts = SlideInferOptions()
    opt = opts.parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    trans = transforms.Compose(transform_list)

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    model.eval()
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, tile):
            tile = trans(tile.copy())
            if opt.direction == 'AtoB':
                model.real_A = torch.unsqueeze(tile, 0).to('cuda')  # put image as input in mode A
                if opt.model == 'pix2pix':
                    model.forward()          # run inference
                    img_B = model.fake_B
                elif opt.model == 'cycle_gan':
                    img_B = model.netG_A(model.real_A)  # only need fake B
                img_B = tensor2im(img_B)
                return img_B
            elif opt.direction == 'BtoA':
                model.real_B = torch.unsqueeze(tile, 0).to('cuda')
                if opt.model == 'pix2pix':
                    model.forward()
                    img_A = model.fake_A
                elif opt.model == 'cycle_gan':
                    img_A = model.netG_B(model.real_B)
                img_A = tensor2im(img_A)
                return img_A

    slide_path = Path(opt.dataroot) 
    save_path = Path(opt.results_dir) 
    batch_size = opt.batch_size
    stride = opt.stride
    if stride == 0:
        stride = None
    else:
        stride = (stride, stride)
    resolution = opt.resolution
    units = 'level'
    if resolution > 0:
        units = 'mpp'
    mask_opt = opt.masks_dir
    if mask_opt not in ["otsu", "morphological", "none"]:
        mask_opt = Path(mask_opt)
    if mask_opt == "none":
        mask_opt = None
    var_thresh = opt.var_thresh
    if var_thresh == 0:
        var_thresh = None
    back_heuristic = opt.bkgrnd_heuristic
    if "*" in slide_path.name:
        slide_filter = slide_path.name
        slide_path = slide_path.parent
    else:
        slide_filter = "*"
    save_path.mkdir(exist_ok=True)
    if slide_path.is_dir():
        slides = list(slide_path.glob(slide_filter))
    else:
        slides = [slide_path]
    slides.sort()

    print(f"found {len(slides)} slides")
    print(f"processing {min(len(slides), opt.num_test)} slides")
    for slide in slides[:min(len(slides), opt.num_test)]:
        # try to get mask
        if isinstance(mask_opt, Path):
            mask = mask_opt / (slide.stem + '_mask.png')
        else:
            mask = mask_opt
        print(f"starting slide {slide}")
        construct_slide(slide, mask, model=ModelWrapper(model), resolution=resolution, units=units, stride=stride, save_path=save_path, back_heuristic=back_heuristic, var_thresh=var_thresh)
      