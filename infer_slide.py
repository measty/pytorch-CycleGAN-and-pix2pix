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
from util.util import tensor2im, tensors2im
import torch
import matplotlib.pyplot as plt
from tiatoolbox.utils.image import imresize
import torch.nn as nn
from torch.utils.data import DataLoader
from tiatoolbox.models.dataset import WSIPatchDataset
import cv2

class ToFloatTensor:
    def __call__(self, tensor):
        return tensor.permute(0,3,1,2) / 255.0

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

def border_mask(slide_dimensions):
    # Step 1: Create a mask of the same aspect ratio as the slide but 5x smaller
    smaller_dims = (slide_dimensions[0] // 5, slide_dimensions[1] // 5)
    mask = np.zeros(smaller_dims, dtype=np.uint8)

    # Step 2: Calculate the dimensions of the inner rectangle (which will be white)
    border_thickness = int(slide_dimensions[1] / 40)
    inner_rectangle_top_left = (border_thickness, border_thickness)
    inner_rectangle_bottom_right = (smaller_dims[1] - border_thickness, smaller_dims[0] - border_thickness)

    # Draw the inner rectangle on the mask
    cv2.rectangle(mask, inner_rectangle_top_left, inner_rectangle_bottom_right, 255, -1)
    return mask

def is_valid_patch(image):
    # Ensure the image has three channels
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must have shape HxWx3.")
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Step 1: Compute the Laplacian of the image and then return the focus measure,
    # which is simply the variance of the Laplacian
    fm = cv2.Laplacian(gray, cv2.CV_32F).var()
    if fm < 65:  # This threshold can be adjusted based on your specific needs
        return False  # Blurry
    
    # Step 2: Analyze the histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    peak = np.max(hist)
    if peak > 0.5 * gray.size:
        brightness = np.sum(gray) / float(gray.size)
        if brightness > 228 or brightness < 50:
            return False  # Mostly bright or mostly dark indicating non-tissue or border
    
    return True

def construct_slide(slide_path, mask, patch_size=256, model=None, model_resolution=0, save_resolution=0, units='level', stride=None, save_path=None, back_heuristic='none', var_thresh=None, **kwargs):
    
    
    filename = Path(slide_path)
    if save_path is None:
        save_path = filename.parent

    tmp_path = save_path / (filename.stem + '_tmp_.npy')
    wsi = WSIReader.open(slide_path)
    mpp = wsi.info.mpp if wsi.info.mpp is not None else np.array([1.0, 1.0])
    if save_resolution == 0:
        canvas_shape = wsi.info.slide_dimensions[::-1]
    else:
        canvas_shape = (wsi.info.slide_dimensions[::-1] * (mpp[::-1] / save_resolution)).astype(int)
    #import pdb; pdb.set_trace()
    back_level = 245
    out_ch = 4 if stride is not None else 3
    back_tile = np.ones((patch_size, patch_size, 3), dtype=np.uint8) * back_level
    if mask == 'otsu':
        get_mask = True
        mask = None
    else:
        get_mask = False
    patch_dataset = WSIPatchDataset(
        img_path=slide_path,
        patch_input_shape=(patch_size, patch_size),
        stride_shape=stride or (patch_size, patch_size),
        mask_path=mask, #'morphological',
        min_mask_ratio=0.1, # only discard patches with very low tissue content
        resolution=model_resolution,
        units=units,
        auto_get_mask=get_mask,
        #preproc_func=lambda x: x.copy(),
    )
    if loader_workers > 0:
        patch_dataset.reader = slide_path
    dataloader = DataLoader(
        patch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader_workers,
        drop_last=False,
        pin_memory=True,
    )

    # patch_extractor = SlidingWindowPatchExtractor(
    #     input_img=slide_path,
    #     patch_size=(patch_size, patch_size),
    #     stride=stride,
    #     input_mask=mask, #'morphological',
    #     min_mask_ratio=0.1, # only discard patches with very low tissue content
    #     within_bound=True,
    #     resolution=resolution,
    #     units=units,
    # ) 

    #locs = patch_extractor.coordinate_list[:, :2]

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

    #for i, tile in tqdm(enumerate(patch_extractor), total=len(patch_extractor)):
    for batch in tqdm(dataloader, total=len(dataloader)):
        ims = batch['image'].float()
        locs = batch['coords']
        # if variance of tile vals less than threshold, skip
        to_proc = np.ones(ims.shape[0], dtype=bool)
        for i, tile in enumerate(ims):
            #import pdb; pdb.set_trace()
            if (var_thresh and tile.var() < var_thresh) or not is_valid_patch(tile.numpy()):
                to_proc[i] = False
        to_keep = []
        if to_proc.any():
            recs = model(ims[to_proc])
            for i, ind in zip(range(recs.shape[0]), np.arange(len(to_proc))[to_proc]):
                if np.mean(recs[i]) < 70 or (var_thresh and np.var(recs[i]) < var_thresh):
                    # if tile is very dark, or flat, will replace with background level
                    to_proc[ind] = False
                else:
                    to_keep.append(i)

        # add to canvas, using back_tile if not using processed tile
        current_ind = 0
        for i, loc in enumerate(locs):
            x, y = loc[0], loc[1]
            if to_proc[i]:
                rec = recs[current_ind]
                current_ind += 1
            else:
                rec = back_tile
            if model_resolution > 0 or save_resolution > 0 and units == 'mpp':
                x, y = int(x * model_resolution / save_resolution), int(y * model_resolution / save_resolution)
                out_size = (np.array(rec.shape[:2]) * model_resolution / save_resolution).astype(int) + 1
                rec = imresize(rec, output_size=out_size)
            if y+rec.shape[0] > canvas_shape[0] or x+rec.shape[1] > canvas_shape[1]:
                # patch overlaps edge of img, cropping.
                rec = rec[:canvas_shape[0]-y, :canvas_shape[1]-x]
            if rec.shape[0] == 0 or rec.shape[1] == 0 or y >= canvas_shape[0] or x >= canvas_shape[1]:
                # patch is completely outside of img, skipping. shouldnt happen but tiatoolbox patchextractor has bug
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
        
        def process_in_tiles():
            # divide by the number of times each pixel was written to, patchwise to avoid memory issues
            for i in tqdm(range(0, cum_canvas.shape[0], 4096)):
                for j in range(0, cum_canvas.shape[1], 4096):
                    # cum_canvas[
                    #     i : min(i + 4096, cum_canvas.shape[0]),
                    #     j : min(j + 4096, cum_canvas.shape[1]),
                    #     :3,
                    # ] = 
                    yield (
                        cum_canvas[
                            i : min(i + 4096, cum_canvas.shape[0]),
                            j : min(j + 4096, cum_canvas.shape[1]),
                            :3,
                        ]
                        / cum_canvas[
                            i : min(i + 4096, cum_canvas.shape[0]),
                            j : min(j + 4096, cum_canvas.shape[1]),
                            3:4,
                        ]
                    ).astype(
                        np.uint8
                    ), i, j
            #cum_canvas = cum_canvas[:,:,:3]
        
    # make a vips image and save it as a pyramidal tiff
    #height, width, bands = cum_canvas.shape
    #linear = cum_canvas.reshape(width * height * bands)
    print("building slide")
    if stride is None:
        vips_img = vips.Image.new_from_memory(
            cum_canvas[:,:,:3].astype(np.uint8).tobytes(),
            canvas_shape[1],
            canvas_shape[0],
            3,
            "uchar"
        )
    else:
        vips_img = None
        for tile, pos_x, pos_y in tqdm(process_in_tiles()):
            temp_vips_image = vips.Image.new_from_memory(
                tile.tobytes(), tile.shape[1], tile.shape[0], 3, "uchar"
            )
            
            if vips_img is None:
                vips_img = temp_vips_image
            else:
                vips_img = vips_img.insert(temp_vips_image, pos_y, pos_x, expand=True)


    # bioformats compatibility stuff
    image_height = vips_img.height
    image_bands = vips_img.bands
    #vips_img = vips.Image.arrayjoin(vips_img.bandsplit(), across=1)
    vips_img.set_type(vips.GValue.gint_type, "page-height", image_height)
    vips_img.set_type(vips.GValue.gstr_type, "image-description",
    f"""<?xml version="1.0" encoding="UTF-8"?>
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
        <Image ID="Image:0">
            <!-- Minimum required fields about image dimensions -->
            <Pixels DimensionOrder="XYCZT"
                    ID="Pixels:0"
                    SizeC="{image_bands}"
                    SizeT="1"
                    SizeX="{vips_img.width}"
                    SizeY="{image_height}"
                    SizeZ="1"
                    Type="uint8">
                    <Channel ID="Channel:0:0" SamplesPerPixel="3">
                        <LightPath/>
                    </Channel>
            </Pixels>
        </Image>
    </OME>""")

    # set resolution metadata - tiffsave expects res in pixels per mm regardless of resunit
    save_path = save_path / (filename.stem + f'{save_suffix}.tiff')
    if save_resolution == 0:
        save_resolution = mpp
    else:
        save_resolution = np.array([save_resolution, save_resolution])
    vips_img.tiffsave(save_path, tile=True, pyramid=True, compression="jpeg", Q=85, bigtiff=True, xres=1000/save_resolution[0], yres=1000/save_resolution[1], resunit="cm", tile_width=512, tile_height=512) #, subifd=True)
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
    if opt.gpu_ids == '-1' or len(opt.gpu_ids) == 0 or opt.name == 'none':
        device = 'cpu'
    else:
        device = 'cuda'
    if opt.name == 'none':
        # hack to use it as slide converter, we just pass the tiles through
        model = None
    else:
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        model.eval()

    transform_list = []
    transform_list += [ToFloatTensor()]
    if opt.model != 'stain_cycle':
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    trans = transforms.Compose(transform_list)

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, tiles):
            if self.model is None:
                return tiles.numpy()
            undo_norm=True
            if opt.model == 'stain_cycle':
                undo_norm=False
            tiles = trans(tiles)
            if opt.direction == 'AtoB':
                model.real_A = tiles.to(device)  # put image as input in mode A
                if opt.model == 'pix2pix':
                    model.forward()          # run inference
                    img_B = model.fake_B
                elif opt.model in ['cycle_gan', 'stain_cycle']:
                    img_B = model.netG_A(model.real_A)  # only need fake B
                return tensors2im(img_B, undo_norm=undo_norm)
                
            elif opt.direction == 'BtoA':
                model.real_B = tiles.to(device)
                if opt.model == 'pix2pix':
                    model.forward()
                    img_A = model.fake_A
                elif opt.model in ['cycle_gan', 'stain_cycle']:
                    img_A = model.netG_B(model.real_B)
                return tensors2im(img_A, undo_norm=undo_norm)

    loader_workers = 4
    slide_path = Path(opt.dataroot) 
    save_path = Path(opt.results_dir)
    names = opt.names
    save_suffix = opt.save_suffix
    batch_size = opt.batch_size
    stride = opt.stride
    if stride == 0:
        stride = None
    else:
        stride = (stride, stride)
    model_resolution = opt.model_resolution
    save_resolution = opt.save_resolution
    units = 'level'
    if model_resolution > 0:
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
    if names:
        named_files = []
        for name in names:
            named_files.extend([f for f in slides if name in f.stem])
        slides = named_files
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
        construct_slide(slide, mask, model=ModelWrapper(model), model_resolution=model_resolution, save_resolution=save_resolution, units=units, stride=stride, save_path=save_path, back_heuristic=back_heuristic, var_thresh=var_thresh)
        

