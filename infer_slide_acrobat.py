"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
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

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

def construct_slide(slide_path, mask_path, patch_size=256, proc_fn=None, save_path=None, **kwargs):
    
    
    filename = Path(slide_path)
    if save_path is None:
        save_path = filename.parent

    tmp_path = filename.parent / (filename.stem + '_tmp_.npy')
    wsi = WSIReader.open(slide_path)
    canvas_shape = wsi.info.slide_dimensions[::-1]
    mpp = wsi.info.mpp
    out_ch = 3
    patch_extractor = SlidingWindowPatchExtractor(
        input_img=slide_path,
        patch_size=(patch_size, patch_size),
        stride=(patch_size, patch_size),
        input_mask=mask_path, # 'otsu', #'morphological',
        min_mask_ratio=0.1, # only discard patches with very low tissue content
        within_bound=True,
        #resolution=0.254,
        #units="mpp",
    ) 

    locs = patch_extractor.coordinate_list[:, :2]

    cum_canvas = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        shape=tuple(canvas_shape) + (out_ch,),
        dtype=np.uint8,
    )
    cum_canvas[:] = 240

    for i, tile in tqdm(enumerate(patch_extractor), total=len(patch_extractor)):
        # if variance of tile vals less than threshold, skip
        if np.var(tile) < 10:
            rec = tile
        else:
            rec = proc_fn(tile, **kwargs)
        #x, y = (locs[i] * (0.254/mpp)).astype(int)
        x, y = locs[i]
        out_size = (np.array(rec.shape[:2]) * 0.254 / mpp).astype(int) + 1
        #rec = imresize(rec, output_size=out_size)
        if y+rec.shape[0] > canvas_shape[0] or x+rec.shape[1] > canvas_shape[1]:
            print("meep")
            x = x - (x+rec.shape[1] - canvas_shape[1])
            y = y - (y+rec.shape[0] - canvas_shape[0])
        cum_canvas[y:y + rec.shape[0], x:x + rec.shape[1], :] = rec

    # make a vips image and save it as a pyramidal tiff
    #height, width, bands = cum_canvas.shape
    #linear = cum_canvas.reshape(width * height * bands)
    vips_img = vips.Image.new_from_memory(
        cum_canvas.tobytes(),
        canvas_shape[1],
        canvas_shape[0],
        out_ch,
        "uchar"
    )
    # set resolution metadata - tiffsave expects res in pixels per mm regardless of resunit
    save_path = save_path / (filename.stem + '_p2p.tiff')
    vips_img.tiffsave(save_path, tile=True, pyramid=True, compression="jpeg", Q=85, bigtiff=True, xres=1000/mpp[0], yres=1000/mpp[1], resunit="cm", tile_width=512, tile_height=512)
    print(f"saved slide {filename.stem} to {save_path}")
    # close memmap and clean up
    cum_canvas._mmap.close()
    del cum_canvas
    os.remove(tmp_path)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
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

    # create a website
    # web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    # if opt.load_iter > 0:  # load_iter is 0 by default
    #     web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    # print('creating web directory', web_dir)
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.

    model.eval()
    def proc_fn(tile):
        tile = trans(tile.copy())
        model.real_A = torch.unsqueeze(tile, 0).to('cuda')  # put image as input in mode A
        if opt.model == 'pix2pix':
            model.forward()          # run inference
            img_B = model.fake_B
        elif opt.model == 'cycle_gan':
            img_B = model.netG_A(model.real_A)  # only need fake B
        img_B = tensor2im(img_B)
        return img_B

    #slide_path = Path("/media/u2071810/Data1/Multiplexstaining/Asmaa_Multiplex_Staining/81ROTS_PHH3.svs")
    slide_path = Path("/media/u2071810/Data1/Acrobat/acrobat_validation_pyramid_1_of_1")
    mask_path = Path("/media/u2071810/Data1/Acrobat/mask_ims")
    save_path = Path("/media/u2071810/Data1/Acrobat/restained_ims_acro")
    save_path.mkdir(exist_ok=True)
    slides = list(slide_path.glob("*.tiff"))
    slides.sort()
    slides_keep=[]
    for slide in slides:
        if "HE" not in slide.stem:
            slides_keep.append(slide)
    slides = slides_keep
    for slide in slides[:4]:
        print(f"starting slide {slide}")
        mask = None # list(mask_path.glob(f"{slide.stem}_mask*"))[0]
        construct_slide(slide, mask, proc_fn=proc_fn, save_path=save_path)
        
