import numpy as np
from pathlib import Path
from PIL import Image

mask_path = Path(r"/media/u2071810/Data1/Acrobat/tissue-masks")
slides_path = Path(r"/media/u2071810/Data1/Acrobat/acrobat_validation_pyramid_1_of_1")
save_path = Path(r"/media/u2071810/Data1/Acrobat/mask_ims")

mask_files = list(mask_path.glob("*.npy"))
slide_files = list(slides_path.glob("*.tiff"))
print(f"Found {len(mask_files)} mask files and {len(slide_files)} slide files")

for slide in slide_files:
    slide_num = slide.stem.split("_")[0]
    stain = slide.stem.split("_")[1]

    if stain =="HE":
        mask_npy = np.load(mask_path / (slide_num + "_fixed.raw.0.npy"))
    else:
        mask_npy = np.load(mask_path / (slide_num + "_moving.raw.0.npy"))

    # save a .png of the mask
    mask_bool = (np.sum(mask_npy[:,:,1:], axis=2, keepdims=False) - mask_npy[:,:,0]) > 0.1
    mask_npy = np.zeros((mask_bool.shape[0], mask_bool.shape[1], 3), dtype=np.uint8)
    mask_npy[mask_bool] = np.array([255, 55, 55], dtype=np.uint8)
    mask_im = Image.fromarray(mask_npy)
    mask_im.save(save_path / (slide.stem + "_mask.png"))