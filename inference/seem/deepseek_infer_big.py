import os
import sys
import logging
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import rasterio
from rasterio.windows import Window
from utils.arguments import load_opt_command
from utils.distributed import init_distributed
from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

logger = logging.getLogger(__name__)

def sliding_window_inference(model, dataset, patch_size, stride, metadata, transform, device='cuda'):
    """
    Perform sliding-window inference on a large georeferenced raster.
    Returns a binary mask in full resolution.
    """
    # Allocate full-size output mask
    height = dataset.height
    width = dataset.width
    mask_full = np.zeros((height, width), dtype=np.uint8)

    # Iterate over windows
    for top in range(0, height, stride):
        for left in range(0, width, stride):
            win = Window(left, top,
                         min(patch_size, width - left),
                         min(patch_size, height - top))
            # Read patch as array (C, H, W)
            patch = dataset.read(window=win)  # shape: (bands, h, w)
            # Skip if not 3-band
            if patch.shape[0] < 3:
                continue
            # Convert to HWC uint8
            img = np.transpose(patch[:3, :, :], (1, 2, 0))
            img_pil = Image.fromarray(img)
            img_resized = transform(img_pil)
            tensor = torch.from_numpy(np.array(img_resized)).permute(2,0,1).float().to(device)
            batch_inputs = [{'image': tensor, 'height': img_resized.size[1], 'width': img_resized.size[0]}]

            with torch.no_grad():
                outputs = model.forward(batch_inputs)
            pano_seg = outputs[-1]['panoptic_seg'][0].cpu().numpy()
            pano_info = outputs[-1]['panoptic_seg'][1]

            # Build binary patch mask
            binary_patch = np.zeros((pano_seg.shape[0], pano_seg.shape[1]), dtype=np.uint8)
            for seg_info in pano_info:
                cid = seg_info['category_id']
                if cid in metadata.thing_dataset_id_to_contiguous_id.values():
                    binary_patch[pano_seg == seg_info['id']] = 1

            # Resize back to original window size if needed
            if binary_patch.shape != (win.height, win.width):
                binary_patch = np.array(Image.fromarray(binary_patch * 255)
                                        .resize((win.width, win.height), resample=Image.NEAREST) // 255,
                                        dtype=np.uint8)

            # Fill into full mask
            mask_full[top:top + win.height, left:left + win.width] = np.maximum(
                mask_full[top:top + win.height, left:left + win.width], binary_patch)

    return mask_full


def main(args=None):
    """
    Main entry: sliding-window inference with geospatial output.
    """
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        opt['base_path'] = os.path.abspath(cmdline_args.user_dir)

    opt = init_distributed(opt)
    pretrained_pth = opt['RESUME_FROM']
    input_tif = opt.get('INPUT_TIF', 'datasets/coco/large_image.tif')
    output_tif = opt.get('OUTPUT_TIF', 'output/pred_mask.tif')

    # Build and load model
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    # Set up metadata
    thing_classes = ['landslide']
    stuff_classes = ['background']
    MetadataCatalog.get('demo').set(
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
        thing_colors=[[0,255,0]],
        stuff_colors=[[0,255,0]],
        thing_dataset_id_to_contiguous_id={i:i for i in range(len(thing_classes))},
        stuff_dataset_id_to_contiguous_id={i+len(thing_classes):i for i in range(len(stuff_classes))}
    )
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
        thing_classes + stuff_classes + ['background'], is_eval=False)
    model.model.sem_seg_head.num_classes = len(thing_classes + stuff_classes)

    # Prepare transform (resize to model input)
    patch_size = opt.get('PATCH_SIZE', 512)
    transform = transforms.Compose([transforms.Resize(patch_size, interpolation=Image.BICUBIC)])
    stride = opt.get('STRIDE', patch_size // 2)

    # Open geotiff
    with rasterio.open(input_tif) as dataset:
        meta = dataset.meta.copy()
        # Run sliding-window inference
        mask = sliding_window_inference(model, dataset, patch_size, stride, metadata, transform)

        # Write mask as geotiff, preserve georeference
        meta.update({
            'driver': 'GTiff',
            'dtype': rasterio.uint8,
            'count': 1
        })
        os.makedirs(os.path.dirname(output_tif), exist_ok=True)
        with rasterio.open(output_tif, 'w', **meta) as dst:
            dst.write(mask, 1)

    logger.info(f"Saved sliding-window prediction mask to {output_tif}")

if __name__ == '__main__':
    main()
    sys.exit(0)
