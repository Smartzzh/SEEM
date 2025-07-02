import os
import sys
import logging
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from utils.arguments import load_opt_command
from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.visualizer import Visualizer
from utils.distributed import init_distributed

logger = logging.getLogger(__name__)

def main(args=None):
    '''
    Main execution point for PyLearn.
    '''
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['base_path'] = absolute_user_dir

    opt = init_distributed(opt)

    # META DATA
    pretrained_pth = os.path.join(opt['RESUME_FROM'])
    output_root = './output/SEEM_AFPN_for2_train2017_test'
    input_folder = 'datasets/coco/train2017/'  # 输入影像文件夹路径

    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    thing_classes = ['landslide']
    stuff_classes = ['background']
    thing_colors = [[0, 255, 0]]
    # thing_colors = [[255, 200, 100]]
    stuff_colors = [[0, 255, 0]]
    thing_dataset_id_to_contiguous_id = {x:x for x in range(len(thing_classes))}
    stuff_dataset_id_to_contiguous_id = {x+len(thing_classes):x for x in range(len(stuff_classes))}

    MetadataCatalog.get("demo").set(
        thing_colors=thing_colors,
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes + stuff_classes + ["background"], is_eval=False)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(thing_classes + stuff_classes)

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # 遍历文件夹中的所有影像文件
    for image_file in os.listdir(input_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # 判断文件是否为图像
            image_pth = os.path.join(input_folder, image_file)
            with torch.no_grad():
                image_ori = Image.open(image_pth).convert("RGB")
                width = image_ori.size[0]
                height = image_ori.size[1]
                image = transform(image_ori)
                image = np.asarray(image)
                image_ori = np.asarray(image_ori)
                images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

                batch_inputs = [{'image': images, 'height': height, 'width': width}]
                outputs = model.forward(batch_inputs)
                visual = Visualizer(image_ori, metadata=metadata)

                pano_seg = outputs[-1]['panoptic_seg'][0]
                pano_seg_info = outputs[-1]['panoptic_seg'][1]

                # 创建一个与预测结果大小相同的全零数组，用于存储二值化的结果
                binary_pred = np.zeros_like(pano_seg.cpu().numpy(), dtype=np.uint8)

                # 将类别为滑坡（landslide）的区域设置为1
                for i, seg_info in enumerate(pano_seg_info):
                    if seg_info['category_id'] in metadata.thing_dataset_id_to_contiguous_id:
                        # 如果是滑坡区域，则设置为1
                        binary_pred[pano_seg.cpu().numpy() == seg_info['id']] = 1

                # 保存二值化的预测结果
                bin_pred_image = Image.fromarray(binary_pred)  # 将二值化图像的值从 [0,1] 转为 [0,255] 范围
                bin_pred_image.save(os.path.join(output_root, f'{os.path.splitext(image_file)[0]}.png'))

if __name__ == "__main__":
    main()
    sys.exit(0)
