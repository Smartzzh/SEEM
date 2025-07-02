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
    output_root = './output/test'
    input_folder = 'datasets/coco/val2017/'  # 影像文件夹路径

    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    stuff_classes = ['background']
    stuff_colors = [[255, 200, 100]]
    stuff_dataset_id_to_contiguous_id = {x: x for x in range(len(stuff_classes))}

    MetadataCatalog.get("demo").set(
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(stuff_classes + ["background"], is_eval=True)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(stuff_classes)

    # 如果输出目录不存在，创建目录
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # 遍历影像文件夹中的所有影像文件
    for image_file in os.listdir(input_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # 判断文件是否为图像
            image_pth = os.path.join(input_folder, image_file)

            with torch.no_grad():
                # 读取影像
                image_ori = Image.open(image_pth).convert("RGB")
                width = image_ori.size[0]
                height = image_ori.size[1]
                image = transform(image_ori)
                image = np.asarray(image)
                image_ori = np.asarray(image_ori)
                images = torch.from_numpy(image.copy()).permute(2, 0, 1).cuda()

                batch_inputs = [{'image': images, 'height': height, 'width': width}]
                outputs = model.forward(batch_inputs)
                visual = Visualizer(image_ori, metadata=metadata)

                # 获取语义分割结果
                sem_seg = outputs[-1]['sem_seg'].max(0)[1]
                demo = visual.draw_sem_seg(sem_seg.cpu(), alpha=0.5)  # rgb Image

                # 保存预测结果
                demo.save(os.path.join(output_root, f'{os.path.splitext(image_file)[0]}_sem.png'))


if __name__ == "__main__":
    main()
    sys.exit(0)
