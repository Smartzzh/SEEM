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
    # 加载命令行参数和配置
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['base_path'] = absolute_user_dir

    opt = init_distributed(opt)

    # 加载预训练模型以及配置文件
    pretrained_pth = os.path.join(opt['RESUME_FROM'])
    output_root = '/public/S/seem/Segment-Everything-Everywhere-All-At-Once/output/big_test/results'  # 输出结果文件夹路径
    input_folder = '/public/S/seem/Segment-Everything-Everywhere-All-At-Once/output/big_test/'  # 输入大图文件夹路径

    # 初始化模型（确保模型处于 eval 模式并转到 GPU 上）
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    # 可选的预处理（如果不需要 resize，直接注释掉）
    transform = transforms.Compose([
        # transforms.Resize(512, interpolation=Image.BICUBIC)
    ])

    # 定义类别与元数据信息
    thing_classes = ['landslide']
    stuff_classes = ['background']
    thing_colors = [[0, 255, 0]]
    stuff_colors = [[0, 255, 0]]
    thing_dataset_id_to_contiguous_id = {x: x for x in range(len(thing_classes))}
    stuff_dataset_id_to_contiguous_id = {x + len(thing_classes): x for x in range(len(stuff_classes))}

    MetadataCatalog.get("demo").set(
        thing_colors=thing_colors,
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    # 初始化文本编码器（如果使用语言引导的分割）
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
        thing_classes + stuff_classes + ["background"], is_eval=False)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(thing_classes + stuff_classes)

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # 设置滑动窗口的参数（可根据需要调整）
    window_size = (1024, 1024)  # (宽, 高)
    stride = (718, 718)       # (水平步长, 垂直步长)

    # 遍历输入文件夹中的所有影像文件
    for image_file in os.listdir(input_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp', '.img')):
            image_pth = os.path.join(input_folder, image_file)
            with torch.no_grad():
                # 读取原图，保留原始分辨率和坐标信息
                image_ori = Image.open(image_pth).convert("RGB")
                width, height = image_ori.size

                # 将原图转换为 NumPy 数组（仅用于可视化，预测时用 PIL）
                image_np = np.asarray(image_ori)
                # 创建一个与原图尺寸一致的预测结果数组（0为背景）
                final_binary_pred = np.zeros((height, width), dtype=np.uint8)

                # 为防止窗口边缘不足的情况，计算所有窗口的起始坐标
                xs = list(range(0, width, stride[0]))
                ys = list(range(0, height, stride[1]))
                # 确保最后一个窗口能覆盖到右边缘和下边缘
                if xs[-1] + window_size[0] < width:
                    xs.append(width - window_size[0])
                if ys[-1] + window_size[1] < height:
                    ys.append(height - window_size[1])
                
                # 遍历所有窗口区域进行推理
                for y in ys:
                    for x in xs:
                        # 裁剪窗口区域；注意边缘处可能比 window_size 小
                        window_img = image_ori.crop((x, y, x + window_size[0], y + window_size[1]))
                        window_np = np.asarray(window_img)
                        # 转换为 tensor（C×H×W）并传入 GPU
                        window_tensor = torch.from_numpy(window_np.copy()).permute(2, 0, 1).float().cuda()
                        
                        # 构造单图 batch 输入，同时传入窗口的原始尺寸
                        batch_inputs = [{'image': window_tensor,
                                         'height': window_np.shape[0],
                                         'width': window_np.shape[1]}]
                        outputs = model.forward(batch_inputs)
                        # 提取当前窗口预测结果
                        pano_seg = outputs[-1]['panoptic_seg'][0]
                        pano_seg_info = outputs[-1]['panoptic_seg'][1]
                        # 将结果转换为 numpy 数组
                        pano_seg_np = pano_seg.cpu().numpy()

                        # 初始化当前窗口的二值预测（0或1）
                        window_binary_pred = np.zeros_like(pano_seg_np, dtype=np.uint8)
                        for seg_info in pano_seg_info:
                            if seg_info['category_id'] in metadata.thing_dataset_id_to_contiguous_id:
                                window_binary_pred[pano_seg_np == seg_info['id']] = 1

                        # 根据原图上窗口的实际覆盖区域计算有效区域尺寸
                        valid_h = min(window_binary_pred.shape[0], height - y)
                        valid_w = min(window_binary_pred.shape[1], width - x)

                        # 将当前窗口的有效预测区域与全图对应区域做或运算
                        final_binary_pred[y:y+valid_h, x:x+valid_w] = np.maximum(
                            final_binary_pred[y:y+valid_h, x:x+valid_w],
                            window_binary_pred[:valid_h, :valid_w]
                        )

                # 保存全图的二值化预测结果，像素坐标与原图一致
                out_image = Image.fromarray(final_binary_pred * 255)  # 将 0/1 映射至 0/255
                out_path = os.path.join(output_root, f'{os.path.splitext(image_file)[0]}.png')
                out_image.save(out_path)
                logger.info(f"Saved prediction: {out_path}")

if __name__ == "__main__":
    main()
    sys.exit(0)
