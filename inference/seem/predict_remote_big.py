import os
from argparse import ArgumentParser
from inference.seem.predict_remote.util import RSImage, RSInferencer
from inference.seem.predict_remote.util import tif_to_shapefile
from mmengine import Config

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('image_folder', help='Root folder containing subfolders with images')
    parser.add_argument('--output-path',help='Directory to save result images',default='results/')
    parser.add_argument('--batch-size',type=int,default=20,help='Maximum number of windows inferred simultaneously')
    parser.add_argument('--window-size',help='Window xsize, ysize',default=(1024, 1024),type=int,nargs=2)
    parser.add_argument('--stride',help='Window xstride, ystride',default=(10, 10),type=int,nargs=2)
    parser.add_argument(
        '--thread', default=1, type=int, help='Number of inference threads')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('command', help='Command: train/evaluate/train-and-evaluate')
    parser.add_argument('--conf_files', nargs='+', required=True, help='Path(s) to the config file(s).')
    
    args = parser.parse_args()
   
    
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.bmp', '.img')
    # 遍历 root 文件夹中的所有子文件夹中的文件
    for root, dirs, files in os.walk(args.image_folder):
        # 筛选出 tif 文件
        tif_files = [
            os.path.join(root, f) for f in files if f.lower().endswith(supported_extensions)
        ]
        if not tif_files:
            continue

        # 为每个子文件夹创建相应的输出路径
        output_path = os.path.join(args.output_path, os.path.relpath(root, args.image_folder))
        os.makedirs(output_path, exist_ok=True)

        # 对每张图像进行预测
        for image_path in tif_files:
            # 初始化推理器
            inferencer = RSInferencer.from_config_path(
                args.command,
                args.conf_files,
                args.batch_size,
                args.thread,
                args.device)

            image = RSImage(image_path)
            # 输出路径的文件名保持为 tif 后缀
            image_name = os.path.splitext(image_path)[0] + '.tif'
            output_path_file = os.path.join(output_path, os.path.basename(image_name))

            print(f"Processing {image_path} -> {output_path_file}")
            cfg = Config.fromfile(args.config)
            inferencer.run(image, args.window_size, args.stride, output_path_file, cfg)

            # 在 output_path 下生成 shapefile 文件夹
            shapefile_folder = os.path.join(output_path, '隐患点')
            os.makedirs(shapefile_folder, exist_ok=True)

            output_shapefile = os.path.join(shapefile_folder, os.path.basename(image_path)[:-4] + '.shp')
            print(f"Generating shapefile {output_shapefile}")
            tif_to_shapefile(output_path_file, output_shapefile)
            
            del inferencer
            del image
           

if __name__ == '__main__':
    main()
