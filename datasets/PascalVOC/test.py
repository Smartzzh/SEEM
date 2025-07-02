import os

def save_filenames_without_extension(folder_path, output_file):
    """
    提取文件夹中所有文件的文件名（不带后缀），保存到指定文件中。

    Args:
        folder_path (str): 要处理的文件夹路径。
        output_file (str): 保存结果的文件路径。
    """
    try:
        # 获取文件夹中所有文件名
        filenames = os.listdir(folder_path)
        # 去除后缀
        filenames_without_extension = [os.path.splitext(filename)[0] for filename in filenames]
        # 保存到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for name in filenames_without_extension:
                f.write(name + '\n')
        print(f"文件名已保存到 {output_file}")
    except Exception as e:
        print(f"发生错误: {e}")

# 使用示例
folder_path = "pth/to/xdecoder_data/PascalVOC/panoptic_val2017"  # 替换为目标文件夹路径
output_file = "pth/to/xdecoder_data/PascalVOC/test.txt"          # 保存的文件路径
save_filenames_without_extension(folder_path, output_file)
