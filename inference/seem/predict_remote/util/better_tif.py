from skimage import io, morphology, measure
import cv2
import numpy as np
import rasterio

def better_tif(mask_path, save_path):
    """
    Generate shapefile from mask
    Args:
        mask: numpy array, mask
    Returns:
        dst: numpy array, mask after removing small objects
    """
    with rasterio.open(mask_path) as src:
        profile = src.profile  # 读取原始影像的元数据（包括坐标系）
        mask = src.read(1)  # 读取第1个波段的像素数据
    Ho, Wo = mask.shape
    mask[mask==1]=255
    mask = cv2.resize(mask,(Ho,Wo),interpolation=cv2.INTER_NEAREST)
    dst=morphology.remove_small_objects(mask.astype(bool),min_size=10000,connectivity=1)
    dst = dst.astype(np.uint8)
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT,(20, 20))
    img = cv2.dilate(dst, kernal, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    max_area = 5000000
    props = measure.regionprops(img)
    for index, prop in enumerate(props, start=1):
        if prop.area > max_area:
            img[img==index] = 0
    # 保持数据类型一致（如果需要）
    profile.update(dtype=rasterio.uint8)
    
    with rasterio.open(save_path, "w", **profile) as dst:
        dst.write(img, 1)


if __name__ == '__main__':
    #随机一个mask
    mask = '/public/S/openmmlab/mmsegmentation/mydata/forseeing/测试/results/mx_clip4.tif'
    save_path='/public/S/openmmlab/mmsegmentation/mydata/forseeing/测试/results/mx_clip4_better.tif'
    img = better_tif(mask,save_path)
        #保存img2