import os
import rasterio
from rasterio.features import shapes
import fiona
from shapely.geometry import shape, mapping
from fiona.crs import from_epsg
from tqdm import tqdm  # 进度条库
from alive_progress import alive_bar
def tif_to_shapefile(tif_file, output_shapefile):
    # 创建shapefile的schema
    schema = {
        'geometry': 'Polygon',
        'properties': {'value': 'int'},
    }
    # 打开shapefile准备写入
    with fiona.open(output_shapefile, 'w', 'ESRI Shapefile', schema=schema, crs=from_epsg(4490)) as shapefile:
        # 打开TIF文件
        with rasterio.open(tif_file) as src:
            # 读取TIF数据
            image = src.read(1)

            # 将像素值为1的区域转换为多边形
            mask = image == 1
            results = shapes(image, mask=mask, transform=src.transform)

            # 使用 tqdm 显示处理多边形的进度
            # for geom, val in tqdm(results, desc="Processing shapes"):
                # 过滤值为1的区域
            with alive_bar(total=None, title="Processing shapes") as bar:
                for geom, val in results:
                    if val == 1:
                        geom_shape = shape(geom)
                        shapefile.write({
                            'geometry': mapping(geom_shape),
                            'properties': {'value': int(val)},
                        })
                    bar()

if __name__ == '__main__':
    tif_file = 'mydata/test/ludingdom_clip1_1.png'  # 替换为你的TIF文件路径
    output_shapefile = 'mydata/test'  # 替换为输出shapefile的路径
    tif_to_shapefile(tif_file, output_shapefile)
