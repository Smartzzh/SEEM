import os
import rasterio
from rasterio.features import shapes
import fiona
from shapely.geometry import shape, mapping
from fiona.crs import from_epsg
from alive_progress import alive_bar

def tif_to_shapefile(tif_file, output_shapefile):
    # 打开TIF文件
    with rasterio.open(tif_file) as src:
        # 获取TIF文件的CRS和transform
        crs = src.crs
        transform = src.transform
        
        # 创建shapefile的schema
        schema = {
            'geometry': 'Polygon',
            'properties': {'value': 'int'},
        }
        
        # 读取TIF数据
        image = src.read(1)

        # 将像素值为1的区域转换为多边形
        mask = image == 1
        results = shapes(image, mask=mask, transform=transform)

        # 打开Shapefile准备写入，并使用与TIF相同的CRS和transform
        with fiona.open(output_shapefile, 'w', driver='ESRI Shapefile', schema=schema, crs=crs, transform=transform) as shapefile:
            # 使用 alive_bar 显示处理进度
            with alive_bar(total=None, title="Processing shapes") as bar:
                for geom, val in results:
                    if val == 1:
                        geom_shape = shape(geom)
                        shapefile.write({
                            'geometry': mapping(geom_shape),
                            'properties': {'value': int(val)},
                        })
                    bar()  # 更新进度条

if __name__ == '__main__':
    tif_file = '/public/S/高分影像/泸定影像/results/ld2/ld.tif'  # 替换为你的TIF文件路径
    output_shapefile = '/public/S/高分影像/泸定影像/results/ld2/output_shapefile.shp'  # 替换为输出shapefile的路径
    tif_to_shapefile(tif_file, output_shapefile)
