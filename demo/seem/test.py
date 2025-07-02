import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping
import fiona

# 参数设置
point_shp = r"C:\Users\xfy\Desktop\识别任务\采样点"            # 点 shp 文件路径
img_folder = r"Z:\03 光学遥感影像\2024年影像\5132阿坝藏族羌族自治州"        # 影像文件夹路径（里面包含 .img 文件）
output_folder = r"Z:\03 光学遥感影像\2024年影像\sample"   # 输出文件夹路径（保存 tif 和 shp）
window_pixels = 1024                       # 指定的像素大小（长宽均为 window_pixels 个像素）

# 若输出文件夹不存在，则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取点 shp 文件（请保证点文件与影像投影一致）
points = gpd.read_file(point_shp)


# 遍历每个点要素
for idx, row in points.iterrows():
    point_geom = row.geometry
    # 使用 os.walk 遍历文件夹及子文件夹中的影像文件
    for root, dirs, files in os.walk(img_folder):
        for file in files:
            if file.lower().endswith('.img'):
                img_path = os.path.join(root, file)
                with rasterio.open(img_path) as src:
                    # 获取影像的空间参考信息和仿射变换参数
                    transform = src.transform
                    
                    #如果投影不一样则跳过
                    if src.crs != points.crs:
                        print(f"影像 {img_path} 和点文件投影不一致，跳过。")
                        continue
                    # 计算影像分辨率（注意：transform.a 为 x 分辨率，transform.e 为 y 分辨率，一般为负数，取绝对值）
                    
                    pixel_width = abs(transform.a)
                    pixel_height = abs(transform.e)
                    # 根据指定像素数计算矩形半宽、半高（单位：影像的空间单位）
                    half_width = (window_pixels * pixel_width) / 2
                    half_height = (window_pixels * pixel_height) / 2

                    # 获取点的坐标
                    x, y = point_geom.x, point_geom.y
                    # 构建以点为中心的矩形（面状）几何对象
                    rect = box(x - half_width, y - half_height, x + half_width, y + half_height)

                    # 判断矩形与影像边界是否相交
                    img_bounds = box(*src.bounds)
                    if not rect.intersects(img_bounds):
                        continue  # 当前影像与该矩形没有交集则跳过

                    # 使用矩形对影像进行裁剪（若部分在影像范围外则自动裁剪交集部分）
                    try:
                        out_image, out_transform = mask(src, [mapping(rect)], crop=True)
                    except Exception as e:
                        print(f"裁剪影像 {img_path} 时出错：{e}")
                        continue

                    # 构造输出文件名，保证 tif 与 shp 名称相同（可根据需要修改命名规则）
                    # 这里使用点索引和影像文件名构造输出文件名
                    base_name = f"point_{idx}_{os.path.splitext(file)[0]}"
                    out_tif = os.path.join(output_folder, base_name + ".tif")
                    out_shp = os.path.join(output_folder, base_name + ".shp")

                    # 保存裁剪后的影像为 tif
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform
                    })
                    with rasterio.open(out_tif, "w", **out_meta) as dest:
                        dest.write(out_image)
                    print(f"保存裁剪影像：{out_tif}")

                    # 保存裁剪范围的 shp 文件，使用 fiona 写出单个多边形
                    schema = {
                        'geometry': 'Polygon',
                        'properties': {'id': 'int'},
                    }
                    # 使用影像的坐标系保存 shp（请确保点 shp 与影像投影一致）
                    with fiona.open(out_shp, 'w', driver='ESRI Shapefile',
                                    crs=src.crs,
                                    schema=schema) as shp:
                        shp.write({
                            'geometry': mapping(rect),
                            'properties': {'id': idx}
                        })
                    print(f"保存裁剪范围 shp：{out_shp}")
