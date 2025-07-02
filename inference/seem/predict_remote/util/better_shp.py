import os
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.geometry.polygon import orient

# 设置最小面积阈值（单位：平方米）
MIN_AREA = 10000  # 根据需求调整

# 文件夹路径
input_folder = r"C:\Users\Administrator\Desktop\ls\mx_合集"
output_folder = r"C:\Users\Administrator\Desktop\ls\mx_合集\re2"

# 确保输出目录存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 填充要素中内部的空隙（孔洞）
def fill_holes(gdf):
    filled_geometries = []
    for geom in gdf.geometry:
        if geom is None:
            filled_geometries.append(geom)
            continue
        # 对于单一多边形
        if geom.geom_type == 'Polygon':
            filled_geometries.append(Polygon(geom.exterior))
        # 对于多多边形（MultiPolygon）
        elif geom.geom_type == 'MultiPolygon':
            new_polys = [Polygon(poly.exterior) for poly in geom.geoms]
            filled_geometries.append(unary_union(new_polys))
        else:
            filled_geometries.append(geom)
    gdf['geometry'] = filled_geometries
    return gdf

# 平滑边界（用多边形的外轮廓来近似平滑）
def smooth_boundaries(gdf):
    smoothed_geometries = []
    for geom in gdf.geometry:
        if isinstance(geom, Polygon):
            smoothed_geom = orient(geom)
            smoothed_geometries.append(smoothed_geom)
        else:
            smoothed_geometries.append(geom)
    gdf['geometry'] = smoothed_geometries
    return gdf

# 处理一个shp文件
def process_shp(file_path, output_path):
    try:
        # 读取shp文件
        gdf = gpd.read_file(file_path)
        # 保存原始坐标系（CGCS2000，通常EPSG:4490）
        orig_crs = gdf.crs

        # 使用一个适合面积计算的等积投影（例如 Asia North Albers Equal Area Conic）
        # 这里使用 ESRI:102012 作为示例（如果有其他更适合的投影，可进行修改）
        equal_area_crs = "ESRI:102012"

        # 临时转换到投影坐标系计算面积
        gdf_proj = gdf.to_crs(equal_area_crs)

        # 根据投影坐标系的面积过滤掉小面积的要素
        filtered_index = gdf_proj[gdf_proj.geometry.area >= MIN_AREA].index
        # 筛选原始数据中符合条件的要素（保持原始的几何和坐标系）
        gdf = gdf.loc[filtered_index]

        # 继续处理：填充孔洞和边界平滑
        gdf = fill_holes(gdf)
        gdf = smooth_boundaries(gdf)

        # 确保输出文件仍使用原始的CGCS2000坐标系
        gdf = gdf.to_crs(orig_crs)

        # 保存处理后的shp文件
        gdf.to_file(output_path)
        print(f"Successfully processed and saved: {output_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# 批量处理文件夹中的所有shp文件
def batch_process_shp(input_folder, output_folder):
    shp_files = [f for f in os.listdir(input_folder) if f.endswith('.shp')]
    for shp_file in shp_files:
        input_file = os.path.join(input_folder, shp_file)
        output_file = os.path.join(output_folder, shp_file)
        print(f"Processing {input_file}...")
        process_shp(input_file, output_file)

# # 执行批量处理
# batch_process_shp(input_folder, output_folder)
