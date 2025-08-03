"""Entry point for the toolbox package."""

from toolbox.pipeline import Pipeline

if __name__ == "__main__":
    pipeline = Pipeline("config.yaml")
    pipeline.run()

# data = pipeline._context['data']
#
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('tkagg')
# fig, ax = plt.subplots()
# ax1 = ax.twinx()
# ax.plot(data['TIME'])
# ax1.plot(data['TIME_QC'], color='red')
# plt.show(block=True)

# import geopandas
# import matplotlib
# from shapely.geometry import Polygon
# import shapely
#

# x, y = data['LONGITUDE'].values, data['LATITUDE'].values
# world = geopandas.read_file(r"C:\Users\banga\Downloads\110m_cultural\ne_110m_admin_0_countries.dbf")
# land_polygon = shapely.ops.unary_union(world.geometry)
# Red_Sea = shapely.geometry.Polygon([(40, 10), (45, 14), (35, 30), (30, 30), (40, 10)])  # (lon, lat)
# Med_Sea = Polygon([(-6, 30), (40, 30), (35, 40), (20, 42), (15, 50), (-5, 40), (-6, 30)])
# red_sea_mask = shapely.contains_xy(Red_Sea, x, y)
# med_sea_mask = shapely.contains_xy(Med_Sea, x, y)
# land_mask = ~shapely.contains_xy(land_polygon, x, y)
#
# world.plot(figsize=(16,8))
# plt.scatter(x, y, c='r')
#
# plt.scatter(x[land_mask], y[land_mask], c='b')
# for poly in list(land_polygon.geoms):
#     poly_x, poly_y = poly.exterior.xy
#     plt.fill(poly_x, poly_y, alpha=0.2, color='m')
#
# # for mask, col in zip([red_sea_mask, med_sea_mask], ['c', 'y']):
# #     plt.scatter(x[mask], y[mask], c=col)
# # for poly in [Red_Sea, Med_Sea]:
# #     poly_x, poly_y = poly.exterior.xy
# #     plt.fill(poly_x, poly_y, alpha=0.2, color='m')
# plt.show(block=True)

