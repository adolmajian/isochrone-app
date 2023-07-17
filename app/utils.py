import base64
import warnings

warnings.filterwarnings(action='ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import pydeck as pdk

import matplotlib.colors as mcolors

from io import BytesIO
from shapely.geometry import shape, MultiPoint, MultiPolygon, Polygon, LineString, MultiLineString
from shapelysmooth import taubin_smooth
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from rasterio.io import MemoryFile
from rasterio.mask import mask

from PIL import Image

PALETTE = {
    'red': [255, 75, 75],
    'white': [255, 255, 255],
    'yellow': [255, 249, 127],
    'teal': [0, 108, 103],
    'aquamarine': [165, 255, 214],
    'melon': [255, 166, 158],
    'beige': [255, 235, 198],
    'orange': [255, 177, 0],
    'purple': [167, 7, 179],
    'onyx': [46, 53, 50],
    'blue': [0, 56, 68],
    'cyan': [0, 148, 198],
    'purple': [136, 67, 204],
}


def get_gdf_corners(gdf):
    xmin, ymin, xmax, ymax = gdf.envelope.total_bounds
    corners = [
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin]
    ]

    return corners


def triangulation_from_points(df):
    # Check gdf crs
    if df.crs.to_epsg() == 4326:
        utm_crs = df.estimate_utm_crs()
        df = df.to_crs(utm_crs)

    # Get points from gdf
    pts = list(df.geometry.apply(lambda p: (p.x, p.y)))

    # Compute Delaunay
    tri = Delaunay(pts)

    # Construct triangles (LineStrings)
    points = tri.points
    triangles = []
    for simplex in tri.simplices:
        vertices = [points[x] for x in simplex]
        triangles.append(LineString(vertices))

    return MultiLineString(triangles)


def plug_shape_holes(geom):
    if geom.geom_type.lower() == 'multipolygon':
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms]).buffer(0)
    if geom.geom_type.lower() == 'polygon':
        return Polygon(geom.exterior)


def extract_exteriors(g):
    if g.geom_type.lower() == 'multipolygon':
        exteriors = [x.exterior for x in g.geoms]
        return MultiLineString(exteriors)
    else:
        return LineString(g.exterior)


def add_color_to_df(df, col, colormap):
    df = df.copy()
    normalized_values = (df[col] - df[col].min()) / (
            df[col].max() - df[col].min())
    df['color_hex'] = normalized_values.apply(lambda x: mcolors.to_hex(colormap(x)))
    df['color_rgb'] = normalized_values.apply(
        lambda x: eval(str(list((np.array(mcolors.to_rgb(colormap(x))) * 255).astype(np.uint8)))))

    return df


def prepare_interpolation_points(subgraph, start_node, acc_nodes, colormap):
    node_dists = nx.single_source_dijkstra_path_length(subgraph, start_node, weight='length')
    node_dists = pd.DataFrame(data=[(k, v) for k, v in node_dists.items()], columns=['osmid', 'dist']).set_index(
        'osmid')
    node_dists = gpd.GeoDataFrame(node_dists.merge(acc_nodes, left_index=True, right_index=True), geometry='geometry',
                                  crs=acc_nodes.crs)

    # Add color columns
    node_dists = add_color_to_df(node_dists, 'dist', colormap)

    return node_dists


def grid_interpolate(node_vals, ref_gdf, target_resolution, algo, p=3, clip_df=True):
    # Get crs
    ref_crs = ref_gdf.crs

    # Get reference to coordinates and distance values from nodes
    points = node_vals.to_crs(ref_crs).geometry.values
    points = [(p.x, p.y) for p in points]
    vals = node_vals['dist'].values

    # Calculate the minimum and maximum bounds of the GeoDataFrame
    xmin, ymin, xmax, ymax = ref_gdf.total_bounds

    # Calculate the number of pixels in the x and y directions based on the target resolution
    num_pixels_x = int((xmax - xmin) / target_resolution)
    num_pixels_y = int((ymax - ymin) / target_resolution)

    # Create a regular grid of points for interpolation based on the target resolution
    grid_x, grid_y = np.meshgrid(np.linspace(xmin, xmax, num_pixels_x), np.linspace(ymin, ymax, num_pixels_y))

    if algo == 'idw':
        # Calculate the distance matrix between the points and grid points
        distances = cdist(points, np.c_[grid_x.ravel(), grid_y.ravel()])

        # Perform IDW interpolation
        weights = 1 / np.power(distances, p)
        # weights = 1 / distances
        interpolated_elevation = np.sum(weights * vals[:, np.newaxis], axis=0) / np.sum(weights, axis=0)

        # Reshape the interpolated elevation values to match the grid shape
        interpolated_elevation = interpolated_elevation.reshape(grid_x.shape)

    if algo == 'tin':
        # Perform TIN interpolation using LinearNDInterpolator
        interpolator = LinearNDInterpolator(points, vals)
        interpolated_elevation = interpolator(grid_x, grid_y)

        # Reshape the interpolated elevation values to match the grid shape
        interpolated_elevation = interpolated_elevation.reshape(grid_x.shape)

    # Create an in-memory raster using rasterio
    height, width = interpolated_elevation.shape
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)  # west, south, east, north

    # Prepare mask geometry (will need to be inverted)
    if clip_df:
        shapes = [ref_gdf.values[0]]

    # Profile
    profile = {
        # 'driver': 'MEM',
        'driver': 'GTiff',
        'dtype': rasterio.float32,
        'count': 1,
        'width': width,
        'height': height,
        'transform': transform,
        'crs': ref_crs
    }

    # Write the interpolated elevation values to the in-memory raster
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            interpolated_elevation = interpolated_elevation[::-1]
            # interpolated_elevation = interpolated_elevation
            dataset.write(interpolated_elevation, 1)

        # Read the dataset right away
        src = memfile.open(**profile)

        # Unmasked image
        img = src.read(1)

        # Masked image
        if clip_df:
            img_masked, transform = mask(src, shapes, crop=True, filled=False)

    if clip_df:
        return img_masked, src, transform
    else:
        return img, src


def colorize_image(img, colormap):
    # Correct the mask (add nodata values from data arr to mask arr)
    img.mask[np.isnan(img.data)] = True

    # Convert the masked array to a regular array by filling the mask with 0s
    img_arr = np.ma.filled(img, fill_value=0).squeeze()

    # Normalize the data to the range [0, 1]
    arr_norm = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())

    # Apply the colormap to the normalized data
    arr_rgba = colormap(arr_norm)

    # Convert to byte
    arr_rgba = (arr_rgba[:, :, :4] * 255).astype(np.uint8)

    # Correct the alpha band values
    arr_rgba[:, :, 3:][img.mask.squeeze()] = 0

    # Build PIL image
    img_rgba = Image.fromarray(arr_rgba)

    return img_rgba


def convert_image_to_bytes_url(img):
    im_file = BytesIO()
    img.save(im_file, format="png")
    im_bytes = im_file.getvalue()  # image in binary format
    im_b64 = str(base64.b64encode(im_bytes))[2:-1]
    im_url = fr'"data:image/png;base64,{im_b64}"'

    return im_url


def extract_contours_from_singleband_raster(img, ref_ds, distance, interval, colormap):
    # Create the levels list
    # levels = list(range(0, distance, interval)) + [distance]
    levels = list(range(0, distance, interval))
    levels_ = [(levels[i], levels[i + 1]) for i in range(len(levels) - 1)]

    # Discreetize the raster
    img_ = img.copy()
    for (low, high) in levels_:
        img_[(img_ > low) & (img_ <= high)] = high

    # Extract features (connected component geometries) for every pixel value
    geoms = []
    contour_vals = []

    for contour in features.shapes(img_, transform=ref_ds.transform):
        contour_geom, contour_val = contour
        geoms.append(shape(contour_geom))
        contour_vals.append(contour_val)

    # Create GeoDataFrame
    geoms = gpd.GeoSeries(geoms, crs=ref_ds.crs)
    contour_vals = pd.Series(contour_vals, name='contour')
    contours = gpd.GeoDataFrame({'contour': contour_vals, 'geometry': geoms}, geometry='geometry')

    # Filter
    contours = contours[contours.contour.isin(levels)].sort_values('contour').reset_index(drop=True)
    contours = contours.dissolve('contour')
    contours.reset_index(inplace=True)

    def smooth(l):
        if l.geom_type.lower() == 'multilinestring':
            return MultiLineString([taubin_smooth(l_) for l_ in l.geoms])
        else:
            return taubin_smooth(l)

    contours_ = contours.copy()
    contours.geometry = contours.geometry.apply(extract_exteriors)

    # Smooth the contours
    contours.geometry = contours.geometry.apply(lambda x: smooth(x))

    # Colorize
    contours = add_color_to_df(contours, 'contour', colormap)

    return contours.to_crs(4326), contours_


def prepare_buffer_layer(buffer):
    return pdk.Layer(
        type="GeoJsonLayer",
        data=buffer,
        get_fill_color=PALETTE['red'] + [25],
        get_line_color=PALETTE['red'] + [200],
        line_width_max_pixels=1,
        stroked=True,
        filled=True,
    )


def prepare_source_layer(source):
    return pdk.Layer(
        type="GeoJsonLayer",
        data=source,
        get_radius=6,
        get_fill_color=PALETTE['yellow'] + [200],
        get_line_color=PALETTE['white'] + [100],
        line_width_max_pixels=3,
        stroked=True,
        pickable=True,
        auto_highlight=True,
    )


def prepare_start_point_layer(start_point):
    return pdk.Layer(
        type="GeoJsonLayer",
        data=start_point,
        get_radius=6,
        get_fill_color=(255, 137, 93, 255),
        get_line_color=PALETTE['white'] + [100],
        line_width_max_pixels=3,
        stroked=True,
        pickable=True,
        auto_highlight=True,
        # tooltip=True
    )


def prepare_edges_layer(edges):
    return pdk.Layer(
        type="GeoJsonLayer",
        data=edges,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
        stroked=True,
        filled=True,
        pickable=True,
        auto_highlight=True,
        # tooltip=True
    )


def prepare_acc_edges_layer(acc_edges):
    return pdk.Layer(
        type="GeoJsonLayer",
        data=acc_edges,
        get_line_color=[255, 75, 75],
        line_width_min_pixels=2,
        stroked=True,
        filled=True,
        pickable=True,
        auto_highlight=True,
        # tooltip=True
    )


def prepare_nodes_layer(nodes):
    return pdk.Layer(
        type="GeoJsonLayer",
        data=nodes,
        get_radius=3,
        # get_fill_color=[255, 75, 75],
        get_fill_color=[255, 255, 255],
        get_line_color=[255, 255, 255, 100],
        line_width_max_pixels=3,
        stroked=True,
        # filled=True,
        pickable=True,
        auto_highlight=True,
        # tooltip=True
    )


def prepare_acc_nodes_layer(acc_nodes):
    return pdk.Layer(
        type="GeoJsonLayer",
        data=acc_nodes,
        get_radius=3,
        get_fill_color=[255, 75, 75],
        # get_fill_color=[255, 255, 255],
        get_line_color=[255, 255, 255, 100],
        line_width_max_pixels=3,
        stroked=True,
        # filled=True,
        pickable=True,
        auto_highlight=True,
        # tooltip=True
    )
