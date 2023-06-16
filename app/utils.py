import base64

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx

import matplotlib.colors as mcolors

from io import BytesIO
from shapely.geometry import shape, MultiPoint, MultiPolygon, Polygon, LineString
from shapelysmooth import taubin_smooth
from scipy.spatial.distance import cdist

import rasterio
from rasterio import features
from rasterio.transform import Affine
from rasterio.io import MemoryFile
from rasterio.mask import mask

from PIL import Image

PALETTE = {
    'red': [255, 75, 75],
    'white': [255, 255, 255],
    'yellow': [255, 249, 127],
    'teal': [0, 108, 103],
    'beige': [255, 235, 198],
    'orange': [255, 177, 0],
    'blue': [0, 56, 68],
    'cyan': [0, 148, 198],
    'green': [46, 204, 113]
}


def distance_between_line_and_point(line, point, interval: int = 10):
    """
    Calculate the ~shortest distance between a line and a point. Geometries must be planar/projected.
    :param line: LineString geometry
    :param point: Point geometry
    :param interval: Interval in meters to split the line
    :return: Distance in meters
    """

    cut_dists = np.arange(0, line.length, interval)
    cut_points = MultiPoint([line.interpolate(cut_dist) for cut_dist in cut_dists] + [line.boundary[1]])


def get_gdf_corners(gdf):
    xmin, ymin, xmax, ymax = gdf.envelope.total_bounds
    corners = [
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin]
    ]

    return corners


def plug_shape_holes(geom):
    if geom.geom_type.lower() == 'multipolygon':
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms]).buffer(0)
    if geom.geom_type.lower() == 'polygon':
        return Polygon(geom.exterior)


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
    # normalized_values = (node_dists['dist'] - node_dists['dist'].min()) / (
    #         node_dists['dist'].max() - node_dists['dist'].min())
    # node_dists['color_hex'] = normalized_values.apply(lambda x: mcolors.to_hex(colormap(x)))
    # node_dists['color_rgb'] = normalized_values.apply(
    #     lambda x: eval(str(list((np.array(mcolors.to_rgb(colormap(x))) * 255).astype(np.uint8)))))

    return node_dists


def idw(node_vals, ref_gdf, target_resolution, p=3, clip_df=True):
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

    # Calculate the distance matrix between the points and grid points
    distances = cdist(points, np.c_[grid_x.ravel(), grid_y.ravel()])

    # Perform IDW interpolation
    weights = 1 / np.power(distances, p)
    # weights = 1 / distances
    interpolated_elevation = np.sum(weights * vals[:, np.newaxis], axis=0) / np.sum(weights, axis=0)

    # Reshape the interpolated elevation values to match the grid shape
    interpolated_elevation = interpolated_elevation.reshape(grid_x.shape)

    # Create an in-memory raster using rasterio
    height, width = interpolated_elevation.shape
    # transform = from_origin(xmin, ymax, (xmax - xmin) / width, (ymax - ymin) / height)
    transform = Affine.translation(xmin - target_resolution / 2, ymin - target_resolution / 2) * Affine.scale(
        target_resolution, target_resolution)

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
            dataset.write(interpolated_elevation[::-1], 1)

        # Read the dataset right away
        src = memfile.open(**profile)

        # Unmasked image
        img = src.read(1)

        # Masked image
        if clip_df:
            img_masked, transform = mask(src, shapes, crop=True, filled=False)

    if clip_df:
        return img_masked, src
    else:
        return img, src


def colorize_image(img, colormap):
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

    for contour in features.shapes(img_[::-1], transform=ref_ds.transform):
        contour_geom, contour_val = contour
        # geoms.append(shape(contour_geom).boundary.simplify(target_resolution, preserve_topology=True))
        geoms.append(shape(contour_geom))
        contour_vals.append(contour_val)

    # Create GeoDataFrame
    geoms = gpd.GeoSeries(geoms, crs=ref_ds.crs)
    contour_vals = pd.Series(contour_vals, name='contour')
    contours = gpd.GeoDataFrame({'contour': contour_vals, 'geometry': geoms}, geometry='geometry')

    # Filter
    contours = contours[contours.contour.isin(levels)].sort_values('contour').reset_index(drop=True)
    contours = contours.dissolve('contour')

    # Get correct ring
    def get_correct_ring(g):
        if g.geom_type.lower() == 'multipolygon':
            idx_max = np.array([x.area for x in g.geoms]).argmax()
            interiors = g.geoms[idx_max].interiors

            if len(interiors) > 1:

                idx_max = np.array([x.length for x in interiors]).argmax()
                ring = interiors[int(idx_max)]

                return LineString(ring)
            else:
                return g.geoms[idx_max].boundary
        else:
            interiors = g.interiors

            if len(interiors) > 1:

                idx_max = np.array([x.length for x in interiors]).argmax()
                ring = interiors[int(idx_max)]

                return LineString(ring)
            else:
                return g.boundary

    contours.geometry = contours.geometry.apply(get_correct_ring)

    # Smooth the contours
    contours.geometry = contours.geometry.apply(lambda x: taubin_smooth(x))

    # Reset index and colorize
    contours.reset_index(inplace=True)
    contours = add_color_to_df(contours, 'contour', colormap)

    return contours.to_crs(4326)
