import numpy as np
from shapely.geometry import MultiPoint

PALETTE = {
    'red': [255, 75, 75],
    'white': [255, 255, 255],
    'yellow': [255, 249, 127],
    'teal': [0, 108, 103],
    'beige': [255, 235, 198],
    'orange': [255, 177, 0],
    'blue': [0, 56, 68]
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

