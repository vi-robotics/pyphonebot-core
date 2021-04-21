"""
Workspace Characterization.
"""

import numpy as np

try:
    import cv2
except ImportError as e:
    print('Skipping OpenCV Import : {}'.format(e))

from shapely import geometry
import shapely.ops
from shapely.ops import cascaded_union, polygonize
from shapely.geometry import Point, Polygon, MultiPoint, LineString, MultiLineString

from phonebot.core.common.math.transform import Position
from phonebot.core.common.config import PhonebotSettings


def grid_max_rect(data):
    """
    Maximum Inscribed Rectangle in a Grid.

    Reference:
        http://stackoverflow.com/a/30418912/5008845
    """
    nrows, ncols = data.shape
    w = np.zeros(dtype=np.int32, shape=data.shape)
    h = np.zeros(dtype=np.int32, shape=data.shape)
    skip = 1

    best_area = 0.0
    best_rect = np.zeros(dtype=np.int32, shape=4)

    for r in range(nrows):
        # print('proc {}/{}'.format(r, nrows))
        for c in range(ncols):
            if data[r][c] == skip:
                continue
            if r == 0:
                h[r][c] = 1
            else:
                h[r][c] = h[r - 1][c] + 1
            if c == 0:
                w[r][c] = 1
            else:
                w[r][c] = w[r][c - 1] + 1
            minw = w[r][c]
            for dh in range(h[r][c]):
                minw = min(minw, w[r - dh][c])
                area = (dh + 1) * minw
                if area > best_area:
                    best_area = area
                    best_rect[:] = (r - dh, c - minw + 1, r, c)

    return (best_area, best_rect)


def max_rect(poly: Polygon, resolution: int):
    """
    Maximum Incribed Rectangle within a closed polygon.

    Parameters:
        poly : Bounding hull (shapely.Polygon)
        resolution : The resolution at which to rasterize the polygon.
    """
    # rasterize polygon -> find minimum inscribed rectangle
    mnx, mny, mxx, mxy = poly.bounds
    x, y = np.float32(poly.exterior.coords.xy)
    xsc = (x - mnx) * resolution
    ysc = (y - mny) * resolution

    w = int(np.ceil((mxx - mnx) * resolution))
    h = int(np.ceil((mxy - mny) * resolution))

    grid = np.zeros(shape=(h, w), dtype=np.uint8)
    cntr = np.stack([xsc, ysc], axis=-1)

    cv2.drawContours(grid,
                     [cntr.astype(np.int32)],
                     0,
                     (255, 0, 0),
                     -1)
    _, rect = grid_max_rect(grid == 0)
    ry0, rx0, ry1, rx1 = np.array(rect).ravel() / resolution
    rx0 += mnx
    rx1 += mnx
    ry0 += mny
    ry1 += mny
    return [rx0, ry0], [rx1, ry1]


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points. Intended for workspace approximation.

    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    # NOTE(ycho): Internal import to prevent scipy dependency
    # (which is not importable in kivy/android)
    from scipy.spatial import ConvexHull, Delaunay

    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])
    coords = np.array([point.coords[0]
                       for point in points])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = np.sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2)
        b = np.sqrt((pb[0] - pc[0])**2 + (pb[1] - pc[1])**2)
        c = np.sqrt((pc[0] - pa[0])**2 + (pc[1] - pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c) / 2.0
        # Area of triangle by Heron's formula
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        # Here's the radius filter.
        # print circum_r
        if circum_r < 1.0 / alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points


def workspace_from_endpoints(endpoints, alpha=100.0):
    """ alpha-shape approximation of feasible workspace from endpoints """
    workspace_hull, edge_points = alpha_shape(MultiPoint(endpoints), alpha)
    rect = max_rect(workspace_hull, 2000.0)
    hull_xy = np.transpose(workspace_hull.exterior.coords.xy)
    rep_pt = workspace_hull.representative_point().xy
    return hull_xy, rect, rep_pt


def get_workspace(buffer_radius: float = 0.0,
                  config: PhonebotSettings = PhonebotSettings(),
                  return_poly=False):
    """
    Analytical phonebot workspace.
    """
    small_radius = config.knee_link_length - config.hip_link_length
    large_radius = config.knee_link_length + config.hip_link_length

    min_circle_a = Point(config.hip_joint_offset,
                         0).buffer(small_radius)
    min_circle_b = Point(-config.hip_joint_offset,
                         0).buffer(small_radius)

    max_circle_a = Point(config.hip_joint_offset,
                         0).buffer(large_radius)
    max_circle_b = Point(-config.hip_joint_offset,
                         0).buffer(large_radius)
    outer = max_circle_a.intersection(max_circle_b)
    inner = min_circle_a.union(min_circle_b)

    workspace = outer.symmetric_difference(inner)

    # Only return lower part of the workspace.
    # (NOTE(yycho0108) : since `y` points downwards, argmax retrieves the lower part.)
    ws_parts = shapely.ops.split(
        workspace, LineString([(-1, 0), (1, 0)]))
    workspace = ws_parts[np.argmax([part.centroid.y for part in ws_parts])]
    workspace = workspace.buffer(buffer_radius)

    if return_poly:
        return workspace

    # Extract and convert points to leg origin frame.
    x, y = workspace.exterior.xy
    boundary = []
    for x, y in zip(*workspace.exterior.xy):
        boundary.append([x, y, 0])
    boundary = Position(boundary)
    return boundary
