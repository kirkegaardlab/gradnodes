from collections import namedtuple
import argparse
from datetime import datetime
from functools import partial
import itertools
import json
import pathlib
import jax
import jax.numpy as jnp
import numba as nb
import numpy as np
import optax
from scipy.spatial import Delaunay, Voronoi
from tqdm import tqdm
from simple_pytree import Pytree, dataclass

jax.config.update("jax_enable_x64", True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.5, help="The exponent of the power dissipation.")
    parser.add_argument("--n_nodes", type=int, default=100, help="The number of nodes in the network.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-2, help="The learning rate of the optimizer.")
    parser.add_argument("--init_noise", "-in", type=float, default=0.0, help="Initial noise to the positions")
    parser.add_argument("--num_iters", type=int, default=10_000, help="The number of iterations.")
    parser.add_argument("--rtol", type=float, default=1e-8, help="Relative tolerance for convergence.")
    parser.add_argument("--atol", type=float, default=1e-10, help="Absolute tolerance for convergence.")
    parser.add_argument("--beta", type=float, default=0.4, help="The β parameter of the leaf.")
    parser.add_argument("--theta", type=float, default=0.0, help="The rotation of the leaf.")
    parser.add_argument("--save_interval", type=int, default=10, help="Interval to save the network.")
    parser.add_argument("--out", type=str, default="runs", help="Output folder to save the results.")
    parser.add_argument("--name", type=str, default="network", help="Name of the run.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator.")
    args = parser.parse_args()
    return args


def find_voronoi_nodes(points):
    (ax, ay), (bx, by), (cx, cy) = points
    D = 2 * ((ax - cx) * (by - cy) - (bx - cx) * (ay - cy))
    ux = (ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)
    uy = (ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)
    return jnp.stack((ux, uy)) / jnp.where(D == 0, 1, D)


def is_point_inside_poly(point, poly, mask):
    """Winding number algorithm."""

    # Shift the polygon to the left
    vertices_x, vertices_y = poly.T
    rev_vertices_x, rev_vertices_y = jnp.roll(poly, -1, axis=0).T
    (x, y) = point

    # A valid edge is one without invalid vertices
    valid_edge = jnp.logical_and(mask, jnp.roll(mask, -1, axis=0))

    # Calculate every time the point crosses an edge.
    y_cross = (vertices_y > y) != (rev_vertices_y > y)
    y_edge = jnp.where(y_cross, (rev_vertices_y - vertices_y), 1.0)
    x_cross = x < ((rev_vertices_x - vertices_x) * (y - vertices_y) / y_edge + vertices_x)

    # If the point crosses an edge, add or subtract 1 to the winding number.
    # If 1 at the end, the point is inside the polygon.
    is_inside = (jnp.sum(valid_edge * (x_cross & y_cross)) % 2) == 1
    return is_inside


are_points_inside_poly = jax.vmap(is_point_inside_poly, in_axes=(0, None, None))


def line_intersect(A, B, C, D, mask):
    (x1, y1), (x2, y2) = A, B

    denom = jnp.cross(B - A, D - C)
    cond = denom != 0
    denom = jnp.where(cond, denom, 1.0)

    ua = jnp.cross(D - C, A - C) / denom
    cond = jnp.where((ua < 0) | (ua > 1), False, cond)

    ub = jnp.cross(B - A, A - C) / denom
    cond = jnp.where((ub < 0) | (ub > 1), False, cond)

    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)

    cond = cond & mask
    return jnp.where(cond, jnp.stack((x, y, 1)), jnp.array([0, 0, 0]))


def find_all_points(subject_polygon, until, clipping_polygon):
    mask = jnp.arange(len(subject_polygon)) < until

    subject_inside = are_points_inside_poly(
        subject_polygon, clipping_polygon, jnp.ones(len(clipping_polygon), dtype=mask.dtype)
    )
    subject_inside = jnp.logical_and(subject_inside, mask)

    subject_polygon = jnp.concatenate((subject_polygon[until - 1][None], subject_polygon), axis=0)[:-1]
    mask2 = jnp.concatenate((jnp.ones(1, dtype=mask.dtype), mask))[:-1]
    clipping_inside = are_points_inside_poly(clipping_polygon, subject_polygon, mask2)

    subject_polygon_r = jnp.roll(subject_polygon, -1, axis=0)
    mask2 = jnp.logical_and(mask2, jnp.roll(mask2, -1, axis=0))
    clipping_polygon_r = jnp.roll(clipping_polygon, -1, axis=0)
    line_intersect_vmap = jax.vmap(
        jax.vmap(line_intersect, in_axes=(None, None, 0, 0, None)), in_axes=(0, 0, None, None, 0)
    )

    intersections = line_intersect_vmap(subject_polygon, subject_polygon_r, clipping_polygon, clipping_polygon_r, mask2)
    intersections = intersections.reshape((-1, 3))
    valid_intersection = intersections[:, 2] > 0.5
    intersections = intersections[:, :2]

    return subject_inside, clipping_inside, intersections, valid_intersection


def clip(region, until, boundary):
    region_inside, boundary_inside, intersections, valid_intersection = find_all_points(region, until, boundary)

    points = jnp.concatenate((region, boundary, intersections))
    p_valid = jnp.concatenate((region_inside, boundary_inside, valid_intersection))

    d = points - jnp.mean(points, where=p_valid[:, None], axis=0, keepdims=True)
    theta = -jnp.arctan2(d[:, 1], d[:, 0])
    idxs = jnp.argsort(theta - 10 * p_valid)

    points = points[idxs]
    p_valid = p_valid[idxs]

    _, unique_idx = jnp.unique(points, size=len(p_valid), return_index=True, axis=0, fill_value=-1)
    p_valid = jnp.zeros_like(p_valid).at[unique_idx].set(p_valid[unique_idx])
    return points, p_valid


def no_clip(region, until, boundary):
    intersections = jnp.zeros(shape=(len(region) * len(boundary), 2))
    points = jnp.concatenate((region, boundary, intersections))
    p_valid = jnp.arange(len(points)) < until
    return points, p_valid


@partial(jax.vmap, in_axes=(0, 0, None, 0))
def clip_all(region, until, boundary, mask):
    cs, cs_valid = jax.lax.cond(mask, clip, no_clip, region, until, boundary)
    return cs, cs_valid


@nb.njit
def ccw(A, B, C):
    # https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
    # NOTE(albert): This is just the cross product of (C-A) x (B-A)...
    return (C[..., 1] - A[..., 1]) * (B[..., 0] - A[..., 0]) > (B[..., 1] - A[..., 1]) * (C[..., 0] - A[..., 0])


@nb.njit
def intersect(A, B, C, D):
    return (ccw(A, C, D) != ccw(B, C, D)) & (ccw(A, B, C) != ccw(A, B, D))


@nb.njit(parallel=False)  # parallel can be set true for very large problems....
def needs_clipping(regions, points, clipping_polygon):
    ret = np.zeros(len(regions), dtype=nb.bool_)
    for i in nb.prange(len(regions)):
        x = points[regions[i][regions[i] >= 0]]
        for j in range(len(x)):
            for k in range(len(clipping_polygon)):
                a = intersect(
                    x[j], x[(j + 1) % len(x)], clipping_polygon[k], clipping_polygon[(k + 1) % len(clipping_polygon)]
                )
                if a:
                    ret[i] = True
                    break
            if ret[i]:
                break
    return ret


def voronoi_precalc_callback(x_ext, boundary, num_nodes, size=30, tri_size=100):
    # Calculate Voronoi using scipy QHull (somewhat efficient).
    # Voronoi regions are shape (num_nodes, pad_value) and always must end with a -2. (Possible bug later on)
    x_ext = np.asarray(x_ext)
    boundary = np.asarray(boundary)
    vor = Voronoi(x_ext)
    vor_regions = np.full((num_nodes, size), -2)
    for i, idx in enumerate(vor.point_region[:num_nodes]):
        vor_regions[i, : len(vor.regions[idx])] = vor.regions[idx]

    needs_clipping_mask = needs_clipping(vor_regions, vor.vertices, boundary)
    delaunay = Delaunay(x_ext)
    simplices = np.full((tri_size, 3), fill_value=-2, dtype=np.int32)
    simplices[: len(delaunay.simplices)] = delaunay.simplices.astype(np.int32)

    simplices = jnp.array(simplices)
    needs_clipping_mask = jnp.array(needs_clipping_mask)
    vor_regions = jnp.array(vor_regions)

    return needs_clipping_mask, vor_regions.astype(np.int32), simplices


def find_voronoi_regions(x, boundary):
    """Given a set of points and a boundary, calculate the clipped voronoi network."""

    num_nodes = len(x)

    vor_size = len(boundary) + (len(x) - 1)
    tri_size = (len(x)+len(boundary)) * 10

    # We extend `x` by "100 * boundary" to avoid issues of infinite voronoi regions and ghost points.
    extended_boundary = boundary + (boundary - jnp.mean(x, axis=0, keepdims=True)) * 100
    x_ext = jnp.concatenate((x, extended_boundary))

    # Some of the steps here do not need to be differentiable, so we can do a first pass with scipy/numpy/numba
    # and use the results to directly get the gradients with Jax.
    # Figure out which Voronoi regions need clipping (border cells)
    # Calculate delaunay to recalculate Voronoi using Jax on the triangles.
    xx = jax.lax.stop_gradient(x_ext)
    out_eval_shape_dtype = (
        jax.ShapeDtypeStruct((num_nodes,), bool),
        jax.ShapeDtypeStruct((num_nodes, vor_size), np.int32),
        jax.ShapeDtypeStruct((tri_size, 3), np.int32),
    )
    callback_outs = jax.pure_callback( # pyright: ignore
        voronoi_precalc_callback, out_eval_shape_dtype, xx, boundary, num_nodes, vor_size, tri_size
    )
    needs_clipping_mask, vor_regions, delaunay_simplices = callback_outs

    # With the callbacks results, continue building the Voronoi network.
    nodes = jax.vmap(find_voronoi_nodes)(x_ext[delaunay_simplices])

    # Clip regions. Evaluate when the -2 start appearing.
    mask = vor_regions >= 0
    until = jnp.sum(mask, axis=1)

    # Pad the regions with -2 to avoid issues with the clipping.
    # Clips the polygons that need clipping and pad the other ones.
    nodes_regions = jnp.take(nodes, vor_regions, axis=0)
    nodes_regions, mask_regions = clip_all(nodes_regions, until, boundary, needs_clipping_mask)

    # Find the last valid index of the clipped regions, so that we can crop useless values (up to a rounding).
    # This should be more efficient than just brute forcing all (~ not convinced, maybe removing 2*len(boundary)?).
    max_region_size = vor_size
    nodes_regions = nodes_regions[:, :max_region_size]
    mask_regions = mask_regions[:, :max_region_size]
    return nodes_regions, mask_regions


def split_into_triangles(polygon, center, mask):
    """
    Finds the three vertices of each the triangles inside the cell from the center x to the voronoi nodes.
    The polygon has (n, 2) shape but mask indicates which of those are valid.
    Returns the triangles with (n, 3, 2) shape.
    """
    last_element = jnp.argmin(mask) # This is the last valid element of the polygon.
    first_triangle = jnp.stack((center, polygon[last_element - 1], polygon[0]), axis=0)[:, None]
    first_vertex = center[None, :].repeat(len(polygon), axis=0)
    rest_vertices = jnp.stack((first_vertex, polygon, jnp.roll(polygon, -1, axis=0)), axis=0)[:, :-1]
    triangles = jnp.concatenate((first_triangle, rest_vertices), axis=1).swapaxes(0, 1)
    triangles = mask[:, None, None] * triangles
    return triangles


def integrate_triangle(vertices, mask):
    """
    Given three vertices shape (3,2) of a triangle, calculate the area x length of the triangle.
    If the mask is False, then the vertices are the same value and should not be integrated.

    Deriviation in Supplemental Material at [TODO(albert): add link to SI]

    Returns:
        - The average length of the triangle.
        - The area of the triangle.
    """
    dummy_vertices = jnp.array([[0., 0.], [1., 1.], [1., 0.]])
    A, B, C = vertices + (1 - mask) * dummy_vertices

    # Rotate the triangle so that B and C are always to the right of A
    M = (B+C)/2 - A
    c, s = M / jnp.linalg.norm(M)
    rotation_matrix = jnp.array([[c, s], [-s, c]])
    B = A + rotation_matrix @ (B - A)
    C = A + rotation_matrix @ (C - A)

    # Calculate the analytical solution for the integral of length x area
    I0 = 1/3 * jnp.cross(A-C, B-C)**3
    (cos1, sin1) = (B - A) / jnp.linalg.norm(B - A)
    (cos2, sin2) = (C - A) / jnp.linalg.norm(C - A)
    dx, dy = C - B
    d = jnp.linalg.norm(C - B)

    def integral(cos, sin):
        """Integral solution in terms of cos(theta) and sin(theta)"""
        tanhalf = sin / (1 + cos)
        term1 = jnp.arctanh((dx + dy * tanhalf) / d) / d**3
        term2 = (dx * cos + dy * sin) / (2 * d**2 * (dy * cos - dx * sin) ** 2)
        return term1 + term2

    LA = jnp.abs(I0 * (integral(cos2, sin2) - integral(cos1, sin1)))
    area = jnp.abs(jnp.cross(A-C, B-C) / 2)

    return LA * mask, area * mask


def integrate_cell(triangles, mask):
    """Given the triangles of a Voronoi cell (#, 3, 2), calculate the average length to area ratio of the cell."""
    weighted_length, areas = jax.vmap(integrate_triangle)(triangles, mask)
    cell_area = jnp.sum(areas)
    average_length = jnp.sum(weighted_length) / cell_area
    return average_length, cell_area


def calc_sinks(x, boundary, sources, sink_fluctuation=0.0):
    # Calculate the voronoi network
    nodes_regions, mask = find_voronoi_regions(x, boundary)

    # Calculate the average length of each region and the area of the region using its triangles.
    triangles = jax.vmap(split_into_triangles)(nodes_regions, x, mask)
    avg_length, areas = jax.vmap(integrate_cell)(triangles, mask)
    normed_areas = areas / jnp.sum(areas)

    n_sources = len(sources)
    n_sinks = len(areas)

    # No sink fluctuation: S = column vector (n_sources + n_sinks, 1)
    if sink_fluctuation == 0:
        S_sources = jnp.ones((n_sources, 1)) / n_sources
        S_sinks = -normed_areas[:, None]
    # Non-zero sink fluctuation: S = matrix (n_sources + n_sinks, n_sinks)
    else:
        # Sources
        S_sources = jnp.ones((n_sources, n_sinks)) / n_sources

        # Sinks
        s_avg = (1 - sink_fluctuation) / n_sinks
        Cs = 1 / (sink_fluctuation * areas + s_avg * jnp.sum(areas))
        S_sinks = -areas[:,None] * Cs
        sfs = jnp.diag(sink_fluctuation * jnp.ones(n_sinks))
        sfs = sfs + s_avg * jnp.ones((n_sinks, n_sinks))
        S_sinks = S_sinks * sfs

    S = jnp.vstack((S_sources, S_sinks))

    return S, avg_length, normed_areas


def calc_flow_squared(C, L, S, B):
    C_eff = C / (L + 1e-5)
    T = -jnp.eye(S.shape[0], k=-1, dtype='float64')
    T = T.at[0].set(1.0)[:, :-1]
    A = (B * C_eff) @ B.T
    A_inv = jnp.linalg.inv(T.T @ A @ T)
    A_pinv = T @ A_inv @ T.T

    # sink_fluctuation = 0.0: S.shape = (n_sources + n_sinks, 1)
    # sink_fluctuation != 0.0: S.shape = (n_sources + n_sinks, n_sinks)
    ps = A_pinv @ S

    F = C_eff[:,None] * (B.T @ ps)
    F_squared = F**2
    # Produces same resulting array shape from matrix as from column vector
    # avg_F_squared.shape = (n_sources + n_sinks,)
    avg_F_squared = F_squared.mean(axis=1)

    return avg_F_squared


def calc_power(F_squared, C, L):
    return jnp.sum(F_squared / C * L)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def fixed_point(fun, x_init, i, args):

    def cond_fun(carry):
        x_prev, x, i = carry
        return jnp.any(jnp.abs(x - x_prev) > 1e-9) & (i <= 2000)

    def body_fun(carry):
        _, x, i = carry
        x_new = fun(x, i, args)
        return x, x_new, i + 1

    _, x, _ = jax.lax.while_loop(cond_fun, body_fun, (x_init, fun(x_init, i, args), i))
    return x


def fixed_point_fwd(fun, x, i, args):
    x = fixed_point(fun, x, i, args)
    return x, (x, i, args)


def fixed_point_bwd(fun, res, vT):
    # Get residuals
    x_star, i_last, diff_args = res

    def bwd_iter(u, _, args):
        # Corresponds to matrix A
        _, vjp_x = jax.vjp(lambda x: fun(x, i_last, args), x_star)
        uT_A = vjp_x(u)[0]
        return vT + uT_A

    _, vjp_args = jax.vjp(lambda args: fun(x_star, i_last, args), diff_args)

    # Solve wT = vT @ A
    w_guess = vT
    wT = fixed_point(bwd_iter, w_guess, 0, diff_args)

    # wT @ B
    args_bar = vjp_args(wT)[0]
    return jnp.zeros_like(x_star), 0, args_bar


fixed_point.defvjp(fixed_point_fwd, fixed_point_bwd)


def optimize_conductivity(C, B, L, S, gamma, dt, c0, metabolic_cost=0.1):
    iterations = 1000
    c0 = metabolic_cost

    def diff_step(C, i, args):
        B, L, S = args
        F_squared = calc_flow_squared(C, L, S, B)

        # Update C on the transport and sinks.
        C += dt * ((F_squared / C ** (1 + gamma) - gamma * c0) * C
                   + c0 * jnp.exp(-20 * i / iterations))
        return C

    C = fixed_point(diff_step, C, 0, (B, L, S))
    return C


def find_edge_lengths(x, edges):
    lenghts = jnp.sqrt(jnp.sum((x[edges[:, 0], :] - x[edges[:, 1], :]) ** 2, axis=-1))
    return lenghts


@dataclass
class GielisDomain(Pytree):
    beta: float
    theta: float

    def g(self, theta):
        phi = theta - self.theta
        return 1 / (jnp.abs(jnp.cos(phi/4)) + jnp.abs(jnp.sin(phi/4)))**(1/self.beta)

    def transform_in(self, x):
        theta = jnp.arctan2(x[...,1], x[...,0])
        r = jnp.linalg.norm(x, axis=-1)
        r_hat = r / (self.g(theta) - r)
        return r_hat, theta

    def transform_out(self, r_hat, theta):
        r = (r_hat * self.g(theta)) / (1 + r_hat)
        x = jnp.stack((r * jnp.cos(theta), r * jnp.sin(theta)), axis=-1)
        return x


def find_transport_network(r, theta, boundary, sources, edges, C, B, gamma, domain, ct, dt, sink_fluctuation=0.0, c_hat=1.0, c0=1e-3):
    # Transform the coordinates from polar to cartesian
    x = domain.transform_out(r, theta)

    # Find the length of the dynamic nodes (also with the sources)
    x_ext = jnp.concatenate((sources, x))
    L = find_edge_lengths(x_ext, edges)

    # Calculate the sink values of the network (alongside its average length)
    # sink_fluctuation = 0.0: S.shape = (n_sources + n_sinks, 1)
    # sink_fluctuation != 0.0: S.shape = (n_sources + n_sinks, n_sinks)
    S, sink_lengths, normed_areas = calc_sinks(x, boundary, sources, sink_fluctuation)

    # Minimize the power by optimizing the conductivities
    C = optimize_conductivity(C, B, L, S, gamma, dt, c0, metabolic_cost=ct)

    # Calculate the flow (remove the sink connections, although they are not a big chunk...)
    F2 = calc_flow_squared(C, L, S, B)

    # Calculate the power required to transport the flow
    P_transport = calc_power(F2, C, L)
    P_cost = jnp.sum(C**gamma * L) * ct
    P_delivery = 1.0 / c_hat * jnp.sum(sink_lengths * normed_areas**2)
    P = P_transport + P_cost + P_delivery
    return P, (P, C, normed_areas, P_transport, P_delivery, P_cost)


def create_network(n_nodes, beta, n_bpoints, theta=0.0, local=False, noise=0.0):
    # Define the boundary of the leaf using Geilis Equation
    phis = np.linspace(0, 2 * np.pi, n_bpoints + 1)[:-1]
    r = 1 / (np.abs(np.cos(phis / 4)) + np.abs(np.sin(phis / 4))) ** (1 / beta)
    boundary = np.stack((r * np.cos(phis), r * np.sin(phis)), axis=1)
    # TODO: fix this! -- This avoids voronoi region being computed wrong.
    boundary += 1e-4 * np.random.randn(*boundary.shape)

    # Define the sources
    source_idx = np.argmin(np.abs(phis - np.pi))
    rot_mat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    boundary = boundary @ rot_mat
    sources = boundary[source_idx][None]

    # Find the dx value to fit n_nodes on the leaf. Assume dy=dx/2 to have a diagonal grid.
    dx = np.sqrt(np.sqrt(4/3) * np.sum(r**2) * np.abs(phis[0] - phis[1]) / 2) * 0.95 / np.sqrt(n_nodes)
    n_attemps = 0
    while True:
        _x, _y = np.arange(-10, 10, dx), np.arange(-10, 10, dx * np.sqrt(3/4))
        xx, yy = np.meshgrid(_x, _y)
        xx[::2] += dx/2

        if noise > 0:
            xx += dx * noise * np.random.randn(*xx.shape)
            yy += dx * noise * np.random.randn(*yy.shape)

        X = np.stack((xx.ravel(), yy.ravel()), axis=1)

        # The source should be on top of a node, so we want to translate the grid to match it.
        # Find the closest node to the sources
        dists = np.linalg.norm(X - sources, axis=1)
        offset = X[np.argmin(dists)] - sources
        X = X - offset

        # Generate a grid of nodes with the sources at the beginning
        boundary_small = boundary * 0.95
        edges_boundary = np.diff(boundary_small, axis=0, append=boundary_small[:1])
        A = np.stack((-edges_boundary[:, 1], edges_boundary[:, 0]), axis=1)
        b = (A * boundary_small).sum(axis=1)
        mask_inside = np.all((A @ X.T) > b[:, None], axis=0)
        X = X[mask_inside]
        n_attemps += 1
        if len(X) == n_nodes - 1:
            break
        if len(X) < n_nodes - 1:
            dx *= 0.99
        else:
            dx *= 1.01
        if n_attemps > 1000:
            raise ValueError(f"Could not find a valid grid of nodes. {len(X)} nodes found.")


    #place a new node text to the source (poiting to the center)
    new_addition = jnp.array([0.01, 0.00]) @ rot_mat + sources
    X = np.concatenate((X, new_addition), axis=0)
    _X = np.concatenate((sources, X), axis=0)

    # The adjacency matrix tells us how many edges are between the nodes
    n_nodes = len(_X)
    if local:
        adjacency = np.zeros((n_nodes, n_nodes))
        distances = np.linalg.norm(_X[:, None] - _X[None], axis=-1)
        for i, j in itertools.product(range(n_nodes), repeat=2):
            if (i < j) and distances[i, j] < (3 * dx):
                adjacency[i, j] = 1
                adjacency[j, i] = 1
    else:
        adjacency = np.ones((n_nodes, n_nodes)) - np.identity(n_nodes)

    n_edges = int(np.sum(np.triu(adjacency)))

    # Calculate incident matrix
    B = np.zeros((n_nodes, n_edges))
    edges = np.zeros((n_edges, 2), dtype=int)
    e = 0
    for (v1, v2) in itertools.product(range(n_nodes), repeat=2):
        if (v1 < v2) and adjacency[v1, v2]:
            B[v1, e] = -1
            B[v2, e] = 1
            edges[e] = [v1, v2]
            e += 1

    return X, B, edges, boundary, sources


def main(args):

    # Set the seed for numpy's random number generator
    np.random.seed(args.seed)

    # Initialise the transport network, the sources positions and the boundary polygon.
    x, B, edges, boundary, sources = create_network(args.n_nodes, beta=args.beta, theta=args.theta, n_bpoints=50, noise=args.init_noise)

    # Create a folder to save results and remove previous runs
    outdir = pathlib.Path(args.out) / (datetime.now().strftime("%Y-%m-%d_%H%M%S") + f"_{args.name}")
    outdir.mkdir(exist_ok=True, parents=True)
    print("Results will be saved in:", outdir)
    (outdir / "params.json").write_text(json.dumps(vars(args), indent=4))
    arraydir = outdir / "arrays"
    arraydir.mkdir(exist_ok=True, parents=True)

    domain = GielisDomain(beta=args.beta, theta=args.theta)
    dt = 0.1

    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(domain.transform_in(x))

    ct =  0.1 * np.sqrt(len(x))
    c_hat = 1 / len(x)**2
    C = jnp.ones(len(edges))

    def init_power(x, edges, C, B):
        x_ext = jnp.concatenate((sources, x))
        L = find_edge_lengths(x_ext, edges)
        S, sink_lengths, normed_areas = calc_sinks(x, boundary, sources, 0.0)
        C = optimize_conductivity(C, B, L, S, args.gamma, dt, 0.1, metabolic_cost=ct)
        F2 = calc_flow_squared(C, L, S, B)
        P_transport = calc_power(F2, C, L)
        P_cost = jnp.sum(C**args.gamma * L) * ct
        P_delivery =  1/c_hat * jnp.sum(sink_lengths * normed_areas**2)
        return P_transport + P_cost +  P_delivery


    P0 = jax.jit(init_power)(x, edges, C, B)
    Powers = namedtuple("Powers", ["P", "Pt", "Pd", "Pc"])

    @jax.jit
    def optimization_step(x, edges, C, B, opt_state, c0):
        """Function to perform the gradient descent on the nodes of the network."""
        r, theta = domain.transform_in(x)

        @partial(jax.grad, has_aux=True, argnums=(0, 1))
        def grad_fun(r, theta):
            P, aux = find_transport_network(r, theta, boundary, sources, edges, C, B, args.gamma, domain, ct, dt, False, c_hat, c0)
            return P/P0, aux

        # auxs = [P, C, normed_areas, P_transport, P_delivery, P_cost]
        grads, auxs = grad_fun(r, theta)
        powers = Powers(P=auxs[0], Pt=auxs[3], Pd=auxs[4], Pc=auxs[5])
        C = auxs[1]
        updates, opt_state = optimizer.update(grads, opt_state)
        r_new, theta_new = optax.apply_updates((r, theta), updates) # pyright: ignore
        x_new = domain.transform_out(r_new, theta_new)
        return x_new, (opt_state, C, x, powers)


    P_old = jnp.inf
    for step in (bar := tqdm(range(args.num_iters))):
        c0 = 0.1 * np.exp(-step/200) + 1e-3

        x, (opt_state, C, Xp, PS) = optimization_step(x, edges, C, B, opt_state, c0)


        if (step % args.save_interval) == 0:
            if step == 0:
                tqdm.write(f'N={len(x)}, beta={args.beta}, theta={args.theta}')
                tqdm.write(f'Initial values: P={PS.P:0.4g}, Pt={PS.Pt+PS.Pc:0.4g}, Pd={PS.Pd:0.4g}')
            np.savez(arraydir / f"arrays_{step:08d}.npz", X=Xp, edges=edges, C=C, power=PS.P,
                     step=step, boundary=boundary, sources=sources, pt=PS.Pt, pd=PS.Pd, pc=PS.Pc)

        bar.set_description(f"P={PS.P:0.4g} Pt={PS.Pt+PS.Pc:0.4g} Pd={PS.Pd:0.4g}")

        if abs(PS.P - P_old) < (args.atol + args.rtol * abs(PS.P)) and step > 100:
            np.savez(arraydir / f"arrays_{step:08d}.npz", X=x, edges=edges, C=C, power=PS.P,
                     step=step, boundary=boundary, sources=sources, pt=PS.Pt, pd=PS.Pd, pc=PS.Pc)
            break

        P_old = PS.P


if __name__ == "__main__":
    args = parse_args()
    main(args)
