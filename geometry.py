import jax
import jax.numpy as jnp
import jax.random as jrnd
from scipy.spatial import Voronoi
from scipy.spatial.distance import pdist
from shapely.geometry import Polygon
from typing import Tuple, Union
import numpy as np

def compute_fundamental_region(plane_group) -> Tuple[Polygon, float]:
    """
    Compute the fundamental region of a plane group using Voronoi diagrams.
    
    Args:
        plane_group: Crystallographic plane group
        
    Returns:
        Tuple of (fundamental region polygon, minimum radius)
    """
    # Get the centroid of the fundamental region
    verts = plane_group.asu.numpy_vertices
    centroid = jnp.mean(verts, axis=0)
    
    # Generate orbits for Voronoi computation
    grid = jnp.array([0, -2, -1, 1, 2])
    centroid_orbits = plane_group.orbit_points(jnp.atleast_2d(centroid), txty=grid)
    vor = Voronoi(centroid_orbits)
    
    # Process ridge vertices
    ridge_vertices = jnp.array(vor.ridge_vertices)
    positive = jnp.all(ridge_vertices >= 0, axis=1)
    ridge_vertices = ridge_vertices[positive, :]
    ridge_centers = jnp.mean(vor.vertices[ridge_vertices, :], axis=1)
    
    # Compute minimum radius and get central polygon
    min_radius = jnp.min(jnp.linalg.norm(ridge_centers - centroid, axis=1))
    verts = vor.vertices[vor.regions[vor.point_region[0]]]
    
    return Polygon(verts), min_radius

def resample_polygon(polygon: Union[Polygon, jnp.ndarray], total_pts: int) -> jnp.ndarray:
    """
    Resample a polygon to have a specified number of evenly spaced points.
    
    Args:
        polygon: Input polygon (Shapely polygon or array of points)
        total_pts: Desired number of points
        
    Returns:
        Array of resampled points
    """
    # Convert input to points if necessary
    if isinstance(polygon, Polygon):
        pts = jnp.array(polygon.exterior.coords.xy).T
    else:
        pts = polygon
        
    # If we already have enough points, return early
    if pts.shape[0] >= total_pts:
        return pts
        
    # Ensure the polygon is closed
    if not jnp.allclose(pts[0], pts[-1]):
        pts = jnp.concatenate([pts, pts[0:1]], axis=0)
        
    # Compute segment lengths and cumulative distances
    segs = jnp.linalg.norm(pts[1:,:] - pts[:-1,:], axis=-1)
    # Add small epsilon to prevent zero-length segments
    segs = segs + 1e-10
    cum_segs = jnp.concatenate([jnp.array([0.]), jnp.cumsum(segs)])
    total_len = cum_segs[-1]
    
    # Generate new points with jitter
    num_new = total_pts - pts.shape[0]
    if num_new <= 0:
        return pts
        
    new_pts = jnp.linspace(0, total_len, num_new+2)[1:-1]
    
    rng = jrnd.PRNGKey(0)
    jitter_scale = 0.01 * (new_pts[1] - new_pts[0]) if num_new > 1 else 0.01 * total_len
    new_pts += jitter_scale * jrnd.uniform(rng, (num_new,), minval=-1, maxval=1)
    
    # Combine and sort points
    all_pts = jnp.concatenate([new_pts, cum_segs])
    new_pts = jnp.sort(all_pts)
    
    # Find segment indices, ensuring they stay in bounds
    seg_idx = jnp.clip(jnp.searchsorted(cum_segs, new_pts) - 1, 0, len(segs)-1)
    
    # Compute interpolation factors
    dists = new_pts - cum_segs[seg_idx]
    alpha = jnp.clip(dists / segs[seg_idx], 0., 1.)
    alpha = alpha[:,jnp.newaxis]
    
    # Interpolate points
    start_pts = pts[seg_idx]
    end_pts = pts[seg_idx + 1]
    resampled_pts = (1-alpha)*start_pts + alpha*end_pts
    
    # Ensure first and last points match
    resampled_pts = resampled_pts.at[-1,:].set(pts[0,:])
    
    # Check for duplicates
    dists = pdist(resampled_pts[:-1,:])
    if jnp.min(dists) < 1e-6:
        raise ValueError('Polygon has repeated points')
        
    return resampled_pts

def offset_polygon(poly: Polygon, offset: float, num_pts: int) -> Polygon:
    """
    Create an offset version of a polygon with resampling.
    
    Args:
        poly: Input polygon
        offset: Offset distance (negative for inward offset)
        num_pts: Number of points for resampling
        
    Returns:
        Offset polygon
    """
    offset_poly = poly.buffer(-offset)
    
    # Handle potential MultiPolygon result from buffer operation
    if not isinstance(offset_poly, Polygon):
        if offset_poly.is_empty:
            raise ValueError("Offset resulted in empty polygon - offset may be too large")
        # Get the largest polygon if buffer creates multiple
        offset_poly = max(offset_poly.geoms, key=lambda p: p.area)
    
    # Get coordinates and ensure we have enough points
    coords = np.array(offset_poly.exterior.coords)
    if len(coords) < 3:
        raise ValueError(f"Offset polygon has insufficient points: {len(coords)}")
        
    # Resample the polygon
    grid = resample_polygon(offset_poly, num_pts)[:-1,:]
    return Polygon(grid)