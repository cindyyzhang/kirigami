from typing import Optional, List
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from space_groups import PlaneGroup
from geometry import compute_fundamental_region, offset_polygon

@dataclass
class TileConfig:
    """Configuration for tile generation"""
    num_points: int
    boundary_offset: float
    inner_offset: float

class Tile:
    """Represents a single tile in the pattern with its geometry and legs"""
    
    def __init__(self, vertices: jnp.ndarray, boundary_offset: float, inner_offset: float):
        """
        Initialize a tile with its vertices and offsets.
        
        Args:
            vertices: Points defining the tile boundary
            boundary_offset: Offset distance for the boundary polygon
            inner_offset: Offset distance for the inner polygon
        """
        self.vertices = vertices
        self.boundary = self._offset_polygon(vertices, boundary_offset)
        self.inner = self._offset_polygon(vertices, inner_offset)
        self.anchor_points: Optional[jnp.ndarray] = None
        self.hinge_points: Optional[jnp.ndarray] = None
        self.start_hinge_points: Optional[jnp.ndarray] = None
        self.neighbor_indices: List[int] = []
    
    @classmethod
    def from_fundamental_region(cls, 
                              plane_group: PlaneGroup, 
                              config: TileConfig) -> "Tile":
        """
        Create a tile from the fundamental region of a plane group.
        
        Args:
            plane_group: Crystallographic plane group
            config: Configuration for tile generation
            
        Returns:
            New Tile instance
        """
        # Compute fundamental region
        fr_poly, _ = compute_fundamental_region(plane_group)
        
        # Create the polygons with offsets
        tile_poly = offset_polygon(fr_poly, 0, config.num_points)
        boundary_poly = offset_polygon(fr_poly, config.boundary_offset, config.num_points)
        inner_poly = offset_polygon(fr_poly, config.inner_offset, config.num_points)
        
        # Extract vertices and create tile
        vertices = jnp.array(tile_poly.exterior.xy).T[:-1]  # Remove duplicate end point
        
        tile = cls(vertices, config.boundary_offset, config.inner_offset)
        tile.boundary = jnp.array(boundary_poly.exterior.xy).T[:-1]
        tile.inner = jnp.array(inner_poly.exterior.xy).T[:-1]
        
        return tile
        
    @staticmethod
    def _offset_polygon(vertices: jnp.ndarray, offset: float) -> jnp.ndarray:
        """Creates an offset polygon from the input vertices"""
        # Simple scaling implementation - could be replaced with proper offsetting
        centroid = jnp.mean(vertices, axis=0)
        return centroid + (vertices - centroid) * (1 - offset)