from typing import List, Tuple
import jax
import jax.numpy as jnp
from space_groups import PlaneGroup
from space_groups.utils import sympy_to_numpy
from tile import Tile

class Pattern:
    """Manages a collection of tiles arranged according to space group symmetry"""
    
    def __init__(self, plane_group: PlaneGroup, width: int, height: int):
        """
        Initialize a pattern with its space group and dimensions.
        
        Args:
            plane_group: The crystallographic plane group defining symmetries
            width: Number of unit cells in x direction
            height: Number of unit cells in y direction
        """
        self.group = plane_group
        self.width = width
        self.height = height
        self.tiles: List[Tile] = []
        
    def add_tile(self, tile: Tile):
        """
        Adds a tile to the pattern.
        
        Args:
            tile: Tile to add to the pattern
        """
        self.tiles.append(tile)
        
    def generate_tiling(self) -> List[Tile]:
        """
        Generates the full tiling pattern using space group operations.
        
        Returns:
            List of tiles forming the complete pattern
        """
        tiled_tiles = []
        for tile in self.tiles:
            orbits = self._get_orbits(tile.vertices)
            for orbit in orbits:
                new_tile = Tile(orbit, 
                              boundary_offset=0.1,  # These could be parameters
                              inner_offset=0.2)
                tiled_tiles.append(new_tile)
        return tiled_tiles
    
    def _get_orbits(self, points: jnp.ndarray) -> jnp.ndarray:
        """Computes orbit of points under space group operations."""
        points_sb = jnp.linalg.solve(self.group.basis, points.T).T
        operations = jnp.array([jnp.array(sympy_to_numpy(op)) for op in self.group.operations])
        
        # Apply symmetry operations
        op_points = jax.vmap(
            jax.vmap(self._apply_operation, in_axes=(None, 0)),
            in_axes=(0, None))(operations, points_sb)
        
        # Add translations
        tx, ty = jnp.meshgrid(jnp.arange(self.width), jnp.arange(self.height))
        translations = jnp.stack([tx.flatten(), ty.flatten()], axis=-1)
        
        tiled_points = op_points[jnp.newaxis,:,:,:] + \
                      translations[:,jnp.newaxis,jnp.newaxis,:]
        tiled_points = tiled_points.reshape(-1, *points.shape)
        return tiled_points @ self.group.basis.T
    
    @staticmethod
    def _apply_operation(op, point):
        """
        Applies a single space group operation to a point.
        
        Args:
            op: Space group operation matrix
            point: Point to transform
            
        Returns:
            Transformed point
        """
        op_A = op[:2,:2]
        op_t = op[:2,2]
        return point @ op_A.T + op_t

    def get_all_vertices(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Gets vertices for all tiles in the pattern before applying symmetry operations.
        
        Returns:
            Tuple containing:
            - Array of all vertex coordinates shape (total_points, 2)
            - Array of indices mapping back to original tiles
        """
        # First get all orbits for each tile
        all_vertices = []
        tile_indices = []
        
        for tile_idx, tile in enumerate(self.tiles):
            orbits = self._get_orbits(tile.vertices)
            all_vertices.append(orbits)
            tile_indices.append(jnp.full(orbits.shape[0], tile_idx))
        
        # Concatenate all vertices and indices
        vertices = jnp.concatenate(all_vertices, axis=0)
        indices = jnp.concatenate(tile_indices, axis=0)
        
        return vertices, indices