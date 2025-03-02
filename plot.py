import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from tile import Tile
from pattern import Pattern
import numpy as np

def plot_tiling(pattern: Pattern,
                figsize: Tuple[int, int] = (15, 15),
                tile_color: str = "#636363",
                pore_color: str = "#636363",
                ax: Optional[plt.Axes] = None,
                save_path: Optional[str] = None) -> plt.Axes:
    """Plot a complete tiling pattern.
    
    Args:
        pattern: The pattern to plot
        figsize: Size of the figure
        tile_color: Color of the tiles
        pore_color: Color of the pores
        ax: Optional axes to plot on
        save_path: Optional path to save the figure (e.g., 'flow.png')
    
    Returns:
        The matplotlib axes with the plot
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Set white background
    ax.set_facecolor('white')
    if fig:
        fig.set_facecolor('white')
    
    # Get the tiled pattern
    tiled_tiles = pattern.tiles
    
    # Plot each tile
    for idx, tile in enumerate(tiled_tiles):
        
        # Plot the base tile shape with scaled vertices
        # ax.fill(scaled_vertices[:, 0], scaled_vertices[:, 1], 
        #         color=tile_color, alpha=0.3)
                
        # Plot the pore with scaled inner points
        # ax.fill(scaled_inner[:, 0], scaled_inner[:, 1], 
        #         color=pore_color, alpha=0.5)

        # Plot boundaries - append first point to close the loop
        vertices_closed = np.vstack((tile.vertices, tile.vertices[0]))
        inner_closed = np.vstack((tile.inner, tile.inner[0]))
        
        # Calculate centroids
        vertices_centroid = np.mean(tile.vertices, axis=0)
        inner_centroid = np.mean(tile.inner, axis=0)
        shrink_factor = 0.9

        # Shrink vertices and inner points around their centroids
        # shrunk_vertices = vertices_centroid + (vertices_closed - vertices_centroid) * shrink_factor
        # shrunk_inner = inner_centroid + (inner_closed - inner_centroid) * shrink_factor
        
        # Fill the inside with grey first
        ax.fill(inner_closed[:, 0], inner_closed[:, 1], 
                color=tile_color, alpha=1.0)
        
        # Then plot the white border on top
        # ax.plot(shrunk_inner[:, 0], shrunk_inner[:, 1], 
        #        color='white', linewidth=2, alpha=1.0)
    
    # Set equal aspect ratio and remove axes
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Ensure the black background extends to the edges
    ax.margins(0)
    
    # Save the figure if a save path is provided
    if save_path and fig:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        print(f"Figure saved to {save_path}")
    
    return ax