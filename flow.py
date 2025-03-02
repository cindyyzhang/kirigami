import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.experimental.ode import odeint
from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable
from space_groups import PlaneGroup
from space_groups.utils import sympy_to_numpy
from space_groups.invariance.orbifold import torus

@dataclass
class FlowConfig:
    """Configuration for the equivariant flow"""
    num_feats: int
    lengthscale: float
    timescale: float
    amplitude: float
    duration: float
    num_steps: int
    start_time: float = 0.0

class EquivariantFlow:
    """Handles the continuous normalizing flow with space group symmetries"""
    
    def __init__(self, plane_group: PlaneGroup, config: FlowConfig):
        self.group = plane_group
        self.config = config
        self.operations = jnp.array([jnp.array(sympy_to_numpy(op)) 
                                   for op in plane_group.operations])
        self.base_function, self.params = self.create_vector_field(jrnd.PRNGKey(200))
        
    def create_vector_field(self, rng_key: jrnd.PRNGKey) -> Tuple[Callable, dict]:
        """Creates a random vector field with the specified symmetries"""
        Wx = jrnd.normal(rng_key, (self.config.num_feats, 4)) / self.config.lengthscale
        Wt = jrnd.normal(rng_key, (self.config.num_feats,)) / self.config.timescale
        V = jrnd.normal(rng_key, (self.config.num_feats, 2)) / jnp.sqrt(self.config.num_feats)
        b = jrnd.uniform(rng_key, (self.config.num_feats,)) * 2 * jnp.pi
        
        @jax.jit
        def base_function(w, xy, t):
            return self.config.amplitude * jnp.cos(torus(xy) @ Wx.T + t*Wt + b) @ w
        
        return base_function, V
    
    def get_equivariant_field(self, base_function, params):
        """Creates an equivariant vector field from the base function"""
        @jax.jit
        def vector_field(xy, t):
            # Apply summation trick over all operations
            op_uv = jax.vmap(self._apply_operation, 
                           in_axes=(None, None, None, None, 0))(base_function, params, xy, t, self.operations)
            return jnp.mean(op_uv, axis=0)
        return vector_field
    
    def _apply_operation(self, base_function, params, xy, t, op):
        """Applies a single group operation to the vector field"""
        op_A = op[:2,:2]
        op_t = op[:2,2]
        op_xy = xy @ op_A.T + op_t
        op_uv = base_function(params, op_xy, t)
        return op_uv @ jnp.linalg.inv(op_A).T

    def integrate_left_right(self, vector_field: Callable, points: jnp.ndarray, start_time: float = 0.01, end_time: float = 1.0) -> jnp.ndarray:
        """
        Integrates the flow up to different times based on x-coordinate.
        Points further right will be integrated for longer times.
        """
        # Normalize x-coordinates to [0, 1]
        x_coords = points[..., 0]  # Shape: (B, N)
        min_x = jnp.min(x_coords)
        max_x = jnp.max(x_coords)
        normalized_x = (x_coords - min_x) / (max_x - min_x)  # Shape: (B, N)
        
        # Map to integration end times for each point
        start_t = start_time
        end_t = end_time
        integration_times = start_t + normalized_x * (end_t - start_t)  # Shape: (B, N)
        
        # For each point, integrate up to its specific time
        def integrate_point(point, end_time):
            times = jnp.linspace(0, end_time, self.config.num_steps)
            trajectory = odeint(lambda xy, t: vector_field(xy, t), point, times)
            return trajectory[-1]  # Return final position
        
        transformed_points = jax.vmap(jax.vmap(integrate_point))(points, integration_times)
        return transformed_points

    def integrate(self, vector_field: Callable, initial_points: jnp.ndarray) -> jnp.ndarray:
        """Regular integration of the flow"""
        times = jnp.linspace(0, self.config.duration, self.config.num_steps)
        return odeint(lambda xy, t: vector_field(xy, t), 
                     initial_points, 
                     times)
