"""Tools to help handle coordinate systems and transformations."""

import jax
import jax.numpy as jnp
import quaxed.numpy as qnp
import unxt as u

__all__ = ["get_u_vec", "get_tangent_basis"]


def get_u_vec(lon: u.Angle, lat: u.Angle) -> jax.Array:
    """
    Given two sky coordinates at a longitude and latitude (e.g., RA, Dec), return a unit
    vector that points in the direction of the sky position.
    """
    return jnp.array(
        [qnp.cos(lon) * qnp.cos(lat), qnp.sin(lon) * qnp.cos(lat), qnp.sin(lat)]
    )


def get_tangent_basis(lon: u.Angle, lat: u.Angle) -> jax.Array:
    """
    row vectors are the tangent-space basis at (lon, lat, r)
    """
    return jnp.array(
        [
            [-qnp.sin(lon), qnp.cos(lon), qnp.zeros_like(lon.value)],
            [-qnp.sin(lat) * qnp.cos(lon), -qnp.sin(lat) * qnp.sin(lon), qnp.cos(lat)],
            [qnp.cos(lat) * qnp.cos(lon), qnp.cos(lat) * qnp.sin(lon), qnp.sin(lat)],
        ]
    )
