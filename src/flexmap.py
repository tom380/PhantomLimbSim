from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import mujoco
import numpy as np


@dataclass(frozen=True)
class FlexSurfaceInfo:
    flex_id: int
    name: str
    vertices: np.ndarray
    vertadr: int
    dim: int
    theta: np.ndarray
    theta_unwrapped: np.ndarray
    axial: np.ndarray
    radius: float
    circumference: float
    elements: np.ndarray
    body_id: int
    axis: np.ndarray
    origin: np.ndarray

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])


def _normalise(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= 0:
        return vec
    return vec / norm


def _cylindrical_unwrap(vertices: np.ndarray) -> Dict[str, np.ndarray]:
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid

    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = _normalise(axis)

    axial = centered @ axis
    radial_vecs = centered - np.outer(axial, axis)
    radial_norm = np.linalg.norm(radial_vecs, axis=1)

    valid = radial_norm > 1e-8
    if not np.any(valid):
        raise ValueError("Unable to compute cylindrical unwrap; radial extent is zero.")

    ref_vec = radial_vecs[valid][np.argmax(radial_norm[valid])]
    ref_vec = _normalise(ref_vec)
    binormal = _normalise(np.cross(axis, ref_vec))

    angles = np.zeros(vertices.shape[0])
    angles[valid] = np.arctan2(radial_vecs[valid] @ binormal, radial_vecs[valid] @ ref_vec)
    theta = np.mod(angles, 2 * np.pi)

    radius = radial_norm[valid].mean()
    circumference = 2 * np.pi * radius

    return {
        "axis": axis,
        "origin": centroid,
        "theta": theta,
        "axial": axial,
        "radius": radius,
        "circumference": circumference,
    }


def _build_adjacency(elements: np.ndarray, vertex_count: int) -> List[List[int]]:
    adjacency: List[List[int]] = [[] for _ in range(vertex_count)]
    if elements.size == 0:
        return adjacency

    for elem in elements:
        valid_vertices = [int(v) for v in elem if 0 <= int(v) < vertex_count]
        for i in range(len(valid_vertices)):
            vi = valid_vertices[i]
            neighbors = adjacency[vi]
            for j in range(i + 1, len(valid_vertices)):
                vj = valid_vertices[j]
                if vj not in neighbors:
                    neighbors.append(vj)
                if vi not in adjacency[vj]:
                    adjacency[vj].append(vi)
    return adjacency


def _unwrap_angles(theta: np.ndarray, elements: np.ndarray, vertex_count: int) -> np.ndarray:
    unwrapped = np.array(theta, dtype=float, copy=True)
    adjacency = _build_adjacency(elements, vertex_count)
    visited = np.zeros(vertex_count, dtype=bool)
    two_pi = 2.0 * np.pi

    for start in range(vertex_count):
        if visited[start]:
            continue

        visited[start] = True
        stack = [start]

        while stack:
            current = stack.pop()
            current_theta = unwrapped[current]

            for nb in adjacency[current]:
                raw_theta = theta[nb]
                offset = np.round((current_theta - raw_theta) / two_pi)
                candidate = raw_theta + offset * two_pi

                if not visited[nb]:
                    unwrapped[nb] = candidate
                    visited[nb] = True
                    stack.append(nb)
                else:
                    diff = unwrapped[nb] - candidate
                    if abs(diff) > np.pi:
                        adjustment = np.round(diff / two_pi)
                        unwrapped[nb] -= adjustment * two_pi

    min_theta = float(np.min(unwrapped))
    if np.isfinite(min_theta):
        unwrapped -= min_theta

    return unwrapped


def compute_flex_surfaces(model: mujoco.MjModel) -> Dict[int, FlexSurfaceInfo]:
    surfaces: Dict[int, FlexSurfaceInfo] = {}

    for flex_id in range(model.nflex):
        vert_start = model.flex_vertadr[flex_id]
        vert_count = model.flex_vertnum[flex_id]
        vertices = np.array(model.flex_vert[vert_start : vert_start + vert_count], copy=True)

        unwrap = _cylindrical_unwrap(vertices)

        dim = int(model.flex_dim[flex_id])
        elem_width = dim + 1
        elem_start = model.flex_elemadr[flex_id]
        elem_count = model.flex_elemnum[flex_id]
        elem_slice = slice(elem_start, elem_start + elem_count * elem_width)
        elements = np.array(model.flex_elem[elem_slice], copy=True).reshape(elem_count, elem_width)
        theta_unwrapped = _unwrap_angles(unwrap["theta"], elements, vert_count)

        node_start = model.flex_nodeadr[flex_id]
        node_count = model.flex_nodenum[flex_id]
        node_bodies = np.array(
            model.flex_nodebodyid[node_start : node_start + node_count], copy=True
        )
        body_id = int(node_bodies[0]) if node_bodies.size else -1

        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, flex_id)
        if name is None:
            name = f"flex_{flex_id}"

        surfaces[flex_id] = FlexSurfaceInfo(
            flex_id=flex_id,
            name=name,
            vertices=vertices,
            vertadr=int(vert_start),
            dim=dim,
            theta=unwrap["theta"],
            theta_unwrapped=theta_unwrapped,
            axial=unwrap["axial"],
            radius=float(unwrap["radius"]),
            circumference=float(unwrap["circumference"]),
            elements=elements,
            body_id=body_id,
            axis=unwrap["axis"],
            origin=unwrap["origin"],
        )

    return surfaces
