from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def _infer_sh_degree(property_names: list[str]) -> int:
    rest_names = [name for name in property_names if name.startswith("f_rest_")]
    rest_names = sorted(rest_names, key=lambda name: int(name.split("_")[-1]))
    if not rest_names:
        return 0
    if len(rest_names) % 3 != 0:
        raise ValueError(
            f"Unsupported PLY SH layout: found {len(rest_names)} f_rest properties, which is not divisible by 3."
        )

    coeff_count = (len(rest_names) + 3) // 3
    root = int(round(coeff_count ** 0.5))
    if root * root != coeff_count:
        raise ValueError(
            f"Unsupported PLY SH layout: found {len(rest_names)} f_rest properties, which is not a valid 3DGS degree."
        )
    return root - 1


@dataclass
class GaussianCloud:
    xyz: np.ndarray
    opacity: np.ndarray
    features_dc: np.ndarray
    features_rest: np.ndarray
    scaling: np.ndarray
    rotation: np.ndarray
    sh_degree: int

    @property
    def count(self) -> int:
        return int(self.xyz.shape[0])

    @classmethod
    def load_ply(cls, path: str | Path) -> "GaussianCloud":
        ply = PlyData.read(str(path))
        vertex = ply.elements[0]
        property_names = [prop.name for prop in vertex.properties]
        sh_degree = _infer_sh_degree(property_names)

        xyz = np.stack(
            [
                np.asarray(vertex["x"], dtype=np.float32),
                np.asarray(vertex["y"], dtype=np.float32),
                np.asarray(vertex["z"], dtype=np.float32),
            ],
            axis=1,
        )
        opacity = np.asarray(vertex["opacity"], dtype=np.float32)[:, None]

        features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
        features_dc[:, 0, 0] = np.asarray(vertex["f_dc_0"], dtype=np.float32)
        features_dc[:, 1, 0] = np.asarray(vertex["f_dc_1"], dtype=np.float32)
        features_dc[:, 2, 0] = np.asarray(vertex["f_dc_2"], dtype=np.float32)

        rest_names = sorted(
            [name for name in property_names if name.startswith("f_rest_")],
            key=lambda name: int(name.split("_")[-1]),
        )
        rest_values = np.zeros((xyz.shape[0], len(rest_names)), dtype=np.float32)
        for index, name in enumerate(rest_names):
            rest_values[:, index] = np.asarray(vertex[name], dtype=np.float32)
        if rest_names:
            features_rest = rest_values.reshape((xyz.shape[0], 3, (sh_degree + 1) ** 2 - 1))
        else:
            features_rest = np.zeros((xyz.shape[0], 3, 0), dtype=np.float32)

        scale_names = sorted(
            [name for name in property_names if name.startswith("scale_")],
            key=lambda name: int(name.split("_")[-1]),
        )
        scaling = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
        for index, name in enumerate(scale_names):
            scaling[:, index] = np.asarray(vertex[name], dtype=np.float32)

        rotation_names = sorted(
            [name for name in property_names if name.startswith("rot_") or name.startswith("rot")],
            key=lambda name: int(name.split("_")[-1]),
        )
        rotation = np.zeros((xyz.shape[0], len(rotation_names)), dtype=np.float32)
        for index, name in enumerate(rotation_names):
            rotation[:, index] = np.asarray(vertex[name], dtype=np.float32)

        return cls(
            xyz=xyz,
            opacity=opacity,
            features_dc=features_dc,
            features_rest=features_rest,
            scaling=scaling,
            rotation=rotation,
            sh_degree=sh_degree,
        )

    def bbox(self) -> dict[str, list[float]]:
        xyz_min = self.xyz.min(axis=0)
        xyz_max = self.xyz.max(axis=0)
        return {
            "min": xyz_min.tolist(),
            "max": xyz_max.tolist(),
            "center": ((xyz_min + xyz_max) * 0.5).tolist(),
        }

    def subset(self, mask: np.ndarray) -> "GaussianCloud":
        return GaussianCloud(
            xyz=self.xyz[mask].copy(),
            opacity=self.opacity[mask].copy(),
            features_dc=self.features_dc[mask].copy(),
            features_rest=self.features_rest[mask].copy(),
            scaling=self.scaling[mask].copy(),
            rotation=self.rotation[mask].copy(),
            sh_degree=self.sh_degree,
        )

    def to_ply_bytes(self) -> bytes:
        attribute_names = ["x", "y", "z", "nx", "ny", "nz"]
        attribute_names.extend(f"f_dc_{index}" for index in range(self.features_dc.shape[1] * self.features_dc.shape[2]))
        attribute_names.extend(
            f"f_rest_{index}" for index in range(self.features_rest.shape[1] * self.features_rest.shape[2])
        )
        attribute_names.append("opacity")
        attribute_names.extend(f"scale_{index}" for index in range(self.scaling.shape[1]))
        attribute_names.extend(f"rot_{index}" for index in range(self.rotation.shape[1]))

        dtype = [(name, "f4") for name in attribute_names]
        elements = np.empty(self.count, dtype=dtype)

        normals = np.zeros_like(self.xyz, dtype=np.float32)
        flattened_dc = self.features_dc.transpose(0, 2, 1).reshape(self.count, -1)
        flattened_rest = self.features_rest.transpose(0, 2, 1).reshape(self.count, -1)
        attributes = np.concatenate(
            [
                self.xyz,
                normals,
                flattened_dc,
                flattened_rest,
                self.opacity,
                self.scaling,
                self.rotation,
            ],
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))

        stream = io.BytesIO()
        PlyData([PlyElement.describe(elements, "vertex")], text=False).write(stream)
        return stream.getvalue()
