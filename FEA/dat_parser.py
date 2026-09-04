from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class Material:
    mat_id: int
    E: float
    nu: float
    sigma_t: float
    K_Ic: float
    density: float | None = None


@dataclass
class DatModel:
    path: Path
    nodes: np.ndarray
    node_ids: np.ndarray
    node_id_to_index: dict[int, int]
    elements: np.ndarray
    element_ids: np.ndarray
    element_material_ids: np.ndarray
    constraints: dict[int, dict[str, float]]
    prescribed_uy: dict[int, float]
    wall_nodes: list[int]
    materials: dict[int, Material]

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.elements.shape[0])


def _strip_comment(line: str) -> str:
    return line.strip()


def _section_name(line: str) -> str | None:
    text = line.strip()
    low = text.lower()
    if low in {
        "coordinates",
        "element",
        "moment-load",
        "presure",
        "wall",
        "material properties",
    }:
        return low
    if low.startswith("end_"):
        return None
    return None


def read_dat(path: str | Path) -> DatModel:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        raise ValueError(f"{path} is empty")

    header = lines[0].split()
    expected_nodes = expected_elements = None
    if len(header) >= 2 and header[0].isdigit() and header[1].isdigit():
        expected_nodes = int(header[0])
        expected_elements = int(header[1])

    nodes: dict[int, tuple[float, float]] = {}
    elements: list[tuple[int, int, int, int, int, int]] = []
    constraints: dict[int, dict[str, float]] = {}
    prescribed_uy: dict[int, float] = {}
    wall_nodes: list[int] = []
    materials: dict[int, Material] = {}

    section: str | None = None
    for raw in lines[1:]:
        line = _strip_comment(raw)
        if not line or line.startswith("$"):
            continue

        low = line.lower()
        if low.startswith("end_"):
            section = None
            continue

        new_section = _section_name(line)
        if new_section is not None:
            section = new_section
            continue

        if section == "coordinates":
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                node_id = int(parts[0])
                nodes[node_id] = (float(parts[1]), float(parts[2]))

        elif section == "element":
            if line.lower().startswith(("volumes", "surfaces", "lines")):
                continue
            parts = line.split()
            if len(parts) >= 6 and parts[0].isdigit():
                eid = int(parts[0])
                n1, n2, n3, n4 = (int(parts[i]) for i in range(1, 5))
                mat_id = int(parts[5])
                elements.append((eid, n1, n2, n3, n4, mat_id))

        elif section == "moment-load":
            if not line.lower().startswith("node"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                node_id = int(parts[1])
                dofs: dict[str, float] = {}
                for i in range(2, len(parts) - 1, 2):
                    key = parts[i].lower()
                    if key in {"ux", "uy"}:
                        dofs[key] = float(parts[i + 1])
                if dofs:
                    constraints[node_id] = dofs

        elif section == "presure":
            if not line.lower().startswith("node"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                prescribed_uy[int(parts[1])] = float(parts[2])

        elif section == "wall":
            if not line.lower().startswith("node"):
                continue
            match = re.match(r"Node\s*,\s*(\d+)", line, flags=re.IGNORECASE)
            if match:
                wall_nodes.append(int(match.group(1)))

        elif section == "material properties":
            parts = line.split()
            if len(parts) >= 5 and parts[0].isdigit():
                mat_id = int(parts[0])
                density = float(parts[5]) if len(parts) >= 6 else None
                materials[mat_id] = Material(
                    mat_id=mat_id,
                    E=float(parts[1]),
                    nu=float(parts[2]),
                    sigma_t=float(parts[3]),
                    K_Ic=float(parts[4]),
                    density=density,
                )

    if not nodes:
        raise ValueError("No coordinates section was parsed")
    if not elements:
        raise ValueError("No four-node Element section was parsed")
    if not materials:
        raise ValueError("No MATERIAL PROPERTIES section was parsed")

    node_ids = np.array(sorted(nodes), dtype=int)
    node_id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
    node_array = np.array([nodes[node_id] for node_id in node_ids], dtype=float)

    elem_ids = np.array([row[0] for row in elements], dtype=int)
    elem_mat_ids = np.array([row[5] for row in elements], dtype=int)
    elem_nodes = np.array(
        [[node_id_to_index[nid] for nid in row[1:5]] for row in elements],
        dtype=int,
    )

    missing = [int(mid) for mid in np.unique(elem_mat_ids) if int(mid) not in materials]
    if missing:
        raise ValueError(f"Elements reference missing material ids: {missing}")

    if expected_nodes is not None and expected_nodes != len(nodes):
        raise ValueError(f"Header node count {expected_nodes} != parsed {len(nodes)}")
    if expected_elements is not None and expected_elements != len(elements):
        raise ValueError(
            f"Header element count {expected_elements} != parsed {len(elements)}"
        )

    return DatModel(
        path=path,
        nodes=node_array,
        node_ids=node_ids,
        node_id_to_index=node_id_to_index,
        elements=elem_nodes,
        element_ids=elem_ids,
        element_material_ids=elem_mat_ids,
        constraints=constraints,
        prescribed_uy=prescribed_uy,
        wall_nodes=wall_nodes,
        materials=materials,
    )
