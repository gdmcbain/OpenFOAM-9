from pathlib import Path
from typing import List, Optional

import numpy as np

import meshio
import skfem
import skfem.io.json

from polymesh import PolyMesh


def main(
    mesh_file: str,
    front_patches: Optional[List[str]] = None,
    case: Optional[Path] = Path("."),
    transform: Optional[np.ndarray] = None,
) -> None:

    poly_mesh = PolyMesh.read(case)
    mesh = poly_mesh.to_meshquad(front_patches, transform=transform)
    meshio.write(mesh_file, mesh, file_format="gmsh22", binary=False)

    left_lines = mesh.cells[1].data[
        mesh.cell_data["gmsh:physical"][1] == mesh.field_data["left"][0]
    ]
    left_points, inverse = np.unique(left_lines, return_inverse=True)
    left_mesh = skfem.MeshLine(
        mesh.points[left_points, 1], inverse.reshape(left_lines.shape).T
    )

    skfem.io.json.to_file(left_mesh, "left.json")


if __name__ == "__main__":

    from argparse import ArgumentParser
    import json

    parser = ArgumentParser()
    parser.add_argument("--case", type=Path, default=Path("."))
    parser.add_argument(
        "json", type=Path, help="name of file containing input parameters"
    )

    args = parser.parse_args()

    with open(args.json, "r") as fjson:
        parameters = json.load(fjson)

    main(**parameters)
