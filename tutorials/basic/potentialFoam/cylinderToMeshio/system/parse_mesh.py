from pathlib import Path
from typing import List, Optional

import numpy as np

import meshio
import skfem
import skfem.io.json

from polymesh import PolyMesh


def main(
    mesh_file: str,
    front_patches: List[str],
    transform: np.ndarray,
    case: Optional[Path] = Path("."),
) -> None:

    poly_mesh = PolyMesh.read(case)
    mio = poly_mesh.to_meshioquad(front_patches, transform=transform)
    meshio.write(mesh_file, mio, file_format="gmsh22", binary=False)

    mesh = poly_mesh.to_meshquad(front_patches, transform=transform)
    mesh.save(Path(mesh_file).with_suffix(".xdmf"))

    left_lines = mio.cells[1].data[
        mio.cell_data["gmsh:physical"][1] == mio.field_data["left"][0]
    ]
    left_points, inverse = np.unique(left_lines, return_inverse=True)
    left_mesh = skfem.MeshLine(
        mio.points[left_points, 1], inverse.reshape(left_lines.shape).T
    )

    skfem.io.json.to_file(left_mesh, "left.json")


if __name__ == "__main__":

    from argparse import ArgumentParser
    import json

    parser = ArgumentParser()
    parser.add_argument("--case", type=Path, default=Path("."))
    parser.add_argument(
        "--json",
        type=Path,
        help="name of file containing input parameters",
        default=Path(__file__).with_name("cylinder.json"),
    )

    args = parser.parse_args()

    with open(args.json, "r") as fjson:
        parameters = json.load(fjson)

    main(**parameters)
