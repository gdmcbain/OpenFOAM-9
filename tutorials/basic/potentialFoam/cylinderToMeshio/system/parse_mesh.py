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

    mesh = PolyMesh.read(case).to_meshquad(front_patches, transform=transform)
    mesh.save(Path(mesh_file).with_suffix(".xdmf"))

    left_basis = skfem.BoundaryFacetBasis(
        mesh, skfem.ElementQuad0(), facets=mesh.boundaries["left"]
    )
    skfem.io.json.to_file(
        left_basis.trace(left_basis.zeros(), lambda x: x[1])[0].mesh, "left.json"
    )


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
