from os import PathLike
from pathlib import Path
from typing import List, Optional

import numpy as np

import meshio

from polymesh import PolyMesh


def main(
    case: PathLike,
    mesh_file: str,
    front_patches: Optional[List[str]] = None,
    transform: Optional[np.ndarray] = None,
) -> None:

    case = Path(case)
    print("case:", case)
    poly_mesh = PolyMesh.read(case)
    mesh = poly_mesh.to_meshquad(front_patches, transform=transform)
    meshio.write(
        case / "constant" / mesh_file, mesh, file_format="gmsh22", binary=False
    )


if __name__ == "__main__":

    from argparse import ArgumentParser
    import json

    parser = ArgumentParser()
    parser.add_argument("--case", type=Path, default=Path(__file__).parents[2])
    parser.add_argument(
        "--json",
        type=Path,
        help="name of file containing input parameters",
        default=Path(__file__).with_suffix(".json"),
    )

    args = vars(parser.parse_args())

    with open(args.pop("json"), "r") as fjson:
        args |= json.load(fjson)

    main(**args)
