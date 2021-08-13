from pathlib import Path

import skfem
import skfem.io.json

def main(meshfile: Path) -> None:
    left_mesh = skfem.io.json.from_file(meshfile)
    left_basis = skfem.CellBasis(left_mesh, skfem.ElementLineP0())
    inlet_velocity = skfem.project(
        lambda y: (left_mesh.p.max() - y[0]) ** 2, basis_to=left_basis
    )
    with open(Path("0") / "inletVelocity", "w") as fout:
        fout.write("inletVelocity nonuniform List<vector>\n")
        fout.write(f"{left_mesh.t.shape[1]}\n")
        fout.write("(\n")
        fout.write("\n".join(f"({u} 0 0)" for u in inlet_velocity))
        fout.write("\n);\n")


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--meshfile", type=Path, default=Path(__file__).stem + ".json")
    args = parser.parse_args()
    main(args.meshfile)
