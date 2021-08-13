from pathlib import Path

import numpy as np

import skfem
from skfem.models.poisson import laplace
from skfem.visuals.matplotlib import plot, savefig


def main(meshfile: Path) -> None:
    mesh = skfem.MeshQuad.load("quads.msh")
    basis = skfem.InteriorBasis(mesh, skfem.ElementQuad2())

    outlet = basis.get_dofs(mesh.boundaries["right"]).all()
    y_max = mesh.p[1, basis.get_dofs(mesh.boundaries["left"]).nodal["u"]].max()
    inlet_basis = skfem.FacetBasis(mesh, basis.elem, facets=mesh.boundaries["left"])

    @skfem.LinearForm
    def inlet(v, w):
        return v * (y_max - w.x[1]) ** 2

    A = skfem.asm(laplace, basis)
    b = skfem.asm(inlet, inlet_basis)

    phi = np.zeros(basis.N)
    phi = skfem.solve(*skfem.condense(A, b, phi, D=outlet))

    plot(mesh, phi[basis.nodal_dofs[0]])
    savefig(Path(".") / (Path(__file__).stem + "-skfem.png"))


if __name__ == "__main__":

    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument("meshfile", type=Path)
    args = parser.parse_args()
    main(args.meshfile)
