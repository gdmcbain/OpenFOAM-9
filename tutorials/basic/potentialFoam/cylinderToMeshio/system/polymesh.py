from dataclasses import dataclass
from pathlib import Path
from shlex import split
from typing import TextIO, Dict, Tuple, Optional, List

from more_itertools import chunked

import numpy as np

import meshio


@dataclass
class PolyMesh:
    points: np.ndarray
    faces: np.ndarray
    boundary: Dict[str, slice]
    neighbour: np.ndarray
    owner: np.ndarray
    note: Dict[str, int]

    @classmethod
    def read(cls, case: Optional[Path] = None):
        case = Path(".") if case is None else case
        neighbour, note = cls.read_index(case, "neighbour")
        owner, _ = cls.read_index(case, "owner")
        return cls(
            cls.read_points(case),
            cls.read_faces(case),
            cls.read_boundary(case),
            neighbour,
            owner,
            note,
        )

    @staticmethod
    def polymesh(case: Path) -> Path:
        return (case / "constant") / "polyMesh"

    @staticmethod
    def swallow_block_comment(fp: TextIO) -> None:
        for line in fp:
            if line.lstrip().startswith("/*"):
                break
        for line in fp:
            if line.rstrip().endswith("*/"):
                break

    @staticmethod
    def is_line_comment(line: str) -> bool:
        return line.lstrip().startswith("//")

    @staticmethod
    def read_foamfile(fp: TextIO) -> Dict[str, int]:
        note = {}
        assert fp.readline().rstrip() == "FoamFile"
        assert fp.readline().strip().startswith("{")
        for line in fp:
            if line.strip() == "}":
                break
            words = split(line.rstrip().rstrip(";"))
            if words[0] == "note":
                for k, v in chunked(words[1].split(), 2):
                    note[k.rstrip(":")] = int(v)
        return note

    @classmethod
    def read_points(cls, case: Path) -> np.ndarray:
        with open(cls.polymesh(case) / "points", "r") as fin:
            cls.swallow_block_comment(fin)
            cls.read_foamfile(fin)
            assert cls.is_line_comment(fin.readline())

            for line in fin:
                if len(line.strip()) == 0:
                    continue
                npoints = int(line)
                break

            assert fin.readline().strip() == "("
            return np.array(
                [np.fromstring(fin.readline()[1:-2], sep=" ") for _ in range(npoints)]
            )

    @classmethod
    def read_faces(cls, case: Path) -> np.ndarray:
        with open(cls.polymesh(case) / "faces", "r") as fin:
            cls.swallow_block_comment(fin)
            cls.read_foamfile(fin)
            assert cls.is_line_comment(fin.readline())

            for line in fin:
                if len(line.strip()) == 0:
                    continue
                nfaces = int(line)
                break

            assert fin.readline().strip() == "("
            return np.array(
                [
                    np.fromstring(fin.readline()[2:-2], int, sep=" ")
                    for _ in range(nfaces)
                ]
            )

    @classmethod
    def read_boundary(cls, case: Path) -> Dict[str, slice]:
        with open(cls.polymesh(case) / "boundary", "r") as fin:
            cls.swallow_block_comment(fin)
            cls.read_foamfile(fin)
            assert cls.is_line_comment(fin.readline())

            for line in fin:
                if len(line.strip()) == 0:
                    continue
                npatches = int(line)
                print(f"{npatches} patches")
                break

            boundary = {}
            assert fin.readline().strip() == "("
            for _ in range(npatches):
                name = fin.readline().strip()
                assert fin.readline().strip() == "{"
                start_face, n_faces = 0, 0
                while start_face * n_faces == 0:
                    line = fin.readline().strip().rstrip(";")
                    words = line.split()
                    if words[0] == "startFace":
                        start_face = int(words[1])
                    elif words[0] == "nFaces":
                        n_faces = int(words[1])
                assert fin.readline().strip() == "}"
                boundary[name] = slice(start_face, start_face + n_faces)

        return boundary

    @classmethod
    def read_index(cls, case: Path, name: str) -> Tuple[np.ndarray, Dict[str, int]]:
        with open(cls.polymesh(case) / name, "r") as fin:
            cls.swallow_block_comment(fin)
            note = cls.read_foamfile(fin)
            assert cls.is_line_comment(fin.readline())

            for line in fin:
                try:
                    nindices = int(line)
                except ValueError:
                    continue
                break

            assert fin.readline().rstrip() == "("
            index = np.genfromtxt(fin, int, max_rows=nindices)

        return index, note

    def to_meshquad(
        self,
        front_patches: Optional[List[str]] = None,
        transform: Optional[np.ndarray] = None,
    ) -> meshio.Mesh:

        front_patches = ["front"] if front_patches is None else front_patches
        front_faces = np.concatenate(
            [self.faces[self.boundary[label]] for label in front_patches]
        )
        front_points, indices, inverse = np.unique(
            front_faces, return_index=True, return_inverse=True
        )

        subdomains = {}
        cell_index = 0
        for label in front_patches:
            ncells = self.boundary[label].stop - self.boundary[label].start
            subdomains[label] = cell_index + np.arange(ncells)
            cell_index += ncells

        physical_names = {}

        quad_tags = np.zeros(front_faces.shape[0])
        for i, (name, quads) in enumerate(subdomains.items(), 1):
            physical_names[name] = (i, 2)
            quad_tags[quads] = i

        lines = np.empty((0, 2), dtype=int)
        line_tags = []
        line_tag = len(subdomains)

        for label, patch in self.boundary.items():
            if label in front_patches:
                continue
            patch_faces = self.faces[patch]
            boundary_lines = np.searchsorted(
                front_points, patch_faces[np.isin(patch_faces, front_points).nonzero()]
            ).reshape((-1, 2))
            if boundary_lines.size > 0:
                lines = np.concatenate([lines, boundary_lines])
                line_tag += 1
                line_tags += [line_tag] * len(boundary_lines)
                physical_names[label] = (line_tag, 1)

        front_owners = np.concatenate(
            [self.owner[self.boundary[label]] for label in front_patches]
        )
        points = self.points[front_points]
        if transform:
            points @= np.hstack([transform, np.zeros((3, 1))])
        return meshio.Mesh(
            points,
            [("quad", inverse.reshape(front_faces.shape)), ("line", lines)],
            cell_data={
                "gmsh:geometrical": [quad_tags, np.array(line_tags)],
                "gmsh:physical": [quad_tags, np.array(line_tags)],
                "owner": [front_owners, np.zeros_like(line_tags)],
            },
            field_data=physical_names,
        )


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--case", dest="case", type=Path, default=Path("."))
    args = parser.parse_args()

    polymesh = PolyMesh.read(args.case)
    print(polymesh)
