from pathlib import Path

import numpy as np

vertices = np.genfromtxt(Path(__file__).with_name(f"old_vertices.txt"))

arcs = np.genfromtxt(Path(__file__).with_name("old_edges.txt"))
termini = arcs[:, :2].astype(np.uint)
knots = arcs[:, 2:]

blocks = np.genfromtxt(Path(__file__).with_name("old_blocks.txt"), dtype=np.uint)
block_vertices, block_size, _ = np.split(blocks, [8, 11], axis=1)


corners = {"min": vertices.min(0), "max": vertices.max(0)}
patches = {
    "left": vertices[:, 0] == corners["min"][0],
    "right": vertices[:, 0] == corners["max"][0],
    "down": vertices[:, 1] == corners["min"][1],
    "up": vertices[:, 1] == corners["max"][1],
}

for v, size in zip(block_vertices, block_size):
    if np.any(patches["right"][v]):
        size[0] = 100
    if np.any(patches["down"][v]):
        size[1] = 10

patches["interior"] = np.logical_not(np.stack(list(patches.values()))).prod(
    0, dtype=bool
)

vertices[patches["left"], 0] = 0.0
vertices[patches["right"], 0] = 2.2
vertices[patches["down"], 1] = 0.0
vertices[patches["up"], 1] = 0.2 + 0.21

indices = patches["left"] & ~(patches["down"] | patches["up"])
y = vertices[indices, 1]
radius = np.linalg.norm(vertices[patches["interior"], :2], axis=1).min()
vertices[indices, 1] = y * 0.05 / radius + 0.2

indices = patches["right"] & ~(patches["down"] | patches["up"])
y = vertices[indices, 1]
vertices[indices, 1] = y * 0.05 / radius + 0.2

indices = patches["down"] & ~(patches["left"] | patches["right"])
x = vertices[indices, 0]
vertices[indices, 0] = x * 0.05 / radius + 0.2

indices = patches["up"] & ~(patches["left"] | patches["right"])
x = vertices[indices, 0]
vertices[indices, 0] = x * 0.05 / radius + 0.2

centre = vertices[patches["interior"]].mean(0)  # (0, 0, 0)


def transform_circle(x: np.ndarray, indices: np.ndarray = None) -> None:
    indices = np.ones(x.shape[:1], dtype=bool) if indices is None else indices
    radii = np.linalg.norm(x[indices, :2] - centre[:2], axis=1)
    x[indices, :2] = 0.05 * np.tile(radii, (2, 1)).T / min(radii) * (
        x[indices, :2] - centre[:2]
    ) + np.array([0.2, 0.2])


transform_circle(vertices, patches["interior"])

with open(Path(__file__).with_name("vertices.txt"), "w") as fout:
    fout.write("(\n")
    for x, y, z in vertices:
        fout.write(f"({x} {y} {z})\n")
    fout.write(")\n")

transform_circle(knots, np.ones(knots.shape[:1], dtype=bool))

with open(Path(__file__).with_name("edges.txt"), "w") as fedges:
    fedges.write("(\n")
    for (start, finish), (x, y, z) in zip(termini, knots):
        fedges.write(f"arc {start} {finish} ({x} {y} {z})\n")
    fedges.write(")\n")

with open(Path(__file__).with_name("blocks.txt"), "w") as fblocks:
    fblocks.write("(\n")
    for v, n in zip(block_vertices, block_size):
        fblocks.write("hex (" + " ".join(f"{i}" for i in v) + ") (" + " ".join(f"{s}" for s in n) + ") simpleGrading (1 1 1)\n")
    fblocks.write(")\n")
