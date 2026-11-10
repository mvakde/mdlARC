"""Reference for the four demonstrated shapes, independent of the generator."""
import numpy as np


def _shape_and_ink(grid):
    grid = np.asarray(grid)
    if grid.shape != (3, 3):
        raise ValueError("27a28665 uses 3 by 3 inputs")
    colors = set(grid.flat)
    if 0 not in colors or len(colors) != 2:
        raise ValueError("expected black background and one foreground color")
    occupied = frozenset(map(tuple, np.argwhere(grid != 0)))
    vocabulary = (
        frozenset(((0, 1), (0, 2), (1, 1), (1, 2), (2, 0))),
        frozenset(((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))),
        frozenset(((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))),
        frozenset(((0, 0), (0, 1), (1, 0), (1, 2), (2, 1))),
    )
    if occupied not in vocabulary:
        raise ValueError("shape is outside the original four-mask vocabulary")
    return vocabulary.index(occupied), int(next(color for color in colors if color != 0))


def solve(train_pairs, test_input):
    lookup = {}
    for pair in train_pairs:
        shape, _ = _shape_and_ink(pair["input"])
        output = np.asarray(pair["output"])
        if output.shape != (1, 1):
            raise ValueError("expected a one-cell color label")
        label = int(output[0, 0])
        if shape in lookup and lookup[shape] != label:
            raise ValueError("contradictory shape labels")
        lookup[shape] = label
    shape, _ = _shape_and_ink(test_input)
    if shape not in lookup:
        raise ValueError("test shape has no demonstrated color mapping")
    return np.array([[lookup[shape]]], dtype=int)


def features(input_grid, output_grid):
    shape, ink = _shape_and_ink(input_grid)
    return {"source_shape": shape, "shape_ink": ink}
