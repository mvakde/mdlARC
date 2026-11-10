"""Measure colored rectangles and orthogonal three-spoke forks from returned grids.

This does not import the generator or assert the correctness of its motion rule.
"""

import numpy as np


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    while remaining:
        cells = {remaining.pop()}
        queue = list(cells)
        while queue:
            r, c = queue.pop()
            for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if point in remaining:
                    remaining.remove(point)
                    cells.add(point)
                    queue.append(point)
        yield cells


def _fork_geometry(mask):
    for turns in range(4):
        view = np.rot90(mask, turns)
        base = 0
        while base < len(view) and bool(np.all(view[base])):
            base += 1
        if not 0 < base < len(view):
            continue
        arms = view[base:]
        lengths = np.sum(arms, axis=0).tolist()
        if any(not np.array_equal(arms[:, c], np.arange(len(arms)) < lengths[c]) for c in range(arms.shape[1])):
            continue
        runs = []
        start = None
        for index, length in enumerate(lengths + [0]):
            if length and start is None:
                start = index
            if not length and start is not None:
                group = lengths[start:index]
                if len(set(group)) != 1:
                    runs = []
                    break
                runs.append((index-start, group[0]))
                start = None
        if len(runs) == 3:
            return (view.shape[1], tuple(int(run[1]) for run in runs))
        if len(runs) == 2 and lengths[0] and lengths[-1]:
            return (view.shape[1], (int(runs[0][1]), 0, int(runs[1][1])))
    raise ValueError("could not identify a three-spoke orthogonal fork")


def features(input_grid, output_grid):
    colors, counts = np.unique(input_grid, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    forks, fork_colors, token_colors = [], set(), set()
    for color in colors:
        if int(color) == background:
            continue
        for cells in _components(input_grid == color):
            rows, cols = zip(*cells)
            mask = np.zeros((max(rows)-min(rows)+1, max(cols)-min(cols)+1), dtype=bool)
            for r, c in cells:
                mask[r-min(rows), c-min(cols)] = True
            if bool(np.all(mask)):
                token_colors.add(int(color))
            else:
                forks.append(_fork_geometry(mask))
                fork_colors.add(int(color))
    if not forks:
        raise ValueError("expected fork objects")
    trail = sorted(set(map(int, np.unique(output_grid))) - set(map(int, colors)))
    if len(trail) != 1:
        raise ValueError("expected one introduced trail color")
    lengths = [length for _, arms in forks for length in arms]
    return {
        "fork_color_count": len(fork_colors),
        "token_color_count": len(token_colors),
        "fork_palette": sorted(fork_colors),
        "token_palette": sorted(token_colors),
        "fork_widths": sorted(set(int(width) for width, _ in forks)),
        "spoke_lengths": sorted(set(lengths)),
        "min_spoke_length": min(lengths),
        "fork_geometries": sorted([int(width), list(arms)] for width, arms in forks),
        "trail_color": trail[0],
    }
