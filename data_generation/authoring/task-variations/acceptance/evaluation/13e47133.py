"""Protected contour-band reference and region measurements.

Roles use the dominant canvas color and second-most-frequent divider convention
of the reviewed input distribution. The solver independently erodes each region
and decodes cue colors by depth; it does not import any generator code.
"""

import numpy as np
from host_checks import component_sizes, structural_signature


def _roles(grid):
    colors, counts = np.unique(grid, return_counts=True)
    order = sorted(range(len(colors)), key=lambda i: (-int(counts[i]), int(colors[i])))
    if len(order) < 2:
        raise ValueError("expected canvas and divider colors")
    return int(colors[order[0]]), int(colors[order[1]])


def _regions(mask):
    remaining = set(zip(*np.nonzero(mask)))
    while remaining:
        queue = [remaining.pop()]
        region = np.zeros(mask.shape, dtype=bool)
        while queue:
            r, c = queue.pop()
            region[r, c] = True
            for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if point in remaining:
                    remaining.remove(point)
                    queue.append(point)
        yield region


def solve(train_pairs, test_input):
    """Predict a grid from observable input only; train labels are not needed here."""
    neutral, divider = _roles(test_input)
    answer = test_input.copy()
    h, w = test_input.shape
    for region in _regions(test_input != divider):
        current = region.copy()
        depths = np.full(test_input.shape, -1, dtype=int)
        depth = 0
        while np.any(current):
            padded = np.pad(current, 1, constant_values=False)
            interior = current.copy()
            for dr in range(3):
                for dc in range(3):
                    interior &= padded[dr:dr+h, dc:dc+w]
            depths[current & ~interior] = depth
            current = interior
            depth += 1
        cues = np.argwhere(region & (test_input != neutral))
        if not len(cues):
            answer[region] = neutral
            continue
        period = [neutral] * (max(int(depths[r, c]) for r, c in cues) + 1)
        observed = {}
        for r, c in cues:
            level, color = int(depths[r, c]), int(test_input[r, c])
            if level in observed and observed[level] != color:
                raise ValueError("conflicting key colors at one contour depth")
            observed[level] = color
            period[level] = color
        for level, color in enumerate(period):
            answer[region & (depths % len(period) == level)] = color
    return answer


def features(input_grid, output_grid):
    _, divider = _roles(input_grid)
    mask = input_grid != divider
    return {
        "divider_color": divider,
        "region_count": len(component_sizes(mask)),
        "divider_geometry": structural_signature((input_grid == divider).astype(int)),
    }
