"""Diagonally tapped periodic reservoirs. Standard library and NumPy only.

The solver is independent of the sampler. See task_spec.md for the full rule.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np


_ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_OTHER_SHIFTS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                      if (r, c) != (-1, -1))


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _roles(grid):
    colors, counts = np.unique(grid, return_counts=True)
    repeated = [int(c) for c, n in zip(colors, counts) if c != 0 and n > 1]
    if len(repeated) != 1 or 0 not in colors:
        raise ValueError("need zero ground and exactly one repeated nonzero wall color")
    wall = repeated[0]
    sources = [(int(r), int(c), int(grid[r, c]))
               for r, c in np.argwhere((grid != 0) & (grid != wall))]
    return wall, sources


def _sheltered(walls, periodic=True, diagonal=True):
    """King-neighborhood clearance; sources deliberately do not enter this mask."""
    h, w = walls.shape
    near = np.zeros_like(walls, dtype=bool)
    padded = None if periodic else np.pad(walls, 1, constant_values=False)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if not diagonal and dr != 0 and dc != 0:
                continue
            if periodic:
                near |= np.roll(walls, (dr, dc), axis=(0, 1))
            else:
                near |= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return ~near


def _components(mask, periodic=True, diagonal=False):
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=np.int16)
    components = []
    steps = _ORTHOGONAL + (_DIAGONAL if diagonal else ())
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if labels[r, c] >= 0:
            continue
        label = len(components)
        labels[r, c] = label
        todo, cells = [(r, c)], []
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            for dy, dx in steps:
                a, b = y + dy, x + dx
                if periodic:
                    a, b = a % h, b % w
                if (0 <= a < h and 0 <= b < w and mask[a, b]
                        and labels[a, b] < 0):
                    labels[a, b] = label
                    todo.append((a, b))
        components.append(cells)
    return labels, components


def _arrivals(sources, labels, shift=(-1, -1), wrap=True):
    h, w = labels.shape
    arrivals = [[] for _ in range(int(labels.max()) + 1)]
    missed = []
    for r, c, color in sources:
        y, x = r + shift[0], c + shift[1]
        if wrap:
            y, x = y % h, x % w
        if 0 <= y < h and 0 <= x < w and labels[y, x] >= 0:
            arrivals[int(labels[y, x])].append((r, c, color))
        else:
            missed.append((r, c, color))
    return arrivals, missed


def _complete(grid, *, shift=(-1, -1), erosion_periodic=True,
              flood_periodic=True, clearance_diagonal=True,
              flood_diagonal=False, erode=True, sources_block=False,
              wrap_taps=True, point_only=False, first_wins=False,
              min_size=0, largest_only=False, overwrite_beacons=False):
    """Rule and explicit alternative hypotheses, all operating on input alone."""
    wall, sources = _roles(grid)
    walls = (grid != 0) if sources_block else (grid == wall)
    safe = (_sheltered(walls, erosion_periodic, clearance_diagonal)
            if erode else ~walls)
    labels, components = _components(safe, flood_periodic, flood_diagonal)
    arrivals, _ = _arrivals(sources, labels, shift, wrap_taps)
    out = grid.copy()
    largest = max(map(len, components), default=0)
    for index, cells in enumerate(components):
        taps = arrivals[index]
        if not taps or (not first_wins and len(taps) != 1):
            continue
        if len(cells) < min_size or (largest_only and len(cells) < largest):
            continue
        color = taps[0][2]  # reading order for the deliberately incorrect first-wins rule
        if point_only:
            r, c, _ = taps[0]
            targets = [((r + shift[0]) % grid.shape[0],
                        (c + shift[1]) % grid.shape[1])]
        else:
            targets = cells
        for r, c in targets:
            if grid[r, c] == 0 or (overwrite_beacons and grid[r, c] != wall):
                out[r, c] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve any well-formed tile from its contents, without generator state."""
    grid = _grid(input_grid)
    if max(grid.shape) > 10:
        raise ValueError("tile axes must be at most 10 for a 3-by-3 output")
    return np.tile(_complete(grid), (3, 3))


def _alternatives():
    result = {f"tap_{r}_{c}": {"shift": (r, c)} for r, c in _OTHER_SHIFTS}
    result.update({
        "clipped_clearance": {"erosion_periodic": False},
        "planar_flood": {"flood_periodic": False},
        "fully_planar": {"erosion_periodic": False, "flood_periodic": False},
        "unwrapped_taps": {"wrap_taps": False},
        "orthogonal_clearance": {"clearance_diagonal": False},
        "diagonal_flood": {"flood_diagonal": True},
        "sources_as_walls": {"sources_block": True},
        "no_erosion": {"erode": False},
        "just_taps": {"point_only": True},
        "first_source_wins": {"first_wins": True},
        "minimum_five": {"min_size": 5},
        "largest_only": {"largest_only": True},
        "overwrite_beacons": {"overwrite_beacons": True},
    })
    return result


def _compatible_alternatives(train):
    survivors = _alternatives()
    for pair in train:
        grid = np.asarray(pair["input"])
        h, w = grid.shape
        tile = np.asarray(pair["output"])[:h, :w]
        survivors = {name: kwargs for name, kwargs in survivors.items()
                     if np.array_equal(_complete(grid, **kwargs), tile)}
        if not survivors:
            break
    return set(survivors)


def _wall_mask(rng, h, w):
    """A vocabulary of wall constructions, not a vocabulary of fixed grids."""
    mode = int(rng.integers(6))
    walls = np.zeros((h, w), dtype=bool)
    if mode in (0, 1):
        vertical = bool(rng.integers(2))
        n = w if vertical else h
        start = int(rng.integers(n))
        gap = int(rng.integers(4, max(5, n - 2)))
        for position in (start, (start + gap) % n):
            if vertical:
                walls[:, position] = True
            else:
                walls[position, :] = True
        extras = int(rng.integers(1, 4)) if mode == 1 else int(rng.integers(2))
        for _ in range(extras):
            walls[int(rng.integers(h)), int(rng.integers(w))] = True
    elif mode == 2:
        walls = rng.random((h, w)) < float(rng.uniform(.08, .20))
        for _ in range(int(rng.integers(1, 4))):
            y, x = int(rng.integers(h)), int(rng.integers(w))
            length = int(rng.integers(2, 6))
            vertical = bool(rng.integers(2))
            for i in range(length):
                walls[(y + i * vertical) % h, (x + i * (not vertical)) % w] = True
    elif mode == 3:
        walls = rng.random((h, w)) < float(rng.uniform(.06, .15))
        y, x = int(rng.integers(h)), int(rng.integers(w))
        for _ in range(int(rng.integers(h, 2 * h + w))):
            walls[y, x] = True
            dy, dx = _ORTHOGONAL[int(rng.integers(4))]
            y, x = (y + dy) % h, (x + dx) % w
    elif mode == 4:
        walls[:] = True
        for _ in range(int(rng.integers(2, 5))):
            y, x = int(rng.integers(h)), int(rng.integers(w))
            ph, pw = int(rng.integers(3, 7)), int(rng.integers(3, 7))
            for i in range(ph):
                for j in range(pw):
                    walls[(y + i) % h, (x + j) % w] = False
        if rng.random() < .5:
            walls[int(rng.integers(h)), int(rng.integers(w))] = True
    else:
        walls = rng.random((h, w)) < .06
        y, x = int(rng.integers(h)), int(rng.integers(w))
        walls[y, :] = True
        walls[:, x] = True
        y2 = (y + int(rng.integers(4, h - 1))) % h
        x2 = (x + int(rng.integers(4, w - 1))) % w
        for i in range(int(rng.integers(3, w + 1))):
            walls[y2, (x + i) % w] = True
        for i in range(int(rng.integers(3, h + 1))):
            walls[(y + i) % h, x2] = True
    return walls


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _ports(components, h, w):
    return [[((r + 1) % h, (c + 1) % w) for r, c in cells]
            for cells in components]


def _make_pair(rng, profile, used):
    """Bounded rejection only for visible properties of a prospective example."""
    for _ in range(1800):
        minimum = 6 if profile == "general" else 7
        h, w = int(rng.integers(minimum, 11)), int(rng.integers(minimum, 11))
        walls = _wall_mask(rng, h, w)
        if walls.sum() < 4 or walls.sum() > walls.size - 10:
            continue
        safe = _sheltered(walls)
        labels, components = _components(safe)
        if not components:
            continue
        ports = _ports(components, h, w)
        inactive = [(int(r), int(c)) for r, c in np.argwhere(~walls)
                    if labels[(int(r) - 1) % h, (int(c) - 1) % w] < 0]
        selected = []
        if profile == "crossing":
            if len(components) < 2 or not inactive:
                continue
            planar, _ = _components(safe, periodic=False)
            choices = []
            for i, cells in enumerate(components):
                edge_ports = [p for p in ports[i] if p[0] == 0 or p[1] == 0]
                if (len(cells) >= 5 and edge_ports
                        and len({int(planar[p]) for p in cells}) >= 2):
                    choices.append((i, edge_ports))
            if not choices:
                continue
            _, edge_ports = _choose(rng, choices)
            selected = [_choose(rng, edge_ports), _choose(rng, inactive)]
        elif profile == "competition":
            small = [i for i, cells in enumerate(components) if 1 <= len(cells) <= 4]
            large = [i for i, cells in enumerate(components) if len(cells) >= 5]
            if not small or not large:
                continue
            one, two = _choose(rng, small), _choose(rng, large)
            indices = rng.choice(len(ports[two]), size=2, replace=False)
            selected = [_choose(rng, ports[one])] + [ports[two][int(i)] for i in indices]
        elif profile == "test":
            if len(components) < 2:
                continue
            multiple = [i for i, cells in enumerate(components) if len(cells) >= 2]
            if not multiple:
                continue
            contested = _choose(rng, multiple)
            unique = _choose(rng, [i for i in range(len(components)) if i != contested])
            indices = rng.choice(len(ports[contested]), size=2, replace=False)
            selected = [_choose(rng, ports[unique])] + [ports[contested][int(i)] for i in indices]
            count = int(rng.integers(4, 6))
            # More independent reservoirs can fill, while the chosen one stays contested.
            unused_components = [i for i in range(len(components)) if i not in (unique, contested)]
            if unused_components and rng.random() < .55:
                selected.append(_choose(rng, ports[_choose(rng, unused_components)]))
            extra_options = inactive + ports[contested]
            for _ in range(count - len(selected)):
                options = [p for p in extra_options if p not in selected]
                if not options:
                    break
                selected.append(_choose(rng, options))
            if len(selected) != count:
                continue
        else:
            count = int(rng.integers(1, 4))
            unique = int(rng.integers(len(components)))
            selected = [_choose(rng, ports[unique])]
            available = [(int(r), int(c)) for r, c in np.argwhere(~walls)
                         if labels[(int(r) - 1) % h, (int(c) - 1) % w] != unique]
            if len(available) < count - 1:
                continue
            if count > 1:
                selected += [available[int(i)] for i in rng.choice(
                    len(available), size=count - 1, replace=False)]
        # Palette is entirely observable in this input, and can vary per pair.
        palette = [int(c) for c in rng.permutation(np.arange(1, 10))]
        grid = np.zeros((h, w), dtype=np.int8)
        grid[walls] = palette[0]
        rng.shuffle(selected)
        for p, color in zip(selected, palette[1:]):
            grid[p] = color
        if _key(grid) in used:
            continue
        tile = _complete(grid)
        if not np.any((grid == 0) & (tile != 0)):
            continue
        return {"input": grid, "output": np.tile(tile, (3, 3))}
    raise RuntimeError(f"bounded sampler exhausted for {profile}")


def _key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    ntrain = _integer(num_train, "num_train", 2, 10)
    ntest = _integer(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    # Two demonstrations must distinguish every supported competing hypothesis.
    # This check does not use test inputs or labels.
    for _ in range(240):
        a = _make_pair(rng, "crossing", set())
        b = _make_pair(rng, "competition", {_key(a["input"])})
        train = [a, b]
        if not _compatible_alternatives(train):
            break
    else:
        raise RuntimeError("could not obtain diagnostic training demonstrations")
    used = {_key(p["input"]) for p in train}
    for _ in range(ntrain - 2):
        pair = _make_pair(rng, "general", used)
        used.add(_key(pair["input"]))
        train.append(pair)
    rng.shuffle(train)
    test = []
    for _ in range(ntest):
        pair = _make_pair(rng, "test", used)
        used.add(_key(pair["input"]))
        test.append(pair)
    return {"train": train, "test": test}


def _features(grid):
    wall, sources = _roles(grid)
    safe = _sheltered(grid == wall)
    labels, components = _components(safe)
    arrivals, missed = _arrivals(sources, labels)
    planar, _ = _components(safe, periodic=False)
    single = [i for i, taps in enumerate(arrivals) if len(taps) == 1]
    return {
        "wall": wall, "sources": sources, "safe": safe, "labels": labels,
        "components": components, "arrivals": arrivals, "missed": missed,
        "filled_sizes": [len(components[i]) for i in single],
        "contested": sum(len(taps) >= 2 for taps in arrivals),
        "untapped": sum(not taps for taps in arrivals),
        "wrapped_success": any((r == 0 or c == 0) for i in single
                               for r, c, _ in arrivals[i]),
        "crossing_fill": any(len({int(planar[p]) for p in components[i]}) > 1
                             for i in single),
    }


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
    seen, train_features = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            h, w = grid.shape
            assert 6 <= h <= 10 and 6 <= w <= 10
            assert output.shape == (3 * h, 3 * w)
            assert _key(grid) not in seen
            seen.add(_key(grid))
            f = _features(grid)
            assert int(np.sum(grid == f["wall"])) >= 4
            count = len(f["sources"])
            assert (1 <= count <= 3) if split == "train" else (4 <= count <= 5)
            tile = output[:h, :w]
            assert np.array_equal(output, np.tile(tile, (3, 3)))
            assert np.array_equal(output, solve(grid))
            assert np.array_equal(tile[grid != 0], grid[grid != 0])
            changed = tile != grid
            assert np.any(changed) and np.all(grid[changed] == 0)
            assert np.all(f["safe"][changed])
            # Check reservoir ownership and each pixel, not just solver equality.
            for i, cells in enumerate(f["components"]):
                taps = f["arrivals"][i]
                for r, c in cells:
                    if grid[r, c] == 0:
                        assert tile[r, c] == (taps[0][2] if len(taps) == 1 else 0)
            assert set(map(int, np.unique(output))) == set(map(int, np.unique(grid)))
            if split == "train":
                train_features.append(f)
            else:
                assert f["contested"] >= 1 and f["filled_sizes"]
    assert any(f["missed"] for f in train_features)
    assert any(f["untapped"] for f in train_features)
    assert any(f["contested"] for f in train_features)
    assert any(f["wrapped_success"] for f in train_features)
    assert any(f["crossing_fill"] for f in train_features)
    assert any(any(s <= 4 for s in f["filled_sizes"]) for f in train_features)
    assert any(any(s >= 5 for s in f["filled_sizes"]) for f in train_features)
    assert not _compatible_alternatives(puzzle["train"])


def _signature(puzzle):
    return tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                 for split in ("train", "test") for p in puzzle[split])


def _unit_checks():
    # A corner wall affects the opposite corner and edges, including diagonals.
    walls = np.zeros((7, 8), dtype=bool)
    walls[0, 0] = True
    safe = _sheltered(walls)
    assert not safe[-1, -1] and not safe[0, -1] and not safe[-1, 0]
    assert safe[2, 2] and int(safe.sum()) == 7 * 8 - 9
    # Diagonal contact is not flood adjacency; orthogonal seam contact is.
    mask = np.zeros((7, 8), dtype=bool)
    mask[2, 2] = mask[3, 3] = mask[0, 6] = mask[-1, 6] = True
    labels, groups = _components(mask)
    assert len(groups) == 3 and labels[0, 6] == labels[-1, 6]
    assert labels[2, 2] != labels[3, 3]
    # Literal expected result: two horizontal reservoirs between wall rows.
    grid = np.zeros((10, 8), dtype=np.int8)
    grid[[0, 5], :] = 7
    grid[3, 3] = 2  # tap (2,2), upper reservoir
    grid[2, 5] = 3  # inside upper reservoir, but its own tap misses; preserve it
    expected = grid.copy()
    expected[2:4, :] = 2
    expected[grid != 0] = grid[grid != 0]
    assert np.array_equal(solve(grid), np.tile(expected, (3, 3)))
    grid[4, 5] = 4  # tap (3,4), now contested
    assert np.array_equal(solve(grid), np.tile(grid, (3, 3)))
    grid[8, 1] = 6  # tap (7,0), activates only the lower reservoir
    expected = grid.copy()
    expected[7:9, :] = 6
    expected[grid != 0] = grid[grid != 0]
    assert np.array_equal(solve(grid), np.tile(expected, (3, 3)))


def self_test():
    """Bounded checks with measured structural coverage over exactly 42 seeds."""
    _unit_checks()
    signatures, shapes, structures, wall_colors, beacon_counts = set(), set(), set(), set(), set()
    sizes, density_bins, category = set(), set(), Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                f = _features(grid)
                shapes.add(grid.shape)
                walls = grid == f["wall"]
                structures.add((grid.shape, walls.tobytes(), (grid != 0).tobytes()))
                wall_colors.add(f["wall"])
                beacon_counts.add(len(f["sources"]))
                sizes.update(map(len, f["components"]))
                density_bins.add(int(10 * walls.mean()))
                category["rectangular"] += grid.shape[0] != grid.shape[1]
                category["crossing_fill"] += f["crossing_fill"]
                category["contested"] += f["contested"] > 0
                category["multiple_fills"] += len(f["filled_sizes"]) > 1
                category["missed"] += bool(f["missed"])
                category["full_output_axis"] += 10 in grid.shape
    assert len(signatures) == 42 and len(shapes) >= 15
    assert len(structures) >= 190 and len(wall_colors) == 9
    assert beacon_counts == {1, 2, 3, 4, 5}
    assert {1, 2, 3, 4} <= sizes and max(sizes) >= 20
    assert len(density_bins) >= 4
    assert all(category[k] >= n for k, n in (
        ("rectangular", 70), ("crossing_fill", 42), ("contested", 84),
        ("multiple_fills", 5), ("missed", 50), ("full_output_axis", 40)))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(10000 + 100 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
            validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.0, 1), (True, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _signature(generate_puzzle(817, 7, 3)) == _signature(generate_puzzle(817, 7, 3))
    assert _signature(generate_puzzle(817)) != _signature(generate_puzzle(818))
    return {"seeds_checked": 42, "distinct_puzzles": len(signatures),
            "input_shapes": sorted(shapes), "structural_signatures": len(structures),
            "wall_colors": sorted(wall_colors), "beacon_counts": sorted(beacon_counts),
            "component_sizes": sorted(sizes), "wall_density_deciles": sorted(density_bins),
            "features": dict(category), "independent_solver": False}


if __name__ == "__main__":
    report = self_test()
    Path("scratch").mkdir(exist_ok=True)
    Path("scratch/self-test.json").write_text(json.dumps(report, separators=(",", ":")) + "\n")
    print(json.dumps(report, separators=(",", ":")))
