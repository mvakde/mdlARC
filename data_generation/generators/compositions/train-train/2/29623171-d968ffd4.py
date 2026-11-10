"""Dominant beacons in a folded corridor. Standard library and NumPy only."""

from itertools import combinations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=False)


def _candidate(a, background, wall, door):
    """Parse a possible door color; all coordinates below include the halo."""
    inner = a[1:-1, 1:-1]
    structural = (inner == wall) | (inner == door)
    rr = np.flatnonzero(np.all(structural, axis=1)) + 1
    cc = np.flatnonzero(np.all(structural, axis=0)) + 1
    assert len(rr) >= 2 and len(cc) >= 2
    assert rr[0] == cc[0] == 1
    assert rr[-1] == a.shape[0] - 2 and cc[-1] == a.shape[1] - 2
    assert len(set(np.diff(rr))) == len(set(np.diff(cc))) == 1
    height, width = int(rr[1] - rr[0] - 1), int(cc[1] - cc[0] - 1)
    assert 2 <= height <= 6 and 2 <= width <= 6
    boxes, counts, colors = {}, {}, {}
    for r in range(len(rr) - 1):
        for c in range(len(cc) - 1):
            box = (int(rr[r] + 1), int(rr[r + 1]),
                   int(cc[c] + 1), int(cc[c + 1]))
            r0, r1, c0, c1 = box
            tile = a[r0:r1, c0:c1]
            if np.all(tile == wall):
                continue
            assert not np.any((tile == wall) | (tile == door))
            payload = set(int(x) for x in np.unique(tile)) - {background}
            assert len(payload) <= 1
            node = (r, c)
            boxes[node] = box
            counts[node] = int(np.count_nonzero(tile != background))
            colors[node] = next(iter(payload)) if payload else None
    assert len(boxes) >= 2 and max(counts.values()) > 0
    # Cropping to the used coarse bounding box forbids ambiguous all-wall bands.
    assert {r for r, c in boxes} == set(range(len(rr) - 1))
    assert {c for r, c in boxes} == set(range(len(cc) - 1))
    adjacency = {node: [] for node in boxes}
    edge_set = set()
    row_index = {int(x): i for i, x in enumerate(rr)}
    col_index = {int(x): i for i, x in enumerate(cc)}
    for r, c in np.argwhere(a == door):
        r, c = int(r), int(c)
        if r in row_index and c not in col_index:
            i, j = row_index[r], int(np.searchsorted(cc, c) - 1)
            assert 0 < i < len(rr) - 1 and 0 <= j < len(cc) - 1
            ends = ((i - 1, j), (i, j))
        elif c in col_index and r not in row_index:
            i, j = int(np.searchsorted(rr, r) - 1), col_index[c]
            assert 0 < j < len(cc) - 1 and 0 <= i < len(rr) - 1
            ends = ((i, j - 1), (i, j))
        else:
            raise AssertionError("door must be on a shared wall, away from intersections")
        assert all(node in boxes for node in ends)
        edge = tuple(sorted(ends))
        assert edge not in edge_set, "exactly one door per connected room pair"
        edge_set.add(edge)
        x, y = ends
        adjacency[x].append(y)
        adjacency[y].append(x)
    assert len(edge_set) == len(boxes) - 1
    assert all(1 <= len(v) <= 2 for v in adjacency.values())
    ends = sorted(node for node, v in adjacency.items() if len(v) == 1)
    assert len(ends) == 2
    path, previous, node = [], None, ends[0]
    while node is not None:
        assert node not in path
        path.append(node)
        following = [v for v in adjacency[node] if v != previous]
        previous, node = node, (following[0] if following else None)
    assert len(path) == len(boxes), "room graph must be one connected path"
    return {"background": background, "wall": wall, "door": door,
            "boxes": boxes, "counts": counts, "colors": colors,
            "adjacency": adjacency, "path": path,
            "geometry": (len(rr) - 1, len(cc) - 1, height, width)}


def _analyze(value):
    a = _grid(value)
    if min(a.shape) < 6:
        raise ValueError("missing framed rooms")
    background, wall = int(a[0, 0]), int(a[1, 1])
    if background == wall:
        raise ValueError("halo and wall must differ")
    if not (np.all(a[0] == background) and np.all(a[-1] == background)
            and np.all(a[:, 0] == background) and np.all(a[:, -1] == background)):
        raise ValueError("expected a one-cell background halo")
    inner = a[1:-1, 1:-1]
    if not (np.all(inner[0] == wall) and np.all(inner[-1] == wall)
            and np.all(inner[:, 0] == wall) and np.all(inner[:, -1] == wall)):
        raise ValueError("expected a closed outer wall")
    candidates = []
    for door in sorted(set(int(v) for v in np.unique(a)) - {background, wall}):
        try:
            candidates.append(_candidate(a, background, wall, door))
        except (AssertionError, ValueError, IndexError):
            continue
    if len(candidates) != 1:
        raise ValueError("room and door structure is ambiguous or invalid")
    return a, candidates[0]


def _paint_room(out, info, node, color):
    r0, r1, c0, c1 = info["boxes"][node]
    out[r0:r1, c0:c1] = color


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, info = _analyze(input_grid)
    out = a.copy()
    path, counts = info["path"], info["counts"]
    maximum = max(counts.values())
    sources = [i for i, node in enumerate(path) if counts[node] == maximum]
    for node in path:
        _paint_room(out, info, node, info["background"])
    for i in sources:
        _paint_room(out, info, path[i], info["colors"][path[i]])
    # Bisection is in room steps, through doors, regardless of pixel position.
    for left, right in zip(sources, sources[1:]):
        half = (right - left - 1) // 2
        for i in range(left + 1, left + half + 1):
            _paint_room(out, info, path[i], info["colors"][path[left]])
        for i in range(right - half, right):
            _paint_room(out, info, path[i], info["colors"][path[right]])
    return out


def _path(rng, length, height, width):
    """Random self-avoiding corridor; nearby rooms need not have a door."""
    max_r, max_c = min(6, 27 // (height + 1)), min(6, 27 // (width + 1))
    boards = [(r, c) for r in range(2, max_r + 1)
              for c in range(2, max_c + 1) if r * c >= length + 1]
    for _ in range(32):
        nr, nc = boards[int(rng.integers(len(boards)))]
        start = (int(rng.integers(nr)), int(rng.integers(nc)))
        # Bounded randomized depth-first search with backtracking.
        path, visited = [start], {start}
        choices = []
        def neighbors(node):
            r, c = node
            return [v for v in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                    if 0 <= v[0] < nr and 0 <= v[1] < nc]
        choices.append([neighbors(start)[i] for i in rng.permutation(len(neighbors(start)))])
        for step in range(6000):
            if len(path) == length:
                r0, c0 = min(r for r, c in path), min(c for r, c in path)
                return [(r - r0, c - c0) for r, c in path]
            while choices[-1] and choices[-1][-1] in visited:
                choices[-1].pop()
            if choices[-1]:
                node = choices[-1].pop()
                visited.add(node)
                path.append(node)
                ns = neighbors(node)
                choices.append([ns[i] for i in rng.permutation(len(ns))])
            else:
                visited.remove(path.pop())
                choices.pop()
                if not path:
                    break
    raise RuntimeError("bounded corridor search failed")


def _positions(rng, height, width, count, mode):
    cells = [(r, c) for r in range(height) for c in range(width)]
    if not count:
        return []
    if mode == "random":
        return [cells[int(i)] for i in rng.choice(len(cells), count, replace=False)]
    anchor = cells[int(rng.integers(len(cells)))]
    if mode == "compact":
        return sorted(cells, key=lambda x: (abs(x[0] - anchor[0]) + abs(x[1] - anchor[1]),
                                             float(rng.random())))[:count]
    selected = [anchor]
    cells.remove(anchor)
    while len(selected) < count:
        distances = [min(abs(r - x) + abs(c - y) for x, y in selected) for r, c in cells]
        options = [i for i, d in enumerate(distances) if d == max(distances)]
        selected.append(cells.pop(int(rng.choice(options))))
    return selected


def _sources(rng, length, mode):
    if mode == "low":
        return [int(rng.integers(0, 2)), length - 1 - int(rng.integers(0, 2))]
    if mode == "high":
        options = [s for s in combinations(range(length), 3)
                   if min(np.diff(s)) >= 3
                   and {(s[1] - s[0] - 1) % 2, (s[2] - s[1] - 1) % 2} == {0, 1}
                   and (s[0] > 0 or s[-1] < length - 1)]
        return list(options[int(rng.integers(len(options)))])
    if mode == "test":
        count = int(rng.integers(4, 7))
    else:
        count = int(rng.choice([1, 2, 2, 3, 3, 4, 5]))
    return sorted(int(x) for x in rng.choice(length, min(length, count), replace=False))


def _make_pair(rng, height, width, maximum, mode, contrast=None):
    length = int(rng.integers(*( {"low": (8, 14), "high": (9, 15),
                                 "test": (16, 24), "extra": (6, 16)}[mode])))
    path = _path(rng, length, height, width)
    sources = _sources(rng, length, mode)
    palette = [int(x) for x in rng.permutation(10)]
    background, wall, door = palette[:3]
    payloads = palette[3:3 + int(rng.integers(2, 6))]
    counts = [int(rng.integers(maximum)) for _ in path]
    colors = [int(rng.choice(payloads)) for _ in path]
    for i in sources:
        counts[i] = maximum
    if mode == "low":
        colors[sources[0]], colors[sources[1]] = payloads[:2]
    elif mode == "high":
        # A same-color odd gap shows that midpoint blanking is geometric.
        odd = next(j for j in range(2) if (sources[j + 1] - sources[j] - 1) % 2)
        for i in sources:
            colors[i] = payloads[1]
        colors[sources[odd]] = colors[sources[odd + 1]] = payloads[0]
        midpoint = (sources[odd] + sources[odd + 1]) // 2
        counts[midpoint] = contrast
    elif len(sources) >= 2:
        colors[sources[0]], colors[sources[-1]] = payloads[:2]
    nr, nc = max(r for r, c in path) + 1, max(c for r, c in path) + 1
    shape = (nr * (height + 1) + 3, nc * (width + 1) + 3)
    inp = np.full(shape, background, dtype=np.int8)
    inp[1:-1, 1:-1] = wall
    out = inp.copy()
    for i, (r, c) in enumerate(path):
        top, left = 2 + r * (height + 1), 2 + c * (width + 1)
        inp[top:top + height, left:left + width] = background
        out[top:top + height, left:left + width] = background
        placement = ("compact", "dispersed", "random")[int(rng.integers(3))]
        for x, y in _positions(rng, height, width, counts[i], placement):
            inp[top + x, left + y] = colors[i]
        if sources[0] <= i <= sources[-1]:
            distances = [abs(i - j) for j in sources]
            nearest = [j for j, d in zip(sources, distances) if d == min(distances)]
            if len(nearest) == 1:
                out[top:top + height, left:left + width] = colors[nearest[0]]
    for (r, c), (s, t) in zip(path, path[1:]):
        if r == s:
            x = 2 + r * (height + 1) + int(rng.integers(height))
            y = 1 + max(c, t) * (width + 1)
        else:
            x = 1 + max(r, s) * (height + 1)
            y = 2 + c * (width + 1) + int(rng.integers(width))
        inp[x, y] = out[x, y] = door
    return {"input": inp, "output": out}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _winners(info):
    maximum = max(info["counts"].values())
    return [i for i, node in enumerate(info["path"]) if info["counts"][node] == maximum]


def _geometric_disagreement(info):
    """A physical Manhattan-nearest shortcut on the same active interval."""
    sources, path = _winners(info), info["path"]
    _, _, height, width = info["geometry"]
    for i in range(sources[0], sources[-1] + 1):
        distances = [abs(i - s) for s in sources]
        near = [s for s, d in zip(sources, distances) if d == min(distances)]
        expected = info["colors"][path[near[0]]] if len(near) == 1 else info["background"]
        r, c = path[i]
        physical = [abs(r - path[s][0]) * (height + 1)
                    + abs(c - path[s][1]) * (width + 1) for s in sources]
        other = [s for s, d in zip(sources, physical) if d == min(physical)]
        shortcut = info["colors"][path[other[0]]] if len(other) == 1 else info["background"]
        if shortcut != expected:
            return True
    return False


def _checked_integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < low or (high is not None and value > high)):
        raise ValueError(f"illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _checked_integer(seed, "seed", 0)
    num_train = _checked_integer(num_train, "num_train", 2, 10)
    num_test = _checked_integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    geometry_low = tuple(int(x) for x in rng.integers(2, 5, size=2))
    geometry_high = tuple(int(x) for x in rng.integers(2, 5, size=2))
    cap_low, cap_high = int(np.prod(geometry_low)), int(np.prod(geometry_high))
    low_max = int(rng.integers(2, min(cap_low, cap_high - 1) + 1))
    high_max = int(rng.integers(low_max + 1, cap_high + 1))
    core = [("low", geometry_low, low_max), ("high", geometry_high, high_max)]
    if bool(rng.integers(2)):
        core.reverse()
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train" and index < 2:
                mode, (height, width), maximum = core[index]
            else:
                mode = "test" if split == "test" else "extra"
                height, width = (int(x) for x in rng.integers(2, 5, size=2))
                maximum = int(rng.integers(1, height * width + 1))
            for attempt in range(256):
                pair = _make_pair(rng, height, width, maximum, mode, low_max)
                _, info = _analyze(pair["input"])
                key = _key(pair["input"])
                if key in seen or (mode == "low" and not _geometric_disagreement(info)):
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("could not construct a distinct diagnostic corridor")
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_semantics(inp, out, info):
    """Check room graph distances and structure, independently of solve's splits."""
    assert inp.shape == out.shape
    editable = np.zeros(inp.shape, dtype=bool)
    for r0, r1, c0, c1 in info["boxes"].values():
        editable[r0:r1, c0:c1] = True
    assert np.array_equal(inp[~editable], out[~editable]), "walls, doors, halo must stay"
    path, sources = info["path"], _winners(info)
    distance_maps = []
    for source in sources:
        distances = {path[source]: 0}
        queue = [path[source]]
        for node in queue:
            for neighbor in info["adjacency"][node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        distance_maps.append(distances)
    extended, cleared, midpoints = 0, 0, 0
    for i, node in enumerate(path):
        expected = info["background"]
        if sources[0] <= i <= sources[-1]:
            ds = [d[node] for d in distance_maps]
            nearest = [s for s, d in zip(sources, ds) if d == min(ds)]
            if len(nearest) == 1:
                expected = info["colors"][path[nearest[0]]]
                extended += i not in sources
            else:
                midpoints += 1
        else:
            cleared += 1
        r0, r1, c0, c1 = info["boxes"][node]
        assert np.all(out[r0:r1, c0:c1] == expected), "whole-room distance ownership"
    assert np.array_equal(solve(inp), out)
    return {"extended": extended, "tails": cleared, "midpoints": midpoints}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, infos, metrics = set(), {"train": [], "test": []}, {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, info = _analyze(pair["input"])
            out = _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            infos[split].append(info)
            metrics[split].append(_check_semantics(inp, out, info))
            assert not np.array_equal(inp, out)
    first = infos["train"][:2]
    low, high = sorted(first, key=lambda x: max(x["counts"].values()))
    m, big = max(low["counts"].values()), max(high["counts"].values())
    assert m < big and m in high["counts"].values(), "winning-then-losing count contrast"
    assert len(_winners(low)) == 2 and len(_winners(high)) == 3
    assert _geometric_disagreement(low), "teach door distance rather than physical closeness"
    high_sources = _winners(high)
    assert {(b - a - 1) % 2 for a, b in zip(high_sources, high_sources[1:])} == {0, 1}
    assert any((b - a - 1) % 2 == 1
               and high["colors"][high["path"][a]] == high["colors"][high["path"][b]]
               for a, b in zip(high_sources, high_sources[1:])), "same-color midpoint evidence"
    assert any(x["tails"] for x in metrics["train"][:2])
    assert all(x["extended"] > 0 for x in metrics["train"][:2]), "B ablation must change labels"
    # Ignoring A and activating every marked room fills every such room in its
    # original color. The three-source core has a genuine blank midpoint or tail.
    high_index = next(i for i, info in enumerate(first) if info is high)
    high_output = np.asarray(puzzle["train"][high_index]["output"])
    assert any(high["counts"][node] > 0 and
               np.all(high_output[
                   high["boxes"][node][0]:high["boxes"][node][1],
                   high["boxes"][node][2]:high["boxes"][node][3]] == high["background"])
               for node in high["path"]), "A ablation must change labels"
    assert all(len(x["path"]) > max(len(t["path"]) for t in infos["train"])
               for x in infos["test"]), "test shift is a longer connected corridor"
    assert all(4 <= len(_winners(x)) <= 6 for x in infos["test"])


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded 42-seed structural, semantic, metamorphic and API checks."""
    shapes, layouts, palettes, maxima, source_counts, gaps = (set() for _ in range(6))
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, info = _analyze(pair["input"])
                shapes.add(inp.shape)
                layouts.add((info["geometry"], tuple(info["path"])))
                palettes.add((info["background"], info["wall"], info["door"]))
                maxima.add(max(info["counts"].values()))
                sources = _winners(info)
                source_counts.add(len(sources))
                gaps.update(b - a - 1 for a, b in zip(sources, sources[1:]))
                # Rotation and reflection must not change a topological rule.
                for transformed_in, transformed_out in (
                        (np.rot90(inp), np.rot90(pair["output"])),
                        (np.fliplr(inp), np.fliplr(pair["output"]))):
                    assert np.array_equal(solve(transformed_in), transformed_out)
        # Rearranging dots within a room preserves its evidence and its answer.
        pair = puzzle["train"][0]
        shuffled, info = _analyze(pair["input"].copy())
        for r0, r1, c0, c1 in info["boxes"].values():
            tile = shuffled[r0:r1, c0:c1]
            tile[:] = np.flip(tile.copy())
        assert np.array_equal(solve(shuffled), pair["output"])
    assert len(set(puzzles)) == 42
    assert len(shapes) >= 30 and len(layouts) >= 150 and len(palettes) >= 80
    assert {1, 2, 3, 4}.issubset(maxima) and max(maxima) >= 12
    assert {1, 2, 3, 4, 5, 6}.issubset(source_counts)
    assert {0, 1, 2, 3, 4}.issubset(gaps) and max(gaps) >= 7
    assert _puzzle_key(generate_puzzle(0)) == puzzles[0]
    # Hand-specified straight boundary cases also exercise unsampled 6-cell
    # room dimensions, all-solid/all-source identity, g=0, g=1, and one source.
    fixtures = (
        (6, 2, [12, 12, 12, 12], [1, 2, 1, 2], [1, 2, 1, 2]),
        (2, 6, [1, 0, 0], [1, 2, 2], [1, 0, 0]),
        (3, 3, [2, 1, 2], [1, 2, 1], [1, 0, 1]),
        (2, 3, [2, 1, 0, 2], [1, 2, 2, 2], [1, 1, 2, 2]),
        (2, 3, [2, 2, 1], [1, 2, 1], [1, 2, 0]),
        (3, 2, [1, 3, 1, 2, 3, 0], [2, 1, 2, 2, 2, 1], [0, 1, 1, 2, 2, 0]),
    )
    for height, width, counts, colors, labels in fixtures:
        inp = np.zeros((height + 4, len(counts) * (width + 1) + 3), dtype=np.int8)
        inp[1:-1, 1:-1] = 5
        expected = inp.copy()
        for i, (count, color, label) in enumerate(zip(counts, colors, labels)):
            left = 2 + i * (width + 1)
            tile = np.zeros((height, width), dtype=np.int8)
            tile.flat[:count] = color
            inp[2:2 + height, left:left + width] = tile
            expected[2:2 + height, left:left + width] = label
            if i:
                x, y = 2 + i % height, left - 1
                inp[x, y] = expected[x, y] = 9
        assert np.array_equal(solve(inp), expected)
        assert np.array_equal(solve(np.rot90(inp)), np.rot90(expected))
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(400 + nt * 10 + ns, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, None), (4.0, 1), (4, "1")):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, room-distance semantics and all legal counts")
