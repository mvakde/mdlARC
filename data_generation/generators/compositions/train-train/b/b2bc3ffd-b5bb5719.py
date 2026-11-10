"""Area-paced echoes of a queue which loses its last two objects each tick.

Only NumPy and the standard library are used.  Coordinates below are canonical:
the rail is the bottom edge, and the queue is read from left to right.
"""

from __future__ import annotations

import numpy as np


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= s <= 30 for s in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((0 <= g) & (g <= 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return g.astype(np.int8, copy=True)


def _components(g, colors):
    seen, objects = set(), []
    h, w = g.shape
    for r, c in np.ndindex(g.shape):
        color = int(g[r, c])
        if color not in colors or (r, c) in seen:
            continue
        stack, cells = [(r, c)], []
        seen.add((r, c))
        while stack:
            rr, cc = stack.pop()
            cells.append((rr, cc))
            for y, x in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if (0 <= y < h and 0 <= x < w and (y, x) not in seen
                        and int(g[y, x]) == color):
                    seen.add((y, x))
                    stack.append((y, x))
        objects.append((color, tuple(sorted(cells))))
    return objects


def _shape_key(cells):
    bottom = max(r for r, _ in cells)
    left = min(c for _, c in cells)
    return tuple(sorted((r - bottom, c - left) for r, c in cells))


def _parse(value):
    g = _grid(value)
    candidates = []
    for k in range(4):
        canonical = np.rot90(g, -k)
        rail = int(canonical[-1, 0])
        if (np.all(canonical[-1] == rail)
                and np.count_nonzero(canonical == rail) == canonical.shape[1]):
            candidates.append((k, canonical.copy(), rail))
    if len(candidates) != 1:
        raise ValueError("input must have one exclusive, uniform full-edge rail")
    k, g, rail = candidates[0]
    bg = int(g[0, 0])
    signals = sorted(set(int(v) for v in np.unique(g)) - {bg, rail})
    if bg == rail or not np.all(g[0] == bg) or len(signals) != 2:
        raise ValueError("blank opposite edge and two visible signal colors required")
    objects = _components(g, set(signals))
    objects.sort(key=lambda o: min(c for _, c in o[1]))
    if len(objects) not in (3, 5, 7, 9):
        raise ValueError("queue must have 3, 5, 7, or 9 objects")
    previous_right = -2
    for _, cells in objects:
        left, right = min(c for _, c in cells), max(c for _, c in cells)
        if (not 1 <= len(cells) <= 5 or max(r for r, _ in cells) != g.shape[0] - 2
                or left < previous_right + 2):
            raise ValueError("objects must rest on the rail in separated column spans")
        previous_right = right
    return {"grid": g, "rotation": k, "background": bg, "rail": rail,
            "signals": tuple(signals), "objects": objects}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct the changing queue from this input alone, retaining its history."""
    info = _parse(input_grid)
    out = info["grid"].copy()
    a, b = info["signals"]
    current = [(color, list(cells)) for color, cells in info["objects"]]
    while len(current) > 1:
        # The sequence operation and object motions happen in the same tick.
        current = current[:-2]
        next_frame = []
        for color, cells in current:
            area = len(cells)
            moved = [(r - area, c + 1) for r, c in cells]
            phase = b if color == a else a
            for r, c in moved:
                if not (0 <= r < out.shape[0] - 1 and 0 <= c < out.shape[1]):
                    raise ValueError("echo leaves the canvas")
                if out[r, c] != info["background"]:
                    raise ValueError("echoes overlap")
                out[r, c] = phase
            next_frame.append((phase, moved))
        current = next_frame
    return np.rot90(out, info["rotation"]).copy()


def _stamps(info):
    """Closed form used by construction/validation, separately from solve's ticks."""
    a, b = info["signals"]
    n = len(info["objects"])
    result = []
    for index, (color, cells) in enumerate(info["objects"]):
        lifetime = (n - 1 - index) // 2
        for tick in range(lifetime + 1):
            phase = color if tick % 2 == 0 else (b if color == a else a)
            footprint = tuple((r - tick * len(cells), c + tick) for r, c in cells)
            result.append((phase, footprint, index, tick))
    return result


def _render_checked(info):
    g, bg = info["grid"], info["background"]
    out = np.full_like(g, bg)
    out[-1] = info["rail"]
    owner = {}
    stamps = _stamps(info)
    for stamp_id, (color, cells, _, _) in enumerate(stamps):
        for r, c in cells:
            if not (0 <= r < g.shape[0] - 1 and 0 <= c < g.shape[1]):
                raise ValueError("stamp out of bounds")
            if (r, c) in owner:
                raise ValueError("stamp collision")
            owner[r, c] = (stamp_id, color)
            out[r, c] = color
    # Same-color echoes stay separate monochromatic components. Different-color
    # neighbors are allowed; touching footprints never overlap or occlude cells.
    for (r, c), (stamp_id, color) in owner.items():
        for pos in ((r + 1, c), (r, c + 1)):
            if pos in owner and owner[pos][0] != stamp_id and owner[pos][1] == color:
                raise ValueError("same-color stamps merge")
    return out


def _alternative(info, *, distance="area", crop="prefix", flip=True, drift=True):
    """Concrete tempting shortcuts, used only to audit demonstration evidence.

    Alternative writes outside the canvas are ignored. This makes a visible
    in-bounds disagreement necessary; mere invalid boundary behavior is not used
    as evidence against a shortcut.
    """
    out = info["grid"].copy()
    n = len(info["objects"])
    a, b = info["signals"]
    for tick in range(1, (n + 1) // 2):
        indices = (range(n - 2 * tick) if crop == "prefix" else
                   range(tick, n - tick) if crop == "symmetric" else range(n))
        for index in indices:
            color, cells = info["objects"][index]
            height = max(r for r, _ in cells) - min(r for r, _ in cells) + 1
            width = max(c for _, c in cells) - min(c for _, c in cells) + 1
            step = {"area": len(cells), "unit": 1, "height": height,
                    "box": height * width}[distance]
            phase = (b if color == a else a) if flip and tick % 2 else color
            for r, c in cells:
                rr, cc = r - tick * step, c + (tick if drift else 0)
                if 0 <= rr < out.shape[0] - 1 and 0 <= cc < out.shape[1]:
                    out[rr, cc] = phase
    return out


_SHORTCUTS = (
    {"distance": "unit"}, {"distance": "height"}, {"distance": "box"},
    {"crop": "symmetric"}, {"crop": "none"}, {"flip": False}, {"drift": False},
)


def _random_shape(rng, area, kind=None):
    if kind is None:
        kind = str(rng.choice(("horizontal", "vertical", "hook", "grow", "grow")))
    if kind == "horizontal":
        cells = [(0, c) for c in range(area)]
    elif kind == "vertical":
        cells = [(-r, 0) for r in range(area)]
    elif kind == "hook" and area >= 3:
        arm = int(rng.integers(2, area))
        cells = [(0, c) for c in range(arm)] + [(-r, 0) for r in range(1, area - arm + 1)]
    else:
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = sorted({(r + dr, c + dc) for r, c in cells
                               for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                               if (r + dr, c + dc) not in cells})
            cells.add(frontier[int(rng.integers(len(frontier)))])
    if rng.integers(2):
        cells = [(r, -c) for r, c in cells]
    return _shape_key(cells)


def _motif(info):
    a, _ = info["signals"]
    return tuple((int(color != a), _shape_key(cells)) for color, cells in info["objects"])


def _key(g):
    return (g.shape, g.tobytes())


def _make_pair(rng, n, max_area, mode, palette, rotation):
    bg, rail, a, b = palette
    areas = [int(v) for v in rng.integers(1, max_area + 1, size=n)]
    kinds = [None] * n
    if mode == "equal":
        areas[0] = areas[1] = int(rng.integers(3, 5))
        areas[2] = 7 - areas[0]
        kinds[:2] = ["horizontal", "vertical"]
        kinds[2] = "hook"
        if rng.integers(2):
            kinds[:2] = kinds[1::-1]
    elif mode == "contrast":
        areas[:3] = [int(v) for v in rng.choice(np.arange(1, 5), 3, replace=False)]
    shapes = [_random_shape(rng, area, kind) for area, kind in zip(areas, kinds)]
    widths = [max(c for _, c in shape) + 1 for shape in shapes]
    base = sum(widths) + n - 1
    if base > 30:
        raise ValueError("queue too wide")
    width = max(8, base) + int(rng.integers(0, min(6, 30 - max(8, base)) + 1))
    gaps = [0] + [1] * (n - 1) + [0]
    slack = width - base
    allocation = rng.multinomial(slack, np.full(n + 1, 1 / (n + 1)))
    gaps = [gap + int(extra) for gap, extra in zip(gaps, allocation)]
    minimum = max(8, max(2 - min(r for r, _ in shape) + ((n - 1 - i) // 2) * area
                         for i, (area, shape) in enumerate(zip(areas, shapes))))
    height = minimum + int(rng.integers(0, min(4, 30 - minimum) + 1))
    phase = int(rng.integers(2))
    first, second = (a, b) if phase == 0 else (b, a)
    if mode in ("equal", "contrast"):
        colors = [int(v) for v in rng.choice([a, b], size=n)]
        colors[:3] = [first, second, first]
    else:
        style = int(rng.integers(4))
        if style == 0:
            colors = [first if i % 2 == 0 else second for i in range(n)]
        elif style == 1:
            cut = int(rng.integers(1, n))
            colors = [first] * cut + [second] * (n - cut)
        elif style == 2:
            colors = [first] * (n - 2) + [second, int(rng.choice([a, b]))]
        else:
            colors = [int(v) for v in rng.choice([a, b], size=n)]
            if len(set(colors)) == 1:
                colors[int(rng.integers(n))] = second if colors[0] == first else first
    g = np.full((height, width), bg, dtype=np.int8)
    g[-1] = rail
    cursor = gaps[0]
    for index, (shape, color) in enumerate(zip(shapes, colors)):
        for r, c in shape:
            g[height - 2 + r, cursor + c] = color
        cursor += widths[index] + gaps[index + 1]
    info = _parse(g)
    out = _render_checked(info)
    # Require all geometry shortcuts to be visibly contradicted in the equal
    # comparison. In particular, a genuinely nonrectangular active shape is
    # needed somewhere to distinguish cell count from bounding-box area.
    if mode == "equal" and any(np.array_equal(_alternative(info, **kw), out)
                               for kw in _SHORTCUTS):
        raise ValueError("insufficient diagnostic evidence")
    return {"input": np.rot90(g, rotation).copy(),
            "output": np.rot90(out, rotation).copy()}


def _count(value, name, lower, upper):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lower <= int(value) <= upper):
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    palette = tuple(int(v) for v in rng.permutation(10)[:4])
    first_angle = int(rng.integers(4))
    angles = [first_angle] * num_train
    if rng.integers(2):
        angles[1] = (first_angle + int(rng.integers(1, 4))) % 4
        for i in range(2, num_train):
            angles[i] = int(rng.integers(4))
    jobs = [(5, 4, "equal"), (7, 4, "contrast")]
    jobs += [(int(rng.choice([3, 5, 7])), 4, "random") for _ in range(num_train - 2)]
    rng.shuffle(jobs)
    puzzle = {"train": [], "test": []}
    seen, motifs = set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if split == "train":
                n, max_area, mode = jobs[i]
                angle = angles[i]
            else:
                n = 9 if i == 0 else int(rng.choice([3, 5, 7, 9]))
                max_area, mode = (4 if i == 0 else 5), "random"
                angle = int(rng.choice(sorted(set(angles))))
            for _ in range(1500):
                try:
                    pair = _make_pair(rng, n, max_area, mode, palette, angle)
                except ValueError:
                    continue
                key, motif = _key(pair["input"]), _motif(_parse(pair["input"]))
                if key not in seen and motif not in motifs:
                    seen.add(key)
                    motifs.add(motif)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct collision-free queue")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, motifs, conventions = set(), set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            x, y = _grid(pair["input"]), _grid(pair["output"])
            assert x.shape == y.shape and _key(x) not in seen
            seen.add(_key(x))
            info = _parse(x)
            motif = _motif(info)
            assert motif not in motifs
            motifs.add(motif)
            conventions.add((info["background"], info["rail"], info["signals"]))
            canonical_y = np.rot90(y, -info["rotation"])
            expected = _render_checked(info)
            assert np.array_equal(canonical_y, expected)
            assert np.array_equal(solve(x), y) and not np.array_equal(x, y)
            assert np.array_equal(canonical_y[-1], info["grid"][-1])
            for color, cells in info["objects"]:
                assert all(canonical_y[r, c] == color for r, c in cells)
            n = len(info["objects"])
            # Every historical footprint is still an intact monochromatic
            # component, including the original queue and the final singleton.
            components = _components(canonical_y, set(info["signals"]))
            stamps = _stamps(info)
            assert len(components) == ((n + 1) // 2) ** 2
            assert {(color, cells) for color, cells in components} == {
                (color, tuple(sorted(cells))) for color, cells, _, _ in stamps}
            assert sum(len(cells) for _, cells in components) == sum(
                len(cells) * (1 + (n - 1 - i) // 2)
                for i, (_, cells) in enumerate(info["objects"]))
            if split == "train":
                assert n <= 7 and all(len(cells) <= 4 for _, cells in info["objects"])
            infos[split].append((info, canonical_y))
    assert len(conventions) == 1
    train = infos["train"]
    assert {5, 7} <= {len(info["objects"]) for info, _ in train}
    assert len(infos["test"][0][0]["objects"]) == 9
    assert {info["rotation"] for info, _ in infos["test"]} <= {
        info["rotation"] for info, _ in train}
    equal_shapes, color_areas = {}, {}
    for info, _ in train:
        for index, (color, cells) in enumerate(info["objects"][:-2]):
            equal_shapes.setdefault(len(cells), set()).add(_shape_key(cells))
            color_areas.setdefault(color, set()).add(len(cells))
    assert any(len(shapes) >= 2 for shapes in equal_shapes.values())
    assert len(equal_shapes) >= 3
    assert any(len(areas) >= 2 for areas in color_areas.values())
    for shortcut in _SHORTCUTS:
        assert any(not np.array_equal(_alternative(info, **shortcut), out)
                   for info, out in train), f"undistinguished shortcut: {shortcut}"
    return None


def _puzzle_key(puzzle):
    return tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded semantic, diversity, determinism, count, and equivariance checks."""
    puzzles, dimensions, profiles, shapes, palettes, rotations = set(), set(), set(), set(), set(), set()
    test_counts, areas, flags = set(), set(), set()
    for seed in range(42):
        # Exercise multi-test variation too, without changing the public defaults.
        nt = 1 + seed % 4
        nr = max(nt + 1, 2 + seed % 9)
        p = generate_puzzle(seed, nr, nt)
        assert _puzzle_key(p) == _puzzle_key(generate_puzzle(seed, nr, nt))
        puzzles.add(_puzzle_key(p))
        for split in ("train", "test"):
            for pair in p[split]:
                x, y = pair["input"], pair["output"]
                info = _parse(x)
                objects = info["objects"]
                dimensions.add(x.shape)
                profiles.add(tuple(len(cells) for _, cells in objects))
                palettes.add((info["background"], info["rail"], info["signals"]))
                rotations.add(info["rotation"])
                shapes.update(_shape_key(cells) for _, cells in objects)
                areas.update(len(cells) for _, cells in objects)
                if split == "test":
                    test_counts.add(len(objects))
                if min(c for _, cells in objects for _, c in cells) == 0:
                    flags.add("flush_source")
                if any(min(r for r, _ in cells) == 0 for _, cells, _, _ in _stamps(info)):
                    flags.add("far_edge_echo")
                if any(len(cells) < (1 - min(r for r, _ in _shape_key(cells))) *
                       (1 + max(c for _, c in _shape_key(cells))) for _, cells in objects[:-2]):
                    flags.add("nonrectangular_active")
                for k in range(4):
                    assert np.array_equal(solve(np.rot90(x, k)), np.rot90(y, k))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert len(puzzles) == 42 and len(palettes) >= 30
    assert len(dimensions) >= 35 and len(profiles) >= 100 and len(shapes) >= 25
    assert rotations == {0, 1, 2, 3} and test_counts == {3, 5, 7, 9}
    assert areas == {1, 2, 3, 4, 5}
    assert flags == {"flush_source", "far_edge_echo", "nonrectangular_active"}
    return {"seeds": 42, "distinct_puzzles": len(puzzles), "dimensions": len(dimensions),
            "area_profiles": len(profiles), "footprints": len(shapes),
            "palettes": len(palettes), "rotations": len(rotations),
            "test_object_counts": sorted(test_counts), "areas": sorted(areas)}


if __name__ == "__main__":
    print(self_test())
