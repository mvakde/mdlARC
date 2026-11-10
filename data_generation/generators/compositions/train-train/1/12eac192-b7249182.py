"""Fragment foundry: conserve small-component material in a midpoint frame.

Only NumPy and the standard library are used. No parent generator is imported.
The public solver reads all roles, quantities, and geometry from its argument.
"""
from __future__ import annotations

import json
import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30")
    return a.astype(int, copy=True)


def _components(a, bg, diagonal=False, merge_colors=False):
    """Segment the original input. Optional alternatives are for ablation checks."""
    seen, found = set(), []
    steps = _STEPS + (((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else ())
    h, w = a.shape
    for r in range(h):
        for c in range(w):
            if (r, c) in seen or a[r, c] == bg:
                continue
            color = int(a[r, c])
            cells, stack = [], [(r, c)]
            seen.add((r, c))
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in steps:
                    p = yy, xx = y + dy, x + dx
                    if (0 <= yy < h and 0 <= xx < w and p not in seen
                            and a[p] != bg and (merge_colors or a[p] == color)):
                        seen.add(p)
                        stack.append(p)
            found.append((color, tuple(cells)))
    return found


def _square(cells):
    if len(cells) != 9:
        return False
    ys, xs = zip(*cells)
    return max(ys) - min(ys) == 2 and max(xs) - min(xs) == 2


def _geometry(shape, centers, mass):
    """Return coordinate sets in a coordinate system relative to the anchor axis."""
    if mass < 5 or mass % 2 != 1:
        raise ValueError("each color must supply the same odd mass, at least five")
    horizontal = centers[0][0] == centers[1][0]
    axis, cross = (1, 0) if horizontal else (0, 1)
    if not horizontal and centers[0][1] != centers[1][1]:
        raise ValueError("terminal centers must be collinear")
    first, second = sorted(centers, key=lambda p: p[axis])
    distance = second[axis] - first[axis]
    if distance < 9 or distance % 2 != 1:
        raise ValueError("terminal separation must be odd and at least nine")
    span = mass - 2
    start = (first[axis] + second[axis] - 3) // 2
    low = first[cross] - span // 2
    if low < 0 or low + span > shape[cross]:
        raise ValueError("the material-sized frame does not fit")

    def xy(along, across):
        return (across, along) if horizontal else (along, across)

    halves = [set(), set()]
    box = {xy(u, v) for u in range(start, start + 4)
           for v in range(low, low + span)}
    for u in range(start, start + 4):
        for v in range(low, low + span):
            if u in (start, start + 3) or v in (low, low + span - 1):
                halves[int(u >= start + 2)].add(xy(u, v))
    arms = [set(), set()]
    # Include the terminal center and wall attachment; overlap is intentional.
    for u in range(first[axis], start + 1):
        arms[0].add(xy(u, first[cross]))
    for u in range(start + 3, second[axis] + 1):
        arms[1].add(xy(u, first[cross]))
    return {"horizontal": horizontal, "axis": axis, "centers": (first, second),
            "distance": distance, "span": span, "mass": mass, "start": start,
            "low": low, "halves": halves, "arms": arms, "box": box,
            "reserved": box | arms[0] | arms[1]}


def _analyze(a):
    corners = (int(a[0, 0]), int(a[0, -1]), int(a[-1, 0]), int(a[-1, -1]))
    if len(set(corners)) != 1:
        raise ValueError("four equal corners must identify the background")
    bg = corners[0]
    comps = _components(a, bg)
    terminals = [(color, cells) for color, cells in comps if _square(cells)]
    if len(terminals) != 2 or terminals[0][0] == terminals[1][0]:
        raise ValueError("expected two maximal solid 3x3 terminals of different colors")
    centers = [(min(y for y, _ in cells) + 1, min(x for _, x in cells) + 1)
               for _, cells in terminals]
    colors = {color for color, _ in terminals}
    if {color for color, _ in comps} != colors:
        raise ValueError("fragments must use the terminal colors")
    small = [(color, cells) for color, cells in comps if len(cells) <= 2]
    masses = {color: sum(len(cells) for k, cells in small if k == color) for color in colors}
    if len(set(masses.values())) != 1:
        raise ValueError("the two movable color masses must be equal")
    geo = _geometry(a.shape, centers, next(iter(masses.values())))
    color_at = {p: color for p, (color, _) in zip(centers, terminals)}
    ordered_colors = tuple(color_at[p] for p in geo["centers"])
    terminal_cells = set(p for _, cells in terminals for p in cells)
    for _, cells in comps:
        if len(cells) >= 3 and not _square(cells) and set(cells) & geo["reserved"]:
            raise ValueError("retained objects must leave the frame and its arms clear")
    geo.update(background=bg, components=comps, small=small,
               colors=ordered_colors, terminal_cells=terminal_cells)
    return geo


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Gather only sizes 1/2 into a two-color frame; extend the stable terminals."""
    a = _grid(input_grid)
    info = _analyze(a)
    out = a.copy()
    for _, cells in info["small"]:
        for p in cells:
            out[p] = info["background"]
    for i, color in enumerate(info["colors"]):
        for p in info["halves"][i] | info["arms"][i]:
            out[p] = color
    return out


def _normalize(cells):
    y0, x0 = min(y for y, _ in cells), min(x for _, x in cells)
    return tuple(sorted((y - y0, x - x0) for y, x in cells))


def _variant(cells, rng):
    reflect = bool(rng.integers(2))
    result = [(y, -x if reflect else x) for y, x in cells]
    for _ in range(int(rng.integers(4))):
        result = [(x, -y) for y, x in result]
    return _normalize(result)


def _shape(size, rng):
    """Connected random paths/blobs/branches, with occasional rings and bridges."""
    style = int(rng.integers(5))
    if style == 0 and size >= 8:
        h, w = int(rng.integers(3, 5)), int(rng.integers(3, 6))
        cells = {(y, x) for y in range(h) for x in range(w)
                 if y in (0, h - 1) or x in (0, w - 1)}
    elif style == 1 and size >= 10:
        length = int(rng.integers(3, 8))
        cells = {(0, x) for x in range(length + 3)}
        cells.update({(1, 0), (1, 1), (1, length + 1), (1, length + 2)})
    else:
        cells = {(0, 0)}
        tip = (0, 0)
        while len(cells) < size:
            if style == 2:
                options = sorted({(tip[0] + dy, tip[1] + dx) for dy, dx in _STEPS} - cells)
            else:
                options = sorted({(y + dy, x + dx) for y, x in cells for dy, dx in _STEPS} - cells)
            if not options:
                options = sorted({(y + dy, x + dx) for y, x in cells for dy, dx in _STEPS} - cells)
            tip = options[int(rng.integers(len(options)))]
            cells.add(tip)
    return _variant(cells, rng)


def _can_place(a, cells, color, bg, forbidden):
    h, w = a.shape
    for y, x in cells:
        if (not (0 <= y < h and 0 <= x < w) or (y, x) in forbidden
                or (y in (0, h - 1) and x in (0, w - 1)) or a[y, x] != bg):
            return False
        for dy, dx in _STEPS:
            p = yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and p not in cells and a[p] == color:
                return False
    return True


def _place(a, shape, color, bg, forbidden, rng):
    shape = _variant(shape, rng)
    sy, sx = max(y for y, _ in shape) + 1, max(x for _, x in shape) + 1
    h, w = a.shape
    if sy > h or sx > w or _square(shape):
        return False
    for _ in range(180):
        top, left = int(rng.integers(h - sy + 1)), int(rng.integers(w - sx + 1))
        cells = {(top + y, left + x) for y, x in shape}
        if _can_place(a, cells, color, bg, forbidden):
            for p in cells:
                a[p] = color
            return True
    return False


def _place_witness(a, colors, bg, reserved, rng):
    """A singleton diagonally meets its own triomino; a different color touches both."""
    group = [(colors[0], ((0, 0),)),
             (colors[0], ((1, 1), (1, 2), (2, 2))),
             (colors[1], ((0, 1),))]
    reflect, turns = bool(rng.integers(2)), int(rng.integers(4))
    transformed = []
    for color, cells in group:
        cells = [(y, -x if reflect else x) for y, x in cells]
        for _ in range(turns):
            cells = [(x, -y) for y, x in cells]
        transformed.append((color, cells))
    y0 = min(y for _, cells in transformed for y, _ in cells)
    x0 = min(x for _, cells in transformed for _, x in cells)
    for _ in range(220):
        top, left = int(rng.integers(a.shape[0] - 2)), int(rng.integers(a.shape[1] - 2))
        groups = [(color, {(top + y - y0, left + x - x0) for y, x in cells})
                  for color, cells in transformed]
        # Keep this complete teaching motif outside the construction for clarity.
        if all(_can_place(a, cells, color, bg, reserved) for color, cells in groups):
            for color, cells in groups:
                for p in cells:
                    a[p] = color
            return True
    return False


def _sample_pair(rng, bg, palette, mass, horizontal, distance, witness, fragmentation):
    span = mass - 2
    along = int(rng.integers(distance + 3, min(30, distance + 10) + 1))
    minimum_cross = max(11, span)
    across = int(rng.integers(minimum_cross, 31))
    if rng.random() < .16 and along >= minimum_cross:
        across = along
    first_u = int(rng.integers(1, along - distance - 1))
    radius = max(2, span // 2)
    cross = int(rng.integers(radius, across - radius))
    centers = ((cross, first_u), (cross, first_u + distance)) if horizontal else (
        (first_u, cross), (first_u + distance, cross))
    shape = (across, along) if horizontal else (along, across)
    a = np.full(shape, bg, dtype=int)
    colors = tuple(palette if rng.integers(2) else palette[::-1])
    for center, color in zip(centers, colors):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                a[center[0] + dy, center[1] + dx] = color
    geo = _geometry(a.shape, centers, mass)
    if witness and not _place_witness(a, colors, bg, geo["reserved"], rng):
        return None
    # Every pair has a nonterminal retained component, including a size-three
    # cutoff witness in the first pair. Additional object counts are independent.
    if not _place(a, ((0, 0), (1, 0), (1, 1)), colors[1], bg, geo["reserved"], rng):
        return None
    for _ in range(int(rng.integers(0, 5))):
        _place(a, _shape(int(rng.integers(3, 19)), rng),
               int(rng.choice(colors)), bg, geo["reserved"], rng)
    for index, color in enumerate(colors):
        # The first teaching pair uses equal cell masses with unequal object
        # counts, explicitly separating cells from connected-component counts.
        dominoes = ((mass // 2 if index == 0 else 1) if fragmentation
                    else int(rng.integers(0, mass // 2 + 1)))
        singletons = mass - 2 * dominoes - int(witness)
        if singletons < 0:
            return None
        pieces = [((0, 0), (0, 1))] * dominoes + [((0, 0),)] * singletons
        rng.shuffle(pieces)
        for piece in pieces:
            if not _place(a, piece, color, bg, set(), rng):
                return None
    try:
        info = _analyze(a)
    except ValueError:
        return None
    if info["mass"] != mass:
        return None
    if witness:
        correct = _movable_set(a, info)
        construction = geo["halves"][0] | geo["halves"][1] | geo["arms"][0] | geo["arms"][1]
        alternatives = (_movable_set(a, info, diagonal=True),
                        _movable_set(a, info, merge_colors=True),
                        _movable_set(a, info, cutoff=1),
                        _movable_set(a, info, cutoff=3))
        if any(not ((correct ^ other) - construction) for other in alternatives):
            return None
    return {"input": a, "output": solve(a)}


def _movable_set(a, info, diagonal=False, merge_colors=False, cutoff=2):
    return {p for _, cells in _components(a, info["background"], diagonal, merge_colors)
            if len(cells) <= cutoff for p in cells}


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _count(value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"count must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    roles = [int(c) for c in rng.permutation(10)]
    bg, palette = roles[0], roles[1:3]
    # Literal roles are fixed in a puzzle; sides, geometry, and fragmentation vary.
    orientation0 = bool(rng.integers(2))
    first_distances = [int(v) for v in rng.choice([9, 11, 13, 15, 17, 19, 21], 2, replace=False)]
    # First vary mass within one orientation. Then, when enough demonstrations
    # exist, repeat a mass under rotation before combining new sizes and axes.
    train_masses = [5, 7] + ([5] if num_train >= 3 else []) + ([9] if num_train >= 4 else [])
    train_masses += [int(rng.choice([5, 7, 9])) for _ in range(num_train - len(train_masses))]
    mixed_axes = num_train >= 3 and bool(rng.random() < .8)
    train_axes = [orientation0, orientation0]
    if num_train >= 3:
        train_axes.append(not orientation0 if mixed_axes else orientation0)
    train_axes += [(bool(rng.integers(2)) if mixed_axes else orientation0)
                   for _ in range(num_train - len(train_axes))]
    max_train_mass = max(train_masses)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            mass = train_masses[index] if split == "train" else max_train_mass + int(rng.choice([2, 4]))
            horizontal = train_axes[index] if split == "train" else bool(rng.choice(train_axes))
            distance = first_distances[index] if split == "train" and index < 2 else int(rng.choice([9, 11, 13, 15, 17, 19, 21, 23, 25]))
            witness = (split == "train" and index == 0) or (split == "test" and index == 0)
            for _ in range(100):
                pair = _sample_pair(rng, bg, palette, mass, horizontal, distance,
                                    witness, split == "train" and index == 0)
                if pair is not None and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    """Check returned-grid semantics, material conservation, and teaching evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _analyze(a)
            infos[split].append(info)
            assert np.array_equal(solve(a), out)
            assert not np.array_equal(a, out)
            small = _movable_set(a, info)
            frame = info["halves"][0] | info["halves"][1]
            arms = info["arms"][0] | info["arms"][1]
            # Check the actual output with a per-cell ownership law in addition
            # to the main solver. This also checks the hollow background interior.
            for p in np.ndindex(a.shape):
                owner = next((i for i in (0, 1) if p in info["halves"][i] | info["arms"][i]), None)
                expected = (info["colors"][owner] if owner is not None
                            else info["background"] if p in small else a[p])
                assert out[p] == expected
            assert all(out[p] == info["background"] for p in info["box"] - frame)
            for i, color in enumerate(info["colors"]):
                assert len(info["halves"][i]) == info["mass"]
                assert all(out[p] == color for p in info["halves"][i])
                assert sum(len(cells) for k, cells in info["small"] if k == color) == len(info["halves"][i])
            for _, cells in info["components"]:
                if len(cells) >= 3:
                    assert all(out[p] == a[p] for p in cells)
            assert all(out[p] == info["background"] for p in small - frame - arms)
    all_infos = infos["train"] + infos["test"]
    assert len({(q["background"], frozenset(q["colors"])) for q in all_infos}) == 1
    assert {q["horizontal"] for q in infos["test"]}.issubset({q["horizontal"] for q in infos["train"]})
    assert infos["train"][0]["horizontal"] == infos["train"][1]["horizontal"]
    assert len({q["distance"] for q in infos["train"]}) >= 2
    train_masses = {q["mass"] for q in infos["train"]}
    assert len(train_masses) >= 2
    assert min(q["mass"] for q in infos["test"]) > max(train_masses)
    a = _grid(puzzle["train"][0]["input"])
    first = infos["train"][0]
    selected = _movable_set(a, first)
    construction = first["halves"][0] | first["halves"][1] | first["arms"][0] | first["arms"][1]
    for other in (_movable_set(a, first, diagonal=True),
                  _movable_set(a, first, merge_colors=True),
                  _movable_set(a, first, cutoff=1),
                  _movable_set(a, first, cutoff=3)):
        # Even if the correct frame were supplied for free, wrong segmentation
        # would visibly preserve/erase a different cell away from the drawing.
        assert (selected ^ other) - construction
    cleared = a.copy()
    for p in selected:
        cleared[p] = first["background"]
    assert not np.array_equal(cleared, puzzle["train"][0]["output"])
    assert {1, 2, 3}.issubset({len(cells) for _, cells in first["components"]})
    assert len({sum(1 for k, _ in first["small"] if k == color) for color in first["colors"]}) == 2
    assert any(len(cells) >= 3 and not _square(cells) for _, cells in first["components"])


def self_test() -> dict:
    """Bounded 42-seed variation survey plus API and input-only solver checks."""
    shapes, structures, signatures, palettes, masses, distances = set(), set(), set(), set(), set(), set()
    counts, retained_sizes, piece_profiles, backgrounds = set(), set(), set(), set()
    tight, edge, crossed, same_mass_different_fragments = 0, 0, 0, 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signature = tuple((_key(q["input"]), _key(q["output"])) for s in ("train", "test") for q in p[s])
        signatures.add(signature)
        if seed in (0, 17, 41):
            again = generate_puzzle(seed)
            assert signature == tuple((_key(q["input"]), _key(q["output"])) for s in ("train", "test") for q in again[s])
        for s in ("train", "test"):
            for q in p[s]:
                a = q["input"]
                info = _analyze(a)
                assert np.array_equal(solve(a.tolist()), q["output"])
                shapes.add(a.shape)
                remap = {info["background"]: 0, info["colors"][0]: 1, info["colors"][1]: 2}
                structures.add((a.shape, tuple(remap[int(v)] for v in a.flat)))
                palettes.add((info["background"], info["colors"]))
                backgrounds.add(info["background"])
                masses.add(info["mass"])
                distances.add(info["distance"])
                counts.add(len(info["components"]))
                retained_sizes.update(len(cells) for _, cells in info["components"] if len(cells) >= 3)
                profile = tuple(tuple(sorted(len(cells) for k, cells in info["small"] if k == color)) for color in info["colors"])
                piece_profiles.add(profile)
                same_mass_different_fragments += int(len(profile[0]) != len(profile[1]))
                cross_axis = 0 if info["horizontal"] else 1
                tight += int(a.shape[cross_axis] == info["span"])
                edge += int(min(y for y, _ in info["terminal_cells"]) == 0 or min(x for _, x in info["terminal_cells"]) == 0
                            or max(y for y, _ in info["terminal_cells"]) == a.shape[0] - 1
                            or max(x for _, x in info["terminal_cells"]) == a.shape[1] - 1)
                crossed += int((info["horizontal"] and a.shape[0] > a.shape[1]) or (not info["horizontal"] and a.shape[1] > a.shape[0]))
    assert len(signatures) == 42 and len(shapes) >= 50 and len(structures) >= 200
    assert len(palettes) >= 40 and len(backgrounds) >= 8
    assert {5, 7, 9, 11, 13}.issubset(masses) and len(distances) >= 8
    assert len(counts) >= 10 and len(piece_profiles) >= 40 and max(retained_sizes) >= 15
    assert edge > 0 and crossed > 0 and same_mass_different_fragments >= 42
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(4200 + 10 * nt + ns, nt, ns)
            assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Prove the validator rejects a plausible near miss, not just malformed data.
    bad = generate_puzzle(0, 2, 1)
    info = _analyze(bad["test"][0]["input"])
    point = min(info["halves"][0])
    bad["test"][0]["output"][point] = info["background"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing frame pixel was accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "grid_shapes": len(shapes), "color_normalized_inputs": len(structures),
            "role_assignments": len(palettes), "masses": sorted(masses),
            "distances": sorted(distances), "component_counts": sorted(counts),
            "retained_sizes": sorted(retained_sizes), "fragment_profiles": len(piece_profiles),
            "terminal_edge_pairs": edge, "crossed_aspect_pairs": crossed,
            "tight_frame_pairs": tight,
            "unequal_fragment_counts_equal_mass_pairs": same_mass_different_fragments}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
