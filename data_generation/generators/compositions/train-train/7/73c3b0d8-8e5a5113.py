"""Clockwork splashes: falling stencils, turning diagonal echoes, ink union.

All solving conventions are visible in the input. No parent generator, file,
random state, or puzzle history is used by solve().
"""

from collections import defaultdict
import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "Grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "Grid cells must be integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "Invalid ARC color")
    return a


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(int(x) for x in a.flat)


def _window(size, row, col):
    pitch = size + 1
    return (slice(row * pitch + 1, (row + 1) * pitch),
            slice(col * pitch + 1, (col + 1) * pitch))


def _parse_input(value):
    a = _grid(value)
    lattice = int(a[0, 0])
    _require(np.all(a[0] == lattice) and np.all(a[:, 0] == lattice),
             "The board must have a complete lattice border")
    full_rows = np.flatnonzero(np.all(a == lattice, axis=1))
    full_cols = np.flatnonzero(np.all(a == lattice, axis=0))
    _require(len(full_rows) >= 2 and len(full_cols) >= 2,
             "Missing lattice lines")
    pitch = int(full_rows[1])
    size = pitch - 1
    _require(2 <= size <= 4 and int(full_cols[1]) == pitch,
             "The board must use square stencil cells of side 2..4")
    ph, pw = a.shape
    _require((ph - 1) % pitch == (pw - 1) % pitch == 0,
             "Incomplete board cell")
    height, width = (ph - 1) // pitch, (pw - 1) // pitch
    _require(height >= 4 and width >= 3, "The board is too small")
    lattice_mask = ((np.arange(ph)[:, None] % pitch == 0) |
                    (np.arange(pw)[None, :] % pitch == 0))
    _require(np.array_equal(a == lattice, lattice_mask),
             "Lattice color must occur exactly on the grid lines")
    colors, counts = np.unique(a[~lattice_mask], return_counts=True)
    _require(len(colors) == 3 and np.count_nonzero(counts == counts.max()) == 1,
             "Need distinct, unambiguous background, floor and ink roles")
    background = int(colors[np.argmax(counts)])
    tiles = [[a[_window(size, r, c)] for c in range(width)]
             for r in range(height)]
    candidates = []
    for r in range(height):
        color = int(tiles[r][0][0, 0])
        if color != background and all(np.all(t == color) for t in tiles[r]):
            candidates.append((r, color))
    _require(len(candidates) == 1, "Need exactly one complete logical floor row")
    floor_row, floor = candidates[0]
    _require(2 <= floor_row < height, "Floor must have room above it")
    ink = (set(int(x) for x in colors) - {background, floor}).pop()
    particles = []
    for r in range(height):
        for c in range(width):
            tile = tiles[r][c]
            if r == floor_row:
                _require(np.all(tile == floor), "Broken floor")
                continue
            _require(np.all((tile == background) | (tile == ink)),
                     "Floor color may occur only in its floor row")
            mask = tile == ink
            if np.any(mask):
                _require(not np.all(mask), "A stencil must have at least one hole")
                particles.append((r, c, mask.copy()))
    _require(bool(particles), "Need at least one falling stencil")
    occupied = {(r, c) for r, c, _ in particles}
    for r, c, _ in particles:
        _require(r < height - 1 and r != floor_row - 1,
                 "Every stencil needs an in-bounds landing cell above or below the floor")
        _require(np.all(tiles[r + 1][c] == background),
                 "Every landing cell must initially be blank")
        _require(not any((r + dr, c + dc) in occupied
                         for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))),
                 "Input stencils must be isolated in logical four-connectivity")
    return {"array": a, "size": size, "height": height, "width": width,
            "background": background, "lattice": lattice, "floor": floor,
            "ink": ink, "floor_row": floor_row, "particles": particles,
            "lattice_mask": lattice_mask}


def _layers(info, turn=1, splash=True):
    """Collect independent ink contributions, before merging any of them."""
    layers = defaultdict(list)
    b, w = info["floor_row"], info["width"]
    for row, col, motif in info["particles"]:
        layers[row + 1, col].append(motif)
        if splash and row == b - 2:
            for distance in range(1, b):
                echo = np.rot90(motif, k=-turn * distance)
                for target_col in (col - distance, col + distance):
                    if 0 <= target_col < w:
                        layers[b - 1 - distance, target_col].append(echo)
    return layers


def _render(info, layers):
    out = info["array"].copy()
    for r, c, _ in info["particles"]:
        out[_window(info["size"], r, c)] = info["background"]
    for (r, c), masks in layers.items():
        tile = out[_window(info["size"], r, c)]
        tile[np.logical_or.reduce(masks)] = info["ink"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Move one logical cell down; impact emits successively clockwise echoes."""
    info = _parse_input(input_grid)
    return _render(info, _layers(info))


def _features(value):
    info = _parse_input(value)
    layers = _layers(info)
    out = _render(info, layers)
    b = info["floor_row"]
    triggers = [(r, c, m) for r, c, m in info["particles"] if r == b - 2]
    merged = [masks for masks in layers.values() if len(masks) > 1]
    return {
        "info": info, "triggers": len(triggers),
        "ordinary": len(info["particles"]) - len(triggers),
        "above": any(r < b - 2 for r, _, _ in info["particles"]),
        "below": any(r > b for r, _, _ in info["particles"]),
        "collision": bool(merged),
        "visible_merge": any(not np.all(np.logical_or.reduce(ms))
                             and np.any(np.sum(np.stack(ms), axis=0) >= 2)
                             and all(not np.array_equal(np.logical_or.reduce(ms), m)
                                     for m in ms) for ms in merged),
        "rotation_visible": not np.array_equal(out, _render(info, _layers(info, turn=0))),
        "direction_visible": not np.array_equal(out, _render(info, _layers(info, turn=-1))),
        "splash_visible": not np.array_equal(out, _render(info, _layers(info, splash=False))),
        "long_sequence": any(
            b >= 4 and len({_key(np.rot90(m, k=k)) for k in range(4)}) == 4
            and all(any(0 <= tc < info["width"]
                        and len(layers[b - 1 - d, tc]) == 1
                        for tc in (c - d, c + d)) for d in (1, 2, 3))
            for _, c, m in triggers),
        "edge": any(c in (0, info["width"] - 1) for _, c, _ in triggers),
        "branching": any(0 < c < info["width"] - 1 for _, c, _ in triggers),
        "side_clip": any(min(c, info["width"] - 1 - c) < b - 1
                         for _, c, _ in triggers),
        "roles": tuple(info[k] for k in ("background", "lattice", "floor", "ink")),
    }


def _make_input(size, height, width, floor_row, particles, roles):
    background, lattice, floor, ink = roles
    a = np.full((height * (size + 1) + 1, width * (size + 1) + 1),
                lattice, dtype=np.int8)
    for r in range(height):
        for c in range(width):
            a[_window(size, r, c)] = floor if r == floor_row else background
    for r, c, mask in particles:
        tile = a[_window(size, r, c)]
        tile[mask] = ink
    return a


def _motif(rng, size, strong=False):
    """Sparse/dense, connected/disconnected, and deliberately structured stencils."""
    for _ in range(100):
        mode = int(rng.integers(5))
        mask = np.zeros((size, size), dtype=bool)
        if mode == 0:
            mask = rng.random((size, size)) < rng.uniform(0.2, 0.75)
        elif mode == 1:
            r, c = (int(x) for x in rng.integers(size, size=2))
            mask[r, c] = True
            target = int(rng.integers(2, size * size))
            while int(mask.sum()) < target:
                neighbors = {(int(r) + dr, int(c) + dc)
                             for r, c in np.argwhere(mask)
                             for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                             if 0 <= r + dr < size and 0 <= c + dc < size
                             and not mask[r + dr, c + dc]}
                options = sorted(neighbors)
                r, c = options[int(rng.integers(len(options)))]
                mask[r, c] = True
        elif mode == 2:
            mask[int(rng.integers(size)), :] = True
            mask[:, int(rng.integers(size))] = True
            mask = np.rot90(mask, k=int(rng.integers(4)))
        elif mode == 3:
            np.fill_diagonal(mask, True)
            mask[int(rng.integers(size)), int(rng.integers(size))] = True
        else:
            mask[0, :int(rng.integers(1, size + 1))] = True
            mask[:int(rng.integers(1, size + 1)), 0] = True
            mask = np.rot90(mask, k=int(rng.integers(4)))
        if not 1 <= int(mask.sum()) < size * size:
            continue
        if strong and len({_key(np.rot90(mask, k=k)) for k in range(4)}) < 4:
            continue
        return mask.copy()
    # A corner L has four different rotations for every supported stencil size.
    mask = np.zeros((size, size), dtype=bool)
    mask[0, 0] = mask[0, 1] = mask[1, 0] = True
    return mask


def _positions(rng, height, width, b, trigger_count, above, below, preset=None):
    points = []

    def add(candidates, count):
        for i in rng.permutation(len(candidates)):
            point = candidates[int(i)]
            if all(abs(point[0] - r) + abs(point[1] - c) > 1 for r, c in points):
                points.append(point)
                count -= 1
                if count == 0:
                    return True
        return count == 0

    if preset is not None:
        points.extend((b - 2, int(c)) for c in preset)
    elif trigger_count and not add([(b - 2, c) for c in range(width)], trigger_count):
        return None
    if above and not add([(r, c) for r in range(b - 2)
                          for c in range(width)], above):
        return None
    if below and not add([(r, c) for r in range(b + 1, height - 1)
                          for c in range(width)], below):
        return None
    return points


def _sample_pair(rng, size, profile, roles, collision_allowed,
                 long=False, require_merge=False, high_floor=False,
                 low_floor=False, both_halves=False, require_rotation=False):
    maximum = 29 // (size + 1)
    for _ in range(500):
        min_h = 7 if both_halves and low_floor else 6 if both_halves else 5
        height = int(rng.integers(min_h, maximum + 1))
        min_w = 7 if require_merge else 5 if long else 3
        width = int(rng.integers(min_w, maximum + 1))
        floors = [b for b in range(2, height)
                  if (not long or b >= 4)
                  and (not high_floor or b <= 3)
                  and (not low_floor or b >= 4)
                  and (not both_halves or 3 <= b <= height - 3)]
        if not floors:
            continue
        b = int(rng.choice(floors))
        nt = 0 if profile == "plain" else (
            int(rng.integers(1, min(3, (width + 1) // 2) + 1))
            if collision_allowed and not long else 1)
        preset = None
        if require_merge:
            options = [(left, right) for left in range(width)
                       for right in range(left + 4, width, 2)
                       if (right - left) // 2 <= b - 1
                       and (left > 0 or right < width - 1)]
            if not options:
                continue
            preset = options[int(rng.integers(len(options)))]
            nt = 2
        elif long:
            cols = [c for c in range(1, width - 1)
                    if max(c, width - 1 - c) >= 3
                    and min(c, width - 1 - c) < b - 1]
            if not cols:
                continue
            preset = [int(rng.choice(cols))]
        na = nb = 0
        if profile in ("plain", "mixed"):
            if both_halves:
                na = nb = 1
            else:
                options = []
                if b >= 3:
                    options.append("above")
                if b <= height - 3:
                    options.append("below")
                if not options:
                    continue
                side = str(rng.choice(options))
                na, nb = (1, 0) if side == "above" else (0, 1)
                if rng.random() < 0.35:
                    if b >= 3:
                        na += int(rng.integers(2))
                    if b <= height - 3:
                        nb += int(rng.integers(2))
        points = _positions(rng, height, width, b, nt, na, nb, preset)
        if points is None:
            continue
        particles = [(r, c, _motif(rng, size, strong=long)) for r, c in points]
        inp = _make_input(size, height, width, b, particles, roles)
        f = _features(inp)
        if long and not (f["long_sequence"] and f["rotation_visible"]
                         and f["direction_visible"] and f["branching"] and f["side_clip"]):
            continue
        if require_rotation and not (f["rotation_visible"] and f["splash_visible"]):
            continue
        if require_merge and not f["visible_merge"]:
            continue
        if not collision_allowed and f["collision"]:
            continue
        return {"input": inp, "output": solve(inp)}
    raise RuntimeError("Could not construct the requested evidence profile")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shift = str(rng.choice(("stencil_scale", "higher_floor", "mixed_population")))
    collision_allowed = bool(rng.integers(2))
    local_palette = bool(rng.integers(2))
    fixed_roles = tuple(int(x) for x in rng.choice(10, size=4, replace=False))
    sizes = list(rng.permutation((2, 3)))
    seen, palettes = set(), set()
    puzzle = {"train": [], "test": []}

    def append(split, size, profile, **kwargs):
        for _ in range(80):
            roles = (tuple(int(x) for x in rng.choice(10, size=4, replace=False))
                     if local_palette else fixed_roles)
            if local_palette and roles in palettes:
                continue
            pair = _sample_pair(rng, int(size), profile, roles,
                                collision_allowed, **kwargs)
            key = _key(pair["input"])
            if key not in seen:
                puzzle[split].append(pair)
                seen.add(key)
                palettes.add(roles)
                return
        raise RuntimeError("Could not construct a distinct input")

    # These two examples alone suffice at the minimum legal train count:
    # an observable full rotation cycle (and union if needed), then plain fall
    # both above and below the floor. Their presentation order is randomized.
    append("train", sizes[0], "fan", long=True, require_merge=collision_allowed,
           low_floor=shift == "higher_floor")
    append("train", sizes[1], "plain", both_halves=True,
           low_floor=shift == "higher_floor")
    for _ in range(num_train - 2):
        modes = ("fan", "plain") if shift == "mixed_population" else ("fan", "plain", "mixed")
        profile = str(rng.choice(modes))
        size = int(rng.choice((2, 3) if shift == "stencil_scale" else (2, 3, 4)))
        append("train", size, profile, low_floor=shift == "higher_floor")
    rng.shuffle(puzzle["train"])

    for i in range(num_test):
        profile = ("mixed" if shift == "mixed_population" else
                   str(rng.choice(("fan", "mixed") if i == 0 else ("fan", "mixed", "plain"))))
        size = 4 if shift == "stencil_scale" else int(rng.choice((2, 3, 4)))
        append("test", size, profile, high_floor=shift == "higher_floor", require_rotation=i == 0)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "Puzzle must have exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "Splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "Illegal example counts")
    features, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "Pair must have exactly input and output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "Board dimensions must stay fixed")
            _require(_key(inp) not in seen, "Duplicate input")
            seen.add(_key(inp))
            f = _features(inp)
            info = f["info"]
            expected = _render(info, _layers(info))
            _require(np.array_equal(expected, out), "Incorrect stencil fall/echo/union")
            _require(not np.array_equal(inp, out), "Identity pair")
            _require(np.array_equal(inp[info["lattice_mask"]], out[info["lattice_mask"]]),
                     "Lattice changed")
            for c in range(info["width"]):
                _require(np.all(out[_window(info["size"], info["floor_row"], c)] == info["floor"]),
                         "Floor changed")
            _require(set(int(x) for x in np.unique(out)) <= set(int(x) for x in np.unique(inp)),
                     "Invented output colors")
            features[split].append(f)
    train, test = features["train"], features["test"]
    _require(any(f["long_sequence"] and f["rotation_visible"] and f["direction_visible"]
                 for f in train), "Training must expose the four orientations and clockwise direction")
    _require(any(f["branching"] and f["side_clip"] for f in train),
             "Training must expose two branches and side-boundary clipping")
    _require(any(f["triggers"] == 0 and f["above"] and f["below"] for f in train),
             "Training must contrast plain fall both above and below the floor")
    _require(any(f["splash_visible"] for f in train), "A's splash must change an answer")
    _require(any(f["rotation_visible"] for f in train), "B's rotations must change an answer")
    _require(any(f["rotation_visible"] and f["splash_visible"] for f in test),
             "At least one test must require both inherited mechanisms")
    if any(f["collision"] for f in train + test):
        _require(any(f["visible_merge"] for f in train),
                 "A strict ink union with shared ink and remaining holes must precede any shared destination")
    if len({f["roles"] for f in train + test}) > 1:
        _require(len({f["roles"] for f in train}) >= 2,
                 "Changing local color roles must be demonstrated")
    _require(len({(f["info"]["size"], f["info"]["height"], f["info"]["width"])
                  for f in train}) >= 2, "Training must vary board/stencil geometry")
    # At least one explicit structural extrapolation must be visible in the
    # returned grids; this is checked without a stored shift label.
    train_sizes = {f["info"]["size"] for f in train}
    scale_shift = all(f["info"]["size"] not in train_sizes for f in test)
    floor_shift = max(f["info"]["floor_row"] for f in test) < min(
        f["info"]["floor_row"] for f in train)
    population_shift = all(not (f["triggers"] and f["ordinary"]) for f in train) and all(
        f["triggers"] and f["ordinary"] for f in test)
    _require(scale_shift or floor_shift or population_shift, "Missing structural train/test shift")


def _serial(puzzle):
    return json.dumps({s: [{k: np.asarray(v).tolist() for k, v in p.items()}
                           for p in puzzle[s]] for s in ("train", "test")},
                      separators=(",", ":"))


def self_test():
    """Bounded semantic fixtures, all count combinations, and 42-seed diversity."""
    # Hand-specified crossing: two turns arrive at the same logical cell.
    left = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)
    right = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]], dtype=bool)
    ordinary = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)
    inp = _make_input(3, 7, 5, 4,
                      [(2, 0, left), (2, 4, right), (5, 2, ordinary)], (0, 8, 2, 4))
    out = solve(inp)
    expected_tiles = {(3, 0): left, (3, 4): right,
                      (2, 1): np.rot90(left, -1), (2, 3): np.rot90(right, -1),
                      (1, 2): np.rot90(left, 2) | np.rot90(right, 2),
                      (0, 3): np.rot90(left, 1), (0, 1): np.rot90(right, 1),
                      (6, 2): ordinary}
    for r in range(7):
        for c in range(5):
            tile = out[_window(3, r, c)]
            if r == 4:
                assert np.all(tile == 2)
            else:
                assert np.array_equal(tile == 4, expected_tiles.get((r, c), np.zeros((3, 3), bool)))
    # Color identities have no operational meaning.
    permutation = np.array([7, 9, 5, 8, 1, 3, 0, 2, 6, 4], dtype=np.int8)
    assert np.array_equal(solve(permutation[inp]), permutation[out])

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, structures, sizes, widths, heights, floors, counts, palettes = (set() for _ in range(8))
    collisions = plain = mixed = symmetric = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair["input"])
                info = f["info"]
                shapes.add(info["array"].shape)
                sizes.add(info["size"])
                widths.add(info["width"])
                heights.add(info["height"])
                floors.add(info["floor_row"])
                counts.add(len(info["particles"]))
                palettes.add(f["roles"])
                structures.add((info["size"], info["height"], info["width"], info["floor_row"],
                                tuple((r, c, tuple(bool(v) for v in m.flat))
                                      for r, c, m in info["particles"])))
                collisions += f["visible_merge"]
                plain += f["triggers"] == 0
                mixed += bool(f["triggers"] and f["ordinary"])
                symmetric += sum(np.array_equal(m, np.rot90(m, 2))
                                 for _, _, m in info["particles"])
    assert sizes == {2, 3, 4}
    assert len(shapes) >= 25 and len(structures) >= 150 and len(palettes) >= 30
    assert len(widths) >= 6 and len(heights) >= 5 and len(floors) >= 5 and len(counts) >= 4
    assert collisions >= 10 and plain >= 42 and mixed >= 15 and symmetric >= 1
    assert len({_serial(p) for p in puzzles}) == 42
    assert _serial(generate_puzzle(702292019)) == _serial(generate_puzzle(702292019))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(156182150, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    return {"seeds_checked": 42, "pairs_checked": sum(len(p[s]) for p in puzzles for s in p),
            "input_shapes": len(shapes), "stencil_sizes": sorted(sizes),
            "logical_heights": sorted(heights), "logical_widths": sorted(widths),
            "floor_rows": sorted(floors), "particle_counts": sorted(counts),
            "role_palettes": len(palettes), "structural_layouts": len(structures),
            "visible_merges": collisions, "plain_pairs": plain, "mixed_pairs": mixed,
            "half_turn_symmetric_stencils": symmetric}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
