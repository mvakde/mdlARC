"""Magnetic square shells: sparse radial anchors and two-sided rigid docking.

Only NumPy and the standard library are used.  Coordinates in the geometry
helpers are relative to the center; no generator state is used by solve().
"""

from functools import lru_cache
from itertools import combinations

import numpy as np


_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_BASE_SHAPES = {
    "bar": (((0, 0), (0, 1)), ((0, 0), (0, 1), (0, 2)),
            ((0, 0), (0, 1), (0, 2), (0, 3))),
    "bent": (((0, 0), (1, 0), (1, 1)),
             ((0, 0), (1, 0), (2, 0), (2, 1)),
             ((0, 0), (0, 1), (0, 2), (1, 1)),
             ((0, 0), (0, 1), (1, 1), (1, 2))),
    "block": (((0, 0), (0, 1), (1, 0), (1, 1)),),
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _normalize(cells):
    top = min(p[0] for p in cells)
    left = min(p[1] for p in cells)
    return tuple(sorted((r - top, c - left) for r, c in cells))


def _orientations(shapes):
    result = set()
    for shape in shapes:
        for swap in (False, True):
            for sr in (-1, 1):
                for sc in (-1, 1):
                    cells = [(sr * (c if swap else r), sc * (r if swap else c))
                             for r, c in shape]
                    result.add(_normalize(cells))
    return tuple(sorted(result))


_SHAPES = {kind: _orientations(shapes) for kind, shapes in _BASE_SHAPES.items()}


@lru_cache(maxsize=15)
def _shell(radius):
    return frozenset((r, c) for r in range(-radius, radius + 1)
                     for c in range(-radius, radius + 1)
                     if max(abs(r), abs(c)) == radius)


def _shift(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _touches(first, second):
    return any((r + dr, c + dc) in second for r, c in first for dr, dc in _DIR4)


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, found = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIR4:
                neighbor = (r + dr, c + dc)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    found.add(neighbor)
        result.append(frozenset(found))
    return result


def _kind(cells):
    shape = _normalize(cells)
    if len({r for r, _ in shape}) == 1 or len({c for _, c in shape}) == 1:
        return "bar"
    return "block" if shape == ((0, 0), (0, 1), (1, 0), (1, 1)) else "bent"


def _direction(cells, radius):
    """Return (unit translation, inside/outside) from visible geometry alone."""
    lo_r, hi_r = min(r for r, _ in cells), max(r for r, _ in cells)
    lo_c, hi_c = min(c for _, c in cells), max(c for _, c in cells)
    if -radius < lo_r <= hi_r < radius and -radius < lo_c <= hi_c < radius:
        # Distance from the component's extremal cells to each inner face.
        gaps = (radius + lo_r, radius - hi_r, radius + lo_c, radius - hi_c)
        closest = min(gaps)
        if gaps.count(closest) != 1:
            raise ValueError("interior fragment has tied closest faces")
        return _DIR4[gaps.index(closest)], "inside"

    def sector(low, high):
        if high < -radius:
            return -1
        if low > radius:
            return 1
        if -radius <= low <= high <= radius:
            return 0
        raise ValueError("fragment straddles a target-square sector boundary")

    row_sector, col_sector = sector(lo_r, hi_r), sector(lo_c, hi_c)
    if row_sector == col_sector == 0:
        raise ValueError("fragment overlaps its shell")
    return (-row_sector, -col_sector), "outside"


def _plan_fragment(cells, radius, canvas_radius):
    ring = _shell(radius)
    if cells & ring or _touches(cells, ring):
        raise ValueError("a source fragment must be detached from its virtual shell")
    direction, side = _direction(cells, radius)
    current = cells
    for steps in range(1, 2 * canvas_radius + 2):
        current = _shift(current, *direction)
        if any(max(abs(r), abs(c)) > canvas_radius for r, c in current):
            raise ValueError("docking ray leaves the canvas")
        if current & ring:
            raise ValueError("ray overlaps shell before reaching edge contact")
        if _touches(current, ring):
            return {"start": cells, "end": current, "direction": direction,
                    "side": side, "steps": steps, "radius": radius,
                    "kind": _kind(cells)}
    raise ValueError("no docking position")


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("expected a 2D integer ARC grid")
    return grid.astype(np.int8, copy=False)


def _decode(input_grid):
    grid = _grid(input_grid)
    n, width = grid.shape
    if n != width or n % 2 != 1 or not 17 <= n <= 29:
        raise ValueError("expected an odd square of side 17..29")
    center = n // 2
    counts = np.bincount(grid.ravel(), minlength=10)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("background must be the unique dominant color")
    background = int(np.argmax(counts))
    colors = [int(c) for c in np.unique(grid) if c != background]
    if not 2 <= len(colors) <= 4:
        raise ValueError("expected two to four shell colors")
    anchors, fragments = {}, []
    for color in colors:
        cells = {(int(r) - center, int(c) - center)
                 for r, c in np.argwhere(grid == color)}
        parts = _components(cells)
        clues = [part for part in parts if len(part) == 1]
        loose = [part for part in parts if len(part) != 1]
        if len(clues) != 1 or not 1 <= len(loose) <= 3:
            raise ValueError("each color needs one singleton beacon and 1..3 fragments")
        beacon = next(iter(clues[0]))
        radius = max(abs(beacon[0]), abs(beacon[1]))
        if not 2 <= radius <= center - 3:
            raise ValueError("beacon radius out of range")
        anchors[color] = {"beacon": beacon, "radius": radius}
        for part in loose:
            if not 2 <= len(part) <= 4:
                raise ValueError("satellite area must be 2..4")
            plan = _plan_fragment(part, radius, center)
            plan["color"] = color
            fragments.append(plan)
    radii = sorted(anchor["radius"] for anchor in anchors.values())
    if any(b - a < 3 for a, b in zip(radii, radii[1:])):
        raise ValueError("shell radii must differ by at least three")
    return grid, background, anchors, fragments


def _render(shape, background, anchors, fragments, *, move=True):
    result = np.full(shape, background, dtype=np.int8)
    center = shape[0] // 2
    occupied = set()
    for color, anchor in anchors.items():
        ring = _shell(anchor["radius"])
        occupied.update(ring)
        for r, c in ring:
            result[r + center, c + center] = color
    for fragment in fragments:
        cells = fragment["end" if move else "start"]
        if move and cells & occupied:
            raise ValueError("final fragments may not overlap any ring or one another")
        occupied.update(cells)
        for r, c in cells:
            result[r + center, c + center] = fragment["color"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode all anchors and plan every translation independently from the input."""
    grid, background, anchors, fragments = _decode(input_grid)
    return _render(grid.shape, background, anchors, fragments)


def _crosses_other_shell(fragment, radii):
    a = [max(abs(r), abs(c)) for r, c in fragment["start"]]
    b = [max(abs(r), abs(c)) for r, c in fragment["end"]]
    return any(q != fragment["radius"] and
               ((max(a) < q < min(b)) or (max(b) < q < min(a))) for q in radii)


def _features(anchors, fragments):
    radii = [anchor["radius"] for anchor in anchors.values()]
    return {
        "sides": {f["side"] for f in fragments},
        "classes": {"cardinal" if 0 in f["direction"] else "diagonal" for f in fragments},
        "kinds": {f["kind"] for f in fragments},
        "crossing": any(_crosses_other_shell(f, radii) for f in fragments),
        "remote": any(not _touches(f["end"], {anchors[f["color"]]["beacon"]})
                      for f in fragments),
        "depth": any(any(not _touches({p}, _shell(f["radius"])) for p in f["end"])
                     for f in fragments),
        "corner": any(abs(a["beacon"][0]) == abs(a["beacon"][1]) for a in anchors.values()),
    }


@lru_cache(maxsize=30)
def _radius_options(canvas_radius, count):
    return tuple(rs for rs in combinations(range(2, canvas_radius - 2), count)
                 if all(b - a >= 3 for a, b in zip(rs, rs[1:])))


def _candidate(rng, canvas_radius, radius, allowed, inside=None):
    kinds = [k for k in ("bar", "bent", "block") if k in allowed]
    weights = np.array([{"bar": .45, "bent": .40, "block": .15}[k] for k in kinds])
    kind = kinds[int(rng.choice(len(kinds), p=weights / weights.sum()))]
    shapes = _SHAPES[kind]
    shape = shapes[int(rng.integers(len(shapes)))]
    height = max(r for r, _ in shape) + 1
    width = max(c for _, c in shape) + 1
    interior = bool(rng.random() < .45) if inside is None else inside
    face = int(rng.integers(4))
    normal = _DIR4[face]
    # Propose a final pose with one extremum immediately beside the chosen face.
    bound = radius - 1 if interior else radius + 1
    tangent_extent = width if face < 2 else height
    if interior:
        low, high = -radius + 1, radius - tangent_extent
    else:
        low, high = -radius - tangent_extent, radius + 1
    if high < low:
        return None
    tangent = int(rng.integers(low, high + 1))
    if face == 0:
        top = -bound if interior else -bound - height + 1
        left = tangent
    elif face == 1:
        top = bound - height + 1 if interior else bound
        left = tangent
    elif face == 2:
        left = -bound if interior else -bound - width + 1
        top = tangent
    else:
        left = bound - width + 1 if interior else bound
        top = tangent
    end = _shift(shape, top, left)
    if interior:
        direction = normal
        step_limit = radius - 1
    else:
        direction = (-normal[0], -normal[1])
        if rng.random() < .45:
            tangent_sign = int(rng.choice((-1, 1)))
            direction = ((direction[0], tangent_sign) if face < 2
                         else (tangent_sign, direction[1]))
        step_limit = canvas_radius - radius
    steps = int(rng.integers(1, step_limit + 1))
    start = _shift(end, -steps * direction[0], -steps * direction[1])
    if any(max(abs(r), abs(c)) > canvas_radius for r, c in start | end):
        return None
    try:
        plan = _plan_fragment(start, radius, canvas_radius)
    except ValueError:
        return None
    if plan["end"] != end or (plan["side"] == "inside") != interior:
        return None
    return plan


def _make_pair(rng, canvas_radius, shell_count, fragment_count, background,
               allowed, witness=False):
    options = _radius_options(canvas_radius, shell_count)
    if not options:
        raise ValueError("not enough room for requested shells")
    for _ in range(35):
        radii = options[int(rng.integers(len(options)))]
        colors = [int(c) for c in rng.choice([c for c in range(10) if c != background],
                                            size=shell_count, replace=False)]
        quotas = [1] * shell_count
        for _ in range(fragment_count - shell_count):
            eligible = [i for i, q in enumerate(quotas) if q < 3]
            quotas[int(rng.choice(eligible))] += 1
        specs = [(i, j) for i, count in enumerate(quotas) for j in range(count)]
        # A witness crosses a foreign shell; another docks to an inner surface.
        if witness:
            specs.sort(key=lambda ij: (0 if ij == (0, 0) else
                                      1 if ij == (shell_count - 1, 0) else 2))
        else:
            rng.shuffle(specs)
        fragments, used_input = [], set()
        used_output = set().union(*(_shell(r) for r in radii))
        by_color = [set() for _ in radii]
        for i, j in specs:
            need_cross = witness and (i, j) == (0, 0)
            inside = False if need_cross else (True if witness and
                     (i, j) == (shell_count - 1, 0) else None)
            for _ in range(650):
                candidate = _candidate(rng, canvas_radius, radii[i], allowed, inside)
                if candidate is None:
                    continue
                if (candidate["start"] & used_input or candidate["end"] & used_output
                        or _touches(candidate["start"], by_color[i])):
                    continue
                if need_cross and not _crosses_other_shell(candidate, radii):
                    continue
                candidate["color"] = colors[i]
                fragments.append(candidate)
                used_input.update(candidate["start"])
                used_output.update(candidate["end"])
                by_color[i].update(candidate["start"])
                break
            else:
                break
        if len(fragments) != fragment_count:
            continue
        anchors = {}
        for i, (color, radius) in enumerate(zip(colors, radii)):
            cells = sorted(_shell(radius) - used_input)
            if witness and i == 0:
                preferred = [p for p in cells if abs(p[0]) == abs(p[1])]
            else:
                # Some clues are side centers, some corners, most scatter.
                style = int(rng.integers(4))
                preferred = ([p for p in cells if 0 in p] if style == 0 else
                             [p for p in cells if abs(p[0]) == abs(p[1])]
                             if style == 1 else cells)
            if not cells:
                break
            choices = preferred or cells
            beacon = choices[int(rng.integers(len(choices)))]
            anchors[color] = {"beacon": beacon, "radius": radius}
        if len(anchors) != shell_count:
            continue
        if witness and not _features(anchors, fragments)["corner"]:
            continue
        n = 2 * canvas_radius + 1
        inp = np.full((n, n), background, dtype=np.int8)
        for color, anchor in anchors.items():
            r, c = anchor["beacon"]
            inp[r + canvas_radius, c + canvas_radius] = color
        for fragment in fragments:
            for r, c in fragment["start"]:
                inp[r + canvas_radius, c + canvas_radius] = fragment["color"]
        out = _render(inp.shape, background, anchors, fragments)
        return {"input": inp, "output": out}
    raise RuntimeError("bounded placement search exhausted")


def _signature(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def _puzzle_signature(puzzle):
    return tuple((_signature(pair["input"]), _signature(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def _legal_count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _legal_count(num_train, "num_train", 2, 10)
    num_test = _legal_count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    regime = str(rng.choice(("scale", "shells", "crowding")))
    base_shells = int(rng.choice((2, 3))) if regime == "crowding" else 2
    shared_background = int(rng.integers(10)) if rng.random() < .5 else None
    for _ in range(25):
        puzzle, seen = {"train": [], "test": []}, set()
        train_features, train_kinds = [], set()
        train_backgrounds = []
        for index in range(num_train):
            radius = int(rng.integers(8, 12)) if regime == "scale" else int(
                rng.integers(11 if base_shells == 3 else 8, 15))
            count = base_shells + (1 if regime == "crowding" else int(rng.integers(1, 3)))
            if shared_background is None:
                choices = [c for c in range(10) if index != 1 or c != train_backgrounds[0]]
                background = int(rng.choice(choices))
            else:
                background = shared_background
            pair = _make_pair(rng, radius, base_shells, count, background,
                              set(_SHAPES), witness=index == 0)
            if _signature(pair["input"]) in seen:
                break
            seen.add(_signature(pair["input"]))
            puzzle["train"].append(pair)
            train_backgrounds.append(background)
            _, _, anchors, fragments = _decode(pair["input"])
            feature = _features(anchors, fragments)
            train_features.append(feature)
            train_kinds.update(feature["kinds"])
        if len(puzzle["train"]) != num_train:
            continue
        classes = set().union(*(f["classes"] for f in train_features))
        if not {"cardinal", "diagonal"} <= classes or not {"bar", "bent"} <= train_kinds:
            continue
        if not any(f["depth"] for f in train_features):
            continue
        rng.shuffle(puzzle["train"])
        for _ in range(num_test):
            if regime == "scale":
                radius, shell_count = int(rng.integers(12, 15)), 2
                count = int(rng.integers(3, 6))
            elif regime == "shells":
                shell_count = int(rng.choice((3, 3, 4)))
                radius = 14 if shell_count == 4 else int(rng.integers(11, 15))
                count = shell_count + int(rng.integers(1, 3))
            else:
                radius = int(rng.integers(11 if base_shells == 3 else 8, 15))
                shell_count = base_shells
                count = shell_count + int(rng.integers(2, 4))
            background = (int(rng.integers(10)) if shared_background is None
                          else shared_background)
            for _ in range(10):
                pair = _make_pair(rng, radius, shell_count, count, background, train_kinds)
                if _signature(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("could not draw a distinct test input")
            seen.add(_signature(pair["input"]))
            puzzle["test"].append(pair)
        validate_generated_puzzle(puzzle)
        return puzzle
    raise RuntimeError("could not obtain training coverage within the bounded search")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "invalid example counts")
    seen, info = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, background, anchors, fragments = _decode(pair["input"])
            out = _grid(pair["output"])
            _require(inp.shape == out.shape, "output shape changed")
            key = _signature(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            expected = _render(inp.shape, background, anchors, fragments)
            _require(np.array_equal(out, expected), "wrong shell completion or docking")
            _require(np.array_equal(out, solve(inp)), "input-only solver disagreement")
            _require(not np.array_equal(out, _render(inp.shape, background, anchors, fragments,
                                                    move=False)), "ignoring translation gives the answer")
            center = inp.shape[0] // 2
            for color, anchor in anchors.items():
                radius = anchor["radius"]
                _require(all(out[r + center, c + center] == color for r, c in _shell(radius)),
                         "incomplete or overwritten shell")
                _require(np.count_nonzero(out == color) - np.count_nonzero(inp == color)
                         == 8 * radius - 1, "fragment cells were created or lost")
            for f in fragments:
                dr, dc = f["direction"]
                _require(f["steps"] > 0 and _normalize(f["start"]) == _normalize(f["end"]),
                         "fragment failed rigid translation")
                _require(f["end"] == _shift(f["start"], dr * f["steps"], dc * f["steps"]),
                         "incorrect displacement")
                ring = _shell(f["radius"])
                _require(_touches(f["end"], ring) and not f["end"] & ring,
                         "destination is not edge contact")
                for step in range(f["steps"]):
                    earlier = _shift(f["start"], dr * step, dc * step)
                    _require(not _touches(earlier, ring) and not earlier & ring,
                             "fragment passed first contact")
            feature = _features(anchors, fragments)
            info[split].append({**feature, "n": inp.shape[0], "shells": len(anchors),
                                "fragments": len(fragments)})
    train = info["train"]
    _require(set().union(*(i["sides"] for i in train)) == {"inside", "outside"},
             "training must demonstrate both surfaces")
    _require(set().union(*(i["classes"] for i in train)) == {"cardinal", "diagonal"},
             "training must demonstrate both motion classes")
    kinds = set().union(*(i["kinds"] for i in train))
    _require({"bar", "bent"} <= kinds, "training needs straight and bent rigid fragments")
    for witness in ("crossing", "remote", "depth", "corner"):
        _require(any(i[witness] for i in train), f"missing training witness: {witness}")
    for item in info["test"]:
        _require(item["kinds"] <= kinds, "test uses an undemonstrated shape category")
        _require(any(item[key] > max(i[key] for i in train)
                     for key in ("n", "shells", "fragments")),
                 "test needs a scale, shell-count, or fragment-count shift")
    return True


def self_test():
    """Bounded semantic, API, metamorphic and structural checks over 42 seeds."""
    signatures, structures, dimensions, radii_seen, palettes = set(), set(), set(), set(), set()
    directions, areas, kinds, shell_counts, loads, shifts = set(), set(), set(), set(), set(), set()
    inward = outward = crossings = fixed_backgrounds = mixed_backgrounds = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_signature(puzzle))
        backgrounds = set()
        maxima = {key: 0 for key in ("n", "shells", "fragments")}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, bg, anchors, fragments = _decode(pair["input"])
                dimensions.add(grid.shape)
                backgrounds.add(bg)
                radii_seen.update(a["radius"] for a in anchors.values())
                palettes.add((bg, tuple(sorted(anchors))))
                shell_counts.add(len(anchors))
                loads.add(len(fragments))
                for f in fragments:
                    directions.add(f["direction"])
                    areas.add(len(f["start"]))
                    kinds.add(f["kind"])
                    inward += f["side"] == "outside"
                    outward += f["side"] == "inside"
                crossings += _features(anchors, fragments)["crossing"]
                roles = {bg: 0}
                for color in sorted(anchors, key=lambda c: anchors[c]["radius"]):
                    roles[color] = len(roles)
                structures.add((grid.shape, bytes(roles[int(v)] for v in grid.flat)))
                vals = {"n": grid.shape[0], "shells": len(anchors), "fragments": len(fragments)}
                if split == "train":
                    for key in maxima:
                        maxima[key] = max(maxima[key], vals[key])
                else:
                    shifts.update(key for key in maxima if vals[key] > maxima[key])
        fixed_backgrounds += len(backgrounds) == 1
        mixed_backgrounds += len(backgrounds) > 1
        # Rotations/reflections and arbitrary palette permutations commute with solve.
        if seed < 4:
            pair = puzzle["train"][0]
            for turn in (1, 2, 3):
                _require(np.array_equal(solve(np.rot90(pair["input"], turn)),
                                       np.rot90(pair["output"], turn)), "rotation covariance failed")
            _require(np.array_equal(solve(np.fliplr(pair["input"])), np.fliplr(pair["output"])),
                     "reflection covariance failed")
            permutation = np.random.default_rng(1000 + seed).permutation(10)
            _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]),
                     "palette covariance failed")
            _require(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "list input failed")
            # Moving a beacon around its shell cannot change either the shell
            # or the docking poses. This specifically probes A's radial role.
            _, bg, anchors, _ = _decode(pair["input"])
            center = pair["input"].shape[0] // 2
            for color, anchor in anchors.items():
                old_r, old_c = anchor["beacon"]
                alternatives = [p for p in sorted(_shell(anchor["radius"]))
                                if pair["input"][p[0] + center, p[1] + center] == bg]
                _require(bool(alternatives), "no alternate beacon location")
                new_r, new_c = alternatives[0]
                changed = pair["input"].copy()
                changed[old_r + center, old_c + center] = bg
                changed[new_r + center, new_c + center] = color
                _require(np.array_equal(solve(changed), pair["output"]),
                         "beacon angle affected a radial anchor")
    _require(len(signatures) == 42 and len(structures) >= 180, "insufficient structural variation")
    _require(len(dimensions) == 7 and len(palettes) >= 70, "scale or palette coverage too narrow")
    _require(shell_counts == {2, 3, 4} and len(loads) >= 4, "count variation missing")
    _require(areas == {2, 3, 4} and kinds == set(_SHAPES), "shape variation missing")
    _require(len(directions) == 8 and inward > 100 and outward > 50, "motion variation missing")
    _require(crossings >= 42 and shifts == {"n", "shells", "fragments"}, "interaction or shift missing")
    _require(fixed_backgrounds >= 5 and mixed_backgrounds >= 5, "background regimes missing")
    _require(_puzzle_signature(generate_puzzle(999, 7, 3)) ==
             _puzzle_signature(generate_puzzle(999, 7, 3)), "seed determinism failed")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            task = generate_puzzle(5000 + a * 10 + b, a, b)
            _require(len(task["train"]) == a and len(task["test"]) == b, "explicit counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures),
            "normalized_input_structures": len(structures),
            "dimensions": sorted(dimensions), "radii": sorted(radii_seen),
            "palettes": len(palettes), "shell_counts": sorted(shell_counts),
            "fragment_counts": sorted(loads), "fragment_areas": sorted(areas),
            "directions": len(directions), "interior_fragments": outward,
            "exterior_fragments": inward, "pairs_with_foreign_shell_crossings": crossings,
            "fixed_background_puzzles": fixed_backgrounds,
            "mixed_background_puzzles": mixed_backgrounds, "shift_axes": sorted(shifts)}
