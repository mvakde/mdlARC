"""Capped combs: registration completes switches before beads can settle.

Only NumPy and the standard library are used. All interpretation is from pixels;
the sampler's profiles are never used by solve or by semantic validation.
"""
from __future__ import annotations

import itertools
import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(cells, diagonal=False):
    remaining = set(cells)
    directions = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                       if (r or c) and (diagonal or not (r and c)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        part, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _paths(cells):
    """Order every four-connected simple path; reject cycles and branches."""
    paths = []
    for part in _components(cells):
        neighbors = {q: [v for dr, dc in _DIRS
                         if (v := (q[0] + dr, q[1] + dc)) in part] for q in part}
        ends = sorted(q for q in part if len(neighbors[q]) == 1)
        if len(ends) != 2 or any(len(v) not in (1, 2) for v in neighbors.values()):
            raise ValueError("the completed circuit must contain only simple paths")
        ordered, previous, current = [ends[0]], None, ends[0]
        while current != ends[1]:
            choices = [q for q in neighbors[current] if q != previous]
            if len(choices) != 1:
                raise ValueError("ambiguous path")
            previous, current = current, choices[0]
            ordered.append(current)
        if len(ordered) != len(part):
            raise ValueError("cycle")
        # A monotone path has no internal gravity trap.
        changes = {b[0] - a[0] for a, b in zip(ordered, ordered[1:]) if b[0] != a[0]}
        if len(changes) > 1:
            raise ValueError("nonmonotone circuit")
        paths.append(ordered)
    return paths


def _gate_runs(gates):
    runs = _components(gates)
    if any(len({r for r, c in run}) != 1 or len(run) not in (1, 2, 3)
           or _bbox(run)[3] - _bbox(run)[2] + 1 != len(run) for run in runs):
        raise ValueError("switches must be separate horizontal runs of one to three cells")
    return runs


def _circuit(parsed, omit_seam=False):
    """Build simultaneous agreeing switch writes, retaining original bead counts."""
    ordinary = parsed["ordinary"]
    bg, track, bead, gate, marker, cap = parsed["palette"]
    track_cells = {q for q, color in ordinary.items() if color in (track, bead)}
    beads = {q for q, color in ordinary.items() if color == bead}
    gate_cells = {q for q, color in ordinary.items() if color == gate}
    active, inactive, writes, overlaps = [], [], {}, 0
    for run in _gate_runs(gate_cells):
        if len(run) < 3 or (omit_seam and min(c for r, c in run) < 0 <= max(c for r, c in run)):
            inactive.append(run)
            continue
        r, _, lo, hi = _bbox(run)
        c = lo + 1
        if r % 2 != 1 or not all((y, x) in track_cells
                                for y in (r - 1, r + 1) for x in range(lo, hi + 1)):
            raise ValueError("a complete switch needs two fully backed lanes")
        if (r - 1, c) in beads or (r + 1, c) in beads:
            raise ValueError("a bead cannot occupy a switch pivot")
        active.append((r, c))
        for y in range(r - 1, r + 2):
            for x in range(c - 1, c + 2):
                color = gate if x == c else track
                if (y, x) in writes:
                    if writes[y, x] != color:
                        raise ValueError("conflicting switch neighborhoods")
                    overlaps += 1
                writes[y, x] = color
    structural = {q: track for q in track_cells}
    structural.update(writes)
    paths = _paths({q for q, color in structural.items() if color == track})
    ends = {q for p in paths for q in (p[0], p[-1])}
    if not beads <= ends:
        raise ValueError("beads must begin at exposed terminals of completed paths")
    return dict(structural=structural, paths=paths, beads=beads, active=active,
                inactive=inactive, overlaps=overlaps)


def _settle(info, sign=1):
    targets = set()
    single = double = level = empty = 0
    for path in info["paths"]:
        occupied = info["beads"] & {path[0], path[-1]}
        count = len(occupied)
        if not count:
            empty += 1
        elif path[0][0] == path[-1][0]:
            level += 1
            targets.update(occupied)
        else:
            inward = path if sign * path[0][0] > sign * path[-1][0] else path[::-1]
            here = set(inward[:count])
            single += count == 1 and here != occupied
            double += count == 2
            targets.update(here)
    if len(targets) != len(info["beads"]):
        raise ValueError("bead conservation failed")
    return targets, dict(single=single, double=double, level=level, empty=empty)


def _render(parsed, info, targets):
    bg, track, bead, gate, marker, cap = parsed["palette"]
    top, bottom, left, right = _bbox(info["structural"])
    result = np.full((bottom - top + 1, right - left + 1), bg, dtype=np.int8)
    for (r, c), color in info["structural"].items():
        result[r - top, c - left] = bead if (r, c) in targets else color
    return np.rot90(result, -parsed["rotation"]).copy()


def _parse(input_grid):
    """Recover all six roles, the two fragment owners, and cap direction."""
    original = _grid(input_grid)
    colors = tuple(map(int, np.unique(original)))
    if len(colors) != 6:
        raise ValueError("exactly six visible semantic colors are required")
    candidates = []
    for rotation in range(4):
        grid = np.rot90(original, rotation)
        by_color = {color: {tuple(map(int, q)) for q in np.argwhere(grid == color)}
                    for color in colors}
        for marker in colors:
            bars = _components(by_color[marker])
            if len(bars) != 2:
                continue
            length = len(bars[0])
            if length < 3 or length % 2 != 1 or len(bars[1]) != length:
                continue
            if any(len({c for r, c in b}) != 1 or _bbox(b)[1] - _bbox(b)[0] + 1 != length
                   for b in bars):
                continue
            cap_positions = {(_bbox(b)[1] + 1, _bbox(b)[2]) for b in bars}
            cap_colors = [color for color in colors if by_color[color] == cap_positions]
            for cap in cap_colors:
                for bg in colors:
                    if bg in (marker, cap):
                        continue
                    components = _components(set.union(*(by_color[c] for c in colors if c != bg)), True)
                    if len(components) != 2:
                        continue
                    fragments = []
                    for part in components:
                        owned = [b for b in bars if b <= part]
                        if len(owned) != 1:
                            break
                        bar = owned[0]
                        r0, bottom, c0, _ = _bbox(bar)
                        if (bottom + 1, c0) not in part:
                            break
                        pixels = {(r - r0, c - c0): int(grid[r, c]) for r, c in part
                                  if grid[r, c] not in (marker, cap)}
                        if not pixels or any(c == 0 for r, c in pixels):
                            break
                        signs = {int(np.sign(c)) for r, c in pixels}
                        if len(signs) != 1:
                            break
                        fragments.append((next(iter(signs)), pixels, (r0, c0)))
                    if len(fragments) != 2 or {f[0] for f in fragments} != {-1, 1}:
                        continue
                    remaining = [c for c in colors if c not in (bg, marker, cap)]
                    for track, bead, gate in itertools.permutations(remaining):
                        valid = True
                        for side, pixels, origin in fragments:
                            band = {q: color for q, color in pixels.items() if 0 <= q[0] < length}
                            outside = {q: color for q, color in pixels.items() if q not in band}
                            if any(color != gate for (r, c), color in band.items() if r % 2):
                                valid = False
                                break
                            if any(color not in (track, bead) for (r, c), color in band.items() if r % 2 == 0):
                                valid = False
                                break
                            for row in range(0, length, 2):
                                xs = sorted(c for r, c in band if r == row)
                                expected = list(range(min(xs), 0)) if xs and side < 0 else list(range(1, max(xs) + 1)) if xs else []
                                if len(xs) < 3 or xs != expected:
                                    valid = False
                                    break
                                bead_x = [c for c in xs if pixels[row, c] == bead]
                                if bead_x and bead_x != [min(xs) if side < 0 else max(xs)]:
                                    valid = False
                                    break
                            if not valid:
                                break
                            # An outer end on the first or last lane can bend
                            # away from the lane band. These tails make ruler
                            # registration differ from bounding-box alignment.
                            for top_tail, row in ((True, 0), (False, length - 1)):
                                tail = {q: color for q, color in outside.items()
                                        if (q[0] < 0) == top_tail}
                                if not tail:
                                    continue
                                xs = [c for r, c in band if r == row]
                                outer = min(xs) if side < 0 else max(xs)
                                ys = sorted(r for r, c in tail)
                                expected = list(range(min(ys), 0)) if top_tail else list(range(length, max(ys) + 1))
                                terminal = (min(ys) if top_tail else max(ys), outer)
                                if (any(c != outer for r, c in tail) or ys != expected
                                        or band[row, outer] != track
                                        or any(color != track and not (q == terminal and color == bead)
                                               for q, color in tail.items())):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        fragments.sort(key=lambda f: f[0])
                        ordinary = dict(fragments[0][1])
                        ordinary.update({(r, c - 1): color for (r, c), color in fragments[1][1].items()})
                        p = dict(palette=(bg, track, bead, gate, marker, cap), rotation=rotation,
                                 height=length, ordinary=ordinary, fragments=fragments)
                        try:
                            info = _circuit(p)
                        except ValueError:
                            continue
                        if info["active"] and info["beads"]:
                            candidates.append(p)
    if len(candidates) != 1:
        raise ValueError(f"expected one geometric interpretation, found {len(candidates)}")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Translate, complete/turn switches, pack beads toward the ruler caps, crop."""
    parsed = _parse(input_grid)
    info = _circuit(parsed)
    targets, _ = _settle(info)
    return _render(parsed, info, targets)


def _sample_network(rng, lanes, arm_range, richness, require_level, require_stub, require_tail=False):
    """Grow a pixel network, checking its whole graph rather than a route template."""
    lo, hi = arm_range
    for attempt in range(160):
        left = rng.integers(lo, hi + 1, size=lanes).tolist()
        right = rng.integers(lo, hi + 1, size=lanes).tolist()
        track_cells = {(2 * i, c) for i in range(lanes) for c in range(-left[i], right[i])}
        rows = [[] for _ in range(lanes - 1)]
        tail_top = lanes == 7 or rng.random() < .5
        has_tail = require_tail or rng.random() < .6
        if has_tail:
            depth = 1 if lanes == 7 else int(rng.integers(1 if tail_top else 2, 4))
            i = 0 if tail_top else lanes - 1
            c = -left[i] if rng.random() < .5 else right[i] - 1
            base = 2 * i
            track_cells.update((base + (-d if tail_top else d), c) for d in range(1, depth + 1))
        # An isolated edge lane gives an explicit level-retention demonstration.
        if require_level and has_tail:
            blank = lanes - 2 if tail_top else 0
        else:
            blank = int(rng.choice([0, lanes - 2])) if require_level or require_stub else -1
        available = [i for i in range(lanes - 1) if i != blank]
        seam_row = int(rng.choice(available))
        rows[seam_row] = [int(rng.choice([-1, 0]))]
        for i in rng.permutation(lanes - 1):
            i = int(i)
            if i == blank:
                continue
            wanted = int(rng.choice([0, 1, 1, 2, 3])) if richness else int(rng.choice([0, 1, 1]))
            wanted = max(wanted, len(rows[i]))
            low = -min(left[i], left[i + 1]) + 2
            high = min(right[i], right[i + 1]) - 3
            for raw in rng.permutation(np.arange(low, high + 1)):
                c = int(raw)
                if len(rows[i]) >= wanted:
                    break
                if (all(abs(c - old) >= 4 for old in rows[i])
                        and all(abs(c - old) >= 2 for j in (i - 1, i + 1) if 0 <= j < len(rows)
                                for old in rows[j])):
                    rows[i].append(c)
        gates = {(2 * i + 1, x) for i, cs in enumerate(rows) for c in cs for x in range(c - 1, c + 2)}
        stub_options = [i for i, cs in enumerate(rows) if all(abs(c) >= 4 for c in cs)]
        if require_stub or rng.random() < .45:
            if not stub_options:
                continue
            i = int(rng.choice(stub_options))
            length = int(rng.integers(1, 3))
            xs = range(-length, 0) if rng.random() < .5 else range(length)
            gates.update((2 * i + 1, c) for c in xs)
        ordinary = {q: 1 for q in track_cells}
        ordinary.update({q: 3 for q in gates})
        p = dict(ordinary=ordinary, palette=(0, 1, 2, 3, 4, 5), rotation=0)
        try:
            info = _circuit(p)
        except ValueError:
            continue
        if not any(p[0][0] != p[-1][0] for p in info["paths"]):
            continue
        if require_level and not any(p[0][0] == p[-1][0] for p in info["paths"]):
            continue
        return ordinary, info
    # Constructive bounded fallback: one completed seam switch, optional loose
    # half-switch, and an isolated lane. No stored grids or parent pairs.
    left = rng.integers(lo, hi + 1, size=lanes).tolist()
    right = rng.integers(lo, hi + 1, size=lanes).tolist()
    ordinary = {(2 * i, c): 1 for i in range(lanes) for c in range(-left[i], right[i])}
    if require_tail:
        ordinary[-1, -left[0]] = 1
    ordinary.update({(1, c): 3 for c in (-2, -1, 0)})
    if require_stub:
        ordinary[2 * lanes - 3, 0] = 3
    return ordinary, _circuit(dict(ordinary=ordinary, palette=(0, 1, 2, 3, 4, 5), rotation=0))


def _stage(rng, fragments, height, palette, stage, rotation):
    bg, track, bead, gate, marker, cap = palette
    shapes = []
    for pixels in fragments:
        own = dict(pixels)
        own.update({(r, 0): marker for r in range(height)})
        own[height, 0] = cap
        top, bottom, left, right = _bbox(own)
        own = {(r - top, c - left): color for (r, c), color in own.items()}
        shapes.append((own, bottom - top + 1, right - left + 1))
    if rng.random() < .5:
        shapes.reverse()
    a, ah, aw = shapes[0]
    b, bh, bw = shapes[1]
    gap_limit = 30 - (ah + bh if stage != "horizontal" else aw + bw)
    gap = int(rng.integers(1, min(5, gap_limit) + 1))
    if stage == "horizontal":
        offsets = ((0, 0), (int(rng.integers(0, 4)), aw + gap))
    elif stage == "vertical":
        offsets = ((0, int(rng.integers(0, 4))), (ah + gap, int(rng.integers(0, 4))))
    else:
        # Diagonal staging uses the same independent offset in both axes.
        horizontal_gap = int(rng.integers(1, min(5, 30 - aw - bw) + 1))
        offsets = ((0, 0), (ah + gap, aw + horizontal_gap))
    cells = {}
    for own, (dy, dx) in zip((a, b), offsets):
        cells.update({(r + dy, c + dx): color for (r, c), color in own.items()})
    top, bottom, left, right = _bbox(cells)
    h, w = bottom - top + 1, right - left + 1
    pads = []
    for n in (h, w):
        total = int(rng.integers(0, min(4, 30 - n) + 1))
        before = int(rng.integers(total + 1))
        pads.append((before, total - before))
    grid = np.full((h + sum(pads[0]), w + sum(pads[1])), bg, dtype=np.int8)
    for (r, c), color in cells.items():
        grid[r - top + pads[0][0], c - left + pads[1][0]] = color
    return np.rot90(grid, rotation).copy()


def _make_pair(rng, lanes, arm_range, richness, density, palette, stage, rotation,
               single=False, double=False, level=False, stub=False):
    ordinary, info = _sample_network(rng, lanes, arm_range, richness, level, stub, require_tail=single)
    chosen = set()
    for path in info["paths"]:
        chosen.update(q for q in (path[0], path[-1]) if rng.random() < density)
    nonlevel = [p for p in info["paths"] if p[0][0] != p[-1][0]]
    rng.shuffle(nonlevel)
    if single:
        # The single-bead demonstration specifically crosses a registration-
        # completed gate: its destination differs if seam gates are omitted.
        p = dict(ordinary=ordinary, palette=(0, 1, 2, 3, 4, 5), rotation=0)
        omitted = _circuit(p, omit_seam=True)
        suitable = []
        for path in nonlevel:
            high, low = min(path[0], path[-1]), max(path[0], path[-1])
            other = next(route for route in omitted["paths"] if high in (route[0], route[-1]))
            alternative = high if other[0][0] == other[-1][0] else max(other[0], other[-1])
            if alternative != low:
                suitable.append(path)
        if not suitable:
            raise ValueError("missing registration-sensitive single bead")
        path = suitable[0]
        nonlevel.remove(path)
        chosen.difference_update((path[0], path[-1]))
        chosen.add(min(path[0], path[-1]))
    if double:
        path = nonlevel.pop()
        chosen.update((path[0], path[-1]))
    if level:
        path = [p for p in info["paths"] if p[0][0] == p[-1][0]][0]
        chosen.add(path[int(rng.choice([0, len(path) - 1]))])
    if not chosen:
        chosen.add(nonlevel[0][0])
    for q in chosen:
        ordinary[q] = 2
    colored = {q: palette[color] for q, color in ordinary.items()}
    left = {q: color for q, color in colored.items() if q[1] < 0}
    right = {(r, c + 1): color for (r, c), color in colored.items() if c >= 0}
    grid = _stage(rng, (left, right), 2 * lanes - 1, palette, stage, rotation)
    return {"input": grid, "output": solve(grid)}


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count("num_train", num_train, 2, 10)
    ntest = _count("num_test", num_test, 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must be greater than num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    fixed_palette = tuple(map(int, rng.choice(10, 6, replace=False)))
    change_palette = rng.random() < .5
    mixed_rotation = rng.random() < .5
    base_rotation = int(rng.integers(4))
    shift = int(rng.integers(3))
    stages = list(rng.permutation(["horizontal", "vertical", "diagonal"]))
    teaching = list(map(int, rng.permutation(ntrain)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            first = split == "train" and index == teaching[0]
            second = split == "train" and index == teaching[1]
            if shift == 0:
                lanes = int(rng.integers(3, 5) if split == "train" else rng.integers(5, 8))
                arm_range = (3, 10)
            elif shift == 1:
                lanes = int(rng.integers(3, 8))
                arm_range = (7, 11) if split == "train" else (3, 6)
            else:
                lanes = int(rng.integers(3, 8))
                arm_range = (3, 11)
            density = float(rng.uniform(.12, .85))
            palette = tuple(map(int, rng.choice(10, 6, replace=False))) if change_palette else fixed_palette
            rotation = (base_rotation + (index % 2 if split == "train" else int(rng.integers(4)))) % 4 if mixed_rotation else base_rotation
            stage = str(stages[0] if split == "train" and shift == 2 else stages[1 + index % 2]
                        if split == "test" and shift == 2 else rng.choice(stages))
            for attempt in range(24):
                pair = _make_pair(rng, lanes, arm_range, not first and rng.random() < .7, density, palette, stage, rotation,
                                  single=first, double=second, level=first, stub=second)
                signature = (pair["input"].shape, pair["input"].tobytes())
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    parsed = _parse(pair["input"])
    info = _circuit(parsed)
    targets, behaviors = _settle(info)
    invalid_without_completion = False
    try:
        omitted = _circuit(parsed, omit_seam=True)
        omitted_targets, _ = _settle(omitted)
    except ValueError:
        # Removing one switch from an interlaced arrangement can produce a
        # cycle or a trap. Record that separately from a changed bead answer.
        omitted_targets = targets
        invalid_without_completion = True
    f = dict(behaviors)
    f.update(palette=parsed["palette"], rotation=parsed["rotation"], lanes=(parsed["height"] + 1) // 2,
             switches=len(info["active"]), stubs=len(info["inactive"]), overlaps=info["overlaps"],
             beads=len(info["beads"]), paths=len(info["paths"]),
             seam_switches=sum(c in (-1, 0) for r, c in info["active"]),
             registration_changes_beads=targets != omitted_targets,
             registration_invalid_without_completion=invalid_without_completion,
             bent_terminals=sum(r < 0 or r >= parsed["height"] for r, c in parsed["ordinary"]),
             bounds_differ=(_bbox(parsed["fragments"][0][1])[:2] != _bbox(parsed["fragments"][1][1])[:2]),
             packing_changes_beads=targets != info["beads"],
             level_beads={q for path in info["paths"] if path[0][0] == path[-1][0]
                          for q in (path[0], path[-1]) if q in info["beads"]})
    return parsed, info, targets, f


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            signature = (inp.shape, inp.tobytes())
            assert signature not in seen
            seen.add(signature)
            p, info, targets, f = _features(pair)
            assert np.array_equal(out, _render(p, info, targets)), "incorrect assembly/rewiring/packing/crop"
            bg, track, bead, gate, marker, cap = p["palette"]
            assert marker not in out and cap not in out
            assert np.count_nonzero(out == bead) == len(info["beads"])
            assert np.count_nonzero(out == gate) == 3 * len(info["active"])
            assert f["seam_switches"] > 0
            assert all(q in targets for q in f["level_beads"])
            for path in info["paths"]:
                cells = set(path)
                assert len(targets & cells) == len(info["beads"] & cells), "per-path conservation"
            if split == "train":
                training.append(f)
    for behavior in ("single", "double", "level", "stubs", "registration_changes_beads", "packing_changes_beads", "bounds_differ"):
        assert any(f[behavior] for f in training), f"training does not expose {behavior}"


def self_test() -> dict:
    """Bounded 42-seed survey, semantic ablations, equivariance, negative labels."""
    shapes, normalized, palettes, rotations, lanes, switch_counts, bead_counts = (set() for _ in range(7))
    fingerprints, all_features = set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprint = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                p, info, targets, f = _features(pair)
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(solve(inp.copy()), out)
                assert np.array_equal(solve(np.rot90(inp)), np.rot90(out))
                role_map = {color: i for i, color in enumerate(p["palette"])}
                roles = np.array([[role_map[int(c)] for c in row] for row in inp], dtype=np.int8)
                normalized.add((roles.shape, roles.tobytes()))
                shapes.add(inp.shape)
                palettes.add(p["palette"])
                rotations.add(p["rotation"])
                lanes.add(f["lanes"])
                switch_counts.add(f["switches"])
                bead_counts.add(f["beads"])
                all_features.append(f)
                fingerprint.append((inp.shape, inp.tobytes(), out.shape, out.tobytes()))
        fingerprints.add(tuple(fingerprint))
    assert len(fingerprints) == 42 and len(normalized) >= 180 and len(shapes) >= 40
    assert len(palettes) >= 40 and rotations == set(range(4)) and lanes == set(range(3, 8))
    assert len(switch_counts) >= 5 and len(bead_counts) >= 8
    assert any(f["overlaps"] for f in all_features) and any(f["empty"] for f in all_features)
    assert any(f["bent_terminals"] for f in all_features)
    # Literal palette permutation preserves the entire answer, including roles
    # that were never taught using the same digits.
    sample = generate_puzzle(0)["train"][0]
    perm = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int8)
    assert np.array_equal(solve(perm[sample["input"]]), perm[sample["output"]])
    # Hand-derived regression: a 2+1 seam bar, a loose 2-cell bar, one moving
    # bead, a two-bead collision and an occupied level route. It is not sampled.
    manual_input = ("02111400411112", "00033400430000", "11111400411200",
                    "00000400433000", "00211400411110", "00000500500000")
    manual_output = ("0111311111", "0001310000", "2111312200", "0000000000", "0021111110")
    digits = lambda rows: np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
    assert np.array_equal(solve(digits(manual_input)), digits(manual_output))
    # A smaller valid family member is supported despite not being sampled.
    assert np.array_equal(solve(digits(manual_input[:3] + (manual_input[-1],))), digits(manual_output[:3]))
    # Wrong packed bead placement must be rejected, even when counts agree.
    corrupt = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in generate_puzzle(0)[s]]
               for s in ("train", "test")}
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(73, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.2), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(314159, 6, 3), generate_puzzle(314159, 6, 3)
    assert all(np.array_equal(pa[k], pb[k]) for s in a for pa, pb in zip(a[s], b[s]) for k in pa)
    return dict(seeds=42, pairs=len(all_features), distinct_puzzles=len(fingerprints),
                normalized_inputs=len(normalized), input_shapes=len(shapes), palettes=len(palettes),
                lanes=sorted(lanes), switch_counts=sorted(switch_counts), bead_counts=sorted(bead_counts),
                rotations=sorted(rotations), registration_sensitive_pairs=sum(f["registration_changes_beads"] for f in all_features),
                packed_pairs=sum(f["packing_changes_beads"] for f in all_features),
                bent_terminal_pairs=sum(bool(f["bent_terminals"]) for f in all_features),
                overlapping_neighborhood_pairs=sum(bool(f["overlaps"]) for f in all_features))
