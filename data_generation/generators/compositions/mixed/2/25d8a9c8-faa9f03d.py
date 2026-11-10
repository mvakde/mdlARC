"""Repair/reroute thin traces, then report whole-row color equality.

Fixed family roles: 0 background, 2 repair, 4 direction, 5 positive.
Payload colors are 1,3,6,7,8,9. No parent code or data is loaded.
All tests are in memory; this module writes no artifacts.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np


BG, REPAIR, DIRECTION, POSITIVE = 0, 2, 4, 5
PAYLOAD = (1, 3, 6, 7, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0,219169629,288785498,914531811,589426625,17110585,643699796,568252339,682921986,590028407,241602523,768783326,845522603,32137891,592371457,484904402,920782000,624034271,6126513,452895339,179923831,133379568,689525439,947298992,340179759,242735633,502198311,88178554,901516287,858165937,759307463,753236903,583131697,863742678,570574065,746919518,632807070,563532798,813974871,593649497,8096830,345009358)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"illegal {name}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(a.dtype.kind in "iu" and np.all((0 <= a) & (a <= 9)), "invalid grid colors")
    return a.astype(np.int8, copy=True)


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _line(a, b):
    _require(a[0] == b[0] or a[1] == b[1], "nonorthogonal segment")
    n = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    d = (int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1])))
    return [(a[0] + i*d[0], a[1] + i*d[1]) for i in range(n + 1)]


def _polyline(points):
    cells = [points[0]]
    for a, b in zip(points, points[1:]):
        cells.extend(_line(a, b)[1:])
    return cells


def _ray(p, d, shape):
    cells = set()
    p = _add(p, d)
    while _inside(p, shape):
        cells.add(p)
        p = _add(p, d)
    return cells


def _neighbors(p, cells):
    return [q for d in DIRS if (q := _add(p, d)) in cells]


def _straight(p, cells, vertical=None):
    ns = _neighbors(p, cells)
    ok = len(ns) == 2 and tuple(ns[0][i] + ns[1][i] for i in (0, 1)) == (2*p[0], 2*p[1])
    return ok and (vertical is None or ((ns[0][1] == p[1]) == vertical))


def _path_ok(walk):
    cells = set(walk)
    return len(cells) == len(walk) and all(len(_neighbors(p, cells)) <= 2 for p in cells)


def _reconstruct(inp):
    """Local marked waypoints and supported isolated straight gaps only."""
    work = inp.copy()
    pending = set(map(tuple, np.argwhere(inp == REPAIR).tolist()))
    for _ in range(len(pending) + 1):
        resolved = {}
        for p in sorted(pending):
            adjacent = {int(work[q]) for d in DIRS if _inside(q := _add(p, d), inp.shape)
                        and int(work[q]) in PAYLOAD}
            aligned = set()
            for d in ((1, 0), (0, 1)):
                a, b = _add(p, d), _add(p, (-d[0], -d[1]))
                if _inside(a, inp.shape) and _inside(b, inp.shape):
                    if work[a] == work[b] and int(work[a]) in PAYLOAD:
                        aligned.add(int(work[a]))
            choices = adjacent if len(adjacent) == 1 else aligned
            if len(choices) == 1:
                resolved[p] = next(iter(choices))
        if not resolved:
            break
        for p, color in resolved.items():
            work[p] = color
        pending.difference_update(resolved)
    _require(not pending, "ambiguous repair waypoint")
    layers = {c: set(map(tuple, np.argwhere(work == c).tolist()))
              for c in PAYLOAD if np.any(work == c)}
    _require(layers, "no trace")
    for color, cells in layers.items():
        observed = set(cells)
        additions = set()
        for p in sorted(observed):
            for d in ((1, 0), (0, 1)):
                mid, q = _add(p, d), _add(p, (2*d[0], 2*d[1]))
                if q not in observed or int(inp[mid]) == DIRECTION:
                    continue
                # At least one fragment continues straight beyond the gap.
                # Parallel complete rows are not evidence of vertical strokes.
                if _add(p, (-d[0], -d[1])) in observed or _add(q, d) in observed:
                    additions.add(mid)
        cells.update(additions)
    return work, layers


def _edits(inp, layers):
    edits = []
    for q in map(tuple, np.argwhere(inp == DIRECTION).tolist()):
        choices = []
        for d in DIRS:
            pivot = _add(q, (-d[0], -d[1]))
            first = _add(pivot, (-d[0], -d[1]))
            for color, cells in layers.items():
                if pivot not in cells or first not in cells or q in cells:
                    continue
                if not any(_add(pivot, t) in cells for t in ((-d[1], d[0]), (d[1], -d[0]))):
                    continue
                previous, current, old = pivot, first, []
                for _ in range(inp.size):
                    if current in old:
                        break
                    old.append(current)
                    following = [p for p in _neighbors(current, cells) if p != previous]
                    if not following:
                        if current[0] in (0, inp.shape[0]-1) or current[1] in (0, inp.shape[1]-1):
                            choices.append({"pivot": pivot, "color": color, "direction": d,
                                            "old": set(old), "walk": tuple([pivot] + old),
                                            "new": _ray(pivot, d, inp.shape)})
                        break
                    if len(following) != 1:
                        break
                    previous, current = current, following[0]
        _require(len(choices) == 1, "direction must identify one terminal branch")
        edits.append(choices[0])
    for i, a in enumerate(edits):
        for b in edits[i+1:]:
            _require(a["color"] != b["color"] or not a["old"] & b["old"], "overlapping deletions")
            _require(a["color"] == b["color"] or not a["new"] & b["new"], "different-color new rays meet")
            _require(a["color"] != b["color"] or
                     (a["pivot"] not in b["old"] and b["pivot"] not in a["old"]), "removed pivot")
    return edits


def _render(shape, owner, layers, edits=()):
    kept = {c: set(cells) for c, cells in layers.items()}
    for e in edits:
        kept[e["color"]].difference_update(e["old"])
    memberships = {}
    for color, cells in kept.items():
        for p in cells:
            memberships.setdefault(p, []).append(color)
    out = np.zeros(shape, dtype=np.int8)
    for p, colors in memberships.items():
        _require(len(colors) <= 2, "three-layer crossing")
        if len(colors) == 1:
            out[p] = colors[0]
        else:
            _require(int(owner[p]) in colors, "crossing owner is not visible")
            out[p] = owner[p]
    # Deletion is simultaneous; every new ray is drawn afterward.
    for e in edits:
        for p in e["new"]:
            out[p] = e["color"]
    return out


def _mask(grid):
    return np.all(grid == grid[:, :1], axis=1)


def _binary(grid):
    return np.repeat(np.where(_mask(grid), POSITIVE, BG).astype(np.int8)[:, None], grid.shape[1], axis=1)


def _analyze(input_grid):
    inp = _grid(input_grid)
    _require(not np.any(inp == POSITIVE), "positive marker is output-only")
    work, layers = _reconstruct(inp)
    edits = _edits(inp, layers)
    before = _render(inp.shape, work, layers)
    after = _render(inp.shape, work, layers, edits)
    return inp, work, layers, edits, before, after


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only the supplied grid, with fixed, documented family conventions."""
    return _binary(_analyze(input_grid)[-1])


def _compatible(cells, others):
    for other in others.values():
        for p in cells & other:
            if not _straight(p, cells) or not _straight(p, other):
                return False
            if set(_neighbors(p, cells)) & set(_neighbors(p, other)):
                return False
        for p in cells - other:
            ns = _neighbors(p, other)
            if ns and not any(q in cells for q in ns):
                return False
    return True


def _random_route(rng, shape, color, max_bends):
    side = int(rng.integers(4))
    if side < 2:
        start = (0 if side == 0 else shape[0]-1, int(rng.integers(2, shape[1]-2)))
        axis = 0
    else:
        start = (int(rng.integers(2, shape[0]-2)), 0 if side == 2 else shape[1]-1)
        axis = 1
    points = [start]
    for _ in range(int(rng.integers(max_bends + 1)) + 1):
        options = [v for v in range(1, shape[axis]-1) if abs(v-points[-1][axis]) >= 3]
        rng.shuffle(options)
        for v in options:
            p = list(points[-1])
            p[axis] = v
            if _path_ok(_polyline(points + [tuple(p)])):
                points.append(tuple(p))
                break
        else:
            return None
        axis = 1-axis
    pivot, prev = points[-1], points[-2]
    d = (int(np.sign(pivot[0]-prev[0])), int(np.sign(pivot[1]-prev[1])))
    q = _add(pivot, d)
    if not _inside(q, shape):
        return None
    options = [v for v in range(shape[axis]) if 2 <= abs(v-pivot[axis]) <= 7]
    rng.shuffle(options)
    for v in options:
        end = list(pivot)
        end[axis] = v
        walk = _polyline(points + [tuple(end)])
        if _path_ok(walk) and q not in walk:
            at = walk.index(pivot)
            return set(walk), {"pivot": pivot, "color": color, "direction": d,
                               "old": set(walk[:at]), "walk": tuple(reversed(walk[:at+1])),
                               "new": _ray(pivot, d, shape)}
    return None


def _scene(rng, shape, colors, count, max_bends):
    layers, edits = {}, []
    for color in colors[:count]:
        for _ in range(60):
            proposal = _random_route(rng, shape, color, max_bends)
            if proposal is None:
                continue
            cells, edit = proposal
            qs = [_add(e["pivot"], e["direction"]) for e in edits]
            q = _add(edit["pivot"], edit["direction"])
            if not _compatible(cells, layers) or any(p in cells for p in qs):
                continue
            if any(q in s for s in layers.values()) or q in qs:
                continue
            if any(edit["new"] & e["new"] for e in edits):
                continue
            layers[color] = cells
            edits.append(edit)
            break
        else:
            return None
    return layers, edits


def _draw_initial(shape, layers, rng):
    out = np.zeros(shape, dtype=np.int8)
    colors = list(layers)
    rng.shuffle(colors)
    for c in colors:
        for p in layers[c]:
            out[p] = c
    return out


def _add_probe(layers, row, width, colors, style, rng):
    a, b = colors[:2]
    if style == "split":
        cut = int(rng.integers(3, width-2))
        segments = [(a, 0, cut), (b, cut, width)]
    elif style == "left":
        segments = [(a, 1, width)]
    elif style == "right":
        segments = [(a, 0, width-1)]
    elif style == "short":
        lo = int(rng.integers(0, width-4))
        hi = int(rng.integers(lo+3, width+1))
        segments = [(a, lo, hi)]
    else:
        segments = [(a, 0, width)]
    for c, lo, hi in segments:
        layers.setdefault(c, set()).update((row, col) for col in range(lo, hi))


def _edit_key(edits):
    return {(e["pivot"], e["direction"], e["color"], frozenset(e["old"])) for e in edits}


def _bend_count(edit):
    ds = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(edit["walk"], edit["walk"][1:])]
    return sum(a != b for a, b in zip(ds, ds[1:]))


def _without_repair(inp, complete, final, edits):
    # Diagnostic ablation, given the actual edits: leave every surviving
    # corruption unresolved, while still executing the correct rerouting.
    result = final.copy()
    overwritten = set().union(*(e["old"] | e["new"] for e in edits)) if edits else set()
    for p in map(tuple, np.argwhere((inp != complete) & (inp != DIRECTION)).tolist()):
        if p not in overwritten:
            result[p] = inp[p]
    return result


def _damage(rng, shape, layers, edits, force_old_crossings=False):
    complete = _draw_initial(shape, layers, rng)
    if force_old_crossings:
        # A visible old crossing must actually interrupt the full probe in
        # the simple demonstration. Ownership is recorded in the input.
        for e in edits:
            for p in e["old"]:
                if any(p in s for c, s in layers.items() if c != e["color"]):
                    complete[p] = e["color"]
    inp = complete.copy()
    for e in edits:
        q = _add(e["pivot"], e["direction"])
        if inp[q] != BG:
            return None
        inp[q] = DIRECTION
    try:
        _, recovered = _reconstruct(inp)
        if recovered != layers or _edit_key(_edits(inp, recovered)) != _edit_key(edits):
            return None
        final = _render(shape, complete, layers, edits)
    except AssertionError:
        return None
    if not np.any(_mask(final)) or np.all(_mask(final)):
        return None
    if np.array_equal(_mask(complete), _mask(final)):
        return None
    # Essential repair on a final complete foreground row, away from crossings.
    candidates = [p for p in set().union(*layers.values())
                  if sum(p in s for s in layers.values()) == 1]
    rng.shuffle(candidates)
    positive_sites = [p for p in candidates if _mask(final)[p[0]] and final[p] != BG
                      and complete[p] == final[p] and 1 <= p[1] < shape[1]-1]
    if len(positive_sites) < 2:
        return None
    def try_change(p, replacement):
        previous = int(inp[p])
        inp[p] = replacement
        try:
            work, recovered = _reconstruct(inp)
            _require(recovered == layers, "damage loses geometry")
            inferred = _edits(inp, recovered)
            _require(_edit_key(inferred) == _edit_key(edits), "damage changes pivot interpretation")
            _require(np.array_equal(_render(shape, work, recovered, inferred), final), "damage changes crossing")
        except AssertionError:
            inp[p] = previous
            return False
        return True

    used, successes, gaps = set(), 0, 0
    # Two surviving full-row damages establish a necessary repair operation.
    for p in positive_sites:
        used.add(p)
        replacement = BG if successes == 0 else REPAIR
        if try_change(p, replacement):
            successes += 1
            gaps += replacement == BG
        if successes == 2:
            break
    # Then put damage on the rest of the network, including marked pivots,
    # bends and endpoints, rather than keeping every repair in the probe row.
    special = [p for p in candidates if not _straight(p, layers[int(complete[p])])]
    rng.shuffle(special)
    ordered = ([e["pivot"] for e in edits] if rng.random() < .6 else []) + special[:2] + candidates + special[2:]
    target = int(rng.integers(3, 8))
    for p in ordered:
        if p in used or inp[p] in (REPAIR, DIRECTION):
            continue
        used.add(p)
        replacement = BG if rng.random() < .35 else REPAIR
        if try_change(p, replacement):
            successes += 1
            gaps += replacement == BG
        if successes >= target:
            break
    if successes < 2 or not gaps or not np.any(inp == REPAIR):
        return None
    if np.array_equal(_mask(_without_repair(inp, complete, final, edits)), _mask(final)):
        return None
    return {"input": inp, "output": _binary(final)}


def _teaching_pair(rng, payloads):
    shape = (int(rng.integers(20, 29)), int(rng.integers(10, 29)))
    h, w = shape
    row = int(rng.integers(11, h-4))
    col = int(rng.integers(2, w-2))
    endpoint = (row, col + 2 if col+2 < w else col-2)
    color = payloads[0]
    walk = _polyline([(0, col), (row, col), endpoint])
    layers = {color: set(walk)}
    edit = {"pivot": (row, col), "color": color, "direction": (1, 0),
            "old": set(_line((0, col), (row-1, col))),
            "walk": tuple(_line((row, col), (0, col))),
            "new": _ray((row, col), (1, 0), shape)}
    rows = sorted(map(int, rng.choice(np.arange(1, row-4), 4, replace=False)))
    rows = [r+i for i, r in enumerate(rows)]
    for r, style in zip(rows, ("full", "split", "left", "right")):
        _add_probe(layers, r, w, payloads[3:], style, rng)
    _add_probe(layers, int(rng.integers(row+2, h-1)), w, payloads[3:], "full", rng)
    pair = _damage(rng, shape, layers, [edit], force_old_crossings=True)
    if pair is not None:
        if rng.random() < .5:
            pair = {k: np.flipud(a).copy() for k, a in pair.items()}
        if rng.random() < .5:
            pair = {k: np.fliplr(a).copy() for k, a in pair.items()}
    return pair


def _random_pair(rng, payloads, count, max_bends, test, teach_bends=False):
    shape = (int(rng.integers(14 if not test else 17, 31)), int(rng.integers(10, 31)))
    scene = _scene(rng, shape, payloads, count, max_bends)
    if scene is None:
        return None
    layers, edits = scene
    if test and max(map(_bend_count, edits)) < 2:
        return None
    if teach_bends and (max(map(_bend_count, edits)) < 1 or not any(e["direction"][1] for e in edits)):
        return None
    marker_rows = {_add(e["pivot"], e["direction"])[0] for e in edits}
    eligible = [r for r in range(1, shape[0]-1) if r not in marker_rows and all(
        _straight(p, s, vertical=True) for s in layers.values() for p in s if p[0] == r)]
    rng.shuffle(eligible)
    rows = []
    desired = int(rng.integers(3, 7))
    for r in eligible:
        if all(abs(r-a) >= 2 for a in rows):
            rows.append(r)
            if len(rows) == desired:
                break
    if len(rows) < 3:
        return None
    for i, r in enumerate(rows):
        style = "full" if i < 2 else str(rng.choice(("full", "split", "left", "right", "short")))
        probe_colors = list(payloads[3:])
        rng.shuffle(probe_colors)
        _add_probe(layers, r, shape[1], probe_colors, style, rng)
    return _damage(rng, shape, layers, edits)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 0, 2**64-1, "seed")
    nt = _integer(num_train, 2, 10, "num_train")
    nq = _integer(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    payloads = tuple(map(int, rng.permutation(PAYLOAD)))
    puzzle, seen = {"train": [], "test": []}, set()
    for index in range(nt + nq):
        test = index >= nt
        pair_payloads = tuple(map(int, rng.permutation(payloads)))
        # Two examples always provide simple and combined directional evidence.
        count = 1 if index == 0 else (2 if index == 1 else int(rng.integers(1 if not test else 2, 4)))
        max_bends = 1 if not test else 3
        for _ in range(180):
            pair = _teaching_pair(rng, pair_payloads) if index == 0 else _random_pair(
                rng, pair_payloads, count, max_bends, test, teach_bends=index == 1)
            if pair is None:
                continue
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in seen:
                seen.add(key)
                puzzle["test" if test else "train"].append(pair)
                break
        else:
            raise RuntimeError(f"bounded construction exhausted at pair {index}")
    # Demonstrations need not reveal the construction order.
    rng.shuffle(puzzle["train"])
    return puzzle


def _features(inp):
    inp, work, layers, edits, before, after = _analyze(inp)
    mask = _mask(after)
    full = [row for row in after if np.all(row != BG) and not np.all(row == row[0])]
    turns = [_bend_count(e) for e in edits]
    return {"shape": tuple(inp.shape), "colors": len(layers), "redirects": len(edits),
            "directions": {e["direction"] for e in edits}, "bends": turns,
            "positive_rows": int(mask.sum()), "mask": tuple(map(bool, mask)),
            "full_positive": bool(np.any(mask & np.any(after != BG, axis=1))),
            "empty_positive": bool(np.any(np.all(after == BG, axis=1))),
            "full_mixed": bool(full),
            "endpoint_trap": bool(np.any((after[:, 0] == after[:, -1]) & ~mask)),
            "first_exception": any(row[0] != row[1] and np.all(row[1:] == row[1]) for row in after),
            "last_exception": any(row[-1] != row[0] and np.all(row[:-1] == row[0]) for row in after),
            "repair_essential": not np.array_equal(_mask(_without_repair(inp, before, after, edits)), mask),
            "reroute_essential": not np.array_equal(_mask(before), mask),
            "restored_foreground_equality": bool(np.any(~_mask(before) & mask & np.any(after != BG, axis=1))),
            "broken_foreground_equality": bool(np.any(_mask(before) & ~mask & np.any(before != BG, axis=1))),
            "crossings": sum(sum(p in s for s in layers.values()) > 1 for p in set().union(*layers.values())),
            "gaps": int(np.count_nonzero((inp == BG) & (before != BG))),
            "repair_markers": int(np.count_nonzero(inp == REPAIR))}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid counts")
    seen, teaching = set(), Counter()
    train_bends, test_bends = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape changed")
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _require(set(map(int, np.unique(out))) <= {BG, POSITIVE}, "not a binary mask")
            _require(np.array_equal(solve(inp.copy()), out), "incorrect output")
            features = _features(inp)
            _require(features["reroute_essential"] and features["repair_essential"], "inactive inherited operation")
            _require(features["full_positive"], "missing complete foreground row")
            _require(features["gaps"] >= 1 and features["repair_markers"] >= 1, "missing repair evidence")
            if split == "train":
                teaching.update({k: int(features[k]) for k in ("full_mixed", "endpoint_trap", "first_exception", "last_exception", "empty_positive", "restored_foreground_equality", "broken_foreground_equality")})
                teaching["multi"] += features["redirects"] > 1
                teaching["horizontal"] += any(d[1] for d in features["directions"])
                train_bends.extend(features["bends"])
            else:
                test_bends.extend(features["bends"])
                _require(features["redirects"] >= 2 and max(features["bends"]) >= 2, "missing test shift")
    _require(all(teaching[k] for k in ("full_mixed", "endpoint_trap", "first_exception", "last_exception", "empty_positive", "multi", "horizontal", "restored_foreground_equality", "broken_foreground_equality")),
             "training lacks necessary equality or multi-edit contrasts")
    _require(1 in train_bends and max(train_bends) <= 1 and max(test_bends) >= 2, "branch-depth shift not established")


def self_test():
    """Bounded, streaming 42-seed semantic and diversity checks; no saved grids."""
    # A hand-constructed regression: the 1 branch leaves rows 0..9, uncovers
    # a 6 row with marked/gapped cells, and moves below row 10.
    # Mixed 3/9, clipped 7/8, and the new crossing on row 13 stay negative.
    rows = ("00010000000", "00010000000", "66616662606", "00010000000",
            "33313999999", "00010000000", "07717777777", "00010000000",
            "88818888880", "00010000000", "00011110000", "00040000000",
            "00000000000", "99999999999", "00000000000", "00000000000", "00000000000")
    regression = np.array([list(map(int, row)) for row in rows], dtype=np.int8)
    expected = np.zeros_like(regression)
    expected[[0, 1, 2, 3, 5, 7, 9]] = POSITIVE
    _require(np.array_equal(solve(regression), expected), "hand-constructed row-mask regression")
    shapes, structures, signatures, masks, directions, bends, counts, palettes = (set() for _ in range(8))
    pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                f = _features(inp)
                shapes.add(f["shape"])
                masks.add(f["mask"])
                directions.update(f["directions"])
                bends.update(f["bends"])
                counts.add(f["redirects"])
                palettes.add(tuple(map(int, np.unique(inp))))
                roles = {}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in inp.flat)
                structures.add((inp.shape, hashlib.sha256(norm).digest()))
                digest.update(bytes(inp.shape) + inp.tobytes() + out.tobytes())
                # Reflections preserve row classification; transposition does not.
                for axis in (0, 1):
                    _require(np.array_equal(solve(np.flip(inp, axis)), np.flip(out, axis)), "reflection error")
                _, work, layers, edits, _, after = _analyze(inp)
                _require(np.array_equal(_render(inp.shape, work, layers, list(reversed(edits))), after), "edit order dependence")
                pairs += 1
        signatures.add(digest.digest())
    _require(len(signatures) == 42 and len(shapes) >= 50 and len(structures) >= 190, "weak structural variation")
    _require(directions == set(DIRS) and {0, 1, 2} <= bends and {1, 2, 3} <= counts, "missing route variation")
    _require(len(masks) >= 80 and len(palettes) >= 20, "weak mask or palette variation")
    a, b = generate_puzzle(0), generate_puzzle(0)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x), "nondeterministic")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(123 + nt*10 + nq, nt, nq))
    for nt, nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1),(4,None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for whole_row in (False, True):
        bad = {s: [dict(p) for p in a[s]] for s in a}
        bad["test"][0]["output"] = bad["test"][0]["output"].copy()
        out = bad["test"][0]["output"]
        if whole_row:
            out[0] = POSITIVE - out[0]
        else:
            out[0, 0] = POSITIVE - out[0, 0]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupt binary output accepted")
    return {"seeds": 42, "pairs": pairs, "shapes": len(shapes), "structures": len(structures),
            "masks": len(masks), "palettes": len(palettes), "directions": sorted(directions),
            "old_bends": sorted(bends), "redirect_counts": sorted(counts)}
