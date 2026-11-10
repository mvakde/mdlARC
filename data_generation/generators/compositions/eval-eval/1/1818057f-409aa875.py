"""Remote constellation acknowledgement: virtual pluses select their senders.

Only standard-library modules and NumPy are used. No files are read or written.
Construction works backwards from dot patterns; solve works forwards from pixels.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np


DISTANCE = 5
SINGLE = 8
COLLISION = 2
DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))
INPUT_COLORS = tuple(c for c in range(10) if c not in (SINGLE, COLLISION))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grids must be integer 2D arrays, sides 1..30, colors 0..9")
    return a


def _arrow_cells(tip, direction):
    r, c = tip
    dr, dc = direction
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return frozenset((r + a, c + b) for a, b in offsets)


def _parse(input_grid):
    """Read monochrome 8-connected arrow components from the input alone."""
    a = _grid(input_grid)
    colors, amounts = np.unique(a, return_counts=True)
    background = int(colors[int(np.argmax(amounts))])
    if int(amounts.max()) <= a.size / 2 or background not in INPUT_COLORS:
        raise ValueError("a non-acknowledgement background must occupy over half the grid")
    if any(int(c) not in INPUT_COLORS for c in colors):
        raise ValueError("input colors 2 and 8 are reserved for acknowledgements")
    remaining = set(map(tuple, np.argwhere(a != background).tolist()))
    if not remaining:
        raise ValueError("at least one arrow is required")
    arrows = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, pending = {start}, [start]
        while pending:
            r, c = pending.pop()
            near = {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)} & remaining
            remaining.difference_update(near)
            component.update(near)
            pending.extend(near)
        if len(component) != 3 or len({int(a[p]) for p in component}) != 1:
            raise ValueError("arrows must be separated, monochrome triominoes")
        matches = [(tip, d, frozenset(component)) for tip in sorted(component)
                   for d in DIRECTIONS if _arrow_cells(tip, d) == component]
        if len(matches) != 1:
            raise ValueError("unrecognized or ambiguous arrowhead")
        tip, direction, cells = matches[0]
        endpoint = (tip[0] + DISTANCE * direction[0], tip[1] + DISTANCE * direction[1])
        if not (0 <= endpoint[0] < a.shape[0] and 0 <= endpoint[1] < a.shape[1]):
            raise ValueError("every five-step endpoint must be inside the canvas")
        arrows.append((tip, direction, cells, endpoint))
    return background, arrows


def _mask_union(mask):
    centers = np.zeros(mask.shape, dtype=bool)
    centers[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
                          & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:])
    union = centers.copy()
    union[1:] |= centers[:-1]
    union[:-1] |= centers[1:]
    union[:, 1:] |= centers[:, :-1]
    union[:, :-1] |= centers[:, 1:]
    return centers, union


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Project, recognize virtual pluses simultaneously, acknowledge senders."""
    a = _grid(input_grid)
    _, arrows = _parse(a)
    counts = np.zeros(a.shape, dtype=np.int16)
    for _, _, _, endpoint in arrows:
        counts[endpoint] += 1
    _, selected = _mask_union(counts > 0)
    out = a.copy()
    for _, _, cells, endpoint in arrows:
        if selected[endpoint]:
            color = SINGLE if counts[endpoint] == 1 else COLLISION
            for p in cells:
                out[p] = color
    return out


def _plus(p):
    r, c = p
    return {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _scan_points(points):
    """Set-based check of local geometry, separate from production mask shifts."""
    centers = {p for p in points if _plus(p) <= points}
    selected = set().union(*(_plus(p) for p in centers)) if centers else set()
    return centers, selected


def _facts(input_grid):
    bg, arrows = _parse(input_grid)
    counts = Counter(o[3] for o in arrows)
    centers, selected = _scan_points(set(counts))
    owners = {p for o in arrows for p in o[2]}
    inactive = set(counts) - selected
    sequential_points, sequential_selected = set(counts), set()
    for p in sorted(counts):
        cross = _plus(p)
        if cross <= sequential_points:
            sequential_points -= cross
            sequential_selected |= cross
    # A concrete plausible alternative: stop at the first physical arrow before
    # the five-step destination, then apply the same constellation rule there.
    stopped = []
    for tip, d, cells, endpoint in arrows:
        for step in range(1, DISTANCE):
            p = (tip[0] + step * d[0], tip[1] + step * d[1])
            if p in owners and p not in cells:
                endpoint = p
                break
        stopped.append(endpoint)
    stop_counts = Counter(stopped)
    _, stop_selected = _scan_points(set(stopped))
    first_stop_witness = False
    for arrow, endpoint in zip(arrows, stopped):
        original = arrow[3]
        old_role = (SINGLE if counts[original] == 1 else COLLISION) if original in selected else -1
        stop_role = (SINGLE if stop_counts[endpoint] == 1 else COLLISION) if endpoint in stop_selected else -1
        first_stop_witness |= old_role != stop_role
    return {"background": bg, "arrows": arrows, "counts": counts,
            "centers": centers, "selected": selected, "inactive": inactive,
            "overlap": len(selected) < 5 * len(centers),
            "sequential_witness": sequential_selected != selected,
            "first_stop_witness": first_stop_witness,
            "active_collision": any(counts[p] > 1 for p in selected),
            "inactive_collision": any(counts[p] > 1 for p in inactive),
            "attached_inactive": any((r + dr, c + dc) in selected for r, c in inactive
                                     for dr, dc in CARDINALS),
            "endpoint_on_arrow": bool(set(counts) & owners)}


def _assign_arrows(rng, targets, shape, node_limit=1200):
    """Finite constraint search: invert tip+5*direction for each required dot.

    Every branch places one arrow. Candidates and recursion are bounded by
    24 targets * 8 directions and node_limit nodes. No search trace is retained.
    """
    dots = set(targets)
    pools = []
    for r, c in targets:
        pool = []
        for d in DIRECTIONS:
            tip = (r - DISTANCE * d[0], c - DISTANCE * d[1])
            cells = _arrow_cells(tip, d)
            if all(0 <= a < shape[0] and 0 <= b < shape[1] for a, b in cells) and not cells & dots:
                halo = frozenset((a + x, b + y) for a, b in cells
                                 for x in (-1, 0, 1) for y in (-1, 0, 1))
                # A one-cell padded, lossless bitmap makes the hot conflict
                # checks integer operations. No layouts or traces are saved.
                stride = shape[1] + 2
                cell_bits = sum(1 << ((a + 1) * stride + b + 1) for a, b in cells)
                halo_bits = sum(1 << ((a + 1) * stride + b + 1) for a, b in halo)
                pool.append((tip, d, cells, (r, c), cell_bits, halo_bits))
        rng.shuffle(pool)
        pools.append(pool)
    visits = 0

    def search(left, occupied, result):
        nonlocal visits
        visits += 1
        if visits > node_limit:
            return None
        if not left:
            return result
        options = []
        for i in left:
            candidates = [v for v in pools[i] if not occupied & v[5]]
            if not candidates:
                return None
            options.append((len(candidates), i, candidates))
        _, i, candidates = min(options, key=lambda x: x[0])
        rest = [j for j in left if j != i]
        for arrow in candidates:
            answer = search(rest, occupied | arrow[4], result + [arrow])
            if answer is not None:
                return answer
            if visits > node_limit:
                break
        return None

    found = search(list(range(len(targets))), 0, [])
    return [o[:4] for o in found] if found is not None else None


def _transform_points(points, turns, reflection):
    result = set()
    for r, c in points:
        if reflection:
            c = -c
        for _ in range(turns):
            r, c = -c, r
        result.add((r, c))
    return result


def _dot_pattern(rng, kind, axis, essential):
    """Sample a virtual structure, then recompute every center after clutter."""
    if kind == "single":
        dots = _plus((0, 0))
    elif kind in ("adjacent", "shared"):
        dots = _plus((0, 0)) | _plus((0, 1 if kind == "adjacent" else 2))
    elif kind == "double":
        # Unequal row/column offsets, straight rows and diagonal arrangements.
        offset = (int(rng.integers(4, 8)), int(rng.integers(-6, 7)))
        dots = _plus((0, 0)) | _plus(offset)
    elif kind in ("chain", "corner", "lattice"):
        centers = ([(0, 0), (0, 2), (0, 4)] if kind == "chain" else
                   [(0, 0), (0, 2), (2, 0)] if kind == "corner" else
                   [(0, 0), (0, 2), (2, 0), (2, 2)])
        dots = set().union(*(_plus(p) for p in centers))
    elif kind == "block":
        dots = {(r, c) for r in range(3) for c in range(int(rng.integers(4, 6)))}
    elif kind == "near":
        mode = int(rng.integers(4))
        dots = ({(0, 0), (0, -1), (0, 1), (1, 0)} if mode == 0 else
                {(0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)} if mode == 1 else
                {(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c} if mode == 2 else
                {(0, c) for c in range(int(rng.integers(2, 7)))})
    else:
        raise ValueError("unknown virtual layout")
    # In the teaching single, extend one arm with a nonparticipating endpoint.
    # Other examples may be clean, extended, diagonal, or a broken nearby motif.
    if essential == "single":
        dots.add((0, 2))
    elif kind != "near" and rng.random() < .60:
        dots.add((0, -2))
        if rng.random() < .40:
            dots.add((0, -3))
    if essential != "single" and rng.random() < .25:
        dots.add((2, 2))
    dots = _transform_points(dots, axis, bool(rng.integers(2)))
    _, selected = _scan_points(dots)
    targets = sorted(dots)
    inactive = sorted(dots - selected)
    if essential == "single":
        # Both collision outcomes are visible in this one demonstration.
        targets += [sorted(selected)[int(rng.integers(len(selected)))], inactive[0]]
    elif essential != "overlap":
        if selected and rng.random() < .50:
            p = sorted(selected)[int(rng.integers(len(selected)))]
            targets += [p] * (2 if rng.random() < .12 else 1)
        if inactive and rng.random() < .40:
            targets.append(inactive[int(rng.integers(len(inactive)))])
    return targets


def _key(a):
    return a.shape, np.asarray(a, dtype=np.uint8).tobytes()


def _render(rng, arrows, shape, palette):
    bg = int(rng.choice(INPUT_COLORS))
    available = [c for c in palette if c != bg]
    if not available:
        available = [c for c in INPUT_COLORS if c != bg]
    a = np.full(shape, bg, dtype=np.int8)
    base = int(rng.choice(available))
    mono = rng.random() < .30
    for arrow in arrows:
        color = base if mono else int(rng.choice(available))
        for p in arrow[2]:
            a[p] = color
    return {"input": a, "output": solve(a)}


def _make_pair(rng, kind, axis, palette, seen, essential=None):
    for attempt in range(48):
        targets = _dot_pattern(rng, kind, axis, essential)
        lows = tuple(min(p[j] for p in targets) for j in (0, 1))
        highs = tuple(max(p[j] for p in targets) for j in (0, 1))
        span = tuple(highs[j] - lows[j] + 1 for j in (0, 1))
        # Shape and translation vary independently of endpoint topology.
        # Initial compact attempts; later attempts are more spacious.
        if attempt < 24:
            shape = tuple(int(rng.integers(max(15, s + 10), 31)) for s in span)
            if rng.random() < .25:
                shape = (max(shape), max(shape))
            offset = tuple(int(rng.integers(4 - lows[j], shape[j] - highs[j] - 3))
                           for j in (0, 1))
        else:
            shape = (30, 30)
            offset = tuple((30 - span[j]) // 2 - lows[j] for j in (0, 1))
        shifted = [(r + offset[0], c + offset[1]) for r, c in targets]
        arrows = _assign_arrows(rng, shifted, shape)
        if arrows is None:
            continue
        # Crop redundant margins, sometimes reaching the actual canvas edge;
        # all arrow cells AND projected dots participate in the crop bounds.
        if rng.random() < .65:
            points = [p for o in arrows for p in o[2]] + shifted
            low = [min(p[j] for p in points) for j in (0, 1)]
            high = [max(p[j] for p in points) for j in (0, 1)]
            pads = rng.integers(0, 4, size=4)
            start = [max(0, low[j] - int(pads[j])) for j in (0, 1)]
            end = [min(shape[j], high[j] + 1 + int(pads[j + 2])) for j in (0, 1)]
            shape = tuple(end[j] - start[j] for j in (0, 1))
            arrows = [((o[0][0] - start[0], o[0][1] - start[1]), o[1],
                       frozenset((r - start[0], c - start[1]) for r, c in o[2]),
                       (o[3][0] - start[0], o[3][1] - start[1])) for o in arrows]
        pair = _render(rng, arrows, shape, palette)
        if _key(pair["input"]) in seen:
            continue
        f = _facts(pair["input"])
        if essential == "single" and not (f["active_collision"] and f["inactive_collision"]
                                           and f["attached_inactive"]):
            continue
        if essential == "overlap" and not (f["sequential_witness"] and f["first_stop_witness"]):
            continue
        return pair
    # Exhaustion is explicit: no silent switch to a different logical scene.
    raise RuntimeError(f"bounded geometric packing exhausted for {kind}")


def _check_counts(num_train, num_test):
    for name, count, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer))
                or not lo <= int(count) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    # One convention throughout; a geometric shift is chosen once per puzzle.
    train_axis = int(rng.integers(4))
    overlap_kind = str(rng.choice(("shared", "adjacent")))
    query_kind = str(rng.choice(("double", "adjacent" if overlap_kind == "shared" else "shared",
                                 "chain", "corner", "lattice")))
    palette = [int(c) for c in rng.choice(INPUT_COLORS, size=int(rng.integers(3, 9)), replace=False)]
    train_specs = [("single", "single"), (overlap_kind, "overlap")]
    train_specs += [(str(rng.choice(("single", overlap_kind, "near", "block"))), None)
                    for _ in range(int(num_train) - 2)]
    # Do not accidentally teach query topology through an optional block.
    if query_kind == "adjacent":
        train_specs = [(overlap_kind if k == "block" else k, e) for k, e in train_specs]
    puzzle, seen = {"train": [], "test": []}, set()
    for kind, essential in train_specs:
        pair = _make_pair(rng, kind, train_axis, palette, seen, essential)
        seen.add(_key(pair["input"]))
        puzzle["train"].append(pair)
    for _ in range(int(num_test)):
        pair = _make_pair(rng, query_kind, (train_axis + 1) % 4, palette, seen)
        seen.add(_key(pair["input"]))
        puzzle["test"].append(pair)
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(int(num_train))]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    witnesses = {"active_collision": False, "inactive_collision": False,
                 "attached_inactive": False, "sequential_witness": False,
                 "first_stop_witness": False}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            f = _facts(a)
            # Generator avoids physical impact/target ambiguity. The solver's
            # source-acknowledgement semantics also support such arrangements.
            assert not f["endpoint_on_arrow"], "generated endpoints must be background"
            expected = a.copy()
            mask = np.zeros(a.shape, dtype=bool)
            for _, _, cells, endpoint in f["arrows"]:
                if endpoint in f["selected"]:
                    color = SINGLE if f["counts"][endpoint] == 1 else COLLISION
                    for p in cells:
                        expected[p] = color
                        mask[p] = True
            assert np.array_equal(out, expected), "incorrect acknowledgement labels"
            assert np.array_equal(a != out, mask), "only complete accepted arrowheads may change"
            assert np.array_equal(out[a == f["background"]], a[a == f["background"]])
            assert np.array_equal(solve(a), expected)
            if split == "train":
                for name in witnesses:
                    witnesses[name] |= f[name]
    assert all(witnesses.values()), "training lacks a required disambiguating witness"


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(split.encode())
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streaming seeds, bounded geometry regressions and mutation rejection.

    Returns descriptive measurements; writes no artifacts and saves no traces.
    """
    fingerprints, shapes, structures, arrow_counts, center_counts = set(), set(), set(), set(), set()
    directions, backgrounds, palettes = set(), set(), set()
    measures = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                f = _facts(a)
                shapes.add(a.shape)
                roles = {}
                canonical = bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)
                structures.add((a.shape, hashlib.sha256(canonical).hexdigest()))
                arrow_counts.add(len(f["arrows"]))
                center_counts.add(len(f["centers"]))
                directions.update(o[1] for o in f["arrows"])
                backgrounds.add(f["background"])
                palettes.add(tuple(sorted(set(map(int, a.flat)))))
                measures["pairs"] += 1
                for name in ("active_collision", "inactive_collision", "overlap", "attached_inactive",
                             "sequential_witness", "first_stop_witness"):
                    measures[name] += bool(f[name])
                measures["no_plus"] += not f["centers"]
                measures["clean"] += bool(f["selected"]) and not f["inactive"]
                measures["multiple_colors"] += len({int(a[next(iter(o[2]))]) for o in f["arrows"]}) > 1
                measures["triple_collision"] += max(f["counts"].values()) >= 3
                measures["edge_arrow"] += any(r in (0, a.shape[0]-1) or c in (0, a.shape[1]-1)
                                              for o in f["arrows"] for r, c in o[2])
                # Omitting A would accept every arrow; omitting B's projection
                # and applying A directly to these separated triominoes changes none.
                if split == "train":
                    measures["no_A_counterexample"] += bool(f["inactive"])
                    measures["no_B_counterexample"] += bool(f["selected"])
    assert len(fingerprints) == 42 and len(structures) > 150
    assert len(shapes) >= 40 and len(arrow_counts) >= 7 and len(center_counts) >= 5
    assert directions == set(DIRECTIONS) and len(backgrounds) == 8 and len(palettes) >= 30
    for name in ("no_plus", "clean", "triple_collision", "edge_arrow", "multiple_colors"):
        assert measures[name] > 0, name
    assert _fingerprint(generate_puzzle(17)) == _fingerprint(generate_puzzle(17))
    # Independent set reasoning for all 512 virtual binary 3x3 neighborhoods.
    for bits in range(512):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        dots = set(map(tuple, np.argwhere(mask).tolist()))
        cs, us = _scan_points(dots)
        centers, union = _mask_union(mask)
        assert set(map(tuple, np.argwhere(centers).tolist())) == cs
        assert set(map(tuple, np.argwhere(union).tolist())) == us
    assert len(_scan_points(_plus((0, 0)) | _plus((0, 1)))[1]) == 8
    assert len(_scan_points(_plus((0, 0)) | _plus((0, 2)))[1]) == 9
    assert not _scan_points({(0, 0), (0, 1), (1, 0), (0, 2)})[1]
    # Rotation, reflection, translation and palette changes preserve the rule.
    p = generate_puzzle(9182, 2, 1)
    for turns in range(4):
        for reflection in (False, True):
            transformed = {s: [{k: np.rot90(np.fliplr(v) if reflection else v, turns).copy()
                               for k, v in pair.items()} for pair in p[s]] for s in p}
            validate_generated_puzzle(transformed)
    # Wrong labels and partial-arrow labels must fail the validator.
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    changed = np.argwhere(damaged["test"][0]["input"] != damaged["test"][0]["output"])[0]
    point = tuple(changed)
    damaged["test"][0]["output"][point] = damaged["test"][0]["input"][point]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("partial acknowledgement was accepted")
    for a in range(2, 11):
        for b in range(1, min(4, a-1) + 1):
            generate_puzzle(83, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4, False),
                 (4.0, 1), (4, 1.0), (None, 1), (4, None), ("4", 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "shapes": len(shapes),
            "structures": len(structures), "arrow_counts": sorted(arrow_counts),
            "center_counts": sorted(center_counts), "measurements": dict(measures)}
