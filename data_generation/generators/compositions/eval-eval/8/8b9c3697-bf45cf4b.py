"""Docking looms: center-anchored, probe-gated exact tile substitution.

Standard library + NumPy only. The solver reads only its input. Construction
paints local scenes independently of the parser and solver.
"""

from __future__ import annotations

import numpy as np


PROBE = 9
_OUTWARD = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return grid.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    point = r + dr, c + dc
                    if point in remaining:
                        remaining.remove(point)
                        cells.add(point)
                        stack.append(point)
        result.append(cells)
    return result


def _bbox(cells):
    rows, cols = zip(*cells)
    return min(rows), min(cols), max(rows) + 1, max(cols) + 1


def _rect(box):
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)}


def _parse_fork(cells):
    candidates = []
    for out in _OUTWARD:
        perp = (out[1], -out[0])
        kmin = min(r * out[0] + c * out[1] for r, c in cells)
        qmin = min(r * perp[0] + c * perp[1] for r, c in cells)
        local = {(r * out[0] + c * out[1] - kmin,
                  r * perp[0] + c * perp[1] - qmin) for r, c in cells}
        width = max(q for _, q in local) + 1
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        lengths = [sum((k, q) in local for k in range(1, 30)) for q in range(width)]
        expected = {(0, q) for q in range(width)}
        expected |= {(k, q) for q, length in enumerate(lengths)
                     for k in range(1, length + 1)}
        if expected != local:
            continue
        runs, q = [], 0
        while q < width:
            if lengths[q] == 0:
                q += 1
                continue
            qa, length = q, lengths[q]
            while q < width and lengths[q] == length:
                q += 1
            runs.append((qa, q, length))
        if len(runs) != 3 or any(runs[j][1] >= runs[j + 1][0] for j in (0, 1)):
            continue
        if runs[1][0] + runs[1][1] != width:
            continue
        candidates.append(dict(cells=cells, out=out, perp=perp, kmin=kmin,
                               qmin=qmin, width=width, runs=tuple(runs),
                               lengths=tuple(run[2] for run in runs),
                               q0=runs[1][0], span=runs[1][1] - runs[1][0],
                               stop=runs[1][2] + 1))
    if len(candidates) != 1:
        raise ValueError("a fork must have one base and three positive spokes")
    return candidates[0]


def _local(fork, point):
    r, c = point
    return (r * fork["out"][0] + c * fork["out"][1] - fork["kmin"],
            r * fork["perp"][0] + c * fork["perp"][1] - fork["qmin"])


def _global(fork, k, q):
    k, q = k + fork["kmin"], q + fork["qmin"]
    return (k * fork["out"][0] + q * fork["perp"][0],
            k * fork["out"][1] + q * fork["perp"][1])


def _translated_box(box, dr, dc):
    return box[0] + dr, box[1] + dc, box[2] + dr, box[3] + dc


def _candidate(fork, tile, grid, background, blockers):
    footprint = _rect(tile["box"])
    local = [_local(fork, p) for p in footprint]
    qs = {q for _, q in local}
    if qs != set(range(fork["q0"], fork["q0"] + fork["span"])):
        return None
    a, end = min(k for k, _ in local), max(k for k, _ in local)
    s = fork["stop"]
    if a <= s:
        return None
    route = {_global(fork, k, q) for k in range(s, end + 1) for q in qs}
    for r, c in route:
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            return None
        if (r, c) not in footprint and int(grid[r, c]) not in (background, PROBE):
            return None
        if (r, c) not in footprint and (r, c) in blockers:
            return None
    distance, length = a - s, end - a + 1
    boxes = [_translated_box(tile["box"], (j * length - distance) * fork["out"][0],
                             (j * length - distance) * fork["out"][1])
             for j in range(distance // length + 1)]
    return dict(route=route, distance=distance, length=length, boxes=boxes)


def _analyze(input_grid):
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    if int(counts.max()) <= grid.size / 2 or background == PROBE:
        raise ValueError("input must have a majority non-probe background")
    forks, tiles = [], []
    for cells in _components((grid != background) & (grid != PROBE)):
        colors = {int(grid[p]) for p in cells}
        if len(colors) == 1:
            fork = _parse_fork(cells)
            fork["color"] = next(iter(colors))
            forks.append(fork)
        else:
            box = _bbox(cells)
            tile = grid[box[0]:box[2], box[1]:box[3]].copy()
            if PROBE in tile:
                raise ValueError("probes cannot be inside a source tile")
            if set(map(tuple, np.argwhere(
                    (grid[box[0]:box[2], box[1]:box[3]] != background)))) != {
                    (r - box[0], c - box[1]) for r, c in cells}:
                raise ValueError("a source tile cannot enclose another object")
            tiles.append(dict(cells=cells, colors=colors, box=box, stamp=tile))
    if not forks or not tiles:
        raise ValueError("need a fork and a patterned tile")
    fork_colors = {f["color"] for f in forks}
    payload = set().union(*(t["colors"] for t in tiles))
    if not fork_colors.isdisjoint(payload):
        raise ValueError("fork and payload roles must be disjoint within an input")
    blockers = set().union(*(f["cells"] for f in forks),
                           *(_rect(t["box"]) for t in tiles))
    matches, used, occupied = [], set(), set()
    for fi, fork in enumerate(forks):
        candidates = []
        for ti, tile in enumerate(tiles):
            candidate = _candidate(fork, tile, grid, background, blockers)
            if candidate is not None:
                candidates.append((ti, candidate))
        if len(candidates) > 1:
            raise ValueError("multiple matching tiles for a fork")
        if candidates:
            ti, match = candidates[0]
            if ti in used or not occupied.isdisjoint(match["route"]):
                raise ValueError("assignments or successful sweeps interact")
            used.add(ti)
            occupied.update(match["route"])
            match.update(fi=fi, ti=ti)
            hits, probe_slots, blocked_slots = {0}, set(), set()
            for j, box in enumerate(match["boxes"]):
                if j == 0:
                    continue
                probes = grid[box[0]:box[2], box[1]:box[3]] == PROBE
                if probes.any():
                    probe_slots.add(j)
                    if np.any(probes & (tiles[ti]["stamp"] != background)):
                        hits.add(j)
                    else:
                        blocked_slots.add(j)
            match.update(hits=hits, probe_slots=probe_slots, blocked_slots=blocked_slots)
            matches.append(match)
    return dict(grid=grid, background=background, forks=forks, tiles=tiles,
                matches=matches, unmatched=set(range(len(tiles))) - used)


def _write_stamp(output, box, stamp):
    output[box[0]:box[2], box[1]:box[3]] = stamp


def _render(info, *, contact=True, solid=False, original_phase=False,
            retain_unmatched=False):
    """Also supplies explicit, documented ablations for witness checks."""
    output, background = info["grid"].copy(), info["background"]
    output[output == PROBE] = background
    for ti, tile in enumerate(info["tiles"]):
        if retain_unmatched and ti in info["unmatched"]:
            continue
        r0, c0, r1, c1 = tile["box"]
        output[r0:r1, c0:c1] = background
    for match in info["matches"]:
        tile = info["tiles"][match["ti"]]
        fork = info["forks"][match["fi"]]
        stamp = tile["stamp"]
        if solid:
            stamp = np.full(stamp.shape, min(tile["colors"]), dtype=np.int8)
        _write_stamp(output, match["boxes"][0], stamp)
        if original_phase:
            # Anchor later blocks to the original leading face, but retain the
            # mandatory correct docking so this specifically tests registration.
            remainder = match["distance"] % match["length"]
            boxes = [_translated_box(b, remainder * fork["out"][0],
                                      remainder * fork["out"][1])
                     for b in match["boxes"][1:]]
            for box in boxes:
                if not _rect(box) <= match["route"]:
                    continue
                dots = info["grid"][box[0]:box[2], box[1]:box[3]] == PROBE
                if np.any(dots & (tile["stamp"] != background)):
                    _write_stamp(output, box, stamp)
        else:
            active = match["hits"] if contact else match["probe_slots"] | {0}
            for j in sorted(active - {0}):
                _write_stamp(output, match["boxes"][j], stamp)
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, match, contact, and output cell from this input only."""
    return _render(_analyze(input_grid))


def _asymmetric(stamp):
    return all(not np.array_equal(stamp, transform)
               for k in range(4) for transform in
               ((np.rot90(stamp, k), np.fliplr(np.rot90(stamp, k)))
                if k else (np.fliplr(stamp),)))


def _make_stamp(rng, length, span, background, payload, asymmetric):
    """No compulsory palette row; connected support with visible crop sides."""
    rr, cc = np.indices((length, span))
    for _ in range(80):
        kind = int(rng.integers(5))
        n = len(payload)
        if kind == 0:
            labels = rr + cc
        elif kind == 1:
            labels = rr if rng.integers(2) else cc
        elif kind == 2:
            labels = np.minimum.reduce((rr, cc, length - 1 - rr, span - 1 - cc))
        elif kind == 3:
            labels = abs(2 * rr - length + 1) + abs(2 * cc - span + 1)
        else:
            labels = rng.integers(n, size=(length, span))
        stamp = np.asarray(payload, dtype=np.int8)[(labels + int(rng.integers(n))) % n]
        stamp = stamp.copy()
        # At least one hole is necessary for contact-vs-dot witnesses.
        count = int(rng.integers(1, max(2, stamp.size // 3 + 1)))
        holes = rng.choice(stamp.size, size=count, replace=False)
        stamp.flat[holes] = background
        fg = stamp != background
        if not all(np.any(edge) for edge in (fg[0], fg[-1], fg[:, 0], fg[:, -1])):
            continue
        if len(_components(fg)) != 1 or len(set(map(int, stamp[fg]))) < 2:
            continue
        if asymmetric and not _asymmetric(stamp):
            continue
        return stamp
    # A 2x2-or-larger, connected asymmetric motif, with a retained corner hole.
    stamp = np.full((length, span), payload[0], dtype=np.int8)
    stamp[0, 0] = background
    stamp[-1, 0] = payload[1]
    # The unique hole fixes a corner; different bottom-left/top-right pixels
    # break the only remaining corner-preserving square reflection.
    assert not asymmetric or _asymmetric(stamp)
    return stamp


def _module(rng, *, train, index, ordinal, count, rotation, background,
            fork_color, payload, active, mixed, fallback):
    # A local loom opens downward. Rotating the entire scene keeps displayed
    # motifs and probes registered; generation does not call the solver.
    length = 2 if mixed or fallback or (train and index == 0) else (
        3 if train and index == 1 else int(rng.integers(2, 4)))
    span = 2 if fallback else int(rng.integers(2, 5))
    gap = 1 if fallback or count == 3 else int(rng.integers(1, 3))
    width, q0 = span + 2 * (1 + gap), 1 + gap
    if train and index == 0 and ordinal == 0:
        center, left, right = 1, 3, 3
    elif train and index == 0 and ordinal == 1:
        center, left, right = 3, 1, 2
    elif train and index == 1 and ordinal == 0:
        center, left, right = 3, 1, 2
    elif train and index == 1:
        center = int(rng.integers(2, 4))
        left = int(rng.integers(1, 5))
        right = left % 4 + 1
    else:
        cap = 3 if mixed or fallback else 5
        center, left, right = map(int, rng.integers(1, cap + 1, size=3))
    n = (4 if index == 1 else int(rng.integers(3, 5))) if train else (
        5 if mixed or fallback else int(rng.integers(5, 8)))
    remainder = 1 if train and index == 1 else int(rng.integers(length))
    stop, distance = center + 1, n * length + remainder
    start, depth = stop + distance, stop + distance + length
    if depth > 30:
        # Fit bounds shorten the quotient without changing semantic behavior.
        n = (30 - stop - length - remainder) // length
        distance, start = n * length + remainder, stop + n * length + remainder
        depth = start + length
    assert n >= (3 if train else 5)
    inp = np.full((max(depth, left + 1, right + 1), width), background, dtype=np.int8)
    out = inp.copy()
    for g in (inp, out):
        g[0, :] = fork_color
        g[1:left + 1, 0] = fork_color
        g[1:right + 1, -1] = fork_color
        g[1:center + 1, q0:q0 + span] = fork_color
    stamp = _make_stamp(rng, length, span, background, payload,
                        asymmetric=train and index < 2)
    if not active:
        mode = "wrong_span" if train and index == 0 else str(rng.choice(
            ["wrong_span", "off_axis", "missing"]))
        if mode == "wrong_span":
            wrong = _make_stamp(rng, length, span + 1, background, payload, False)
            inp[start:start + length, q0:q0 + span + 1] = wrong
        elif mode == "off_axis":
            inp[start:start + length, q0 + 1:q0 + 1 + span] = stamp
        # Inactive probes are harmless; the semantic rule erases all of them.
        inp[stop + length, q0] = PROBE
        return np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()

    inp[start:start + length, q0:q0 + span] = stamp
    out[stop:stop + length, q0:q0 + span] = stamp
    ink = list(map(tuple, np.argwhere(stamp != background)))
    holes = list(map(tuple, np.argwhere(stamp == background)))
    slots = list(map(int, rng.permutation(np.arange(1, n))))
    positive, negative = slots[:2]
    blank = slots[2] if n >= 4 else None
    for j in range(1, n):
        if j == blank:
            continue
        selected = []
        if j == positive:
            selected.append(ink[int(rng.integers(len(ink)))])
            if rng.random() < .45:
                selected.append(holes[int(rng.integers(len(holes)))])
        elif j == negative:
            selected.append(holes[int(rng.integers(len(holes)))])
        elif rng.random() < .7:
            pool = ink if rng.random() < .6 else holes
            selected.append(pool[int(rng.integers(len(pool)))])
        for rr, cc in selected:
            inp[stop + j * length + rr, q0 + cc] = PROBE
        if any(stamp[p] != background for p in selected):
            out[stop + j * length:stop + (j + 1) * length, q0:q0 + span] = stamp
    # A contact in the exposed portion of the final full block can print over
    # part of the original footprint after erasure. Training 1 shows this.
    if remainder and (train and index == 1 or rng.random() < .6):
        options = [(r, c) for r, c in ink if r < remainder]
        if options:
            rr, cc = options[int(rng.integers(len(options)))]
            inp[stop + n * length + rr, q0 + cc] = PROBE
            out[stop + n * length:stop + (n + 1) * length, q0:q0 + span] = stamp
    return np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()


def _pack(rng, modules, background):
    count = len(modules)
    options = []
    for axis in (0, 1):
        gap = int(rng.integers(1, 4))
        total = sum(m[0].shape[axis] for m in modules) + gap * (count - 1)
        other = max(m[0].shape[1 - axis] for m in modules)
        if total <= 30 and other <= 30:
            options.append((axis, gap, total, other))
    if not options:
        raise ValueError("local loom boxes do not fit")
    axis, gap, total, other = options[int(rng.integers(len(options)))]
    shape = [0, 0]
    shape[axis] = total + int(rng.integers(min(30 - total, 5) + 1))
    shape[1 - axis] = other + int(rng.integers(min(30 - other, 6) + 1))
    inp = np.full(shape, background, dtype=np.int8)
    out = inp.copy()
    cursor = int(rng.integers(shape[axis] - total + 1))
    for i in rng.permutation(count):
        mi, mo = modules[int(i)]
        origin = [0, 0]
        origin[axis] = cursor
        origin[1 - axis] = int(rng.integers(shape[1 - axis] - mi.shape[1 - axis] + 1))
        r, c = origin
        inp[r:r + mi.shape[0], c:c + mi.shape[1]] = mi
        out[r:r + mo.shape[0], c:c + mo.shape[1]] = mo
        cursor += mi.shape[axis] + gap
    if np.count_nonzero(inp == background) <= inp.size / 2:
        raise ValueError("packing must retain a majority background")
    return dict(input=inp, output=out)


def _generate_pair(rng, train, index, axis, palette, stable):
    for attempt in range(64):
        fallback = attempt >= 48
        colors = palette if stable else list(map(int, rng.permutation(9)))
        background = colors[0]
        count = 2 if index < 2 and train else int(rng.integers(1 if train else 2, 4))
        mixed = count == 2 and rng.random() < .30 and not fallback
        base_rotation = (axis if train and index < 2 else int(rng.integers(2)))
        rotations = [base_rotation + 2 * int(rng.integers(2)) for _ in range(count)]
        if mixed:
            rotations[1] = (rotations[0] + int(rng.choice([1, 3]))) % 4
        if train and index == 0:
            active = [True, False]
        elif train and index == 1:
            active = [True, True]
        else:
            active = [True] + [bool(rng.random() < .8) for _ in range(count - 1)]
        nf = int(rng.integers(1, count + 1))
        fork_colors = colors[1:1 + nf]
        payload = colors[1 + nf:1 + nf + int(rng.integers(2, min(5, 8 - nf) + 1))]
        modules = []
        for ordinal in range(count):
            modules.append(_module(rng, train=train, index=index, ordinal=ordinal,
                                   count=count, rotation=rotations[ordinal],
                                   background=background,
                                   fork_color=fork_colors[ordinal % nf],
                                   payload=payload, active=active[ordinal],
                                   mixed=mixed, fallback=fallback))
        try:
            pair = _pack(rng, modules, background)
            info = _analyze(pair["input"])
            if len(info["matches"]) != sum(active):
                continue
            if not np.array_equal(_render(info), pair["output"]):
                continue
            # This witness must distinguish lattice origins, not just have a
            # nonzero remainder whose motif happens to mask the difference.
            if train and index == 1 and np.array_equal(
                    _render(info, original_phase=True), pair["output"]):
                continue
            return pair
        except ValueError:
            continue
    raise RuntimeError("bounded packing attempts exhausted")


def _valid_integer(value, low, high):
    return (isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_)) and low <= int(value) <= high)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    if not _valid_integer(num_train, 2, 10) or not _valid_integer(num_test, 1, 4):
        raise ValueError("counts must be integers: 2..10 train, 1..4 test")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(9)))
    stable = bool(rng.random() < .25)
    axis = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            for _ in range(24):
                pair = _generate_pair(rng, split == "train", index,
                                      (axis + index) % 2, palette, stable)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not produce a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, information = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            assert grid.shape == output.shape and _key(grid) not in seen
            seen.add(_key(grid))
            info = _analyze(grid)
            information[split].append(info)
            assert info["matches"], "need a successful loom"
            assert np.array_equal(_render(info), output), "wrong registered contact output"
            assert PROBE not in output
            for fork in info["forks"]:
                assert min(fork["lengths"]) >= 1
                assert all(output[p] == fork["color"] for p in fork["cells"])
            for ti in info["unmatched"]:
                box = info["tiles"][ti]["box"]
                assert np.all(output[box[0]:box[2], box[1]:box[3]] == info["background"])
            for match in info["matches"]:
                stamp = info["tiles"][match["ti"]]["stamp"]
                assert np.any(stamp == info["background"])
                assert match["blocked_slots"] and len(match["hits"]) >= 2
                for j, box in enumerate(match["boxes"]):
                    actual = output[box[0]:box[2], box[1]:box[3]]
                    if j in match["hits"]:
                        assert np.array_equal(actual, stamp)
                    else:
                        assert np.all(actual == info["background"])
                # Exactly the structures and activated complete blocks survive.
                allowed = set().union(*(f["cells"] for f in info["forks"]))
                for m in info["matches"]:
                    for j in m["hits"]:
                        allowed |= _rect(m["boxes"][j])
                assert all(output[p] == info["background"] for p in match["route"] - allowed)
                pitches = match["distance"] // match["length"]
                assert 3 <= pitches <= 4 if split == "train" else 5 <= pitches <= 7
    train = information["train"]
    assert any(i["unmatched"] for i in train), "show erased wrong-span tiles"
    axes = {abs(i["forks"][m["fi"]]["out"][0]) for i in train for m in i["matches"]}
    assert axes == {0, 1}, "show both movement axes"
    assert any(max(i["forks"][m["fi"]]["lengths"][0],
                   i["forks"][m["fi"]]["lengths"][2])
               < i["forks"][m["fi"]]["lengths"][1]
               for i in train for m in i["matches"]), "show an active long center"
    assert any(f["lengths"][0] != f["lengths"][2] for i in train for f in i["forks"])
    assert any(m["distance"] % m["length"] for i in train for m in i["matches"])
    assert any(len(m["boxes"]) - 1 > len(m["probe_slots"]) for i in train for m in i["matches"])
    assert any(_asymmetric(i["tiles"][m["ti"]]["stamp"]) for i in train for m in i["matches"])
    for ablation in (dict(contact=False), dict(solid=True), dict(original_phase=True),
                     dict(retain_unmatched=True)):
        assert any(not np.array_equal(_render(i, **ablation), _render(i)) for i in train)
    return None


def _puzzle_key(puzzle):
    return tuple((_key(_grid(p["input"])), _key(_grid(p["output"])))
                 for s in ("train", "test") for p in puzzle[s])


def _normalized(array, background):
    mapping, result = {int(background): 0}, []
    for v in array.ravel():
        v = int(v)
        result.append(mapping.setdefault(v, len(mapping)))
    return array.shape, tuple(result)


def _semantic_regression():
    """Hand-built larger member: thick guards, OR, a hole, and residual travel.

    These are new-family fixtures, not copied parent pairs. Expected labels are
    explicit block writes without the input parser or contact renderer.
    """
    inp = np.zeros((28, 13), dtype=np.int8)
    inp[0, :] = 4
    inp[1:8, 0:2] = 4
    inp[1:5, 4:9] = 4
    inp[1:4, 11:13] = 4
    expected = inp.copy()
    rr, cc = np.indices((5, 5))
    stamp = ((rr + 2 * cc) % 3 + 1).astype(np.int8)
    stamp[2, 2] = 0
    inp[23:28, 4:9] = stamp
    # OR in block 1; a hole-only block 2; block 3 crosses the old footprint.
    for p in ((10, 4), (12, 6), (17, 6), (20, 4), (6, 7), (22, 2)):
        inp[p] = PROBE
    for start in (5, 10, 20):
        expected[start:start + 5, 4:9] = stamp
    for k in range(4):
        for reflect in (False, True):
            source, target = np.rot90(inp, k), np.rot90(expected, k)
            if reflect:
                source, target = np.fliplr(source), np.fliplr(target)
            assert np.array_equal(solve(source), target)
    return None


def self_test():
    """Bounded 42-seed semantic, count, and structural variation audit."""
    shapes, geometries, motifs, patterns, directions = set(), set(), set(), set(), set()
    counts, backgrounds, fingerprints, remainders = set(), set(), set(), set()
    for seed in range(42):
        # Exercise varied counts while keeping the public default at (4, 1).
        nt = 1 + seed % 4
        nr = nt + 1 + (seed // 4) % (10 - nt)
        puzzle = generate_puzzle(seed, nr, nt)
        fingerprints.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                shapes.add(info["grid"].shape)
                counts.add((len(info["forks"]), len(info["matches"])))
                backgrounds.add(info["background"])
                patterns.add(_normalized(info["grid"], info["background"]))
                for f in info["forks"]:
                    geometries.add((f["width"], f["runs"]))
                    directions.add(f["out"])
                for m in info["matches"]:
                    motifs.add(_normalized(info["tiles"][m["ti"]]["stamp"], info["background"]))
                    remainders.add(m["distance"] % m["length"])
                # The exact same rule must commute with spatial reflection.
                assert np.array_equal(solve(np.fliplr(pair["input"])), np.fliplr(pair["output"]))
    assert len(fingerprints) == 42
    assert len(shapes) >= 30 and len(geometries) >= 60 and len(motifs) >= 100
    assert len(patterns) >= 200 and len(backgrounds) >= 7
    assert directions == set(_OUTWARD) and {0, 1, 2} <= remainders
    assert {n for n, _ in counts} == {1, 2, 3}
    _semantic_regression()
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + a * 10 + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    for seed in (-1, True, 1.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    assert _puzzle_key(generate_puzzle(194, 3, 2)) == _puzzle_key(generate_puzzle(194, 3, 2))
    damaged = generate_puzzle(195, 2, 1)
    info = _analyze(damaged["test"][0]["input"])
    box = info["matches"][0]["boxes"][0]
    damaged["test"][0]["output"][box[0], box[1]] = PROBE
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    malformed = {(0, q) for q in range(7)} | {(k, q) for k in (1, 2) for q in (0, 6)}
    try:
        _parse_fork(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("zero center accepted")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(shapes), "fork_geometries": len(geometries),
            "normalized_motifs": len(motifs), "normalized_inputs": len(patterns),
            "backgrounds": len(backgrounds), "directions": len(directions),
            "receiver_active_counts": sorted(counts),
            "travel_remainders": sorted(map(int, remainders))}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
