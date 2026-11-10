"""A continuously clocked scanning ruler steered by unary-coded detours.

Only the standard library and NumPy are used. No evidence files, random state,
training outputs or sampling metadata are consulted by solve(). No disk writes.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np


_TEACH = ("odd_overwrite", "ordinary_overwrite", "remote_highlight",
          "remote_preserved", "sliced_component", "rotating_scan",
          "depth_clock_differs", "reset_clock_differs", "unary_differs",
          "downstream_phase")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _walk(vertices):
    """List unit-step positions; each shared corner occurs exactly once."""
    points = [vertices[0]]
    for (r, c), (rr, cc) in zip(vertices, vertices[1:]):
        if (r, c) == (rr, cc):
            # A deliberately wrong fixed-clearance counterfactual may need no
            # lateral move. Actual decoded routes always have nonzero detours.
            continue
        assert (r == rr) != (c == cc), "nonzero axis-aligned segment required"
        dr, dc = int(np.sign(rr - r)), int(np.sign(cc - c))
        for _ in range(abs(rr - r) + abs(cc - c)):
            r, c = r + dr, c + dc
            points.append((r, c))
    return points


def _route(height, initial, bars, fixed_clearance=False):
    vertices, current = [(0, initial)], initial
    for f, lo, hi, n, sign in bars:
        d = 2 if fixed_clearance else n + 1
        target = hi + d if sign == 1 else lo - d
        vertices.extend(((f - d, current), (f - d, target)))
        current = target
    vertices.append((height - 1, current))
    return _walk(vertices)


def _timing(points, mode="normal"):
    """Tangent is outgoing, except at the last cell. Clock never resets normally."""
    result, last_bend, previous = [], 0, None
    for s, p in enumerate(points):
        q = points[s + 1] if s + 1 < len(points) else points[s - 1]
        delta = (q[0] - p[0], q[1] - p[1])
        if s + 1 == len(points):
            delta = (-delta[0], -delta[1])
        if previous is not None and delta != previous:
            last_bend = s
        phase = p[0] if mode == "depth" else s - last_bend if mode == "reset" else s
        result.append((p[0], p[1], phase % 6, bool(delta[0])))
        previous = delta
    return result


def _render(a, info, points=None, mode="normal", rows_only=False):
    points = info["points"] if points is None else points
    bg, tick, high, bar, marker = info["roles"]
    out = a.copy()
    out[out == marker] = bar
    on_path = np.zeros(a.shape, dtype=bool)
    for r, c in points:
        on_path[r, c] = True
    objects = np.isin(a, info["object_colors"])
    scan = np.zeros(a.shape, dtype=bool)
    timed = _timing(points, mode)
    for r, c, phase, forward in timed:
        if phase == 4:
            if forward or rows_only:
                scan[r, :] = True
            else:
                scan[:, c] = True
    out[scan & objects & ~on_path] = high
    out[on_path] = bg
    for r, c, phase, _ in timed:
        if phase % 2 == 0:
            out[r, c] = high if phase == 4 else tick
    return out


def _decode_bars(a, roles):
    bg, tick, high, bar, marker = roles
    structure = (a == bar) | (a == marker)
    rows = np.flatnonzero(structure.any(axis=1))
    if not 1 <= len(rows) <= 6:
        raise ValueError("expected one to six bars")
    initial = int(np.flatnonzero(a[0] == tick)[0])
    bars, current, previous = [], initial, -1
    for raw_f in rows:
        f = int(raw_f)
        cs = np.flatnonzero(structure[f])
        lo, hi = int(cs[0]), int(cs[-1])
        markers = tuple(map(int, np.flatnonzero(a[f] == marker)))
        n = len(markers)
        if (not 0 < f < len(a) - 1 or len(cs) != hi - lo + 1
                or not 1 <= n <= 4 or hi - lo + 1 < 2 * n + 1):
            raise ValueError("invalid bar geometry")
        low = markers == tuple(lo + 2 * j for j in range(n))
        high_end = markers == tuple(hi - 2 * j for j in reversed(range(n)))
        if low == high_end:
            raise ValueError("endpoint code is not asymmetric")
        sign, d = (1 if low else -1), n + 1
        target = hi + d if sign == 1 else lo - d
        if (not 0 <= target < a.shape[1] or f - d < 1
                or f - d <= previous or not lo <= current <= hi):
            raise ValueError("infeasible detour")
        bars.append((f, lo, hi, n, sign))
        current, previous = target, f
    objects = tuple(int(c) for c in np.unique(a) if c not in roles)
    if not 1 <= len(objects) <= 3 or any(np.sum(a == c) < 5 for c in objects):
        raise ValueError("one to three visible foreground colors required")
    mask = np.isin(a, objects)
    halo = structure.copy()
    for dr, dc in itertools.product((-1, 0, 1), repeat=2):
        src_r = slice(max(0, -dr), min(len(a), len(a) - dr))
        src_c = slice(max(0, -dc), min(a.shape[1], a.shape[1] - dc))
        dst_r = slice(max(0, dr), min(len(a), len(a) + dr))
        dst_c = slice(max(0, dc), min(a.shape[1], a.shape[1] + dc))
        halo[dst_r, dst_c] |= structure[src_r, src_c]
    if np.any(mask & halo):
        raise ValueError("foreground touches a steering bar")
    for cells in _components(mask):
        rr, cc = np.array(cells).T
        if len(np.unique(a[rr, cc])) != 1:
            raise ValueError("data components must be monochromatic")
    points = _route(len(a), initial, bars)
    if len(points) != len(set(points)) or any(structure[r, c] for r, c in points):
        raise ValueError("route collides or revisits")
    return dict(roles=roles, object_colors=objects, bars=bars, initial=initial,
                points=points)


def _decode(input_grid):
    g = _grid(input_grid)
    values, counts = np.unique(g, return_counts=True)
    if not 6 <= len(values) <= 8 or counts.max() <= g.size / 2:
        raise ValueError("six to eight colors and majority background required")
    bg = int(values[counts.argmax()])
    sizes = dict(zip(map(int, values), map(int, counts)))
    heads = []
    for k in range(4):
        a = np.rot90(g, -k)
        occupied = np.flatnonzero(a[0] != bg)
        if len(occupied) != 3 or not np.all(np.diff(occupied) == 1):
            continue
        left, center, right = map(int, occupied)
        if left == 0 or right == a.shape[1] - 1:
            continue
        high, tick = int(a[0, left]), int(a[0, center])
        if (a[0, right] == high and high != tick and sizes.get(tick) == 1
                and sizes.get(high) == 2):
            heads.append((k, a, tick, high))
    if len(heads) != 1:
        raise ValueError("input needs a unique boundary head")
    k, a, tick, high = heads[0]
    remaining = [int(c) for c in values if c not in (bg, tick, high)]
    candidates = []
    for bar, marker in itertools.permutations(remaining, 2):
        try:
            info = _decode_bars(a, (bg, tick, high, bar, marker))
        except ValueError:
            continue
        candidates.append(info)
    if len(candidates) != 1:
        raise ValueError("input needs one unique marked-bar interpretation")
    info = candidates[0]
    info["rotation"] = k
    return a, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the head, count markers and transport the clock using input only."""
    a, info = _decode(input_grid)
    return np.rot90(_render(a, info), info["rotation"]).copy()


def _span_plan(rng, width, counts, repeated, forced_repeat, long_leg):
    """A finite depth-first search; failures memoized, candidate lists transient."""
    failed = set()
    repeat_index, repeat_sign = forced_repeat if forced_repeat is not None else (-2, 0)

    def search(i, current, previous):
        if i == len(counts):
            return []
        state = (i, current, previous)
        if state in failed:
            return None
        n, candidates = counts[i], []
        signs = (-previous,) if previous and not repeated else (-1, 1)
        if i in (repeat_index, repeat_index + 1):
            signs = (repeat_sign,)
        for lo in range(current + 1):
            for hi in range(max(current, lo + 2 * n), width):
                for sign in signs:
                    target = hi + n + 1 if sign == 1 else lo - n - 1
                    if 0 <= target < width and (not long_leg or i or abs(target - current) >= 7):
                        candidates.append((lo, hi, target, sign))
        for raw in rng.permutation(len(candidates)):
            lo, hi, target, sign = candidates[int(raw)]
            tail = search(i + 1, target, sign)
            if tail is not None:
                return [(lo, hi, target, sign)] + tail
        failed.add(state)
        return None

    for raw in rng.permutation(np.arange(2, width - 2)):
        initial = int(raw)
        spans = search(0, initial, 0)
        if spans is not None:
            return initial, spans
    return None


def _geometry(rng, counts, repeated, force_repeat, training, shift):
    minimum = sum(n + 1 for n in counts) + len(counts) + 2
    lower, upper = max(12, minimum), 30
    if shift == 2:
        lower, upper = (max(12, minimum), 24) if training else (max(26, minimum), 30)
    height = int(rng.integers(lower, upper + 1))
    width = height if rng.random() < .23 else int(rng.integers(12, 31))
    repeat = None
    if force_repeat:
        repeat = (int(rng.integers(len(counts) - 1)), int(rng.choice((-1, 1))))
    plan = None
    for span in dict.fromkeys((width, min(30, width + 5), 30)):
        plan = _span_plan(rng, span, counts, repeated, repeat, training)
        if plan is not None:
            width = span
            break
    assert plan is not None, "feasible width-30 plan not found"
    initial, spans = plan
    slack = height - minimum
    if rng.random() < .25:
        allocation = np.zeros(len(counts) + 1, dtype=int)
        allocation[int(rng.integers(len(allocation)))] = slack
    else:
        allocation = rng.multinomial(slack, np.full(len(counts) + 1, 1 / (len(counts) + 1)))
    previous, bars = -1, []
    for i, (n, (lo, hi, _, sign)) in enumerate(zip(counts, spans)):
        turn = (1 if i == 0 else previous + 1) + int(allocation[i])
        f = turn + n + 1
        bars.append((f, lo, hi, n, sign))
        previous = f
    return (height, width), initial, bars


def _object_mask(rng, allowed):
    """Draw unconstrained shapes, then keep only finished components of size >=5."""
    h, w = allowed.shape
    mask = np.zeros(allowed.shape, dtype=bool)
    target = int(rng.uniform(.10, .26) * h * w)
    starts = np.argwhere(allowed)
    for _ in range(36):
        if mask.sum() >= target:
            break
        r, c = map(int, starts[int(rng.integers(len(starts)))])
        kind = int(rng.integers(4))
        if kind < 2:
            hh, ww = int(rng.integers(2, 9)), int(rng.integers(2, 9))
            local = np.ones((hh, ww), dtype=bool)
            if kind == 1 and hh >= 4 and ww >= 4:
                local[1:hh - 1, 1:ww - 1] = False
                if rng.random() < .5:
                    local[hh - 1, 1:ww - 1] = False
            rr, cc = min(h, r + hh), min(w, c + ww)
            mask[r:rr, c:cc] |= local[:rr - r, :cc - c] & allowed[r:rr, c:cc]
        else:
            active, seen = [(r, c)], {(r, c)}
            for _ in range(int(rng.integers(8, 60))):
                if not active:
                    break
                j = len(active) - 1 if kind == 3 else int(rng.integers(len(active)))
                rr, cc = active.pop(j)
                mask[rr, cc] = True
                neighbors = ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1))
                for index in rng.permutation(4):
                    q = neighbors[int(index)]
                    if 0 <= q[0] < h and 0 <= q[1] < w and allowed[q] and q not in seen:
                        seen.add(q)
                        active.append(q)
        # This also bounds the majority-background constraint before adding bars.
        if mask.sum() > .31 * h * w:
            break
    result = np.zeros_like(mask)
    for cells in _components(mask):
        if len(cells) >= 5:
            rr, cc = np.array(cells).T
            result[rr, cc] = True
    return result


def _make_candidate(rng, shape, initial, bars, palette, max_object_colors):
    bg, tick, high, bar, marker = palette[:5]
    a = np.full(shape, bg, dtype=np.int8)
    a[0, initial - 1:initial + 2] = (high, tick, high)
    allowed = np.ones(shape, dtype=bool)
    allowed[0] = False
    for f, lo, hi, n, sign in bars:
        a[f, lo:hi + 1] = bar
        for j in range(n):
            a[f, lo + 2 * j if sign == 1 else hi - 2 * j] = marker
        allowed[f - 1:f + 2, max(0, lo - 1):min(shape[1], hi + 2)] = False
    mask = _object_mask(rng, allowed)
    parts = _components(mask)
    if not parts:
        return None
    count = min(len(parts), max_object_colors)
    # The local count genuinely varies; every selected color appears in a component.
    count = int(rng.integers(1, count + 1))
    object_colors = tuple(palette[5:5 + count])
    for i, cells in enumerate(parts):
        rr, cc = np.array(cells).T
        a[rr, cc] = object_colors[i % count]
    if np.sum(a == bg) <= a.size / 2:
        return None
    info = dict(roles=tuple(palette[:5]), object_colors=object_colors, bars=bars,
                initial=initial, points=_route(shape[0], initial, bars))
    return a, _render(a, info), info


def _evidence(a, out, info):
    bg, tick, high, _, _ = info["roles"]
    objects = np.isin(a, info["object_colors"])
    path = np.zeros(a.shape, dtype=bool)
    odd, ordinary = False, False
    for r, c, phase, _ in _timing(info["points"]):
        path[r, c] = True
        odd |= bool(objects[r, c] and phase % 2)
        ordinary |= bool(objects[r, c] and phase in (0, 2))
    off = objects & ~path
    sliced = any(any(out[r, c] == high for r, c in cells)
                 and any(out[r, c] == a[r, c] for r, c in cells)
                 for cells in _components(objects))
    rows_only = _render(a, info, rows_only=True)
    depth_output = _render(a, info, mode="depth")
    after_bar = np.arange(len(a))[:, None] > info["bars"][0][0]
    fixed = _route(len(a), info["initial"], info["bars"], fixed_clearance=True)
    return dict(odd_overwrite=odd, ordinary_overwrite=ordinary,
                remote_highlight=bool(np.any(off & (out == high))),
                remote_preserved=bool(np.any(off & (out == a))),
                sliced_component=sliced,
                rotating_scan=bool(np.any(off & (out == high) & (rows_only != high))),
                depth_clock_differs=not np.array_equal(out, depth_output),
                reset_clock_differs=not np.array_equal(out, _render(a, info, mode="reset")),
                unary_differs=not np.array_equal(out, _render(a, info, points=fixed)),
                downstream_phase=bool(np.any(off & after_bar & (out == high) & (depth_output != high))),
                multi=len(info["object_colors"]) > 1)


def _covered(evidence, need_multi):
    return all(any(e[k] for e in evidence) for k in _TEACH) and (
        not need_multi or any(e["multi"] for e in evidence))


def _check_count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _check_count(num_train, "num_train", 2, 10)
    ns = _check_count(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    maximum = int(rng.integers(2, 5))
    max_objects = int(rng.integers(1, 4))
    variable_head = bool(rng.integers(2))
    repeated = bool(rng.random() < .7)
    shift = int(rng.integers(3))
    base = int(rng.integers(4))
    train_dirs = [base, (base + 1) % 4] if shift == 0 else list(range(4))
    test_dirs = [(base + 2) % 4, (base + 3) % 4] if shift == 0 else train_dirs
    max_train_bars = 3 if shift in (1, 2) else 4
    counts = list(map(int, rng.integers(1, max_train_bars + 1, size=nt)))
    counts[0] = int(rng.integers(1, 3))
    counts[1] = max(2, maximum - counts[0])
    slots = [(i, j) for i, count in enumerate(counts) for j in range(count)]
    requirements = [{} for _ in counts]
    for n, slot in zip(range(1, maximum + 1), rng.permutation(len(slots))):
        i, j = slots[int(slot)]
        requirements[i][j] = n
    sequences = []
    for count, required in zip(counts, requirements):
        seq = list(map(int, rng.integers(1, maximum + 1, size=count)))
        for j, n in required.items():
            seq[j] = n
        sequences.append(seq)
    repeat_pair = int(rng.choice([i for i, count in enumerate(counts) if count >= 2])) if repeated else -1
    puzzle, seen, evidence, signs = {"train": [], "test": []}, set(), [], set()

    def add(split, seq, force_repeat=False, last_train=False):
        direction = int(rng.choice(train_dirs if split == "train" else test_dirs))
        for _ in range(48):
            shape, initial, bars = _geometry(rng, seq, repeated, force_repeat, split == "train", shift)
            pair_palette = palette.copy()
            if variable_head:
                available = [c for c in range(10) if c not in palette[3:5 + max_objects]]
                pair_palette[:3] = list(map(int, rng.choice(available, 3, replace=False)))
            for _ in range(12):
                candidate = _make_candidate(rng, shape, initial, bars, pair_palette, max_objects)
                if candidate is None:
                    continue
                a, out, info = candidate
                # Require both endpoint meanings across the demonstrations.
                if last_train and len(signs | {b[-1] for b in bars}) == 1:
                    a, out = a[:, ::-1].copy(), out[:, ::-1].copy()
                    _, info = _decode(a)
                observed = _evidence(a, out, info) if split == "train" else None
                if last_train and not _covered(evidence + [observed], max_objects > 1):
                    continue
                oriented = np.rot90(a, direction).copy()
                if _key(oriented) in seen:
                    continue
                # Decode every accepted input: generation plans cannot bypass ambiguity checks.
                _, decoded = _decode(oriented)
                assert decoded["roles"] == info["roles"]
                seen.add(_key(oriented))
                puzzle[split].append(dict(input=oriented, output=np.rot90(out, direction).copy()))
                if observed is not None:
                    evidence.append(observed)
                    signs.update(b[-1] for b in info["bars"])
                return
        raise RuntimeError("bounded natural-scene construction exhausted")

    for i, seq in enumerate(sequences):
        add("train", seq, i == repeat_pair, i == nt - 1)
    for _ in range(ns):
        count = int(rng.integers(4, 6)) if shift == 1 else int(rng.integers(1, 6))
        seq = list(map(int, rng.integers(1, maximum + 1, size=count)))
        while sum(seq) + 2 * count + 2 > 30:
            j = int(rng.choice([j for j, n in enumerate(seq) if n > 1]))
            seq[j] -= 1
        add("test", seq)
    for split in puzzle:
        puzzle[split] = [puzzle[split][int(i)] for i in rng.permutation(len(puzzle[split]))]
    validate_generated_puzzle(puzzle)
    return puzzle


def _pixel_prediction(a, info):
    """Direct per-cell invariants, separately from the scan/overwrite painter."""
    bg, tick, high, bar, marker = info["roles"]
    timed = _timing(info["points"])
    distances = {(r, c): phase for r, c, phase, _ in timed}
    scan_rows = {r for r, _, phase, forward in timed if phase == 4 and forward}
    scan_cols = {c for _, c, phase, forward in timed if phase == 4 and not forward}
    out = a.copy()
    for r in range(len(a)):
        for c in range(a.shape[1]):
            if (r, c) in distances:
                phase = distances[r, c]
                out[r, c] = bg if phase % 2 else high if phase == 4 else tick
            elif a[r, c] == marker:
                out[r, c] = bar
            elif a[r, c] in info["object_colors"] and (r in scan_rows or c in scan_cols):
                out[r, c] = high
    return out


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, evidence = set(), []
    measured = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, label = _grid(pair["input"]), _grid(pair["output"])
            assert g.shape == label.shape and _key(g) not in seen
            seen.add(_key(g))
            a, info = _decode(g)
            b = np.rot90(label, -info["rotation"])
            assert np.array_equal(b, _pixel_prediction(a, info)), "wrong output pixel"
            assert np.array_equal(solve(g), label), "input-only solve disagrees"
            bg, tick, high, bar, marker = info["roles"]
            assert not np.any(b == marker)
            assert np.all(b[a == bar] == bar)
            assert b[0, info["initial"]] == tick
            assert np.all(b[0, [info["initial"] - 1, info["initial"] + 1]] == high)
            assert len(info["points"]) <= a.size
            # The geometric odometer has consumed every lateral displacement.
            lateral, previous_c = 0, info["initial"]
            for s, (r, c) in enumerate(info["points"]):
                lateral += abs(c - previous_c)
                assert s == r + lateral
                previous_c = c
            measured[split].append((a, info))
            if split == "train":
                evidence.append(_evidence(a, b, info))
    train = measured["train"]
    test = measured["test"]
    assert _covered(evidence, any(len(x["object_colors"]) > 1 for _, x in test))
    train_ns = {b[3] for _, x in train for b in x["bars"]}
    assert len(train_ns) >= 2
    assert {b[-1] for _, x in train for b in x["bars"]} == {-1, 1}
    assert {b[3] for _, x in test for b in x["bars"]} <= train_ns
    for a, info in train:
        correct = _render(a, info)
        no_b = _render(a, info, points=[(r, info["initial"]) for r in range(len(a))])
        no_a = a.copy()
        no_a[no_a == info["roles"][4]] = info["roles"][3]
        for r, c in info["points"]:
            no_a[r, c] = info["roles"][1]
        assert not np.array_equal(correct, no_b), "B is dispensable"
        assert not np.array_equal(correct, no_a), "A is dispensable"
    has_repeat = lambda items: any(any(x[-1] == y[-1] for x, y in zip(info["bars"], info["bars"][1:])) for _, info in items)
    assert not has_repeat(test) or has_repeat(train)
    # Every generated puzzle exposes one actual, input-measured distribution shift.
    novel_dir = bool({x["rotation"] for _, x in test} - {x["rotation"] for _, x in train})
    longer = min(len(x["bars"]) for _, x in test) > max(len(x["bars"]) for _, x in train)
    larger = min(len(a) for a, _ in test) > max(len(a) for a, _ in train)
    assert novel_dir or longer or larger, "missing train/test shift"


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _regressions():
    # Literal geometry and scan expectations, not labels supplied by solve().
    a = np.zeros((13, 15), dtype=np.int8)
    a[0, 3:6] = (4, 3, 4)
    a[6, 2:9] = 8
    a[6, 2] = 2
    a[8:13, 4] = 5
    expected = a.copy()
    expected[6, 2] = 8
    expected[8:13, 4] = 4
    points = ([(r, 4) for r in range(5)] + [(4, c) for c in range(5, 11)]
              + [(r, 10) for r in range(5, 13)])
    for s, (r, c) in enumerate(points):
        expected[r, c] = 0 if s % 2 else 4 if s % 6 == 4 else 3
    assert len(points) == 19
    assert np.array_equal(solve(a), expected)
    # Slice a single component; don't flood-fill the whole highlighted object.
    a[8:13, 5] = 5
    expected[8:13, 5] = 5
    expected[10, 5] = 4
    assert np.array_equal(solve(a), expected)
    permutation = np.array([9, 7, 1, 6, 2, 8, 5, 4, 0, 3], dtype=np.int8)
    for k in range(4):
        for mirror in (False, True):
            aa = np.fliplr(a) if mirror else a
            bb = np.fliplr(expected) if mirror else expected
            assert np.array_equal(solve(np.rot90(permutation[aa], k)), np.rot90(permutation[bb], k))
    # A scan intersects a different route segment: track stencil wins.
    aa = a.copy()
    aa[4, 7] = 5
    assert solve(aa)[4, 7] == 0  # s=7 is odd, despite a scan along row 4.
    # Four markers: turn and clearance are five, with a visible permanent tail.
    aa = np.zeros((19, 24), dtype=np.int8)
    aa[0, 3:6] = (4, 3, 4)
    aa[10, 2:15] = 8
    aa[10, [2, 4, 6, 8]] = 2
    aa[13:18, 1] = 5
    forward, info = _decode(aa)
    assert info["bars"] == [(10, 2, 14, 4, 1)]
    assert (5, 4) in info["points"] and (10, 19) in info["points"]
    assert np.array_equal(solve(aa), _pixel_prediction(forward, info))


def self_test():
    """Bounded streaming audit: no generated grids or execution traces saved."""
    _regressions()
    hashes, shapes, layouts, directions, counts, marker_counts = set(), set(), set(), set(), set(), set()
    transitions, role_sets, component_counts, object_counts, lengths = set(), [set() for _ in range(5)], set(), set(), set()
    total = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        signature = _fingerprint(p)
        assert signature not in hashes
        hashes.add(signature)
        assert signature == _fingerprint(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in p[split]:
                a, info = _decode(pair["input"])
                total += 1
                shapes.add(pair["input"].shape)
                directions.add((split, info["rotation"]))
                counts.add((split, len(info["bars"])))
                marker_counts.update(b[3] for b in info["bars"])
                transitions.update((x[-1], y[-1]) for x, y in zip(info["bars"], info["bars"][1:]))
                for values, color in zip(role_sets, info["roles"]):
                    values.add(color)
                objects = np.isin(a, info["object_colors"])
                component_counts.add(len(_components(objects)))
                object_counts.add(len(info["object_colors"]))
                lengths.update(b[2] - b[1] + 1 for b in info["bars"])
                normalized = np.zeros_like(a)
                for j, color in enumerate((*info["roles"], *info["object_colors"])):
                    normalized[a == color] = j
                layouts.add(_key(normalized))
    assert len(hashes) == 42 and total == 210
    assert len(shapes) >= 50 and len(layouts) >= 200
    assert directions == set(itertools.product(("train", "test"), range(4)))
    assert marker_counts == {1, 2, 3, 4}
    assert transitions == set(itertools.product((-1, 1), repeat=2))
    assert all(len(values) >= 8 for values in role_sets)
    assert object_counts == {1, 2, 3} and len(component_counts) >= 5 and len(lengths) >= 15
    assert {n for s, n in counts if s == "train"} >= {1, 2, 3, 4}
    assert {n for s, n in counts if s == "test"} == {1, 2, 3, 4, 5}
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    for seed in (-1, True, 1.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid seed accepted")
    damaged = generate_puzzle(7)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged output accepted")
    return {"seeds": 42, "pairs": total, "distinct_shapes": len(shapes),
            "distinct_normalized_layouts": len(layouts), "all_count_combinations": 30,
            "independent_solver": False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
