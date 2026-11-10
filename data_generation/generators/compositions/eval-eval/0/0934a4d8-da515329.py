"""Veiled echo pinwheels: reflected pigments terminate four winding probes.

All conventions are either geometric or visible in the input. No parent code,
saved puzzle, seed, or generation record is consulted by the solver.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("A grid must be a 1..30 integer matrix with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _detect(a):
    """Find the two reserved roles by shape, never by literal color."""
    candidates = []
    for cross in np.unique(a):
        points = np.argwhere(a == cross)
        if len(points) % 4 or not 4 <= len(points) <= 24:
            continue
        length = len(points) // 4
        y = int(points[:, 0].min()) + length
        x = int(points[:, 1].min()) + length
        if not (length <= y < a.shape[0]-length
                and length <= x < a.shape[1]-length):
            continue
        expected = np.zeros(a.shape, dtype=bool)
        for d in range(1, length+1):
            expected[y-d, x] = expected[y+d, x] = True
            expected[y, x-d] = expected[y, x+d] = True
        if not np.array_equal(a == cross, expected):
            continue
        for veil in np.unique(a):
            if veil == cross:
                continue
            p = np.argwhere(a == veil)
            r, s = int(p[:, 0].min()), int(p[:, 0].max())+1
            c, d = int(p[:, 1].min()), int(p[:, 1].max())+1
            if not (r <= y-length and y+length < s
                    and c <= x-length and x+length < d):
                continue
            region = np.zeros(a.shape, dtype=bool)
            region[r:s, c:d] = True
            if np.array_equal(region, (a == veil) | expected):
                candidates.append((int(veil), int(cross), length, (y, x), (r, s, c, d)))
    if len(candidates) != 1:
        raise ValueError("Require one hollow cross inside one otherwise solid veil")
    return candidates[0]


def _axes(a, unknown):
    """Infer supported, possibly offset, axial reflections from visible pixels."""
    rr, cc = np.indices(a.shape)
    result = [[], []]
    for axis, n in enumerate(a.shape):
        coord = (rr, cc)[axis]
        for total in range(2*n-1):
            overlap = n - abs(total-(n-1))
            if overlap < .65*n:
                continue
            mate = total-coord
            selected = (mate >= 0) & (mate < n) & (coord < mate)
            r, c = rr[selected], cc[selected]
            mr, mc = (total-r, c) if axis == 0 else (r, total-c)
            visible = ~unknown[r, c] & ~unknown[mr, mc]
            if visible.sum() < max(18, .12*a.size):
                continue
            left, right = a[r[visible], c[visible]], a[mr[visible], mc[visible]]
            if len(np.unique(left)) >= 2 and np.array_equal(left, right):
                result[axis].append(total)
    return result


def _recover(a, record):
    veil, cross, _, _, box = record
    unknown = (a == veil) | (a == cross)
    axes = _axes(a, unknown)
    r, s, c, d = box
    queries = [(y, x) for y in range(r, s) for x in range(c, d)]
    answers, models = [], []
    # A single axis can suffice; two commuting reflections also allow a
    # surviving diagonal member of the orbit. Every used witness is in bounds.
    for hs in [None] + axes[0]:
        for vs in [None] + axes[1]:
            if hs is None and vs is None:
                continue
            values = []
            for y, x in queries:
                ys = [y] if hs is None else [y, hs-y]
                xs = [x] if vs is None else [x, vs-x]
                witnesses = {int(a[yy, xx]) for yy in ys for xx in xs
                             if 0 <= yy < a.shape[0] and 0 <= xx < a.shape[1]
                             and not unknown[yy, xx]}
                if len(witnesses) != 1:
                    break
                values.append(witnesses.pop())
            if len(values) == len(queries):
                answers.append(np.array(values, dtype=np.int8).reshape(s-r, d-c))
                models.append((hs, vs))
    if not answers or any(not np.array_equal(answers[0], b) for b in answers[1:]):
        raise ValueError("The veil needs complete, agreeing reflected witnesses")
    full = a.copy()
    full[r:s, c:d] = answers[0]
    return full, axes, models


def _walk(length, phase, reach, hand=1):
    """Ordered infinite-plane path, long enough for all visits within reach.

    The only clipping happens at the consumer. In particular an exit does not
    end a path, which may subsequently re-enter the query window.
    """
    radius = max(2, length)
    dy, dx = -1, 0
    for _ in range(phase):
        dy, dx = dx, -dy
    for distance in range(1, radius+1):
        yield distance*dy, distance*dx, False, 0
    y, x = radius*dy, radius*dx
    dy, dx = hand*dx, -hand*dy
    ordinal = 0
    # Every four stages move the path's envelope outward by eight cells.
    for k in range(4*(reach//8+2)):
        for vy, vx, steps in ((dy, dx, 4*k+2),
                              (hand*dx, -hand*dy, radius-2),
                              (dy, dx, radius)):
            for _ in range(steps):
                y, x = y+vy, x+vx
                ordinal += 1
                yield y, x, True, ordinal
        dy, dx = hand*dx, -hand*dy


def _draw(pigments, length, center, veil, *, stop=True, hand=1):
    """Draw using a completed query window, recording causal first-hit evidence."""
    h, w = pigments.shape
    cy, cx = center
    reach = max(cy, cx, h-1-cy, w-1-cx)
    answer = np.full((h, w), veil, dtype=np.int8)
    owner = np.full((h, w), -1, dtype=np.int8)
    records = []
    for phase, (dy, dx) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
        payload = int(pigments[cy+length*dy, cx+length*dx])
        visits, gate, outside, reentry = [], None, False, False
        for ry, rx, eligible, ordinal in _walk(length, phase, reach, hand):
            y, x = cy+ry, cx+rx
            if not (0 <= y < h and 0 <= x < w):
                outside = True
                continue
            reentry |= outside
            assert owner[y, x] in (-1, phase), "Winding arms cannot collide"
            owner[y, x] = phase
            answer[y, x] = payload
            visits.append((y, x, eligible, ordinal))
            if stop and eligible and int(pigments[y, x]) == payload:
                gate = (y, x, ordinal)
                break
        records.append({"payload": payload, "visits": visits, "gate": gate,
                        "reentry": reentry})
    assert answer[cy, cx] == veil
    return answer, records


def _analyse(a):
    record = _detect(a)
    full, axes, models = _recover(a, record)
    veil, cross, length, (y, x), (r, s, c, d) = record
    pigments = full[r:s, c:d]
    center = (y-r, x-c)
    answer, arms = _draw(pigments, length, center, veil)
    return answer, {"veil": veil, "cross": cross, "length": length,
                    "box": (r, s, c, d), "center": center,
                    "pigments": pigments, "axes": axes, "models": models,
                    "arms": arms}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input alone. Clockwise is fixed throughout the family."""
    return _analyse(_grid(input_grid))[0]


def _pattern(rng, shape, palette, style):
    p = np.asarray(palette, dtype=np.int8)
    if style == 0:
        return rng.choice(p, size=shape)
    if style == 1:
        small = rng.choice(p, size=((shape[0]+1)//2, (shape[1]+1)//2))
        return np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)[:shape[0], :shape[1]]
    if style == 2:
        motif = rng.choice(p, size=(3, 4))
        tiled = np.tile(motif, ((shape[0]+2)//3, (shape[1]+3)//4))
        a = tiled[:shape[0], :shape[1]].copy()
        noise = rng.random(shape) < .20
        a[noise] = rng.choice(p, size=int(noise.sum()))
        return a
    weights = np.full(len(p), .5/(len(p)-1))
    weights[0] = .5
    return rng.choice(p, size=shape, p=weights)


def _field(rng, palette, style):
    bh, bw = (int(v) for v in rng.integers(10, 17, size=2))
    base = _pattern(rng, (bh, bw), palette, style)
    odd_h, odd_w = bool(rng.integers(2)), bool(rng.integers(2))
    upper = np.concatenate((base, base[:, -2::-1] if odd_w else base[:, ::-1]), axis=1)
    full = np.concatenate((upper, upper[-2::-1] if odd_h else upper[::-1]), axis=0)
    cuts = []
    for n in full.shape:
        amount = int(rng.integers(max(0, n-30), max(0, n-30)+4))
        leading = int(rng.integers(amount+1))
        cuts.append((leading, n-amount+leading))
    a = full[cuts[0][0]:cuts[0][1], cuts[1][0]:cuts[1][1]].copy()
    sums = (full.shape[0]-1-2*cuts[0][0], full.shape[1]-1-2*cuts[1][0])
    return a, sums


def _starts(n, size, total, side, interior):
    """Feasible starts wholly on one side of an axis, or crossing it."""
    lo, hi = (1, n-size-1) if interior else (0, n-size)
    result = []
    for start in range(lo, hi+1):
        end = start+size-1
        if side == -1 and 2*end < total:
            result.append(start)
        elif side == 1 and 2*start > total:
            result.append(start)
        elif side == 0 and 2*start < total < 2*end:
            result.append(start)
    return result


def _make_pair(rng, length, placement, regime, roles, style, teaching, seen):
    veil, cross, palette = roles
    radius = max(2, length)
    minimum = 2*radius+5 if teaching else 2*length+1
    for attempt in range(240):
        # Coherent motifs remain an optional distribution preference. Training
        # evidence takes priority if that preference keeps giving short probes.
        field, sums = _field(rng, palette, 0 if teaching and attempt >= 60 else style)
        h, w = field.shape
        if placement == "direct":
            mh = int(rng.integers(minimum, min(16, h)+1))
            mw = int(rng.integers(minimum, min(16, w)+1))
            rs = _starts(h, mh, sums[0], int(rng.choice([-1, 1])), True)
            cs = _starts(w, mw, sums[1], int(rng.choice([-1, 1])), True)
        else:
            # A tall seam window and its quarter-turns; the other dimension is
            # on one side, retaining reflected witnesses for the seam overlap.
            limit = min(21, h-2) if teaching else min(24, h)
            mh = int(rng.integers(minimum, limit+1))
            mw = int(rng.integers(minimum, min(16, w)+1))
            rs = _starts(h, mh, sums[0], 0, placement != "edge")
            cs = _starts(w, mw, sums[1], int(rng.choice([-1, 1])), placement != "edge")
            if placement == "edge":
                # Hold out an edge of the veil, not a new pigment or path rule.
                if regime == "tight":
                    mh = minimum
                rs = list(range(h-mh+1))
                cs = [v for v in cs if v in (0, w-mw)]
        if not rs or not cs:
            continue
        r, c = int(rng.choice(rs)), int(rng.choice(cs))
        padding = 2 if teaching else 0
        low = length+padding
        cy = int(rng.integers(low, mh-low))
        cx = int(rng.integers(low, mw-low))
        # Radius-two completion for L1 may be clipped in nonteaching windows.
        center = cy, cx
        pigments = field[r:r+mh, c:c+mw]
        answer, arms = _draw(pigments, length, center, veil)
        ungated, whole_arms = _draw(pigments, length, center, veil, stop=False)
        if not np.any(answer != ungated):
            continue
        if teaching:
            payloads = {arm["payload"] for arm in arms}
            gates = [arm["gate"] for arm in arms if arm["gate"] is not None]
            if len(payloads) < 2 or len(gates) < 2:
                continue
            # A surviving first full forward/jog/forward triple is visible.
            if not any(g[2] >= 2*radius for g in gates):
                continue
            later_echo = any(arm["gate"] is not None and any(
                eligible and ordinal > arm["gate"][2]
                and int(pigments[yy, xx]) == arm["payload"]
                for yy, xx, eligible, ordinal in whole["visits"])
                for arm, whole in zip(arms, whole_arms))
            inner_differs = any(int(pigments[cy+dy, cx+dx]) != arm["payload"]
                                for (dy, dx), arm in zip(
                                    ((-1, 0), (0, 1), (1, 0), (0, -1)), arms))
            if not later_echo or not inner_differs:
                continue
            reflected, _ = _draw(pigments, length, center, veil, hand=-1)
            if np.array_equal(answer, reflected):
                continue
        a = field.copy()
        a[r:r+mh, c:c+mw] = veil
        for distance in range(1, length+1):
            a[r+cy-distance, c+cx] = a[r+cy+distance, c+cx] = cross
            a[r+cy, c+cx-distance] = a[r+cy, c+cx+distance] = cross
        # Quarter turns change windows, seam orientation and asymmetric margins
        # while preserving the fixed clockwise winding.
        turn = int(rng.integers(4))
        a, answer = np.rot90(a, turn).copy(), np.rot90(answer, turn).copy()
        if _key(a) in seen:
            continue
        try:
            prediction, info = _analyse(a)
        except ValueError:
            continue
        if not np.array_equal(prediction, answer):
            continue
        seen.add(_key(a))
        return {"input": a, "output": answer}
    raise RuntimeError("Bounded sampler could not find a witnessed echo pinwheel")


def _count(value, lo, hi, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, "num_train")
    nq = _count(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    regime = str(rng.choice(["edge", "scale", "tight"]))
    cap = int(rng.integers(3, 5))
    train_lengths = [cap-1, cap]
    train_high = cap if regime == "scale" else 6
    train_lengths.extend(int(rng.integers(1, train_high+1)) for _ in range(nt-2))
    test_lengths = [cap+1]*nq if regime == "scale" else [
        int(rng.choice(train_lengths)) for _ in range(nq)]
    common = bool(rng.random() < .25)
    shared_roles = None
    style_preference = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for index, length in enumerate(train_lengths+test_lengths):
        if shared_roles is None or not common:
            colors = rng.permutation(10)
            size = int(rng.integers(5 if index < 2 else 3, 9))
            roles = (int(colors[0]), int(colors[1]), tuple(int(v) for v in colors[2:2+size]))
            if shared_roles is None:
                shared_roles = roles
        else:
            roles = shared_roles
        teaching = index < 2
        style = style_preference if rng.random() < .6 else int(rng.integers(4))
        if index < nt:
            placement = "direct" if index == 0 else "seam"
            split = "train"
        else:
            placement = "seam" if regime == "scale" else "edge"
            split = "test"
        pair = _make_pair(rng, length, placement, regime, roles, style, teaching, seen)
        puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check returned pixels, visible witnesses, gates, and teaching evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, teaching, echo_evidence, tip_evidence = set(), [], [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen, "Duplicate input"
            seen.add(_key(a))
            expected, info = _analyse(a)
            assert np.array_equal(expected, b), "Wrong ordered first-hit trails"
            veil, cross, length = info["veil"], info["cross"], info["length"]
            pigments, center, arms = info["pigments"], info["center"], info["arms"]
            assert cross not in b and veil not in pigments and cross not in pigments
            assert b[center] == veil and np.count_nonzero(b != veil) > 4*length
            assert np.any(b == veil) and not np.array_equal(b, pigments)
            assert set(int(v) for v in np.unique(b)) <= {
                veil, *(arm["payload"] for arm in arms)}
            # Each eligible prefix is a nonmatching pigment until its terminal
            # matching cell. The gate is included, and no arm resumes afterward.
            for arm in arms:
                eligible = [(y, x, k) for y, x, e, k in arm["visits"] if e]
                if arm["gate"] is not None:
                    assert eligible[-1] == arm["gate"]
                    y, x, _ = arm["gate"]
                    assert int(pigments[y, x]) == arm["payload"]
                    eligible = eligible[:-1]
                assert all(int(pigments[y, x]) != arm["payload"] for y, x, _ in eligible)
            ungated, whole_arms = _draw(pigments, length, center, veil, stop=False)
            assert not np.array_equal(b, ungated), "Stopping must affect every pair"
            if split == "train":
                teaching.append((info, b))
                echo_evidence.append(any(arm["gate"] is not None and any(
                    eligible and ordinal > arm["gate"][2]
                    and int(pigments[yy, xx]) == arm["payload"]
                    for yy, xx, eligible, ordinal in whole["visits"])
                    for arm, whole in zip(arms, whole_arms)))
                cy, cx = center
                tip_evidence.append(any(int(pigments[cy+dy, cx+dx]) != arm["payload"]
                                        for (dy, dx), arm in zip(
                                            ((-1, 0), (0, 1), (1, 0), (0, -1)), arms)))
    # Even the smallest two-demonstration request teaches different payloads,
    # long-spoke jogs, handedness and causal stopping.
    assert any(len({arm["payload"] for arm in i["arms"]}) >= 2 for i, _ in teaching)
    assert any(i["length"] >= 3 and any(
        arm["gate"] is not None and arm["gate"][2] >= 2*i["length"]
        for arm in i["arms"]) for i, _ in teaching)
    assert any(not np.array_equal(b, _draw(i["pigments"], i["length"],
               i["center"], i["veil"], hand=-1)[0]) for i, b in teaching)
    assert any(echo_evidence) and any(tip_evidence)


def _analytic_mask(length, shape, center):
    """Interval membership cross-check for the ungated recurrence, not a solver."""
    radius = max(2, length)
    yy, xx = np.indices(shape)
    yy, xx = yy-center[0], xx-center[1]
    reach = max(int(np.abs(yy).max()), int(np.abs(xx).max()))
    mask = np.zeros(shape, dtype=bool)
    for _ in range(4):
        mask |= (xx == 0) & (yy >= -radius) & (yy <= -1)
        for q in range(reach//8+3):
            starts = ((-radius-8*q, -8*q), (-2-8*q, radius+8*q+2),
                      (radius+8*q+4, 8*q+4), (8*q+6, -radius-8*q-6))
            for phase, ((y, x), (dy, dx)) in enumerate(zip(
                    starts, ((0, 1), (1, 0), (0, -1), (-1, 0)))):
                along, across = (yy-y)*dy+(xx-x)*dx, (yy-y)*dx-(xx-x)*dy
                run = 16*q+4*phase+2
                mask |= (across == 0) & (along >= 0) & (along <= run)
                mask |= (along == run) & (across >= 0) & (across <= radius-2)
                mask |= (across == radius-2) & (along >= run) & (along <= run+radius)
        yy, xx = xx, -yy
    return mask


def _gate_fixture():
    """A small constructed audit case: two north matches, only the first stops."""
    base = np.full((12, 13), 5, dtype=np.int8)
    # Query starts at (1,1), center is (5,5) in the field, length is two.
    # The inner spoke cells have pigment 5; the actual tips carry 1,2,3,4.
    for y, x, color in ((3, 5, 1), (5, 7, 2), (7, 5, 3), (5, 3, 4),
                        (3, 7, 1), (7, 7, 2), (7, 3, 3), (3, 3, 4),
                        (4, 9, 1)):
        base[y, x] = color
    upper = np.concatenate((base, base[:, ::-1]), axis=1)
    full = np.concatenate((upper, upper[::-1]), axis=0)
    a = full.copy()
    a[1:10, 1:10] = 0
    for distance in (1, 2):
        a[5-distance, 5] = a[5+distance, 5] = 9
        a[5, 5-distance] = a[5, 5+distance] = 9
    return a, full[1:10, 1:10].copy()


def self_test():
    """Bounded audits of 42 seeds, explicit counts, geometry, and mutations."""
    signatures, input_shapes, output_shapes, lengths, palettes = set(), set(), set(), set(), set()
    features = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple((_key(p["input"]), _key(p["output"]))
                          for s in ("train", "test") for p in puzzle[s])
        signatures.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                _, i = _analyse(a)
                input_shapes.add(a.shape)
                output_shapes.add(b.shape)
                lengths.add(i["length"])
                palettes.add((i["veil"], i["cross"], tuple(sorted(np.unique(i["pigments"]).tolist()))))
                r, s, c, d = i["box"]
                features["edge"] += int(r == 0 or c == 0 or s == a.shape[0] or d == a.shape[1])
                features["offset"] += int(any(t != a.shape[k]-1 for k in (0, 1) for t in i["axes"][k]))
                features["seam"] += int(any(2*(r, c)[k] < t < 2*((s, d)[k]-1)
                                            for k in (0, 1) for t in i["axes"][k]))
                features["reentry"] += int(any(arm["reentry"] for arm in i["arms"]))
                features["no_gate"] += sum(arm["gate"] is None for arm in i["arms"])
                features["multiple_payloads"] += int(len({arm["payload"] for arm in i["arms"]}) >= 3)
                assert np.array_equal(solve(a.copy()), b)
        if seed in (0, 7, 41):
            again = generate_puzzle(seed)
            assert signature == tuple((_key(p["input"]), _key(p["output"]))
                                      for s in ("train", "test") for p in again[s])
    assert len(signatures) == 42
    assert len(input_shapes) >= 35 and len(output_shapes) >= 25 and len(palettes) >= 40
    assert lengths == set(range(1, 7))
    assert all(features[k] > 0 for k in ("edge", "offset", "seam", "reentry", "no_gate", "multiple_payloads"))
    for length in range(1, 7):
        for shape in ((2*length+1, 2*length+3), (17, 23), (30, 30)):
            center = (length, length)
            pigments = np.ones(shape, dtype=np.int8)
            ungated, _ = _draw(pigments, length, center, 0, stop=False)
            assert np.array_equal(ungated != 0, _analytic_mask(length, shape, center))
    a, pigments = _gate_fixture()
    expected, arms = _draw(pigments, 2, (4, 4), 0)
    assert np.array_equal(solve(a), expected)
    assert [arm["payload"] for arm in arms] == [1, 2, 3, 4]
    assert arms[0]["gate"] == (2, 6, 2)
    assert pigments[3, 8] == 1 and expected[3, 8] == 0  # Later echo stays unpainted.
    assert pigments[2, 5] == 5 and expected[2, 5] == 1  # Carry, rather than reveal.
    assert not np.array_equal(expected, pigments)  # Reconstruction alone fails.
    assert not np.array_equal(expected, np.where(_analytic_mask(2, (9, 9), (4, 4)), 9, 0))
    # On an 18x18 reflected field, erasing both central seams removes every
    # witness of the central orbit; there must be no invented reconstruction.
    full = np.concatenate((pigments, pigments[:, ::-1]), axis=1)
    full = np.concatenate((full, full[::-1]), axis=0)
    broken = full.copy()
    broken[6:12, 6:12] = 0
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        broken[8+dy, 8+dx] = 9
    try:
        solve(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("A completely erased orbit was accepted")
    # All legal count combinations; no stochastic count inference is needed.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(913, nt, nq)
            assert (len(p["train"]), len(p["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    p = generate_puzzle(0)
    # Palette equivariance includes changing mask and cross marker identities.
    permutation = np.array([6, 9, 0, 4, 7, 1, 8, 3, 5, 2], dtype=np.int8)
    for pair in p["train"]+p["test"]:
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
        assert np.array_equal(solve(np.rot90(pair["input"])), np.rot90(pair["output"]))
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("A mutated label passed validation")
    return {"seeds": 42, "pairs": 210, "input_shapes": len(input_shapes),
            "output_shapes": len(output_shapes), "palettes": len(palettes),
            "lengths": sorted(lengths), "features": dict(features)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
