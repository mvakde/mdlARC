"""A comb adjusts striped pins carrying symmetry heads; fixed motifs are copied.

Only NumPy and the standard library are used. No generated files, parent modules,
examples, RNG, or seed are accessed by solve(). Canonical geometry is documented
in task_spec.md; all eight whole-scene orientations are supported.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json

import numpy as np


HOST_SEEDS = (0,374711116,945610396,623120591,371416479,392593158,325616938,
401848668,995702444,478704606,16490397,506530920,589339923,617937932,
160652905,705868825,777586622,106503609,687106025,592468601,667805161,
337805815,714089346,120092518,166831556,180626193,757298785,747776768,
250052447,169384583,430220824,772883092,394989516,325078275,442558585,
333868781,89617049,294742080,630937910,732241405,721947856,949794372)


def _orient(a, code):
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4).copy()


def _unorient(a, code):
    b = np.rot90(a, -(code % 4))
    return (np.fliplr(b) if code >= 4 else b).copy()


def _grid(a):
    a = np.asarray(a)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("Expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return (a.shape, a.tobytes())


def _connected(cells):
    remaining = set(cells)
    if not remaining:
        return False
    todo = [remaining.pop()]
    while todo:
        r, c = todo.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
    return not remaining


def _destination(lane, axis=None, kind=None):
    a = lane["axis"] if axis is None else axis
    point = (lane["kind"] if kind is None else kind) == "P"
    return [(2*a-r, 2*lane["x"]-c if point else c, v)
            for r, c, v in lane["source"]]


def _witness(lane):
    """Refute unturned translation into the opposite bounding box, with colors."""
    src, dst = lane["source"], _destination(lane)
    dr = min(r for r, c, v in dst) - min(r for r, c, v in src)
    dc = min(c for r, c, v in dst) - min(c for r, c, v in src)
    return set(dst) != {(r+dr, c+dc, v) for r, c, v in src}


def _canonical_read(g, bg, s):
    """Recognize the complete hardware from pixels, then assign local motifs."""
    rr, cc = np.nonzero(g == s)
    assert len(rr)
    top, k = int(rr.min()), int(rr.max())
    left, right = int(cc.min()), int(cc.max())
    assert right-left >= 15 and top < k
    assert np.all(g[top, left:right+1] == s)
    b = top
    while b+1 < g.shape[0] and g[b+1, left] == s:
        b += 1
    br = top
    while br+1 < g.shape[0] and g[br+1, right] == s:
        br += 1
    assert br == b and b-top >= 8 and k > b+2
    # The detached base is the lowest structural run; its teeth align with pins.
    basecols = np.flatnonzero(g[k] == s)
    assert len(basecols) >= 8
    xs = np.flatnonzero(g[k-1] == s).tolist()
    assert 2 <= len(xs) <= 4 and min(np.diff(xs)) >= 7
    assert basecols.tolist() == list(range(xs[0], xs[-1]+1))
    assert xs[0]-left >= 4 and right-xs[-1] >= 4
    covered = np.zeros(g.shape, bool)
    covered[top, left:right+1] = True
    covered[top:b+1, left] = covered[top:b+1, right] = True
    covered[k, xs[0]:xs[-1]+1] = True
    heights, lanes, marker = [], [], None
    for x in xs:
        t = k-1
        while t >= 0 and g[t, x] == s:
            t -= 1
        h = k-1-t
        assert h >= 1 and k-h > b+1
        heights.append(h)
        covered[k-h:k, x] = True
        d = top+1
        while d < b and g[d, x] != bg:
            d += 1
        d -= 1
        assert d > top+2 and d < b-1
        m = int(g[d, x])
        assert m not in (bg, s) and (marker is None or m == marker)
        marker = m
        seq = tuple(int(v) for v in g[top+1:d+1, x])
        assert marker not in seq[:-1]
        lo = hi = x
        while lo > 0 and g[d, lo-1] == marker:
            lo -= 1
        while hi+1 < g.shape[1] and g[d, hi+1] == marker:
            hi += 1
        assert 1 <= hi-lo+1 <= 5 and left < lo <= x <= hi < right
        assert x-2 <= lo and hi <= x+2
        covered[top+1:d+1, x] = True
        covered[d, lo:hi+1] = True
        a = b-h-1
        shift = d-a
        assert a > top and shift >= 0
        lanes.append(dict(x=x, bottom=d, axis=a, shift=shift, seq=seq,
                          head=tuple(range(lo, hi+1)),
                          kind="P" if lo == hi else "H", source=[]))
    # Payload lies in a visible, nonoverlapping band to the right of its pin.
    for r, c in np.argwhere((g != bg) & ~covered):
        r, c = int(r), int(c)
        eligible = [lane for lane in lanes
                    if lane["x"] < c <= lane["x"]+3 and lane["bottom"] < r < b]
        assert len(eligible) == 1
        v = int(g[r, c])
        assert v not in (s, marker)
        eligible[0]["source"].append((r, c, v))
    targets = set()
    for lane in lanes:
        src = lane["source"]
        assert 2 <= len(src) <= 9
        assert _connected((r, c) for r, c, v in src)
        assert max(r for r, c, v in src)-min(r for r, c, v in src) < 3
        assert min(r for r, c, v in src)-lane["bottom"] in (1, 2)
        for r, c, v in _destination(lane):
            assert top < r < b and left < c < right
            assert g[r, c] == bg and (r, c) not in targets
            targets.add((r, c))
    return dict(shape=g.shape, bg=bg, structure=s, marker=marker, top=top,
                base=b, left=left, right=right, keybase=k,
                xs=tuple(xs), heights=tuple(heights), lanes=lanes)


def _read(a):
    colors, counts = np.unique(a, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    bg = int(colors[counts.argmax()])
    candidates = []
    for code in range(8):
        g = _unorient(a, code)
        for s in colors:
            if int(s) == bg:
                continue
            try:
                m = _canonical_read(g, bg, int(s))
            except (AssertionError, IndexError, ValueError):
                continue
            m["orientation"] = code
            candidates.append((g, m))
    assert len(candidates) == 1, "Machine must have exactly one geometric interpretation"
    return candidates[0]


def _simulate(g, m):
    """Move actual input cells one step at a time; then use the moved markers."""
    out = g.copy()
    bg, t, b, k = m["bg"], m["top"], m["base"], m["keybase"]
    keycells = [(int(r), int(c), int(g[r, c]))
                for r, c in np.argwhere((g != bg) & (np.indices(g.shape)[0] > b))]
    for r, c, v in keycells:
        out[r, c] = bg
    for lane in m["lanes"]:
        x, d = lane["x"], lane["bottom"]
        out[t+1:d+1, x] = bg
        out[d, list(lane["head"])] = bg
    for r, c, v in keycells:
        out[r-(k-b), c] = v
    for lane in m["lanes"]:
        x, d = lane["x"], lane["bottom"]
        tooth = b
        while out[tooth-1, x] == m["structure"]:
            tooth -= 1
        cells = [(r, x, int(g[r, x])) for r in range(t+1, d)]
        cells += [(d, c, int(g[d, c])) for c in lane["head"]]
        bottom = d
        while bottom >= tooth:
            cells = [(r-1, c, v) for r, c, v in cells if r-1 > t]
            bottom -= 1
        assert bottom == tooth-1 and cells
        for r, c, v in cells:
            out[r, c] = v
        head = [(r, c) for r, c, v in cells if v == m["marker"]]
        assert head and len({r for r, c in head}) == 1
        axis = head[0][0]
        for r, c, v in lane["source"]:
            q = (2*axis-r, 2*x-c if len(head) == 1 else c)
            assert out[q] == bg
            out[q] = v
    return _orient(out, m["orientation"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution. Color roles, direction, pin contents, and motifs are read."""
    a = _grid(input_grid)
    try:
        g, m = _read(a)
    except AssertionError as exc:
        raise ValueError(str(exc)) from exc
    return _simulate(g, m)


def _motif(rng, height, width, witness):
    if witness:
        mask = np.ones((height, width), bool)
        mask[0, 0] = False
    else:
        kind = int(rng.integers(5))
        mask = np.zeros((height, width), bool)
        if kind == 0:
            mask[:] = True
        elif kind == 1:
            mask[:, 0] = mask[-1, :] = True
        elif kind == 2:
            mask[0, :] = mask[-1, :] = True
            mask[:, 0] = mask[:, -1] = True
        elif kind == 3:
            for r in range(height):
                mask[r, r % width] = True
        else:
            mask[rng.integers(height), rng.integers(width)] = True
            for _ in range(int(rng.integers(2, height*width+1))):
                frontier = sorted({(r+dr, c+dc) for r, c in np.argwhere(mask)
                                   for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                                   if 0 <= r+dr < height and 0 <= c+dc < width
                                   and not mask[r+dr, c+dc]})
                if frontier:
                    mask[frontier[int(rng.integers(len(frontier)))]] = True
            # Keep the requested depth visible, including in depth-transfer tests.
            col = int(np.argwhere(mask)[0, 1])
            mask[:, col] = True
        if mask.sum() < 2:
            mask[-1, :] = True
    if rng.random() < .5:
        mask = mask[:, ::-1]
    rows, cols = np.nonzero(mask)
    return mask[rows.min():rows.max()+1, cols.min():cols.max()+1].copy()


def _make_model(rng, n, roles, orientation, deep, demonstration=False):
    bg, s, marker = roles
    pool = [int(v) for v in rng.choice([c for c in range(10) if c not in roles],
                                      size=int(rng.integers(2, 6)), replace=False)]
    top = int(rng.integers(0, 3))
    span = int(rng.integers(12, 19))
    cap = min(7, (span-2)//2)
    slack = 30-(9+7*(n-1))
    gaps = [7]*(n-1)
    for j in rng.permutation(n-1):
        extra = int(rng.integers(0, min(2, slack)+1))
        gaps[int(j)] += extra
        slack -= extra
    lm = int(rng.integers(0, min(2, slack)+1))
    slack -= lm
    rm = int(rng.integers(0, min(2, slack)+1))
    xs = [lm+4]
    for step in gaps:
        xs.append(xs[-1]+step)
    b = top+span
    kinds = ["H", "P"] + [str(rng.choice(("H", "P"))) for _ in range(n-2)]
    rng.shuffle(kinds)
    flat = rng.random() < .2 and not demonstration
    flat_h = int(rng.integers(4, cap+1))
    heights, lanes = [], []
    for i, x in enumerate(xs):
        ph = 3 if deep else 2
        pw = int(rng.integers(2, 4))
        mask = _motif(rng, ph, pw, demonstration)
        ph, pw = mask.shape
        # Two-cell gaps occur where the same fitting/contact arithmetic permits.
        gap = 1 if demonstration else int(rng.choice((1, 1, 2)))
        low = ph+gap-1 + int(demonstration and i == 1)
        h = flat_h if flat else int(rng.integers(low, cap+1))
        h = max(h, low)
        a = b-h-1
        maxshift = min(b-gap-ph-a, a-top-gap-ph)
        assert maxshift >= 0
        shift = int(rng.integers(0, maxshift+1))
        if demonstration and i == 0:
            shift = 0
        if demonstration and i == 1:
            shift = int(rng.integers(1, maxshift+1))
        d = a+shift
        # Ordered runs make both discarded and surviving colors inspectable.
        seq = [s]*(d-top-1)
        style = int(rng.integers(3))
        if style:
            cuts = sorted(set([0, len(seq)] + rng.integers(0, len(seq)+1, 4).tolist()))
            for lo, hi in zip(cuts, cuts[1:]):
                seq[lo:hi] = [int(rng.choice([s]+pool))]*(hi-lo)
        if demonstration and i == 1:
            seq[0] = pool[0]
            seq[shift] = pool[0]
            seq[shift+1] = pool[1]
        seq.append(marker)
        if kinds[i] == "P":
            head = (x,)
        else:
            lo = int(rng.integers(-2, 1))
            hi = int(rng.integers(max(0, lo+1), 3))
            head = tuple(range(x+lo, x+hi+1))
        colors = [int(rng.choice(pool)) for _ in range(int(rng.integers(1, len(pool)+1)))]
        source = [(d+gap+int(r), x+1+int(c), int(rng.choice(colors)))
                  for r, c in np.argwhere(mask)]
        heights.append(h)
        lanes.append(dict(x=x, bottom=d, axis=a, shift=shift, seq=tuple(seq),
                          kind=kinds[i], head=head, source=source))
    k = b+max(heights)+2
    bottompad = int(rng.integers(0, min(2, 29-k)+1))
    return dict(shape=(k+1+bottompad, xs[-1]+5+rm), bg=bg, structure=s,
                marker=marker, top=top, base=b, left=lm, right=xs[-1]+4,
                keybase=k, xs=tuple(xs), heights=tuple(heights), lanes=lanes,
                orientation=orientation)


def _render(m, output=False):
    """Direct construction. The input-only predictor uses stepwise pixel motion."""
    g = np.full(m["shape"], m["bg"], np.int8)
    t, b, l, r, s = m["top"], m["base"], m["left"], m["right"], m["structure"]
    g[t, l:r+1] = s
    g[t:b+1, l] = g[t:b+1, r] = s
    k = b if output else m["keybase"]
    g[k, m["xs"][0]:m["xs"][-1]+1] = s
    for h, lane in zip(m["heights"], m["lanes"]):
        x = lane["x"]
        g[k-h:k, x] = s
        shift = lane["shift"] if output else 0
        for j, v in enumerate(lane["seq"], t+1):
            if j-shift > t:
                g[j-shift, x] = v
        headrow = lane["bottom"]-shift
        g[headrow, list(lane["head"])] = m["marker"]
        for rr, cc, v in lane["source"]:
            assert g[rr, cc] == m["bg"]
            g[rr, cc] = v
        if output:
            for rr, cc, v in _destination(lane):
                assert g[rr, cc] == m["bg"]
                g[rr, cc] = v
    return _orient(g, m["orientation"])


def _comparison(m):
    # Same hardware/source pixels, a one-cell shorter tooth on the moving lane.
    other = {**m, "lanes": [{**lane, "source": list(lane["source"])} for lane in m["lanes"]]}
    h = list(m["heights"])
    i = next(i for i, lane in enumerate(m["lanes"]) if lane["shift"] > 0)
    h[i] -= 1
    other["heights"] = tuple(h)
    other["lanes"][i]["axis"] += 1
    other["lanes"][i]["shift"] -= 1
    return other


def _pair(m, seen):
    # Finite uniqueness fallback: 49 two-cell assignments exceed 14 pair slots.
    inp = _render(m)
    if _key(inp) in seen:
        pool = [c for c in range(10) if c not in (m["bg"], m["structure"], m["marker"])]
        source = m["lanes"][0]["source"]
        original = list(source)
        for a in pool:
            for b in pool:
                source[:] = [(r, c, a if i == 0 else b if i == 1 else v)
                             for i, (r, c, v) in enumerate(original)]
                inp = _render(m)
                if _key(inp) not in seen:
                    break
            else:
                continue
            break
        else:
            raise AssertionError("Exceeded the 14-pair bound")
    seen.add(_key(inp))
    return {"input": inp, "output": _render(m, True)}


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError("Counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    roles = tuple(int(c) for c in rng.choice(10, 3, replace=False))
    orientation = int(rng.integers(8))
    orientation_mode = str(rng.choice(("fixed", "mixed", "transfer")))
    recolor = bool(rng.integers(2))
    depth_transfer = bool(rng.integers(2))
    first = _make_model(rng, 2, roles, orientation, False, True)
    models = [first, _comparison(first)]
    for _ in range(num_train-2):
        n = int(rng.integers(2, 5 if depth_transfer else 4))
        role = tuple(int(c) for c in rng.choice(10, 3, replace=False)) if recolor else roles
        code = int(rng.integers(8)) if orientation_mode == "mixed" else orientation
        models.append(_make_model(rng, n, role, code, False if depth_transfer else bool(rng.integers(2))))
    puzzle, seen = {"train": [], "test": []}, set()
    for model in models:
        puzzle["train"].append(_pair(model, seen))
    rng.shuffle(puzzle["train"])
    for _ in range(num_test):
        n = int(rng.integers(2, 5)) if depth_transfer else 4
        role = tuple(int(c) for c in rng.choice(10, 3, replace=False)) if recolor else roles
        if orientation_mode == "transfer":
            code = (orientation+int(rng.integers(1, 8))) % 8
        elif orientation_mode == "mixed":
            code = int(rng.integers(8))
        else:
            code = orientation
        model = _make_model(rng, n, role, code, True)
        puzzle["test"].append(_pair(model, seen))
    return puzzle


def _hardware_signature(g, m):
    return (g.shape, m["orientation"], g[:m["base"]+1].tobytes())


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, witnesses, training, hardware = set(), set(), [], {}
    colored_clipping = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            g, m = _read(inp)
            expected = _simulate(g, m)
            assert np.array_equal(out, expected)
            assert not np.array_equal(inp, out)
            o = _unorient(out, m["orientation"])
            assert np.all(o[m["keybase"], m["xs"][0]:m["xs"][-1]+1] == m["bg"])
            for lane in m["lanes"]:
                x, a, shift = lane["x"], lane["axis"], lane["shift"]
                assert tuple(int(v) for v in o[m["top"]+1:a+1, x]) == lane["seq"][shift:]
                for r, c, v in lane["source"]:
                    assert int(o[r, c]) == v
                assert np.all(o[a, list(lane["head"])] == m["marker"])
                for r, c, v in _destination(lane):
                    assert int(o[r, c]) == v
                if split == "train" and _witness(lane):
                    witnesses.add(lane["kind"])
            if split == "train":
                training.extend(m["lanes"])
                hardware.setdefault(_hardware_signature(g, m), []).append(m["heights"])
                colored_clipping |= any(any(v != m["structure"] for v in lane["seq"][:lane["shift"]])
                                        for lane in m["lanes"])
    assert witnesses == {"H", "P"}, "Training must distinguish both true symmetry operations"
    assert any(lane["shift"] == 0 for lane in training)
    assert any(lane["shift"] > 0 for lane in training)
    assert colored_clipping, "Training must expose loss of a colored pin cell at the roof"
    assert any(len(set(profiles)) > 1 for profiles in hardware.values()), "Missing controlled key comparison"


def self_test():
    """42 seeds streamed in memory; no files, execution traces, or retained runs."""
    shapes, profiles, sequences, silhouettes, role_sets, orientations = set(), set(), set(), set(), set(), set()
    puzzle_hashes, flags, counts = set(), Counter(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                digest.update(str(inp.shape).encode())
                digest.update(inp.tobytes())
                g, m = _read(inp)
                shapes.add(inp.shape)
                profiles.add(m["heights"])
                counts.add(len(m["lanes"]))
                role_sets.add((m["bg"], m["structure"], m["marker"]))
                orientations.add(m["orientation"])
                for lane in m["lanes"]:
                    seq = lane["seq"]
                    sequences.add(seq)
                    src = lane["source"]
                    r0, c0 = min(r for r, c, v in src), min(c for r, c, v in src)
                    silhouettes.add(tuple(sorted((r-r0, c-c0) for r, c, v in src)))
                    flags["moving" if lane["shift"] else "stationary"] += 1
                    flags[lane["kind"]] += 1
                    flags["colored_clip"] += any(v != m["structure"] for v in seq[:lane["shift"]])
                    flags["retained_stripes"] += len(set(seq[lane["shift"]:-1])) > 1
                    flags["diagonal"] += any(not any(abs(r-rr)+abs(c-cc) == 1
                                                    for rr, cc, vv in src if (r, c) != (rr, cc))
                                               for r, c, v in src)
                    flags["multicolor"] += len({v for r, c, v in src}) > 1
                    flags["double_shift"] += lane["shift"] > 0 and min(r for r, c, v in _destination(lane, lane["bottom"])) - min(r for r, c, v in _destination(lane)) == 2*lane["shift"]
                    if split == "train":
                        flags["omit_A_witness"] += set(_destination(lane, lane["bottom"])) != set(_destination(lane))
                        flags["all_mirrors_witness"] += set(_destination(lane, kind="H")) != set(_destination(lane))
                        flags["omit_B_witness"] += len(_destination(lane)) > 0
                flags["tied_heights"] += len(set(m["heights"])) < len(m["heights"])
                flags["flat"] += len(set(m["heights"])) == 1
        puzzle_hashes.add(digest.hexdigest())
    assert len(puzzle_hashes) == 42
    assert len(shapes) >= 35 and len(profiles) >= 70 and len(sequences) >= 100
    assert len(silhouettes) >= 15 and len(role_sets) >= 25
    assert counts == {2, 3, 4} and orientations == set(range(8))
    assert all(flags[k] >= 10 for k in ("moving", "stationary", "H", "P", "colored_clip",
                                      "retained_stripes", "multicolor", "double_shift", "tied_heights", "flat",
                                      "omit_A_witness", "all_mirrors_witness", "omit_B_witness"))
    p, q = generate_puzzle(0), generate_puzzle(0)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    pair = p["train"][0]
    for code in range(8):
        perm = np.array([(3*i+code) % 10 for i in range(10)], np.int8)
        assert np.array_equal(solve(_orient(perm[pair["input"]], code)), _orient(perm[pair["output"]], code))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(103+a*11+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["train"][0]["output"][0, 0] = (int(bad["train"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    return {"seeds": len(HOST_SEEDS), "pairs": 5*len(HOST_SEEDS), "shapes": len(shapes),
            "profiles": len(profiles), "ordered_sequences": len(sequences),
            "silhouettes": len(silhouettes), "role_palettes": len(role_sets),
            "orientations": sorted(orientations), "pin_counts": sorted(counts), "features": dict(flags)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
