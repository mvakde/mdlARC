"""Stamp first, then collapse adjacent identical full vectors of the whole scene.

All semantics are visible in the input. Banded masks are a sampling choice,
not an extra decoding convention. No parent generator is imported or executed.
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0,549277063,304346528,32698035,34701067,17949947,244079928,
              44591385,792146077,415361023,657143104,744399159,65560941,
              179140184,922952976,816093783,518422689,810859954,617037611,
              120151036,124477662,210722888,913615137,404921037,191115158,
              110463383,286467131,28334261,795575941,893762016,917742702,
              42080847,772535421,835385961,241659649,879687239,707041775,
              545359605,778806350,766946696,622398059,798109610)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "color bounds")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _starts(a, axis):
    lines = a if axis == 0 else a.T
    return np.r_[0, 1 + np.flatnonzero(np.any(lines[1:] != lines[:-1], axis=1))]


def _collapse(a):
    return a[np.ix_(_starts(a, 0), _starts(a, 1))].copy()


def _reference_collapse(a):
    # Generation's expected values use ordinary whole-vector comparisons.
    rows = [0] + [i for i in range(1, len(a)) if list(a[i]) != list(a[i-1])]
    cols = [0] + [i for i in range(1, a.shape[1])
                  if list(a[:, i]) != list(a[:, i-1])]
    return a[np.ix_(rows, cols)].copy()


def _runs(a, axis):
    return tuple(map(int, np.diff(np.r_[_starts(a, axis), a.shape[axis]])))


def _connected(mask):
    points = set(map(tuple, np.argwhere(mask)))
    if not points:
        return False
    seen = {next(iter(points))}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                p = r + dr, c + dc
                if p in points and p not in seen:
                    seen.add(p)
                    stack.append(p)
    return seen == points


def _parse(inp):
    """Recover geometry, actual colored masks, and paired coordinate sets."""
    h, w = inp.shape
    candidates = []
    for d in range(1, w-5):
        if not np.all(inp[:, d] == inp[0, d]):
            continue
        interior = inp[1:-1, d+2:-1]
        if not interior.size or not np.all(interior == interior[0, 0]):
            continue
        bg, divider, frame = map(int, (interior[0, 0], inp[0, d], inp[0, d+1]))
        if len({bg, divider, frame}) != 3:
            continue
        if not all(inp[r, c] == frame for r, c in ((0, w-1), (h-1, d+1), (h-1, w-1))):
            continue
        stage = inp[:, :d]
        if not np.any(stage != bg) or np.any((stage == divider) | (stage == frame)):
            continue
        if not np.array_equal(inp[0, d+1:], inp[-1, d+1:]):
            continue
        if not np.array_equal(inp[:, d+1], inp[:, -1]):
            continue
        edge = np.r_[inp[0, d+1:], inp[1:-1, d+1]]
        if np.any((edge == bg) | (edge == divider)):
            continue
        candidates.append((d, bg, divider, frame))
    _require(len(candidates) == 1, "scene needs one visible divider/frame")
    d, bg, divider, frame = candidates[0]
    templates = []
    for color in np.unique(inp[:, :d]):
        color = int(color)
        if color == bg:
            continue
        cells = np.argwhere(inp[:, :d] == color)
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        shape = hi - lo + 1
        _require(bool(np.all(shape % 2 == 1)), "odd tight bounding boxes required")
        mask = inp[lo[0]:hi[0]+1, lo[1]:hi[1]+1] == color
        _require(_connected(mask), "one 8-connected template per color")
        rows = list(map(int, 1 + np.flatnonzero(inp[1:-1, d+1] == color)))
        cols = list(map(int, d+2 + np.flatnonzero(inp[0, d+2:-1] == color)))
        offsets = cells - (lo + hi)//2
        if rows and cols:
            for r in rows:
                for c in cols:
                    dest = offsets + (r, c)
                    _require(bool(np.all((dest[:, 0] >= 1) & (dest[:, 0] < h-1)
                                         & (dest[:, 1] >= d+2) & (dest[:, 1] < w-1))),
                             "stamp must fit strictly inside frame")
        templates.append(dict(color=color, cells=cells, lo=tuple(map(int, lo)),
                              mask=mask, offsets=offsets, rows=rows, cols=cols,
                              state="A" if rows and cols else "R" if rows else "C" if cols else "N"))
    templates.sort(key=lambda t: t["lo"])
    _require(len({t["lo"] for t in templates}) == len(templates), "source order must have no ties")
    return dict(d=d, bg=bg, divider=divider, frame=frame, templates=templates)


def _stamp(inp, parsed=None, *, order="source", opaque=False, erase_boxes=False, paint=True):
    p = _parse(inp) if parsed is None else parsed
    out = inp.copy()
    active = [t for t in p["templates"] if t["state"] == "A"]
    for t in active:
        if erase_boxes:
            r, c = t["lo"]
            h, w = t["mask"].shape
            out[r:r+h, c:c+w] = p["bg"]
        else:
            out[t["cells"][:, 0], t["cells"][:, 1]] = p["bg"]
    if not paint:
        return out
    if order == "reverse":
        active.reverse()
    elif order == "destination":
        active.sort(key=lambda t: (min(t["rows"]), min(t["cols"])))
    elif order == "size":
        active.sort(key=lambda t: t["mask"].size)
    claimed = np.zeros(inp.shape, dtype=bool)
    for t in active:
        offsets = t["offsets"]
        if opaque:
            offsets = np.argwhere(np.ones(t["mask"].shape, dtype=bool)) - np.array(t["mask"].shape)//2
        for r in t["rows"]:
            for c in t["cols"]:
                dest = offsets + (r, c)
                for y, x in dest:
                    if not claimed[y, x]:
                        out[y, x] = t["color"]
                        claimed[y, x] = True
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only rule, independent of seed, curriculum, and generator state."""
    inp = _grid(input_grid)
    return _collapse(_stamp(inp))


def _ring(h=7, w=7, thickness=2):
    mask = np.ones((h, w), dtype=bool)
    mask[thickness:-thickness, thickness:-thickness] = False
    return mask


def _template(mask, top, left, state="A", rows=(), cols=()):
    return dict(mask=mask.copy(), top=int(top), left=int(left), state=state,
                rows=list(rows), cols=list(cols))


def _assemble(rng, h, sw, fw, ts, absent=False):
    # Palette is fresh for each example. Black has no privileged role.
    colors = list(map(int, rng.permutation(10)))
    bg, divider, frame = colors[:3]
    inp = np.full((h, sw+1+fw), bg, dtype=np.int8)
    inp[:, sw] = divider
    inp[[0, -1], sw+1:] = frame
    inp[:, [sw+1, -1]] = frame
    ts = sorted(ts, key=lambda t: (t["top"], t["left"]))
    for t, color in zip(ts, colors[3:]):
        t["color"] = color
        bh, bw = t["mask"].shape
        view = inp[t["top"]:t["top"]+bh, t["left"]:t["left"]+bw]
        _require(bool(np.all(view[t["mask"]] == bg)), "source occupancy collision")
        view[t["mask"]] = color
        for r in t["rows"]:
            _require(inp[r, sw+1] == frame, "row coordinate collision")
            inp[r, sw+1] = inp[r, -1] = color
        for c in t["cols"]:
            _require(inp[0, sw+1+c] == frame, "column coordinate collision")
            inp[0, sw+1+c] = inp[-1, sw+1+c] = color
    if absent:
        color = colors[3+len(ts)]
        rs = [r for r in range(1, h-1) if inp[r, sw+1] == frame]
        cs = [c for c in range(1, fw-1) if inp[0, sw+1+c] == frame]
        if rs and cs:
            r, c = int(rng.choice(rs)), int(rng.choice(cs))
            inp[r, sw+1] = inp[r, -1] = color
            inp[0, sw+1+c] = inp[-1, sw+1+c] = color
    # Expected stamping uses construction records and reverse-order painting.
    result = inp.copy()
    for t in ts:
        if t["rows"] and t["cols"]:
            bh, bw = t["mask"].shape
            view = result[t["top"]:t["top"]+bh, t["left"]:t["left"]+bw]
            view[t["mask"]] = bg
    for t in reversed(ts):
        if not (t["rows"] and t["cols"]):
            continue
        bh, bw = t["mask"].shape
        for r in t["rows"]:
            for c in t["cols"]:
                view = result[r-bh//2:r+bh//2+1, sw+1+c-bw//2:sw+1+c+bw//2+1]
                view[t["mask"]] = t["color"]
    return {"input": inp, "output": _reference_collapse(result)}


def _priority_pair(rng):
    h, sw = int(rng.integers(18, 29)), int(rng.integers(7, 10))
    fw = int(rng.integers(13, 30-sw))
    bh, bw = int(rng.choice([3, 5])), int(rng.choice([3, 5]))
    top = int(rng.integers(0, h-7-bh))
    first = _template(_ring(), top, int(rng.integers(sw-6)))
    second = _template(np.ones((bh, bw), bool), top+8, int(rng.integers(sw-bw+1)))
    r, c = int(rng.integers(5, h-4)), int(rng.integers(5, fw-4))
    # Source-first ring lands below/right (or below/left) of the later solid.
    dc = int(rng.choice([-1, 1]))
    first["rows"], first["cols"] = [r], [c]
    second["rows"], second["cols"] = [r-1], [c-dc]
    return _assemble(rng, h, sw, fw, [first, second])


def _replication_pair(rng):
    h, sw = int(rng.integers(20, 31)), int(rng.integers(7, 10))
    fw = int(rng.integers(18, 30-sw))
    # Change copy spacing and slack; no global regular downscale is possible.
    r1 = int(rng.integers(4, 7))
    r2 = int(rng.integers(r1+8, h-4))
    c1 = 4
    c2 = int(rng.integers(12, fw-4))
    top, left = int(rng.integers(0, h-7)), int(rng.integers(sw-6))
    ring = _template(_ring(), top, left, rows=[r1, r2], cols=[c1, c2])
    states = list(rng.permutation(["N", "R", "C"]))
    ts = [ring]
    # Three independent templates fit in the transparent source hole.
    for i, state in enumerate(states):
        t = _template(np.ones((1, 3), bool), top+2+i, left+2, str(state))
        if state == "R":
            t["rows"] = [h-2]
        if state == "C":
            t["cols"] = [fw-2]
        ts.append(t)
    return _assemble(rng, h, sw, fw, ts, absent=bool(rng.integers(2)))


def _band_mask(rng, h, w):
    if min(h, w) == 1:
        return np.ones((h, w), dtype=bool)
    motifs = (np.ones((3, 3), bool), _ring(3, 3, 1),
              np.array([[0,1,0],[1,1,1],[0,1,0]], bool),
              np.eye(3, dtype=bool), np.array([[0,1,0],[1,0,1],[0,1,0]], bool))
    if rng.random() < .55:
        coarse = motifs[int(rng.integers(len(motifs)))].copy()
    else:
        for _ in range(32):
            coarse = rng.random((3, 3)) < float(rng.uniform(.35, .85))
            if coarse[0].any() and coarse[-1].any() and coarse[:, 0].any() and coarse[:, -1].any() and _connected(coarse):
                break
        else:
            coarse = np.ones((3, 3), bool)
    def bands(n):
        choices = [(a, b, n-a-b) for a in range(1, n-1) for b in range(1, n-a) if n-a-b > 0]
        return choices[int(rng.integers(len(choices)))]
    mask = np.repeat(np.repeat(coarse, bands(h), axis=0), bands(w), axis=1)
    if rng.integers(2):
        mask = mask[::-1]
    if rng.integers(2):
        mask = mask[:, ::-1]
    return mask.copy()


def _random_pair(rng):
    sw = int(rng.integers(5, 10))
    fw = int(rng.integers(12, 30-sw))
    count = int(rng.integers(1, 5))
    dims = [(int(rng.choice([3,5,7])), int(rng.choice([3,5] if sw < 7 else [3,5,7])))
            for _ in range(count)]
    # Some small inactive templates keep scenes within 30 rows.
    states = [str(rng.choice(["A", "N", "R", "C"], p=[.72,.10,.09,.09])) for _ in dims]
    states[int(rng.integers(count))] = "A"
    dims = [(1, int(rng.choice([1,3]))) if s != "A" else d for s, d in zip(states, dims)]
    while sum(d[0] for d in dims) + count + 1 > 30:
        i = max(range(count), key=lambda k: dims[k][0])
        dims[i] = (dims[i][0]-2, dims[i][1])
    minimum = max(15, sum(d[0] for d in dims) + count + 1)
    h = int(rng.integers(minimum, 31))
    slack = h - sum(d[0] for d in dims) - (count-1)
    gaps = rng.multinomial(slack, np.full(count+1, 1/(count+1)))
    top, ts = int(gaps[0]), []
    for i, ((bh, bw), state) in enumerate(zip(dims, states)):
        ts.append(_template(_band_mask(rng, bh, bw), top, int(rng.integers(sw-bw+1)), state))
        top += bh + 1 + int(gaps[i+1])
    for axis, length, name in ((0, h, "rows"), (1, fw, "cols")):
        used = set()
        eligible = [t for t in ts if t["state"] in ("A", "R" if axis == 0 else "C")]
        eligible.sort(key=lambda t: -t["mask"].shape[axis])
        # Reserve one center for every eligible color before any replication.
        for t in eligible:
            rad = t["mask"].shape[axis]//2 if t["state"] == "A" else 0
            free = [v for v in range(1+rad, length-1-rad) if v not in used]
            v = int(rng.choice(free))
            t[name] = [v]
            used.add(v)
        for t in eligible:
            if t["state"] != "A":
                continue
            rad = t["mask"].shape[axis]//2
            free = [v for v in range(1+rad, length-1-rad) if v not in used]
            count_extra = min(len(free), int(rng.choice([0,1,2], p=[.55,.35,.10])))
            extra = list(map(int, rng.choice(free, size=count_extra, replace=False)))
            t[name] += extra
            used.update(extra)
    return _assemble(rng, h, sw, fw, ts, absent=bool(rng.random() < .25))


def _different(a, b):
    return not np.array_equal(a, b)


def _features(inp):
    p = _parse(inp)
    stamped = _stamp(inp, p)
    out = _collapse(stamped)
    cleared = _stamp(inp, p, paint=False)
    runs = (_runs(stamped, 0), _runs(stamped, 1))
    colored_runs = []
    for axis in (0, 1):
        starts = _starts(stamped, axis)
        good = 0
        for start, length in zip(starts, runs[axis]):
            line = stamped[start, p["d"]+2:-1] if axis == 0 else stamped[1:-1, start]
            good += length > 1 and np.any((line != p["bg"]) & (line != p["frame"]) & (line != p["divider"]))
        colored_runs.append(int(good))
    repeated = []
    for lines in (out, out.T):
        repeated.append(any(np.array_equal(lines[a], lines[b])
                            for a in range(len(lines)) for b in range(a+2, len(lines))))
    def mutation(**kwargs):
        return _different(out, _collapse(_stamp(inp, p, **kwargs)))
    # Freezing the run boundaries before moving the templates is also wrong:
    # source erasure and destination painting must precede line comparison.
    premature = stamped[np.ix_(_starts(inp, 0), _starts(inp, 1))]
    return dict(shape=inp.shape, output_shape=out.shape,
                roles=(p["bg"], p["divider"], p["frame"]),
                masks=tuple((t["mask"].shape, t["mask"].tobytes()) for t in p["templates"]),
                count=len(p["templates"]), active=sum(t["state"] == "A" for t in p["templates"]),
                states="".join(t["state"] for t in p["templates"]),
                products=tuple((len(t["rows"]), len(t["cols"])) for t in p["templates"] if t["state"] == "A"),
                runs=runs, colored_runs=tuple(colored_runs), separated=tuple(repeated),
                no_a=_different(out, _collapse(inp)), no_b=_different(out, stamped),
                premature=_different(out, premature),
                boundary_dependency=any(not np.array_equal(_starts(stamped, ax), _starts(cleared, ax)) for ax in (0,1)),
                reverse=mutation(order="reverse"), destination=mutation(order="destination"),
                size=mutation(order="size"), opaque=mutation(opaque=True), erase=mutation(erase_boxes=True))


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    puzzle, seen = {"train": [], "test": []}, set()
    training_schedules = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            make = _priority_pair if split == "train" and i == 0 else _replication_pair if split == "train" and i == 1 else _random_pair
            for _ in range(128):
                pair = make(rng)
                try:
                    f = _features(pair["input"])
                except AssertionError:
                    continue
                if _key(pair["input"]) in seen or not (f["no_a"] and f["no_b"] and f["boundary_dependency"] and f["premature"]):
                    continue
                if split == "train" and i == 0:
                    if not all(f[k] for k in ("reverse", "destination", "size", "opaque")):
                        continue
                    if min(f["colored_runs"]) < 1:
                        continue
                if split == "train" and i == 1:
                    if not (f["erase"] and all(f["separated"]) and min(f["colored_runs"]) >= 1):
                        continue
                if split == "test" and f["runs"] in training_schedules:
                    continue
                break
            else:
                raise RuntimeError("bounded construction exhausted")
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
            if split == "train":
                training_schedules.add(f["runs"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "exact puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "pair lists required")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "exact pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            p = _parse(inp)
            stamped = _stamp(inp, p)
            _require(np.array_equal(out, _reference_collapse(stamped)), "wrong composed output")
            _require(np.array_equal(out, solve(inp)), "input-only solve mismatch")
            _require(len(_starts(out, 0)) == len(out) and len(_starts(out, 1)) == out.shape[1], "uncollapsed adjacent lines")
            rows, cols = _runs(stamped, 0), _runs(stamped, 1)
            reconstructed = np.repeat(np.repeat(out, rows, axis=0), cols, axis=1)
            _require(np.array_equal(reconstructed, stamped), "retained values/order changed")
            _require(set(map(int, out.flat)) == set(map(int, stamped.flat)), "collapse lost a value")
            _require(set(map(int, inp.flat)) == set(map(int, stamped.flat)), "stamping lost a labeled color")
            _require(np.array_equal(stamped[[0,-1], p["d"]+1:], inp[[0,-1], p["d"]+1:]), "stamping changed horizontal borders")
            _require(np.array_equal(stamped[:, [p["d"],p["d"]+1,-1]], inp[:, [p["d"],p["d"]+1,-1]]), "stamping changed divider/borders")
            for t in p["templates"]:
                values = stamped[t["cells"][:, 0], t["cells"][:, 1]]
                _require(bool(np.all(values == (p["bg"] if t["state"] == "A" else t["color"]))), "source erasure/preservation")
            f = _features(inp)
            _require(f["no_a"] and f["no_b"] and f["boundary_dependency"] and f["premature"], "both parents must affect answer and boundaries after stamping")
            features[split].append(f)
    train = features["train"]
    for feature in ("reverse", "destination", "size", "opaque", "erase"):
        _require(any(f[feature] for f in train), f"missing training witness: {feature}")
    _require(any((2,2) in f["products"] for f in train), "training needs Cartesian replication")
    _require(all(any(state in f["states"] for f in train) for state in "NRC"), "training needs all inactive states")
    for ax in (0, 1):
        _require(any(f["colored_runs"][ax] and len(set(f["runs"][ax])) > 1 for f in train), "need nonuniform colored equality runs")
        _require(any(f["separated"][ax] for f in train), "separated repetitions must survive")
    schedules = {f["runs"] for f in train}
    _require(all(f["runs"] not in schedules for f in features["test"]), "test needs a new equality-run schedule")
    return True


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 supplied seeds, metamorphic checks, and all legal counts."""
    shapes, masks, palettes, schedules, structures, fingerprints = set(), set(), set(), set(), set(), set()
    products, counts, backgrounds = set(), set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        _require(_signature(puzzle) == _signature(generate_puzzle(seed)), "nondeterministic puzzle")
        fingerprints.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                f = _features(inp)
                shapes.add(inp.shape)
                masks.update(f["masks"])
                palettes.add(f["roles"])
                backgrounds.add(f["roles"][0])
                schedules.add(f["runs"])
                counts.add((f["count"], f["active"]))
                products.update(f["products"])
                role_map = {}
                structures.add((inp.shape, bytes(role_map.setdefault(int(c), len(role_map)) for c in inp.flat)))
                # Bijections preserve color equality, including black templates.
                recolor = np.roll(np.arange(10, dtype=np.int8), 3)
                _require(np.array_equal(solve(recolor[inp]), recolor[out]), "palette equivariance")
                stamped = _stamp(inp)
                _require(np.array_equal(_collapse(_collapse(stamped)), out), "collapse idempotence")
                _require(np.array_equal(_collapse(stamped.T).T, out), "axis symmetry of collapse")
        # Check mutated outputs are rejected, without saving payloads.
        bad = {s: list(puzzle[s]) for s in ("train", "test")}
        bad["train"][0] = dict(puzzle["train"][0], output=_collapse(_stamp(puzzle["train"][0]["input"], order="reverse")))
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted wrong priority")
    _require(len(fingerprints) == 42, "seed diversity")
    for value, minimum, name in ((len(shapes),35,"geometry"), (len(masks),35,"masks"),
                                  (len(palettes),50,"palette roles"), (len(schedules),100,"equality schedules"),
                                  (len(structures),150,"color-normalized structures"), (len(counts),8,"eligibility/counts")):
        _require(value >= minimum, f"insufficient {name}: {value}")
    _require(backgrounds == set(range(10)), "background role coverage")
    _require({(1,1),(2,1),(1,2),(2,2),(3,1),(1,3)} <= products, "coordinate multiplicity coverage")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(123456+a*10+b, a, b)
            _require((len(p["train"]),len(p["test"])) == (a,b), "explicit counts")
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, input_shapes=len(shapes), masks=len(masks), palette_roles=len(palettes),
                equality_schedules=len(schedules), normalized_inputs=len(structures),
                eligibility_counts=len(counts), coordinate_products=sorted(products))
