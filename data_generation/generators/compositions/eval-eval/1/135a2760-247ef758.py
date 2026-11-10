"""Fault-addressed stamping: periodic error positions are colored coordinates.

Standard library + NumPy only. No parent imports, stored tasks, or generation
state enter solve(). Experiments keep at most one small puzzle in memory.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or high is not None and value > high:
        raise ValueError(f"{name} outside the supported range")
    return value


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid colors must be integers 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    seen = {next(iter(cells))}
    todo = list(seen)
    while todo:
        r, c = todo.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                p = (r + dr, c + dc)
                if p in cells and p not in seen:
                    seen.add(p)
                    todo.append(p)
    return seen == cells


def _recover_tape(tape, neutral):
    """All supported strict-majority periodic explanations must agree.

    Four observations per phase, <= floor(length/5) errors, and only a
    neutral/color exchange at each error. Redundant periods are harmless.
    """
    length = len(tape)
    candidates = []
    for period in range(1, length // 4 + 1):
        unit = np.empty(period, dtype=np.int8)
        for phase in range(period):
            values = tape[phase::period]
            counts = np.bincount(values, minlength=10)
            winner = int(np.argmax(counts))
            if counts[winner] * 2 <= len(values):
                break
            unit[phase] = winner
        else:
            repaired = unit[np.arange(length) % period]
            error = repaired != tape
            if neutral not in unit or len(np.unique(unit)) < 2:
                continue
            if int(error.sum()) > length // 5:
                continue
            if not np.all((tape[error] == neutral) ^ (repaired[error] == neutral)):
                continue
            candidates.append((period, repaired))
    if not candidates:
        raise ValueError("no supported periodic ruler")
    if any(not np.array_equal(candidates[0][1], x[1]) for x in candidates[1:]):
        raise ValueError("ambiguous periodic ruler")
    return candidates[0][1].copy(), candidates[0][0]


def _faults(tape, repaired, neutral, mode="faults"):
    result = {}
    for i, (a, b) in enumerate(zip(tape, repaired)):
        a, b = int(a), int(b)
        label = None
        if mode == "raw_marks" and a != neutral:
            label = a
        elif mode == "expected_marks" and b != neutral:
            label = b
        elif mode in ("faults", "positive_only", "deletions_only") and a != b:
            if mode == "positive_only" and a == neutral:
                continue
            if mode == "deletions_only" and b == neutral:
                continue
            label = b if a == neutral else a
        if label is not None:
            result.setdefault(label, []).append(i)
    return result


def _canonical(a):
    """Parse only the canonical orientation, deriving every role geometrically."""
    h, w = a.shape
    if h < 12:
        raise ValueError("field too short")
    found = []
    for d in range(3, min(6, w - 11)):
        divider = int(a[0, d])
        if not np.all(a[:, d] == divider):
            continue
        f = d + 1
        neutral, bg = int(a[0, f]), int(a[2, f + 1])
        if len({divider, neutral, bg}) != 3:
            continue
        if not (np.all(a[0, f:] == neutral) and np.all(a[-1, f:] == neutral)
                and np.all(a[:, f] == neutral) and np.all(a[:, -1] == neutral)):
            continue
        if not (np.all(a[1, f+1:f+3] == bg) and np.all(a[2, f+1:-1] == bg)
                and np.all(a[3:-1, f+2] == bg) and np.all(a[3:-1, f+3:-1] == bg)):
            continue
        stage = a[:, :d]
        colors = [int(c) for c in np.unique(stage) if c != bg]
        if not 1 <= len(colors) <= 4 or neutral in colors or divider in colors:
            continue
        top, left = a[1, f+3:-1], a[3:-1, f+1]
        if min(len(top), len(left)) < 8:
            continue
        if not set(map(int, np.r_[top, left])) <= {neutral, *colors}:
            continue
        try:
            clean_top, ptop = _recover_tape(top, neutral)
            clean_left, pleft = _recover_tape(left, neutral)
            rows = _faults(left, clean_left, neutral)
            cols = _faults(top, clean_top, neutral)
            templates = []
            for color in colors:
                cells = np.argwhere(stage == color)
                lo, hi = cells.min(axis=0), cells.max(axis=0)
                bh, bw = map(int, hi - lo + 1)
                mask = stage[lo[0]:hi[0]+1, lo[1]:hi[1]+1] == color
                if bh % 2 != 1 or bw % 2 != 1 or not _connected(mask):
                    raise ValueError("template requires a tight odd 8-connected mask")
                rs, cs = rows.get(color, []), cols.get(color, [])
                offsets = cells - (lo + hi) // 2
                if rs and cs:
                    for r in rs:
                        for c in cs:
                            points = offsets + [r, c]
                            if not np.all((points >= 0) & (points < [len(left), len(top)])):
                                raise ValueError("stamp would cross the field boundary")
                templates.append(dict(color=color, cells=cells, mask=mask,
                                      offsets=offsets, rows=rs, cols=cs,
                                      first=tuple(map(int, cells[0]))))
            templates.sort(key=lambda t: t["first"])
            found.append(dict(a=a, d=d, f=f, bg=bg, neutral=neutral, divider=divider,
                              top=top, left=left, clean_top=clean_top, clean_left=clean_left,
                              periods=(pleft, ptop), templates=templates))
        except ValueError:
            continue
    if len(found) != 1:
        raise ValueError("no unique canonical layout")
    return found[0]


def _locate(a):
    matches = []
    for rotation in range(4):
        try:
            p = _canonical(np.rot90(a, rotation))
            matches.append((rotation, p))
        except ValueError:
            pass
    if len(matches) != 1:
        raise ValueError("no unique oriented layout")
    return matches[0]


def _render(p, *, mode="faults", shape="mask", order="source", cartesian=True,
            repair_only=False):
    """Forward visible-input rendering; alternatives are diagnostic ablations."""
    out = p["a"].copy()
    f, bg, neutral = p["f"], p["bg"], p["neutral"]
    out[1, f+3:-1] = p["clean_top"]
    out[3:-1, f+1] = p["clean_left"]
    if repair_only:
        return out
    rows = _faults(p["left"], p["clean_left"], neutral, mode)
    cols = _faults(p["top"], p["clean_top"], neutral, mode)
    active = [t for t in p["templates"] if rows.get(t["color"]) and cols.get(t["color"])]
    for t in active:
        out[t["cells"][:, 0], t["cells"][:, 1]] = bg
    if order == "reverse":
        active = active[::-1]
    elif order == "destination":
        active = sorted(active, key=lambda t: (min(rows[t["color"]]), min(cols[t["color"]])))
    elif order == "size":
        active = sorted(active, key=lambda t: t["mask"].size)
    field = out[3:-1, f+3:-1]
    occupied = np.zeros(field.shape, dtype=bool)
    for t in active:
        color = t["color"]
        rs, cs = rows[color], cols[color]
        centers = [(r, c) for r in rs for c in cs] if cartesian else list(zip(rs, cs))
        offsets = t["offsets"]
        if shape == "centers":
            offsets = np.array([[0, 0]], dtype=int)
        elif shape == "solid":
            bh, bw = t["mask"].shape
            offsets = np.argwhere(np.ones((bh, bw), dtype=bool)) - [bh // 2, bw // 2]
        for r, c in centers:
            for dr, dc in offsets:
                y, x = r + int(dr), c + int(dc)
                # Diagnostic raw-mark hypotheses can request illegal edge copies.
                # Cropping these alternatives does not affect the valid rule.
                if 0 <= y < field.shape[0] and 0 <= x < field.shape[1] and not occupied[y, x]:
                    field[y, x] = color
                    occupied[y, x] = True
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover and execute the rule using this input alone; never mutate it."""
    rotation, p = _locate(_grid(input_grid))
    return np.rot90(_render(p), -rotation).copy()


def _sample_mask(rng, h, w):
    mask = np.zeros((h, w), dtype=bool)
    mode = int(rng.integers(7))
    if min(h, w) == 1 or mode == 0:
        mask[:] = True
    elif mode == 1:
        mask[[0, -1], :] = True
        mask[:, [0, -1]] = True
    elif mode == 2 and h == w:
        y, x = np.indices((h, w))
        mask = abs(y - h // 2) + abs(x - w // 2) == h // 2
    elif mode == 3:
        for y in range(h):
            mask[y, round(y * (w - 1) / (h - 1))] = True
        for x in range(w):
            mask[round(x * (h - 1) / (w - 1)), x] = True
    elif mode == 4:
        mask[h // 2, :] = True
        mask[:, w // 2] = True
    else:
        points = [(0, int(rng.integers(w))), (h-1, int(rng.integers(w))),
                  (int(rng.integers(h)), 0), (int(rng.integers(h)), w-1)]
        rng.shuffle(points)
        y, x = points[0]
        mask[y, x] = True
        for ty, tx in points[1:]:
            while (y, x) != (ty, tx):
                y += int(np.sign(ty-y))
                x += int(np.sign(tx-x))
                mask[y, x] = True
        target = int(rng.integers(int(mask.sum()), h*w+1))
        for _ in range(h*w):
            if int(mask.sum()) >= target:
                break
            frontier = [(y, x) for y in range(h) for x in range(w) if not mask[y, x]
                        and mask[max(0,y-1):y+2, max(0,x-1):x+2].any()]
            y, x = frontier[int(rng.integers(len(frontier)))]
            mask[y, x] = True
    if rng.random() < .5:
        mask = mask[::-1]
    if rng.random() < .5:
        mask = mask[:, ::-1]
    return mask.copy()


def _priority_masks(rng, sw):
    for _ in range(64):
        bh, bw = (3, 5) if sw >= 5 and rng.random() < .5 else (5, 3)
        a, b = _sample_mask(rng, bh, bw), _sample_mask(rng, 3, 3)
        dc = int(rng.choice([-1, 1]))
        aa = {(int(r)-bh//2+1, int(c)-bw//2+dc) for r, c in np.argwhere(a)}
        bb = {(int(r)-1, int(c)-1) for r, c in np.argwhere(b)}
        box = {(r-bh//2+1, c-bw//2+dc) for r in range(bh) for c in range(bw)}
        if aa & bb and (box-aa) & bb and aa-bb and bb-aa:
            return a, b, dc
    a = np.ones((5, 3), dtype=bool)
    a[1:-1, 1] = False
    return a, np.ones((3, 3), dtype=bool), 1


def _positions(rng, length, masks, counts, period, forced=None):
    forced = {} if forced is None else forced
    result = [[] for _ in masks]
    phases = set()
    for i, positions in forced.items():
        for pos in positions:
            if pos % period in phases:
                raise ValueError("forced coordinates share a phase")
            phases.add(pos % period)
            result[i].append(int(pos))
    requests = [i for i in range(len(masks)) for _ in range(counts[i]-len(result[i]))]
    rng.shuffle(requests)
    for i in requests:
        rad = masks[i].shape[0] // 2
        choices = [v for v in range(rad, length-rad) if v % period not in phases]
        if not choices:
            raise ValueError("coordinate packing failed")
        v = int(rng.choice(choices))
        result[i].append(v)
        phases.add(v % period)
    return [sorted(x) for x in result]


def _make_tape(rng, length, neutral, colors, positions, period, both=False):
    faults = [(pos, colors[i]) for i, ps in enumerate(positions) for pos in ps]
    for attempt in range(160):
        unit = rng.choice([neutral] + colors, size=period).astype(np.int8)
        polarities = rng.integers(0, 2, size=len(faults))
        if both:
            polarities[:2] = rng.permutation([0, 1])
        for (pos, color), polarity in zip(faults, polarities):
            unit[pos % period] = neutral if polarity else color
        if neutral not in unit or len(np.unique(unit)) < 2:
            continue
        clean = unit[np.arange(length) % period]
        noisy = clean.copy()
        for pos, color in faults:
            noisy[pos] = color if clean[pos] == neutral else neutral
        try:
            repaired, recovered_period = _recover_tape(noisy, neutral)
        except ValueError:
            continue
        if np.array_equal(clean, repaired):
            return noisy, clean, recovered_period
    raise ValueError("bounded ruler construction exhausted")


def _pair(rng, bg, kind, rotation):
    """Construct clean tapes and labels separately from the solver renderer."""
    test = kind == "test"
    for attempt in range(120):
        rlen = int(rng.integers(20, 25) if test else rng.integers(16, 19))
        clen = int(rng.integers(19, 23) if test else rng.integers(16, 19))
        sw = int(rng.integers(3, min(5, 25-clen)+1))
        palette = [int(c) for c in rng.permutation([c for c in range(10) if c != bg])]
        divider, neutral = palette[:2]
        if kind == "cartesian":
            n = 2
            active = int(rng.integers(2))
            nr = [2 if i == active else 0 for i in range(n)]
            nc = nr.copy()
        elif kind == "priority":
            n = 4
            active_ids = sorted(map(int, rng.choice(4, size=2, replace=False)))
            a, b = active_ids
            rest = [i for i in range(n) if i not in active_ids]
            nr, nc = [0]*n, [0]*n
            nr[a] = nr[b] = nc[a] = nc[b] = 1
            nr[rest[0]], nc[rest[1]] = 1, 1
        else:
            n = int(rng.integers(1, 5))
            capr, capc = rlen // 5, clen // 5
            active_ids = list(map(int, rng.choice(n, size=int(rng.integers(1, min(n,capr,capc)+1)), replace=False)))
            nr = [int(i in active_ids) for i in range(n)]
            nc = nr.copy()
            for i in rng.permutation(n):
                i = int(i)
                if i in active_ids:
                    continue
                state = int(rng.integers(3))
                if state == 1 and sum(nr) < capr:
                    nr[i] = 1
                elif state == 2 and sum(nc) < capc:
                    nc[i] = 1
            for counts, cap in ((nr, capr), (nc, capc)):
                for _ in range(cap-sum(counts)):
                    if rng.random() < .7:
                        counts[int(rng.choice(active_ids))] += 1
        colors = palette[2:2+n]
        masks = []
        budget = rlen + 4 - (n-1)
        for i in range(n):
            hs = [h for h in (1, 3, 5) if h <= budget-(n-i-1)]
            bh = int(rng.choice(hs))
            bw = int(rng.choice([1, 3] + ([5] if sw >= 5 else [])))
            if bh == bw == 1:
                bw = 3
            masks.append(_sample_mask(rng, bh, bw))
            budget -= bh
        dc = 1
        if kind == "priority":
            masks[a], masks[b], dc = _priority_masks(rng, sw)
            for i in rest:
                masks[i] = _sample_mask(rng, 1, 3)
        if sum(m.shape[0] for m in masks) + n-1 > rlen+4:
            continue
        pr_choices = list(range(max(2, sum(nr)), rlen//4+1))
        pc_choices = list(range(max(2, sum(nc)), clen//4+1))
        if not pr_choices or not pc_choices:
            continue
        pr, pc = int(rng.choice(pr_choices)), int(rng.choice(pc_choices))
        fr, fc = {}, {}
        if kind == "priority":
            rr = int(rng.integers(3, rlen-4))
            cc = int(rng.integers(3, clen-4))
            fr = {a: [rr+1], b: [rr]}
            fc = {a: [cc+dc], b: [cc]}
        try:
            rows = _positions(rng, rlen, masks, nr, pr, fr)
            cols = _positions(rng, clen, [m.T for m in masks], nc, pc, fc)
            noisy_r, clean_r, recovered_r = _make_tape(rng, rlen, neutral, colors, rows, pr, kind == "cartesian")
            noisy_c, clean_c, recovered_c = _make_tape(rng, clen, neutral, colors, cols, pc, kind == "cartesian")
            if kind == "priority" and recovered_r == recovered_c:
                continue
        except ValueError:
            continue
        h, w = rlen+4, sw+clen+5
        f = sw+1
        inp = np.full((h, w), bg, dtype=np.int8)
        inp[:, sw] = divider
        inp[[0, -1], f:] = neutral
        inp[:, [f, w-1]] = neutral
        inp[1, f+3:-1] = noisy_c
        inp[3:-1, f+1] = noisy_r
        slack = h - sum(m.shape[0] for m in masks) - (n-1)
        gaps = rng.multinomial(slack, np.full(n+1, 1/(n+1)))
        source_cells = []
        top = int(gaps[0])
        for i, (mask, color) in enumerate(zip(masks, colors)):
            bh, bw = mask.shape
            left = int(rng.integers(sw-bw+1))
            cells = np.argwhere(mask) + [top, left]
            inp[cells[:, 0], cells[:, 1]] = color
            source_cells.append(cells)
            top += bh + 1 + int(gaps[i+1])
        out = inp.copy()
        out[1, f+3:-1] = clean_c
        out[3:-1, f+1] = clean_r
        for i in range(n):
            if rows[i] and cols[i]:
                cells = source_cells[i]
                out[cells[:, 0], cells[:, 1]] = bg
        # Ground truth is reverse-order painting of construction masks, whereas
        # solve uses parsed color cells and a forward occupancy bitmap.
        for i in reversed(range(n)):
            if not rows[i] or not cols[i]:
                continue
            bh, bw = masks[i].shape
            for r in rows[i]:
                for c in cols[i]:
                    view = out[3+r-bh//2:3+r+bh//2+1, f+3+c-bw//2:f+3+c+bw//2+1]
                    view[masks[i]] = colors[i]
        # Witnesses must distinguish actual rendered answers, including any
        # coincidental merging of copies, rather than just requested features.
        if kind in ("cartesian", "priority"):
            try:
                parsed = _canonical(inp)
                if not np.array_equal(_render(parsed), out):
                    continue
                fcheck = _features(parsed)["differences"]
                required = (("raw_marks", "expected_marks", "positive_only", "deletions_only", "zip", "centers")
                            if kind == "cartesian" else ("solid", "reverse", "destination", "size"))
                if not all(fcheck[k] for k in required):
                    continue
            except ValueError:
                continue
        return {"input": np.rot90(inp, rotation).copy(), "output": np.rot90(out, rotation).copy()}
    raise RuntimeError("bounded pair construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    ne = _integer(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    backgrounds = list(map(int, rng.permutation(10)))
    base_rotation = int(rng.integers(4))
    puzzle = {"train": [], "test": []}
    seen = set()
    for serial in range(nt+ne):
        split = "train" if serial < nt else "test"
        kind = ("cartesian" if serial == 0 else "priority" if serial == 1 else "random") if split == "train" else "test"
        # Minimal demonstrations expose both ruler orientations; test rotation
        # recombines those geometric roles, with no changed anchoring convention.
        rotation = (base_rotation + serial) % 4 if serial < 2 else int(rng.integers(4))
        pair = _pair(rng, backgrounds[serial % 10], kind, rotation)
        original = {k: v.copy() for k, v in pair.items()}
        # Twenty distinct bijections exceed the maximum thirteen prior inputs.
        for variant in range(20):
            mapping = np.arange(10, dtype=np.int8)
            if variant >= 10:
                x = backgrounds[serial % 10]
                y = (x+1) % 10
                mapping[x], mapping[y] = mapping[y], mapping[x]
            mapping = (mapping + variant % 10) % 10
            pair = {k: mapping[v] for k, v in original.items()}
            if _key(pair["input"]) not in seen:
                break
        else:
            raise RuntimeError("distinct-input bijections exhausted")
        seen.add(_key(pair["input"]))
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(p):
    ts = p["templates"]
    out = _render(p)
    modes = ("raw_marks", "expected_marks", "positive_only", "deletions_only")
    differences = {mode: int(np.count_nonzero(out != _render(p, mode=mode))) for mode in modes}
    differences.update({"zip": int(np.count_nonzero(out != _render(p, cartesian=False))),
                        "centers": int(np.count_nonzero(out != _render(p, shape="centers"))),
                        "solid": int(np.count_nonzero(out != _render(p, shape="solid"))),
                        "reverse": int(np.count_nonzero(out != _render(p, order="reverse"))),
                        "destination": int(np.count_nonzero(out != _render(p, order="destination"))),
                        "size": int(np.count_nonzero(out != _render(p, order="size"))),
                        "repair_only": int(np.count_nonzero(out != _render(p, repair_only=True)))})
    return dict(differences=differences, count=len(ts), periods=p["periods"],
                lengths=(len(p["left"]), len(p["top"])),
                products=[(len(t["rows"]), len(t["cols"])) for t in ts],
                states="".join("A" if t["rows"] and t["cols"] else "R" if t["rows"] else "C" if t["cols"] else "N" for t in ts))


def validate_generated_puzzle(puzzle):
    """Check returned arrays, semantic invariants, and distinguishing evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            rotation, p = _locate(inp)
            canonical_out = np.rot90(out, rotation)
            assert np.array_equal(_render(p), canonical_out), "wrong periodic repair, address, mask, or preservation"
            assert np.array_equal(solve(inp), out)
            # Changed cells can occur only on rulers, moved source pixels, and
            # occupied destination mask cells. Neutral framing/padding survives.
            protected = np.ones(inp.shape if rotation % 2 == 0 else inp.T.shape, dtype=bool)
            f = p["f"]
            protected[1, f+3:-1] = False
            protected[3:-1, f+1] = False
            protected[3:-1, f+3:-1] = False
            for t in p["templates"]:
                active = bool(t["rows"] and t["cols"])
                if active:
                    protected[t["cells"][:, 0], t["cells"][:, 1]] = False
                    assert np.all(canonical_out[t["cells"][:, 0], t["cells"][:, 1]] == p["bg"])
            assert np.array_equal(p["a"][protected], canonical_out[protected])
            for tape, clean in ((p["left"], p["clean_left"]), (p["top"], p["clean_top"])):
                changed = tape != clean
                assert 1 <= int(changed.sum()) <= len(tape)//5
                assert np.all((tape[changed] == p["neutral"]) ^ (clean[changed] == p["neutral"]))
            info = _features(p)
            infos[split].append(info)
            assert info["differences"]["repair_only"] > 0
    train = infos["train"]
    for shortcut in ("raw_marks", "expected_marks", "positive_only", "deletions_only", "zip", "centers", "solid", "reverse", "destination", "size"):
        assert any(f["differences"][shortcut] for f in train), f"missing training witness against {shortcut}"
    assert set("NRC") <= set("".join(f["states"] for f in train))
    assert any(any(a >= 2 and b >= 2 for a, b in f["products"]) for f in train)
    assert any(f["periods"][0] != f["periods"][1] for f in train)
    # The split shift is geometric length extrapolation on both rulers, measured
    # after orientation normalization, not a hidden semantic mode.
    for axis in (0, 1):
        assert min(f["lengths"][axis] for f in infos["test"]) > max(f["lengths"][axis] for f in train)


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed coverage, metamorphic checks, and wrong-answer rejection."""
    signatures, dimensions, structures, masks, periods, products, rotations = (set() for _ in range(7))
    counts, backgrounds, states = set(), set(), set()
    mutation_checks = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        assert _digest(puzzle) == _digest(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                rotation, p = _locate(inp)
                rotations.add(rotation)
                dimensions.add(inp.shape)
                counts.add(len(p["templates"]))
                backgrounds.add(p["bg"])
                periods.update(p["periods"])
                f = _features(p)
                states.update(f["states"])
                products.update(m for m in f["products"] if min(m) > 0)
                roles = {}
                structures.add(bytes(roles.setdefault(int(c), len(roles)) for c in inp.flat))
                for t in p["templates"]:
                    masks.add((t["mask"].shape, t["mask"].tobytes()))
        # Bijections and rotations may change every literal role but no rule.
        pair = puzzle["test"][0]
        mapping = np.random.default_rng(seed+1000).permutation(10).astype(np.int8)
        assert np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]])
        assert np.array_equal(solve(np.rot90(pair["input"])), np.rot90(pair["output"]))
        if seed == 0:
            rotation, p = _locate(puzzle["train"][0]["input"])
            for kwargs in ({"mode": "raw_marks"}, {"shape": "centers"}, {"cartesian": False}):
                bad = {s: [dict(x) for x in puzzle[s]] for s in ("train", "test")}
                bad["train"][0]["output"] = np.rot90(_render(p, **kwargs), -rotation)
                try:
                    validate_generated_puzzle(bad)
                except AssertionError:
                    mutation_checks += 1
                else:
                    raise AssertionError("validator accepted a shortcut output")
    assert len(signatures) == 42 and len(dimensions) >= 30 and len(structures) >= 180
    assert len(masks) >= 35 and {2, 3, 4} <= periods
    assert counts == {1, 2, 3, 4} and backgrounds == set(range(10))
    assert rotations == set(range(4)) and states == set("ANRC")
    assert {(1, 1), (2, 1), (1, 2), (2, 2)} <= products
    assert mutation_checks == 3
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(73, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "dimensions": len(dimensions),
            "normalized_structures": len(structures), "masks": len(masks),
            "periods": sorted(periods), "products": sorted(products),
            "mutation_checks": mutation_checks, "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
