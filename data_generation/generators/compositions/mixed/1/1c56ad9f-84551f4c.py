"""Rigid dominoes on a repeatedly shearing carrier. No parent imports or I/O."""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grids required")
    _require(np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=False)


def _offset(d):
    return (0, -1, 0, 1)[d % 4]


def _parse(value):
    """All roles except the fixed blue initiator are identified geometrically."""
    a = _grid(value)
    bg = int(a[0, 0])
    _require(bg != 1, "blue is reserved for initiators")
    occupied = np.argwhere(a != bg)
    _require(len(occupied) > 0, "empty scene")
    bottom = int(occupied[:, 0].max())
    colors = np.unique(a[bottom][a[bottom] != bg])
    _require(len(colors) == 1 and int(colors[0]) != 1, "missing carrier bottom bar")
    carrier = int(colors[0])
    mask = a == carrier
    rr, cc = np.where(mask)
    top, left, right = int(rr.min()), int(cc.min()), int(cc.max())
    height = bottom - top + 1
    _require(height >= 6 and height % 4 == 2, "carrier top must have left-moving phase")
    _require(top >= 3 and left >= 1 and right < a.shape[1] - 1, "missing margins")
    _require(np.all(mask[top, left:right + 1]) and np.all(mask[bottom, left:right + 1]),
             "carrier needs complete top and bottom bars")
    core = mask[top + 1:bottom, left:right + 1]
    rails = np.all(core, axis=0)
    _require(rails[0] and rails[-1] and np.sum(~rails) >= 1, "missing open supports")
    _require(sum(np.array_equal(row, rails) for row in core) >= 2, "two bare rail rows required")
    starts = np.flatnonzero(rails & ~np.r_[False, rails[:-1]])
    ends = np.flatnonzero(rails & ~np.r_[rails[1:], False])
    supports = tuple((int(x), int(y)) for x, y in zip(starts, ends))
    _require(len(supports) >= 2, "carrier needs separate supports")
    bars = []
    for r, row in enumerate(core, top + 1):
        _require(np.all(row[rails]), "interrupted support")
        for (_, e), (s, _) in zip(supports, supports[1:]):
            piece = row[e + 1:s]
            _require(np.all(piece) or not np.any(piece), "broken crossbar")
        if not np.array_equal(row, rails):
            bars.append(r)
    _require(np.all((a[top:bottom + 1] == bg) | mask[top:bottom + 1]), "foreign carrier cells")
    rods = []
    rod_colors = set()
    baseline = top - 1
    for x in range(a.shape[1]):
        rows = np.flatnonzero(a[:top, x] != bg)
        if len(rows):
            h = len(rows)
            color = int(a[rows[0], x])
            _require(h >= 2 and np.array_equal(rows, np.arange(top - h, top)), "invalid rod")
            _require(color != carrier and np.all(a[rows, x] == color), "nonuniform rod")
            _require(left <= x <= right and x + h <= a.shape[1], "unmounted or clipped rod")
            rods.append((x, h, color))
            rod_colors.add(color)
    _require(rods and 1 in rod_colors and len(rod_colors - {1}) <= 1, "invalid rod roles")
    for (x, h, _), (y, _, _) in zip(rods, rods[1:]):
        _require(y >= x + h, "rod inside another rod's landing interval")
    return dict(grid=a, bg=bg, carrier=carrier, mask=mask, top=top, bottom=bottom,
                left=left, right=right, height=height, baseline=baseline, rods=rods,
                supports=supports, bars=tuple(bars), padding=a.shape[0] - bottom - 1)


def _dynamics(rods, *, step=-1, recursive=True, wave=True, wave_limit=None):
    """Finish each contact closure before one carrier step. No saved grid traces."""
    xs = [x for x, _, _ in rods]
    fallen = {}
    pending = {i for i, (_, _, color) in enumerate(rods) if color == 1}
    batches = []
    waves = 0
    while pending:
        this_batch = []
        while pending:
            i = min(pending)
            pending.remove(i)
            if i in fallen:
                continue
            fallen[i] = xs[i]
            this_batch.append(i)
            if recursive:
                end = xs[i] + rods[i][1]
                pending.update(j for j in range(len(rods)) if j not in fallen and xs[j] == end)
        batches.append(tuple(this_batch))
        if not wave or (wave_limit is not None and waves >= wave_limit):
            break
        waves += 1
        for i in range(len(rods)):
            if i not in fallen:
                xs[i] += step
        ends = {x + rods[i][1] for i, x in fallen.items()}
        pending = {i for i, x in enumerate(xs) if i not in fallen and x in ends}
        _require(waves <= len(rods), "nonterminating dynamics")
    return dict(xs=xs, fallen=fallen, waves=waves, batches=tuple(batches))


def _render(info, state, *, anchor="bottom"):
    a = info["grid"]
    out = np.full_like(a, info["bg"])
    for r in range(info["top"], info["bottom"] + 1):
        d = info["bottom"] - r
        if anchor == "canvas":
            d = a.shape[0] - 1 - r
        elif anchor == "top":
            d = r - info["top"]
        cols = np.flatnonzero(info["mask"][r]) + state["waves"] * _offset(d)
        _require(np.all((cols >= 0) & (cols < a.shape[1])), "carrier wave clipped")
        out[r, cols] = info["carrier"]
    b = info["baseline"]
    for i, (_, h, color) in enumerate(info["rods"]):
        x = state["fallen"].get(i, state["xs"][i])
        _require(x >= 0 and x + h <= a.shape[1], "rod clipped")
        if i in state["fallen"]:
            out[b, x:x + h] = color
        else:
            out[b - h + 1:b + 1, x] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    return _render(info, _dynamics(info["rods"]))


def _sample_rods(rng, kind):
    if kind == "resume":
        n = int(rng.integers(4, 7))
        hs = rng.integers(2, 6, size=n).tolist()
        gaps = [0, 1, int(rng.integers(2, 5))] + rng.integers(0, 3, size=n - 4).tolist()
        roles = [True] + [False] * (n - 1)
        # Sometimes make the first rod a preserved passive before the initiator.
        if n >= 5 and rng.random() < 0.35:
            roles[0], roles[1] = False, True
            gaps[1:4] = [0, 1, int(rng.integers(2, 4))]
    elif kind == "sustain":
        n = 6
        hs = rng.integers(2, 4, size=n).tolist()
        gaps = [2, 2, int(rng.integers(0, 2)), 1, 1]
        roles = [True, False, False, True, False, False]
    elif kind == "deep":
        n = int(rng.integers(5, 8))
        hs = rng.integers(2, 5, size=n).tolist()
        gaps = [1] * int(rng.integers(3, min(4, n - 1) + 1))
        gaps += [0] * (n - 1 - len(gaps))
        rng.shuffle(gaps)
        roles = [True] + [False] * (n - 1)
        if rng.random() < 0.25:
            roles[-1] = True
        if n >= 6 and rng.random() < 0.22:
            roles[0], roles[1] = False, True
    else:
        n = int(rng.integers(2, 8))
        hs = rng.integers(2, 7, size=n).tolist()
        gaps = rng.choice((0, 1, 2, 3), size=n - 1, p=(0.35, 0.35, 0.2, 0.1)).tolist()
        roles = (rng.random(n) < 0.3).tolist()
        roles[int(rng.integers(n))] = True
        if all(roles):
            roles[int(rng.integers(n))] = False
    columns = [0]
    for h, gap in zip(hs, gaps):
        columns.append(columns[-1] + h + gap)
    return [(x, int(h), 1 if role else -1) for x, h, role in zip(columns, hs, roles)]


def _pair(rng, palette, kind, bottom_padding=None):
    bg, carrier, passive = palette
    for _ in range(600):
        raw = _sample_rods(rng, kind)
        state = _dynamics(raw)
        k = state["waves"]
        if kind == "resume" and (k != 2 or len(state["batches"][0]) < 2):
            continue
        if kind == "sustain" and k != 3:
            continue
        if kind == "deep" and k not in (4, 5):
            continue
        if kind == "general" and k > 3:
            continue
        extent = raw[-1][0] + raw[-1][1]
        if extent + 2 * k + 2 > 30:
            continue
        max_extra = 30 - extent - 2 * k - 2
        extra_left = int(rng.integers(0, min(2, max_extra) + 1))
        extra_right = int(rng.integers(0, min(2, max_extra - extra_left) + 1))
        frame_width = extent + extra_left + extra_right
        spare = 30 - frame_width - 2 * k - 2
        lp = k + 1 + int(rng.integers(0, spare + 1))
        rp = k + 1 + int(rng.integers(0, spare - (lp - k - 1) + 1))
        max_h = max(h for _, h, _ in raw)
        pad = int(rng.integers(0, 4)) if bottom_padding is None else bottom_padding
        allowed = [h for h in (6, 10, 14, 18, 22) if h + max_h + pad + 1 <= 30]
        frame_height = int(rng.choice(allowed))
        head = int(rng.integers(1, min(4, 30 - frame_height - max_h - pad) + 1))
        top = max_h + head
        bottom = top + frame_height - 1
        a = np.full((bottom + pad + 1, lp + frame_width + rp), bg, dtype=np.int8)
        left, right = lp, lp + frame_width - 1
        a[top, left:right + 1] = carrier
        a[bottom, left:right + 1] = carrier
        rail_mask = np.zeros(frame_width, dtype=bool)
        rail_mask[0] = rail_mask[-1] = True
        # Thin or thick, unequally spaced supports, with visible open gaps.
        for _ in range(int(rng.integers(0, 5)) if frame_width > 4 else 0):
            start = int(rng.integers(2, frame_width - 2))
            thick = int(rng.integers(1, 3))
            end = min(frame_width - 2, start + thick)
            if not np.any(rail_mask[start - 1:end + 1]):
                rail_mask[start:end] = True
        a[top:bottom + 1, left:right + 1][:, rail_mask] = carrier
        starts = np.flatnonzero(rail_mask & ~np.r_[False, rail_mask[:-1]])
        ends = np.flatnonzero(rail_mask & ~np.r_[rail_mask[1:], False])
        bare = set(rng.choice(np.arange(top + 1, bottom), size=2, replace=False).tolist())
        density = float(rng.uniform(0.0, 0.55))
        for row in range(top + 1, bottom):
            if row not in bare and rng.random() < density:
                first = int(rng.integers(0, len(starts) - 1))
                last = int(rng.integers(first + 1, len(starts)))
                a[row, left + ends[first]:left + starts[last] + 1] = carrier
        for x, h, color in raw:
            x += left + extra_left
            a[top - h:top, x] = 1 if color == 1 else passive
        info = _parse(a)
        return {"input": a, "output": _render(info, _dynamics(info["rods"]))}
    raise RuntimeError("bounded geometry sampler exhausted")


def _count(n, name, low, high):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, numbers.Integral) or not low <= n <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(n)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, "num_train", 2, 10)
    ntest = _count(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = tuple(int(c) for c in rng.choice([c for c in range(10) if c != 1], 3, replace=False))
    puzzle = {"train": [], "test": []}
    seen = set()
    plans = [("train", "resume", int(rng.choice((0, 4)))), ("train", "sustain", 1)]
    plans += [("train", "general", None)] * (ntrain - 2)
    plans += [("test", "deep", None)] * ntest
    for split, kind, pad in plans:
        for _ in range(24):
            pair = _pair(rng, palette, kind, pad)
            key = _key(pair["input"])
            if key not in seen:
                puzzle[split].append(pair)
                seen.add(key)
                break
        else:
            raise RuntimeError("bounded uniqueness sampler exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid counts")
    seen, palettes = set(), set()
    stats = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            info = _parse(inp)
            state = _dynamics(info["rods"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            palettes.add((info["bg"], info["carrier"], tuple(sorted({c for _, _, c in info["rods"]}))))
            _require(np.array_equal(out, _render(info, state)), "wrong dynamics output")
            _require(inp.shape == out.shape and not np.array_equal(inp, out), "unchanged or resized pair")
            # Conservation checks catch silent overwriting between distinct rods.
            for color in np.unique(inp):
                _require(np.sum(inp == color) == np.sum(out == color), "color mass not conserved")
            for i, (_, h, color) in enumerate(info["rods"]):
                x = state["fallen"].get(i, state["xs"][i])
                if i not in state["fallen"]:
                    _require(np.all(out[info["top"] - h:info["top"], x] == color), "upright rod deformed")
            _require(state["waves"] == len(state["batches"]), "missing final wave")
            _require(state["waves"] <= len(state["fallen"]), "a wave has no causal cascade")
            endpoints = {x + info["rods"][i][1] for i, x in state["fallen"].items()}
            _require(all(x not in endpoints for i, x in enumerate(state["xs"]) if i not in state["fallen"]),
                     "stopped before exact-contact closure")
            # A removed means ordinary domino closure: no carrier shift, no resumed contacts.
            no_a = _dynamics(info["rods"], wave=False)
            b_effect = len(state["fallen"]) > len(no_a["fallen"])
            stats[split].append((info, state, b_effect))
    _require(len(palettes) == 1, "puzzle color conventions changed")
    train = stats["train"]
    _require(any(s["waves"] == 2 and len(s["batches"][0]) >= 2 and effect for _, s, effect in train),
             "missing recursive cascade and post-wave resumption demonstration")
    _require(any(s["waves"] == 3 and len(s["batches"][0]) >= 2 for _, s, _ in train),
             "missing multiple-trigger sustained-wave demonstration")
    _require(any(len(s["fallen"]) < len(i["rods"]) for i, s, _ in train), "no preserved passive demonstrated")
    _require(len({i["padding"] % 4 for i, _, _ in train}) >= 2, "canvas anchor ambiguous")
    _require(all(s["waves"] <= 3 for _, s, _ in train), "training depth shift lost")
    _require(all(s["waves"] >= 4 and effect for _, s, effect in stats["test"]), "test must extrapolate iterations")
    # Both parent ablations alter rod decisions, not only decorative scaffold pixels.
    _require(any(effect for _, _, effect in train), "A never changes toppling decisions")
    _require(any(any(c != 1 and j in s["fallen"] for j, (_, _, c) in enumerate(i["rods"]))
                 for i, s, _ in train), "B never recursively activates a passive")
    for info, state, _ in train:
        _require(not np.array_equal(_render(info, state, anchor="top"), _render(info, state)),
                 "top-anchored wave matches")
    _require(any(not np.array_equal(_render(i, s, anchor="canvas"), _render(i, s)) for i, s, _ in train),
             "canvas-anchored wave matches")


def self_test():
    """42 streamed seeds, compact summaries only; no file creation or grid retention."""
    shapes, geometries, palettes, signatures = set(), set(), set(), set()
    rod_counts, heights, depths, support_counts, bar_counts = set(), set(), set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        q = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for a, b in zip(p[split], q[split]):
                for role in ("input", "output"):
                    _require(_key(a[role]) == _key(b[role]), "seed not reproducible")
                    digest.update(a[role].tobytes())
                info = _parse(a["input"])
                state = _dynamics(info["rods"])
                before = a["input"].copy()
                _require(np.array_equal(solve(before), a["output"]) and np.array_equal(before, a["input"]),
                         "solver changed input or disagreed")
                shapes.add(a["input"].shape)
                palettes.add((info["bg"], info["carrier"]))
                rod_counts.add(len(info["rods"]))
                heights.update(h for _, h, _ in info["rods"])
                depths.add(state["waves"])
                support_counts.add(len(info["supports"]))
                bar_counts.add(len(info["bars"]))
                # Translation- and palette-free structural identity, not just canvas identity.
                rods = tuple((x - info["rods"][0][0], h, c == 1) for x, h, c in info["rods"])
                geometries.add((info["height"], info["right"] - info["left"], info["supports"],
                                tuple(r - info["top"] for r in info["bars"]), rods))
        signatures.add(digest.digest())
    _require(len(signatures) == 42 and len(geometries) >= 150, "insufficient structural variation")
    _require(len(shapes) >= 35 and len(palettes) >= 25, "insufficient shape or palette variation")
    _require(len(rod_counts) >= 5 and heights >= {2, 3, 4, 5, 6}, "rod scale variation missing")
    _require(depths >= {1, 2, 3, 4, 5} and len(support_counts) >= 3 and len(bar_counts) >= 5,
             "iteration or scaffold variation missing")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            generate_puzzle(693573993, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures), "geometries": len(geometries),
            "canvas_shapes": len(shapes), "palette_assignments": len(palettes), "rod_counts": sorted(rod_counts),
            "rod_heights": sorted(heights), "wave_counts": sorted(depths), "support_counts": sorted(support_counts)}
