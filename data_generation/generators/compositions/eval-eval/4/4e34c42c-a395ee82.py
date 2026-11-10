"""Registered stencil parity: landmark registration and anchored substitution.

Only standard-library modules and NumPy are used. Frames group sparse charts;
exact repeated 2x2 landmarks fix translations. Each physical command observation
applies one foreground-color swap to the corresponding template copy.
"""
from collections import Counter, deque
import json

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _groups(points):
    rest = set(points)
    groups = []
    while rest:
        p = min(rest)
        rest.remove(p)
        group, todo = {p}, [p]
        while todo:
            r, c = todo.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in rest:
                    rest.remove(q)
                    group.add(q)
                    todo.append(q)
        groups.append(group)
    return groups


def _box(points):
    return (min(r for r, c in points), min(c for r, c in points),
            max(r for r, c in points), max(c for r, c in points))


def _gap(a, b):
    return (a[2]+1 < b[0] or b[2]+1 < a[0]
            or a[3]+1 < b[1] or b[3]+1 < a[1])


def _pick(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _decode(input_grid):
    """All roles, charts, offsets and vote counts come from the input pixels."""
    inp = _grid(input_grid)
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 4, "four local color roles required"
    assert counts.max()*2 > inp.size, "background must occupy over half the grid"
    bg = int(values[counts.argmax()])
    # The frame role is the unique color whose components are hollow rectangles.
    candidates = []
    for color in map(int, values):
        if color == bg:
            continue
        groups = _groups(map(tuple, np.argwhere(inp == color)))
        boxes = []
        for group in groups:
            t, l, b, r = _box(group)
            border = {(y, x) for y in range(t, b+1) for x in range(l, r+1)
                      if y in (t, b) or x in (l, r)}
            if (b-t < 3 or r-l < 3 or (b-t) % 3 or (r-l) % 3
                    or group != border):
                break
            boxes.append((t, l, b, r))
        else:
            if 2 <= len(boxes) <= 4:
                candidates.append((color, sorted(boxes)))
    assert len(candidates) == 1, "ambiguous chart frames"
    frame, boxes = candidates[0]
    assert all(_gap(a, b) for i, a in enumerate(boxes) for b in boxes[i+1:])
    outside = inp != bg
    for t, l, b, r in boxes:
        outside[t:b+1, l:r+1] = False
    glyph_points = set(map(tuple, np.argwhere(outside)))
    assert glyph_points and len(_groups(glyph_points)) == 1
    ink_colors = {int(inp[p]) for p in glyph_points}
    assert len(ink_colors) == 1
    ink = ink_colors.pop()
    assert ink not in (bg, frame)
    marker = next(int(c) for c in values if c not in (bg, frame, ink))
    tr, tc, br, bc = _box(glyph_points)
    assert all(_gap((tr, tc, br, bc), box) for box in boxes)
    mask = inp[tr:br+1, tc:bc+1] == ink
    h, w = mask.shape
    assert 2 <= h <= 5 and 2 <= w <= 5 and int(mask.sum()) >= 3
    charts = []
    for t, l, b, r in boxes:
        mh, mw = (b-t)//3, (r-l)//3
        commands, landmarks = {}, {}
        expected = np.full((b-t+1, r-l+1), bg, dtype=np.int8)
        expected[[0, -1], :] = frame
        expected[:, [0, -1]] = frame
        for y in range(mh):
            for x in range(mw):
                rr, cc = t+1+3*y, l+1+3*x
                patch = inp[rr:rr+2, cc:cc+2]
                occupied = patch != bg
                if not occupied.any():
                    continue
                if occupied.sum() == 1 and occupied[0, 0]:
                    color = int(patch[0, 0])
                    assert color in (ink, marker)
                    commands[y, x] = color
                else:
                    assert occupied.all() and set(map(int, patch.flat)) == {ink, marker}
                    signature = tuple(map(int, patch.flat))
                    assert signature not in landmarks, "landmark repeats within a chart"
                    landmarks[signature] = (y, x)
                expected[1+3*y:3+3*y, 1+3*x:3+3*x] = patch
        assert np.array_equal(expected, inp[t:b+1, l:r+1]), "unrecognized chart pixels"
        assert landmarks and commands
        points = list(commands) + list(landmarks.values())
        assert _box(points) == (0, 0, mh-1, mw-1), "chart has padded lattice bands"
        charts.append({"commands": commands, "landmarks": landmarks, "shape": (mh, mw)})
    offsets = {0: (0, 0)}
    todo = deque([0])
    edge_count = 0
    adjacency = [[] for _ in charts]
    for i, a in enumerate(charts):
        for j in range(i+1, len(charts)):
            b = charts[j]
            shared = set(a["landmarks"]) & set(b["landmarks"])
            if not shared:
                continue
            deltas = {(a["landmarks"][s][0]-b["landmarks"][s][0],
                       a["landmarks"][s][1]-b["landmarks"][s][1]) for s in shared}
            assert len(deltas) == 1, "inconsistent landmark registrations"
            dr, dc = deltas.pop()
            adjacency[i].append((j, (dr, dc)))
            adjacency[j].append((i, (-dr, -dc)))
            edge_count += 1
    while todo:
        i = todo.popleft()
        for j, delta in adjacency[i]:
            proposed = (offsets[i][0]+delta[0], offsets[i][1]+delta[1])
            if j in offsets:
                assert offsets[j] == proposed, "inconsistent registration cycle"
            else:
                offsets[j] = proposed
                todo.append(j)
    assert len(offsets) == len(charts), "disconnected landmark graph"
    votes, colors, identities, locations = Counter(), {}, {}, {}
    for i, chart in enumerate(charts):
        dr, dc = offsets[i]
        for signature, (r, c) in chart["landmarks"].items():
            p = r+dr, c+dc
            assert signature not in identities or identities[signature] == p
            assert p not in locations or locations[p] == signature
            identities[signature], locations[p] = p, signature
        for (r, c), color in chart["commands"].items():
            p = r+dr, c+dc
            assert p not in colors or colors[p] == color, "conflicting command roles"
            colors[p] = color
            votes[p] += 1
    assert not (set(locations) & set(colors)), "landmark/command collision"
    anchors = [p for p in colors if colors[p] == ink]
    assert len(anchors) == 1, "need one distinct exceptional command"
    anchor = anchors[0]
    for r, c in colors:
        rr, cc = tr+(r-anchor[0])*h, tc+(c-anchor[1])*w
        assert 0 <= rr <= inp.shape[0]-h and 0 <= cc <= inp.shape[1]-w, "expansion clipped"
    return {"bg": bg, "ink": ink, "marker": marker, "frame": frame,
            "mask": mask, "template_top": (tr, tc), "anchor": anchor,
            "votes": votes, "colors": colors, "charts": charts,
            "offsets": offsets, "landmarks": locations, "edges": edge_count}


def _paint(shape, meta, ignore_parity=False, pitch=None, fill=False):
    out = np.full(shape, meta["bg"], dtype=np.int8)
    mask = np.ones_like(meta["mask"]) if fill else meta["mask"]
    h, w = mask.shape if pitch is None else pitch
    mr, mc = np.nonzero(mask)
    tr, tc = meta["template_top"]
    ar, ac = meta["anchor"]
    for (r, c), count in meta["votes"].items():
        color = meta["colors"][r, c]
        if count % 2 or ignore_parity:
            color = meta["marker"] if color == meta["ink"] else meta["ink"]
        out[tr+(r-ar)*h+mr, tc+(c-ac)*w+mc] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    return _paint(inp.shape, _decode(inp))


def _registration_needed(meta):
    """Compare registered commands with the concrete zero-translation shortcut."""
    raw, roles, anchors = Counter(), {}, set()
    for chart in meta["charts"]:
        for p, color in chart["commands"].items():
            raw[p] += 1
            if p in roles and roles[p] != color:
                return True
            roles[p] = color
            if color == meta["ink"]:
                anchors.add(p)
    if len(anchors) != 1:
        return True
    def relative(votes, colors, anchor):
        return sorted((r-anchor[0], c-anchor[1], count, colors[r, c])
                      for (r, c), count in votes.items())
    return relative(raw, roles, next(iter(anchors))) != relative(
        meta["votes"], meta["colors"], meta["anchor"])


def _bands(meta):
    points = meta["votes"]
    t, l, b, r = _box(points)
    return (b-t+1-len({y for y, x in points}), r-l+1-len({x for y, x in points}))


def _steps(meta):
    points = meta["votes"]
    return tuple(any((r+(axis == 0), c+(axis == 1)) in points for r, c in points)
                 for axis in (0, 1))


def _mask(rng, h, w, informative=False):
    # Random monotone path spans the tight box; frontier growth adds branches.
    mask = np.zeros((h, w), dtype=bool)
    r = c = 0
    mask[r, c] = True
    steps = [0]*(h-1) + [1]*(w-1)
    rng.shuffle(steps)
    for axis in steps:
        r += axis == 0
        c += axis == 1
        mask[r, c] = True
    goal = max(int(mask.sum()), int(round(h*w*float(_pick(rng, (0.25, 0.45, 0.65, 0.85, 1.0))))))
    while mask.sum() < goal:
        frontier = set()
        for r, c in np.argwhere(mask):
            for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= y < h and 0 <= x < w and not mask[y, x]:
                    frontier.add((int(y), int(x)))
        mask[_pick(rng, sorted(frontier))] = True
    if informative:
        mask[:] = True
        mask[0, 0] = False
    if rng.integers(2):
        mask = np.fliplr(mask)
    if rng.integers(2):
        mask = np.flipud(mask)
    return mask.copy()


def _survey(rng, n, extended=False, topology="chain", anchor_count=None):
    """Make partial observations, not whole-parent puzzles or latent solutions."""
    maximum = 4 if extended and n <= 3 else 3
    windows = []
    for i in range(n):
        h, w = (3, 3) if n == 4 else (int(rng.integers(2, maximum+1)), int(rng.integers(2, maximum+1)))
        # Every survey includes a common site but its coordinate origin varies.
        r0, c0 = (-int(rng.integers(2)), -int(rng.integers(2))) if n == 4 else (
            -int(rng.integers(h)), -int(rng.integers(w)))
        windows.append({(r0+r, c0+c) for r in range(h) for c in range(w)})
    charts = [dict() for _ in range(n)]
    used = {(0, 0)}
    commands = {(0, 0): "ordinary"}
    for chart in charts:
        chart[0, 0] = ("command", "ordinary")
    # Show every overlap multiplicity 1..n. Attendance is deliberately varied.
    for count in range(2, n):
        subset = tuple(map(int, rng.choice(n, count, replace=False)))
        available = sorted(set.intersection(*(windows[i] for i in subset))-used)
        if not available:
            return None
        p = _pick(rng, available)
        used.add(p)
        commands[p] = "ordinary"
        for i in subset:
            charts[i][p] = ("command", "ordinary")
    graph = [(i, i+1) for i in range(n-1)] if topology == "chain" else [(0, i) for i in range(1, n)]
    key_sites = {}
    for i, j in graph:
        available = sorted((windows[i] & windows[j])-used)
        if not available:
            return None
        p = _pick(rng, available)
        identity = len(key_sites)
        key_sites[p] = identity
        used.add(p)
        charts[i][p] = charts[j][p] = ("key", identity)
    available = sorted(p for p in set.union(*windows)-used
                       if anchor_count is None or sum(p in window for window in windows) >= anchor_count)
    if not available:
        return None
    anchor = _pick(rng, available)
    used.add(anchor)
    commands[anchor] = "anchor"
    observers = [i for i in range(n) if anchor in windows[i]]
    rng.shuffle(observers)
    attendance = anchor_count if anchor_count is not None else int(rng.integers(1, len(observers)+1))
    for i in observers[:attendance]:
        charts[i][anchor] = ("command", "anchor")
    for i in range(n):
        available = sorted(windows[i]-used)
        if not available:
            return None
        p = _pick(rng, available)
        used.add(p)
        commands[p] = "ordinary"
        charts[i][p] = ("command", "ordinary")
    # Existing landmarks can recur in additional charts, making cycles/triangles.
    for p, identity in key_sites.items():
        for i in range(n):
            if p in windows[i] and rng.random() < .3:
                charts[i][p] = ("key", identity)
    # More distinct commands and independent subsets of observations.
    for p in sorted(set.union(*windows)-used):
        if rng.random() < .48:
            commands[p] = "ordinary"
            observers = [i for i in range(n) if p in windows[i]]
            rng.shuffle(observers)
            for i in observers[:int(rng.integers(1, len(observers)+1))]:
                charts[i][p] = ("command", "ordinary")
    # Optional extra landmarks broaden the exact overlap signature relationships.
    for p in sorted(set.union(*windows)-set(commands)-set(key_sites)):
        observers = [i for i in range(n) if p in windows[i]]
        if len(observers) >= 2 and len(key_sites) < 8 and rng.random() < .3:
            identity = len(key_sites)
            key_sites[p] = identity
            for i in observers:
                charts[i][p] = ("key", identity)
    return charts, commands, anchor, key_sites


def _render_chart(chart, patterns, roles):
    bg, ink, marker, frame = roles
    t, l, b, r = _box(chart)
    h, w = b-t+1, r-l+1
    result = np.full((3*h+1, 3*w+1), bg, dtype=np.int8)
    result[[0, -1], :] = frame
    result[:, [0, -1]] = frame
    for (y, x), (kind, value) in chart.items():
        rr, cc = 1+3*(y-t), 1+3*(x-l)
        if kind == "command":
            result[rr, cc] = ink if value == "anchor" else marker
        else:
            result[rr:rr+2, cc:cc+2] = patterns[value]
    return result


def _pack(rng, cards, charts, mask, commands, anchor, roles):
    bg, ink, marker, frame = roles
    h, w = mask.shape
    t, l, b, r = _box(commands)
    extent_h, extent_w = (b-t+1)*h, (r-l+1)*w
    min_h = max(extent_h, max(a.shape[0] for a in cards))
    min_w = max(extent_w, max(a.shape[1] for a in cards))
    if max(min_h, min_w) > 30:
        return None
    foreground = int(mask.sum()) + sum(np.count_nonzero(a != bg) for a in cards)
    canvases = [(int(rng.integers(max(min_h, 16), 31)),
                 int(rng.integers(max(min_w, 16), 31))) for _ in range(7)] + [(30, 30)]
    for gh, gw in canvases:
        if gh*gw <= 2*foreground:
            continue
        for attempt in range(10):
            tr = int(rng.integers((anchor[0]-t)*h, gh-(b-anchor[0]+1)*h+1))
            tc = int(rng.integers((anchor[1]-l)*w, gw-(r-anchor[1]+1)*w+1))
            occupied = [(tr, tc, tr+h-1, tc+w-1)]
            placements = {}
            order = sorted(range(len(cards)), key=lambda i: -(cards[i].size+float(rng.random())*30))
            for i in order:
                ch, cw = cards[i].shape
                positions = [(y, x) for y in range(gh-ch+1) for x in range(gw-cw+1)
                             if all(_gap((y, x, y+ch-1, x+cw-1), box) for box in occupied)]
                if not positions:
                    break
                y, x = _pick(rng, positions)
                placements[i] = (y, x)
                occupied.append((y, x, y+ch-1, x+cw-1))
            else:
                inp = np.full((gh, gw), bg, dtype=np.int8)
                inp[tr:tr+h, tc:tc+w][mask] = ink
                for i, (y, x) in placements.items():
                    ch, cw = cards[i].shape
                    inp[y:y+ch, x:x+cw] = cards[i]
                # Construction uses world-coordinate commands and direct writes;
                # the solver instead parses cards and solves a registration graph.
                out = np.full_like(inp, bg)
                votes = Counter(p for chart in charts for p, token in chart.items()
                                if token[0] == "command")
                mr, mc = np.nonzero(mask)
                for (y, x), kind in commands.items():
                    color = ink if kind == "anchor" else marker
                    if votes[y, x] % 2:
                        color = marker if kind == "anchor" else ink
                    out[tr+(y-anchor[0])*h+mr, tc+(x-anchor[1])*w+mc] = color
                return {"input": inp, "output": out}
    return None


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    nt, nq = _counts(num_train, num_test)
    consistent = bool(rng.random() < .25)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    train_topology = _pick(rng, ("chain", "star"))
    puzzle = {"train": [], "test": []}
    seen = set()
    training_dims = set()
    previous_dims = None
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            n = (2 if index == 0 else 3 if index == 1 else 4 if index == 2
                 else int(rng.integers(2, 5))) if split == "train" else (3 if nt == 2 else 4)
            dims = [(h, w) for h in range(2, 6) for w in range(2, 6)
                    if not (split == "train" and index == 0 and h == w)
                    and not (split == "train" and index == 1
                             and (h == previous_dims[0] or w == previous_dims[1]))
                    and not (split == "test" and (h, w) in training_dims)]
            for attempt in range(300):
                h, w = _pick(rng, dims)
                mask = _mask(rng, h, w, informative=(split == "train" and index == 0))
                topology = train_topology if split == "train" else ("star" if train_topology == "chain" else "chain")
                survey = _survey(rng, n, extended=(split == "test" or index > 1), topology=topology,
                                 anchor_count=index+1 if split == "train" and index < 2 else None)
                if survey is None:
                    continue
                charts, commands, anchor, key_sites = survey
                # Both teaching examples show both expansion axes.
                t, l, b, r = _box(commands)
                if split == "train" and index < 2:
                    if b == t or r == l:
                        continue
                    if index == 0 and anchor == ((t+b)//2, (l+r)//2):
                        continue
                roles = palette if consistent else tuple(map(int, rng.choice(10, 4, replace=False)))
                bg, ink, marker, frame = roles
                ids = list(map(int, rng.choice(np.arange(1, 15), len(key_sites), replace=False)))
                patterns = [np.array([ink if code & (1 << bit) else marker for bit in range(4)],
                                     dtype=np.int8).reshape(2, 2) for code in ids]
                cards = [_render_chart(chart, patterns, roles) for chart in charts]
                pair = _pack(rng, cards, charts, mask, commands, anchor, roles)
                if pair is None or _key(pair["input"]) in seen:
                    continue
                meta = _decode(pair["input"])
                if split == "train" and index == 0:
                    if (not _registration_needed(meta) or not all(_steps(meta))
                            or meta["votes"][meta["anchor"]] != 1):
                        continue
                if split == "train" and index == 1:
                    first = _decode(puzzle["train"][0]["input"])
                    a = tuple(anchor[i]-_box(commands)[i] for i in (0, 1))
                    old = tuple(first["anchor"][i]-_box(first["colors"])[i] for i in (0, 1))
                    if any(a[i] == old[i] for i in (0, 1)) or meta["votes"][meta["anchor"]] != 2:
                        continue
                if split == "test":
                    training_bands = [_bands(_decode(q["input"])) for q in puzzle["train"]]
                    if any(_bands(meta)[i] and not any(v[i] for v in training_bands) for i in (0, 1)):
                        continue
                assert np.array_equal(_paint(pair["input"].shape, meta), pair["output"])
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                if split == "train" and index == 0:
                    previous_dims = h, w
                if split == "train":
                    training_dims.add((h, w))
                break
            else:
                raise RuntimeError(f"bounded construction exhausted at {split}/{index}, {n} cards")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, info = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            m = _decode(inp)
            assert np.array_equal(_paint(inp.shape, m), out), "wrong anchored parity expansion"
            assert set(m["votes"].values()) == set(range(1, len(m["charts"])+1))
            assert np.count_nonzero(out != m["bg"]) == len(m["votes"])*int(m["mask"].sum())
            assert not np.array_equal(_paint(inp.shape, m, ignore_parity=True), out), "overlap parity has no effect"
            assert not np.array_equal(inp, out)
            info[split].append(m)
    train = info["train"]
    assert len({m["mask"].shape[0] for m in train}) >= 2
    assert len({m["mask"].shape[1] for m in train}) >= 2
    assert any(m["mask"].shape[0] != m["mask"].shape[1] and not m["mask"].all() for m in train)
    assert any(_registration_needed(m) for m in train), "registration never demonstrated"
    assert any(all(_steps(m)) for m in train), "both unit lattice steps need witnesses"
    normalized_anchors = {tuple(m["anchor"][i]-_box(m["votes"])[i] for i in (0, 1)) for m in train}
    assert len({a[0] for a in normalized_anchors}) >= 2
    assert len({a[1] for a in normalized_anchors}) >= 2
    assert {1, 2} <= {m["votes"][m["anchor"]] for m in train}, "anchor parity not demonstrated"
    assert max(len(m["charts"]) for m in train) >= max(len(m["charts"]) for m in info["test"])
    for m in info["test"]:
        assert m["mask"].shape not in {q["mask"].shape for q in train}, "missing scale-combination shift"
        assert all(not _bands(m)[i] or any(_bands(q)[i] for q in train) for i in (0, 1))
    return None


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    metas, pairs = [], []
    for p in puzzles:
        validate_generated_puzzle(p)
        for split in ("train", "test"):
            for pair in p[split]:
                pairs.append(pair)
                m = _decode(pair["input"])
                metas.append(m)
                assert np.array_equal(solve(pair["input"]), pair["output"])
    signature = lambda p: tuple(_key(q[k]) for s in ("train", "test") for q in p[s] for k in ("input", "output"))
    assert len({signature(p) for p in puzzles}) == 42
    assert signature(puzzles[7]) == signature(generate_puzzle(7))
    # Measures reflect parsed pixels, not merely sampled parameters.
    metrics = {"seeds": 42, "pairs": len(pairs),
               "canvas_shapes": len({q["input"].shape for q in pairs}),
               "template_shapes": len({m["mask"].shape for m in metas}),
               "template_masks": len({_key(m["mask"].astype(np.int8)) for m in metas}),
               "chart_shapes": len({c["shape"] for m in metas for c in m["charts"]}),
               "command_layouts": len({tuple(sorted((r-min(y for y, x in m["votes"]),
                                                        c-min(x for y, x in m["votes"]))
                                                       for r, c in m["votes"])) for m in metas}),
               "occupied_counts": sorted({len(m["votes"]) for m in metas}),
               "chart_counts": sorted({len(m["charts"]) for m in metas}),
               "backgrounds": sorted({m["bg"] for m in metas}),
               "overlap_multiplicities": sorted({n for m in metas for n in m["votes"].values()}),
               "palette_roles": len({tuple(m[k] for k in ("bg", "ink", "marker", "frame")) for m in metas}),
               "anchor_multiplicities": sorted({m["votes"][m["anchor"]] for m in metas})}
    assert metrics["canvas_shapes"] >= 25
    assert metrics["template_shapes"] >= 12 and metrics["template_masks"] >= 70
    assert metrics["chart_shapes"] >= 8 and metrics["command_layouts"] >= 100
    assert metrics["chart_counts"] == [2, 3, 4]
    assert metrics["overlap_multiplicities"] == [1, 2, 3, 4]
    assert len(metrics["occupied_counts"]) >= 8 and len(metrics["backgrounds"]) == 10
    assert any(m["mask"].all() for m in metas)
    assert any(m["mask"].mean() < .6 for m in metas)
    assert any(any(m["votes"][p] == 2 and m["colors"][p] == m["marker"] for p in m["votes"]) for m in metas)
    # Ablation A: discarding translations changes positions/multiplicities.
    m = metas[0]
    assert _registration_needed(m)
    # Ablation B: replacing masks by boxes disagrees on informative demonstrations.
    assert not np.array_equal(_paint(pairs[0]["input"].shape, m, fill=True), pairs[0]["output"])
    bad = {s: [{k: np.array(v, copy=True) for k, v in q.items()} for q in puzzles[0][s]] for s in puzzles[0]}
    bad["train"][0]["output"][m["template_top"]] = m["frame"]
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupt output accepted")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(4200+nt*10+nq, nt, nq))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return metrics


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
