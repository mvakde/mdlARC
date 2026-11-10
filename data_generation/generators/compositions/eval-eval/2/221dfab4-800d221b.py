"""Clocked cross-section ballots. Standard library and NumPy only.

No files, source fixtures, generator state, or demonstrations are used by solve.
All diagnostics in self_test are bounded and in memory.
"""
from __future__ import annotations

import hashlib
import numpy as np

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _canonical(a, side):
    # side: top, right, bottom, left; lateral coordinates need not be clockwise.
    return (a, a.T[::-1], a[::-1], a.T)[side]


def _restore(a, side):
    return (a, a[::-1].T, a[::-1], a.T)[side].copy()


def _components(mask):
    h, w = mask.shape
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    components = []
    ids = np.full(mask.shape, -1, dtype=np.int16)
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        stack, cells = [root], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            ids[r, c] = len(components)
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
        components.append(cells)
    return components, ids


def _bars(a, color):
    candidates = []
    for side in range(4):
        g = _canonical(a, side)
        rr, cc = np.where(g == color)
        if len(rr) and np.all(rr == 0):
            start, end = int(cc.min()), int(cc.max()) + 1
            if end - start == len(cc):
                candidates.append((side, start, end))
    return candidates


def _interpret(g, bg, marker, wire, signals, start, end, side):
    panels, ids = _components(np.isin(g, signals))
    assert 2 <= len(panels) <= 6
    boxes, local, minorities = [], [], []
    for cells in panels:
        rr, cc = np.array(cells).T
        r0, r1, c0, c1 = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
        assert len(cells) == (r1 - r0) * (c1 - c0)
        counts = [int(np.count_nonzero(g[rr, cc] == s)) for s in signals]
        assert min(counts) > 0 and counts[0] != counts[1]
        local.append(signals[int(counts[1] > counts[0])])
        minorities.append(min(counts))
        boxes.append((r0, r1, c0, c1))
    routes, _ = _components(g == wire)
    assert routes
    owner = np.full(g.shape, -1, dtype=np.int16)
    used = set()
    for cells in routes:
        neighbors = set()
        for r, c in cells:
            for dr, dc in _DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < len(g) and 0 <= nc < g.shape[1] and ids[nr, nc] >= 0:
                    neighbors.add(int(ids[nr, nc]))
        assert len(neighbors) == 1
        pid = neighbors.pop()
        used.add(pid)
        for p in cells:
            owner[p] = pid
    assert used == set(range(len(panels)))
    cuts, weights, winners, margins = [], [], [], []
    for r in range(4, len(g), 6):
        counts = np.bincount(owner[r, owner[r] >= 0], minlength=len(panels))
        tally = [sum(int(n) for n, s in zip(counts, local) if s == signal) for signal in signals]
        assert sum(tally) > 0 and tally[0] != tally[1], "sampled line needs a strict vote"
        cuts.append(r)
        weights.append(tuple(map(int, counts)))
        winners.append(signals[int(tally[1] > tally[0])])
        margins.append(abs(tally[1] - tally[0]))
    assert cuts, "at least one census line"
    return dict(g=g, background=bg, marker=marker, wire=wire, signals=signals,
                side=side, start=start, end=end, panels=panels, boxes=boxes,
                local=tuple(local), minorities=minorities, routes=routes,
                owner=owner, panel_ids=ids, cuts=cuts, weights=weights,
                winners=winners, margins=margins)


def _parse(a):
    values, counts = np.unique(a, return_counts=True)
    assert len(values) == 5
    bg = int(values[np.argmax(counts)])
    assert int(counts.max()) > a.size / 2
    bars = [(int(c), _bars(a, int(c))) for c in values if c != bg]
    bars = [(c, b[0]) for c, b in bars if len(b) == 1]
    assert len(bars) == 1, "one unambiguous whole-color boundary bar"
    marker, (side, start, end) = bars[0]
    # Corner singletons have two possible directions and were excluded above.
    assert len(_bars(a, marker)) == 1
    g = _canonical(a, side)
    remaining = [int(c) for c in values if c not in (bg, marker)]
    interpretations = []
    for wire in remaining:
        signals = tuple(c for c in remaining if c != wire)
        try:
            interpretations.append(_interpret(g, bg, marker, wire, signals, start, end, side))
        except AssertionError:
            continue
    assert len(interpretations) == 1, "one panel/wire interpretation required"
    return interpretations[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete output from the visible input only."""
    info = _parse(_grid(input_grid))
    g = info["g"]
    out = g.copy()
    owner = info["owner"]
    for pid, color in enumerate(info["local"]):
        out[owner == pid] = color
    objects = (g != info["background"]) & (g != info["marker"])
    for r, winner in zip(info["cuts"], info["winners"]):
        out[r, objects[r]] = winner
    start, end = info["start"], info["end"]
    out[:, start:end] = info["background"]
    for r in range(0, len(g), 2):
        if r % 6 != 4:
            out[r, start:end] = info["marker"]
    for r, winner in zip(info["cuts"], info["winners"]):
        out[r, start:end] = winner
    return _restore(out, info["side"])


def _cellwise_expected(info):
    """Validation formula, separate from solve's mask painting."""
    g = info["g"]
    special = dict(zip(info["cuts"], info["winners"]))
    out = np.empty_like(g)
    for r in range(len(g)):
        for c in range(g.shape[1]):
            old = int(g[r, c])
            if info["start"] <= c < info["end"]:
                color = (info["background"] if r % 2 else special[r] if r % 6 == 4
                         else info["marker"])
            elif r in special and old not in (info["background"], info["marker"]):
                color = special[r]
            elif old == info["wire"]:
                color = info["local"][int(info["owner"][r, c])]
            else:
                color = old
            out[r, c] = color
    return _restore(out, info["side"])


def _measure(info):
    g, owner = info["g"], info["owner"]
    rr, cc = np.indices(g.shape)
    outside = (cc < info["start"]) | (cc >= info["end"])
    special = rr % 6 == 4
    wire = owner >= 0
    panel = info["panel_ids"] >= 0
    objects = wire | panel
    panel_disagree = False
    raw_disagree = False
    override = panel_broadcast = 0
    raw = [int(np.count_nonzero(g == s)) for s in info["signals"]]
    for r, weights, winner in zip(info["cuts"], info["weights"], info["winners"]):
        nwin = sum(n > 0 and s == winner for n, s in zip(weights, info["local"]))
        nlose = sum(n > 0 and s != winner for n, s in zip(weights, info["local"]))
        panel_disagree |= nwin < nlose
        raw_disagree |= raw[info["signals"].index(winner)] < raw[1 - info["signals"].index(winner)]
        for c in range(g.shape[1]):
            if outside[r, c] and wire[r, c]:
                override += info["local"][int(owner[r, c])] != winner
            if outside[r, c] and panel[r, c]:
                panel_broadcast += int(g[r, c]) != winner
    local_visible = all(np.any((owner == i) & outside & ~special) for i in range(len(info["panels"])))
    minority_preserved = False
    for pid, color in enumerate(info["local"]):
        minority_preserved |= bool(np.any((info["panel_ids"] == pid) & (g != color) & outside & ~special))
    padded = np.pad(wire, 1)
    degree = sum(padded[1 + dr:1 + dr + len(g), 1 + dc:1 + dc + g.shape[1]] for dr, dc in _DIRS)
    areas = [len(cells) for cells in info["panels"]]
    return dict(shape=tuple(g.shape), side=info["side"], panels=len(areas),
                panel_areas=areas, local_margins=[a - 2 * m for a, m in zip(areas, info["minorities"])],
                wire_cells=int(wire.sum()), forks=int(np.sum(wire & (degree >= 3))),
                weight_profiles=info["weights"], winners=tuple(info["winners"]),
                margins=tuple(info["margins"]), dynamic=len(set(info["winners"])) > 1,
                panel_disagree=bool(panel_disagree), raw_disagree=bool(raw_disagree),
                override=int(override), panel_broadcast=int(panel_broadcast),
                local_visible=bool(local_visible), minority_preserved=bool(minority_preserved),
                clearing=bool(np.any(objects & ~outside & (rr % 2 == 1))),
                ordinary_overwrite=bool(np.any(objects & ~outside & (rr > 0) & (rr % 2 == 0) & ~special)),
                background_gaps=bool(np.any((g == info["background"]) & outside & special)),
                roles=(info["background"], info["marker"], info["wire"], *info["signals"]))


def _teaching(measures):
    witness = any(m["dynamic"] and m["panel_disagree"] and m["raw_disagree"] for m in measures)
    keys = ("clearing", "ordinary_overwrite", "panel_broadcast", "override",
            "local_visible", "minority_preserved", "background_gaps")
    return witness and all(any(m[k] for m in measures) for k in keys)


def _territories(rng, width, count):
    # Random bounded integer composition, not a bank of rendered layouts.
    gap = 1 if width < count * 5 else int(rng.integers(1, 3))
    available = width - (count - 1) * gap
    widths = [3] * count
    for _ in range(available - sum(widths)):
        choices = [i for i, w in enumerate(widths) if w < 10]
        if not choices:
            break
        widths[int(rng.choice(choices))] += 1
    slack = width - sum(widths) - gap * (count - 1)
    cursor = int(rng.integers(slack + 1))
    result = []
    for w in widths:
        result.append((cursor, cursor + w))
        cursor += w + gap
    return result


def _network(rng, depth, left, right, box, max_branches):
    r0, r1, c0, c1 = box
    cells = set()
    for header, direction, boundary in ((r0 - 1, -1, 1), (r1, 1, depth - 1)):
        if not 1 <= header < depth:
            continue
        attach = int(rng.integers(c0, c1))
        branch_count = int(rng.integers(1, min(max_branches, right - left) + 1))
        columns = list(map(int, rng.choice(np.arange(left, right), branch_count, replace=False)))
        for index, column in enumerate(columns):
            for c in range(min(attach, column), max(attach, column) + 1):
                cells.add((header, c))
            # One spanning arm, others end independently. The field is readable
            # even when a particular cut has only one participating panel.
            if index == 0 or rng.random() < .28:
                last = boundary
            else:
                last = int(rng.integers(min(header, boundary), max(header, boundary) + 1))
            c = column
            for r in range(header, last + direction, direction):
                cells.add((r, c))
                if rng.random() < .20:
                    new_c = int(np.clip(c + rng.choice([-1, 1]), left, right - 1))
                    cells.add((r, new_c))
                    c = new_c
    return cells


def _geometry(rng, depth, width, count, max_branches, placement):
    panels, networks = [], []
    for left, right in _territories(rng, width, count):
        ph = int(rng.integers(2, min(7, depth - 3) + 1))
        pw = int(rng.integers(2, right - left + 1))
        if placement in ("ends", "middle"):
            candidates = []
            for r in range(1, depth - ph + 1):
                center = (r + (ph - 1) / 2) / (depth - 1)
                if ((placement == "ends" and (center <= .30 or center >= .70)) or
                        (placement == "middle" and .36 <= center <= .64)):
                    candidates.append(r)
            r0 = int(rng.choice(candidates))
        else:
            r0 = int(rng.integers(1, depth - ph + 1))
        c0 = int(rng.integers(left, right - pw + 1))
        box = r0, r0 + ph, c0, c0 + pw
        panels.append(box)
        networks.append(_network(rng, depth, left, right, box, max_branches))
    return panels, networks


def _assignments(networks, depth, witness):
    cuts = list(range(4, depth, 6))
    weights = [[sum(r == cut for r, c in cells) for cells in networks] for cut in cuts]
    n = len(networks)
    options = []
    for bits in range(1, (1 << n) - 1):
        labels = [1 if bits >> i & 1 else -1 for i in range(n)]
        totals = [sum(w * label for w, label in zip(ws, labels)) for ws in weights]
        if any(t == 0 for t in totals):
            continue
        dynamic = min(totals) < 0 < max(totals)
        wrong_panel = any(
            sum(label for label, w in zip(labels, ws) if w) * total < 0
            for ws, total in zip(weights, totals))
        if not witness or (dynamic and wrong_panel):
            options.append(bits)
    return options, weights


def _pattern(rng, shape, majority, minority):
    h, w = shape
    size = h * w
    band = int(rng.integers(3))
    nminor = ((size - 1) // 2 if band == 0 else
              1 if band == 1 else max(1, int(size * rng.uniform(.16, .36))))
    nminor = min(nminor, (size - 1) // 2)
    rr, cc = np.indices(shape)
    style = int(rng.integers(4))
    if style == 0:
        order = rng.permutation(size)
    elif style == 1:
        r, c = int(rng.integers(h)), int(rng.integers(w))
        order = np.argsort((abs(rr - r) + abs(cc - c) + rng.random(shape) * .4).ravel())
    elif style == 2:
        order = np.argsort(((rr + cc) % 2 * size + rng.random(shape)).ravel())
    else:
        order = np.argsort((rr * w + cc if rng.random() < .5 else cc * h + rr).ravel())
    flat = np.full(size, majority, dtype=np.int8)
    flat[order[:nminor]] = minority
    return flat.reshape(shape)


def _paint_records(g, roles, panels, networks, labels, start, end):
    """Construction output from sampled records, never recovered solver state."""
    bg, marker, wire, s0, s1 = roles
    out = g.copy()
    for cells, label in zip(networks, labels):
        for p in cells:
            out[p] = label
    for r in range(4, len(g), 6):
        tally = sum((1 if label == s0 else -1) * sum(x == r for x, c in cells)
                    for cells, label in zip(networks, labels))
        assert tally
        winner = s0 if tally > 0 else s1
        out[r, (g[r] != bg) & (g[r] != marker)] = winner
    out[:, start:end] = bg
    for r in range(0, len(g), 2):
        if r % 6 == 4:
            tally = sum((1 if label == s0 else -1) * sum(x == r for x, c in cells)
                        for cells, label in zip(networks, labels))
            out[r, start:end] = s0 if tally > 0 else s1
        else:
            out[r, start:end] = marker
    return out


def _make_pair(rng, roles, side, depth_bounds, branch_budget, placement, witness, rich, panel_options):
    for _ in range(240):
        depth = int(rng.integers(*depth_bounds))
        count = int(rng.choice(panel_options))
        width = int(rng.integers(max(16, 4 * count + count - 1), 31))
        panels, networks = _geometry(rng, depth, width, count, branch_budget, placement)
        choices, _ = _assignments(networks, depth, witness)
        if not choices:
            continue
        bits = int(rng.choice(choices))
        labels = [roles[3 + ((bits >> i) & 1)] for i in range(count)]
        bg, marker, wire, s0, s1 = roles
        g = np.full((depth, width), bg, dtype=np.int8)
        for box, cells, label in zip(panels, networks, labels):
            r0, r1, c0, c1 = box
            g[r0:r1, c0:c1] = _pattern(rng, (r1 - r0, c1 - c0), label, s1 if label == s0 else s0)
            for p in cells:
                g[p] = wire
        length = int(rng.integers(1, min(6, width // 3) + 1))
        start = int(rng.integers(1, width - 1)) if length == 1 else int(rng.integers(width - length + 1))
        end = start + length
        g[0, start:end] = marker
        if np.count_nonzero(g == bg) <= g.size / 2:
            continue
        inp = _restore(g, side)
        try:
            info = _parse(inp)
        except AssertionError:
            continue
        m = _measure(info)
        if witness and not (m["dynamic"] and m["panel_disagree"] and m["raw_disagree"]):
            continue
        if rich and not all(m[k] for k in ("clearing", "ordinary_overwrite", "panel_broadcast", "override",
                                          "local_visible", "minority_preserved", "background_gaps")):
            continue
        output = _restore(_paint_records(g, roles, panels, networks, labels, start, end), side)
        assert np.array_equal(output, _cellwise_expected(info))
        return {"input": inp, "output": output}
    raise RuntimeError("bounded network sampling exhausted")


def _key(a):
    return a.shape, hashlib.sha256(a.astype(np.int8).tobytes()).digest()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one coherent family member with visibly determined conventions."""
    nt, ns = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(4))
    base_side = int(rng.integers(4))
    palette = list(map(int, rng.choice(10, 5, replace=False)))
    varying = bool(rng.integers(2))
    witness_slot = int(rng.integers(nt))
    # The second rich example makes the minimum two-example request informative.
    rich_slot = (witness_slot + int(rng.integers(1, nt))) % nt
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", ns)):
        test = split == "test"
        for index in range(count):
            witness = not test and index == witness_slot
            rich = not test and index == rich_slot
            roles = list(map(int, rng.choice(10, 5, replace=False))) if varying else palette
            side = (base_side + 1) % 4 if mode == 0 and test else base_side
            if mode != 0 and rng.random() < .30:
                side = int(rng.integers(4))
            depth_bounds = (25, 31) if mode == 1 and test else (19, 23) if mode == 1 else (19, 31) if witness or rich else (14, 31)
            branches = 5 if mode == 2 and test else 3 if mode == 2 else 4
            placement = "middle" if mode == 3 and test else "ends" if mode == 3 else "any"
            panel_options = ((4, 5) if test else (3,)) if mode == 2 else (3, 4) if witness else (2, 3, 4, 5)
            for _ in range(12):
                pair = _make_pair(rng, roles, side, depth_bounds, branches, placement, witness, rich, panel_options)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, teaching = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _parse(inp)
            assert np.array_equal(_cellwise_expected(info), out), "wrong output"
            assert not np.array_equal(inp, out)
            if split == "train":
                teaching.append(_measure(info))
    assert _teaching(teaching), "demonstrations must expose the interacting mechanisms"


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """42 seeds; bounded metrics, count sweep, metamorphic and mutation checks."""
    shapes, palettes, structures, sides, panel_counts = set(), set(), set(), set(), set()
    profiles, margins, wire_counts = set(), set(), set()
    changing = weighted = broadcasts = forks = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _parse(inp)
                m = _measure(info)
                assert np.array_equal(solve(inp), out)
                shapes.add(inp.shape)
                palettes.add(m["roles"])
                sides.add((split, info["side"]))
                panel_counts.add(m["panels"])
                profiles.update(m["weight_profiles"])
                margins.update(m["margins"])
                wire_counts.add(m["wire_cells"])
                norm = {}
                normalized = bytes(norm.setdefault(int(c), len(norm)) for c in inp.flat)
                structures.add((inp.shape, hashlib.sha256(normalized).digest()))
                changing += m["dynamic"]
                weighted += m["panel_disagree"]
                broadcasts += m["panel_broadcast"] > 0
                forks += m["forks"] > 0
        # Rotation and a complete role permutation must commute with solving.
        pair = puzzle["test"][0]
        assert np.array_equal(solve(np.rot90(pair["input"])), np.rot90(pair["output"]))
        perm = np.random.default_rng(999 + seed).permutation(10)
        assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
    assert len(shapes) >= 60 and len(structures) == 210
    assert len(palettes) >= 40 and len(wire_counts) >= 60
    assert sides == {(s, d) for s in ("train", "test") for d in range(4)}
    assert panel_counts == {2, 3, 4, 5}
    assert len(profiles) >= 80 and len(margins) >= 8
    assert changing >= 42 and weighted >= 42 and broadcasts >= 60 and forks >= 100
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(300 + nt * 10 + ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a = generate_puzzle(92, 2, 1)
    assert _equal(a, generate_puzzle(92, 2, 1))
    assert not _equal(a, generate_puzzle(93, 2, 1))
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    info = _parse(bad["test"][0]["input"])
    r = info["cuts"][0]
    changed = _canonical(bad["test"][0]["output"], info["side"])
    changed[r, info["start"]] = info["marker"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("mutated special tick accepted")
    _topology_regression()
    _short_regression()
    # A rearranged panel has the same local vote. Only its unsampled, outside-
    # strip panel pixels should follow the rearrangement in the output.
    inp = a["train"][0]["input"].copy()
    info = _parse(inp)
    g = info["g"]
    r0, r1, c0, c1 = info["boxes"][0]
    block = np.sort(g[r0:r1, c0:c1], axis=None).reshape((r1 - r0, c1 - c0))
    g[r0:r1, c0:c1] = block
    expected = _canonical(a["train"][0]["output"], info["side"]).copy()
    for r in range(r0, r1):
        for c in range(c0, c1):
            if r % 6 != 4 and not info["start"] <= c < info["end"]:
                expected[r, c] = g[r, c]
    assert np.array_equal(solve(_restore(g, info["side"])), _restore(expected, info["side"]))


def _topology_regression():
    """Six panels, lateral wiring and nested geometry outside the sampler lanes."""
    # Filled rectangles and wire components are deliberately hand placed; the
    # outer network surrounds other panels but is separated from their wires.
    g = np.zeros((24, 30), dtype=np.int8)
    roles = [0, 9, 8, 2, 5]
    boxes = [(1, 3, 2, 5), (1, 3, 10, 13), (1, 3, 19, 22),
             (18, 21, 4, 7), (18, 21, 13, 16), (18, 21, 23, 26)]
    labels = [2, 5, 2, 2, 5, 2]
    nets = [set() for _ in boxes]
    for i, (r0, r1, c0, c1) in enumerate(boxes):
        g[r0:r1, c0:c1] = labels[i]
        g[r0, c0] = 5 if labels[i] == 2 else 2
        # Mostly separate strokes, with a long sideways perimeter route below.
        c = c0 + 1
        for r in range(3 if i < 3 else 5, 18 if i < 3 else 18):
            nets[i].add((r, c))
    # Make the first owner's network extend laterally outside its panel column.
    nets[0] |= {(3, c) for c in range(1, 4)} | {(r, 1) for r in range(3, 23)}
    nets[0] |= {(22, c) for c in range(1, 29)} | {(r, 28) for r in range(4, 23)}
    for cells in nets:
        for p in cells:
            assert g[p] == 0
            g[p] = 8
    g[0, 8:10] = 9
    info = _parse(g)
    assert len(info["panels"]) == 6
    expected = _paint_records(g, roles, boxes, nets, labels, 8, 10)
    assert np.array_equal(solve(g), expected)


def _short_regression():
    # A truncated ruler, boundary wires, and a strip that can cross an entire
    # panel are legal even though the default sampler favors longer sweeps.
    for start, end in ((5, 6), (1, 4), (0, 11)):
        g = np.zeros((5, 12), dtype=np.int8)
        g[1:3, 1:4] = 2
        g[1, 1] = 5
        g[1:3, 7:10] = 5
        g[1, 7] = 2
        g[3:5, 1:3] = 8
        g[3:5, 8] = 8
        g[0, start:end] = 9
        for side in range(4):
            inp = _restore(g, side)
            expected = g.copy()
            expected[3:5, 1:3] = 2
            expected[3:5, 8] = 5
            expected[4, g[4] == 8] = 2
            expected[:, start:end] = 0
            expected[0, start:end] = expected[2, start:end] = 9
            expected[4, start:end] = 2
            assert np.array_equal(solve(inp), _restore(expected, side))
