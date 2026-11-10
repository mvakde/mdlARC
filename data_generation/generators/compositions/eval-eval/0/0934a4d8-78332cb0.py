"""Echo-wing chains: jointly assemble panels and repair a reflected seam.

Only standard-library modules and NumPy are used. There is no file I/O during
generation, solving, validation, or self_test. All inference starts from pixels.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import hashlib

import numpy as np


HOST_SEEDS = (0, 353902540, 759082242, 676152718, 34854230, 926570764,
              769853161, 254064250, 885295013, 821547776, 520209522,
              599386619, 546829695, 784143005, 422736717, 48279409,
              950203872, 335456564, 490149300, 945708988, 785841399,
              205620975, 182859866, 549656334, 539595286, 820726576,
              203771382, 838767531, 330172800, 897270075, 156283833,
              806336665, 295886628, 366669403, 964738581, 650860017,
              843039053, 22827586, 422722456, 768717931, 868386729,
              444389841)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _layout(panels, order, shape, divider):
    nr, nc = shape
    h, w = panels[0].shape
    a = np.full((nr * (h + 1) - 1, nc * (w + 1) - 1), divider, dtype=np.int8)
    for slot, index in enumerate(order):
        r, c = divmod(slot, nc)
        a[r * (h + 1):r * (h + 1) + h,
          c * (w + 1):c * (w + 1) + w] = panels[index]
    return a


def _partition(a):
    candidates = []
    for color in np.unique(a):
        rows = list(map(int, np.flatnonzero(np.all(a == color, axis=1))))
        cols = list(map(int, np.flatnonzero(np.all(a == color, axis=0))))
        if not rows and not cols:
            continue
        rr, cc = [-1] + rows + [a.shape[0]], [-1] + cols + [a.shape[1]]
        hh = [b - x - 1 for x, b in zip(rr, rr[1:])]
        ww = [b - x - 1 for x, b in zip(cc, cc[1:])]
        if (min(hh + ww) < 5 or len(set(hh)) != 1 or len(set(ww)) != 1
                or not 2 <= len(hh) * len(ww) <= 5):
            continue
        panels = [a[r + 1:rr[i + 1], c + 1:cc[j + 1]].copy()
                  for i, r in enumerate(rr[:-1]) for j, c in enumerate(cc[:-1])]
        if any(np.any(p == color) for p in panels):
            continue
        corners = {int(p[r, c]) for p in panels for r, c in
                   ((0, 0), (0, -1), (-1, 0), (-1, -1))}
        if len(corners) != 1:
            continue
        candidates.append(dict(panels=panels, background=corners.pop(),
                               divider=int(color), layout=(len(hh), len(ww))))
    if len(candidates) != 1:
        raise ValueError("No unique regular panel partition")
    return candidates[0]


def _occluder(parsed):
    panels = parsed["panels"]
    colors = set(int(x) for p in panels for x in np.unique(p))
    candidates = []
    for color in colors - {parsed["background"], parsed["divider"]}:
        indices = [i for i, p in enumerate(panels) if np.any(p == color)]
        if len(indices) != 1:
            continue
        i = indices[0]
        rr, cc = np.nonzero(panels[i] == color)
        r, s, c, t = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
        if len(rr) >= 2 and len(rr) == (s - r) * (t - c):
            candidates.append((color, i))
    if len(candidates) != 1:
        raise ValueError("Occluder must be the unique rectangular panel-local color")
    return candidates[0]


def _wings(p):
    d = p.shape[1] // 2
    # Both returned arrays run from the outer edge towards the center spine.
    return p[:, :d], p[:, d + 1:][:, ::-1]


def _orders(panels, bg, marker, ports_only=False):
    wings = [_wings(p) for p in panels]
    blank = lambda a: bool(np.all(a == bg))
    starts = [i for i, (left, _) in enumerate(wings) if blank(left)]
    ends = [i for i, (_, right) in enumerate(wings) if blank(right)]
    if len(starts) != 1 or len(ends) != 1 or starts[0] == ends[0]:
        return []
    if any((blank(left) and i != starts[0]) or (blank(right) and i != ends[0])
           for i, (left, right) in enumerate(wings)):
        return []
    edges = {}
    for i, (_, right) in enumerate(wings):
        for j, (left, _) in enumerate(wings):
            if i == j or blank(right) or blank(left):
                continue
            x, y = (right[:, 0], left[:, 0]) if ports_only else (right, left)
            visible = (x != marker) & (y != marker)
            matches = (x != bg) == (y != bg) if ports_only else x == y
            if np.all(matches[visible]):
                edges[i, j] = True
    result = []

    def visit(path, remaining):
        if len(result) >= 2:
            return
        if not remaining:
            if path[-1] == ends[0]:
                result.append(tuple(path))
            return
        if path[-1] == ends[0]:
            return
        for j in remaining:
            if edges.get((path[-1], j), False):
                visit(path + [j], tuple(k for k in remaining if k != j))

    visit([starts[0]], tuple(i for i in range(len(panels)) if i != starts[0]))
    return result


def _connected(mask):
    points = {tuple(map(int, p)) for p in np.argwhere(mask)}
    if not points:
        return False
    stack = [points.pop()]
    while stack:
        r, c = stack.pop()
        for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if q in points:
                points.remove(q)
                stack.append(q)
    return not points


def _decode(input_grid):
    a = _grid(input_grid)
    parsed = _partition(a)
    marker, target = _occluder(parsed)
    bg, candidates = parsed["background"], []
    for axis in ("H", "V"):
        panels = [p.copy() if axis == "H" else p.T.copy() for p in parsed["panels"]]
        h, w = panels[0].shape
        d = w // 2
        if not (5 <= h <= 11 and 5 <= w <= 13 and w % 2 == 1):
            continue
        if any(np.any(p[0] != bg) or np.any(p[-1] != bg) for p in panels):
            continue
        rr, cc = np.nonzero(panels[target] == marker)
        box = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
        r, s, c, t = box
        if not (1 <= r < s <= h - 1 and (t <= d or c >= d + 1)):
            continue
        orders = _orders(panels, bg, marker)
        if len(orders) != 1:
            continue
        order = orders[0]
        position = order.index(target)
        side = "L" if t <= d else "R"
        if (side == "L" and position == 0) or (side == "R" and position == len(order) - 1):
            continue
        source = order[position - 1] if side == "L" else order[position + 1]
        raw_source = panels[source][:, d + 1:] if side == "L" else panels[source][:, :d]
        partner = raw_source[:, ::-1]
        completed = [p.copy() for p in panels]
        wing = completed[target][:, :d] if side == "L" else completed[target][:, d + 1:]
        hidden = wing == marker
        reflection_effect = bool(np.any(partner[hidden] != raw_source[hidden]))
        wing[hidden] = partner[hidden]
        if any(not _connected(p != bg) for p in completed):
            continue
        if any(not np.array_equal(_wings(completed[i])[1], _wings(completed[j])[0])
               for i, j in zip(order, order[1:])):
            continue
        # Count how many nonempty opposite wings could fill this hole locally.
        # This measures actual ambiguity before enforcing the complete chain.
        unknown = _wings(panels[target])[0 if side == "L" else 1]
        local_sources, local_predictions = [], set()
        for i, p in enumerate(panels):
            other = _wings(p)[1 if side == "L" else 0]
            visible = (unknown != marker) & (other != marker)
            if (i != target and np.any(other != bg) and not np.any(other == marker)
                    and np.all((unknown == other)[visible])):
                local_sources.append(i)
                local_predictions.add(other[unknown == marker].tobytes())
        candidates.append(dict(parsed, axis=axis, panels=panels, completed=completed,
                               marker=marker, target=target, source=source, side=side,
                               box=box, order=order, reflection_effect=reflection_effect,
                               local_sources=tuple(local_sources),
                               local_predictions=len(local_predictions),
                               port_orders=len(_orders(panels, bg, marker, ports_only=True))))
    if len(candidates) != 1:
        raise ValueError("Need one axis and one complete reflected-wing chain")
    return candidates[0]


def _answer(record):
    panels = record["completed"]
    result = _layout(panels, record["order"], (1, len(panels)), record["divider"])
    return result if record["axis"] == "H" else result.T.copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer partition, occlusion, axis, order, and missing cells from input only."""
    return _answer(_decode(input_grid))


def _choose(rng, choices):
    return choices[int(rng.integers(len(choices)))]


def _wing(rng, h, d, signature, bg, colors):
    occupied = {(r, 0) for r in signature}
    hub = int(rng.integers(1, h - 1))
    occupied.add((hub, d - 1))
    for row in signature:
        bend = int(rng.integers(1, d))
        occupied.update((row, c) for c in range(bend + 1))
        occupied.update((r, bend) for r in range(min(row, hub), max(row, hub) + 1))
        occupied.update((hub, c) for c in range(bend, d))
    allowed = {(r, c) for r in range(1, h - 1) for c in range(1, d)}
    for _ in range(int(rng.integers(0, max(2, len(allowed) // 2)))):
        frontier = sorted({q for r, c in occupied for q in
                           ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                           if q in allowed and q not in occupied})
        if not frontier:
            break
        occupied.add(_choose(rng, frontier))
    a = np.full((h, d), bg, dtype=np.int8)
    for point in sorted(occupied):
        a[point] = int(_choose(rng, colors))
    return a


def _make_panels(links, h, d, bg, colors, rng):
    panels = []
    blank = np.full((h, d), bg, dtype=np.int8)
    for i in range(len(links) + 1):
        left = blank if i == 0 else links[i - 1]
        right = blank if i == len(links) else links[i]
        p = np.full((h, 2 * d + 1), bg, dtype=np.int8)
        p[:, :d], p[:, d + 1:] = left, right[:, ::-1]
        roots = sorted(set(map(int, np.flatnonzero(left[:, -1] != bg)))
                       | set(map(int, np.flatnonzero(right[:, -1] != bg))))
        low = max(1, roots[0] - int(rng.integers(2)))
        high = min(h - 2, roots[-1] + int(rng.integers(2)))
        p[low:high + 1, d] = rng.choice(colors, size=high - low + 1)
        panels.append(p)
    return panels


def _make_pair(rng, axis, n, full_mask, colors_count, port_witness=False):
    # Each attempt is discarded in memory. At most 96 attempts; no trace/log I/O.
    for _ in range(96):
        palette = list(map(int, rng.permutation(10)))
        bg, divider, marker = palette[:3]
        colors = palette[3:3 + colors_count]
        widths = [w for w in range(5, 14, 2) if n * (w + 1) - 1 <= 30]
        w = _choose(rng, widths)
        h = int(rng.integers(7 if port_witness else 5, 12))
        d = w // 2
        pool = [s for size in range(1, min(3, h - 4) + 1)
                for s in combinations(range(1, h - 1), size)]
        common_signature = _choose(rng, pool)
        signatures = [common_signature if port_witness or rng.random() < .5
                      else _choose(rng, pool) for _ in range(n - 1)]
        links = [_wing(rng, h, d, s, bg, colors) for s in signatures]
        if not port_witness and n >= 3 and rng.random() < .25:
            j = int(rng.integers(n - 2))
            links[j + 1] = links[j].copy()
        if port_witness and len({_key(link) for link in links}) != n - 1:
            continue
        panels = _make_panels(links, h, d, bg, colors, rng)
        link = int(rng.integers(n - 1))
        side = _choose(rng, ("L", "R"))
        target = link + 1 if side == "L" else link
        if full_mask:
            r, s, c, t = 1, h - 1, 0, d
        else:
            r = int(rng.integers(1, h - 1))
            s = int(rng.integers(r + 1, h))
            c = int(rng.integers(d))
            t = int(rng.integers(c + 1, d + 1))
            if (s - r) * (t - c) < 2 or (r, s, c, t) == (1, h - 1, 0, d):
                continue
        offset = 0 if side == "L" else d + 1
        crop = panels[target][r:s, c + offset:t + offset]
        if not np.any(crop == bg) or not np.any(crop != bg):
            continue
        masked = [p.copy() for p in panels]
        masked[target][r:s, c + offset:t + offset] = marker
        if axis == "V":
            masked = [p.T.copy() for p in masked]
            complete = [p.T.copy() for p in panels]
        else:
            complete = panels
        ph, pw = masked[0].shape
        layouts = [(nr, n // nr) for nr in range(1, n + 1) if n % nr == 0
                   and nr * (ph + 1) - 1 <= 30 and (n // nr) * (pw + 1) - 1 <= 30]
        shape = _choose(rng, layouts)
        order = list(map(int, rng.permutation(n)))
        if order == list(range(n)):
            order[0], order[-1] = order[-1], order[0]
        inp = _layout(masked, order, shape, divider)
        out_shape = (1, n) if axis == "H" else (n, 1)
        out = _layout(complete, list(range(n)), out_shape, divider)
        try:
            record = _decode(inp)
        except ValueError:
            continue
        if not np.array_equal(_answer(record), out):
            continue
        if port_witness and (record["port_orders"] < 2 or not record["reflection_effect"]):
            continue
        if full_mask and n >= 3 and record["local_predictions"] < 2:
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("Failed to sample a unique echo chain within 96 attempts")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        seed %= 1 << 64
    rng = np.random.default_rng(seed)
    first = _choose(rng, ("H", "V"))
    second = "V" if first == "H" else "H"
    plans = [("train", first, 4, False, 1, True),
             ("train", second, 3, True, 2, False)]
    plans += [("train", _choose(rng, ("H", "V")), int(rng.integers(2, 5)),
               bool(rng.integers(2)), int(rng.integers(1, 4)), False)
              for _ in range(nt - 2)]
    plans += [("test", _choose(rng, ("H", "V")), int(rng.integers(4, 6)),
               True, int(rng.integers(1, 4)), False) for _ in range(nq)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, axis, n, full, nc, witness in plans:
        for _ in range(8):
            pair = _make_pair(rng, axis, n, full, nc, witness)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("Distinct-input retry budget exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _measure(pair):
    a, b = _grid(pair["input"]), _grid(pair["output"])
    record = _decode(a)
    expected = _answer(record)
    assert np.array_equal(expected, b), "Wrong repaired chain"
    n = len(record["panels"])
    h, w = record["panels"][0].shape
    bg, marker = record["background"], record["marker"]
    assert marker not in b, "Occlusion must disappear completely"
    # Recover output tiles by fixed dimensions, independently of its dividers.
    oriented = b if record["axis"] == "H" else b.T
    assert oriented.shape == (h, n * (w + 1) - 1)
    observed = [oriented[:, j * (w + 1):j * (w + 1) + w] for j in range(n)]
    assert Counter(_key(p) for p in observed) == Counter(_key(p) for p in record["completed"])
    changed = 0
    for before, after in zip(record["panels"], record["completed"]):
        assert np.array_equal(before[before != marker], after[before != marker])
        assert _connected(after != bg)
        changed += int(np.count_nonzero(before != after))
    r, s, c, t = record["box"]
    assert changed == (s - r) * (t - c)
    crop = record["completed"][record["target"]][r:s, c:t]
    assert np.any(crop == bg) and np.any(crop != bg), "Teach recovery of both ink and background"
    joins = [_wings(record["completed"][i])[1] for i in record["order"][:-1]]
    signatures = [tuple(map(int, np.flatnonzero(p[:, 0] != bg))) for p in joins]
    assert all(signatures), "Every chain join has a nonempty geometric port"
    return dict(axis=record["axis"], n=n, panel_shape=(h, w), layout=record["layout"],
                input_shape=a.shape, output_shape=b.shape, mask_shape=(s - r, t - c),
                side=record["side"], mask_area=changed,
                full_mask=(s - r == h - 2 and t - c == w // 2),
                reflection_effect=record["reflection_effect"],
                local_sources=len(record["local_sources"]), port_orders=record["port_orders"],
                local_predictions=record["local_predictions"],
                reordered=record["order"] != tuple(range(n)),
                rearranged=(record["order"] != tuple(range(n)) or record["layout"] !=
                            ((1, n) if record["axis"] == "H" else (n, 1))),
                background=bg, divider=record["divider"], marker=marker,
                ink_colors=tuple(sorted(int(x) for x in np.unique(b) if x not in (bg, record["divider"]))),
                repeated_ports=len(set(signatures)) < len(signatures),
                repeated_wings=len({_key(p) for p in joins}) < len(joins),
                max_port=max(map(len, signatures)),
                wing_keys=tuple(_key((p != bg).astype(np.int8)) for p in joins))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a = _grid(pair["input"])
            key = _key(a)
            assert key not in seen, "Duplicate inputs"
            seen.add(key)
            records[split].append(_measure(pair))
    train = records["train"]
    assert {r["axis"] for r in train} == {"H", "V"}, "Expose both spatial directions"
    assert any(r["port_orders"] >= 2 and r["reflection_effect"] and len(r["ink_colors"]) == 1
               and not r["full_mask"] and r["rearranged"] for r in train), "Need the reflected-texture witness"
    assert any(r["full_mask"] and r["local_predictions"] >= 2 and len(r["ink_colors"]) >= 2
               for r in train), "Need the global-chain witness"
    # No specific literal role, layout, side, or split restriction is semantic.


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, content = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(content)
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed coverage, equivariance, counts, and rejection checks."""
    signatures, features, geometry = set(), [], set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                feature = _measure(pair)
                geometry.update(feature.pop("wing_keys"))
                features.append(feature)
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
        if seed in (0, 353902540):
            assert _puzzle_digest(generate_puzzle(seed)) == _puzzle_digest(puzzle)
    assert len(signatures) == 42
    assert {r["n"] for r in features} == {2, 3, 4, 5}
    assert len({r["input_shape"] for r in features}) >= 25
    assert len({r["mask_shape"] for r in features}) >= 15
    assert len({r["panel_shape"] for r in features}) >= 15
    assert len(geometry) >= 100
    assert {r["marker"] for r in features} == set(range(10))
    assert {r["background"] for r in features} == set(range(10))
    assert {r["divider"] for r in features} == set(range(10))
    assert {r["side"] for r in features} == {"L", "R"}
    assert any(r["max_port"] > 1 for r in features)
    assert any(r["repeated_wings"] for r in features)
    assert any(r["layout"] == (2, 2) for r in features)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1517 + nt * 11 + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    # Transposition and independent color permutations preserve the rule.
    perm = np.array([6, 0, 8, 1, 9, 3, 5, 2, 7, 4], dtype=np.int8)
    transformed = {s: [{k: perm[v.T].copy() for k, v in pair.items()} for pair in p[s]]
                   for s in ("train", "test")}
    validate_generated_puzzle(transformed)
    # Wrong labels, lost mirror evidence, and two swappable loops must fail.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Wrong label accepted")
    # Destroy a visible joint without touching the marker or divider. The
    # monochrome witness has three distinct wings, so it cannot silently relink.
    witness = next(pair for pair in p["train"] if _measure(pair)["port_orders"] >= 2)
    damaged = _decode(witness["input"])
    broken = [x.copy() for x in damaged["panels"]]
    bg, marker = damaged["background"], damaged["marker"]
    first_tile = damaged["order"][0]
    other = damaged["order"][1]
    if first_tile == damaged["target"]:
        first_tile = damaged["order"][-1]
        wing = broken[first_tile][:, :broken[first_tile].shape[1] // 2]
    else:
        wing = broken[first_tile][:, broken[first_tile].shape[1] // 2 + 1:]
    fresh = next(c for c in range(10) if c not in set(int(v) for x in broken for v in np.unique(x)))
    wing[1:-1] = fresh
    if damaged["axis"] == "V":
        broken = [x.T.copy() for x in broken]
    invalid_input = _layout(broken, list(range(len(broken))), damaged["layout"], damaged["divider"])
    try:
        solve(invalid_input)
    except ValueError:
        pass
    else:
        raise AssertionError("Conflicting visible reflected evidence accepted")
    record = _decode(p["train"][0]["input"])
    panels = [x.copy() for x in record["completed"]]
    # Give two distinct panels identical input/output wings: their spines differ,
    # so two valid permutations would produce different outputs.
    bg, marker = record["background"], record["marker"]
    h, w = panels[0].shape
    d = w // 2
    wing = _wings(panels[record["order"][0]])[1].copy()
    blank = np.full_like(wing, bg)
    example = []
    for left, right in ((blank, wing), (wing, wing), (wing, wing), (wing, blank)):
        tile = np.full((h, w), bg, dtype=np.int8)
        tile[:, :d], tile[:, d + 1:] = left, right[:, ::-1]
        example.append(tile)
    example[1][1:-1, d] = bg
    example[2][1:-1, d] = marker
    assert len(_orders(example, bg, marker)) == 2, "Reject interchangeable full-wing loops"
    return {"seeds_checked": 42, "pairs_checked": len(features),
            "distinct_puzzles": len(signatures), "wing_geometries": len(geometry),
            "input_shapes": len({r["input_shape"] for r in features}),
            "panel_shapes": len({r["panel_shape"] for r in features}),
            "mask_shapes": len({r["mask_shape"] for r in features}),
            "port_ambiguous_pairs": sum(r["port_orders"] >= 2 for r in features),
            "locally_ambiguous_repairs": sum(r["local_predictions"] >= 2 for r in features),
            "reflection_changes_fill": sum(r["reflection_effect"] for r in features)}
