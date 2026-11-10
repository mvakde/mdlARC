"""Last-seen stencils: a causal, anchor-keyed dictionary on an L-keyed scene.

No parent code, files, labels, or generation state are used by solve().
Experiments and self_test keep grids in memory; this module writes no files.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import hashlib
import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _forward(a, code):
    """Physical scene to a frame with fast right and slow down."""
    b = np.rot90(a, code % 4)
    return np.fliplr(b) if code >= 4 else b


def _inverse(a, code):
    b = np.fliplr(a) if code >= 4 else a
    return np.rot90(b, -(code % 4)).copy()


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _D4:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
        result.append(cells)
    return result


def _find_key(a):
    candidates = []
    for code in range(8):
        b = _forward(a, code)
        if min(b.shape) < 6:
            continue
        bg, origin, fast, slow = map(int, (b[-1, -1], b[0, 0], b[0, 1], b[1, 0]))
        if len({bg, origin, fast, slow}) != 4:
            continue
        expected = np.where(np.arange(1, b.shape[1]) % 2 == 1, fast, bg)
        if (np.array_equal(b[0, 1:], expected) and np.all(b[1:, 0] == slow)
                and np.all(b[1, 1:] == bg) and np.all(b[-1, 1:] == bg)
                and np.all(b[1:, 1] == bg) and np.all(b[1:, -1] == bg)):
            candidates.append((b.copy(), code, (bg, origin, fast, slow)))
    if len(candidates) != 1:
        raise ValueError("the dashed-fast, solid-slow L key must be unique")
    return candidates[0]


def _signature(cells, a, anchor):
    ar, ac = anchor
    return tuple(sorted((r - ar, c - ac, int(a[r, c])) for r, c in cells))


def _decoration(sig):
    return tuple(p for p in sig if p[:2] != (0, 0))


def _parse(input_grid):
    a, code, roles = _find_key(_grid(input_grid))
    bg = roles[0]
    mask = a != bg
    mask[0, :] = mask[:, 0] = False
    events, keys = [], set()
    for cells in _components(mask):
        if any(not (1 < r < a.shape[0] - 1 and 1 < c < a.shape[1] - 1)
               for r, c in cells):
            raise ValueError("events must leave a clear interior margin")
        if len(cells) == 1:
            anchor = cells[0]
            key = int(a[anchor])
            sig = None
        else:
            if len(cells) < 3:
                raise ValueError("a complete motif needs an anchor and repeated decoration")
            counts = Counter(int(a[p]) for p in cells)
            singletons = [k for k, count in counts.items() if count == 1]
            if len(singletons) != 1:
                raise ValueError("each complete motif needs exactly one singleton color")
            key = singletons[0]
            anchor = next(p for p in cells if int(a[p]) == key)
            sig = _signature(cells, a, anchor)
            keys.add(key)
        events.append({"anchor": anchor, "key": key, "signature": sig})
    if not events or not keys:
        raise ValueError("a scene needs complete reference motifs")
    if any(e["key"] not in keys for e in events):
        raise ValueError("a bare anchor has no visible definition")
    if any(color in keys for e in events if e["signature"] is not None
           for _, _, color in _decoration(e["signature"])):
        raise ValueError("anchor identities and decoration colors must be disjoint")
    events.sort(key=lambda e: e["anchor"])
    return {"grid": a, "code": code, "roles": roles, "events": events, "keys": keys}


def _choose(d, mode="causal"):
    """Resolve query signatures. Alternative modes are diagnostics only."""
    events = d["events"]
    complete = [i for i, e in enumerate(events) if e["signature"] is not None]
    memory, chosen, latest = {}, {}, None
    order = list(range(len(events)))
    if mode == "reverse":
        order.reverse()
    for i in order:
        e = events[i]
        key = e["key"]
        if e["signature"] is not None:
            memory[key], latest = e["signature"], e["signature"]
            continue
        if mode in ("causal", "reverse"):
            if key not in memory:
                raise ValueError("a query occurs before its first definition")
            sig = memory[key]
        elif mode == "key_blind":
            if latest is None:
                raise ValueError("no previous complete motif")
            sig = latest
        else:
            eligible = [j for j in complete if events[j]["key"] == key]
            if mode == "first":
                j = eligible[0]
            elif mode == "last":
                j = eligible[-1]
            elif mode == "nearest":
                ar, ac = e["anchor"]
                j = min(eligible, key=lambda t: (
                    (events[t]["anchor"][0] - ar) ** 2
                    + (events[t]["anchor"][1] - ac) ** 2, t))
            else:
                raise ValueError("unknown diagnostic selection")
            sig = events[j]["signature"]
        chosen[i] = tuple(sorted(((0, 0, key),) + _decoration(sig)))
    return chosen


def _complete(d, mode="causal"):
    a, events, bg = d["grid"], d["events"], d["roles"][0]
    result = a.copy()
    for i, sig in _choose(d, mode).items():
        ar, ac = events[i]["anchor"]
        for dr, dc, color in sig:
            r, c = ar + dr, ac + dc
            if not (1 < r < a.shape[0] - 1 and 1 < c < a.shape[1] - 1):
                raise ValueError("a copied motif crosses the clear margin")
            if (dr, dc) != (0, 0) and int(result[r, c]) != bg:
                raise ValueError("stamps may not overwrite another event")
            result[r, c] = color
    # The intended completed footprints stay separated, even after filling.
    mask = result != bg
    mask[0, :] = mask[:, 0] = False
    if len(_components(mask)) != len(events):
        raise ValueError("completed motifs must remain separate components")
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Fill bare anchors from preceding same-key definitions, using input only."""
    d = _parse(input_grid)
    return _inverse(_complete(d), d["code"])


def _bounds(sig):
    rows = [r for r, _, _ in sig]
    cols = [c for _, c, _ in sig]
    return min(rows), min(cols), max(rows), max(cols)


def _asymmetric(sig):
    s = set(sig)
    return all(s != {(r * sr, c * sc, k) for r, c, k in sig}
               for sr, sc in ((-1, 1), (1, -1), (-1, -1)))


def _template(rng, key, decorations, previous):
    # Connected frontier growth, not a finite catalogue of stored silhouettes.
    for _ in range(80):
        h, w = int(rng.integers(1, 4)), int(rng.integers(2, 6))
        if h * w < 3:
            h = 2
        n = int(rng.integers(3, min(12, h * w) + 1))
        points = {(int(rng.integers(h)), int(rng.integers(w)))}
        style = int(rng.integers(3))
        while len(points) < n:
            frontier = sorted({(r + dr, c + dc) for r, c in points for dr, dc in _D4
                               if 0 <= r + dr < h and 0 <= c + dc < w} - points)
            neighbors = np.array([sum((r + dr, c + dc) in points for dr, dc in _D4)
                                  for r, c in frontier], dtype=float)
            weights = (np.ones(len(frontier)) if style == 0 else
                       neighbors ** (2 if style == 1 else -2))
            p = frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))]
            points.add(p)
        points = sorted(points)
        ar, ac = points[int(rng.integers(len(points)))]
        offsets = [(r - ar, c - ac) for r, c in points if (r, c) != (ar, ac)]
        count = int(rng.integers(1, min(3, len(decorations), len(offsets) // 2) + 1))
        colors = list(map(int, rng.choice(decorations, count, replace=False)))
        assignment = colors * 2 + [colors[int(rng.integers(count))]
                                   for _ in range(len(offsets) - 2 * count)]
        rng.shuffle(assignment)
        sig = tuple(sorted([(0, 0, key)] + [(r, c, k) for (r, c), k in zip(offsets, assignment)]))
        if sig not in previous:
            return sig
    raise RuntimeError("bounded template sampling exhausted")


def _plan(rng, keys, decorations, versions):
    chains = []
    for key, count in zip(keys, versions):
        signatures, chain = [], []
        for _ in range(count + 1):
            sig = _template(rng, key, decorations, signatures)
            signatures.append(sig)
        for sig in signatures[:-1]:
            chain.extend([(True, key, sig), (False, key, sig)])
        # An unused final definition is legal; it also makes reverse traversal
        # a well-defined, observably different counterfactual in every puzzle.
        chain.append((True, key, signatures[-1]))
        chains.append(chain)
    slack = 15 - sum(map(len, chains))
    for _ in range(int(rng.integers(min(3, slack) + 1))):
        chain = chains[int(rng.integers(len(chains)))]
        index = int(rng.integers(len(chain) - 1))
        chain.insert(index + 1, chain[index])
    positions, merged = [0] * len(chains), []
    while any(p < len(ch) for p, ch in zip(positions, chains)):
        available = [i for i, ch in enumerate(chains) if positions[i] < len(ch)]
        i = available[int(rng.integers(len(available)))]
        merged.append(chains[i][positions[i]])
        positions[i] += 1
    return merged


def _gaps(rng, count, spare):
    weights = rng.dirichlet(np.full(count + 1, float(rng.choice([0.4, 1.0, 3.0]))))
    return (1 + rng.multinomial(spare, weights)).tolist()


def _layout(rng, events):
    boxes = [_bounds(sig) for _, _, sig in events]
    n = len(events)
    options = {}
    for lanes in range(1 if n <= 4 else 2, min(4, n) + 1):
        choices = []
        for cuts in combinations(range(1, n), lanes - 1):
            edges = (0,) + cuts + (n,)
            groups = [boxes[a:b] for a, b in zip(edges, edges[1:])]
            ups = [-min(b[0] for b in group) for group in groups]
            downs = [max(b[2] for b in group) for group in groups]
            widths = [sum(b[3] - b[1] + 1 for b in group) + len(group) - 1
                      for group in groups]
            height, width = sum(u + v + 1 for u, v in zip(ups, downs)) + lanes + 1, max(widths) + 2
            if height <= 29 and width <= 29:
                choices.append((edges, ups, downs, widths, height, width))
        if choices:
            options[lanes] = choices
    if not options:
        raise RuntimeError("no bounded ARC packing")
    lane_count = list(options)[int(rng.integers(len(options)))]
    choices = options[lane_count]
    edges, ups, downs, widths, minh, minw = choices[int(rng.integers(len(choices)))]
    extra_limit = int(rng.choice([3, 8, 29]))
    h = minh + int(rng.integers(min(29 - minh, extra_limit) + 1))
    w = minw + int(rng.integers(min(29 - minw, extra_limit) + 1))
    vertical = _gaps(rng, lane_count, h - minh)
    cursor, places = vertical[0], []
    for lane, (a, b) in enumerate(zip(edges, edges[1:])):
        horizontal = _gaps(rng, b - a, w - widths[lane] - 2)
        col = horizontal[0]
        ar = cursor + ups[lane] + 1
        for j, i in enumerate(range(a, b)):
            lr, lc, hr, hc = boxes[i]
            places.append((ar, col - lc + 1))
            col += hc - lc + 1 + horizontal[j + 1]
        cursor += ups[lane] + downs[lane] + 1 + vertical[lane + 1]
    return (h + 1, w + 1), places


def _paint(rng, events, roles, code):
    shape, places = _layout(rng, events)
    bg, origin, fast, slow = roles
    a = np.full(shape, bg, dtype=np.int8)
    a[0, 1::2] = fast
    a[0, 0], a[1:, 0] = origin, slow
    out = a.copy()
    for (complete, key, sig), (ar, ac) in zip(events, places):
        a[ar, ac] = key
        for dr, dc, color in sig:
            out[ar + dr, ac + dc] = color
            if complete:
                a[ar + dr, ac + dc] = color
    return {"input": _inverse(a, code), "output": _inverse(out, code)}


def pair_features(pair):
    """Small review metrics derived from visible input, without labels."""
    d = _parse(pair["input"])
    events, chosen = d["events"], _choose(d)
    lanes = sorted({e["anchor"][0] for e in events})
    queries = list(chosen)
    definitions = [e for e in events if e["signature"] is not None]
    witnesses = {}
    for mode in ("first", "last", "reverse", "nearest", "key_blind"):
        alternative = _choose(d, mode)
        witnesses[mode] = any(chosen[i] != alternative[i] for i in queries)
    nearest_strict = False
    for i in queries:
        ar, ac = events[i]["anchor"]
        eligible = [e for e in definitions if e["key"] == events[i]["key"]]
        distances = [(e["anchor"][0] - ar) ** 2 + (e["anchor"][1] - ac) ** 2 for e in eligible]
        minimum = min(distances)
        nearest_strict |= all(e["signature"] != chosen[i]
                              for e, distance in zip(eligible, distances) if distance == minimum)
    carry = any(i > 0 and events[i - 1]["anchor"][0] != events[i]["anchor"][0]
                for i in queries)
    used = list(chosen.values())
    return {"code": d["code"], "roles": d["roles"], "keys": len(d["keys"]),
            "events": len(events), "queries": len(queries), "sources": len(definitions),
            "lanes": tuple(sum(e["anchor"][0] == r for e in events) for r in lanes),
            "input_shape": tuple(np.asarray(pair["input"]).shape),
            "sizes": tuple(sorted({len(e["signature"]) for e in definitions})),
            "geometries": tuple(sorted({tuple((r, c) for r, c, _ in e["signature"])
                                        for e in definitions})),
            "decor_colors": len({color for e in definitions for _, _, color in _decoration(e["signature"])}),
            "multicolor": any(len({k for _, _, k in _decoration(s)}) > 1 for s in used),
            "asymmetric": any(_asymmetric(s) for s in used),
            "holes": any((hr - lr + 1) * (hc - lc + 1) > len(s)
                         for s in used for lr, lc, hr, hc in [_bounds(s)]),
            "off_center": any(lr + hr != 0 or lc + hc != 0 for s in used
                              for lr, lc, hr, hc in [_bounds(s)]),
            "repeated_definition": len({(e["key"], e["signature"]) for e in definitions}) < len(definitions),
            "carry": carry, "nearest_strict": nearest_strict, "witnesses": witnesses}


def _make_pair(rng, code, kind):
    for _ in range(80):
        roles = tuple(map(int, rng.choice(10, 4, replace=False)))
        pool = [c for c in range(10) if c != roles[0]]
        rng.shuffle(pool)
        if kind == "first":
            count, versions = 1, [int(rng.integers(2, 4))]
        elif kind == "second":
            count, versions = 2, [2, 2]
        elif kind == "test":
            count = int(rng.choice([2, 3]))
            versions = [2, 1] if count == 2 else [2, 1, 1]
            for _ in range(int(rng.integers(3))):
                eligible = [i for i, v in enumerate(versions)
                            if v < 3 and 2 * sum(versions) + count + 2 <= 15]
                if eligible:
                    versions[int(rng.choice(eligible))] += 1
            rng.shuffle(versions)
        else:
            count = int(rng.integers(1, 3))
            versions = [int(rng.integers(1, 4)) for _ in range(count)]
        keys = pool[:count]
        decorations = pool[count:count + int(rng.integers(1, 4))]
        pair = _paint(rng, _plan(rng, keys, decorations, versions), roles, code)
        f = pair_features(pair)
        if kind == "first" and not (f["carry"] and f["asymmetric"] and f["holes"]
                                     and f["nearest_strict"]):
            continue
        if kind == "second" and not (f["carry"] and f["asymmetric"]
                                      and f["witnesses"]["key_blind"]):
            continue
        return pair
    raise RuntimeError("bounded demonstration witness sampling exhausted")


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    first = int(rng.integers(8))
    second = int(rng.choice([c for c in range(8) if c % 2 != first % 2]))
    training_codes = [first, second]
    if rng.random() < 0.5:
        training_codes.append(int(rng.choice([c for c in range(8) if c not in training_codes])))
    test_codes = [c for c in range(8) if c not in training_codes]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            kind = ("first" if i == 0 else "second" if i == 1 else "ordinary") if split == "train" else "test"
            code = training_codes[i] if split == "train" and i < 2 else int(rng.choice(
                training_codes if split == "train" else test_codes))
            for _ in range(8):
                pair = _make_pair(rng, code, kind)
                k = _key(pair["input"])
                if k not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input sampling exhausted")
            seen.add(k)
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape
            k = _key(a)
            assert k not in seen, "duplicate input"
            seen.add(k)
            d = _parse(a)
            expected = _inverse(_complete(d), d["code"])
            assert np.array_equal(expected, b), "incorrect stencil, causal selection, or preservation"
            original_occupied = a != d["roles"][0]
            assert np.array_equal(a[original_occupied], b[original_occupied]), "an existing cell changed"
            assert np.count_nonzero(a != b) >= 2, "no meaningful completion"
            f = pair_features(pair)
            (training if split == "train" else testing).append(f)
    for mode in ("first", "last", "reverse", "nearest", "key_blind"):
        assert any(f["witnesses"][mode] for f in training), f"unresolved shortcut: {mode}"
    assert any(f["carry"] for f in training), "no cross-lane memory demonstration"
    assert any(f["nearest_strict"] for f in training), "nearest-source evidence depends on a distance tie"
    assert any(f["asymmetric"] and f["holes"] for f in training)
    assert {f["code"] % 2 for f in training} == {0, 1}, "both fast-axis orientations need demonstrations"
    assert not ({f["code"] for f in training} & {f["code"] for f in testing}), "missing frame-combination shift"


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded 42-seed variation, equivariance, counts, and negative checks."""
    digests, shapes, geometries, lane_profiles, roles = set(), set(), set(), set(), set()
    codes = {s: set() for s in ("train", "test")}
    key_counts, event_counts, sizes, totals = set(), set(), set(), Counter()
    within_variation = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digests.add(_digest(puzzle))
        local = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                f = pair_features(pair)
                shapes.add(f["input_shape"])
                geometries.update(f["geometries"])
                lane_profiles.add(f["lanes"])
                roles.add(f["roles"])
                codes[split].add(f["code"])
                key_counts.add(f["keys"])
                event_counts.add(f["events"])
                sizes.update(f["sizes"])
                local.add(f["lanes"])
                for name in ("multicolor", "off_center", "holes", "repeated_definition", "carry"):
                    totals[name] += int(f[name])
        within_variation += len(local) > 1
    assert len(digests) == 42
    assert len(shapes) >= 45 and len(geometries) >= 140 and len(lane_profiles) >= 35
    assert len(roles) >= 100 and all(values == set(range(8)) for values in codes.values())
    assert key_counts == {1, 2, 3} and {5, 10, 15} <= event_counts
    assert set(range(3, 13)) <= sizes and within_variation == 42
    assert all(totals[name] >= 12 for name in totals)
    base = generate_puzzle(0)
    assert _digest(base) == _digest(generate_puzzle(0))
    for pair in base["train"] + base["test"]:
        for code in range(8):
            assert np.array_equal(solve(_forward(pair["input"], code)), _forward(pair["output"], code))
        permutation = np.random.default_rng(917).permutation(10)
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 4), (3, 2)):
        p = generate_puzzle(100 + a + b, a, b)
        assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal explicit counts accepted")
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in base[s]] for s in base}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupt label accepted")
    # Remove the first definition of a key, leaving its first query uninitialized.
    d = _parse(base["train"][0]["input"])
    e = next(e for e in d["events"] if e["signature"] is not None)
    broken_input = d["grid"].copy()
    ar, ac = e["anchor"]
    for dr, dc, _ in e["signature"]:
        broken_input[ar + dr, ac + dc] = d["roles"][0]
    try:
        solve(_inverse(broken_input, d["code"]))
    except ValueError:
        pass
    else:
        raise AssertionError("uninitialized query accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(digests),
            "input_shapes": len(shapes), "anchor_relative_geometries": len(geometries),
            "lane_population_profiles": len(lane_profiles), "role_assignments": len(roles),
            "key_counts": sorted(key_counts), "event_counts": sorted(event_counts),
            "motif_sizes": sorted(sizes), "feature_pair_counts": dict(totals),
            "within_puzzle_lane_variation": within_variation,
            "semantic_verification": "generator-owned checks only"}
