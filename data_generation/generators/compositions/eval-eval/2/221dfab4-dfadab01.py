"""Clocked stencil printer: visible codebook, packet registration, XOR ink.

Only Python's standard library and NumPy are used. No generation state is
consulted by solve(). All rotations are genuine quarter turns.
"""
from collections import Counter
from itertools import combinations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid of dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _components(a, bg):
    remaining = set(map(tuple, np.argwhere(a != bg).tolist()))
    result = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        stack, cells = [root], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = (r + dr, c + dc)
                    if p in remaining:
                        remaining.remove(p)
                        stack.append(p)
        result.append(cells)
    return result


def _mask_ok(mask):
    return (mask.ndim == 2 and all(3 <= n <= 5 for n in mask.shape)
            and not mask[0, 0] and mask[0, 1] and mask[1, 0]
            and mask.sum() >= 6 and mask[-1].any() and mask[:, -1].any()
            and len(_components(mask.astype(np.int8), 0)) == 1)


def _decode(value):
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if counts.max() <= a.size / 2:
        raise ValueError("background must occupy more than half the input")
    headers = []
    for k in range(4):
        g = np.rot90(a, k)
        for color in colors:
            color = int(color)
            if color == bg:
                continue
            pts = np.argwhere(g == color)
            if len(pts) < 2 or not np.all(pts[:, 0] == 0):
                continue
            cols = np.sort(pts[:, 1])
            s, length = int(cols[0]), len(cols)
            if not np.array_equal(cols, np.arange(s, s + length)):
                continue
            ends = [c for c in (s - 1, s + length)
                    if 0 <= c < g.shape[1] and int(g[0, c]) not in (bg, color)]
            if len(ends) == 1:
                cap = ends[0]
                if np.count_nonzero(g[0] != bg) == length + 1:
                    headers.append((k, g.copy(), color, s, length, cap, int(g[0, cap])))
    if len(headers) != 1:
        raise ValueError("one unambiguous edge bar with one ink cap is required")
    k, g, marker, start, length, cap, ink = headers[0]
    codes = set(map(int, colors)) - {bg, marker, ink}
    if not 1 <= len(codes) <= 3:
        raise ValueError("expected one to three codes")
    material = g.copy()
    material[0] = bg
    book, live, rejected, bare, exemplars = {}, [], [], [], []
    for cells in _components(material, bg):
        points = [(r, c, int(material[r, c])) for r, c in cells
                  if int(material[r, c]) in codes]
        r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
        if len(points) == 1 and points[0][:2] == (r0, c0):
            live.append(points[0])
            if len(cells) > 1:
                h = max(r for r, _ in cells) - r0 + 1
                w = max(c for _, c in cells) - c0 + 1
                mask = np.zeros((h, w), dtype=bool)
                for r, c in cells:
                    if material[r, c] == ink:
                        mask[r - r0, c - c0] = True
                if not _mask_ok(mask):
                    raise ValueError("invalid full exemplar")
                code = points[0][2]
                if code in book and not np.array_equal(book[code], mask):
                    raise ValueError("conflicting exemplars")
                book[code] = mask
                exemplars.append(points[0])
        elif points:
            rejected.extend(points)
        else:
            bare.append(cells)
    if set(book) != codes:
        raise ValueError("each code needs a full input-visible exemplar")
    if len({m.shape for m in book.values()}) != 1:
        raise ValueError("this family uses one common motif size")
    return dict(g=g, k=k, bg=bg, marker=marker, start=start, length=length,
                cap=cap, ink=ink, book=book, live=sorted(live),
                rejected=sorted(rejected), bare=bare, exemplars=exemplars)


def _target(d, mode):
    if mode == "in_place":
        return d
    if mode == "next":
        return 4 + 6 * max(0, (d - 4 + 5) // 6)
    if mode == "nearest":
        return min(range(4, 35, 6), key=lambda t: (abs(d - t), t))
    return 6 * (d // 6) + 4


def _footprint(info, record, mode="register", point=False, common=False):
    d, c, code = record
    t = _target(d, mode)
    mask = (np.ones((1, 1), dtype=bool) if point else
            info["book"][min(info["book"]) if common else code])
    H, W = info["g"].shape
    return {(t + int(r), c + int(j)) for r, j in np.argwhere(mask)
            if 0 <= t + r < H and 0 <= c + j < W}


def _finish(info, field):
    out = np.full(info["g"].shape, info["bg"], dtype=np.int8)
    out[field] = info["ink"]
    for d in range(len(out)):
        color = (info["bg"] if d % 2 else
                 info["ink"] if d % 6 == 4 else info["marker"])
        out[d, info["start"]:info["start"] + info["length"]] = color
    out[0, info["cap"]] = info["ink"]
    return out


def _render(info, *, mode="register", union=False, all_codes=False,
            point=False, common=False):
    field = np.zeros(info["g"].shape, dtype=bool)
    records = info["live"] + (info["rejected"] if all_codes else [])
    for rec in records:
        points = _footprint(info, rec, mode, point, common)
        if points:
            rr, cc = np.array(sorted(points)).T
            if union:
                field[rr, cc] = True
            else:
                field[rr, cc] ^= True
    return _finish(info, field)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode everything from this input; no training pairs or seed are needed."""
    info = _decode(input_grid)
    return np.rot90(_render(info), -info["k"]).copy()


def _coordinate_prediction(info):
    """Check the toggle renderer by counting covering mask cells per coordinate."""
    out = np.full(info["g"].shape, info["bg"], dtype=np.int8)
    for r in range(len(out)):
        for c in range(out.shape[1]):
            if info["start"] <= c < info["start"] + info["length"]:
                out[r, c] = (info["bg"] if r % 2 else
                             info["ink"] if r % 6 == 4 else info["marker"])
                continue
            total = 0
            for d, lateral, code in info["live"]:
                dr, dc = r - (6 * (d // 6) + 4), c - lateral
                mask = info["book"][code]
                if 0 <= dr < len(mask) and 0 <= dc < mask.shape[1]:
                    total += bool(mask[dr, dc])
            if total % 2:
                out[r, c] = info["ink"]
    out[0, info["cap"]] = info["ink"]
    return out


def _metrics(info):
    target = _render(info)
    points = [_footprint(info, rec) for rec in info["live"]]
    counts = Counter(p for footprint in points for p in footprint)
    s, e = info["start"], info["start"] + info["length"]
    outside = lambda p: not s <= p[1] < e
    partial = False
    for a, b in combinations(points, 2):
        overlap = {p for p in a & b if outside(p)}
        difference = {p for p in a ^ b if outside(p)}
        if (any(counts[p] % 2 == 0 for p in overlap)
                and any(counts[p] % 2 == 1 for p in difference)):
            partial = True
            break
    multiplicity = Counter((d // 6, c, code) for d, c, code in info["live"])
    cancellation = triple_visible = False
    for rec, footprint in zip(info["live"], points):
        d, c, code = rec
        n = multiplicity[d // 6, c, code]
        if n == 2 and any(outside(p) and counts[p] == 2 for p in footprint):
            cancellation = True
        if n == 3 and any(outside(p) and counts[p] == 3 for p in footprint):
            triple_visible = True
    clipping = set()
    H, W = info["g"].shape
    for d, c, code in info["live"]:
        h, w = info["book"][code].shape
        t = 6 * (d // 6) + 4
        if t + h > H:
            clipping.add("far")
        if c + w > W:
            clipping.add("lateral")
        if t + h > H and c + w > W:
            clipping.add("corner")
    result = {name: not np.array_equal(target, _render(info, **kwargs))
              for name, kwargs in (
                  ("registration", dict(mode="in_place")),
                  ("next", dict(mode="next")),
                  ("nearest", dict(mode="nearest")),
                  ("parity", dict(union=True)),
                  ("membership", dict(all_codes=True)),
                  ("substitution", dict(point=True)),
                  ("code_identity", dict(common=True)))}
    result.update(partial=partial, clipping=clipping,
                  lane_cut=any(s <= c < e and r % 2 and n % 2
                               for (r, c), n in counts.items()),
                  max_multiplicity=max(multiplicity.values()),
                  pair=any(n == 2 for n in multiplicity.values()),
                  triple=any(n == 3 for n in multiplicity.values()),
                  cancellation=cancellation, triple_visible=triple_visible,
                  remote=any(not s <= c < e for _, c, _ in info["live"]))
    return result


def _check_material(info):
    """Restrict generated distractors to the visibly taught attachment shapes."""
    material = info["g"].copy()
    material[0] = info["bg"]
    for cells in _components(material, info["bg"]):
        if len(cells) == 1:
            assert int(material[cells[0]]) in info["book"]
            continue
        r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
        actual = {q: int(material[q]) for q in cells}
        matched = False
        for code, mask in info["book"].items():
            ink_cells = {(r0 + int(r), c0 + int(c)): info["ink"]
                         for r, c in np.argwhere(mask)}
            last = max(map(tuple, np.argwhere(mask).tolist()))
            for anchor in (None, (r0, c0), (r0 + last[0] + 1, c0 + last[1] + 1)):
                expected = dict(ink_cells)
                if anchor is not None:
                    expected[anchor] = code
                if expected == actual:
                    matched = True
        assert matched, "unexplained rendered input component"


def _check_counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _sample_mask(rng, h, w):
    kind = int(rng.integers(5))
    mask = np.zeros((h, w), dtype=bool)
    if kind == 0:
        mask[0] = mask[-1] = True
        mask[:, 0] = mask[:, -1] = True
    elif kind == 1:
        mask[:] = True
        for r, c in rng.permutation(np.argwhere(mask))[0:int(rng.integers(1, h * w // 3 + 1))]:
            mask[r, c] = False
    elif kind == 2:
        mask[:, int(rng.integers(w))] = True
        mask[int(rng.integers(h)), :] = True
    elif kind == 3:
        for r in range(h):
            c = round(r * (w - 1) / (h - 1))
            mask[r, c] = mask[r, w - 1 - c] = True
    else:
        r, c = 0, 1
        mask[r, c] = True
        while r < h - 1 or c < w - 1:
            if r == h - 1 or (c < w - 1 and rng.random() < .5):
                c += 1
            else:
                r += 1
            mask[r, c] = True
    if rng.random() < .5:
        mask = mask[:, ::-1].copy()
    if rng.random() < .5:
        mask = mask[::-1].copy()
    mask[0, 0] = False
    mask[0, 1] = mask[1, 0] = mask[-1, -1] = True
    # Connect components by short diagonal paths, never through the blank origin.
    while len(_components(mask.astype(np.int8), 0)) > 1:
        objects = _components(mask.astype(np.int8), 0)
        a, b = min(((a, b) for a in objects[0] for b in objects[1]),
                   key=lambda ab: max(abs(ab[0][0] - ab[1][0]), abs(ab[0][1] - ab[1][1])))
        r, c = a
        while (r, c) != b:
            r += int(np.sign(b[0] - r))
            c += int(np.sign(b[1] - c))
            if (r, c) != (0, 0):
                mask[r, c] = True
    while mask.sum() < 6:
        possible = [(r, c) for r in range(h) for c in range(w)
                    if (r, c) != (0, 0) and not mask[r, c]
                    and mask[max(0, r - 1):r + 2, max(0, c - 1):c + 2].any()]
        mask[possible[int(rng.integers(len(possible)))]] = True
    assert _mask_ok(mask)
    return mask


def _place(g, bg, cells):
    """A one-cell Chebyshev halo separates input objects, not their outputs."""
    H, W = g.shape
    for r, c, _ in cells:
        if not (2 <= r < H and 0 <= c < W):
            return False
        if np.any(g[max(0, r - 1):min(H, r + 2), max(0, c - 1):min(W, c + 2)] != bg):
            return False
    for r, c, color in cells:
        g[r, c] = color
    return True


def _construct_pair(rng, book, bg, marker, ink, k, kind):
    h, w = next(iter(book.values())).shape
    codes = sorted(book)
    for attempt in range(100):
        teacher = kind in ("double", "partial")
        D = int(rng.choice([18, 23, 24, 29, 30] if teacher else
                           [23, 24, 29, 30] if kind == "triple" else list(range(14, 31))))
        W = int(rng.integers(22 if teacher or kind == "triple" else 16, 31))
        length = int(rng.integers(2, 6))
        start = int(rng.integers(1, W - length - 1))
        cap = start - 1 if rng.random() < .5 else start + length
        g = np.full((D, W), bg, dtype=np.int8)
        g[0, start:start + length] = marker
        g[0, cap] = ink
        live = []

        def point(d, c, code):
            if _place(g, bg, [(d, c, code)]):
                live.append((d, c, code))
                return True
            return False

        # The special interacting requests are placed first; their input points
        # are separated, but packet registration makes their output masks meet.
        if kind in ("double", "triple", "partial"):
            b = int(rng.integers(1, max(2, (D - h - 4) // 6 + 1)))
            c = int(rng.integers(0, W - w + 1))
            code = int(rng.choice(codes))
            phases = [0, 2, 4] if kind == "triple" else sorted(map(int, rng.choice([0, 2, 4], 2, replace=False)))
            group = [(6 * b + phase, c, code) for phase in phases]
            if kind == "partial":
                other = int(rng.choice([x for x in codes if x != code] or codes))
                shift = int(rng.integers(1, min(w, W - c)))
                group[-1] = (group[-1][0], c + shift, other)
            if not all(point(*rec) for rec in group):
                continue

        # A late packet request distinguishes registration from next-line snap.
        if teacher:
            b = int(rng.integers(0, max(1, (D - 6) // 6 + 1)))
            if not point(6 * b + 5, int(rng.integers(W - w + 1)), int(rng.choice(codes))):
                continue
        # A stamp crossing the ruler demonstrates its overwrite priority.
        if kind == "double":
            if not point(int(rng.integers(2, D - h)), start - 1, int(rng.choice(codes))):
                continue
        # This one request is clipped on both axes after registration.
        if kind == "partial":
            eligible = [d for d in range(2, D) if 6 * (d // 6) + 4 < D
                        and 6 * (d // 6) + 4 + h > D]
            if not point(int(rng.choice(eligible)), W - 2, int(rng.choice(codes))):
                continue

        success = True
        for code in rng.permutation(codes).tolist():
            cells = [(int(r), int(c), ink) for r, c in np.argwhere(book[code])]
            placed = False
            for _ in range(180):
                r, c = int(rng.integers(2, D - h + 1)), int(rng.integers(W - w + 1))
                component = [(r, c, code)] + [(r + a, c + b, color) for a, b, color in cells]
                if _place(g, bg, component):
                    live.append((r, c, code))
                    placed = True
                    break
            if not placed:
                success = False
                break
        if not success:
            continue

        # A genuine attachment distractor, rather than a hidden invalid singleton.
        if kind == "partial" or rng.random() < .30:
            placed = False
            for _ in range(180):
                code = int(rng.choice(codes))
                mask = book[code]
                last = max(map(tuple, np.argwhere(mask).tolist()))
                r, c = int(rng.integers(2, D - h)), int(rng.integers(W - w))
                cells = [(r + int(a), c + int(b), ink) for a, b in np.argwhere(mask)]
                cells.append((r + last[0] + 1, c + last[1] + 1, code))
                if _place(g, bg, cells):
                    placed = True
                    break
            if not placed and kind == "partial":
                continue
        if rng.random() < .35:
            for _ in range(100):
                code = int(rng.choice(codes))
                r, c = int(rng.integers(2, D - h + 1)), int(rng.integers(W - w + 1))
                if _place(g, bg, [(r + int(a), c + int(b), ink)
                                  for a, b in np.argwhere(book[code])]):
                    break
        for _ in range(int(rng.integers(0, 7))):
            for _ in range(30):
                rec = (int(rng.integers(2, D)), int(rng.integers(W)), int(rng.choice(codes)))
                packet_key = rec[0] // 6, rec[1], rec[2]
                if kind != "triple" and sum((d // 6, c, code) == packet_key for d, c, code in live) >= 2:
                    continue
                if point(*rec):
                    break
        counts = Counter((d // 6, c, code) for d, c, code in live)
        if kind != "triple" and max(counts.values()) > 2:
            continue
        if np.count_nonzero(g == bg) <= g.size / 2:
            continue
        info = dict(g=g, bg=bg, marker=marker, ink=ink, start=start,
                    length=length, cap=cap, book=book, live=sorted(live), rejected=[])
        inp = np.rot90(g, -k).copy()
        decoded = _decode(inp)
        assert decoded["live"] == sorted(live), "input contacts altered membership"
        assert all(np.array_equal(decoded["book"][c], book[c]) for c in book)
        metrics = _metrics(decoded)
        if kind == "double" and not (metrics["cancellation"] and metrics["parity"] and metrics["lane_cut"]):
            continue
        if kind == "partial" and not (metrics["partial"] and metrics["membership"]
                                      and {"far", "lateral", "corner"} <= metrics["clipping"]):
            continue
        if kind == "triple" and not metrics["triple_visible"]:
            continue
        return dict(input=inp, output=np.rot90(_render(info), -k).copy())
    raise RuntimeError("bounded component packing failed")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    n = int(rng.integers(1, 4))
    palette = rng.permutation(10).tolist()
    marker, ink = palette[:2]
    codes = palette[2:2 + n]
    backgrounds = palette[2 + n:4 + n]
    h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
    book = {}
    for code in codes:
        for _ in range(100):
            mask = _sample_mask(rng, h, w)
            if not any(np.array_equal(mask, m) for m in book.values()):
                book[code] = mask
                break
        else:
            raise RuntimeError("could not sample distinct motifs")
    regime, base = int(rng.integers(3)), int(rng.integers(4))
    train_k = ([base + 2 * (i % 2) for i in range(num_train)] if regime == 0 else
               [int(rng.integers(4)) for _ in range(num_train)] if regime == 1 else
               [base] * num_train)
    test_k = ([base + 1] * num_test if regime == 0 else
              [int(rng.integers(4)) for _ in range(num_test)] if regime == 1 else
              [base] * num_test)
    kinds = ["double", "partial"] + ["ordinary"] * (num_train - 2)
    rng.shuffle(kinds)
    for _ in range(24):
        puzzle, seen = {"train": [], "test": []}, set()
        try:
            for split, rotations in (("train", train_k), ("test", test_k)):
                for i, k in enumerate(rotations):
                    kind = kinds[i] if split == "train" else "triple"
                    bg = backgrounds[i % 2] if split == "train" else int(rng.choice(backgrounds))
                    for _ in range(20):
                        pair = _construct_pair(rng, book, bg, marker, ink, k % 4, kind)
                        key = _key(pair["input"])
                        if key not in seen:
                            break
                    else:
                        raise RuntimeError("duplicate inputs exhausted retries")
                    seen.add(key)
                    puzzle[split].append(pair)
            validate_generated_puzzle(puzzle)
            return puzzle
        except (AssertionError, RuntimeError):
            # Whole-puzzle rejection supplies aggregate teaching evidence without
            # inserting a repeated foreground witness at fixed answer coordinates.
            continue
    raise RuntimeError("bounded teaching construction failed")


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, convention, train_metrics, test_metrics = set(), None, [], []
    taught = Counter()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and not np.array_equal(a, b)
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            info = _decode(a)
            current = (info["marker"], info["ink"],
                       tuple((c, _key(m)) for c, m in sorted(info["book"].items())))
            if convention is None:
                convention = current
            assert convention == current, "dictionary and palette roles changed within puzzle"
            _check_material(info)
            canonical_output = np.rot90(b, info["k"])
            assert np.array_equal(canonical_output, _render(info)), "incorrect output"
            assert np.array_equal(canonical_output, _coordinate_prediction(info)), "coordinate check disagrees"
            assert np.all(info["g"][1, max(0, min(info["cap"], info["start"]) - 1):
                                           min(info["g"].shape[1], max(info["cap"], info["start"] + info["length"] - 1) + 2)] == info["bg"])
            assert all(d >= 2 for d, _, _ in info["live"])
            metrics = _metrics(info)
            if split == "train":
                train_metrics.append(metrics)
                assert metrics["max_multiplicity"] <= 2
                taught.update({code for _, _, code in info["exemplars"]})
            else:
                test_metrics.append(metrics)
                assert metrics["triple_visible"], "triple request must affect visible print outside ruler"
    required = ("registration", "next", "nearest", "parity", "membership",
                "substitution", "partial", "lane_cut", "cancellation", "remote")
    for feature in required:
        assert any(m[feature] for m in train_metrics), f"missing teaching witness: {feature}"
    if len(taught) > 1:
        assert any(m["code_identity"] for m in train_metrics)
    assert all(n >= 2 for n in taught.values()), "each code must occur in two training inputs"
    train_clips = set().union(*(m["clipping"] for m in train_metrics))
    test_clips = set().union(*(m["clipping"] for m in test_metrics))
    assert test_clips <= train_clips, "test uses untaught clipping"


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def self_test():
    """Bounded suite: 42 seeds plus count limits, equivariance and corruption."""
    shapes, masks, palettes, layouts, rotations, live_counts = set(), set(), set(), set(), set(), set()
    puzzles = []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        for split in ("train", "test"):
            for pair in p[split]:
                info = _decode(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                shapes.add(pair["input"].shape)
                masks.update(_key(m) for m in info["book"].values())
                palettes.add((info["marker"], info["ink"], tuple(sorted(info["book"]))))
                layouts.add((info["g"].shape, info["start"], info["length"], tuple(info["live"])))
                rotations.add((split, info["k"]))
                live_counts.add(len(info["live"]))
    assert len(shapes) >= 50 and len(masks) >= 35
    assert len(palettes) >= 35 and len(layouts) >= 180 and len(live_counts) >= 7
    assert rotations == {(s, k) for s in ("train", "test") for k in range(4)}
    assert _equal(puzzles[0], generate_puzzle(0))
    assert not _equal(puzzles[0], puzzles[1])
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1234 + a * 10 + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    pair = puzzles[0]["test"][0]
    for k in range(4):
        assert np.array_equal(solve(np.rot90(pair["input"], k)), np.rot90(pair["output"], k))
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
              for s in ("train", "test")}
    out = broken["test"][0]["output"]
    out[0, 0] = (int(out[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt label accepted")
    return dict(seeds=42, grid_shapes=len(shapes), distinct_masks=len(masks),
                palettes=len(palettes), layouts=len(layouts), live_counts=sorted(live_counts))


if __name__ == "__main__":
    print(self_test())
