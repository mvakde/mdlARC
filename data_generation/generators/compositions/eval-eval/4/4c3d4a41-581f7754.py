"""Registered pressure pieces: rigid guide alignment creates a shared pin profile.

Only NumPy and the standard library are used. Generation records construct labels;
solve independently segments the input and simulates each pin's contact movement.
No examples, seeds, files, or mutable generation state are consulted by solve.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _need(bool(np.all((0 <= a) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, pending = {start}, [start]
        while pending:
            r, c = pending.pop()
            for dr, dc in _DIRS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    pending.append(p)
        result.append(cells)
    return result


def _decode(input_grid):
    original = _grid(input_grid)
    values, counts = np.unique(original, return_counts=True)
    bg = int(values[np.argmax(counts)])
    _need(int(counts.max()) > original.size / 2, "background must occupy a majority")
    views = [(k, np.rot90(original, k)) for k in range(4)]
    views = [(k, a) for k, a in views if a.shape[1] >= 5
             and int(a[0, 0]) != bg and np.all(a[0] == a[0, 0])]
    _need(len(views) == 1, "one complete uniform boundary wall required")
    turns, a = views[0]
    a = a.copy()
    h, w = a.shape
    structure = int(a[0, 0])
    remaining = {tuple(map(int, p)) for p in np.argwhere(a != bg) if p[0] != 0}
    pins = []
    for c in range(w):
        r = 1
        while r < h and int(a[r, c]) != bg:
            r += 1
        if r > 1:
            _need(0 < c < w - 1 and r <= h - 2, "pin too close to an outer edge")
            seq = tuple(int(v) for v in a[1:r, c])
            pins.append({"col": c, "seq": seq})
            remaining.difference_update((rr, c) for rr in range(1, r))
    _need(len(pins) >= 2, "at least two hanging pins required")
    _need(all(b["col"] - x["col"] >= 2 for x, b in zip(pins, pins[1:])),
          "pin lanes must be separated")
    anchors, pieces = {}, []
    for cells in _components(remaining):
        if len(cells) == 1 and next(iter(cells))[0] == h - 1:
            p = next(iter(cells))
            color = int(a[p])
            _need(0 < p[1] < w - 1 and color not in (bg, structure), "bad guide marker")
            _need(color not in anchors, "one opposite-edge marker per identity required")
            anchors[color] = p
        else:
            pieces.append({"cells": cells})
    _need(anchors and len(pieces) >= 2, "missing pressure pieces or anchors")
    pin_cells = {(r, p["col"]) for p in pins for r in range(1, len(p["seq"]) + 1)}
    target_owner = {}
    for i, piece in enumerate(pieces):
        cells = piece["cells"]
        _need(len(cells) >= 3 and all(2 <= r <= h - 3 for r, _ in cells),
              "pieces must be complete and separated from the wall and anchor edge")
        _need(all((r + dr, c + dc) not in pin_cells for r, c in cells for dr, dc in _DIRS),
              "source pieces must be disconnected from the pins")
        marks = [p for p in cells if int(a[p]) in anchors]
        _need(len(marks) == 1, "piece requires exactly one registration cell")
        mark = marks[0]
        color = int(a[mark])
        _need(all(int(a[p]) == structure for p in cells if p != mark), "unknown piece color")
        dx = anchors[color][1] - mark[1]
        target = {(r, c + dx): int(a[r, c]) for r, c in cells if 0 <= c + dx < w}
        _need(target, "registration must remain visible")
        for p in target:
            _need(p not in target_owner, "registered pieces may touch but cannot overlap")
            target_owner[p] = i
        piece.update(mark=mark, color=color, dx=dx, target=target,
                     clipped=len(cells) - len(target))
    _need(set(anchors) == {p["color"] for p in pieces}, "unused guide identity")
    for pin in pins:
        seq = pin["seq"]
        first_payload = next((i for i, v in enumerate(seq) if v != structure), len(seq))
        _need(first_payload < len(seq), "pin must carry payload")
        _need(all(v not in (bg, structure, *anchors) for v in seq[first_payload:]),
              "pin payload and marker roles must be disjoint")
        c = pin["col"]
        contacts = [(r, owner) for (r, cc), owner in target_owner.items() if cc == c]
        top, owner = min(contacts) if contacts else (h, -1)
        pin.update(top=top, owner=owner, shift=max(0, len(seq) - top + 1),
                   competitors=len({owner for _, owner in contacts}), prefix=first_payload)
    return {"a": a, "turns": turns, "bg": bg, "structure": structure,
            "pins": pins, "anchors": anchors, "pieces": pieces}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Project actual piece cells; push actual pin sequences one step at a time."""
    d = _decode(input_grid)
    a, bg = d["a"], d["bg"]
    out = np.full_like(a, bg)
    out[0] = d["structure"]
    for color, p in d["anchors"].items():
        out[p] = color
    for piece in d["pieces"]:
        for p, color in piece["target"].items():
            out[p] = color
    for pin in d["pins"]:
        cells = list(enumerate(pin["seq"], 1))
        # A gap does not pull a pin downward. Zero motion is valid contact.
        while cells[-1][0] >= pin["top"]:
            cells = [(r - 1, v) for r, v in cells if r > 1]
            _need(cells, "at least one payload cell must survive")
        for r, color in cells:
            _need(int(out[r, pin["col"]]) == bg, "pin still intersects a piece")
            out[r, pin["col"]] = color
    return np.rot90(out, -d["turns"]).copy()


def _key(a):
    return tuple(a.shape), np.asarray(a, dtype=np.int8).tobytes()


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _shape(rng):
    """Connected silhouettes; the registration cell is chosen separately."""
    kind = int(rng.integers(5))
    h, w = int(rng.integers(2, 7)), int(rng.integers(3, 8))
    if kind == 0:  # irregular comb, including sparse and adjacent teeth
        cells = {(h - 1, c) for c in range(w)}
        teeth = rng.choice(w, int(rng.integers(2, w + 1)), replace=False)
        for c in teeth:
            depth = int(rng.integers(1, h))
            cells.update((r, int(c)) for r in range(h - 1 - depth, h - 1))
    elif kind == 1:  # filled stepped contour
        heights = rng.integers(1, h + 1, w)
        cells = {(r, c) for c, height in enumerate(heights) for r in range(h - int(height), h)}
    elif kind == 2:  # frame, optionally open, with a short outward extension
        h = max(3, h)
        cells = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h - 1) or c in (0, w - 1)}
        if rng.random() < .45:
            cells.remove((0, int(rng.integers(1, w - 1))))
        if rng.random() < .5:
            cells.add((h - 1, w))
    elif kind == 3:  # branch with unequal arms
        mid = int(rng.integers(1, w - 1))
        cells = {(r, mid) for r in range(h)}
        cells.update((h - 1, c) for c in range(w))
        row = int(rng.integers(h - 1))
        cells.update((row, c) for c in range(int(rng.integers(mid + 1)), mid + 1))
    else:  # bounded random growth, without an unbounded rejection loop
        cells = {(0, 0)}
        for _ in range(int(rng.integers(4, 16))):
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _DIRS
                               if 0 <= r + dr < h and 0 <= c + dc < w} - cells)
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    if rng.random() < .3:
        cells = {(c, r) for r, c in cells}
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    return sorted((r - r0, c - c0) for r, c in cells)


def _sequence(rng, length, structure, payload, shift, witness):
    prefix = int(rng.integers(0, length)) if rng.random() < .6 else 0
    if witness == "prefix":
        prefix = shift + 1
    seq = [structure] * prefix + [int(rng.choice(payload))] * (length - prefix)
    # Small palettes and a few runs keep long ordered stacks visually readable.
    if length - prefix >= 3 and rng.random() < .5:
        split = int(rng.integers(prefix + 1, length))
        seq[split:] = [int(rng.choice(payload))] * (length - split)
    if witness == "order":
        prefix = min(prefix, length - 2)
        seq[prefix:] = [int(rng.choice(payload))] * (length - prefix)
        seq[-2:] = list(map(int, rng.choice(payload, 2, replace=False)))
    return tuple(seq)


def _label(h, w, bg, structure, anchors, objects, pins):
    """Constructor: direct destination arithmetic on placement records."""
    inp, out = np.full((h, w), bg, np.int8), np.full((h, w), bg, np.int8)
    inp[0] = out[0] = structure
    for color, col in anchors.items():
        inp[h - 1, col] = out[h - 1, col] = color
    top = [h] * w
    for obj in objects:
        for r, c, color in obj["cells"]:
            inp[r, c] = color
            cc = c + obj["dx"]
            if 0 <= cc < w:
                _need(int(out[r, cc]) == bg, "construction target overlap")
                out[r, cc] = color
                top[cc] = min(top[cc], r)
    for c, seq in pins:
        _need(all(int(inp[r, c]) == bg for r in range(1, len(seq) + 2)), "source pin collision")
        inp[1:len(seq) + 1, c] = seq
        shift = max(0, len(seq) - top[c] + 1)
        for j, color in enumerate(seq, 1):
            dest = j - shift
            if dest > 0:
                _need(int(out[dest, c]) == bg, "construction pin overlap")
                out[dest, c] = color
    return {"input": inp, "output": out}


def _features(d):
    pins, pieces = d["pins"], d["pieces"]
    moving = [p for p in pins if p["shift"] > 0]
    owners = {p["owner"] for p in moving}
    rank_cross = any((x["mark"][1] - y["mark"][1]) *
                     (d["anchors"][x["color"]][1] - d["anchors"][y["color"]][1]) < 0
                     for i, x in enumerate(pieces) for y in pieces[i + 1:])
    eccentric = any(2 * p["mark"][1] != min(c for _, c in p["cells"]) +
                    max(c for _, c in p["cells"]) for p in pieces if p["dx"])
    return {"shape": list(d["a"].shape), "pieces": len(pieces), "systems": len(d["anchors"]),
            "pins": len(pins), "turns": d["turns"], "background": d["bg"],
            "structure": d["structure"], "shifts": [p["shift"] for p in pins],
            "moving": len(moving), "stationary": sum(p["shift"] == 0 for p in pins),
            "contact": sum(p["top"] == len(p["seq"]) + 1 for p in pins),
            "gap": sum(p["top"] > len(p["seq"]) + 1 for p in pins),
            "shared": sum(p["competitors"] >= 2 and p["shift"] > 0 for p in pins),
            "owners": len(owners), "rank_cross": rank_cross, "eccentric": eccentric,
            "piece_clips": sum(p["clipped"] > 0 for p in pieces),
            "repeated_identity": len(pieces) > len(d["anchors"]),
            "sizes": [len(p["cells"]) for p in pieces],
            "profiles": [tuple(min(r for r, cc in p["target"] if cc == c)
                               for c in sorted({cc for _, cc in p["target"]})) for p in pieces],
            "ordered_survivor": any(len(set(p["seq"][p["shift"]:]) - {d["structure"]}) >= 2
                                    for p in moving),
            "moving_prefix": any(0 < p["shift"] < p["prefix"] for p in pins)}


def _sample_scene(rng, n, palette, witness):
    h, w = int(rng.integers(14, 29)), int(rng.integers(16, 31))
    bg, structure = map(int, palette[:2])
    m = int(rng.integers(2, min(3, n) + 1))
    markers = list(map(int, palette[2:2 + m]))
    payload = list(map(int, palette[2 + m:2 + m + int(rng.integers(2, 4))]))
    guide_options = list(range(1, w - 1))
    rng.shuffle(guide_options)
    guides = []
    for c in guide_options:
        if all(abs(c - x) >= 2 for x in guides):
            guides.append(c)
            if len(guides) == m:
                break
    anchors = dict(zip(markers, guides))
    identities = markers + [int(rng.choice(markers)) for _ in range(n - m)]
    rng.shuffle(identities)
    objects, source, target = [], set(), {}
    for marker in identities:
        for _ in range(32):
            points = _shape(rng)
            ph, pw = max(r for r, _ in points) + 1, max(c for _, c in points) + 1
            if ph > h - 5 or pw > w:
                continue
            mark = points[int(rng.integers(len(points)))]
            row, col = int(rng.integers(3, h - ph - 1)), int(rng.integers(0, w - pw + 1))
            cells = {(row + r, col + c) for r, c in points}
            if any(p in source or any((p[0] + dr, p[1] + dc) in source for dr, dc in _DIRS)
                   for p in cells):
                continue
            dx = anchors[marker] - col - mark[1]
            projected = {(r, c + dx) for r, c in cells if 0 <= c + dx < w}
            if projected & set(target):
                continue
            colored = [(r, c, marker if (r, c) == (row + mark[0], col + mark[1]) else structure)
                       for r, c in sorted(cells)]
            objects.append({"cells": colored, "dx": dx})
            source.update(cells)
            target.update({p: len(objects) - 1 for p in projected})
            break
        else:
            return None
    source_top = [min((r for r, cc in source if cc == c), default=h - 1) for c in range(w)]
    tops = [min((r for r, cc in target if cc == c), default=h) for c in range(w)]
    cap = [min(h - 3, source_top[c] - 2,
               source_top[max(0, c - 1)] - 1, source_top[min(w - 1, c + 1)] - 1)
           for c in range(w)]
    moving = [c for c in range(1, w - 1) if tops[c] < h and cap[c] >= tops[c]]
    shared = [c for c in moving if len({owner for (r, cc), owner in target.items() if cc == c}) >= 2]
    if len(moving) < 2 or witness and not shared:
        return None
    rng.shuffle(moving)
    choices = shared if witness else moving
    c1 = int(rng.choice(choices))
    owner1 = target[tops[c1], c1]
    choices2 = [c for c in moving if abs(c - c1) >= 2
                and (not witness or target[tops[c], c] != owner1)]
    if not choices2:
        return None
    c2 = int(rng.choice(choices2))
    stationary = [c for c in range(1, w - 1) if cap[c] >= 1 and
                  abs(c - c1) >= 2 and abs(c - c2) >= 2]
    if not stationary:
        return None
    cs = [c1, c2, int(rng.choice(stationary))]
    candidates = list(range(1, w - 1))
    rng.shuffle(candidates)
    desired = int(rng.integers(3, 9))
    for c in candidates:
        if len(cs) >= desired:
            break
        if cap[c] >= 1 and all(abs(c - x) >= 2 for x in cs):
            cs.append(c)
    pins = []
    first_shift = None
    for i, c in enumerate(cs):
        if i < 2:
            lengths = list(range(tops[c], cap[c] + 1))
            if i == 1:
                lengths = [v for v in lengths if v - tops[c] + 1 != first_shift]
                if not lengths:
                    return None
            length = int(rng.choice(lengths))
        elif i == 2:
            upper = min(cap[c], tops[c] - 1)
            length = upper if tops[c] < h and rng.random() < .5 else int(rng.integers(1, upper + 1))
        else:
            length = int(rng.integers(1, cap[c] + 1))
        shift = max(0, length - tops[c] + 1)
        if i == 0:
            first_shift = shift
        demo = "order" if i == 0 else "prefix" if i == 1 else ""
        seq = _sequence(rng, length, structure, payload, shift, demo)
        pins.append((c, seq))
    pair = _label(h, w, bg, structure, anchors, objects, sorted(pins))
    if np.mean(pair["input"] == bg) <= .5:
        return None
    d = _decode(pair["input"])
    f = _features(d)
    if witness and not (f["rank_cross"] and f["eccentric"] and f["owners"] >= 2):
        return None
    return pair


def _fallback(rng, n, palette, variant):
    """Finite, spacious construction obeying precisely the same contact rule."""
    bg, structure, a, b = map(int, palette[:4])
    anchors = {a: 17, b: 20}
    profiles = [(8, 2, [4, 2, 3, 1, 4], 1, a),
                (15, 0, [3, 5, 2, 4, 3], 2, b),
                (21, 4, [3, 1, 2, 2, 3], 0, a),
                (27, 1, [2, 3, 1, 2, 2], 3, b)]
    objects = []
    for base, left, heights, mark_c, marker in profiles[:n]:
        cells = {(base, left + c) for c in range(5)}
        for c, height in enumerate(heights):
            cells.update((r, left + c) for r in range(base - height, base))
        mark = base, left + mark_c
        dx = anchors[marker] - mark[1]
        objects.append({"cells": [(r, c, marker if (r, c) == mark else structure)
                                  for r, c in sorted(cells)], "dx": dx})
    pins = []
    # Variant and random payloads preserve geometry and all witnesses.
    payload = list(map(int, palette[4:7]))
    for i, (c, length) in enumerate(((12, 3), (16, 10), (18, 13), (20, 19), (22, 18))):
        top = min((r for o in objects for r, cc, _ in o["cells"] if cc + o["dx"] == c), default=30)
        shift = max(0, length - top + 1)
        seq = _sequence(rng, length, structure, payload, shift,
                        "order" if i == 1 else "prefix" if i == 2 else "")
        if i == 0:
            # Three ternary payload cells distinguish 27 fallback inputs,
            # more than the maximum 13 previous pairs, without extra storage.
            seq = tuple(payload[(variant // (3 ** j)) % 3] for j in range(length))
        pins.append((c, seq))
    return _label(30, 30, bg, structure, anchors, objects, pins)


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("counts must be integers")
        if not lo <= int(value) <= hi:
            raise ValueError("counts outside legal range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = rng.permutation(10)
    changing_colors = bool(rng.random() < .75)
    orientation = int(rng.integers(4))
    mixed_orientation = bool(rng.random() < .5)
    train_cap = int(rng.choice((2, 3)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            n = (train_cap if index == 0 else int(rng.integers(2, train_cap + 1))) if split == "train" else train_cap + 1
            colors = rng.permutation(10) if changing_colors else palette
            witness = split == "train" and index == 0
            turns = int(rng.integers(4)) if mixed_orientation else orientation
            pair = None
            for _ in range(160):
                candidate = _sample_scene(rng, n, colors, witness)
                if candidate is not None:
                    candidate = {k: np.rot90(v, turns).copy() for k, v in candidate.items()}
                    if _key(candidate["input"]) not in seen:
                        pair = candidate
                        break
            if pair is None:
                # Twenty distinct stationary-pin codes exceed 13 old inputs.
                # Check uniqueness in the final orientation, not a latent view.
                for variant in range(20):
                    candidate = _fallback(rng, n, colors, variant)
                    candidate = {k: np.rot90(v, turns).copy() for k, v in candidate.items()}
                    if _key(candidate["input"]) not in seen:
                        pair = candidate
                        break
            _need(pair is not None, "distinct input construction exhausted")
            _need(_key(pair["input"]) not in seen, "duplicate input")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_features = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape, "canvas must be preserved")
            _need(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _decode(inp)
            _need(np.array_equal(solve(inp), out), "output does not follow registered pressure")
            _need(not np.array_equal(inp, out), "unchanged puzzle pair")
            # Formula check is independent of solve's one-cell-at-a-time loop.
            canonical_out = np.rot90(out, d["turns"])
            _need(np.array_equal(canonical_out[0], d["a"][0]), "wall changed")
            for color, p in d["anchors"].items():
                _need(int(canonical_out[p]) == color, "anchor moved")
            for pin in d["pins"]:
                surviving = pin["seq"][pin["shift"]:]
                _need(len(surviving) >= 1, "complete pin loss")
                actual = tuple(map(int, canonical_out[1:len(surviving) + 1, pin["col"]]))
                _need(actual == surviving, "wrong pin order or wall clipping")
                _need(len(surviving) < pin["top"], "pin crosses contact envelope")
            for piece in d["pieces"]:
                _need(all(int(canonical_out[p]) == v for p, v in piece["target"].items()),
                      "piece deformed or recolored")
            if split == "train":
                train_features.append(_features(d))
    _need(any(f["shared"] and f["owners"] >= 2 for f in train_features), "shared-envelope witness missing")
    for flag in ("rank_cross", "eccentric", "stationary", "ordered_survivor", "moving_prefix"):
        _need(any(f[flag] for f in train_features), "missing training evidence: " + flag)
    _need(any(len({s for s in f["shifts"] if s > 0}) >= 2 for f in train_features),
          "lane-specific shifts missing")


def _focused_regressions():
    """Small hand-calculated contact, shielding, order and marker experiments."""
    objects = []
    for base, left, heights, mark_col, marker, guide in (
            (7, 2, (4, 1, 3), 3, 2, 10), (12, 0, (2, 4, 1, 2, 3), 2, 3, 13)):
        cells = {(base, left + c) for c in range(len(heights))}
        for c, height in enumerate(heights):
            cells.update((r, left + c) for r in range(base - height, base))
        objects.append({"cells": [(r, c, marker if (r, c) == (base, mark_col) else 5)
                                  for r, c in sorted(cells)], "dx": guide - mark_col})
    pins = [(6, (6, 4)), (9, (5, 5, 5, 8, 8, 4, 6, 1)),
            (11, (5,) * 8 + (8, 8)), (13, (4,) * 12)]
    pair = _label(15, 17, 0, 5, {2: 10, 3: 13}, objects, pins)
    result = solve(pair["input"])
    _need(np.array_equal(result, pair["output"]), "focused constructor disagreement")
    for col, expected in ((6, (6, 4)), (9, (6, 1)), (11, (5, 8, 8)), (13, (4,) * 10)):
        _need(tuple(map(int, result[1:len(expected) + 1, col])) == expected,
              "hand-calculated surviving sequence")
    d = _decode(pair["input"])
    _need([p["shift"] for p in d["pins"]] == [0, 6, 7, 2], "contact arithmetic")
    _need(d["pins"][2]["competitors"] == 2 and d["pins"][2]["top"] == 4,
          "higher piece must shield the lower one at column 11")
    # Same pieces and same pins, but a different visible registration and guide.
    # The first piece now crosses the left canvas edge; column 9 becomes unhit.
    changed = pair["input"].copy()
    changed[14, 10], changed[14, 1] = 0, 2
    changed[7, 3], changed[7, 4] = 5, 2
    altered = solve(changed)
    _need(tuple(map(int, altered[1:9, 9])) == pins[1][1], "registration must alter pin contact")
    cd = _decode(changed)
    _need(sum(p["clipped"] for p in cd["pieces"]) == 5, "exact piece clipping count")
    _need(int(altered[7, 1]) == 2 and int(altered[14, 1]) == 2, "marker or anchor lost")
    for length in (1, 2):
        changed = pair["input"].copy()
        changed[1:9, 9] = 0
        changed[1:1 + length, 9] = 6
        pd = next(p for p in _decode(changed)["pins"] if p["col"] == 9)
        _need(pd["shift"] == 0, "gap and exact contact must not pull pins")
        _need(np.all(solve(changed)[1:1 + length, 9] == 6), "stationary pin changed")
    # Two identities inside one piece would leave its intended guide ambiguous.
    ambiguous = pair["input"].copy()
    ambiguous[7, 2] = 3
    try:
        solve(ambiguous)
    except AssertionError:
        pass
    else:
        raise AssertionError("ambiguous registration accepted")


def self_test():
    """42 seeds in memory; aggregate metrics only, with no disk writes or traces."""
    _focused_regressions()
    fingerprints, shapes, profiles, sizes = set(), set(), set(), set()
    features = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        split_features = {}
        for split in puzzle:
            split_features[split] = []
            for pair in puzzle[split]:
                d = _decode(pair["input"])
                f = _features(d)
                features.append(f)
                split_features[split].append(f)
                shapes.add(tuple(f["shape"]))
                profiles.update(f["profiles"])
                sizes.update(f["sizes"])
        _need(min(f["pieces"] for f in split_features["test"]) >
              max(f["pieces"] for f in split_features["train"]), "missing count transfer")
    _need(len(fingerprints) == 42 and len(shapes) >= 35 and len(profiles) >= 60, "structural variation")
    _need({f["pieces"] for f in features} == {2, 3, 4}, "piece counts")
    _need({f["turns"] for f in features} == set(range(4)), "wall orientations")
    _need({f["background"] for f in features} == set(range(10)), "background roles")
    _need({f["structure"] for f in features} == set(range(10)), "structural roles")
    _need(len(sizes) >= 12, "shape sizes")
    for flag in ("shared", "rank_cross", "eccentric", "piece_clips", "contact", "gap", "repeated_identity"):
        _need(sum(bool(f[flag]) for f in features) >= 3, "variation lacks " + flag)
    p = generate_puzzle(719, 2, 1)
    _need(_fingerprint(p) == _fingerprint(generate_puzzle(719, 2, 1)), "seed determinism")
    rng = np.random.default_rng(20261022)
    for n in (2, 3, 4):
        signatures = set()
        for variant in range(20):
            pair = _fallback(rng, n, np.arange(10), variant)
            signatures.add(_key(pair["input"]))
            _need(np.array_equal(solve(pair["input"]), pair["output"]), "fallback semantics")
            f = _features(_decode(pair["input"]))
            _need(f["shared"] and f["owners"] >= 2 and f["rank_cross"]
                  and f["moving_prefix"] and f["ordered_survivor"], "fallback training evidence")
        _need(len(signatures) == 20, "fallback uniqueness guarantee")
    for turns in range(4):
        mapping = rng.permutation(10)
        transformed = {s: [{k: mapping[np.rot90(v, turns)].astype(np.int8)
                            for k, v in pair.items()} for pair in p[s]] for s in p}
        validate_generated_puzzle(transformed)
    # Both concept omissions change the pin payload result, not only piece locations.
    d = _decode(p["train"][0]["input"])
    _need(any(x["shift"] > 0 for x in d["pins"]), "A omission is invisible")
    for pin in d["pins"]:
        # Without B, original separated pieces are already clear of all pins.
        _need(all(not (c == pin["col"] and r <= len(pin["seq"]))
                  for piece in d["pieces"] for r, c in piece["cells"]), "source contact")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            q = generate_puzzle(900 + 10 * a + b, a, b)
            validate_generated_puzzle(q)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": len(features), "distinct_puzzles": len(fingerprints),
            "canonical_shapes": len(shapes), "piece_profiles": len(profiles),
            "piece_sizes": sorted(sizes),
            "feature_pairs": {flag: sum(bool(f[flag]) for f in features) for flag in
                              ("shared", "rank_cross", "eccentric", "piece_clips", "contact", "gap", "repeated_identity")}}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
