"""A turning courier collects transparent, key-selected stencil layers.

No evidence files, seeds, or previously generated examples are used by solve.
Generation paints labels with bitmap rotations; solving carries sparse offsets.
No files are written by generation, validation, or the bounded self-test.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= g) & (g <= 9))), "colors outside 0..9")
    return g.astype(np.int8, copy=False)


def _rot(p, turns):
    r, c = p
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _direction(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    _require((dr == 0) != (dc == 0), "nonorthogonal or repeated route point")
    return (1 if dc > 0 else 3) if dr == 0 else (2 if dr > 0 else 0)


def _distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _components(cells, diagonal=False):
    unseen = set(cells)
    result = []
    neighbors = ([(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r, c) != (0, 0)] if diagonal else DIRS)
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        component, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _read_rack(rack, frame):
    """Enclosure groups even disconnected paint; a card's center is its key."""
    fill = int(rack[0, 0])
    _require(fill != frame, "rack fill/frame collision")
    _require(np.all(rack[0] == fill) and np.all(rack[-1] == fill)
             and np.all(rack[:, 0] == fill) and np.all(rack[:, -1] == fill),
             "rack must have a fill margin")
    parts = _components(zip(*np.where(rack == frame)))
    _require(2 <= len(parts) <= 3, "two or three cards required")
    templates, radii, mask = {}, set(), np.zeros(rack.shape, dtype=bool)
    for part in parts:
        top = min(r for r, c in part)
        left = min(c for r, c in part)
        bottom = max(r for r, c in part)
        right = max(c for r, c in part)
        side = bottom - top + 1
        _require(side == right - left + 1 and side in (7, 9), "square card size")
        border = {(r, c) for r in range(top, bottom + 1)
                  for c in range(left, right + 1)
                  if r in (top, bottom) or c in (left, right)}
        _require(part == border, "incomplete or connected card frames")
        _require(not mask[top:bottom + 1, left:right + 1].any(), "overlapping cards")
        mask[top:bottom + 1, left:right + 1] = True
        radius = (side - 3) // 2
        radii.add(radius)
        ar, ac = (top + bottom) // 2, (left + right) // 2
        key = int(rack[ar, ac])
        _require(key not in (fill, frame) and key not in templates, "invalid card key")
        paint = {}
        for r in range(top + 1, bottom):
            for c in range(left + 1, right):
                if (r, c) == (ar, ac):
                    continue
                color = int(rack[r, c])
                if color != fill:
                    _require(max(abs(r - ar), abs(c - ac)) == radius, "paint outside ring")
                    _require(color not in (frame, key), "key/paint collision")
                    paint[r - ar, c - ac] = color
        _require(3 <= len(paint) <= 10, "paint count")
        _require(all(n >= 2 for n in Counter(paint.values()).values()), "ambiguous singleton paint")
        templates[key] = paint
    _require(len(radii) == 1, "inconsistent cargo radius")
    _require(np.all(rack[~mask] == fill), "unframed rack contents")
    colors = set().union(*(set(t.values()) for t in templates.values()))
    _require(colors.isdisjoint(templates), "paint must not be a key")
    return templates, radii.pop(), fill, colors


def _identify_arrow(cells, radius):
    matches = []
    for tip in sorted(cells):
        for d in range(4):
            offsets = {_rot((r - tip[0], c - tip[1]), 1 - d) for r, c in cells}
            if any(c > 0 or max(abs(r), abs(c)) >= radius for r, c in offsets):
                continue
            if {p for p in offsets if p[1] == 0} != {(0, 0)}:
                continue
            if max(abs(r) for r, c in offsets) == 0:
                continue
            if {(-r, c) for r, c in offsets} != offsets:
                continue
            matches.append((tuple(sorted(offsets)), tip, d))
    _require(len(matches) == 1, "arrow needs a unique tip and heading")
    return matches[0]


def _route(tip, stations):
    remaining = dict(stations)
    points, headings, keys = [tip], [], []
    while remaining:
        rays = {}
        for q in remaining:
            if tip[0] == q[0] or tip[1] == q[1]:
                d = _direction(tip, q)
                rays.setdefault(d, []).append(q)
        _require(len(rays) == 1, "route must have exactly one occupied ray")
        d, candidates = next(iter(rays.items()))
        q = min(candidates, key=lambda p: _distance(tip, p))
        points.append(q)
        headings.append(d)
        keys.append(remaining.pop(q))
        tip = q
    return points, headings, keys


def _read_board(board, templates, radius, rack_fill, frame, paint_colors):
    colors, counts = np.unique(board, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(int(max(counts)) > board.size // 2, "board background must dominate")
    keys = set(templates)
    excluded = {rack_fill, frame} | keys | paint_colors
    _require(bg not in excluded, "board background role collision")
    body_colors = set(int(c) for c in colors) - keys - {bg}
    _require(len(body_colors) == 1, "one courier ink required")
    ink = body_colors.pop()
    _require(ink not in excluded, "courier ink role collision")
    body = set(zip(*np.where(board == ink)))
    _require(len(_components(body, diagonal=True)) == 1, "disconnected courier")
    mask, tip, initial = _identify_arrow(body, radius)
    stations = {(int(r), int(c)): int(board[r, c])
                for r, c in zip(*np.where((board != bg) & (board != ink)))}
    _require(len(stations) >= 2 and set(stations.values()) <= keys, "unknown stations")
    all_parts = _components(body | set(stations), diagonal=True)
    _require(len(all_parts) == len(stations) + 1, "input components must be separated")
    points, headings, ordered_keys = _route(tip, stations)
    _require(headings[0] == initial, "first leg must follow the tip")
    _require(all(radius <= r < board.shape[0] - radius
                 and radius <= c < board.shape[1] - radius for r, c in points),
             "cargo would cross a boundary")
    return dict(board=board, bg=bg, ink=ink, mask=mask, tip=tip, initial=initial,
                points=points, headings=headings, keys=ordered_keys,
                templates=templates, radius=radius, rack_fill=rack_fill, frame=frame)


def _decode(input_grid):
    g = _grid(input_grid)
    interpretations = []
    # Full background lines are considered, then rejected by actual card grammar.
    for axis in (0, 1):
        for i in range(1, g.shape[axis] - 1):
            line = g[i, :] if axis == 0 else g[:, i]
            if not np.all(line == line[0]):
                continue
            frame = int(line[0])
            halves = ((g[:i, :], g[i + 1:, :]) if axis == 0
                      else (g[:, :i], g[:, i + 1:]))
            for rack_index in (0, 1):
                rack, board = halves[rack_index], halves[1 - rack_index]
                if frame in board or frame not in rack:
                    continue
                try:
                    templates, radius, fill, paint = _read_rack(rack, frame)
                    info = _read_board(board, templates, radius, fill, frame, paint)
                    info.update(axis=axis, rack_index=rack_index, input_shape=g.shape)
                    interpretations.append(info)
                except ValueError:
                    continue
    _require(len(interpretations) == 1, "input needs one rack/route interpretation")
    return interpretations[0]


def _carry(info, mode="normal"):
    cargo, heading = {}, info["initial"]
    overwrites = 0
    for d, key in zip(info["headings"], info["keys"]):
        delta = (d - heading) % 4
        if mode != "no_rotation":
            cargo = {_rot(p, delta): color for p, color in cargo.items()}
        if mode == "last_only":
            cargo = {}
        for p, color in info["templates"][key].items():
            if mode == "orient_new":
                p = _rot(p, d - 1)
            if p in cargo and cargo[p] != color:
                overwrites += 1
            if mode != "first_wins" or p not in cargo:
                cargo[p] = color
        heading = d
    return cargo, overwrites


def _draw_terminal(info, cargo):
    result = np.full(info["board"].shape, info["bg"], dtype=np.int8)
    ar, ac = info["points"][-1]
    for p in info["mask"]:
        r, c = _rot(p, info["headings"][-1] - 1)
        result[ar + r, ac + c] = info["ink"]
    result[ar, ac] = info["keys"][-1]
    for (dr, dc), color in cargo.items():
        result[ar + dr, ac + dc] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(input_grid)
    return _draw_terminal(info, _carry(info)[0])


def _counterfactuals(info):
    expected = _draw_terminal(info, _carry(info)[0])
    alternatives = {mode: not np.array_equal(expected, _draw_terminal(info, _carry(info, mode)[0]))
                    for mode in ("no_rotation", "last_only", "orient_new", "first_wins")}
    alternatives["no_templates"] = not np.array_equal(expected, _draw_terminal(info, {}))
    return alternatives


def _random_templates(rng, radius, keys, paints):
    ring = [(r, c) for r in range(-radius, radius + 1)
            for c in range(-radius, radius + 1) if max(abs(r), abs(c)) == radius]
    templates = {}
    for i, key in enumerate(keys):
        count = int(rng.integers(4, 11))
        choices = rng.choice(len(ring), count, replace=False)
        multicolor = rng.random() < 0.45
        assignment = ([paints[0], paints[0], paints[1], paints[1]]
                      + [int(rng.choice(paints)) for _ in range(count - 4)]
                      if multicolor else [paints[i % len(paints)]] * count)
        rng.shuffle(assignment)
        templates[key] = {ring[int(j)]: int(color) for j, color in zip(choices, assignment)}
    return templates


def _random_mask(rng, radius):
    depth = int(rng.integers(1, radius))
    solid = bool(rng.integers(2))
    mask = {(0, 0)}
    for c in range(1, depth + 1):
        rows = range(-c, c + 1) if solid else (-c, c)
        mask.update((r, -c) for r in rows)
    return tuple(sorted(mask))


def _grow_route(rng, shape, radius, mask, count):
    """Incremental free-coordinate walk; no fixed path bank or spiral."""
    h, w = shape
    for _ in range(48):
        points = [(int(rng.integers(radius, h - radius)),
                   int(rng.integers(radius, w - radius)))]
        headings, start_body = [], None
        for step in range(count):
            options = {}
            for d, (dr, dc) in enumerate(DIRS):
                initial_body = (start_body if start_body is not None else
                                {(points[0][0] + r, points[0][1] + c)
                                 for r, c in (_rot(p, d - 1) for p in mask)})
                for distance in range(2, max(shape)):
                    q = points[-1][0] + dr * distance, points[-1][1] + dc * distance
                    if not (radius <= q[0] < h - radius and radius <= q[1] < w - radius):
                        continue
                    if any(max(abs(q[0] - r), abs(q[1] - c)) <= 1
                           for r, c in set(points) | initial_body):
                        continue
                    valid = True
                    for a, b in zip(points[:-1], points[1:]):
                        if a[0] == q[0] or a[1] == q[1]:
                            if _direction(a, q) != _direction(a, b) or _distance(a, q) <= _distance(a, b):
                                valid = False
                                break
                    if valid:
                        options.setdefault(d, []).append((q, initial_body))
            if not options:
                break
            ds = list(options)
            weights = np.array([1.3 if headings and d == headings[-1] else 1.0 for d in ds])
            d = int(rng.choice(ds, p=weights / weights.sum()))
            choices = options[d]
            q, start_body = choices[int(rng.integers(len(choices)))]
            points.append(q)
            headings.append(d)
        if len(headings) == count:
            return points, headings
    return None


def _bitmap_label(shape, bg, ink, mask, points, headings, keys, templates, radius):
    """Construction uses dense paint and NumPy rotations, separate from _carry."""
    cargo = np.full((2 * radius + 1, 2 * radius + 1), bg, dtype=np.int8)
    previous = headings[0]
    for d, key in zip(headings, keys):
        cargo = np.rot90(cargo, -(d - previous)).copy()
        for (dr, dc), color in templates[key].items():
            cargo[radius + dr, radius + dc] = color
        previous = d
    out = np.full(shape, bg, dtype=np.int8)
    ar, ac = points[-1]
    # Rotate a bitmap of the ACTUAL sampled mask, not a shape-size lookup.
    body = np.zeros(cargo.shape, dtype=bool)
    for dr, dc in mask:
        body[radius + dr, radius + dc] = True
    body = np.rot90(body, -(headings[-1] - 1))
    patch = out[ar - radius:ar + radius + 1, ac - radius:ac + radius + 1]
    patch[body] = ink
    patch[radius, radius] = keys[-1]
    painted = cargo != bg
    patch[painted] = cargo[painted]
    return out


def _make_pair(rng, count, palette, teach):
    # Retry both geometry and content, retaining no failed candidates or logs.
    for _ in range(240):
        radius = int(rng.choice([2, 2, 3]))
        nkeys = int(rng.integers(2, 4)) if radius == 2 else 2
        ref_fill, bg, frame, ink = (int(c) for c in palette[:4])
        key_colors = [int(c) for c in palette[4:4 + nkeys]]
        paints = [int(c) for c in palette[7:9]]
        templates = _random_templates(rng, radius, key_colors, paints)
        side = 2 * radius + 3
        gap = int(rng.integers(1, 3))
        required = nkeys * side + (nkeys - 1) * gap + 2
        if required > 30:
            continue
        width = int(rng.integers(required, 31))
        rack_height = side + 2
        height = int(rng.integers(14 if radius == 2 else 15, 30 - rack_height))
        mask = _random_mask(rng, radius)
        grown = _grow_route(rng, (height, width), radius, mask, count)
        if grown is None:
            continue
        points, headings = grown
        requested = list(key_colors)
        requested += [int(rng.choice(key_colors)) for _ in range(count - nkeys)]
        rng.shuffle(requested)
        board = np.full((height, width), bg, dtype=np.int8)
        sr, sc = points[0]
        for dr, dc in (_rot(p, headings[0] - 1) for p in mask):
            board[sr + dr, sc + dc] = ink
        for (r, c), key in zip(points[1:], requested):
            board[r, c] = key
        rack = np.full((rack_height, width), ref_fill, dtype=np.int8)
        order = rng.permutation(nkeys)
        left = int(rng.integers(1, width - (required - 2)))
        for j in order:
            key = key_colors[int(j)]
            rack[1:1 + side, left:left + side] = frame
            rack[2:side, left + 1:left + side - 1] = ref_fill
            ar, ac = 1 + side // 2, left + side // 2
            rack[ar, ac] = key
            for (dr, dc), color in templates[key].items():
                rack[ar + dr, ac + dc] = color
            left += side + gap
        divider = np.full((1, width), frame, dtype=np.int8)
        inp = np.concatenate([rack, divider, board], axis=0)
        out = _bitmap_label(board.shape, bg, ink, mask, points, headings,
                            requested, templates, radius)
        turns = int(rng.integers(4))
        inp, out = np.rot90(inp, turns).copy(), np.rot90(out, turns).copy()
        info = _decode(inp)
        _require(np.array_equal(_draw_terminal(info, _carry(info)[0]), out), "construction mismatch")
        if teach and not all(_counterfactuals(info).values()):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded construction exhausted; no misleading fallback label")


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(g):
    return g.shape, g.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    shared = bool(rng.random() < 0.2)
    palette = rng.permutation(10)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            length = (3 + index % 2 if split == "train" and index < 2 else
                      int(rng.integers(3, 5)) if split == "train" else int(rng.integers(5, 8)))
            for _ in range(12):
                pair = _make_pair(rng, length, palette if shared else rng.permutation(10),
                                  teach=split == "train" and index < 2)
                fingerprint = _key(pair["input"])
                if fingerprint not in seen:
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
            seen.add(fingerprint)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Raise AssertionError for a malformed label, scene, or teaching regime."""
    try:
        _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
        _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
        _counts(len(puzzle["train"]), len(puzzle["test"]))
        seen, lengths = set(), {}
        for split in ("train", "test"):
            lengths[split] = []
            for i, pair in enumerate(puzzle[split]):
                _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
                inp, out = _grid(pair["input"]), _grid(pair["output"])
                fingerprint = _key(inp)
                _require(fingerprint not in seen, "duplicate input")
                seen.add(fingerprint)
                info = _decode(inp)
                cargo, _ = _carry(info)
                _require(all(max(abs(r), abs(c)) == info["radius"] for r, c in cargo), "cargo ring")
                _require(set(info["keys"]) == set(info["templates"]), "unused reference key")
                _require(np.array_equal(_draw_terminal(info, cargo), out), "incorrect terminal state")
                lengths[split].append(len(info["keys"]))
                if split == "train" and i < 2:
                    _require(all(_counterfactuals(info).values()), "missing teaching witness")
        _require(set(lengths["train"]) <= {3, 4}, "training length regime")
        _require(set(lengths["train"]) == {3, 4}, "training must vary pickup count")
        _require(set(lengths["test"]) <= {5, 6, 7}, "test length regime")
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise AssertionError(f"invalid generated puzzle: {exc}") from exc


def _signature(points):
    origin = points[0]
    return min(tuple(_rot((r - origin[0], sign * (c - origin[1])), t) for r, c in points)
               for sign in (-1, 1) for t in range(4))


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        h.update(split.encode())
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def self_test():
    """Stream 42 seeds and return bounded, grid-derived descriptive measurements."""
    signatures, routes, templates, masks, sizes, palettes = (set() for _ in range(6))
    radii, key_counts, lengths, initial, terminal, layouts = (set() for _ in range(6))
    mixed_turns = straight = multi_paint = overwrite_cases = 0
    total_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                routes.add(_signature(info["points"]))
                masks.add(info["mask"])
                templates.update(tuple(sorted(t)) for t in info["templates"].values())
                sizes.add(info["board"].shape)
                palettes.add((info["bg"], info["ink"], tuple(sorted(info["templates"]))))
                radii.add(info["radius"])
                key_counts.add(len(info["templates"]))
                lengths.add((split, len(info["keys"])))
                initial.add(info["initial"])
                terminal.add(info["headings"][-1])
                layouts.add((info["axis"], info["rack_index"]))
                changes = [(b - a) % 4 for a, b in zip(info["headings"], info["headings"][1:])]
                mixed_turns += int(1 in changes and 3 in changes)
                straight += int(0 in changes)
                multi_paint += int(any(len(set(t.values())) > 1 for t in info["templates"].values()))
                overwrite_cases += int(_carry(info)[1] > 0)
                total_pairs += 1
        # Equivariance checks use only inputs, and transformed expected labels.
        if seed < 4:
            pair = puzzle["test"][0]
            for k in (1, 2, 3):
                assert np.array_equal(solve(np.rot90(pair["input"], k)), np.rot90(pair["output"], k))
            perm = np.random.default_rng(1000 + seed).permutation(10)
            assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
    assert len(signatures) == 42
    assert len(routes) >= 140 and len(templates) >= 200 and len(sizes) >= 45
    assert len(masks) >= 4 and len(palettes) >= 100
    assert radii == {2, 3} and key_counts == {2, 3}
    assert lengths == {("train", 3), ("train", 4), ("test", 5), ("test", 6), ("test", 7)}
    assert initial == terminal == set(range(4))
    assert layouts == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert mixed_turns > 40 and straight > 30 and multi_paint > 50 and overwrite_cases > 100
    assert _digest(generate_puzzle(909, 2, 1)) == _digest(generate_puzzle(909, 2, 1))
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(100 + a + b, a, b)
        assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (2, 2), (3, 4), (6, 5), (4, 0),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("changed label accepted")
    try:
        _route((4, 4), {(4, 7): 1, (7, 4): 2})
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous occupied rays accepted")
    return {"seeds": 42, "pairs": total_pairs, "unique_puzzles": len(signatures),
            "normalized_routes": len(routes), "paint_masks": len(templates),
            "arrow_masks": len(masks), "board_shapes": len(sizes),
            "role_palettes": len(palettes), "radii": sorted(radii),
            "key_counts": sorted(key_counts), "mixed_turn_routes": mixed_turns,
            "straight_continuation_routes": straight, "multicolor_card_pairs": multi_paint,
            "different_color_overwrite_pairs": overwrite_cases}
