"""Registration-latched gravity: a boundary guide is a virtual support.

No parent module, training example, seed or construction record is consulted by
solve. Generation uses explicit rigid pieces; decoding reconstructs them from
the input. See task_spec.md for the complete conventions and supported subset.
"""
from collections import Counter

import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 281303999, 344290349, 375541589, 885878165, 453456576,
              21586007, 130110879, 474191023, 550117862, 911852391,
              310217828, 657274779, 623354323, 414121096, 302105999,
              816184629, 602316737, 414475275, 885089213, 504996456,
              939940334, 792576937, 929547101, 700602042, 590192320,
              727279138, 283015318, 339701311, 483666549, 681603364,
              33215773, 990037784, 96758652, 24576044, 925096685,
              574020011, 31832547, 484249099, 565936172, 222162170,
              851207030)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors outside 0..9")
    return a.astype(np.int8, copy=True)


def _connected(points):
    if not points:
        return False
    start = min(points)
    reached, todo = {start}, [start]
    while todo:
        r, c = todo.pop()
        for dr, dc in DIRS:
            p = r + dr, c + dc
            if p in points and p not in reached:
                reached.add(p)
                todo.append(p)
    return len(reached) == len(points)


def _components(grid, background):
    remaining = {tuple(map(int, p)) for p in np.argwhere(grid != background)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        points, todo = {start}, [start]
        color = int(grid[start])
        while todo:
            r, c = todo.pop()
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in remaining and int(grid[p]) == color:
                    remaining.remove(p)
                    points.add(p)
                    todo.append(p)
        result.append((color, points))
    return result


def _piece(cells, marker=None, target=None, rod_color=None):
    return {"cells": dict(cells), "marker": marker, "target": target,
            "rod_color": rod_color}


def _shift(piece, d):
    return _piece({(r + d, c): v for (r, c), v in piece["cells"].items()},
                  None if piece["marker"] is None else
                  (piece["marker"][0] + d, piece["marker"][1]),
                  piece["target"], piece["rod_color"])


def _settle(pieces, height, ignore_guides=False):
    """Greatest movable set at each tick; no processing-order tie breaker."""
    result = [_shift(p, 0) for p in pieces]
    for _ in range(height + 1):
        cells = [set(p["cells"]) for p in result]
        targets = [{(r + 1, c) for r, c in s} for s in cells]
        moving = {i for i, p in enumerate(result)
                  if all(r < height for r, c in targets[i]) and
                  (ignore_guides or p["marker"] is None or
                   p["marker"][0] < p["target"])}
        while True:
            stationary = set(range(len(result))) - moving
            blocked = {i for i in moving
                       if any(targets[i] & cells[j] for j in stationary)}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return result
        result = [_shift(p, 1 if i in moving else 0) for i, p in enumerate(result)]
    raise AssertionError("gravity did not terminate")


def _decode(input_grid):
    g = _grid(input_grid)
    h, w = g.shape
    _require(h >= 8 and w >= 7, "canvas too small for the conventions")
    bg = int(g[1, 0])
    _require(np.all(g[1] == bg), "missing blank separation row")
    _require(g[0, 0] == bg and g[0, -1] == bg, "control corners must be blank")
    commands = [int(x) for x in g[0] if x != bg]
    _require(1 <= len(commands) <= 3 and len(set(commands)) == len(commands),
             "one to three distinct command swatches required")
    _require(g[-1, 0] == bg and g[-1, -1] == bg, "floor corners must be blank")
    guides, anchors = {}, {}
    for r in range(2, h - 1):
        for c in (0, w - 1):
            color = int(g[r, c])
            if color != bg:
                _require(color not in commands, "command/guide roles overlap")
                _require(color not in guides or guides[color] == r, "inconsistent guide")
                guides[color] = r
                anchors[r, c] = color
                _require(g[r, 1 if c == 0 else w - 2] == bg,
                         "boundary marker touches construction")
                _require(g[r - 1, c] == bg and g[r + 1, c] == bg,
                         "boundary markers must be isolated")
    _require(1 <= len(guides) <= 3, "one to three guides required")
    scene = g.copy()
    scene[:2] = bg
    scene[:, 0] = scene[:, -1] = bg
    comps = _components(scene, bg)
    owner = {p: i for i, (_, points) in enumerate(comps) for p in points}
    used, pieces, body_colors = set(), [], set()
    for i, (color, points) in enumerate(comps):
        if color not in guides:
            continue
        _require(len(points) == 1, "guide identity must occur once per cargo")
        mark = next(iter(points))
        adjacent = {owner[mark[0] + dr, mark[1] + dc] for dr, dc in DIRS
                    if (mark[0] + dr, mark[1] + dc) in owner}
        _require(len(adjacent) == 1, "registration must touch exactly one body")
        j = next(iter(adjacent))
        body_color, body = comps[j]
        _require(body_color not in guides and j not in used and len(body) >= 3,
                 "ambiguous or undersized cargo body")
        _require(mark[0] <= guides[color], "cargo starts beyond its guide")
        used.update((i, j))
        body_colors.add(body_color)
        pieces.append(_piece({p: int(g[p]) for p in body | points}, mark, guides[color]))
    _require(pieces and set(guides) == {int(g[p["marker"]]) for p in pieces},
             "guide without cargo")
    _require(not body_colors.intersection(commands), "commands cannot select cargo bodies")
    for i, (color, points) in enumerate(comps):
        if i in used:
            continue
        _require(color not in guides and color not in body_colors,
                 "unmarked material reuses a cargo role")
        rows, cols = {r for r, _ in points}, {c for _, c in points}
        _require(len(points) >= 2 and (len(rows) == 1 or len(cols) == 1),
                 "unmarked pieces must be straight rods")
        pieces.append(_piece({p: color for p in points}, rod_color=color))
    _require(any(p["rod_color"] is not None for p in pieces), "no rod scaffold")
    return g, bg, commands, guides, anchors, pieces


def _render(shape, bg, band, anchors, pieces):
    out = np.full(shape, bg, dtype=np.int8)
    out[0] = band
    for p, color in anchors.items():
        out[p] = color
    occupied = set(anchors)
    for piece in pieces:
        for p, color in piece["cells"].items():
            _require(2 <= p[0] < shape[0] and 1 <= p[1] < shape[1] - 1,
                     "construction outside arena")
            _require(p not in occupied, "piece overlap")
            occupied.add(p)
            out[p] = color
    return out


def _predict_decoded(decoded, ignore_guides=False, ignore_deletion=False, ignore_gravity=False):
    g, bg, commands, guides, anchors, pieces = decoded
    survivors = [p for p in pieces if ignore_deletion or p["rod_color"] not in commands]
    final = survivors if ignore_gravity else _settle(survivors, g.shape[0], ignore_guides)
    return _render(g.shape, bg, g[0], anchors, final)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode commands, rigid pieces and virtual stops entirely from the input."""
    return _predict_decoded(_decode(input_grid))


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _same_pieces(a, b):
    return all(p["cells"] == q["cells"] and p["marker"] == q["marker"]
               for p, q in zip(a, b)) and len(a) == len(b)


def _edges(pieces):
    cells = [set(p["cells"]) for p in pieces]
    return [(i, j) for i in range(len(pieces)) for j in range(len(pieces))
            if i != j and {(r + 1, c) for r, c in cells[i]} & cells[j]]


def pair_features(pair):
    """Observable metrics and parent-ablation witnesses, without generator state."""
    decoded = _decode(pair["input"])
    g, bg, commands, guides, anchors, pieces = decoded
    survivors = [p for p in pieces if p["rod_color"] not in commands]
    final = _settle(survivors, g.shape[0])
    distances = [min(q["cells"])[0] - min(p["cells"])[0]
                 for p, q in zip(survivors, final)]
    edges = _edges(final)
    cargo = [p for p in pieces if p["marker"] is not None]
    locked_supports = [(i, j) for i, j in edges if final[j]["marker"] is not None and
                       final[j]["marker"][0] == final[j]["target"] and distances[j] > 0]
    return {"shape": g.shape, "background": bg, "guides": len(guides),
            "cargo": len(cargo), "rods": len(pieces) - len(cargo),
            "sizes": tuple(sorted(len(p["cells"]) for p in cargo)),
            "rod_lengths": tuple(sorted(len(p["cells"]) for p in pieces if p["marker"] is None)),
            "vertical_rods": sum(len({c for r, c in p["cells"]}) == 1 for p in pieces if p["marker"] is None),
            "horizontal_rods": sum(len({r for r, c in p["cells"]}) == 1 for p in pieces if p["marker"] is None),
            "commands": len(commands), "deleted": len(pieces) - len(survivors),
            "moved": sum(d > 0 for d in distances),
            "moved_cargo": sum(d > 0 and q["marker"] is not None
                               for d, q in zip(distances, final)),
            "falls": tuple(sorted(set(distances) - {0})),
            "reached": sum(d > 0 and q["marker"] is not None and
                           q["marker"][0] == q["target"] for d, q in zip(distances, final)),
            "blocked": sum(d > 0 and q["marker"] is not None and
                           q["marker"][0] < q["target"] for d, q in zip(distances, final)),
            "locked_supports": len(locked_supports),
            "edges": len(_edges(pieces)), "final_edges": len(edges),
            "bilateral": len(anchors) > len(guides),
            "off_center": sum(2 * p["marker"][0] != min(r for r, c in p["cells"]) +
                              max(r for r, c in p["cells"]) or
                              2 * p["marker"][1] != min(c for r, c in p["cells"]) +
                              max(c for r, c in p["cells"]) for p in cargo),
            "a_matters": not np.array_equal(_predict_decoded(decoded, ignore_guides=True), pair["output"]),
            "b_gravity_matters": not np.array_equal(_predict_decoded(decoded, ignore_gravity=True), pair["output"]),
            "b_selection_matters": not np.array_equal(_predict_decoded(decoded, ignore_deletion=True), pair["output"])}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "pair counts")
    seen, features = set(), {"train": [], "test": []}
    for split in features:
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "changed canvas")
            key = _signature(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            decoded = _decode(inp)
            pieces = decoded[-1]
            _require(_same_pieces(pieces, _settle(pieces, inp.shape[0])), "input not stable")
            survivors = [p for p in pieces if p["rod_color"] not in decoded[2]]
            final = _settle(survivors, inp.shape[0])
            _require(_same_pieces(final, _settle(final, inp.shape[0])), "output not stable")
            _require(np.array_equal(out, _predict_decoded(decoded)), "wrong latched settlement")
            _require(np.array_equal(out[:2], inp[:2]), "controls changed")
            _require(np.array_equal(out[:, (0, -1)], inp[:, (0, -1)]), "anchors changed")
            _require(all(Counter(p["cells"].values()) == Counter(q["cells"].values())
                         for p, q in zip(survivors, final)), "rigid piece colors changed")
            features[split].append(pair_features(pair))
    train = features["train"]
    _require(any(f["reached"] and f["blocked"] and f["locked_supports"] and
                 f["a_matters"] and f["b_gravity_matters"] and f["b_selection_matters"]
                 and f["off_center"] for f in train), "missing interacting demonstration")
    _require(any(f["deleted"] and not f["moved"] for f in train), "missing harmless selection demonstration")
    _require(all(f["cargo"] > max(t["cargo"] for t in train) for f in features["test"]),
             "missing cargo-count recombination shift")
    _require(all(f["reached"] and f["blocked"] and f["locked_supports"] and
                 f["moved_cargo"] >= 3 and f["a_matters"] and f["b_gravity_matters"]
                 for f in features["test"]), "test does not exercise the interaction")


def _shape(rng, h, w):
    """Connected bodies with a full bottom edge and varied upper geometry."""
    kind = int(rng.integers(5))
    if kind == 0:
        points = {(r, c) for r in range(h) for c in range(w)}
    elif kind == 1:
        points = {(r, c) for r in range(h) for c in range(w)
                  if r in (0, h - 1) or c in (0, w - 1)}
    elif kind == 2:
        points = {(h - 1, c) for c in range(w)} | {(r, 0) for r in range(h)}
        points |= {(r, w - 1) for r in range(h)}
    elif kind == 3:
        points = {(h - 1, c) for c in range(w)}
        for c in range(w):
            for r in range(h - int(rng.integers(1, h + 1)), h):
                points.add((r, c))
        points.add((0, int(rng.integers(w))))
        # Connect any tall isolated tip to the bottom spine.
        for r, c in list(points):
            points.update((rr, c) for rr in range(r, h))
    else:
        x = int(rng.integers(w))
        points = {(h - 1, c) for c in range(w)} | {(r, x) for r in range(h)}
        points |= {(0, c) for c in range(max(0, x - 1), min(w, x + 2))}
    if rng.random() < .5:
        points = {(r, w - 1 - c) for r, c in points}
    return points


def _mark_candidates(points, others):
    return [p for p in sorted(points) if _connected(points - {p}) and
            not any((p[0] + dr, p[1] + dc) in others for dr, dc in DIRS)]


def _anchors(rng, guides, width):
    anchors = {}
    for color, row in guides.items():
        ends = [0, width - 1] if rng.random() < .35 else [int(rng.choice([0, width - 1]))]
        for col in ends:
            if any((row + dr, col) in anchors for dr in (-1, 0, 1)):
                return None
            anchors[row, col] = color
    return anchors


def _band(rng, width, background, commands):
    band = np.full(width, background, dtype=np.int8)
    columns = rng.choice(np.arange(1, width - 1), size=len(commands), replace=False)
    for c, color in zip(columns, commands):
        band[c] = color
    return band


def _make_core(rng, palette, large=False):
    bg = palette[0]
    ga, gb = palette[1:3]
    ca, cb = palette[4:6]
    selected, landing = palette[7:9]
    hl, wl = int(rng.integers(2, 5)), int(rng.integers(4, 9))
    hu, wu = int(rng.integers(2, 5)), int(rng.integers(3, min(wl + 2, 8)))
    lower_r = hu + int(rng.integers(4, 7))
    short = int(rng.integers(1, 5))
    land_distance = short + int(rng.integers(2, 5))
    height = lower_r + hl + land_distance + int(rng.integers(3, 11))
    if height > 30:
        return None
    width = int(rng.integers(20 if large else max(11, wl + 5), 31 if large else 25))
    left = int(rng.integers(2, width - wl - 1))
    lp = {(lower_r + r, left + c) for r, c in _shape(rng, hl, wl)}
    bottom = lower_r + hl - 1
    x1, x2 = map(int, rng.choice(np.arange(left, left + wl), size=2, replace=False))
    tall = {(r, x1): selected for r in range(bottom + 1, height)}
    short_post = {(r, x2): landing for r in range(bottom + land_distance + 1, height)}
    if abs(x1 - x2) < 2:
        return None
    up0 = _shape(rng, hu, wu)
    upper_left = int(rng.integers(max(1, left - wu + 2), min(width - wu, left + wl - 1)))
    up = {(2 + r, upper_left + c) for r, c in up0}
    if up & lp:
        return None
    occupied = lp | set(tall) | set(short_post)
    for d in range(height):
        target = {(r + 1, c) for r, c in up}
        if target & occupied or any(r >= height for r, c in target):
            break
        up = target
    if not any((r + 1, c) in lp for r, c in up):
        return None
    lc = _mark_candidates(lp, up | set(tall) | set(short_post))
    uc = _mark_candidates(up, occupied)
    if not lc or not uc:
        return None
    # Deliberately eccentric handles teach registration rather than centering.
    lc = [p for p in lc if 2 * p[0] != lower_r + bottom or 2 * p[1] != 2 * left + wl - 1]
    if not lc:
        return None
    lm = lc[int(rng.integers(len(lc)))]
    um = uc[int(rng.integers(len(uc)))]
    guide_a = lm[0] + short
    choices = [r for r in range(max(3, um[0] + land_distance + 2), height - 1)
               if abs(r - guide_a) >= 2]
    if not choices:
        return None
    guide_b = int(rng.choice(choices))
    lower = _piece({p: ga if p == lm else ca for p in lp}, lm, guide_a)
    upper = _piece({p: gb if p == um else cb for p in up}, um, guide_b)
    pieces = [lower, upper, _piece(tall, rod_color=selected), _piece(short_post, rod_color=landing)]
    return (height, width), bg, {ga: guide_a, gb: guide_b}, pieces


def _insert(rng, shape, guides, pieces, palette, cargo):
    h, w = shape
    occupied = set().union(*(set(p["cells"]) for p in pieces))
    if cargo:
        hh, ww = int(rng.integers(2, 7)), int(rng.integers(3, 9))
        offsets = _shape(rng, hh, ww)
    else:
        vertical = bool(rng.integers(2))
        length = int(rng.integers(2, min(15 if vertical else 20,
                                       h - 3 if vertical else w - 3) + 1))
        offsets = {(i, 0) if vertical else (0, i) for i in range(length)}
    max_col = max(c for r, c in offsets)
    left = int(rng.integers(1, w - max_col - 1))
    if cargo and rng.random() < .65:
        # Favor recombined support paths, rather than only more isolated cargo.
        support_cols = {c for p in pieces if p["marker"] is not None for r, c in p["cells"]}
        choices = [x for x in range(1, w - max_col - 1)
                   if set(range(x, x + max_col + 1)) & support_cols]
        if choices:
            left = int(rng.choice(choices))
    points = {(2 + r, left + c) for r, c in offsets}
    if points & occupied:
        return False
    if cargo:
        candidates = _mark_candidates(points, set())
        if not candidates:
            return False
        mark = candidates[int(rng.integers(len(candidates)))]
        marker_colors = [c for c, row in guides.items() if row >= mark[0]]
        if not marker_colors:
            return False
        marker_color = (max(marker_colors, key=lambda c: guides[c]) if rng.random() < .6
                        else int(rng.choice(marker_colors)))
        color = int(rng.choice(palette[4:7]))
        piece = _piece({p: marker_color if p == mark else color for p in points},
                       mark, guides[marker_color])
    else:
        color = int(rng.choice(palette[7:10]))
        piece = _piece({p: color for p in points}, rod_color=color)
    new = _settle(pieces + [piece], h)
    if not _same_pieces(pieces, new[:-1]):
        return False
    # Check local decoding conventions after placement, including body joins.
    candidate = new[-1]
    if candidate["marker"] is not None:
        m = candidate["marker"]
        if any((m[0] + dr, m[1] + dc) in occupied for dr, dc in DIRS):
            return False
    for p in pieces:
        if p["marker"] is not None:
            m = p["marker"]
            if any((m[0] + dr, m[1] + dc) in candidate["cells"] for dr, dc in DIRS):
                return False
    for pos, value in candidate["cells"].items():
        for dr, dc in DIRS:
            other = pos[0] + dr, pos[1] + dc
            if any(p["cells"].get(other) == value for p in pieces):
                return False
    pieces.append(candidate)
    return True


def _scene(rng, palette, extra_cargo=0, extra_rods=0, large=False):
    for _ in range(100):
        core = _make_core(rng, palette, large)
        if core is None:
            continue
        shape, bg, guides, pieces = core
        if extra_cargo and rng.random() < .6:
            choices = [r for r in range(5, shape[0] - 1)
                       if all(abs(r - q) >= 2 for q in guides.values())]
            if choices:
                guides[palette[3]] = int(rng.choice(choices))
        success = True
        for cargo, count in ((False, extra_rods), (True, extra_cargo)):
            for _ in range(count):
                if not any(_insert(rng, shape, guides, pieces, palette, cargo) for attempt in range(100)):
                    success = False
                    break
            if not success:
                break
        if not success:
            continue
        # Only draw guide colors actually carried in the scene.
        used = {p["cells"][p["marker"]] for p in pieces if p["marker"] is not None}
        guides = {c: row for c, row in guides.items() if c in used}
        anchors = _anchors(rng, guides, shape[1])
        if anchors is None:
            continue
        band = _band(rng, shape[1], bg, [palette[7]])
        inp = _render(shape, bg, band, anchors, pieces)
        try:
            decoded = _decode(inp)
            actual = sorted((tuple(sorted(p["cells"].items())), p["target"]) for p in decoded[-1])
            intended = sorted((tuple(sorted(p["cells"].items())), p["target"]) for p in pieces)
            _require(actual == intended, "construction does not decode uniquely")
        except AssertionError:
            continue
        return shape, bg, guides, anchors, pieces
    raise AssertionError("bounded scene construction exhausted")


def _pair(rng, scene, commands):
    shape, bg, guides, anchors, pieces = scene
    band = _band(rng, shape[1], bg, commands)
    survivors = [p for p in pieces if p["rod_color"] not in commands]
    return {"input": _render(shape, bg, band, anchors, pieces),
            "output": _render(shape, bg, band, anchors, _settle(survivors, shape[0]))}


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a = _count(num_train, "num_train", 2, 10)
    b = _count(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    # Two counterfactual commands establish that deletion is by equality, and
    # teach a target-latched piece supporting a still-misaligned upper piece.
    for _ in range(100):
        scene = _scene(rng, palette)
        release = _pair(rng, scene, [palette[7]])
        f = pair_features(release)
        if f["reached"] and f["blocked"] and f["locked_supports"] and f["a_matters"]:
            break
    else:
        raise AssertionError("no diagnostic release scene")
    train = [release, _pair(rng, scene, [palette[8]])]
    seen = {_signature(p["input"]) for p in train}
    for _ in range(a - 2):
        for attempt in range(30):
            colors = list(map(int, rng.permutation(10)))
            extra = int(rng.integers(0, 3))
            scene = _scene(rng, colors, extra, int(rng.integers(0, 4)))
            commands = list(map(int, rng.choice(colors[7:10], size=int(rng.integers(1, 4)), replace=False)))
            pair = _pair(rng, scene, commands)
            if _signature(pair["input"]) not in seen:
                seen.add(_signature(pair["input"]))
                train.append(pair)
                break
        else:
            raise AssertionError("duplicate training inputs")
    max_cargo = max(pair_features(p)["cargo"] for p in train)
    tests = []
    for _ in range(b):
        for attempt in range(120):
            colors = list(map(int, rng.permutation(10)))
            count = int(rng.integers(max_cargo + 1, min(6, max_cargo + 2) + 1))
            scene = _scene(rng, colors, count - 2, int(rng.integers(1, 5)), large=True)
            options = [[colors[7]], [colors[7], colors[8]], [colors[7], colors[9]], colors[7:10]]
            rng.shuffle(options)
            pair = None
            for commands in options:
                candidate = _pair(rng, scene, commands)
                f = pair_features(candidate)
                if f["reached"] and f["blocked"] and f["locked_supports"] and f["moved_cargo"] >= 3:
                    pair = candidate
                    break
            if pair is not None and _signature(pair["input"]) not in seen:
                seen.add(_signature(pair["input"]))
                tests.append(pair)
                break
        else:
            raise AssertionError("duplicate test inputs")
    rng.shuffle(train)
    rng.shuffle(tests)
    puzzle = {"train": train, "test": tests}
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle):
    return tuple((_signature(p["input"]), _signature(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def _mechanics_test():
    # Coordinate-explicit diagnostic: the lower cargo falls two rows and pins;
    # the upper cargo falls two rows but stops seven rows above its own guide.
    inp = np.zeros((16, 12), dtype=np.int8)
    inp[0, 3] = 1
    inp[10, 0], inp[14, 11] = 2, 3
    inp[7:10, 3:8] = 5
    inp[8, 5] = 2
    inp[5:7, 4:7] = 6
    inp[5, 5] = 3
    inp[10:16, 3] = 1
    inp[14:16, 7] = 7
    expected = np.zeros_like(inp)
    expected[0, 3], expected[10, 0], expected[14, 11] = 1, 2, 3
    expected[9:12, 3:8] = 5
    expected[10, 5] = 2
    expected[7:9, 4:7] = 6
    expected[7, 5] = 3
    expected[14:16, 7] = 7
    _require(np.array_equal(solve(inp), expected), "coordinate diagnostic")
    # Selecting the shorter non-load-bearing post leaves both cargo stationary.
    harmless = inp.copy()
    harmless[0, 3] = 7
    expected_harmless = harmless.copy()
    expected_harmless[14:16, 7] = 0
    _require(np.array_equal(solve(harmless), expected_harmless), "command diagnostic")
    # Moving only the boundary guide changes the same scaffold's landing.
    longer = inp.copy()
    longer[10, 0], longer[13, 0] = 0, 2
    predicted = solve(longer)
    _require(predicted[12, 5] == 2 and predicted[9, 5] == 3,
             "physical post must arrest cargo before its later guide")
    # A mutually supported stack moves in one tick, without row-order priority.
    beam = _piece({(5, c): 4 for c in range(3, 6)}, rod_color=4)
    post = _piece({(r, 4): 7 for r in range(6, 9)}, rod_color=7)
    _require(_same_pieces(_settle([beam, post], 12), [_shift(beam, 3), _shift(post, 3)]),
             "co-moving support stack")


def self_test():
    """Bounded 42-seed semantic, diversity, metamorphic and rejection checks."""
    _mechanics_test()
    fingerprints, shapes, geometry, records = set(), set(), set(), []
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = pair_features(pair)
                records.append(f)
                shapes.add(f["shape"])
                decoded = _decode(pair["input"])
                points = tuple(sorted(tuple(sorted(p["cells"])) for p in decoded[-1]))
                geometry.add((f["shape"], points, tuple(sorted(decoded[3].values()))))
                _require(np.array_equal(solve(pair["input"].copy()), pair["output"]), "solve mismatch")
    _require(len(fingerprints) == 42, "seed diversity")
    _require(len(shapes) >= 45 and len(geometry) >= 150, "structural diversity")
    _require({f["background"] for f in records} == set(range(10)), "background diversity")
    _require({2, 3, 4, 5} <= {f["cargo"] for f in records}, "cargo-count diversity")
    _require({1, 2, 3} <= {f["commands"] for f in records}, "command diversity")
    _require({2, 3} <= {f["guides"] for f in records}, "guide diversity")
    _require(any(f["horizontal_rods"] for f in records), "horizontal rods missing")
    _require(any(len(f["falls"]) > 1 for f in records), "independent fall distances missing")
    _require(any(f["bilateral"] for f in records), "bilateral guides missing")
    _require(len({f["sizes"] for f in records}) >= 70, "body geometry diversity")
    _require(max(n for f in records for n in f["rod_lengths"]) >= 15, "long supports missing")
    _require(_fingerprint(generate_puzzle(7123, 3, 1)) ==
             _fingerprint(generate_puzzle(7123, 3, 1)), "determinism")
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 4)):
        validate_generated_puzzle(generate_puzzle(7124, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    puzzle = generate_puzzle(7125, 2, 1)
    recolor = np.random.default_rng(71).permutation(10)
    altered = {s: [{k: recolor[np.fliplr(v)].astype(np.int8) for k, v in p.items()}
                   for p in puzzle[s]] for s in puzzle}
    validate_generated_puzzle(altered)
    # Deletion, guide and gravity ablations above are geometric counterexamples.
    # Also reject a well-shaped but semantically corrupted answer.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted answer accepted")
