"""Restore optimized checkpoint tracks; use their edge signatures to assemble tiles.

The solver reads only its input.  Construction uses small corridor cycles, but
the solver enumerates simple induced paths, never construction parameters.
No parent module, file, seed, cache, or stored example is used by solve().
"""
from __future__ import annotations

from collections import Counter
from itertools import permutations
import hashlib
import json

import numpy as np


HOST_SEEDS = (0, 433158549, 639305145, 333866160, 989625975, 729912,
              284225757, 786714246, 90907270, 217654691, 330625987,
              420808435, 927767222, 270909140, 955699506, 633285090,
              227851238, 257265480, 852896253, 894705262, 962730105,
              702906539, 191667817, 475307589, 848942433, 808270488,
              23817408, 772290943, 106967254, 445445382, 548408750,
              54056902, 919483310, 278737162, 881610465, 270255399,
              519912845, 65968872, 53182207, 38744699, 94367573, 775313880)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid size")
    _require(np.issubdtype(a.dtype, np.integer) and
             bool(np.all((a >= 0) & (a <= 9))), "integer colors 0..9 required")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _neighbors(p):
    r, c = p
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        part, stack = {p}, [p]
        while stack:
            for q in _neighbors(stack.pop()):
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _cells(mask):
    return {tuple(map(int, p)) for p in np.argwhere(mask)}


def _bends(path):
    directions = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(directions, directions[1:]))


def _simple(path):
    cells = set(path)
    return (len(cells) == len(path) and
            all(b in _neighbors(a) for a, b in zip(path, path[1:])) and
            all(sum(q in cells for q in _neighbors(p)) ==
                (1 if i in (0, len(path) - 1) else 2)
                for i, p in enumerate(path)))


def _route_options(panel, color, background):
    """All simple, non-self-touching completions with checkpoint endpoints.

    Endpoints of an optimum must be supplied cells: any added terminal tail can
    be trimmed.  Enumerating these endpoints therefore loses no optimum.  The
    family has at most two separated corridor cycles per tile; this exact DFS
    is small without a search cap or a generator-provided bound.
    """
    required = _cells(panel == color)
    allowed = _cells((panel == color) | (panel == background))
    adjacent = {p: tuple(q for q in _neighbors(p) if q in allowed) for p in allowed}
    starts = sorted(p for p in required
                    if sum(q in required for q in _neighbors(p)) <= 1)
    _require(bool(starts), "checkpoint fragments cannot branch or form a cycle")
    candidates = {}

    def walk(path, used, left):
        if not left:
            canonical = min(tuple(path), tuple(reversed(path)))
            candidates[canonical] = (len(path) - len(required), _bends(path))
            return
        for q in adjacent[path[-1]]:
            if q in used or sum(p in used for p in adjacent[q]) != 1:
                continue
            path.append(q)
            used.add(q)
            walk(path, used, left - {q})
            used.remove(q)
            path.pop()

    for start in starts:
        walk([start], {start}, required - {start})
    return sorted((cost, path) for path, cost in candidates.items())


def _has_block(mask):
    return bool(np.any(mask[:-1, :-1] & mask[1:, :-1] &
                       mask[:-1, 1:] & mask[1:, 1:]))


def _partition(grid):
    candidates = []
    h, w = grid.shape
    for divider in map(int, np.unique(grid)):
        rows = list(map(int, np.flatnonzero(np.all(grid == divider, axis=1))))
        cols = list(map(int, np.flatnonzero(np.all(grid == divider, axis=0))))
        if not rows and not cols:
            continue
        rr, cc = [-1] + rows + [h], [-1] + cols + [w]
        hh = [b - a - 1 for a, b in zip(rr, rr[1:])]
        ww = [b - a - 1 for a, b in zip(cc, cc[1:])]
        if min(hh + ww) < 5 or len(set(hh)) != 1 or len(set(ww)) != 1:
            continue
        panels = [grid[r + 1:rr[i + 1], c + 1:cc[j + 1]].copy()
                  for i, r in enumerate(rr[:-1]) for j, c in enumerate(cc[:-1])]
        if not 2 <= len(panels) <= 5 or any(np.any(p == divider) for p in panels):
            continue
        corners = {int(p[r, c]) for p in panels
                   for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))}
        if len(corners) != 1:
            continue
        bg = next(iter(corners))
        walls = []
        for panel in panels:
            choices = [int(c) for c in np.unique(panel) if c != bg and
                       _has_block(panel == c)]
            if len(choices) != 1:
                break
            walls.append(choices[0])
        if len(walls) == len(panels) and len(set(walls)) == 1:
            candidates.append(dict(panels=panels, background=bg, wall=walls[0],
                                   divider=divider, layout=(len(hh), len(ww))))
    _require(len(candidates) == 1, "one equal-panel partition with visible roles required")
    return candidates[0]


def _complete_panel(panel, bg, wall):
    ordinary = sorted(int(c) for c in np.unique(panel) if c not in (bg, wall))
    active = [c for c in ordinary if len(_components(_cells(panel == c))) >= 2]
    _require(len(active) == 1, "one active checkpoint color per panel required")
    color = active[0]
    for c in ordinary:
        for part in _components(_cells(panel == c)):
            _require(all(sum(q in part for q in _neighbors(p)) <= 2 for p in part),
                     "supplied fragments cannot branch")
    options = _route_options(panel, color, bg)
    _require(bool(options), "no checkpoint completion")
    _require(len(options) == 1 or options[0][0] != options[1][0],
             "tied shortest/fewest-bend paths")
    out = panel.copy()
    path = options[0][1]
    _require(_simple(path), "completion must be a simple induced path")
    for p in path:
        _require(int(panel[p]) in (bg, color), "foreign occupied cell")
        out[p] = color
    return dict(completed=out, color=color, path=path, options=options,
                raw_mask=panel == color, mask=out == color)


def _orders(signatures):
    starts = [i for i, (a, b) in enumerate(signatures) if not a and b]
    ends = [i for i, (a, b) in enumerate(signatures) if a and not b]
    if len(starts) != 1 or len(ends) != 1:
        return []
    result = []
    for rest in permutations(i for i in range(len(signatures)) if i != starts[0]):
        order = (starts[0],) + rest
        if order[-1] != ends[0]:
            continue
        if all(signatures[a][1] and signatures[a][1] == signatures[b][0]
               for a, b in zip(order, order[1:])):
            result.append(order)
            if len(result) == 2:
                break
    return result


def _signatures(mask, axis):
    p = mask if axis == "H" else mask.T
    return tuple(map(int, np.flatnonzero(p[:, 0]))), tuple(map(int, np.flatnonzero(p[:, -1])))


def _chain(masks):
    candidates = []
    for axis in ("H", "V"):
        oriented = [p if axis == "H" else p.T for p in masks]
        if any(np.any(p[0]) or np.any(p[-1]) for p in oriented):
            continue
        signatures = [_signatures(p, axis) for p in masks]
        orders = _orders(signatures)
        if orders:
            _require(len(orders) == 1, "ambiguous complete panel order")
            candidates.append((axis, orders[0], signatures))
    _require(len(candidates) == 1, "ports must specify one complete directed chain")
    return candidates[0]


def _assemble(panels, order, layout, divider):
    nr, nc = layout
    h, w = panels[0].shape
    out = np.full((nr * (h + 1) - 1, nc * (w + 1) - 1), divider, dtype=np.int8)
    for slot, i in enumerate(order):
        r, c = divmod(slot, nc)
        out[r * (h + 1):r * (h + 1) + h, c * (w + 1):c * (w + 1) + w] = panels[i]
    return out


def _decode(input_grid):
    info = _partition(_grid(input_grid))
    details = [_complete_panel(p, info["background"], info["wall"]) for p in info["panels"]]
    axis, order, signatures = _chain([d["mask"] for d in details])
    info.update(details=details, axis=axis, order=order, signatures=signatures)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(input_grid)
    n = len(info["panels"])
    layout = (1, n) if info["axis"] == "H" else (n, 1)
    out = _assemble([d["completed"] for d in info["details"]], info["order"],
                    layout, info["divider"])
    _require(all(x <= 30 for x in out.shape), "assembled strip exceeds 30")
    return out


def _polyline(keys):
    result = [keys[0]]
    for a, b in zip(keys, keys[1:]):
        _require(a[0] == b[0] or a[1] == b[1], "orthogonal segment required")
        dr = (b[0] > a[0]) - (b[0] < a[0])
        dc = (b[1] > a[1]) - (b[1] < a[1])
        p = a
        while p != b:
            p = p[0] + dr, p[1] + dc
            result.append(p)
    return tuple(result)


def _gadget(w, link, side):
    s, e, kind, k = link
    if kind == "priority" and s > 1:
        a, b = (s - 1, 1), (e, 1)
        preferred = _polyline([a, (s, 1), (s, 0), (e, 0), b])
        alternate = _polyline([a, (s - 1, w - 1), (e, w - 1), b])
    elif kind == "bend":
        d = (w - 1) // 2
        a, b = (s, d), (e, d)
        preferred = _polyline([a, (s, 0), (e, 0), b])
        alternate = _polyline([a, (s, d + 1), (s + k, d + 1),
                               (s + k, w - 1), (e, w - 1), b])
    else:
        a, b = (s, 1), (e, 1)
        preferred = _polyline([a, (s, 0), (e, 0), b])
        alternate = _polyline([a, (s, w - 1), (e, w - 1), b])
    if side == "R":
        flip = lambda p: (p[0], w - 1 - p[1])
        a, b = flip(a), flip(b)
        preferred, alternate = tuple(map(flip, preferred)), tuple(map(flip, alternate))
    _require(_simple(preferred) and _simple(alternate), "invalid corridor arc")
    return a, b, preferred, alternate


def _links(rng, n, h, mode):
    for _ in range(80):
        cut = int(rng.integers(4, h - 9))
        upper = [(s, e) for s in range(1, cut) for e in range(s + 3, min(s + 5, cut) + 1)]
        if mode == "priority":
            upper = [p for p in upper if p[0] >= 2]
        lower = [(s, e) for s in range(cut + 5, h - 4)
                 for e in range(s + 3, min(s + 5, h - 2) + 1)]
        if not upper or not lower:
            continue
        chosen = []
        repeat = n == 4 and bool(rng.integers(2))
        for i in range(n - 1):
            pool = upper if i % 2 == 0 else lower
            if repeat and i == 2:
                chosen.append(chosen[0])
                continue
            available = [p for p in pool if p not in chosen]
            if not available:
                break
            chosen.append(available[int(rng.integers(len(available)))])
        if len(chosen) != n - 1:
            continue
        links = []
        per_signature = {}
        for s, e in chosen:
            if (s, e) not in per_signature:
                kind = mode if mode != "mixed" else ("priority", "bend", "length")[int(rng.integers(3))]
                per_signature[s, e] = (s, e, kind, int(rng.integers(1, e - s - 1)))
            links.append(per_signature[s, e])
        return links
    raise ValueError("unable to sample compatible port intervals")


def _make_panel(rng, h, w, left, right, bg, wall, color, extras, foreign):
    gadgets = [_gadget(w, link, side) for link, side in ((left, "L"), (right, "R"))
               if link is not None]
    gadgets.sort(key=lambda g: g[0])
    free, required, preferred, alternate = set(), set(), [], []
    for i, (a, b, first, second) in enumerate(gadgets):
        if i:
            old = gadgets[i - 1][1]
            middle = (old[0] + a[0]) // 2
            bridge = _polyline([old, (middle, old[1]), (middle, a[1]), a])
            free.update(bridge)
            preferred.extend(bridge[1:-1])
            alternate.extend(bridge[1:-1])
        free.update(first)
        free.update(second)
        required.update((a, b))
        preferred.extend(first)
        alternate.extend(second)
    _require(_simple(preferred) and _simple(alternate), "corridors self-touch")
    inp = np.full((h, w), wall, dtype=np.int8)
    for p in free | {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}:
        inp[p] = bg
    for p in required:
        inp[p] = color
    if extras:
        for p in preferred:
            if p not in required and rng.random() < .24:
                inp[p] = color
        if foreign is not None:
            possible = sorted(p for p in set(alternate) - set(preferred)
                              if 0 < p[0] < h - 1 and 0 < p[1] < w - 1)
            if possible and rng.random() < .35:
                inp[possible[int(rng.integers(len(possible)))]] = foreign
            else:
                # One connected inert run inside a wall; never a second task.
                r = int(rng.integers(1, h - 1))
                c = int(rng.integers(1, w - 1))
                run = [(r, x) for x in range(c, min(w - 1, c + int(rng.integers(1, 4))))]
                if all(inp[p] == wall for p in run):
                    for p in run:
                        inp[p] = foreign
    out = inp.copy()
    for p in preferred:
        out[p] = color
    _require(np.all(out[inp != bg] == inp[inp != bg]), "never overwrite supplied cells")
    decoded = _complete_panel(inp, bg, wall)
    _require(np.array_equal(decoded["completed"], out), "constructed path is not the unique optimum")
    return inp, out


def _make_pair(rng, n, axis, mode, witness=False, test=False):
    for _ in range(100):
        h = int(rng.integers(16 if n == 5 or mode == "priority" else 14, 25))
        if n == 4 and mode == "bend" and rng.random() < .45:
            h = 14
        widths = [w for w in (5, 7, 9, 11, 13) if n * (w + 1) - 1 <= 30]
        w = widths[int(rng.integers(len(widths)))]
        palette = list(map(int, rng.permutation(10)))
        bg, wall, divider = palette[:3]
        route_palette = palette[3:]
        shared = mode == "bend" or bool(rng.integers(2))
        colors = ([route_palette[0]] * n if shared else
                  [route_palette[int(rng.integers(min(n, 5)))] for _ in range(n)])
        try:
            links = _links(rng, n, h, mode)
            inputs, outputs = [], []
            for i in range(n):
                foreign = next(c for c in route_palette if c != colors[i])
                a, b = _make_panel(rng, h, w, None if i == 0 else links[i - 1],
                                   None if i == n - 1 else links[i], bg, wall,
                                   colors[i], not witness and i != 0, foreign)
                if axis == "V":
                    a, b = a.T.copy(), b.T.copy()
                inputs.append(a)
                outputs.append(b)
            ph, pw = inputs[0].shape
            layouts = [(nr, n // nr) for nr in range(1, n + 1) if n % nr == 0 and
                       nr * (ph + 1) - 1 <= 30 and (n // nr) * (pw + 1) - 1 <= 30]
            layout = layouts[int(rng.integers(len(layouts)))]
            order = list(map(int, rng.permutation(n)))
            if order == list(range(n)):
                order[0], order[-1] = order[-1], order[0]
            inp = _assemble(inputs, order, layout, divider)
            out = _assemble(outputs, list(range(n)), (1, n) if axis == "H" else (n, 1), divider)
            _require(np.array_equal(solve(inp), out), "assembly differs from construction")
            if witness:
                info = _decode(inp)
                _require(_reverse_counterfactual(info), "alternative arcs must reverse the chain")
                flags = _contrasts(info)
                _require(flags[0 if mode == "priority" else 1], "missing objective witness")
            return {"input": inp, "output": out}
        except ValueError:
            continue
    raise ValueError("bounded construction attempts exhausted")


def _count(value, low, high):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
             and low <= value <= high, "illegal example count")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _count(num_train, 2, 10), _count(num_test, 1, 4)
    _require(a > b, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, "nonnegative integer seed required")
    rng = np.random.default_rng(seed)
    first_axis = "H" if rng.integers(2) else "V"
    other_axis = "V" if first_axis == "H" else "H"
    result, seen = {"train": [], "test": []}, set()
    for split, count in (("train", a), ("test", b)):
        for i in range(count):
            witness = split == "train" and i < 2
            mode = ("priority", "bend")[i] if witness else "mixed"
            n = (3, 4)[i] if witness else (5 if split == "test" and i == 0 else int(rng.integers(2, 5)))
            axis = (first_axis, other_axis)[i] if witness else ("H" if rng.integers(2) else "V")
            for _ in range(20):
                pair = _make_pair(rng, n, axis, mode, witness, split == "test")
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise ValueError("bounded duplicate retries exhausted")
            seen.add(_key(pair["input"]))
            result[split].append(pair)
    return result


def _path_mask(shape, path):
    mask = np.zeros(shape, dtype=bool)
    for p in path:
        mask[p] = True
    return mask


def _reverse_counterfactual(info):
    masks = [_path_mask(d["mask"].shape, d["options"][-1][1]) for d in info["details"]]
    try:
        axis, order, _ = _chain(masks)
        return axis == info["axis"] and order == info["order"][::-1]
    except ValueError:
        return False


def _contrasts(info):
    primary, secondary = False, False
    for d in info["details"]:
        best = d["options"][0][0]
        signatures = _signatures(d["mask"], info["axis"])
        for cost, path in d["options"][1:]:
            other = _signatures(_path_mask(d["mask"].shape, path), info["axis"])
            if other == signatures:
                continue
            primary |= cost[0] > best[0] and cost[1] < best[1]
            secondary |= cost[0] == best[0] and cost[1] > best[1]
    return bool(primary), bool(secondary)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists required")
    a, b = _count(len(puzzle["train"]), 2, 10), _count(len(puzzle["test"]), 1, 4)
    _require(a > b, "example counts")
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _decode(inp)
            _require(np.array_equal(solve(inp), out), "incorrect solved output")
            parsed = _partition(out)
            completed = [d["completed"] for d in info["details"]]
            _require(Counter(map(_key, completed)) == Counter(map(_key, parsed["panels"])),
                     "completed panel multiset must be conserved")
            for p, d in zip(info["panels"], info["details"]):
                q = d["completed"]
                _require(np.array_equal(q[p != info["background"]], p[p != info["background"]]),
                         "supplied cell changed")
                _require(np.array_equal(q == info["wall"], p == info["wall"]), "wall changed")
                _require(_simple(d["path"]), "branch or self-touch")
            _require(any(not np.array_equal(d["raw_mask"], d["mask"]) for d in info["details"]),
                     "must require completion")
            _require(info["order"] != tuple(range(len(completed))), "must require reordering")
            if split == "train":
                train.append(info)
    _require(any(_contrasts(i)[0] for i in train), "demonstrate length before bends")
    _require(any(_contrasts(i)[1] for i in train), "demonstrate bend tie break")
    _require(sum(_reverse_counterfactual(i) for i in train) >= 2,
             "demonstrate that alternative paths change the chain order")
    _require({i["axis"] for i in train} == {"H", "V"}, "demonstrate both axes")
    _require(any(len({d["color"] for d in i["details"]}) < len(i["details"]) for i in train),
             "demonstrate color-independent port matching")
    return True


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """42 fixed seeds, bounded in-memory diagnostics; writes no files or traces."""
    sets = {k: set() for k in ("shapes", "layouts", "counts", "axes", "backgrounds",
                               "walls", "dividers", "port_lengths", "bends", "structures")}
    records, pair_count, hidden_turns, inert = [], 0, 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), "nondeterministic")
        sizes = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                sizes.append(len(info["panels"]))
                pair_count += 1
                for field, value in (("shapes", info["panels"][0].shape), ("layouts", info["layout"]),
                                     ("counts", len(info["panels"])), ("axes", info["axis"]),
                                     ("backgrounds", info["background"]), ("walls", info["wall"]),
                                     ("dividers", info["divider"])):
                    sets[field].add(value)
                remap = {}
                norm = [[remap.setdefault(int(c), len(remap)) for c in row] for row in pair["input"]]
                sets["structures"].add(hashlib.sha256(json.dumps(norm, separators=(",", ":")).encode()).hexdigest())
                for a, b in info["signatures"]:
                    sets["port_lengths"].update(len(s) for s in (a, b) if s)
                for p, d in zip(info["panels"], info["details"]):
                    sets["bends"].add(_bends(d["path"]))
                    path = d["path"]
                    for i in range(1, len(path) - 1):
                        u, v, w = path[i - 1:i + 2]
                        if (v[0] - u[0], v[1] - u[1]) != (w[0] - v[0], w[1] - v[1]):
                            hidden_turns += int(p[v] == info["background"])
                    inert += len(set(map(int, np.unique(p))) -
                                 {info["background"], info["wall"], d["color"]})
        records.append({"seed": seed, "panel_counts": sizes})
    _require(sets["counts"] == {2, 3, 4, 5} and sets["axes"] == {"H", "V"}, "structural coverage")
    _require(all(len(sets[k]) == 10 for k in ("backgrounds", "walls", "dividers")), "role palettes")
    _require(len(sets["shapes"]) >= 20 and len(sets["structures"]) >= 200, "geometric diversity")
    _require(any(r > 1 and c > 1 for r, c in sets["layouts"]), "array layouts")
    _require(hidden_turns > 100 and inert > 0, "hidden corners and inert evidence")
    for a, b in ((2, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(1234, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A symmetric ring has two equally short, equally bent completions.
    tie = np.full((9, 5), 8, dtype=np.int8)
    for p in _polyline([(2, 2), (2, 0), (6, 0), (6, 2)]) + _polyline([(2, 2), (2, 4), (6, 4), (6, 2)]):
        tie[p] = 0
    tie[2, 2] = tie[6, 2] = 2
    try:
        _complete_panel(tie, 0, 8)
    except ValueError:
        pass
    else:
        raise AssertionError("tied optimum accepted")
    _require(len(_orders([((), (1,)), ((1,), (1,)), ((1,), (1,)), ((1,), ())])) == 2,
             "interchangeable-panel negative")
    # A foreign singleton on the preferred arc forces the other side of a ring.
    a, b, route, alt = _gadget(5, (2, 6, "length", 1), "L")
    panel = np.full((10, 5), 8, dtype=np.int8)
    for p in set(route) | set(alt):
        panel[p] = 0
    panel[a] = panel[b] = 2
    panel[4, 0] = 3
    answer = _complete_panel(panel, 0, 8)
    _require(answer["completed"][4, 0] == 3 and set(answer["path"]) == set(alt), "foreign-cell avoidance")
    bad = generate_puzzle(0, 2, 1)
    bad["test"][0]["output"] = bad["test"][0]["output"].copy()
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("damaged output accepted")
    return {"seeds": len(HOST_SEEDS), "pairs": pair_count, "hidden_turns": hidden_turns,
            "inert_colors": inert, "coverage": {k: len(v) for k, v in sets.items()},
            "records": records, "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
