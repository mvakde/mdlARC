"""Stamp-constrained jigsaws. Only Python's standard library and NumPy.

No parent imports, stored puzzles, solver caches, file writes, or latent codes.
Generation constructs labels from an owner grid; validation re-parses pixels.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions must be 1..30")
    _need(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _need(bool(np.all((0 <= a) & (a <= 9))), "grid colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _components(mask):
    cells = set(zip(*np.nonzero(mask)))
    result = []
    while cells:
        start = min(cells)
        cells.remove(start)
        stack, found = [start], []
        while stack:
            r, c = stack.pop()
            found.append((int(r), int(c)))
            for dr, dc in _D4:
                q = (r + dr, c + dc)
                if q in cells:
                    cells.remove(q)
                    stack.append(q)
        result.append(tuple(sorted(found)))
    return result


def _normalize(cells, dots=(), legend=()):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    shift = lambda points: tuple(sorted((int(r-r0), int(c-c0)) for r, c in points))
    return {"cells": shift(cells), "dots": shift(dots), "legend": shift(legend)}


def _turn(piece, k):
    result = piece
    for _ in range(k % 4):
        h = max(r for r, c in result["cells"]) + 1
        result = {name: tuple(sorted((c, h-1-r) for r, c in points))
                  for name, points in result.items()}
    return result


def _dimensions(piece):
    return (max(r for r, c in piece["cells"])+1, max(c for r, c in piece["cells"])+1)


def _parse(value):
    grid = _grid(value)
    bg = int(grid[0, 0])
    _need(all(np.all(edge == bg) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])),
          "a complete background border is required")
    components = _components(grid != bg)
    rings = []
    for cells in components:
        r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
        r1, c1 = max(r for r, c in cells), max(c for r, c in cells)
        if r1-r0 < 4 or c1-c0 < 4:
            continue
        border = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)
                  if r in (r0, r1) or c in (c0, c1)}
        if set(cells) == border and len({int(grid[r, c]) for r, c in cells}) == 1:
            rings.append((r0, c0, r1, c1, int(grid[r0, c0])))
    _need(len(rings) == 1, "one isolated catalogue frame is required")
    r0, c0, r1, c1, carrier = rings[0]
    inner = grid[r0+1:r1, c0+1:c1]
    _need(all(np.all(edge == bg) for edge in (inner[0], inner[-1], inner[:, 0], inner[:, -1])),
          "catalogue needs an empty inner rim")
    outside = grid != bg
    outside[r0:r1+1, c0:c1+1] = False
    loose = _components(outside)
    _need(len(loose) == 4, "four loose connected pieces are required")
    blocks = []
    for r in range(grid.shape[0]-1):
        for c in range(grid.shape[1]-1):
            if outside[r:r+2, c:c+2].all() and len(set(map(int, grid[r:r+2, c:c+2].flat))) == 4:
                blocks.append((r, c))
    _need(len(blocks) == 1, "one unambiguous four-color 2-by-2 key is required")
    kr, kc = blocks[0]
    key = tuple(map(int, grid[kr:kr+2, kc:kc+2].flat))
    _need(bg not in key and carrier not in key, "key colors must differ from structural colors")
    legend = {(kr+i, kc+j) for i in range(2) for j in range(2)}
    anchor = next(i for i, cells in enumerate(loose) if (kr, kc) in cells)
    loose = [loose[anchor]] + [cells for i, cells in enumerate(loose) if i != anchor]
    remaining = {int(grid[r, c]) for cells in loose for r, c in cells} - {carrier, *key}
    _need(len(remaining) == 1, "one shared neutral dot color is required")
    dot = remaining.pop()
    pieces = []
    for i, cells in enumerate(loose):
        allowed = {carrier, dot, *key} if i == 0 else {carrier, dot}
        _need(all(int(grid[r, c]) in allowed for r, c in cells), "unexpected piece color")
        dots = [(r, c) for r, c in cells if grid[r, c] == dot]
        _need(dots and any(grid[r, c] == carrier for r, c in cells), "each piece needs material and dots")
        if i == 0:
            _need(all(sum(grid[r, c] == color for r, c in cells) == 1 for color in key),
                  "each legend color occurs once on anchor")
            _need(all(max(abs(r-u), abs(c-v)) >= 2 for r, c in dots for u, v in legend),
                  "dots must be separated from the printed key")
        pieces.append(_normalize(cells, dots, legend if i == 0 else ()))
    templates, repetitions = {}, {}
    for cells in _components(inner != bg):
        anchors = [(r, c) for r, c in cells if int(inner[r, c]) in key]
        _need(len(anchors) == 1 and len(cells) >= 3, "every catalogue motif needs exactly one key anchor")
        ar, ac = anchors[0]
        color = int(inner[ar, ac])
        signature = tuple(sorted((r-ar, c-ac, int(inner[r, c])) for r, c in cells))
        _need(all(v not in {bg, carrier, dot, *key} for r, c, v in signature if (r, c) != (0, 0)),
              "decoration roles must be distinct from keys and structural colors")
        _need(color not in templates or templates[color] == signature, "conflicting catalogue entries")
        templates[color] = signature
        repetitions[color] = repetitions.get(color, 0) + 1
    _need(set(templates) == set(key), "catalogue must define all four key colors")
    total = sum(len(p["cells"]) for p in pieces)
    side = math.isqrt(total)
    _need(side*side == total and 1 <= side <= 30, "loose material area must be square")
    return {"grid": grid, "bg": bg, "carrier": carrier, "dot": dot, "key": key,
            "templates": templates, "repetitions": repetitions, "pieces": pieces,
            "side": side, "catalogue_shape": (r1-r0+1, c1-c0+1), "key_box": (kr, kc)}


def _placed(piece, turns, side, corner, template, stamp_mode):
    p = _turn(piece, turns)
    h, w = _dimensions(p)
    if h > side or w > side:
        return None
    dy = 0 if corner < 2 else side-h
    dx = 0 if corner % 2 == 0 else side-w
    cells = {(r+dy, c+dx) for r, c in p["cells"]}
    corners = ((0, 0), (0, side-1), (side-1, 0), (side-1, side-1))
    if [q in cells for q in corners] != [i == corner for i in range(4)]:
        return None
    dots = tuple((r+dy, c+dx) for r, c in p["dots"])
    stamps, footprint = {}, set()
    if stamp_mode != "none":
        if stamp_mode == "piece":  # Explicit counterfactual used only in validation.
            transformed = []
            for r, c, color in template:
                for _ in range(turns):
                    r, c = c, -r
                transformed.append((r, c, color))
            template = transformed
        for ar, ac in dots:
            for dr, dc, color in template:
                q = (ar+dr, ac+dc)
                if q not in cells or q in footprint:
                    return None
                footprint.add(q)
                stamps[q] = color
    mask = sum(1 << (r*side+c) for r, c in cells)
    return {"mask": mask, "stamps": stamps, "dots": dots, "turns": turns, "cells": cells}


def _assemble(info, stamp_mode="page", limit=2):
    """Enumerate ALL allowed corner placements; deduplicate final answers.

    Corner occupancy fixes translation after a turn. There are at most
    6 * 4**3 complete placement combinations with the anchor fixed.
    """
    side, key, pieces = info["side"], info["key"], info["pieces"]
    table = {}
    for corner in range(4):
        for i in ([0] if corner == 0 else range(1, 4)):
            options, seen = [], set()
            for turns in ([0] if corner == 0 else range(4)):
                option = _placed(pieces[i], turns, side, corner, info["templates"][key[corner]], stamp_mode)
                if option is None:
                    continue
                signature = (option["mask"], tuple(sorted(option["stamps"].items())))
                if signature not in seen:
                    seen.add(signature)
                    option["piece"] = i
                    options.append(option)
            table[corner, i] = options
    answers, seen_outputs = [], set()
    full = (1 << (side*side))-1

    def visit(corner, remaining, occupied, placements):
        if len(answers) >= limit:
            return
        if corner == 4:
            if occupied != full:
                return
            out = np.empty((side, side), dtype=np.int8)
            for role, place in enumerate(placements):
                for r, c in place["cells"]:
                    out[r, c] = key[role]
                for (r, c), color in place["stamps"].items():
                    out[r, c] = color
            signature = out.tobytes()
            if signature not in seen_outputs:
                seen_outputs.add(signature)
                answers.append({"output": out, "placements": tuple(placements)})
            return
        for i in ([0] if corner == 0 else remaining):
            for p in table[corner, i]:
                if p["mask"] & occupied:
                    continue
                visit(corner+1, tuple(v for v in remaining if v != i), occupied | p["mask"], placements+[p])
                if len(answers) >= limit:
                    return

    visit(0, (1, 2, 3), 0, [])
    return answers


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    answers = _assemble(info)
    _need(len(answers) == 1, "motif fit and square packing must give one output")
    return answers[0]["output"].copy()


def _partition(rng, side, style):
    """Connected cuts are constructed afresh, never selected from a cut bank."""
    n = side
    y = int(rng.integers(2, n-1))
    x, z = (int(rng.integers(2, n-1)) for _ in range(2))
    owner = np.empty((n, n), dtype=np.int8)
    if style == 2:
        owner.fill(-1)
        frontier = []
        for i, (r, c) in enumerate(((0, 0), (0, n-1), (n-1, 0), (n-1, n-1))):
            owner[r, c] = i
            frontier.extend((r+dr, c+dc, i) for dr, dc in _D4 if 0 <= r+dr < n and 0 <= c+dc < n)
        while frontier:
            j = int(rng.integers(len(frontier)))
            r, c, label = frontier[j]
            frontier[j] = frontier[-1]
            frontier.pop()
            if owner[r, c] != -1:
                continue
            owner[r, c] = label
            frontier.extend((r+dr, c+dc, label) for dr, dc in _D4
                            if 0 <= r+dr < n and 0 <= c+dc < n and owner[r+dr, c+dc] == -1)
    else:
        cuts = [x, z]
        for r in range(n):
            band = int(r >= y)
            if style == 1 and r not in (0, y) and rng.random() < .85:
                cuts[band] = int(np.clip(cuts[band]+rng.choice([-2, -1, 0, 1, 2]), 1, n-1))
            cut = cuts[band]
            owner[r, :cut] = 2*band
            owner[r, cut:] = 2*band+1
    if rng.random() < .5:
        owner = owner.T.copy()
        owner[owner == 1] = 4
        owner[owner == 2] = 1
        owner[owner == 4] = 2
    areas = [int(np.count_nonzero(owner == i)) for i in range(4)]
    if min(areas) < 6 or areas[0] < 10:
        return None
    if any(len(_components(owner == i)) != 1 for i in range(4)):
        return None
    locations = [(r, c) for r in range(n-1) for c in range(n-1)
                 if np.all(owner[r:r+2, c:c+2] == 0)]
    rng.shuffle(locations)
    for kr, kc in locations:
        legend = {(kr+i, kc+j) for i in range(2) for j in range(2)}
        clear = [(r, c) for r, c in zip(*np.nonzero(owner == 0))
                 if all(max(abs(r-u), abs(c-v)) >= 2 for u, v in legend)]
        if clear:
            return owner, legend
    return None


def _sample_template(rng, cells, forbidden_dots, legend, key_color, decor, wanted, small):
    available_dots = sorted(cells - forbidden_dots)
    if not available_dots:
        return None
    for _ in range(45):
        p = available_dots[int(rng.integers(len(available_dots)))]
        height, width = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        # At least one extent is <=3, keeping a four-entry strip <=30 cells.
        if height == width == 4:
            height = 3
        r0, c0 = p[0]-int(rng.integers(height)), p[1]-int(rng.integers(width))
        allowed = {(r, c) for r, c in cells - legend if r0 <= r < r0+height and c0 <= c < c0+width}
        shape = {p}
        target = int(rng.integers(3, 5 if small else 8))
        while len(shape) < target:
            frontier = sorted({(r+dr, c+dc) for r, c in shape for dr, dc in _D4} & allowed - shape)
            if not frontier:
                break
            shape.add(frontier[int(rng.integers(len(frontier)))])
        if len(shape) < 3:
            continue
        offsets = tuple(sorted((r-p[0], c-p[1]) for r, c in shape))
        candidates = [q for q in available_dots if all((q[0]+dr, q[1]+dc) in cells-legend for dr, dc in offsets)]
        rng.shuffle(candidates)
        selected, occupied = [], set()
        for q in candidates:
            stamp = {(q[0]+dr, q[1]+dc) for dr, dc in offsets}
            if not (stamp & occupied):
                selected.append(q)
                occupied.update(stamp)
            if len(selected) == wanted:
                break
        if len(selected) != wanted:
            continue
        colors = [int(decor[0])]*(len(offsets)-1)
        if len(colors) >= 4 and len(decor) == 2:
            colors[:2] = [int(decor[1])]*2
            rng.shuffle(colors)
        signature, j = [], 0
        for dr, dc in offsets:
            if (dr, dc) == (0, 0):
                signature.append((dr, dc, key_color))
            else:
                signature.append((dr, dc, colors[j]))
                j += 1
        return tuple(signature), tuple(selected)
    return None


def _catalogue(rng, templates, key, bg, carrier, row_only=False):
    ids = list(key)
    if rng.random() < .35:
        ids.append(key[int(rng.integers(4))])
    rng.shuffle(ids)
    columns = len(ids) if row_only else int(rng.choice([len(ids), 2, 1]))
    boxes = []
    for color in ids:
        sig = templates[color]
        lr, lc = min(r for r, c, v in sig), min(c for r, c, v in sig)
        h = max(r for r, c, v in sig)-lr+1
        w = max(c for r, c, v in sig)-lc+1
        boxes.append((color, lr, lc, h, w))
    groups = [boxes[i:i+columns] for i in range(0, len(boxes), columns)]
    h = sum(max(b[3] for b in group) for group in groups)+len(groups)-1+4
    w = max(sum(b[4] for b in group)+len(group)-1 for group in groups)+4
    if h > 28 or w > 28:
        return _catalogue(rng, templates, key, bg, carrier, row_only=True)
    panel = np.full((h, w), bg, dtype=np.int8)
    panel[0], panel[-1], panel[:, 0], panel[:, -1] = carrier, carrier, carrier, carrier
    y = 2
    for group in groups:
        x = 2
        for color, lr, lc, bh, bw in group:
            for dr, dc, v in templates[color]:
                panel[y+dr-lr, x+dc-lc] = v
            x += bw+1
        y += max(b[3] for b in group)+1
    return panel


def _shelves(rng, objects, bg, fallback=False):
    dims = [obj.shape for obj in objects]
    candidates = []
    for attempt in range(100):
        if fallback:
            order = [0] + [int(v) for v in rng.permutation([1, 2, 3, 4])]
            groups = [order[:1], order[1:3], order[3:]]
        else:
            order = [int(v) for v in rng.permutation(5)]
            groups, group = [], []
            for j, i in enumerate(order):
                group.append(i)
                if j == 4 or rng.random() < .45:
                    groups.append(group)
                    group = []
        gap = 1 if fallback else int(rng.choice([1, 1, 2]))
        heights = [max(dims[i][0] for i in g) for g in groups]
        widths = [sum(dims[i][1] for i in g)+gap*(len(g)-1) for g in groups]
        h, w = sum(heights)+gap*(len(groups)-1), max(widths)
        if h <= 28 and w <= 28:
            candidates.append((groups, heights, widths, gap, h, w))
        if fallback or len(candidates) == 5:
            break
    if not candidates:
        return None
    groups, heights, widths, gap, h, w = candidates[int(rng.integers(len(candidates)))]
    pads = []
    for extent in (h, w):
        before = int(rng.integers(1, min(3, 29-extent)+1))
        after = int(rng.integers(1, min(3, 30-extent-before)+1))
        pads.append((before, after))
    grid = np.full((h+sum(pads[0]), w+sum(pads[1])), bg, dtype=np.int8)
    y = pads[0][0]
    for group, height, width in zip(groups, heights, widths):
        x = pads[1][0]+int(rng.integers(w-width+1))
        for i in group:
            obj = objects[i]
            r = y+int(rng.integers(height-obj.shape[0]+1))
            view = grid[r:r+obj.shape[0], x:x+obj.shape[1]]
            view[obj != bg] = obj[obj != bg]
            x += obj.shape[1]+gap
        y += height+gap
    return grid


def _transparent_witness(info, answer):
    for corner, p in enumerate(answer["placements"]):
        sig = info["templates"][info["key"][corner]]
        footprint = {(r, c) for r, c, color in sig}
        for ar, ac in p["dots"]:
            for dr in range(min(r for r, c in footprint), max(r for r, c in footprint)+1):
                for dc in range(min(c for r, c in footprint), max(c for r, c in footprint)+1):
                    q = (ar+dr, ac+dc)
                    if (dr, dc) not in footprint and q in p["cells"] and q not in p["stamps"]:
                        return True
    return False


def _make_pair(rng, side, stage):
    # Fixed rejection budgets bound runtime; no attempts or traces are saved.
    for attempt in range(450):
        style = 0 if stage == "first" else 1 if stage == "test" else int(rng.choice([1, 1, 2]))
        cut = _partition(rng, side, style)
        if cut is None:
            continue
        owner, legend = cut
        palette = [int(v) for v in rng.permutation(10)]
        bg, carrier, dot = palette[:3]
        key = tuple(palette[3:7])
        decor = palette[7:9] if rng.random() < .45 else palette[7:8]
        cells = [set(map(tuple, np.argwhere(owner == i).tolist())) for i in range(4)]
        counts = [1]*4
        if stage in ("first", "test"):
            repeat_piece = max(range(4), key=lambda i: len(cells[i])-(5 if i == 0 else 0))
            counts[repeat_piece] = 2
            if stage == "test" and len(cells[repeat_piece]) >= 20 and rng.random() < .25:
                counts[repeat_piece] = 3
        templates, dots = {}, []
        for i in range(4):
            forbidden = {p for p in cells[i] if i == 0 and any(max(abs(p[0]-r), abs(p[1]-c)) < 2 for r, c in legend)}
            sampled = _sample_template(rng, cells[i], forbidden, legend if i == 0 else set(),
                                       key[i], decor, counts[i], stage == "first")
            if sampled is None:
                break
            templates[key[i]], places = sampled
            dots.append(places)
        if len(dots) != 4:
            continue
        pieces = [_normalize(cells[i], dots[i], legend if i == 0 else ()) for i in range(4)]
        irregular = sum(len(p["cells"]) < math.prod(_dimensions(p)) for p in pieces)
        if stage != "first" and irregular < 2:
            continue
        rotations = [0, 1, 2, 3] if stage == "first" else [0]+[int(v) for v in rng.integers(0, 4, size=3)]
        pieces = [_turn(p, k) for p, k in zip(pieces, rotations)]
        info = {"side": side, "key": key, "templates": templates, "pieces": pieces}
        answers = _assemble(info)
        if len(answers) != 1:
            continue
        raw = _assemble(info, "none") if stage in ("first", "test") else []
        if stage in ("first", "test") and len(raw) < 2:
            continue
        if stage == "first":
            if {p["turns"] for p in answers[0]["placements"][1:]} != {1, 2, 3}:
                continue
            if not _transparent_witness(info, answers[0]):
                continue
            wrong = _assemble(info, "piece", limit=385)
            if any(np.array_equal(a["output"], answers[0]["output"]) for a in wrong):
                continue
        # Construct label directly in original assembly coordinates.
        expected = np.asarray(key, dtype=np.int8)[owner]
        for i in range(4):
            for ar, ac in dots[i]:
                for dr, dc, color in templates[key[i]]:
                    expected[ar+dr, ac+dc] = color
        if not np.array_equal(expected, answers[0]["output"]):
            raise AssertionError("construction disagrees with input-only assembly")
        if not set(key) <= set(map(int, expected.flat)):
            continue
        objects = []
        for i, piece in enumerate(pieces):
            obj = np.full(_dimensions(piece), bg, dtype=np.int8)
            for r, c in piece["cells"]:
                obj[r, c] = carrier
            for r, c in piece["dots"]:
                obj[r, c] = dot
            if i == 0:
                r0 = min(r for r, c in piece["legend"])
                c0 = min(c for r, c in piece["legend"])
                obj[r0:r0+2, c0:c0+2] = np.asarray(key).reshape(2, 2)
            objects.append(obj)
        panel = _catalogue(rng, templates, key, bg, carrier)
        grid = _shelves(rng, [panel]+objects, bg)
        if grid is None:
            panel = _catalogue(rng, templates, key, bg, carrier, row_only=True)
            grid = _shelves(rng, [panel]+objects, bg, fallback=True)
        if grid is None:
            continue
        # Parsing also rejects accidental ring/key ambiguities in presentation.
        try:
            parsed = _parse(grid)
            visible_answers = _assemble(parsed)
            if len(visible_answers) != 1 or not np.array_equal(visible_answers[0]["output"], expected):
                continue
        except ValueError:
            continue
        return {"input": grid, "output": expected}
    raise RuntimeError(f"bounded construction exhausted for {stage}, side {side}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _need(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    _need(_integer(num_train) and 2 <= num_train <= 10, "num_train must be an integer in 2..10")
    _need(_integer(num_test) and 1 <= num_test <= 4, "num_test must be an integer in 1..4")
    _need(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(4))
    fixed_side = int(rng.integers(7, 10))
    puzzle, seen, train_keys = {"train": [], "test": []}, set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            stage = "test" if split == "test" else "first" if index == 0 else "other"
            if regime == 0:
                side = int(rng.integers(7, 9) if split == "train" else rng.integers(8, 10))
            elif regime == 1:
                side = int(rng.integers(8, 10) if split == "train" else rng.integers(7, 9))
            elif regime == 2:
                side = fixed_side
            else:
                side = int(rng.integers(7, 10))
            for _ in range(8):
                pair = _make_pair(rng, side, stage)
                signature = (pair["input"].shape, pair["input"].tobytes())
                pair_key = _parse(pair["input"])["key"]
                if split == "train" and index == 1 and pair_key in train_keys:
                    continue
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    if split == "train":
                        train_keys.add(pair_key)
                    break
            else:
                raise RuntimeError("distinct-input retry bound exhausted")
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "exact train/test keys required")
    _need(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal pair counts")
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "exact input/output keys required")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            sig = (inp.shape, inp.tobytes())
            _need(sig not in seen, "duplicate input")
            seen.add(sig)
            info = _parse(inp)
            answers = _assemble(info)
            _need(len(answers) == 1 and np.array_equal(out, answers[0]["output"]), "wrong or ambiguous output")
            _need(out.shape == (info["side"], info["side"]), "area-derived output size mismatch")
            _need(set(info["key"]) <= set(map(int, out.flat)), "every corner color must remain visible")
            _need(not np.array_equal(inp, out), "identity pair")
            records[split].append((info, answers[0]))
    first, first_answer = records["train"][0]
    _need(all(len(p["cells"]) == math.prod(_dimensions(p)) for p in first["pieces"]),
          "first demonstration must isolate registration on rectangular pieces")
    _need(len(_assemble(first, "none")) == 2, "first demonstration must expose geometry-only ambiguity")
    _need(_transparent_witness(first, first_answer), "first demonstration must expose transparency")
    _need(max(len(p["dots"]) for p in first["pieces"]) >= 2, "first demonstration must repeat a motif")
    _need({p["turns"] for p in first_answer["placements"][1:]} == {1, 2, 3},
          "first demonstration must expose all nonzero quarter turns")
    wrong = _assemble(first, "piece", limit=385)
    _need(not any(np.array_equal(a["output"], first_answer["output"]) for a in wrong),
          "first demonstration must distinguish rotating the templates")
    _need(len({info["key"] for info, answer in records["train"]}) >= 2, "training must vary the key")
    for info, answer in records["train"][1:]:
        _need(all(len(p["dots"]) == 1 for p in info["pieces"]), "later training isolates irregular single-dot assembly")
        _need(sum(len(p["cells"]) < math.prod(_dimensions(p)) for p in info["pieces"]) >= 2,
              "later training must demonstrate irregular pieces")
    for info, answer in records["test"]:
        _need(len(_assemble(info, "none")) == 2, "tests must need motif-based registration")
        _need(max(len(p["dots"]) for p in info["pieces"]) >= 2, "tests must combine repetition and assembly")
        _need(sum(len(p["cells"]) < math.prod(_dimensions(p)) for p in info["pieces"]) >= 2,
              "tests must combine irregular pieces and repetition")


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                a = _grid(pair[name])
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds, one puzzle at a time; bounded RAM, zero disk writes."""
    metrics = {name: set() for name in ("puzzles", "canvases", "sides", "cuts", "motifs", "backgrounds", "counts", "rotations", "catalogues")}
    pairs, ambiguous, transparent = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        metrics["puzzles"].add(_digest(puzzle))
        local_cuts = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                answer = _assemble(info)[0]
                _need(np.array_equal(solve(pair["input"].copy()), pair["output"]), "input-only solve failed")
                _need(sum(len(p["cells"]) for p in info["pieces"]) == pair["output"].size, "material conservation")
                masks = tuple(p["mask"] for p in answer["placements"])
                local_cuts.add((info["side"], masks))
                metrics["cuts"].add((info["side"], masks))
                metrics["canvases"].add(pair["input"].shape)
                metrics["catalogues"].add(info["catalogue_shape"])
                metrics["sides"].add(info["side"])
                metrics["backgrounds"].add(info["bg"])
                metrics["counts"].update(len(p["dots"]) for p in info["pieces"])
                metrics["rotations"].update(p["turns"] for p in answer["placements"])
                for sig in info["templates"].values():
                    metrics["motifs"].add(tuple((r, c) for r, c, color in sig))
                ambiguous += len(_assemble(info, "none")) > 1
                transparent += _transparent_witness(info, answer)
                pairs += 1
        _need(len(local_cuts) >= 2, "piece geometry must vary within a puzzle")
    _need(len(metrics["puzzles"]) == 42, "seeds did not change puzzles")
    _need(metrics["sides"] == {7, 8, 9}, "square-size coverage missing")
    _need(len(metrics["canvases"]) >= 25 and len(metrics["cuts"]) >= 100, "geometry coverage too narrow")
    _need(len(metrics["motifs"]) >= 150, "anchor-relative motif coverage too narrow")
    _need(metrics["backgrounds"] == set(range(10)), "background palette coverage missing")
    _need({1, 2} <= metrics["counts"] and metrics["rotations"] == {0, 1, 2, 3}, "repetition or turn coverage missing")
    original = generate_puzzle(0, 2, 1)
    _need(_digest(original) == _digest(generate_puzzle(0, 2, 1)), "determinism failed")
    # Role parsing and page orientation survive visible, seed-free relabeling.
    permutation = np.asarray([9, 2, 0, 8, 1, 7, 3, 6, 4, 5], dtype=np.int8)
    for pair in original["train"]+original["test"]:
        before = pair["input"].copy()
        _need(np.array_equal(solve(permutation[before]), permutation[pair["output"]]), "palette equivariance failed")
        _need(np.array_equal(solve(before.T.copy()), pair["output"].T), "transpose equivariance failed")
        _need(np.array_equal(pair["input"], before), "solver mutated its input")
    # Leave the catalogue and every footprint/dot unchanged; permute key roles.
    changed = original["train"][0]["input"].copy()
    kr, kc = _parse(changed)["key_box"]
    changed[kr, kc], changed[kr, kc+1] = changed[kr, kc+1], changed[kr, kc]
    counterfactual = _assemble(_parse(changed), limit=385)
    _need(not any(np.array_equal(a["output"], original["train"][0]["output"]) for a in counterfactual),
          "key-to-corner/template correspondence did not affect the answer")
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": pairs, "geometry_ambiguous_pairs": ambiguous,
            "transparent_witness_pairs": transparent,
            "distinct": {name: len(values) for name, values in metrics.items()}}
