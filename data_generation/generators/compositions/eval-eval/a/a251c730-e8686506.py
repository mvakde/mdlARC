"""Stencil-guided exact inventory assembly, using only visible input evidence.

The reference badges constrain colors, not piece boundaries or inventory counts.
All search, randomness, and metrics are local to a call. No artifacts are written.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib

import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NODE_LIMIT = 16000


class _SearchLimit(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _components(cells):
    remaining = {(int(r), int(c)) for r, c in cells}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, part = [start], []
        while stack:
            r, c = stack.pop()
            part.append((r, c))
            for dr, dc in _D4:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(tuple(sorted(part)))
    return result


def _normalize(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), "invalid grid shape")
    _require(np.issubdtype(grid.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= grid) & (grid <= 9)), "colors must be 0..9")
    return grid.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _interpret(input_grid):
    """Find the board, distinguish badges from pieces, and read probe constraints."""
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _require(all(np.all(edge == bg) for edge in
                 (grid[0], grid[-1], grid[:, 0], grid[:, -1])), "uniform outer background rim required")
    candidates = []
    # The scaffold has a private color. Its COMPLETE extent has a rectangular
    # border, so neither a solid loose rectangle nor a partial badge is a board.
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        rr, cc = np.where(grid == color)
        top, left, bottom, right = int(rr.min()), int(cc.min()), int(rr.max())+1, int(cc.max())+1
        box = grid[top:bottom, left:right]
        if min(box.shape) < 6:
            continue
        if not all(np.all(edge == color) for edge in (box[0], box[-1], box[:, 0], box[:, -1])):
            continue
        inner = box[1:-1, 1:-1]
        if np.any(inner == bg) and np.any((inner != bg) & (inner != color)):
            candidates.append((top, left, bottom, right, color))
    _require(len(candidates) == 1, "exactly one framed assembly board required")
    top, left, bottom, right, scaffold = candidates[0]
    board = grid[top:bottom, left:right].copy()
    keys = set(map(int, np.unique(board))) - {bg, scaffold}
    _require(1 <= len(keys) <= 3, "one to three visible peg identities required")
    pegs = [(int(r), int(c), int(board[r, c])) for r, c in np.argwhere(
        (board != bg) & (board != scaffold))]
    outside = grid != bg
    outside[top:bottom, left:right] = False
    pieces, templates, repetitions = [], {}, Counter()
    for part in _components(map(tuple, np.argwhere(outside))):
        tagged = [(r, c) for r, c in part if int(grid[r, c]) in keys]
        if tagged:
            _require(len(tagged) == 1, "one peg per reference badge required")
            ar, ac = tagged[0]
            key = int(grid[ar, ac])
            signature = tuple(sorted((r-ar, c-ac, int(grid[r, c])) for r, c in part))
            _require(len(signature) >= 4, "reference badge too small")
            _require(key not in templates or templates[key] == signature, "inconsistent reference badges")
            _require(len({v for dr, dc, v in signature if (dr, dc) != (0, 0)}) >= 2,
                     "each stencil must cross at least two payload colors")
            templates[key] = signature
            repetitions[key] += 1
        else:
            colors = {int(grid[p]) for p in part}
            _require(len(colors) == 1 and scaffold not in colors, "loose pieces must be monochrome")
            pieces.append((next(iter(colors)), _normalize(part)))
    _require(set(templates) == keys and 2 <= len(pieces) <= 14, "missing dictionary or inventory")
    payload = {color for color, shape in pieces}
    _require(not (payload & (keys | {bg, scaffold})), "payload and marker roles must differ")
    requirements, coverage = {}, Counter()
    for ar, ac, key in pegs:
        for dr, dc, color in templates[key]:
            if (dr, dc) == (0, 0):
                continue
            r, c = ar+dr, ac+dc
            _require(0 <= r < board.shape[0] and 0 <= c < board.shape[1], "stencil clipped by board")
            _require(int(board[r, c]) == bg and color in payload, "probe must address a vacant payload cell")
            _require((r, c) not in requirements or requirements[r, c] == color, "conflicting overlapping probes")
            requirements[r, c] = color
            coverage[r, c] += 1
    holes = [tuple(map(int, p)) for p in np.argwhere(board == bg)]
    _require(sum(len(shape) for color, shape in pieces) == len(holes), "inventory area must equal all gaps")
    return {"grid": grid, "board": board, "bbox": (top, left, bottom, right), "bg": bg,
            "scaffold": scaffold, "pegs": pegs, "templates": templates,
            "repetitions": repetitions, "pieces": pieces, "requirements": requirements,
            "coverage": coverage, "holes": holes}


def _assemble(board, bg, pieces, requirements, node_limit=_NODE_LIMIT):
    """Exact cover; return up to two distinct COLORED answers, not piece orders.

    A bounded local cache and node budget limit memory and runtime. Reaching the
    budget is a rejection, never evidence of uniqueness.
    """
    holes = [tuple(map(int, p)) for p in np.argwhere(board == bg)]
    if sum(len(s) for _, s in pieces) != len(holes):
        return [], {"nodes": 0, "max_initial_choices": 0}
    index = {p: i for i, p in enumerate(holes)}
    inventory = Counter((int(color), _normalize(shape)) for color, shape in pieces)
    types = sorted(inventory)
    by_cell = [[] for _ in holes]
    choices_per_type = []
    for t, (color, shape) in enumerate(types):
        height = max(r for r, c in shape)+1
        width = max(c for r, c in shape)+1
        count = 0
        for r in range(board.shape[0]-height+1):
            for c in range(board.shape[1]-width+1):
                cells = [(r+dr, c+dc) for dr, dc in shape]
                if not all(p in index and requirements.get(p, color) == color for p in cells):
                    continue
                ids = [index[p] for p in cells]
                mask = sum(1 << i for i in ids)
                paint = sum((color+1) << (4*i) for i in ids)
                move = (t, mask, paint)
                for i in ids:
                    by_cell[i].append(move)
                count += 1
        choices_per_type.append(count)
    nodes = 0

    @lru_cache(maxsize=4096)
    def search(remaining, counts):
        nonlocal nodes
        nodes += 1
        if node_limit is not None and nodes > node_limit:
            raise _SearchLimit("exact-cover node budget exceeded")
        if not remaining:
            return (0,) if not any(counts) else ()
        best = None
        bits = remaining
        while bits:
            bit = bits & -bits
            i = bit.bit_length()-1
            choices = [m for m in by_cell[i] if counts[m[0]] and m[1] & remaining == m[1]]
            if not choices:
                return ()
            if best is None or len(choices) < len(best):
                best = choices
                if len(best) == 1:
                    break
            bits ^= bit
        answers = set()
        for t, mask, paint in best:
            rest = list(counts)
            rest[t] -= 1
            for code in search(remaining ^ mask, tuple(rest)):
                answers.add(code | paint)
                if len(answers) == 2:
                    return tuple(sorted(answers))
        return tuple(sorted(answers))

    codes = search((1 << len(holes))-1, tuple(inventory[t] for t in types))
    result = []
    for code in codes:
        answer = board.copy()
        for p, i in index.items():
            answer[p] = ((code >> (4*i)) & 15)-1
        result.append(answer)
    return result, {"nodes": nodes, "max_initial_choices": max(choices_per_type, default=0)}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the unique completion from this input alone."""
    info = _interpret(input_grid)
    # The public solver is complete for the finite placement problem. Only
    # construction/diagnostic calls have a time-budget rejection threshold.
    answers, _ = _assemble(info["board"], info["bg"], info["pieces"], info["requirements"], node_limit=None)
    _require(len(answers) == 1, "input must have one colored completion")
    return answers[0]


def _random_shape(rng, size):
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _D4} - cells)
        feasible = [p for p in frontier if
                    max(r for r, c in cells | {p})-min(r for r, c in cells | {p}) < 4 and
                    max(c for r, c in cells | {p})-min(c for r, c in cells | {p}) < 4]
        cells.add(_pick(rng, feasible))
    return _normalize(cells)


def _partition(rng, holes):
    """Include interchangeable silhouettes of different colors, then grow tiles."""
    free = set(holes)
    shape = _random_shape(rng, int(rng.integers(2, 6)))
    candidates = []
    for r, c in sorted(free):
        p = {(r+dr, c+dc) for dr, dc in shape}
        if p <= free:
            candidates.append(p)
    rng.shuffle(candidates)
    for first in candidates:
        others = [p for p in candidates if not p & first]
        if others:
            second = _pick(rng, others)
            parts = [first, second]
            free -= first | second
            break
    else:
        return None
    while free:
        start = _pick(rng, sorted(free))
        part = {start}
        free.remove(start)
        target = int(rng.integers(3, 9))
        while len(part) < target:
            frontier = sorted({(r+dr, c+dc) for r, c in part for dr, dc in _D4} & free)
            frontier = [p for p in frontier if
                        max(r for r, c in part | {p})-min(r for r, c in part | {p}) < 5 and
                        max(c for r, c in part | {p})-min(c for r, c in part | {p}) < 5]
            if not frontier:
                break
            p = _pick(rng, frontier)
            part.add(p)
            free.remove(p)
        parts.append(part)
        if len(parts) > 12:
            return None
    return parts


def _probes(rng, peg, holes, answer, amount):
    """Connected transparent stencil; it does not encode the tile segmentation."""
    cells = {peg}
    for _ in range(amount):
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _D4} & holes - cells)
        frontier = [p for p in frontier if
                    max(r for r, c in cells | {p})-min(r for r, c in cells | {p}) < 5 and
                    max(c for r, c in cells | {p})-min(c for r, c in cells | {p}) < 5]
        if not frontier:
            break
        cells.add(_pick(rng, frontier))
    cells.remove(peg)
    if len(cells) < 3 or len({int(answer[p]) for p in cells}) < 2:
        return None
    return cells


def _scene(rng, board, templates, pieces, bg, copies):
    """Scatter objects with a one-cell background moat; grow canvas only as needed."""
    objects = []
    for key, signature in templates.items():
        for _ in range(copies[key]):
            lr = min(r for r, c, v in signature)
            lc = min(c for r, c, v in signature)
            objects.append(tuple((r-lr, c-lc, v) for r, c, v in signature))
    objects += [tuple((r, c, color) for r, c in shape) for color, shape in pieces]
    for attempt in range(12):
        low = min(28, 20+attempt)
        ch, cw = (int(rng.integers(low, 31)) for _ in range(2)) if attempt < 10 else (30, 30)
        h, w = board.shape
        br, bc = int(rng.integers(1, ch-h)), int(rng.integers(1, cw-w))
        grid = np.full((ch, cw), bg, dtype=np.int8)
        grid[br:br+h, bc:bc+w] = board
        blocked = np.zeros((ch, cw), dtype=bool)
        blocked[0] = blocked[-1] = True
        blocked[:, 0] = blocked[:, -1] = True
        blocked[br-1:br+h+1, bc-1:bc+w+1] = True
        ordered = list(objects)
        rng.shuffle(ordered)
        ordered.sort(key=lambda obj: -(max(r for r, c, v in obj)+1)*(max(c for r, c, v in obj)+1))
        for obj in ordered:
            oh = max(r for r, c, v in obj)+1
            ow = max(c for r, c, v in obj)+1
            positions = [(r, c) for r in range(1, ch-oh) for c in range(1, cw-ow)
                         if all(not blocked[r+dr, c+dc] for dr, dc, v in obj)]
            if not positions:
                break
            r, c = _pick(rng, positions)
            for dr, dc, color in obj:
                rr, cc = r+dr, c+dc
                grid[rr, cc] = color
                blocked[rr, cc] = True
                for dy, dx in _D4:
                    blocked[rr+dy, cc+dx] = True
        else:
            return grid
    return None


def _make_pair(rng, palette, nkeys, first=False, overlap=False):
    # Bounded proposals; each is small and entirely in memory. No fallback may
    # silently weaken uniqueness, conservation, or either-parent witnesses.
    for attempt in range(240):
        h = int(rng.integers(6, 9 if first else 10))
        w = int(rng.integers(6, 9 if first else 11))
        bg, scaffold = map(int, palette[:2])
        keys = list(map(int, palette[2:2+nkeys]))
        ncolors = int(rng.integers(2, 5))
        payload = list(map(int, palette[2+nkeys:2+nkeys+ncolors]))
        board = np.full((h, w), scaffold, dtype=np.int8)
        board[1:-1, 1:-1] = bg
        # Internal scaffold islands are bilateral; the colored completed board
        # need not be symmetric. Symmetry never manufactures an inventory item.
        if not first:
            for _ in range(int(rng.integers(4))):
                r, c = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
                board[r, c] = board[r, w-1-c] = scaffold
        options = [tuple(map(int, p)) for p in np.argwhere(board == bg)]
        rng.shuffle(options)
        pegs = []
        for p in options:
            if all(abs(p[0]-q[0])+abs(p[1]-q[1]) >= 2 for q in pegs):
                pegs.append(p)
                if len(pegs) == nkeys:
                    break
        if len(pegs) != nkeys:
            continue
        for p, key in zip(pegs, keys):
            board[p] = key
        holes = {tuple(map(int, p)) for p in np.argwhere(board == bg)}
        parts = _partition(rng, holes)
        if parts is None:
            continue
        colors = [payload[0], payload[1]] + [_pick(rng, payload) for _ in parts[2:]]
        answer = board.copy()
        pieces = []
        for cells, color in zip(parts, colors):
            for p in cells:
                answer[p] = color
            pieces.append((color, _normalize(cells)))
        for probe_attempt in range(10):
            templates, requirements, cover = {}, {}, Counter()
            for p, key in zip(pegs, keys):
                probes = _probes(rng, p, holes, answer, int(rng.integers(5, 11)))
                if probes is None:
                    break
                templates[key] = tuple(sorted([(0, 0, key)] +
                    [(r-p[0], c-p[1], int(answer[r, c])) for r, c in probes]))
                for q in probes:
                    requirements[q] = int(answer[q])
                    cover[q] += 1
            if len(templates) != nkeys or len(holes)-len(requirements) < 3:
                continue
            if overlap and not any(n > 1 for n in cover.values()):
                continue
            # At least one supplied piece extends a probe into unmarked space.
            if not any(cells & set(requirements) and cells - set(requirements) for cells in parts):
                continue
            try:
                answers, stats = _assemble(board, bg, pieces, requirements)
            except _SearchLimit:
                continue
            if len(answers) != 1 or not np.array_equal(answers[0], answer):
                continue
            if stats["max_initial_choices"] < 2:
                continue
            copies = {k: 2 if first else int(rng.integers(1, 3)) for k in keys}
            grid = _scene(rng, board, templates, pieces, bg, copies)
            if grid is None:
                continue
            turns = int(rng.integers(4))
            grid, output = np.rot90(grid, turns).copy(), np.rot90(answer, turns).copy()
            if rng.random() < .5:
                grid, output = grid[:, ::-1].copy(), output[:, ::-1].copy()
            # Re-prove the transformed input within the construction budget:
            # changing orientation can change the exact-cover search order.
            # The public complete solver never needs this rejection cutoff.
            try:
                visible = _interpret(grid)
                recovered, _ = _assemble(visible["board"], visible["bg"], visible["pieces"],
                                         visible["requirements"])
            except ValueError:
                continue
            if len(recovered) == 1 and np.array_equal(recovered[0], output):
                return {"input": grid, "output": output}
    raise RuntimeError("bounded construction exhausted; no invalid fallback emitted")


def _checked_count(value, name, lo, hi):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _checked_count(num_train, "num_train", 2, 10)
    ne = _checked_count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = rng.permutation(10)
    stable_palette = bool(rng.random() < .3)
    result, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            nkeys = (1 if i == 0 else 2 if i == 1 else int(rng.integers(1, 3))) if split == "train" else 3
            for retry in range(8):
                pair_palette = palette if stable_palette else rng.permutation(10)
                pair = _make_pair(rng, pair_palette, nkeys, first=split == "train" and i == 0,
                                  overlap=nkeys > 1)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct-input retry budget exhausted")
            seen.add(key)
            result[split].append(pair)
    return result


def pair_metrics(pair, ablate=True):
    info = _interpret(pair["input"])
    answer, stats = _assemble(info["board"], info["bg"], info["pieces"], info["requirements"])
    _require(len(answer) == 1 and np.array_equal(answer[0], pair["output"]), "incorrect or ambiguous label")
    out = answer[0]
    fixed = info["board"] != info["bg"]
    _require(np.array_equal(out[fixed], info["board"][fixed]), "fixed cells changed")
    _require(not np.any(out == info["bg"]), "unfilled board")
    expected = Counter()
    for color, shape in info["pieces"]:
        expected[color] += len(shape)
    _require(expected == Counter(map(int, out[~fixed])), "payload conservation failed")
    unprobed = len(info["holes"])-len(info["requirements"])
    _require(unprobed >= 3, "stencil-only shortcut must leave cells unresolved")
    _require(stats["max_initial_choices"] >= 2, "inventory placements are all locally predetermined")
    if ablate:
        without_a, _ = _assemble(info["board"], info["bg"], info["pieces"], {})
        _require(len(without_a) == 2, "stencils must distinguish shape-only colored alternatives")
    template_shapes = [tuple((r, c) for r, c, v in sig) for sig in info["templates"].values()]
    return {"input_shape": tuple(info["grid"].shape), "output_shape": tuple(out.shape),
            "keys": len(info["templates"]), "pieces": len(info["pieces"]),
            "piece_shapes": tuple(sorted(set(shape for color, shape in info["pieces"]))),
            "piece_areas": tuple(sorted({len(shape) for color, shape in info["pieces"]})),
            "template_shapes": tuple(template_shapes), "probe_counts": tuple(len(s)-1 for s in template_shapes),
            "shared_probes": sum(n > 1 for n in info["coverage"].values()), "unprobed": unprobed,
            "background": info["bg"], "nodes": stats["nodes"],
            "without_a_ambiguous": bool(ablate), "without_b_incomplete": True,
            "nonsolid_stencil": any((max(r for r, c in s)-min(r for r, c in s)+1)*
                                    (max(c for r, c in s)-min(c for r, c in s)+1) > len(s)
                                    for s in template_shapes)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[k], list) for k in puzzle), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    seen, train_metrics = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            metrics = pair_metrics({"input": inp, "output": out})
            if split == "train":
                train_metrics.append(metrics)
    _require(any(m["keys"] >= 2 and m["shared_probes"] for m in train_metrics),
             "demonstrations must expose compatible overlapping stencils")


def self_test():
    """Stream 42 seeds, retaining only bounded sets of structural measurements."""
    puzzles, ins, outs, shapes, stencils, areas, counts, backgrounds = (set() for _ in range(8))
    pairs = shared = nonsolid = 0
    max_nodes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                metrics = pair_metrics(pair)
                digest.update(repr(_key(pair["input"])).encode())
                ins.add(metrics["input_shape"])
                outs.add(metrics["output_shape"])
                shapes.update(metrics["piece_shapes"])
                stencils.update(metrics["template_shapes"])
                areas.update(metrics["piece_areas"])
                counts.add(metrics["pieces"])
                backgrounds.add(metrics["background"])
                pairs += 1
                shared += metrics["shared_probes"] > 0
                nonsolid += metrics["nonsolid_stencil"]
                max_nodes = max(max_nodes, metrics["nodes"])
        puzzles.add(digest.digest())
    _require(len(puzzles) == 42, "seed diversity failed")
    _require(len(ins) >= 25 and len(outs) >= 20, "geometry diversity failed")
    _require(len(shapes) >= 80 and len(stencils) >= 100, "procedural shape diversity failed")
    _require(len(areas) >= 6 and len(counts) >= 5 and len(backgrounds) >= 8, "count or palette diversity failed")
    _require(shared >= 80 and nonsolid >= 100, "stencil interaction coverage failed")
    # Distinct colored completions versus harmless identical-piece permutations.
    board = np.array([[0, 8, 0]], dtype=np.int8)
    _require(len(_assemble(board, 0, [(1, ((0, 0),)), (2, ((0, 0),))], {})[0]) == 2,
             "color ambiguity missed")
    _require(len(_assemble(board, 0, [(1, ((0, 0),)), (1, ((0, 0),))], {})[0]) == 1,
             "identical-piece exchange treated as different answer")
    _require(not _assemble(np.zeros((1, 2), dtype=np.int8), 0,
                           [(1, ((0, 0), (1, 0)))], {})[0], "illegal piece rotation accepted")
    first, repeat = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    _require(all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], repeat[s])
                 for k in ("input", "output")), "nondeterministic generation")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Bad labels, lost inventory and inconsistent duplicate badges must fail.
    info = _interpret(first["train"][0]["input"])
    corrupt = first["train"][0]["output"].copy()
    corrupt[0, 0] = info["bg"]
    try:
        pair_metrics({"input": first["train"][0]["input"], "output": corrupt})
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt fixed output accepted")
    inp = first["train"][0]["input"]
    outside = inp != info["bg"]
    top, left, bottom, right = info["bbox"]
    outside[top:bottom, left:right] = False
    components = _components(map(tuple, np.argwhere(outside)))
    keys = set(info["templates"])
    for kind in ("inventory", "reference"):
        bad = inp.copy()
        if kind == "inventory":
            part = next(p for p in components if not any(int(inp[q]) in keys for q in p))
            for p in part:
                bad[p] = info["bg"]
        else:
            part = next(p for p in components if any(int(inp[q]) in keys for q in p))
            cell = next(p for p in part if int(inp[p]) not in keys)
            bad[cell] = next(color for color, shape in info["pieces"] if color != int(inp[cell]))
        try:
            solve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"corrupt {kind} accepted")
    # All eight whole-scene orientations and a full palette permutation preserve
    # the rule, including the reference offset directions and supplied pieces.
    permutation = np.random.default_rng(777).permutation(10)
    for pair in (first["train"][0], first["train"][1], first["test"][0]):
        for turns in range(4):
            for reflect in (False, True):
                inp = np.rot90(pair["input"], turns)
                out = np.rot90(pair["output"], turns)
                if reflect:
                    inp, out = inp[:, ::-1], out[:, ::-1]
                _require(np.array_equal(solve(permutation[inp]), permutation[out]),
                         "whole-scene orientation/palette equivariance failed")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(puzzles), "input_shapes": len(ins),
            "output_shapes": len(outs), "piece_shapes": len(shapes), "stencil_shapes": len(stencils),
            "piece_areas": sorted(areas), "piece_counts": sorted(counts), "backgrounds": sorted(backgrounds),
            "overlap_pairs": shared, "nonsolid_stencil_pairs": nonsolid, "max_search_nodes": max_nodes}
