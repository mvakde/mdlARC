"""Cut a fused stencil, identify its parts by enlarged specimens, and dispatch.

The only solver input is the grid. Frames group disconnected specimens; the
marked frame identifies the stencil. No parent generator is imported or run.
"""
from collections import Counter
from functools import lru_cache
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_SIDES = ("top", "right", "bottom", "left")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid grid colors")
    return a.astype(np.int8, copy=True)


def _key(grid):
    return tuple(grid.shape), grid.astype(np.int8).tobytes()


def _components(cells):
    remaining = set(cells)
    found = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        part, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        found.append(part)
    return found


def _norm(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


def _size(cells):
    return max(r for r, c in cells)+1, max(c for r, c in cells)+1


def _rotate(cells, turns):
    for _ in range(turns % 4):
        cells = [(c, -r) for r, c in cells]
    return _norm(cells)


@lru_cache(maxsize=4096)
def _poses(shape, reflect=True):
    bases = (shape, tuple((r, -c) for r, c in shape)) if reflect else (shape,)
    return tuple(sorted({_rotate(base, q) for base in bases for q in range(4)}))


def _canonical(cells):
    return _poses(_norm(cells))[0]


def _rim_positions(h, w):
    return ([(0, c) for c in range(w)] + [(h-1, c) for c in range(w)]
            + [(r, 0) for r in range(1, h-1)] + [(r, w-1) for r in range(1, h-1)])


def _side(p, shape):
    r, c = p
    h, w = shape
    flags = [s for s, yes in (("top", r == 0), ("right", c == w-1),
                             ("bottom", r == h-1), ("left", c == 0)) if yes]
    _require(len(flags) == 1, "marker must be on one side, away from corners")
    return flags[0]


def _cut(cells, ar, ac, sides):
    parts, fixed = {}, set()
    for r, c in cells:
        if r == ar or c == ac:
            fixed.add((r, c))
            continue
        v = ("top" if r < ar else "bottom") if ar is not None else next(iter(sides))
        h = ("left" if c < ac else "right") if ac is not None else next(iter(sides))
        parts.setdefault((v, h), set()).add((r, c))
    return parts, fixed


def _retained(corner, sides):
    return any(s in sides for s in corner)


def _features(cells, ar, ac, sides):
    parts, fixed = _cut(cells, ar, ac, sides)
    gaps = 0
    for axis, at in ((0, ac), (1, ar)):
        if at is not None:
            hits = sorted(p[axis] for p in cells if p[1-axis] == at)
            if not hits:
                return None
            gaps += sum(b-a-1 for a, b in zip(hits, hits[1:]))
    retained = [v for k, v in parts.items() if _retained(k, sides)]
    classes = {_canonical(v) for v in parts.values()}
    reflection = any(len(_poses(_norm(v))) > len(_poses(_norm(v), False)) for v in retained)
    return {"parts": parts, "fixed": fixed, "gaps": gaps,
            "fragmented": any(len(_components(v)) > 1 for v in retained),
            "chiral": reflection, "classes": classes,
            "equal_area_distinct": any(a != b and len(a) == len(b)
                                       for a in classes for b in classes)}


def _frames(a, bg):
    """Recognize isolated rectangular rings, including the interrupted source ring.

    The union of all non-background colors reconnects frame and marker cells.
    An empty moat separates each ring from its content and from other frames.
    """
    result = []
    for comp in _components(map(tuple, np.argwhere(a != bg))):
        rs, cs = zip(*comp)
        r0, r1, c0, c1 = min(rs), max(rs)+1, min(cs), max(cs)+1
        h, w = r1-r0, c1-c0
        if min(h, w) < 5:
            continue
        board = a[r0:r1, c0:c1].copy()
        s = int(board[0, 0])
        if s == bg or any(board[p] != s for p in ((0, w-1), (h-1, 0), (h-1, w-1))):
            continue
        rim = _rim_positions(h, w)
        if len(comp) != len(rim) or any((r+r0, c+c0) not in comp for r, c in rim):
            continue
        changes = [p for p in rim if board[p] != s]
        if len(changes) > 2 or any(board[p] == bg for p in changes):
            continue
        if changes and len({int(board[p]) for p in changes}) != 1:
            continue
        # A full background moat inside the frame makes grouping explicit.
        if not (np.all(board[1, 1:-1] == bg) and np.all(board[-2, 1:-1] == bg)
                and np.all(board[1:-1, 1] == bg) and np.all(board[1:-1, -2] == bg)):
            continue
        inner_colors = set(map(int, np.unique(board[1:-1, 1:-1]))) - {bg}
        if len(inner_colors) != 1 or s in inner_colors:
            continue
        color = inner_colors.pop()
        if changes and color == int(board[changes[0]]):
            continue
        result.append({"board": board, "bounds": (r0, r1, c0, c1),
                       "structure": s, "color": color, "markers": changes,
                       "cells": set(map(tuple, np.argwhere(board == color)))})
    return result


def _decode(cells, scale):
    normalized = _norm(cells)
    h, w = _size(normalized)
    if h % scale or w % scale or len(cells) % (scale*scale):
        return None
    blocks = Counter((r//scale, c//scale) for r, c in normalized)
    if any(v != scale*scale for v in blocks.values()):
        return None
    return tuple(sorted(blocks))


def _recover(value):
    a = _grid(value)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] > a.size//2, "background must occupy more than half the input")
    frames = _frames(a, bg)
    sources = [f for f in frames if f["markers"]]
    _require(len(sources) == 1, "expected one marked stencil frame")
    source = sources[0]
    specimens = [f for f in frames if not f["markers"]]
    b = source["board"]
    h, w = b.shape
    s, p = source["structure"], source["color"]
    m = int(b[source["markers"][0]])
    _require(len({bg, s, p, m}) == 4, "source roles must be distinct")
    _require(7 <= min(h, w) and max(h, w) <= 15, "stencil dimensions outside family")
    _require(all(f["structure"] == s for f in frames), "all frames share a structural color")
    _require(all(f["color"] not in (bg, s, p, m) for f in specimens), "specimen role conflict")
    covered = np.zeros(a.shape, dtype=bool)
    for f in frames:
        r0, r1, c0, c1 = f["bounds"]
        _require(not np.any(covered[r0:r1, c0:c1]), "overlapping frames")
        # Exterior moats separate frames even when they share colors.
        outside = np.ones(a.shape, dtype=bool)
        outside[r0:r1, c0:c1] = False
        near = np.zeros(a.shape, dtype=bool)
        near[max(0, r0-1):min(a.shape[0], r1+1), max(0, c0-1):min(a.shape[1], c1+1)] = True
        _require(not np.any((a != bg) & near & outside), "missing exterior moat")
        covered[r0:r1, c0:c1] = True
    _require(not np.any((a != bg) & ~covered), "unframed colored content")
    marker_map = {_side(q, b.shape): q for q in source["markers"]}
    _require(len(marker_map) == len(source["markers"]), "duplicate marked side")
    sides = set(marker_map)
    vertical = sides & {"top", "bottom"}
    horizontal = sides & {"left", "right"}
    _require(len(sides) == 1 or len(vertical) == len(horizontal) == 1,
             "two markers must occupy adjacent sides")
    ar = marker_map[next(iter(horizontal))][0] if horizontal else None
    ac = marker_map[next(iter(vertical))][1] if vertical else None
    cells = source["cells"]
    _require(len(_components(cells)) == 1, "fused stencil must be connected")
    f = _features(cells, ar, ac, sides)
    _require(f is not None and len(f["parts"]) == (4 if len(sides) == 2 else 2),
             "each cut region and each axis must contain payload")
    parts = f["parts"]
    _require(len(f["classes"]) >= 2, "at least two different template shapes required")
    _require(len(specimens) == len(parts), "one specimen per cut region, including discarded region")
    target_counts = Counter(_canonical(v) for v in parts.values())
    solutions = []
    for scale in range(2, 5):
        if h*scale > 30 or w*scale > 30:
            continue
        decoded = [_decode(t["cells"], scale) for t in specimens]
        if any(v is None for v in decoded):
            continue
        if Counter(_canonical(v) for v in decoded) != target_counts:
            continue
        color_map = {}
        ambiguous = False
        for t, shape in zip(specimens, decoded):
            k = _canonical(shape)
            if k in color_map and color_map[k] != t["color"]:
                ambiguous = True
            color_map[k] = t["color"]
        _require(not ambiguous, "congruent templates have differently colored specimens")
        reflection = rotation = 0
        for corner, part in parts.items():
            if not _retained(corner, sides):
                continue
            k = _canonical(part)
            if target_counts[k] != 1:
                continue
            shape = decoded[next(i for i, v in enumerate(decoded) if _canonical(v) == k)]
            dest = _norm(part)
            reflection += dest not in _poses(shape, False)
            rotation += dest != shape and dest in _poses(shape, False)
        solutions.append((scale, color_map, reflection, rotation, decoded))
    _require(len(solutions) == 1, "expected exactly one common scale and matching")
    scale, color_map, reflection, rotation, decoded = solutions[0]
    info = {"background": bg, "structure": s, "payload": p, "marker": m,
            "source": source, "specimens": specimens, "sides": sides,
            "markers": marker_map, "ar": ar, "ac": ac, "cells": cells,
            "parts": parts, "fixed": f["fixed"], "scale": scale,
            "color_map": color_map, "reflection": int(reflection), "rotation": int(rotation),
            "gaps": f["gaps"], "fragmented": f["fragmented"],
            "equal_area_distinct": f["equal_area_distinct"], "decoded": decoded}
    return info


def _render(info, *, nearest=False, separate=False, monochrome=False, no_dispatch=False):
    """Render the intended logical answer, or explicit simpler-rule diagnostics."""
    b = info["source"]["board"]
    bg, p, m = info["background"], info["payload"], info["marker"]
    h, w = b.shape
    out = np.full((h, w), bg, dtype=np.int8)
    for q in info["fixed"]:
        out[q] = p
    for side, start in info["markers"].items():
        dr, dc = _DIRS[(_SIDES.index(side)+2) % 4]
        ray, r, c = [], *start
        while 0 <= r < h and 0 <= c < w:
            ray.append((r, c))
            r, c = r+dr, c+dc
        hits = [i for i, q in enumerate(ray) if q in info["cells"]]
        stop = min(hits) if nearest else max(hits)
        for q in ray[:stop+1]:
            if out[q] == bg:
                out[q] = m
    for corner, part in info["parts"].items():
        if not no_dispatch and not _retained(corner, info["sides"]):
            continue
        color = p if monochrome else info["color_map"][_canonical(part)]
        if no_dispatch:
            for q in part:
                out[q] = color
            continue
        groups = _components(part) if separate else [part]
        for group in groups:
            rs, cs = zip(*group)
            dr = -min(rs) if corner[0] == "top" else h-1-max(rs)
            dc = -min(cs) if corner[1] == "left" else w-1-max(cs)
            for r, c in group:
                q = r+dr, c+dc
                if not separate:
                    _require(out[q] == bg, "corner translation collided with a guide or another part")
                out[q] = color
    return np.repeat(np.repeat(out, info["scale"], 0), info["scale"], 1)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover frames, cut templates, common scale and colors from this input."""
    return _render(_recover(input_grid))


def _grow(rng, height, width, area, thin):
    cells = {(int(rng.integers(height)), int(rng.integers(width)))}
    while len(cells) < area:
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _DIRS
                           if 0 <= r+dr < height and 0 <= c+dc < width} - cells)
        if not frontier:
            break
        weights = np.array([sum((r+dr, c+dc) in cells for dr, dc in _DIRS)
                            for r, c in frontier], dtype=float)
        weights = 1/weights if thin else weights**2
        weights /= weights.sum()
        cells.add(frontier[int(rng.choice(len(frontier), p=weights))])
    return cells


def _witness_payload(rng):
    # A chiral L, a two-fragment triad, a straight triomino and a square.
    # Off-axis additions broaden this witness without repairing its axial gaps.
    base = {(0, 2), (1, 2), (2, 2), (2, 1), (0, 4), (2, 4), (2, 5),
            (4, 2), (5, 2), (6, 2), (4, 4), (4, 5), (5, 4), (5, 5),
            (0, 3), (2, 3), (3, 3), (4, 3), (6, 3), (3, 1), (3, 2), (3, 4), (3, 5)}
    cells = base.copy()
    options = [(2, 0), (1, 1), (0, 1), (4, 1), (5, 1), (6, 1), (4, 6), (5, 6), (6, 4)]
    rng.shuffle(options)
    for q in options[:int(rng.integers(0, len(options)+1))]:
        proposed = cells | {q}
        f = _features(proposed, 3, 3, {"top", "right"})
        if (len(_components(proposed)) == 1 and f["fragmented"] and f["chiral"]
                and f["equal_area_distinct"]):
            cells = proposed
    return cells, 3, 3


def _payload(rng, h, w, double, witness):
    if witness:
        return _witness_payload(rng)
    for _ in range(60):
        ph = int(rng.integers(3, min(8, h-3)+1))
        pw = int(rng.integers(3, min(8, w-3)+1))
        low = 7 if double else 5
        area = int(rng.integers(low, max(low+1, min(40, ph*pw*4//5)+1)))
        cells = _grow(rng, ph, pw, area, bool(rng.integers(2)))
        cells = set(_norm(cells))
        sh, sw = _size(cells)
        if min(sh, sw) < 3:
            continue
        candidates = [(ar, ac) for ar in (range(1, sh-1) if double else [None])
                      for ac in range(1, sw-1)]
        rng.shuffle(candidates)
        for ar, ac in candidates:
            f = _features(cells, ar, ac, {"top", "right"} if double else {"top"})
            if f is None or len(f["parts"]) != (4 if double else 2) or len(f["classes"]) < 2:
                continue
            return cells, ar, ac
    # Connected unequal lobes: small fallback for even a scale-four stencil.
    if double:
        cells = {(r, 2) for r in range(5)} | {(2, c) for c in range(5)}
        cells |= {(0, 1), (1, 1), (1, 0), (0, 3), (3, 1), (3, 3), (4, 3)}
        return cells, 2, 2
    return ({(r, 2) for r in range(5)} | {(0, 1), (1, 1), (2, 1), (2, 0),
                                        (1, 3), (2, 3), (3, 3), (3, 4)}), None, 2


def _specimen(rng, shape, scale, color, bg, structural, compact=False):
    sh, sw = _size(shape)
    pads = [1, 1, 1, 1] if compact else list(map(int, rng.integers(1, 3, size=4)))
    t, b, l, r = pads
    a = np.full((sh*scale+t+b+2, sw*scale+l+r+2), bg, dtype=np.int8)
    a[0, :] = a[-1, :] = structural
    a[:, 0] = a[:, -1] = structural
    for rr, cc in shape:
        a[1+t+rr*scale:1+t+(rr+1)*scale, 1+l+cc*scale:1+l+(cc+1)*scale] = color
    return a


def _free(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return []
    sums = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w]
    return np.argwhere(totals == 0)


def _pack(rng, objects, bg):
    if any(max(a.shape) > 30 for a in objects) or sum(a.size for a in objects) > 740:
        return None
    order = sorted(range(len(objects)), key=lambda i: (-objects[i].size, float(rng.random())))
    for attempt in range(36):
        h, w = (int(rng.integers(22, 31)), int(rng.integers(22, 31))) if attempt < 12 else (30, 30)
        canvas = np.full((h, w), bg, dtype=np.int8)
        blocked = np.zeros((h, w), dtype=bool)
        for i in order:
            oh, ow = objects[i].shape
            positions = _free(blocked, oh, ow)
            if len(positions) == 0:
                break
            r, c = map(int, positions[int(rng.integers(len(positions)))])
            canvas[r:r+oh, c:c+ow] = objects[i]
            blocked[max(0, r-1):min(h, r+oh+1), max(0, c-1):min(w, c+ow+1)] = True
        else:
            if np.count_nonzero(canvas == bg) <= canvas.size//2:
                continue
            # Empty outer margins can vary independently of the output canvas.
            if rng.random() < .7:
                points = np.argwhere(canvas != bg)
                lo, hi = points.min(0), points.max(0)+1
                lo = np.maximum(0, lo-rng.integers(0, 3, size=2))
                hi = np.minimum((h, w), hi+rng.integers(0, 3, size=2))
                crop = canvas[lo[0]:hi[0], lo[1]:hi[1]]
                if np.count_nonzero(crop == bg) > crop.size//2:
                    canvas = crop.copy()
            return canvas
    return None


def _make_pair(rng, scale, turn, double, palette, witness=None, large=False, small=False):
    bg, s, p, m = map(int, palette[:4])
    for attempt in range(80):
        limit = 30//scale
        if small:
            limit = min(limit, 12 if witness == "full" else 10)
        low = 11 if witness == "full" else (13 if large and scale == 2 else 7)
        h, w = map(int, rng.integers(low, limit+1, size=2))
        if rng.random() < .25:
            w = h
        if witness == "full":
            cells, ar, ac = _payload(rng, h, w, True, True)
        else:
            cells, ar, ac = _payload(rng, h, w, double, False)
        ph, pw = _size(cells)
        if ph > h-4 or pw > w-4:
            continue
        r0, c0 = int(rng.integers(2, h-ph-1)), int(rng.integers(2, w-pw-1))
        cells = {(r+r0, c+c0) for r, c in cells}
        ar, ac = (ar+r0 if ar is not None else None), ac+c0
        board = np.full((h, w), bg, dtype=np.int8)
        board[0, :] = board[-1, :] = s
        board[:, 0] = board[:, -1] = s
        for q in cells:
            board[q] = p
        board[0, ac] = m
        if double:
            board[ar, -1] = m
        board = np.rot90(board, -turn).copy()
        actual_cells = set(map(tuple, np.argwhere(board == p)))
        marks = {_side(tuple(q), board.shape): tuple(map(int, q)) for q in np.argwhere(board == m)}
        sides = set(marks)
        vertical, horizontal = sides & {"top", "bottom"}, sides & {"left", "right"}
        aar = marks[next(iter(horizontal))][0] if horizontal else None
        aac = marks[next(iter(vertical))][1] if vertical else None
        features = _features(actual_cells, aar, aac, sides)
        parts = features["parts"]
        keys = sorted(features["classes"])
        color_count = len(keys) if witness else int(rng.integers(1, len(keys)+1))
        cmap = {k: int(palette[4+i % color_count]) for i, k in enumerate(keys)}
        specimens = []
        witnessed = False
        for corner, part in sorted(parts.items()):
            dest = _norm(part)
            poses = _poses(dest)
            k = _canonical(part)
            unique = sum(_canonical(v) == k for v in parts.values()) == 1
            eligible = _retained(corner, sides) and unique
            if not witnessed and eligible:
                alternatives = (tuple(q for q in poses if q not in _poses(dest, False))
                                if witness == "full" else
                                tuple(q for q in _poses(dest, False) if q != dest))
                if witness and alternatives:
                    poses = alternatives
                    witnessed = True
            shape = poses[int(rng.integers(len(poses)))]
            specimens.append(_specimen(rng, shape, scale, cmap[k], bg, s, compact=attempt > 15))
        if witness and not witnessed:
            continue
        inp = _pack(rng, [board] + specimens, bg)
        if inp is None:
            continue
        # Construction's output uses the sampled templates directly. Recovery
        # later checks the returned raster; it does not supply generation data.
        direct = {"source": {"board": board}, "background": bg, "payload": p,
                  "marker": m, "markers": marks, "sides": sides,
                  "cells": actual_cells, "fixed": features["fixed"], "parts": parts,
                  "color_map": cmap, "scale": scale}
        out = _render(direct)
        recovered = _recover(inp)
        _require(np.array_equal(_render(recovered), out), "raster recovery disagrees with construction")
        if witness == "full" and not (recovered["reflection"] and recovered["gaps"] and recovered["fragmented"]):
            continue
        if witness == "rotation" and not recovered["rotation"]:
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded construction could not pack a valid matched stencil")


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, "num_train", 2, 10)
    ntest = _count(num_test, "num_test", 1, 4)
    _require(ntrain > ntest, "num_train must exceed num_test")
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    mode = int(rng.integers(4))
    train_scales = ([2]*ntrain if mode in (0, 1) else [2, 3] + [int(rng.choice([2, 3])) for _ in range(ntrain-2)])
    test_scales = ([3]*ntest if mode == 0 else [2]*ntest if mode == 1 else
                   [4]*ntest if mode == 2 else [int(rng.choice([2, 3])) for _ in range(ntest)])
    angles = list(map(int, rng.choice(4, 2, replace=False)))
    unseen_angles = [a for a in range(4) if a not in angles]
    palette = rng.permutation(10)
    varied_palette = bool(rng.random() < .75)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, scales in (("train", train_scales), ("test", test_scales)):
        for index, scale in enumerate(scales):
            if split == "train":
                double = index == 0 or (index > 1 and bool(rng.integers(2)))
                turn = angles[index % 2]
                witness = "full" if index == 0 else "rotation" if index == 1 else None
            else:
                # With the source's empty inner moat, scale four leaves only
                # a 3x3 payload box. Two axes would make four singleton parts.
                double = scale != 4 and bool(rng.integers(2))
                turn = int(rng.choice(unseen_angles if mode == 3 else angles))
                witness = None
            if varied_palette:
                palette = rng.permutation(10)
            for _ in range(12):
                pair = _make_pair(rng, scale, turn, double, palette, witness,
                                  large=split == "test" and mode == 1,
                                  small=split == "train" and mode == 1)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _count(nt, "train length", 2, 10)
    _count(nq, "test length", 1, 4)
    _require(nt > nq, "invalid split counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _recover(inp)
            _require(np.array_equal(_render(info), out), "incorrect output")
            k = info["scale"]
            logical = out[::k, ::k]
            _require(np.array_equal(out, np.repeat(np.repeat(logical, k, 0), k, 1)), "output is not block scaled")
            _require(info["structure"] not in out, "construction frames must disappear")
            _require(np.count_nonzero(out == info["payload"]) == len(info["fixed"])*k*k,
                     "fixed-axis payload count changed")
            expected_counts = Counter()
            for corner, cells in info["parts"].items():
                if _retained(corner, info["sides"]):
                    expected_counts[info["color_map"][_canonical(cells)]] += len(cells)*k*k
            for color in info["color_map"].values():
                _require(np.count_nonzero(out == color) == expected_counts[color], "part area or deletion mismatch")
            # Each template occupies one common-scale block pattern in a frame.
            _require(sum(len(t["cells"]) for t in info["specimens"]) ==
                     sum(len(v) for v in info["parts"].values())*k*k, "specimen/template area mismatch")
            infos[split].append(info)
    train = infos["train"]
    _require({len(i["sides"]) for i in train} == {1, 2}, "both marker arities need demonstrations")
    _require(any(i["reflection"] for i in train), "missing reflection demonstration")
    _require(any(i["rotation"] for i in train), "missing rotation demonstration")
    _require(any(i["fragmented"] and i["gaps"] for i in train), "missing collective/gap demonstration")
    _require(any(any(a != b and len(a) == len(b) and i["color_map"][a] != i["color_map"][b]
                         for a in i["color_map"] for b in i["color_map"]) for i in train),
             "missing differently colored equal-area shape demonstration")
    # Explicit parent-ablation witnesses are verified from the supplied grids.
    for mutant in ({"monochrome": True}, {"no_dispatch": True}, {"nearest": True}, {"separate": True}):
        _require(any(not np.array_equal(_render(i, **mutant), _render(i)) for i in train),
                 "demonstrations fail to distinguish a simpler rule")
    return True


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded 42-seed variation audit, counts, symmetries and negative checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "seeds do not vary puzzles")
    _require(_puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0)), "nondeterministic seed")
    entries = [(p, _recover(p["input"])) for task in puzzles for split in ("train", "test") for p in task[split]]
    infos = [i for p, i in entries]
    _require({i["scale"] for i in infos} == {2, 3, 4}, "scale coverage")
    _require({s for i in infos for s in i["sides"]} == set(_SIDES), "direction coverage")
    _require(len({tuple(i["source"]["board"].shape) for i in infos}) >= 20, "stencil size coverage")
    _require(len({_canonical(v) for i in infos for v in i["parts"].values()}) >= 50, "template vocabulary coverage")
    _require(any(i["equal_area_distinct"] for i in infos), "shape matching reduced to area")
    _require(any(len(i["color_map"]) < len(i["parts"]) for i in infos), "no repeated classes")
    _require({i["background"] for i in infos} == set(range(10)), "palette role coverage")
    # Equivariance checks exercise solver members beyond the sampled schedules.
    pair = puzzles[0]["train"][0]
    for turn in range(4):
        for flip in (False, True):
            inp = np.fliplr(pair["input"]) if flip else pair["input"]
            out = np.fliplr(pair["output"]) if flip else pair["output"]
            _require(np.array_equal(solve(np.rot90(inp, turn)), np.rot90(out, turn)), "scene symmetry failure")
    permutation = np.array([7, 3, 0, 9, 5, 2, 1, 8, 6, 4], dtype=np.int8)
    _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]), "color equivariance failure")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(100+a*10+b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "count mismatch")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    wrong = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    wrong["test"][0]["output"][0, 0] = (int(wrong["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(wrong)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    info = _recover(pair["input"])
    damaged = pair["input"].copy()
    tray = info["specimens"][0]
    r0, r1, c0, c1 = tray["bounds"]
    rr, cc = min(tray["cells"])
    damaged[r0+rr, c0+cc] = info["background"]
    try:
        solve(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete macrocell accepted")
    repeated_pair, repeated = next((p, i) for p, i in entries
                                   if len(i["color_map"]) < len(i["parts"]))
    # Recover the raster for this repeated class, then give one interchangeable
    # specimen a fresh color. Shape matching must report ambiguity, not sort it.
    counts = Counter(_canonical(v) for v in repeated["parts"].values())
    tray = next(t for t, shape in zip(repeated["specimens"], repeated["decoded"])
                if counts[_canonical(shape)] > 1)
    recolored = repeated_pair["input"].copy()
    free_color = next(c for c in range(10) if c not in set(map(int, np.unique(recolored))))
    r0, r1, c0, c1 = tray["bounds"]
    for rr, cc in tray["cells"]:
        recolored[r0+rr, c0+cc] = free_color
    try:
        solve(recolored)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous interchangeable specimen colors accepted")
    return {"seeds": 42, "pairs": len(infos), "scales": sorted({i["scale"] for i in infos}),
            "stencil_shapes": len({tuple(i["source"]["board"].shape) for i in infos}),
            "template_classes": len({_canonical(v) for i in infos for v in i["parts"].values()}),
            "semantic_scope": "generator-owned invariants, ablations and equivariance; no independent review"}
