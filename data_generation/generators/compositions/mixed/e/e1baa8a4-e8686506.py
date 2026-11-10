"""Canonicalize independently stretched objects, then complete a bilateral board.

Only local arrays are retained. No examples, labels, seeds, or scene cache are
consulted by solve. Generation and validation create no files or execution traces.
See task_spec.md for the visible segmentation and reusable-template conventions.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected an integer ARC grid with sides 1..30")
    return a.astype(np.int8, copy=True)


def _starts(a, axis):
    lines = a if axis == 0 else a.T
    return [0] + [i for i in range(1, len(lines))
                  if not np.array_equal(lines[i], lines[i - 1])]


def _runs(a, axis):
    edges = _starts(a, axis) + [a.shape[axis]]
    return tuple(y - x for x, y in zip(edges, edges[1:]))


def _compress(a):
    # Whole-vector equality, independently on the two axes. Never np.unique:
    # separated repetitions carry both geometry and cell values.
    return a[np.ix_(_starts(a, 0), _starts(a, 1))].copy()


def _expand(a, rows, cols):
    return np.repeat(np.repeat(a, rows, axis=0), cols, axis=1)


def _components(mask):
    unseen = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
        result.append(tuple(sorted(cells)))
    return result


def _bbox(cells):
    rr, cc = zip(*cells)
    return min(rr), min(cc), max(rr) + 1, max(cc) + 1


def _normal(cells):
    r, c, _, _ = _bbox(cells)
    return tuple(sorted((y - r, x - c) for y, x in cells))


def _mirror(shape):
    width = max(c for _, c in shape) + 1
    return _normal([(r, width - 1 - c) for r, c in shape])


def _mask(shape):
    a = np.zeros((max(r for r, _ in shape) + 1,
                  max(c for _, c in shape) + 1), dtype=np.int8)
    for cell in shape:
        a[cell] = 1
    return a


def _signature(a, bg):
    return _normal([tuple(map(int, p)) for p in np.argwhere(a != bg)])


def _bilateral(a):
    return (np.array_equal(a, a[:, ::-1])
            or np.array_equal(a, a[::-1, :]))


def _complete(board, bg, catalog):
    """Match complete connected slots, with unlimited stencil reuse."""
    result, counts = board.copy(), Counter()
    for cells in _components(board == bg):
        shape = _normal(cells)
        if shape not in catalog:
            raise ValueError("A gap has no translation-equivalent template")
        color = catalog[shape]
        counts[shape] += 1
        for cell in cells:
            result[cell] = color
    if not counts or set(counts) != set(catalog):
        raise ValueError("All templates must describe at least one whole gap")
    return result, counts


def _analyze(input_grid):
    inp = _grid(input_grid)
    bg = int(inp[0, 0])
    if not (np.all(inp[0] == bg) and np.all(inp[-1] == bg)
            and np.all(inp[:, 0] == bg) and np.all(inp[:, -1] == bg)):
        raise ValueError("The canvas needs a uniform background border")
    objects = _components(inp != bg)
    multicolor = [s for s in objects if len({int(inp[p]) for p in s}) > 1]
    if len(multicolor) != 1:
        raise ValueError("Expected one connected multicolor board scaffold")
    top, left, bottom, right = _bbox(multicolor[0])
    raw_board = inp[top:bottom, left:right]
    if min(raw_board.shape) < 3 or any(np.any(edge == bg) for edge in
            (raw_board[0], raw_board[-1], raw_board[:, 0], raw_board[:, -1])):
        raise ValueError("The board must have a complete fixed frame")
    board = _compress(raw_board)
    if not _bilateral(board):
        raise ValueError("The canonical board must be bilateral")
    fixed_colors = set(map(int, np.unique(board))) - {bg}
    catalog, raw_catalog, template_runs = {}, {}, []
    template_roundtrips = True
    for cells in objects:
        inside = [top <= r < bottom and left <= c < right for r, c in cells]
        if all(inside):
            continue  # Includes any fixed islands enclosed by ring-shaped gaps.
        if any(inside):
            raise ValueError("A payload crosses the board boundary")
        values = {int(inp[p]) for p in cells}
        if len(values) != 1 or values & fixed_colors:
            raise ValueError("Exterior stencils must be monochrome payloads")
        color = next(iter(values))
        r, c, b, d = _bbox(cells)
        raw = inp[r:b, c:d]
        canonical = _compress(raw)
        shape = _signature(canonical, bg)
        if shape in catalog and catalog[shape] != color:
            raise ValueError("The same canonical template has two colors")
        catalog[shape] = color
        raw_catalog[_signature(raw, bg)] = color
        rr, cc = _runs(raw, 0), _runs(raw, 1)
        template_runs.append((rr, cc))
        template_roundtrips &= np.array_equal(_expand(canonical, rr, cc), raw)
    if len(set(catalog.values())) < 2:
        raise ValueError("At least two payload colors are required")
    result, counts = _complete(board, bg, catalog)
    if not _bilateral(result):
        raise ValueError("Template colors must preserve bilateral organization")
    raw_unmatched = sum(_normal(s) not in raw_catalog
                        for s in _components(raw_board == bg))
    template_unmatched = sum(s not in raw_catalog for s in counts)
    return result, {
        "background": bg, "bbox": (top, left, bottom, right),
        "raw_board": raw_board, "board": board, "catalog": catalog,
        "counts": counts, "row_runs": _runs(raw_board, 0),
        "col_runs": _runs(raw_board, 1), "template_runs": template_runs,
        "raw_unmatched": raw_unmatched, "template_unmatched": template_unmatched,
        "template_roundtrips": bool(template_roundtrips),
    }


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover local lattices, fill translated stencils, and crop the board."""
    return _analyze(input_grid)[0]


_T_UP = ((0, 0), (0, 1), (0, 2), (1, 1))
_T_DOWN = ((0, 1), (1, 0), (1, 1), (1, 2))
_SHAPES = (
    _T_UP, _T_DOWN,
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)),
    ((0, 0),),
)


def _random_shape(rng):
    for _ in range(40):
        if rng.random() < .6:
            shape = _SHAPES[int(rng.integers(len(_SHAPES)))]
        else:
            cells = {(0, 0)}
            for _ in range(int(rng.integers(2, 8))):
                neighbors = sorted({(r + dr, c + dc) for r, c in cells
                                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))} - cells)
                cells.add(neighbors[int(rng.integers(len(neighbors)))])
            shape = _normal(cells)
        a = np.rot90(_mask(shape), int(rng.integers(4)))
        if max(a.shape) <= 4 and np.array_equal(_compress(a), a):
            return _signature(a, 0)
    return _T_UP


def _layout(rng, shapes, h, w):
    used, placements = set(), []
    for family, shape in enumerate(shapes):
        sh, sw = _mask(shape).shape
        candidates = []
        for r in range(1, h - sh):
            for c in range(1, w - sw):
                base = frozenset((r + y, c + x) for y, x in shape)
                mirror = frozenset((y, w - 1 - x) for y, x in base)
                orbit = [base] if base == mirror else [base, mirror]
                if family == 0 and len(orbit) == 1:
                    continue  # Every scene must demonstrate stencil reuse.
                if len(orbit) == 2 and (base & mirror or any(
                        (y + dy, x + dx) in mirror for y, x in base
                        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))):
                    continue
                cells = set().union(*orbit)
                if any((y + dy, x + dx) in used for y, x in cells
                       for dy, dx in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))):
                    continue
                candidates.append(orbit)
        if not candidates:
            return None
        chosen = candidates[int(rng.integers(len(candidates)))]
        for cells in chosen:
            used.update(cells)
            placements.append((family, cells))
    return placements


def _sample_runs(rng, length, active=True, test=False):
    values = np.ones(length, dtype=int)
    if active:
        extra = int(rng.integers(2, 7))
        if test:
            index = int(rng.integers(length))
            values[index] = int(rng.integers(4, 6))
            extra = max(0, extra - (values[index] - 1))
        extra = min(extra, int(((5 if test else 3) - values).sum()))
        for _ in range(extra):
            choices = np.flatnonzero(values < (5 if test else 3))
            values[int(rng.choice(choices))] += 1
    return tuple(map(int, values))


def _scatter(rng, raw_board, templates, bg):
    bh, bw = raw_board.shape
    for attempt in range(24):
        ch = int(rng.integers(max(24, bh + 2), 31)) if attempt < 16 else 30
        cw = int(rng.integers(max(24, bw + 2), 31)) if attempt < 16 else 30
        r, c = int(rng.integers(1, ch - bh)), int(rng.integers(1, cw - bw))
        canvas = np.full((ch, cw), bg, dtype=np.int8)
        blocked = np.zeros((ch, cw), dtype=bool)
        canvas[r:r + bh, c:c + bw] = raw_board
        blocked[r - 1:r + bh + 1, c - 1:c + bw + 1] = True
        for tile in sorted(templates, key=lambda a: -a.size):
            th, tw = tile.shape
            candidates = [(y, x) for y in range(1, ch - th)
                          for x in range(1, cw - tw)
                          if not blocked[y:y + th, x:x + tw].any()]
            if not candidates:
                break
            y, x = candidates[int(rng.integers(len(candidates)))]
            canvas[y:y + th, x:x + tw] = tile
            blocked[y - 1:y + th + 1, x - 1:x + tw + 1] = True
        else:
            return canvas
    return None


def _make_pair(rng, introductory=False, test=False):
    for _ in range(96):
        h = int(rng.integers(7, 12 if not test else 14))
        w = int(rng.choice([9, 11, 13]))
        shapes, signatures = [], set()
        family_count = 2 if introductory else int(rng.integers(2, 4))
        if introductory:
            shapes = [_T_UP, _T_DOWN]
        else:
            for _ in range(60):
                shape = _random_shape(rng)
                if not shapes and _mirror(shape) != shape:
                    continue  # One oriented stencil must fill two mirrored slots.
                orbit = {shape, _mirror(shape)}
                if signatures & orbit:
                    continue
                shapes.append(shape)
                signatures.update(orbit)
                if len(shapes) == family_count:
                    break
            if len(shapes) != family_count:
                continue
        placements = _layout(rng, shapes, h, w)
        if placements is None:
            continue
        palette = rng.permutation(10)
        bg = int(palette[0])
        nf = int(rng.integers(2, 4))
        fixed = palette[1:1 + nf]
        payload = palette[1 + nf:1 + nf + family_count]
        board = np.empty((h, w), dtype=np.int8)
        style = int(rng.integers(3))
        for r in range(h):
            for c in range((w + 1) // 2):
                index = (r + c) % nf
                if 0 < r < h - 1 and 0 < c < w // 2:
                    if style == 1:
                        index = (r // 2) % nf
                    elif style == 2:
                        index = int(rng.integers(nf))
                board[r, c] = board[r, w - 1 - c] = fixed[index]
        output, catalog = board.copy(), {}
        for family, cells in placements:
            color = int(payload[family])
            catalog[_normal(cells)] = color
            for cell in cells:
                board[cell] = bg
                output[cell] = color
        mode = 0 if introductory or test else int(rng.integers(3))
        rows = _sample_runs(rng, h, mode != 2, test)
        cols = _sample_runs(rng, w, mode != 1, test)
        raw_board = _expand(board, rows, cols)
        templates = []
        for shape, color in sorted(catalog.items()):
            tile = np.where(_mask(shape), color, bg).astype(np.int8)
            tr = _sample_runs(rng, tile.shape[0], rng.random() < .8)
            tc = _sample_runs(rng, tile.shape[1], rng.random() < .8)
            # Stencil crops have at most four logical lines per axis. Keep
            # their independent stretching small enough to pack on the canvas.
            tr = tuple(min(2, v) for v in tr)
            tc = tuple(min(2, v) for v in tc)
            templates.append(_expand(tile, tr, tc))
        canvas = _scatter(rng, raw_board, templates, bg)
        if canvas is None:
            continue
        rotation = int(rng.integers(4))
        canvas = np.rot90(canvas, rotation).copy()
        output = np.rot90(output, rotation).copy()
        predicted, info = _analyze(canvas)
        if (info["raw_unmatched"] and info["template_unmatched"]
                and np.array_equal(predicted, output)):
            return {"input": canvas, "output": output}
    raise RuntimeError("Bounded construction attempts exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            for _ in range(12):
                pair = _make_pair(rng, split == "train" and i == 0, split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Unable to construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "Invalid pair counts")
    seen, both_axes, repeated, oriented = set(), False, False, False
    train_max = 0
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "Duplicate input")
            seen.add(_key(inp))
            answer, a = _analyze(inp)
            _require(np.array_equal(answer, out), "Incorrect template completion")
            board, raw = a["board"], a["raw_board"]
            rr, cc = a["row_runs"], a["col_runs"]
            _require(np.array_equal(_expand(board, rr, cc), raw), "Lost a lattice crossing")
            _require(a["template_roundtrips"], "Lost a template lattice crossing")
            _require(np.array_equal(_compress(board), board), "Unobservable logical boundary")
            _require(raw.shape != board.shape and a["raw_unmatched"] > 0, "Compression has no geometric role")
            _require(a["template_unmatched"] > 0, "Template canonicalization is not demonstrated")
            fixed = board != a["background"]
            _require(np.array_equal(board[fixed], out[fixed]), "Changed a fixed cell")
            _require(not np.any(out == a["background"]), "Unfilled board")
            _require(any(n > 1 for n in a["counts"].values()), "No reusable stencil")
            for shape in a["catalog"]:
                tile = _mask(shape)
                _require(np.array_equal(_compress(tile), tile), "Stencil is not canonical")
            if split == "train":
                train_max = max(train_max, max(rr), max(cc))
                both_axes |= (max(rr) > 1 and max(cc) > 1
                              and len(set(rr)) > 1 and len(set(cc)) > 1)
                repeated |= any(n > 1 for n in a["counts"].values())
                for x, cx in a["catalog"].items():
                    for y, cy in a["catalog"].items():
                        oriented |= cx != cy and len(x) == len(y) and _mask(x).shape == _mask(y).shape
            else:
                _require(max(max(rr), max(cc)) >= 4, "Test lacks the band-width shift")
    _require(train_max <= 3 and both_axes and repeated and oriented,
             "Training must expose compression, reuse, and shape-sensitive colors")


def self_test():
    """Bounded streaming tests over 42 seeds; retain only small signatures."""
    shapes, canvases, stencils, palettes, layouts, puzzle_hashes = (set() for _ in range(6))
    ring, separated, one_axis, raw_mismatch, pair_count = 0, 0, 0, 0, 0
    template_mismatch, global_mismatch = 0, 0
    backgrounds, fixed_counts, payload_counts, gap_counts = (set() for _ in range(4))
    row_schedules, col_schedules, symmetries = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                digest.update(inp.tobytes())
                _, a = _analyze(inp)
                b = a["board"]
                backgrounds.add(a["background"])
                fixed_counts.add(len(set(map(int, b.ravel())) - {a["background"]}))
                payload_counts.add(len(set(a["catalog"].values())))
                gap_counts.add(sum(a["counts"].values()))
                lr, tb = np.array_equal(b, b[:, ::-1]), np.array_equal(b, b[::-1])
                symmetries.add("both" if lr and tb else "lr" if lr else "tb")
                shapes.add(out.shape)
                canvases.add(inp.shape)
                stencils.update(a["catalog"])
                palettes.add(tuple(map(int, np.unique(inp))))
                normalized, roles = [], {}
                for value in b.ravel():
                    normalized.append(roles.setdefault(int(value), len(roles)))
                layouts.add((b.shape, bytes(normalized)))
                ring += len(_components(b != a["background"])) > 1
                separated += any(np.array_equal(b[i], b[j]) for i in range(len(b))
                                 for j in range(i + 2, len(b)))
                rr, cc = a["row_runs"], a["col_runs"]
                row_schedules.add(rr)
                col_schedules.add(cc)
                one_axis += (max(rr) == 1) != (max(cc) == 1)
                raw_mismatch += a["raw_unmatched"] > 0
                template_mismatch += a["template_unmatched"] > 0
                top, left, bottom, right = a["bbox"]
                global_shape = (sum(top <= r < bottom for r in _starts(inp, 0)),
                                sum(left <= c < right for c in _starts(inp, 1)))
                global_mismatch += global_shape != b.shape
                # A second expression checks exact full-line equality without
                # relying on the run-start helper used by the actual solver.
                keep_r = np.r_[True, np.any(a["raw_board"][1:] != a["raw_board"][:-1], axis=1)]
                keep_c = np.r_[True, np.any(a["raw_board"][:, 1:] != a["raw_board"][:, :-1], axis=0)]
                _require(np.array_equal(a["raw_board"][keep_r][:, keep_c], b),
                         "Whole-vector canonicalization disagrees")
                # Ablation B: compression and cropping alone leaves gaps.
                _require(not np.array_equal(b, out), "B ablation has no effect")
                # Ablation A: physical board dimensions and some raw shapes differ.
                _require(a["raw_board"].shape != out.shape and a["raw_unmatched"], "A ablation has no effect")
                pair_count += 1
        puzzle_hashes.add(digest.digest())
    _require(len(puzzle_hashes) == 42 and len(shapes) >= 25 and len(canvases) >= 20,
             "Insufficient size or seed diversity")
    _require(len(stencils) >= 35 and len(layouts) >= 160 and len(palettes) >= 30,
             "Insufficient structural or palette variation")
    _require(ring > 0 and separated > 0 and one_axis > 0 and raw_mismatch == pair_count,
             "Missing islands, separated repeats, or independent axis stretching")
    _require(template_mismatch == pair_count and global_mismatch > 0,
             "Independent local lattices are not answer-relevant")
    _require(backgrounds == set(range(10)) and len(fixed_counts) >= 2
             and len(payload_counts) >= 2 and len(gap_counts) >= 3,
             "Semantic roles or component counts do not vary")
    _require(len(row_schedules) >= 40 and len(col_schedules) >= 40
             and {"lr", "tb"} <= symmetries, "Missing lattice or orientation diversity")
    # Explicit operator probe: only adjacent duplicates disappear, on both axes.
    probe = np.array([[1, 2, 1], [3, 3, 3], [1, 2, 1]], dtype=np.int8)
    expanded = _expand(probe, (2, 1, 3), (1, 3, 2))
    _require(np.array_equal(_compress(expanded), probe), "Separated repeats were erased")
    # Same-size/area opposite T stencils must retain orientation and color.
    slots = np.full((7, 9), 8, dtype=np.int8)
    for dy, dx in _T_UP:
        slots[1 + dy, 1 + dx] = 0
    for dy, dx in _T_DOWN:
        slots[4 + dy, 5 + dx] = 0
    catalog = {_T_UP: 2, _T_DOWN: 3}
    answer, _ = _complete(slots, 0, catalog)
    _require(answer[1, 1] == 2 and answer[4, 6] == 3, "Orientation was ignored")
    try:
        _complete(slots, 0, {_T_UP: 2})
    except ValueError:
        pass
    else:
        raise AssertionError("A reflected or rotated stencil was accepted")
    # Changing a stencil's supplied color must change exactly its gap family.
    altered, _ = _complete(slots, 0, {_T_UP: 7, _T_DOWN: 3})
    _require(np.array_equal(answer != altered, answer == 2),
             "Payload colors are not causally used")
    # Corrupt an answer, including a fixed frame cell, and demand rejection.
    sample = generate_puzzle(42, 2, 1)
    sample["train"][0]["output"][0, 0] = _analyze(sample["train"][0]["input"])[1]["background"]
    try:
        validate_generated_puzzle(sample)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Validator accepted corrupted fixed cells")
    a, b = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
                 for k in ("input", "output")), "Nondeterministic generation")
    # Re-encode the same visible objects on different independent lattices.
    # This tests answer invariance under changes to band widths and scattering.
    original = a["train"][0]
    _, info = _analyze(original["input"])
    board, bg = info["board"], info["background"]
    rows = tuple(1 + i % 2 for i in range(board.shape[0]))
    cols = tuple(1 + int(i % 3 == 0) for i in range(board.shape[1]))
    tiles = [np.repeat(np.where(_mask(s), color, bg), 2, axis=1).astype(np.int8)
             for s, color in sorted(info["catalog"].items())]
    restretched = _scatter(np.random.default_rng(314159), _expand(board, rows, cols), tiles, bg)
    _require(restretched is not None and np.array_equal(solve(restretched), original["output"]),
             "Independent object stretching changed the answer")
    for rotation in range(4):
        _require(np.array_equal(solve(np.rot90(original["input"], rotation)),
                                np.rot90(original["output"], rotation)),
                 "Whole-scene rotation changed the rule")
    palette = np.roll(np.arange(10, dtype=np.int8), 3)
    _require(np.array_equal(solve(palette[original["input"]]), palette[original["output"]]),
             "Literal colors changed the rule")
    for nt, ne in ((2, 1), (10, 4)):
        p = generate_puzzle(1000 + nt, nt, ne)
        _require(len(p["train"]) == nt and len(p["test"]) == ne, "Ignored explicit counts")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted illegal counts")
    return {"seeds": 42, "pairs": pair_count, "output_shapes": len(shapes),
            "canvas_shapes": len(canvases), "oriented_stencil_shapes": len(stencils),
            "canonical_layouts": len(layouts), "palettes": len(palettes),
            "fixed_island_pairs": ring, "separated_row_repeat_pairs": separated,
            "one_axis_expansion_pairs": one_axis, "raw_shape_mismatch_pairs": raw_mismatch,
            "uncompressed_template_mismatch_pairs": template_mismatch,
            "global_compression_wrong_size_pairs": global_mismatch,
            "backgrounds": sorted(backgrounds), "fixed_color_counts": sorted(fixed_counts),
            "payload_color_counts": sorted(payload_counts), "gap_counts": sorted(gap_counts),
            "row_schedules": len(row_schedules), "column_schedules": len(col_schedules),
            "symmetries": sorted(symmetries)}
