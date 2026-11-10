"""Fold printing: input-carried stencils, geometric folding, and exposure parity.

Only NumPy and the standard library are used. The solver reads the font and all
live addresses from the query itself; generation records never enter its API.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np


BACKGROUND = 0
GUIDE = 9
HOST_SEEDS = (0, 455589553, 485494166, 853572605, 79578269, 68237373,
              870641689, 50228611, 992258743, 262695899, 737088830,
              313693202, 294860504, 401824065, 950466795, 127149027,
              916105668, 768725758, 415530533, 859167077, 495646084,
              380961644, 732714885, 246027311, 282413086, 699277172,
              216158291, 646779129, 150914068, 318284237, 45695098,
              35447241, 355568424, 73780101, 111699030, 503996878,
              297990774, 425681068, 363244674, 843402438, 280655004,
              145888768)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grids must be two-dimensional with dimensions 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and not np.issubdtype(a.dtype, np.bool_),
             "grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in todo:
                        todo.remove(q)
                        stack.append(q)
        result.append(sorted(cells))
    return result


def _bbox(cells):
    return (min(r for r, _ in cells), max(r for r, _ in cells),
            min(c for _, c in cells), max(c for _, c in cells))


def _read_guide(a):
    cells = list(map(tuple, np.argwhere(a == GUIDE).tolist()))
    _require(bool(cells), "missing fold guide")
    r0, r1, c0, c1 = _bbox(cells)
    if len(cells) == 1:
        return "P", r0, c0
    if r0 == r1 and len(cells) == c1 - c0 + 1:
        return "H", r0, c0
    if c0 == c1 and len(cells) == r1 - r0 + 1:
        return "V", r0, c0
    raise ValueError("one contiguous line or one singleton must be the entire guide")


def _transform(q, guide):
    kind, ar, ac = guide
    r, c = q
    return ((2 * ar - r if kind in ("H", "P") else r),
            (2 * ac - c if kind in ("V", "P") else c))


def _same_entry(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1])


def _read_input(a):
    """Recover roles from top-left-keyed specimens, then classify all objects."""
    _require(np.count_nonzero(a == BACKGROUND) > a.size // 2,
             "black must be the strict majority input background")
    guide = _read_guide(a)
    objects = _components((a != BACKGROUND) & (a != GUIDE))
    book = {}
    for cells in objects:
        counts = Counter(int(a[q]) for q in cells)
        r0, r1, c0, c1 = _bbox(cells)
        origin = (r0, c0)
        if len(counts) != 2 or origin not in cells:
            continue
        code = int(a[origin])
        if counts[code] != 1:
            continue
        ink = next(v for v in counts if v != code)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            if int(a[r, c]) == ink:
                mask[r - r0, c - c0] = True
        _require(all(3 <= n <= 5 for n in mask.shape), "specimen size must be 3..5 per axis")
        _require(not mask[0, 0] and len(_components(mask)) == 1,
                 "specimen ink must be connected and leave the key corner blank")
        _require(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any(),
                 "specimen ink must have tight bounds")
        entry = (ink, mask)
        _require(code not in book or _same_entry(book[code], entry),
                 "inconsistent specimens for one code")
        book[code] = entry
    _require(1 <= len(book) <= 3, "each input must visibly define its 1..3 active codes")
    inks = {ink for ink, _ in book.values()}
    _require(len(inks) == len(book) and not inks.intersection(book),
             "code colors and distinct ink colors must be disjoint")
    _require(set(map(int, np.unique(a))) <= {BACKGROUND, GUIDE} | set(book) | inks,
             "undefined color role")
    live, queries, specimens, discarded = [], [], [], []
    for cells in objects:
        points = [q for q in cells if int(a[q]) in book]
        r0, _, c0, _ = _bbox(cells)
        if len(points) == 1 and points[0] == (r0, c0):
            q = points[0]
            record = (q[0], q[1], int(a[q]))
            live.append(record)
            (queries if len(cells) == 1 else specimens).append(record)
        else:
            discarded.append(cells)
    _require(bool(live) and {c for _, _, c in specimens} == set(book),
             "every active code needs a full keyed specimen")
    return guide, book, live, queries, discarded


def _exposures(record, book, guide):
    r, c, code = record
    for dr, dc in np.argwhere(book[code][1]):
        q = (r + int(dr), c + int(dc))
        yield q
        yield _transform(q, guide)


def _fields(shape, book, records, guide, clip_before_fold=False, anchor_fold=False,
            unturned=False):
    """Push each ink exposure, retaining its multiplicity, into the paper."""
    h, w = shape
    fields = {code: Counter() for code in book}
    for r, c, code in records:
        for dr, dc in np.argwhere(book[code][1]):
            q = (r + int(dr), c + int(dc))
            if clip_before_fold and not (0 <= q[0] < h and 0 <= q[1] < w):
                continue
            if anchor_fold:
                ar, ac = _transform((r, c), guide)
                reflected = (ar + int(dr), ac + int(dc))
            elif unturned:
                kind, ar, ac = guide
                mh, mw = book[code][1].shape
                top = 2 * ar - (r + mh - 1) if kind in ("H", "P") else r
                left = 2 * ac - (c + mw - 1) if kind in ("V", "P") else c
                reflected = (top + int(dr), left + int(dc))
            else:
                reflected = _transform(q, guide)
            for x, y in (q, reflected):
                if 0 <= x < h and 0 <= y < w:
                    fields[code][x, y] += 1
    return fields


def _render(shape, book, records, guide, mode="parity", check_conflicts=True, **options):
    fields = _fields(shape, book, records, guide, **options)
    out = np.zeros(shape, dtype=np.int8)
    occupied = set()
    for code, counts in fields.items():
        if check_conflicts:
            _require(not occupied.intersection(counts), "different inks' exposure supports overlap")
        occupied.update(counts)
        for q, n in counts.items():
            keep = n % 2 == 1 if mode == "parity" else (n > 0 if mode == "union" else n == 1)
            if keep:
                out[q] = book[code][0]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode this input's font, print full stencils, fold, XOR, then crop."""
    a = _grid(input_grid)
    guide, book, live, _, _ = _read_input(a)
    return _render(a.shape, book, live, guide)


def _reference_render(shape, book, records, guide):
    """Pull exposures at q and T(q), using dense array indexing, for self-tests.

    This does not call the push renderer or enumerate transformed stencil cells.
    The original mask can be queried outside the paper before the final crop.
    """
    rr, cc = np.indices(shape)
    kind, ar, ac = guide
    tr = 2 * ar - rr if kind in ("H", "P") else rr
    tc = 2 * ac - cc if kind in ("V", "P") else cc
    out = np.zeros(shape, dtype=np.int8)
    for code, (ink, mask) in book.items():
        parity = np.zeros(shape, dtype=bool)
        for r, c, other in records:
            if other != code:
                continue
            for rows, cols in ((rr, cc), (tr, tc)):
                dr, dc = rows - r, cols - c
                valid = (dr >= 0) & (dc >= 0) & (dr < mask.shape[0]) & (dc < mask.shape[1])
                hit = np.zeros(shape, dtype=bool)
                hit[valid] = mask[dr[valid], dc[valid]]
                parity ^= hit
        _require(not np.any(parity & (out != 0)), "different inks collide")
        out[parity] = ink
    return out


def _mask(rng, focus=False):
    h, w = (int(rng.choice((3, 5))), int(rng.choice((3, 5)))) if focus else (
        int(rng.integers(3, 6)), int(rng.integers(3, 6)))
    mask = np.zeros((h, w), dtype=bool)
    family = 0 if focus else int(rng.integers(3))
    if family == 0:
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        mask[1:-1, 1:-1] = rng.random((h - 2, w - 2)) < rng.uniform(0, .85)
    elif family == 1:
        mask[:] = True
        mask[1:-1, 1:-1] = rng.random((h - 2, w - 2)) < rng.uniform(.15, .8)
        for r, c in ((0, w // 2), (h // 2, 0), (h // 2, w - 1), (h - 1, w // 2)):
            if rng.random() < .45:
                mask[r, c] = False
    else:
        # Variable thin routes join three corners and two neighbors of the key.
        points = [(0, 1), (0, w - 1), (h - 1, 0), (h - 1, w - 1), (1, 0)]
        for start, finish in zip(points, points[1:]):
            r, c = start
            mask[r, c] = True
            while (r, c) != finish:
                dr, dc = int(np.sign(finish[0] - r)), int(np.sign(finish[1] - c))
                choices = [(r + dr, c + dc)]
                if dr:
                    choices.append((r + dr, c))
                if dc:
                    choices.append((r, c + dc))
                choices = [q for q in choices if q != (0, 0)]
                r, c = choices[int(rng.integers(len(choices)))]
                mask[r, c] = True
    mask[0, 0] = False
    # A cutout can disconnect a small dense stencil: repair via a connected frame.
    if len(_components(mask)) != 1:
        mask[0, 1:] = mask[-1, :] = True
        mask[1:, 0] = mask[:, -1] = True
    return mask


def _halo(cells, shape):
    h, w = shape
    return {(r + dr, c + dc) for r, c in cells for dr in (-1, 0, 1)
            for dc in (-1, 0, 1) if 0 <= r + dr < h and 0 <= c + dc < w}


def _make_pair(rng, full_book, active, focus, kind, query_count, edge_all,
               artifact_kinds, positive_axis):
    """Free placement; reserve ink supports by color, not rectangular slots."""
    book = {c: full_book[c] for c in active}
    for attempt in range(32):
        lower = 14 if len(active) == 1 and not artifact_kinds else 21
        h, w = (30, 30) if attempt >= 12 else (
            int(rng.integers(lower, 31)), int(rng.integers(lower, 31)))
        offsets = (2, 2) if positive_axis else (int(rng.integers(-2, 3)), int(rng.integers(-2, 3)))
        ar, ac = h // 2 + offsets[0], w // 2 + offsets[1]
        guide = (kind, ar, ac)
        a = np.zeros((h, w), dtype=np.int8)
        if kind == "P":
            a[ar, ac] = GUIDE
        records, query_records, input_cells = [], [], set()
        supports = {c: set() for c in active}

        def place(r, c, code, specimen=False):
            ink, mask = book[code]
            material = {(r, c): code}
            if specimen:
                if r + mask.shape[0] > h or c + mask.shape[1] > w:
                    return False
                material.update({(r + int(x), c + int(y)): ink for x, y in np.argwhere(mask)})
            if not (0 <= r < h and 0 <= c < w):
                return False
            if any(a[q] != BACKGROUND for q in material):
                return False
            if _halo(material, a.shape).intersection(input_cells):
                return False
            record = (r, c, code)
            footprint = {q for q in _exposures(record, book, guide)
                         if 0 <= q[0] < h and 0 <= q[1] < w}
            if any(footprint.intersection(supports[other]) for other in active if other != code):
                return False
            for q, value in material.items():
                a[q] = value
            records.append(record)
            if not specimen:
                query_records.append(record)
            supports[code].update(footprint)
            input_cells.update(material)
            return True

        # Seed a near-fold overlap candidate. Acceptance below requires double
        # and triple exposures in the complete scene; stencil cells, rather
        # than input points, interact across the global fold.
        mh, mw = book[focus][1].shape
        canonical = attempt >= 12
        if kind == "H":
            r = ar - mh // 2 + (int(rng.integers(-1, 2)) if mh == 5 and not canonical else 0)
            c = int(rng.integers(0, w - 2 * mw + 3))
            cluster = ((r, c), (r + (0 if canonical else int(rng.integers(-1, 2))),
                                    c + (mw - 1 if canonical else int(rng.choice((mw - 2, mw - 1, mw))))))
        elif kind == "V":
            r = int(rng.integers(0, h - 2 * mh + 3))
            c = ac - mw // 2 + (int(rng.integers(-1, 2)) if mw == 5 and not canonical else 0)
            cluster = ((r, c), (r + (mh - 1 if canonical else int(rng.choice((mh - 2, mh - 1, mh)))),
                                    c + (0 if canonical else int(rng.integers(-1, 2)))))
        else:
            r = ar - mh // 2 + (int(rng.integers(-1, 2)) if mh == 5 and not canonical else 0)
            c = ac - mw // 2 + (int(rng.integers(-1, 2)) if mw == 5 and not canonical else 0)
            dr, dc = (0, mw - 1) if canonical else (
                ((0, mw - 1), (mh - 1, 0), (0, 1 - mw), (1 - mh, 0))[int(rng.integers(4))])
            cluster = ((r, c), (r + dr, c + dc))
        if not all(place(r, c, focus) for r, c in cluster):
            continue
        failed = False
        # Every code is visibly defined, and its defining sample is also live.
        for code in active:
            mh2, mw2 = book[code][1].shape
            locations = [(r, c) for r in range(h - mh2 + 1) for c in range(w - mw2 + 1)]
            rng.shuffle(locations)
            if not any(place(r, c, code, specimen=True) for r, c in locations):
                failed = True
                break
        if failed:
            continue
        boundary_modes = ["bottom", "right", "corner"] if edge_all else [str(rng.choice(("bottom", "right", "corner")))]
        for mode in boundary_modes:
            locations = ([(h - 2, c) for c in range(w - mw + 1)] if mode == "bottom" else
                         [(r, w - 2) for r in range(h - mh + 1)] if mode == "right" else
                         [(h - 2, w - 2)])
            rng.shuffle(locations)
            if not any(place(r, c, focus) for r, c in locations):
                failed = True
                break
        if failed:
            continue
        locations = [(r, c) for r in range(h) for c in range(w)]
        rng.shuffle(locations)
        for r, c in locations:
            if len(query_records) >= query_count:
                break
            place(r, c, int(rng.choice(active)))
        if len(query_records) != query_count:
            continue
        if kind != "P":
            length = int(rng.integers(2, 8))
            starts = list(range((w if kind == "H" else h) - length + 1))
            rng.shuffle(starts)
            for start in starts:
                cells = [(ar, start + j) if kind == "H" else (start + j, ac) for j in range(length)]
                if all(a[q] == BACKGROUND for q in cells):
                    for q in cells:
                        a[q] = GUIDE
                    break
            else:
                continue
        out = _render(a.shape, book, records, guide)
        printed_support = set().union(*supports.values())
        for artifact in artifact_kinds:
            code = int(rng.choice(active))
            ink, mask = book[code]
            hh, ww = mask.shape
            # Attached keys are interior with a complete prospective stamp;
            # their erasure cannot be explained by a boundary-only shortcut.
            locations = [(r, c) for r in range(h - (2 * hh if artifact == "attached" else hh) + 1)
                         for c in range(w - (2 * ww if artifact == "attached" else ww) + 1)]
            rng.shuffle(locations)
            for r, c in locations:
                material = {(r + int(x), c + int(y)): ink for x, y in np.argwhere(mask)}
                if artifact == "attached":
                    last_c = int(np.flatnonzero(mask[-1])[-1])
                    material[r + hh, c + last_c + 1] = code
                if any(a[q] != BACKGROUND for q in material):
                    continue
                if _halo(material, a.shape).intersection(input_cells) or printed_support.intersection(material):
                    continue
                for q, value in material.items():
                    a[q] = value
                input_cells.update(material)
                break
            else:
                failed = True
                break
        if failed or np.count_nonzero(a == 0) <= a.size // 2:
            continue
        # Reconstruct from input, and require observable witnesses, before acceptance.
        try:
            observed_guide, observed_book, observed_live, _, _ = _read_input(a)
            if set(observed_live) != set(records) or set(observed_book) != set(book):
                continue
            _require(all(_same_entry(book[c], observed_book[c]) for c in book), "font mismatch")
            counts = _fields(a.shape, book, records, observed_guide)
            values = [n for field in counts.values() for n in field.values()]
            if not (2 in values and 3 in values):
                continue
            if np.array_equal(out, _render(a.shape, book, records, observed_guide,
                                          anchor_fold=True, check_conflicts=False)):
                continue
            if np.array_equal(out, _render(a.shape, book, records, observed_guide,
                                          unturned=True, check_conflicts=False)):
                continue
            if np.array_equal(out, _render(a.shape, book, records, observed_guide,
                                          clip_before_fold=True)) and positive_axis:
                continue
            if np.array_equal(out, _render(a.shape, book, [x for x in records if x not in query_records], observed_guide)):
                continue
        except ValueError:
            continue
        return {"input": a, "output": out}
    raise RuntimeError("bounded placement exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    _require(nt > nq, "num_train must be greater than num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    vocabulary = int(rng.integers(1, 4))
    palette = list(map(int, rng.permutation(np.arange(1, 9))))
    codes = palette[:vocabulary]
    book = {code: (palette[vocabulary + i], _mask(rng, focus=i == 0)) for i, code in enumerate(codes)}
    kinds = list(map(str, rng.choice(["H", "V", "P"], int(rng.integers(1, min(3, nt) + 1)), replace=False)))
    schedule = kinds + [str(rng.choice(kinds)) for _ in range(nt - len(kinds))]
    rng.shuffle(schedule)
    rich = bool(rng.random() < .55)
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(nt + nq):
        training = i < nt
        kind = schedule[i] if training else str(rng.choice(kinds))
        if training and i < 2:
            active = list(codes)
        else:
            active = [codes[0]] + [c for c in codes[1:] if rng.random() < .6]
        query_count = 5 if i == 0 else (int(rng.integers(3, 6)) if training else int(rng.integers(6, 10)))
        artifacts = ["bare", "attached"] if rich and i == 0 else (
            [str(rng.choice(("bare", "attached")))] if rich and rng.random() < .4 else [])
        for _ in range(12):
            pair = _make_pair(rng, book, active, codes[0], kind, query_count,
                              i == 0, artifacts, i == 0)
            signature = (pair["input"].shape, pair["input"].tobytes())
            if signature not in seen:
                break
        else:
            raise RuntimeError("could not make distinct inputs")
        seen.add(signature)
        puzzle["train" if training else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_material(a, book, discarded):
    ink_to_code = {ink: code for code, (ink, _) in book.items()}
    features = set()
    for cells in discarded:
        ink_values = {int(a[q]) for q in cells if int(a[q]) in ink_to_code}
        _require(len(ink_values) == 1, "discarded material must have one known ink")
        ink = ink_values.pop()
        code = ink_to_code[ink]
        mask = book[code][1]
        r, _, c, _ = _bbox(cells)
        expected = {(r + int(x), c + int(y)): ink for x, y in np.argwhere(mask)}
        points = [q for q in cells if int(a[q]) in book]
        if points:
            _require(len(points) == 1 and int(a[points[0]]) == code, "invalid attached key")
            last_c = int(np.flatnonzero(mask[-1])[-1])
            q = (r + mask.shape[0], c + last_c + 1)
            _require(points[0] == q, "attachment must be diagonally beyond last bottom-row ink")
            expected[q] = code
            features.add("attached")
        else:
            features.add("bare")
        _require({q: int(a[q]) for q in cells} == expected, "unexplained material")
    return features


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _count(len(puzzle["train"]), "train count", 2, 10)
    _count(len(puzzle["test"]), "test count", 1, 4)
    _require(len(puzzle["train"]) > len(puzzle["test"]), "counts out of order")
    seen, shared_book, taught_kinds, taught_features = set(), {}, set(), set()
    orientation_kinds, membership_features = set(), set()
    training_query_counts = []
    coverage = {"double": False, "triple": False, "orientation": False,
                "addresses": False, "crop_order": False, "no_A": False, "no_B": False}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == out.shape, "output must retain paper dimensions")
            signature = (a.shape, a.tobytes())
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            guide, book, live, queries, discarded = _read_input(a)
            features = _check_material(a, book, discarded)
            for c, entry in book.items():
                if split == "test":
                    _require(c in shared_book, "untaught code")
                _require(c not in shared_book or _same_entry(shared_book[c], entry), "font changed across examples")
                shared_book[c] = entry
            expected = _render(a.shape, book, live, guide)
            _require(np.array_equal(expected, out), "output disagrees with input-derived exposure parity")
            _require(not np.array_equal(a, out), "identity pair")
            _require(set(map(int, np.unique(out))) <= {0} | {ink for ink, _ in book.values()},
                     "guides, keys, or unknown colors survived")
            if split == "train":
                taught_kinds.add(guide[0])
                taught_features.update(features)
                training_query_counts.append(len(queries))
                values = [n for counts in _fields(a.shape, book, live, guide).values() for n in counts.values()]
                coverage["double"] |= 2 in values
                coverage["triple"] |= 3 in values
                coverage["orientation"] |= not np.array_equal(out, _render(a.shape, book, live, guide,
                                                                         anchor_fold=True, check_conflicts=False))
                if not np.array_equal(out, _render(a.shape, book, live, guide,
                                                  unturned=True, check_conflicts=False)):
                    orientation_kinds.add(guide[0])
                if "bare" in features and any(np.all(out[tuple(np.array(cells).T)] == 0)
                                               for cells in discarded if not any(int(a[q]) in book for q in cells)):
                    membership_features.add("bare")
                if "attached" in features:
                    all_points = [(int(r), int(c), int(a[r, c]))
                                  for r, c in np.argwhere(np.isin(a, list(book)))]
                    if not np.array_equal(out, _render(a.shape, book, all_points, guide, check_conflicts=False)):
                        membership_features.add("attached")
                coverage["crop_order"] |= not np.array_equal(out, _render(a.shape, book, live, guide, clip_before_fold=True))
                coverage["addresses"] |= not np.array_equal(out, _render(a.shape, book, [x for x in live if x not in queries], guide))
                # No-A ablation is ordinary B stamping with same-code XOR only.
                unfolded = np.zeros(a.shape, dtype=np.int8)
                for code, (ink, mask) in book.items():
                    field = np.zeros(a.shape, dtype=bool)
                    for r, c, value in live:
                        if value == code:
                            hh, ww = min(mask.shape[0], a.shape[0] - r), min(mask.shape[1], a.shape[1] - c)
                            field[r:r + hh, c:c + ww] ^= mask[:hh, :ww]
                    unfolded[field] = ink
                coverage["no_A"] |= not np.array_equal(unfolded, out)
                # No-B ablation folds point addresses, without any stencil ink.
                point_fold = np.zeros(a.shape, dtype=np.int8)
                counts = {code: Counter() for code in book}
                for r, c, code in live:
                    for q in ((r, c), _transform((r, c), guide)):
                        if 0 <= q[0] < a.shape[0] and 0 <= q[1] < a.shape[1]:
                            counts[code][q] += 1
                for code, field in counts.items():
                    for q, n in field.items():
                        if n % 2:
                            point_fold[q] = code
                coverage["no_B"] |= not np.array_equal(point_fold, out)
            else:
                _require(guide[0] in taught_kinds and features <= taught_features, "untaught test feature")
                _require(len(queries) > max(training_query_counts), "missing query-density shift")
    _require(all(coverage.values()), f"training lacks discriminating witnesses: {coverage}")
    _require(orientation_kinds == taught_kinds, "each taught symmetry needs an unturned-copy counterexample")
    _require(taught_features <= membership_features, "material erasure lacks an observable witness")


def self_test():
    """Bounded 42-seed audit with semantic ablations and label mutations."""
    shapes, masks, kinds, palettes, query_counts, guide_lengths = set(), set(), set(), set(), set(), set()
    artifact_features, max_exposure, input_signatures = set(), 0, set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                guide, book, records, queries, discarded = _read_input(a)
                _require(np.array_equal(_reference_render(a.shape, book, records, guide), pair["output"]),
                         "dense pull check failed")
                shapes.add(a.shape)
                kinds.add(guide[0])
                palettes.add(tuple(sorted((c, ink) for c, (ink, _) in book.items())))
                query_counts.add(len(queries))
                guide_lengths.add(int(np.count_nonzero(a == GUIDE)))
                masks.update((mask.shape, mask.tobytes()) for _, mask in book.values())
                artifact_features.update(_check_material(a, book, discarded))
                input_signatures.add((a.shape, a.tobytes()))
                max_exposure = max(max_exposure, max(n for f in _fields(a.shape, book, records, guide).values() for n in f.values()))
        if seed == 0:
            again = generate_puzzle(seed)
            _require(all(np.array_equal(puzzle[s][i][k], again[s][i][k]) for s in puzzle
                         for i in range(len(puzzle[s])) for k in ("input", "output")), "nondeterminism")
            # An entire missing test print and an additive-fold label must fail.
            for mode in ("erase", "union", "unique", "anchor", "unturned"):
                bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
                pair = bad["test"][0]
                guide, book, live, _, _ = _read_input(pair["input"])
                pair["output"] = (np.zeros_like(pair["output"]) if mode == "erase" else
                                  _render(pair["input"].shape, book, live, guide, anchor_fold=True,
                                          check_conflicts=False) if mode == "anchor" else
                                  _render(pair["input"].shape, book, live, guide, unturned=True,
                                          check_conflicts=False) if mode == "unturned" else
                                  _render(pair["input"].shape, book, live, guide, mode=mode))
                try:
                    validate_generated_puzzle(bad)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"accepted {mode} corruption")
    _require(len(shapes) >= 50 and len(masks) >= 35 and len(palettes) >= 35,
             "insufficient geometry/font/palette diversity")
    _require(kinds == {"H", "V", "P"} and artifact_features == {"bare", "attached"}, "missing mechanisms")
    _require(query_counts >= set(range(3, 10)) and len(guide_lengths) >= 6 and max_exposure >= 4,
             "insufficient counts or overlap variation")
    _require(len(input_signatures) == 42 * 5, "cross-seed duplicate default input")
    # Literal coordinate regression: the reflected top-left key is NOT the
    # reflected stencil's top-left, and outside-source ink can return to paper.
    mask = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    book = {2: (6, mask)}
    records, guide = [(7, 7, 2)], ("P", 5, 5)
    expected = _reference_render((9, 9), book, records, guide)
    _require(expected[1, 1] == 6 and expected[3, 3] == 0, "full-stencil coordinate regression")
    _require(not np.array_equal(expected, _render((9, 9), book, records, guide, clip_before_fold=True)),
             "clip-order regression")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Read-only evidence exclusion: cover both originals and selected variations.
    here = Path(__file__).resolve().parent
    for parent in ("A", "B"):
        for name in ("task.json", "variation.json"):
            path = here / "parents" / parent / name
            if path.exists():
                source = json.loads(path.read_text())
                for split in ("train", "test"):
                    for pair in source[split]:
                        a = np.asarray(pair["input"], dtype=np.int8)
                        _require((a.shape, a.tobytes()) not in input_signatures, "copied parent input")
    return {"seeds": 42, "pairs": 210, "grid_shapes": len(shapes),
            "stencil_masks": len(masks), "palettes": len(palettes),
            "guide_kinds": sorted(kinds), "query_counts": sorted(query_counts),
            "max_exposures_at_one_cell": max_exposure,
            "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
