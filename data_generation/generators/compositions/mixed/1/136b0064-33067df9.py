"""Anchored glyph scanlines -> maximal runs -> exact-span rectangle stacks.

All decoding conventions are family constants. No generation state is used by
solve. The three glyphs retain A's original horizontal stroke meanings; the
tile's coarse column replaces A's single path endpoint. See task_spec.md.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import product

import numpy as np


# R3, L2, L4: exactly the horizontal masks/meanings in parent A's source.
_MASKS = (
    ((1, 1, 0), (1, 0, 1), (0, 1, 0)),
    ((1, 0, 1), (1, 0, 1), (1, 1, 1)),
    ((1, 1, 1), (0, 1, 0), (1, 0, 1)),
)
_MASK_LOOKUP = {tuple(v for row in m for v in row): k for k, m in enumerate(_MASKS)}
_STROKES = ((1, 3), (-1, 2), (-1, 4))  # direction, length
_ROW_AXES = (1, 2, 3, 4, 6)
_COL_AXES = (2, 3, 4, 6)
_SIZE, _INTERIOR, _GAP = 26, 24, 2


def _integer(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _span(anchor, stroke):
    direction, length = stroke
    end = anchor + direction * (length - 1)
    return min(anchor, end), max(anchor, end) + 1


def _read_tokens(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    if (h - 1) % 4 or (w - 1) % 4:
        raise ValueError("glyph tiles require a one-cell scaffold and 3x3 interiors")
    rows, cols = (h - 1) // 4, (w - 1) // 4
    if rows not in _ROW_AXES or cols not in _COL_AXES:
        raise ValueError("unsupported coarse dimensions")
    bg = int(a[0, 0])
    if not (np.all(a[::4, :] == bg) and np.all(a[:, ::4] == bg)):
        raise ValueError("broken background scaffold")
    tokens = []
    for r in range(rows):
        for c in range(cols):
            tile = a[4 * r + 1:4 * r + 4, 4 * c + 1:4 * c + 4]
            ink = tile != bg
            if not np.any(ink):
                continue
            colors = np.unique(tile[ink])
            mask = tuple(int(v) for v in ink.flat)
            if len(colors) != 1 or mask not in _MASK_LOOKUP:
                raise ValueError("each occupied tile must be one monochrome command glyph")
            tokens.append((r, c, _MASK_LOOKUP[mask], int(colors[0])))
    if not tokens:
        raise ValueError("at least one glyph is required")
    return (rows, cols), bg, tuple(tokens)


def _coarse_from_tokens(shape, bg, tokens, mapping=_STROKES):
    coarse = np.full(shape, bg, dtype=np.int8)
    for r, anchor, command, color in tokens:
        left, right = _span(anchor, mapping[command])
        if left < 0 or right > shape[1]:
            raise ValueError("command leaves the coarse canvas")
        occupied = coarse[r, left:right]
        if np.any((occupied != bg) & (occupied != color)):
            raise ValueError("different-colored strokes may not overlap")
        occupied[:] = color
    return coarse


def _decode(input_grid):
    shape, bg, tokens = _read_tokens(input_grid)
    return _coarse_from_tokens(shape, bg, tokens), bg, tokens


def _runs(row, bg):
    result = []
    c = 0
    while c < len(row):
        color = int(row[c])
        if color == bg:
            c += 1
            continue
        end = c + 1
        while end < len(row) and int(row[end]) == color:
            end += 1
        result.append((color, c, end))
        c = end
    return result


def _rectangles(coarse, bg, merge=True):
    """Return (color, top-row, bottom-row-exclusive, left, right-exclusive)."""
    row_runs = [_runs(row, bg) for row in coarse]
    result = []
    for r, runs in enumerate(row_runs):
        for run in runs:
            if merge and r and run in row_runs[r - 1]:
                continue
            end = r + 1
            while merge and end < len(row_runs) and run in row_runs[end]:
                end += 1
            color, left, right = run
            result.append((color, r, end, left, right))
    return result


def _render(coarse, bg, merge=True):
    py, px = _INTERIOR // coarse.shape[0], _INTERIOR // coarse.shape[1]
    out = np.full((_SIZE, _SIZE), bg, dtype=np.int8)
    for color, top, bottom, left, right in _rectangles(coarse, bg, merge):
        out[_GAP + top * py:bottom * py, _GAP + left * px:right * px] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode only the supplied grid; do not consult examples, seeds, or files."""
    coarse, bg, _ = _decode(input_grid)
    return _render(coarse, bg)


@lru_cache(maxsize=5)
def _encodings(length):
    """All small nonredundant-enough covers of a same-color run of length 2..6.

    Each anchor holds at most one glyph. Fewer glyphs than decoded cells also
    keeps the glyph-as-single-pixel shortcut visibly wrong on these run rows.
    This is a finite grammar table, not a cache of generated examples.
    """
    choices = []
    for anchor in range(length):
        legal = [None]
        for command, stroke in enumerate(_STROKES):
            left, right = _span(anchor, stroke)
            if 0 <= left < right <= length:
                legal.append(command)
        choices.append(legal)
    found = []
    for commands in product(*choices):
        selected = tuple((c, k) for c, k in enumerate(commands) if k is not None)
        if not 1 <= len(selected) <= min(4, length - 1):
            continue
        cells = set()
        for c, k in selected:
            left, right = _span(c, _STROKES[k])
            cells.update(range(left, right))
        if len(cells) == length:
            found.append(selected)
    assert found
    return tuple(found)


def _encode(coarse, bg, rng, overrides=None):
    rows, cols = coarse.shape
    inp = np.full((4 * rows + 1, 4 * cols + 1), bg, dtype=np.int8)
    overrides = {} if overrides is None else overrides
    for r, row in enumerate(coarse):
        for color, left, right in _runs(row, bg):
            signature = (r, left, right)
            covers = _encodings(right - left)
            cover = overrides.get(signature)
            if cover is None:
                cover = covers[int(rng.integers(len(covers)))]
            for relative_anchor, command in cover:
                c = left + relative_anchor
                tile = inp[4 * r + 1:4 * r + 4, 4 * c + 1:4 * c + 4]
                assert np.all(tile == bg)
                tile[np.asarray(_MASKS[command], dtype=bool)] = color
    return inp


def _random_row(rng, cols, bg, palette):
    row = np.full(cols, bg, dtype=np.int8)
    if not palette:
        return row
    # Sample intervals rather than unrelated pixels: every maximal run remains
    # constructible from the horizontal glyph vocabulary (minimum length two).
    intervals = [(l, r) for l in range(cols) for r in range(l + 2, cols + 1)]
    for _ in range(int(rng.integers(0, cols // 2 + 1))):
        candidates = [(l, r) for l, r in intervals if np.all(row[l:r] == bg)]
        if not candidates:
            break
        left, right = candidates[int(rng.integers(len(candidates)))]
        row[left:right] = int(rng.choice(palette))
    return row


def _depth(coarse, bg):
    return max((b - a for _, a, b, _, _ in _rectangles(coarse, bg)), default=0)


def _free_training(rng, shape, bg, palette):
    coarse = np.full(shape, bg, dtype=np.int8)
    for r in range(shape[0]):
        if r and rng.random() < 0.3:
            coarse[r] = coarse[r - 1]
            if _depth(coarse[:r + 1], bg) <= 2:
                continue
        coarse[r] = _random_row(rng, shape[1], bg, palette)
        if _depth(coarse[:r + 1], bg) > 2:
            coarse[r] = bg  # explicit interruption, no new decoding exception
    if not np.any(coarse != bg):
        r = int(rng.integers(shape[0]))
        left = int(rng.integers(shape[1] - 1))
        coarse[r, left:left + 2] = palette[0]
    return coarse


def _teaching_pair(rng, role, cols, bg, main, auxiliary):
    """Variable-location witnesses, not complete parent-puzzle templates."""
    rows = 3 if role == "equivalent" else 4
    coarse = np.full((rows, cols), bg, dtype=np.int8)
    overrides = {}
    top = int(rng.integers(rows - 1))
    spare = [r for r in range(rows) if r not in (top, top + 1)]
    if role == "equivalent":
        left = int(rng.integers(cols - 2))
        # A single R3 and two overlapping L2s have identical decoded endpoints.
        covers = [((0, 0),), ((1, 1), (2, 1))]
        if rng.integers(2):
            covers.reverse()
        for row, cover in zip((top, top + 1), covers):
            coarse[row, left:left + 3] = main
            overrides[(row, left, left + 3)] = cover
        left = int(rng.integers(cols - 3))
        coarse[spare[0], left:left + 4] = auxiliary
        overrides[(spare[0], left, left + 4)] = ((3, 2),)
    else:
        left = int(rng.integers(cols - 3))
        first = (left, left + 4)
        controls = [(a, a + n) for n in (2, 3) for a in range(cols - n + 1)
                    if max(a, first[0]) < min(a + n, first[1])]
        second = controls[int(rng.integers(len(controls)))]
        specifications = [(first, ((3, 2),))]
        cover = ((1, 1),) if second[1] - second[0] == 2 else ((0, 0),)
        specifications.append((second, cover))
        if rng.integers(2):
            specifications.reverse()
        for row, ((a, b), cover) in zip((top, top + 1), specifications):
            coarse[row, a:b] = main
            overrides[(row, a, b)] = cover
        # Together with the first pair, every glyph has at least two colors.
        for row, size, command, anchor in zip(spare, (3, 2), (0, 1), (0, 1)):
            a = int(rng.integers(cols - size + 1))
            coarse[row, a:a + size] = auxiliary
            overrides[(row, a, a + size)] = ((anchor, command),)
    return coarse, _encode(coarse, bg, rng, overrides)


def _test_coarse(rng, cols, bg, palette):
    rows = 6
    coarse = np.stack([_random_row(rng, cols, bg, palette) for _ in range(rows)])
    depth = int(rng.integers(3, 6))
    top = int(rng.integers(rows - depth + 1))
    length = int(rng.integers(2, cols + 1))
    left = int(rng.integers(cols - length + 1))
    right = left + length
    color = int(rng.choice(palette))
    others = [c for c in palette if c != color]
    for r in range(top, top + depth):
        coarse[r] = bg
        coarse[r, left:right] = color
        # Other rectangles may flank the main stack, including at either edge.
        coarse[r, :left] = _random_row(rng, left, bg, others)
        coarse[r, right:] = _random_row(rng, cols - right, bg, others)
    boundaries = ([top - 1] if top else []) + ([top + depth] if top + depth < rows else [])
    control_spans = [(a, b) for a in range(cols) for b in range(a + 2, cols + 1)
                     if (a, b) != (left, right) and max(a, left) < min(b, right)]
    if control_spans and rng.random() < 0.75:
        r = int(rng.choice(boundaries))
        a, b = control_spans[int(rng.integers(len(control_spans)))]
        coarse[r] = bg
        coarse[r, a:b] = color
    return coarse


def _novel_recolor(inp, coarse, bg, seen):
    """Finite uniqueness repair using only visible palette bijections.

    At most thirteen inputs precede a pair. Ninety distinct (background, first
    ink) assignments are available, so this fallback is guaranteed to find one.
    It normally returns immediately without changing the sampled palette.
    """
    if _key(inp) not in seen:
        return inp, coarse, bg
    first = int(inp.flat[np.flatnonzero(inp.ravel() != bg)[0]])
    old_other = [x for x in range(10) if x not in (bg, first)]
    for new_bg in range(10):
        for new_first in range(10):
            if new_first == new_bg:
                continue
            new_other = [x for x in range(10) if x not in (new_bg, new_first)]
            table = np.zeros(10, dtype=np.int8)
            table[bg], table[first] = new_bg, new_first
            table[old_other] = new_other
            changed = table[inp]
            if _key(changed) not in seen:
                return changed, table[coarse], new_bg
    raise AssertionError("ninety recolorings cannot collide with thirteen inputs")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer("seed", seed, 0)
    nt = _integer("num_train", num_train, 2, 10)
    nq = _integer("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    colors = [int(c) for c in rng.permutation(10)[:4]]
    widths = [4, 6]
    if rng.integers(2):
        widths.reverse()
    roles = [("equivalent", widths[0], colors[0], colors[1]),
             ("partial", widths[1], colors[2], colors[3])]
    if rng.integers(2):
        roles.reverse()
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_bg = None
    for index in range(nt + nq):
        split = "train" if index < nt else "test"
        if index < 2:
            role, cols, main, auxiliary = roles[index]
            bg = int(rng.choice([c for c in range(10)
                                 if c not in (main, auxiliary, previous_bg)]))
            coarse, inp = _teaching_pair(rng, role, cols, bg, main, auxiliary)
        else:
            bg = int(rng.integers(10))
            available = [c for c in range(10) if c != bg]
            count = int(rng.integers(1, 7))
            palette = [int(c) for c in rng.choice(available, count, replace=False)]
            cols = int(rng.choice(_COL_AXES))
            if split == "train":
                rows = int(rng.integers(1, 5))
                coarse = _free_training(rng, (rows, cols), bg, palette)
            else:
                coarse = _test_coarse(rng, cols, bg, palette)
            inp = _encode(coarse, bg, rng)
        inp, coarse, bg = _novel_recolor(inp, coarse, bg, seen)
        seen.add(_key(inp))
        previous_bg = bg
        puzzle[split].append({"input": inp, "output": _render(coarse, bg)})
    validate_generated_puzzle(puzzle)
    return puzzle


def _cell_render(coarse, bg, *, partial_bridge=False):
    """Authored cross-check: cell cores and bridges, without rectangle painting.

    partial_bridge=True is the deliberately wrong touching-pixels alternative.
    This is not an external or blind reference solver.
    """
    rows, cols = coarse.shape
    py, px = _INTERIOR // rows, _INTERIOR // cols
    out = np.full((_SIZE, _SIZE), bg, dtype=np.int8)
    runs = [_runs(row, bg) for row in coarse]
    for r in range(rows):
        for c in range(cols):
            color = int(coarse[r, c])
            if color == bg:
                continue
            y, x = _GAP + r * py, _GAP + c * px
            out[y:y + py - _GAP, x:x + px - _GAP] = color
            if c + 1 < cols and coarse[r, c + 1] == color:
                out[y:y + py - _GAP, x + px - _GAP:x + px] = color
        if r + 1 == rows:
            continue
        y = (r + 1) * py
        for color, left, right in runs[r]:
            if not partial_bridge and (color, left, right) in runs[r + 1]:
                out[y:y + _GAP, _GAP + left * px:right * px] = color
            elif partial_bridge:
                for lower_color, lower_left, lower_right in runs[r + 1]:
                    a, b = max(left, lower_left), min(right, lower_right)
                    if lower_color == color and a < b:
                        out[y:y + _GAP, _GAP + a * px:b * px] = color
    return out


def _features(coarse, bg, tokens):
    runs = [_runs(row, bg) for row in coarse]
    exact = any(set(a) & set(b) for a, b in zip(runs, runs[1:]))
    partial = any(ca == cb and (la, ra) != (lb, rb) and max(la, lb) < min(ra, rb)
                  for a, b in zip(runs, runs[1:]) for ca, la, ra in a for cb, lb, rb in b)
    # Compare the syntax of the matching run itself. An unrelated neighboring
    # run changing must not count as evidence for equivalent glyph encodings.
    def syntax(r, run):
        color, left, right = run
        return tuple((c, k) for rr, c, k, ink in tokens
                     if rr == r and ink == color and left <= c < right)

    equivalent = any(syntax(r, run) != syntax(r + 1, run)
                     for r in range(len(runs) - 1)
                     for run in set(runs[r]) & set(runs[r + 1]))
    return bool(exact), bool(partial), bool(equivalent)


def _training_dictionaries(infos):
    """Check inferability inside the anchored horizontal-stroke hypothesis class.

    Candidate lengths 1..6 and either direction, not merely the three correct
    commands. Targets are recovered from output cell cores. No test labels.
    """
    catalog = ((1, 1),) + tuple((d, n) for d in (-1, 1) for n in range(2, 7))
    observations = []
    options = []
    for shape, bg, tokens, output in infos:
        py, px = _INTERIOR // shape[0], _INTERIOR // shape[1]
        target = output[_GAP:_SIZE - _GAP:py, _GAP:_SIZE - _GAP:px]
        assert target.shape == shape
        observations.append((shape, bg, tokens, target))
    for command in range(3):
        possible = []
        for stroke in catalog:
            fits = True
            for shape, bg, tokens, target in observations:
                for r, c, k, color in tokens:
                    if k != command:
                        continue
                    left, right = _span(c, stroke)
                    if left < 0 or right > shape[1] or np.any(target[r, left:right] != color):
                        fits = False
                        break
                if not fits:
                    break
            if fits:
                possible.append(stroke)
        options.append(possible)
    answers = []
    for mapping in product(*options):
        if all(np.array_equal(_coarse_from_tokens(shape, bg, tokens, mapping), target)
               for shape, bg, tokens, target in observations):
            answers.append(mapping)
    return answers


def _assert_rectangular_components(out, bg, expected):
    """Each painted 4-connected component is exactly one filled rectangle."""
    visited = set()
    actual = set()
    for r, c in zip(*np.nonzero(out != bg)):
        start = int(r), int(c)
        if start in visited:
            continue
        color = int(out[start])
        stack, cells = [start], []
        visited.add(start)
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= q[0] < _SIZE and 0 <= q[1] < _SIZE and q not in visited
                        and int(out[q]) == color):
                    visited.add(q)
                    stack.append(q)
        top, bottom = min(y for y, _ in cells), max(y for y, _ in cells) + 1
        left, right = min(x for _, x in cells), max(x for _, x in cells) + 1
        assert len(cells) == (bottom - top) * (right - left), "nonrectangular component"
        actual.add((color, top, bottom, left, right))
    assert actual == expected, "incorrect rectangle count, bounds, or color"


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, infos = set(), []
    evidence = [False, False, False]
    colors_by_mask = [set() for _ in range(3)]
    omission_a = omission_b = touching_wrong = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            coarse, bg, tokens = _decode(inp)
            assert out.shape == (_SIZE, _SIZE)
            assert np.array_equal(out, solve(inp))
            assert np.array_equal(out, _cell_render(coarse, bg))
            assert set(np.unique(inp)) == set(np.unique(out)), "lost or invented colors"
            assert np.all(out[:_GAP] == bg) and np.all(out[-_GAP:] == bg)
            assert np.all(out[:, :_GAP] == bg) and np.all(out[:, -_GAP:] == bg)
            assert all(right - left >= 2 for row in coarse for _, left, right in _runs(row, bg))
            py, px = _INTERIOR // coarse.shape[0], _INTERIOR // coarse.shape[1]
            expected = {(color, _GAP + top * py, bottom * py, _GAP + left * px, right * px)
                        for color, top, bottom, left, right in _rectangles(coarse, bg)}
            _assert_rectangular_components(out, bg, expected)
            points = np.full(coarse.shape, bg, dtype=np.int8)
            for r, c, _, color in tokens:
                points[r, c] = color
            a_changes = not np.array_equal(out, _render(points, bg))
            b_changes = not np.array_equal(out, _render(coarse, bg, merge=False))
            if split == "train":
                assert coarse.shape[0] <= 4 and _depth(coarse, bg) <= 2
                evidence = [a or b for a, b in zip(evidence, _features(coarse, bg, tokens))]
                for _, _, k, color in tokens:
                    colors_by_mask[k].add(color)
                infos.append((coarse.shape, bg, tokens, out))
                omission_a |= a_changes
                omission_b |= b_changes
                touching_wrong |= not np.array_equal(out, _cell_render(coarse, bg, partial_bridge=True))
            else:
                assert coarse.shape[0] == 6 and _depth(coarse, bg) >= 3
                assert a_changes and b_changes, "both inherited mechanisms must affect each test"
    assert all(evidence), "need exact, partial-overlap, and different-syntax equality witnesses"
    assert all(len(cs) >= 2 for cs in colors_by_mask), "glyph meaning must not be tied to ink color"
    assert omission_a and omission_b and touching_wrong
    assert _training_dictionaries(infos) == [_STROKES], "ambiguous horizontal glyph meanings"


def _canonical(coarse, bg):
    roles = {bg: 0}
    return coarse.shape, tuple(roles.setdefault(int(c), len(roles)) for c in coarse.flat)


def self_test():
    """Bounded: 42 sequential five-pair puzzles, small regressions, no disk I/O."""
    signatures, layouts, shapes, palettes, glyph_counts, depths, counts = (set() for _ in range(7))
    partial_tests = zeros = empty_rows = empty_cols = different_syntax = 0
    pair_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = tuple((_key(p["input"]), _key(p["output"]))
                          for s in ("train", "test") for p in puzzle[s])
        signatures.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pair_count += 1
                coarse, bg, tokens = _decode(pair["input"])
                layouts.add(_canonical(coarse, bg))
                shapes.add(coarse.shape)
                palettes.add((bg, tuple(sorted(set(int(v) for v in coarse.flat) - {bg}))))
                glyph_counts.add(len(tokens))
                counts.add(len(_rectangles(coarse, bg)))
                zeros += bg != 0 and np.any(coarse == 0)
                empty_rows += np.any(np.all(coarse == bg, axis=1))
                empty_cols += np.any(np.all(coarse == bg, axis=0))
                if split == "test":
                    depths.add(_depth(coarse, bg))
                    partial_tests += _features(coarse, bg, tokens)[1]
                else:
                    different_syntax += _features(coarse, bg, tokens)[2]
        if seed == 0:
            repeat = generate_puzzle(seed)
            assert signature == tuple((_key(p["input"]), _key(p["output"]))
                                      for s in ("train", "test") for p in repeat[s])
    assert len(signatures) == 42
    assert len(layouts) >= 100 and len(shapes) >= 14 and len(palettes) >= 80
    assert len(glyph_counts) >= 10 and len(counts) >= 6
    assert {3, 4, 5}.issubset(depths) and partial_tests >= 10
    assert zeros >= 10 and empty_rows >= 10 and empty_cols >= 10
    assert different_syntax >= 42
    # API extremes; the host additionally exercises every legal count pair.
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(91001 + nt + nq, nt, nq)
        assert (len(p["train"]), len(p["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Same glyph, different anchors: partial overlap must NOT bridge.
    coarse = np.array([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=np.int8)
    assert len(_rectangles(coarse, 0)) == 2
    assert not np.array_equal(_render(coarse, 0), _cell_render(coarse, 0, partial_bridge=True))
    # Same span after an interruption must form two separate rectangles.
    coarse = np.array([[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0]], dtype=np.int8)
    assert len(_rectangles(coarse, 0)) == 2 and _depth(coarse, 0) == 1
    # Joint causal controls: glyph syntax does not decide vertical merging;
    # the maximal decoded color/span does. All inputs are hand-built here.
    controls = (
        ((2, 4), ((0, 0, 0, 1), (1, 1, 1, 1), (1, 2, 1, 1)),
         ((1, 0, 2, 0, 3),)),  # R3 versus two overlapping L2s: merge
        ((2, 4), ((0, 0, 0, 1), (1, 1, 0, 1)),
         ((1, 0, 1, 0, 3), (1, 1, 2, 1, 4))),  # same mask, shifted: separate
        ((2, 4), ((0, 0, 0, 1), (1, 0, 0, 2)),
         ((1, 0, 1, 0, 3), (2, 1, 2, 0, 3))),  # same span, changed ink
        ((1, 6), ((0, 1, 1, 1), (0, 2, 0, 1)),
         ((1, 0, 1, 0, 5),)),  # touching same-color strokes form one run
        ((3, 4), ((0, 0, 0, 1), (2, 0, 0, 1)),
         ((1, 0, 1, 0, 3), (1, 2, 3, 0, 3))),  # empty row interrupts
    )
    for shape, tokens, expected in controls:
        inp = np.zeros((4 * shape[0] + 1, 4 * shape[1] + 1), dtype=np.int8)
        for r, c, command, color in tokens:
            tile = inp[4 * r + 1:4 * r + 4, 4 * c + 1:4 * c + 4]
            tile[np.asarray(_MASKS[command], dtype=bool)] = color
        decoded, bg, _ = _decode(inp)
        assert tuple(_rectangles(decoded, bg)) == expected
        py, px = _INTERIOR // shape[0], _INTERIOR // shape[1]
        bounds = {(color, _GAP + top * py, bottom * py, _GAP + left * px, right * px)
                  for color, top, bottom, left, right in expected}
        _assert_rectangular_components(solve(inp.tolist()), bg, bounds)
    # Mutation checks cover outputs, scaffold, and an invalid command mask.
    source = generate_puzzle(9, 2, 1)
    for mutation in ("output", "scaffold", "glyph"):
        broken = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in source[s]]
                  for s in ("train", "test")}
        p = broken["test"][0]
        if mutation == "output":
            p["output"][0, 0] = (int(p["output"][0, 0]) + 1) % 10
        elif mutation == "scaffold":
            p["input"][0, 0] = (int(p["input"][0, 0]) + 1) % 10
        else:
            _, bg, tokens = _read_tokens(p["input"])
            r, c, _, color = tokens[0]
            p["input"][4 * r + 1:4 * r + 4, 4 * c + 1:4 * c + 4] = color
        try:
            validate_generated_puzzle(broken)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"accepted {mutation} corruption")
    return {"seeds": 42, "pairs": pair_count, "distinct_puzzles": len(signatures),
            "coarse_shapes": sorted(shapes), "normalized_layouts": len(layouts),
            "background_palette_combinations": len(palettes), "glyph_counts": sorted(glyph_counts),
            "rectangle_counts": sorted(counts), "test_stack_depths": sorted(depths),
            "partial_overlap_tests": int(partial_tests), "payload_zero_pairs": int(zeros),
            "empty_row_pairs": int(empty_rows), "empty_column_pairs": int(empty_cols),
            "different_syntax_merge_train_pairs": int(different_syntax),
            "independent_semantic_review": False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
