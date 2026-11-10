"""Locally coded square-adjacency analogies; no parent imports or saved state.

All generation is in memory. The public solver reads only its supplied pixels.
"""
from __future__ import annotations

from itertools import combinations
import hashlib

import numpy as np


_SHORT = (2, 3, 4, 5)
_LONG = (7, 8, 9)
_LENGTHS = _SHORT + _LONG
_DIGITS = (1, 4, 9, 6)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid ARC color")
    return a.astype(np.int8, copy=False)


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(a):
    return a.shape, a.astype(np.uint8, copy=False).tobytes()


def _digit(length):
    return ((int(length) - 1) ** 2) % 10


def _outline(a, box, color):
    r, c, h, w = box[:4]
    a[r, c:c+w] = a[r+h-1, c:c+w] = color
    a[r:r+h, c] = a[r:r+h, c+w-1] = color


def _inside(a, b):
    r, c, h, w = a[:4]
    y, x, bh, bw = b[:4]
    return y < r and x < c and r+h < y+bh and c+w < x+bw


def _disjoint(a, b):
    r, c, h, w = a[:4]
    y, x, bh, bw = b[:4]
    return r+h <= y or y+bh <= r or c+w <= x or x+bw <= c


def _rectangles(a):
    """Enumerate long monochrome outlines, including harmless edge extensions."""
    height, width = a.shape
    found = set()
    for top in range(height - 10):
        left = 0
        while left < width:
            color = int(a[top, left])
            end = left + 1
            while end < width and int(a[top, end]) == color:
                end += 1
            if end - left >= 9:
                for bottom in range(top + 10, height):
                    columns = np.flatnonzero(np.all(a[top:bottom+1, left:end] == color, axis=0)) + left
                    for x, y in combinations(columns.tolist(), 2):
                        if y-x >= 8 and np.all(a[bottom, x:y+1] == color):
                            found.add((top, x, bottom-top+1, y-x+1, color))
            left = end
    return sorted(found)


def _extract(a):
    """Find four equal framed panels and the two disjoint enclosing pairs."""
    rectangles = _rectangles(a)
    by_frame = {}
    for box in rectangles:
        r, c, h, w, color = box
        if 11 <= h <= 13 and 9 <= w <= 13:
            interior = a[r+1:r+h-1, c+1:c+w-1]
            if not np.all(interior == color):
                by_frame.setdefault((h, w, color), []).append(box)
    scenes = {}
    for (_, _, frame), panels in by_frame.items():
        if len(panels) != 4 or any(not _disjoint(x, y) for x, y in combinations(panels, 2)):
            continue
        groups = {}
        for box in rectangles:
            if box[4] == frame:
                continue
            members = tuple(i for i, p in enumerate(panels) if _inside(p, box))
            if len(members) == 2:
                groups.setdefault(members, []).append(box)
        interiors = [a[r+1:r+h-1, c+1:c+w-1].copy() for r, c, h, w, _ in panels]
        blanks = [i for i, p in enumerate(interiors) if np.all(p == p[0, 0])]
        if len(blanks) != 1:
            continue
        blank = blanks[0]
        neutral = int(interiors[blank][0, 0])
        if neutral == frame:
            continue
        for (first, boxes1), (second, boxes2) in combinations(groups.items(), 2):
            if set(first) & set(second):
                continue
            boxes = [(x, y) for x in boxes1 for y in boxes2 if _disjoint(x, y)]
            if not boxes:
                continue
            target_ids, ref_ids = (first, second) if blank in first else (second, first)
            occupied_id = next(i for i in target_ids if i != blank)
            ref_ids = tuple(sorted(ref_ids, key=lambda i: np.count_nonzero(interiors[i] != neutral)))
            signature = (tuple(panels), tuple(sorted((first, second))))
            scenes[signature] = {
                "neutral": neutral, "panels": panels, "blank_id": blank,
                "occupied_id": occupied_id, "reference_ids": ref_ids,
                "reference": [interiors[i] for i in ref_ids],
                "occupied": interiors[occupied_id],
                "boxes": min(boxes, key=lambda pair: (sum(b[2]*b[3] for b in pair), pair)),
            }
    _require(len(scenes) == 1, "need a unique two-pair frame association")
    return next(iter(scenes.values()))


def _read_bars(panel, neutral):
    """Read a source panel; the singleton row identifies the marker, not color."""
    occupied = [np.flatnonzero(row != neutral) for row in panel]
    singles = [r for r, cols in enumerate(occupied) if len(cols) == 1]
    _require(len(singles) == 1, "need exactly one isolated marker row")
    marker_row = singles[0]
    column = int(occupied[marker_row][0])
    marker = int(panel[marker_row, column])
    active = [r for r, cols in enumerate(occupied) if len(cols)]
    _require(all(y-x >= 2 for x, y in zip(active, active[1:])), "active rows must be separated")
    bars = []
    for row, cols in enumerate(occupied):
        if not len(cols) or row == marker_row:
            continue
        start, length = int(cols[0]), len(cols)
        _require(length in _LENGTHS, "unsupported bar length")
        _require(np.array_equal(cols, np.arange(start, start+length)), "bar is not contiguous")
        color = int(panel[row, start])
        _require(np.all(panel[row, cols] == color), "bar must be monochrome")
        _require(start+length <= column or start > column, "bar crosses marker column")
        bars.append((row, start, length, color))
    _require(1 <= len(bars) <= 5, "need one to five bars")
    inks = {bar[3] for bar in bars}
    _require(len(inks) == 1 and marker not in inks, "bar and marker roles must be distinct")
    _require(np.count_nonzero(panel[:, column] != neutral) == 1, "marker column must otherwise be blank")
    return {"panel": panel, "neutral": neutral, "marker_row": marker_row,
            "column": column, "marker": marker, "ink": next(iter(inks)), "bars": bars}


def _analyze(a):
    scene = _extract(a)
    neutral = scene["neutral"]
    source, completed = scene["reference"]
    ref = _read_bars(source, neutral)
    target = _read_bars(scene["occupied"], neutral)
    _require(sorted(b[2] for b in ref["bars"]) == list(_SHORT), "reference must demonstrate lengths 2..5")
    changes = np.zeros(source.shape, dtype=bool)
    code = {}
    for row, _, length, _ in ref["bars"]:
        changes[row, ref["column"]] = True
        color = int(completed[row, ref["column"]])
        _require(color != neutral, "reference label is invisible")
        code[_digit(length)] = color
    _require(np.array_equal(source[~changes], completed[~changes]), "reference changed an original cell")
    _require(np.count_nonzero(source != completed) == 4, "reference needs exactly one label per bar")
    _require(len(set(code.values())) == 4, "four arithmetic classes need distinct colors")
    _require({ref["ink"], ref["marker"]} <= set(code.values()), "code must expose both transferable roles")
    shared = set(code.values()) - {ref["ink"], ref["marker"]}
    # These distinct visible roles make the transfer unique, without a hidden key.
    _require(len({neutral, ref["ink"], ref["marker"], target["ink"], target["marker"], *shared}) == 7,
             "ambiguous or collapsed source/target color roles")
    mapping = {ref["ink"]: target["ink"], ref["marker"]: target["marker"]}
    transferred = {digit: mapping.get(color, color) for digit, color in code.items()}
    _require(all(_digit(b[2]) in transferred for b in target["bars"]), "target class absent from reference")
    scene.update(ref=ref, target=target, code=code, transferred=transferred)
    return scene


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct the blank companion interior from this input alone."""
    scene = _analyze(_grid(input_grid))
    target = scene["target"]
    answer = target["panel"].copy()
    for row, _, length, _ in target["bars"]:
        answer[row, target["column"]] = scene["transferred"][_digit(length)]
    return answer


def _panel(rng, shape, lengths, neutral, ink, marker):
    h, w = shape
    n = len(lengths)
    lengths = [int(x) for x in rng.permutation(lengths)]
    # Allocate unused rows to margins and to the already nonzero row gaps.
    spare = h - (2*(n+1)-1)
    spaces = rng.multinomial(spare, np.full(n+2, 1/(n+2)))
    rows = [int(spaces[0])]
    for i in range(n):
        rows.append(rows[-1] + 2 + int(spaces[i+1]))
    marker_slot = int(rng.integers(n+1))
    marker_row = rows.pop(marker_slot)
    possible = [c for c in range(w) if max(c, w-c-1) >= max(lengths)]
    column = int(rng.choice(possible))
    panel = np.full(shape, neutral, dtype=np.int8)
    panel[marker_row, column] = marker
    bars = []
    for row, length in zip(rows, lengths):
        sides = ([-1] if column >= length else []) + ([1] if w-column-1 >= length else [])
        side = int(rng.choice(sides))
        start = (int(rng.integers(column-length+1)) if side == -1
                 else int(rng.integers(column+1, w-length+1)))
        panel[row, start:start+length] = ink
        bars.append((row, start, length, ink))
    return panel, bars, column


def _layout(rng, shape, axis):
    """axis 0: horizontal panel pairs, stacked groups; axis 1: the reverse."""
    ph, pw = shape[0]+2, shape[1]+2
    gh, gw = (ph+2, 2*pw+3) if axis == 0 else (2*ph+3, pw+2)
    minimum_h, minimum_w = (2*gh+1, gw) if axis == 0 else (gh, 2*gw+1)
    height = int(rng.integers(minimum_h, 31))
    width = int(rng.integers(minimum_w, 31))
    spare = (height-minimum_h) if axis == 0 else (width-minimum_w)
    extra = rng.multinomial(spare, np.full(5, .2))
    spans = [(gh if axis == 0 else gw)+int(extra[i]) for i in range(2)]
    starts = [int(extra[2]), int(extra[2])+spans[0]+1+int(extra[3])]
    boxes, positions = [], []
    for i in range(2):
        if axis == 0:
            bh, bw = spans[i], int(rng.integers(gw, width+1))
            r, c = starts[i], int(rng.integers(width-bw+1))
            gap = int(rng.integers(1, bw-2*pw-1))
            begin = int(rng.integers(1, bw-2*pw-gap))
            pts = [(r+int(rng.integers(1, bh-ph)), c+x) for x in (begin, begin+pw+gap)]
        else:
            bh, bw = int(rng.integers(gh, height+1)), spans[i]
            r, c = int(rng.integers(height-bh+1)), starts[i]
            gap = int(rng.integers(1, bh-2*ph-1))
            begin = int(rng.integers(1, bh-2*ph-gap))
            pts = [(r+x, c+int(rng.integers(1, bw-pw))) for x in (begin, begin+ph+gap)]
        boxes.append((r, c, bh, bw))
        positions.append(pts)
    return height, width, boxes, positions


def _wallpaper(rng, shape, colors):
    rr, cc = np.indices(shape)
    palette = rng.choice(colors, size=int(rng.integers(3, 7)), replace=False)
    a = palette[(rr//int(rng.integers(1, 3)) + cc//int(rng.integers(1, 3))) % len(palette)].copy()
    noise = rng.random(shape) < .04
    a[noise] = rng.choice(palette, size=int(noise.sum()))
    # Short runs remain camouflage; no accidental long straight edge is allowed.
    for r in range(shape[0]):
        for c in range(shape[1]):
            forbidden = set()
            if r >= 2 and a[r-1, c] == a[r-2, c]:
                forbidden.add(int(a[r-1, c]))
            if c >= 2 and a[r, c-1] == a[r, c-2]:
                forbidden.add(int(a[r, c-1]))
            if int(a[r, c]) in forbidden:
                a[r, c] = next(int(x) for x in palette if int(x) not in forbidden)
    return a.astype(np.int8)


def _make_pair(rng, lengths):
    min_h, min_w = max(9, 2*len(lengths)+1), max(7, max(lengths)+1)
    options = [(h, w, axis) for h in range(min_h, 12) for w in range(min_w, 12)
               for axis in (0, 1) if (h <= 10 if axis == 0 else w <= 10)]
    h, w, axis = options[int(rng.integers(len(options)))]
    shape = (h, w)
    colors = [int(x) for x in rng.permutation(10)]
    neutral, ref_ink, ref_marker, ink, marker, shared1, shared2, frame, group1, group2 = colors
    code = dict(zip(_DIGITS, [int(x) for x in rng.permutation([ref_ink, ref_marker, shared1, shared2])]))
    before, ref_bars, ref_column = _panel(rng, shape, _SHORT, neutral, ref_ink, ref_marker)
    after = before.copy()
    for row, _, length, _ in ref_bars:
        after[row, ref_column] = code[_digit(length)]
    target, target_bars, target_column = _panel(rng, shape, lengths, neutral, ink, marker)
    output = target.copy()
    mapping = {ref_ink: ink, ref_marker: marker}
    for row, _, length, _ in target_bars:
        color = code[_digit(length)]
        output[row, target_column] = mapping.get(color, color)
    blank = np.full(shape, neutral, dtype=np.int8)
    reference = [before, after] if rng.integers(2) else [after, before]
    incomplete = [target, blank] if rng.integers(2) else [blank, target]
    groups = [reference, incomplete] if rng.integers(2) else [incomplete, reference]
    height, width, boxes, positions = _layout(rng, shape, axis)
    canvas = _wallpaper(rng, (height, width), colors)
    group_colors = [group1, group1 if rng.random() < .4 else group2]
    for g in range(2):
        _outline(canvas, boxes[g], group_colors[g])
        for (r, c), interior in zip(positions[g], groups[g]):
            _outline(canvas, (r, c, h+2, w+2), frame)
            canvas[r+1:r+h+1, c+1:c+w+1] = interior
    return {"input": canvas, "output": output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    used = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train" and index == 0:
                lengths = list(_SHORT)
            elif split == "train" and index == 1:
                lengths = list(_LONG)
            elif split == "test":
                n = int(rng.integers(2, 6))
                lengths = [int(rng.choice(_SHORT)), int(rng.choice(_LONG))]
                lengths += [int(rng.choice(_LENGTHS)) for _ in range(n-2)]
            else:
                # Training separates the short/long regimes; tests recombine them.
                pool = _SHORT if rng.integers(2) == 0 else _LONG
                lengths = [int(rng.choice(pool)) for _ in range(int(rng.integers(1, 6)))]
            for _ in range(32):
                pair = _make_pair(rng, lengths)
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, learned = set(), set()
    transfer_exposed = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            scene = _analyze(a)
            target = scene["target"]
            lengths = {b[2] for b in target["bars"]}
            mixed = bool(lengths & set(_SHORT)) and bool(lengths & set(_LONG))
            _require(mixed == (split == "test"), "short/long train-test recombination shift is missing")
            _require(out.shape == target["panel"].shape, "output is not the blank interior crop")
            change = np.zeros(out.shape, dtype=bool)
            for row, _, length, _ in target["bars"]:
                change[row, target["column"]] = True
                _require(int(out[row, target["column"]]) == scene["transferred"][_digit(length)],
                         "wrong arithmetic class, color-role transfer, or label column")
            _require(np.array_equal(out[~change], target["panel"][~change]), "original target cells changed")
            _require(np.count_nonzero(out != target["panel"]) == len(target["bars"]), "need one visible label per bar")
            if split == "train":
                learned.update(b[2] for b in target["bars"])
                classes = {_digit(b[2]) for b in target["bars"]}
                transfer_exposed |= classes == set(_DIGITS)
    _require(set(_LENGTHS) <= learned, "training must expose short and long arithmetic cases")
    _require(transfer_exposed, "training must expose all four codes and both transferred roles")


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                a = np.asarray(pair[name], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _replace_panel(canvas, box, interior):
    r, c, h, w = box[:4]
    canvas[r+1:r+h-1, c+1:c+w-1] = interior


def _diagnostics():
    # Handwritten arithmetic regression includes the multi-digit square cases.
    assert [_digit(x) for x in _LENGTHS] == [1, 4, 9, 6, 6, 9, 4]
    p = generate_puzzle(123, 2, 1)
    pair = p["train"][0]
    a = pair["input"]
    original = a.copy()
    scene = _analyze(a)
    expected = pair["output"]
    assert np.array_equal(solve(a), expected) and np.array_equal(a, original)
    # Reordering the complete pair cannot change direction or answer.
    swapped = a.copy()
    source, completed = scene["reference"]
    before_id, after_id = scene["reference_ids"]
    _replace_panel(swapped, scene["panels"][before_id], completed)
    _replace_panel(swapped, scene["panels"][after_id], source)
    assert np.array_equal(solve(swapped), expected)
    # Exact same target, changed reference code: a target-only rule must fail.
    twin = a.copy()
    new_completed = completed.copy()
    old_colors = [scene["code"][digit] for digit in _DIGITS]
    new_code = dict(zip(_DIGITS, old_colors[1:]+old_colors[:1]))
    for row, _, length, _ in scene["ref"]["bars"]:
        new_completed[row, scene["ref"]["column"]] = new_code[_digit(length)]
    _replace_panel(twin, scene["panels"][after_id], new_completed)
    twin_answer = solve(twin)
    assert not np.array_equal(twin_answer, expected)
    assert np.array_equal(_analyze(twin)["occupied"], scene["occupied"])
    # The inherited color roles are causal, not just shared literal colors.
    literal = scene["occupied"].copy()
    for row, _, length, _ in scene["target"]["bars"]:
        literal[row, scene["target"]["column"]] = scene["code"][_digit(length)]
    assert not np.array_equal(literal, expected)
    # Omitting adjacency subtraction (L^2) or squaring (L-1) changes known codes.
    assert scene["transferred"][_digit(2)] != scene["transferred"][2**2 % 10]
    assert scene["transferred"][_digit(5)] != scene["transferred"][5-1]
    # Both spatial reflections preserve the horizontal arithmetic rule.
    for transformed, answer in ((a[:, ::-1], expected[:, ::-1]), (a[::-1], expected[::-1])):
        assert np.array_equal(solve(transformed.copy()), answer)
    damaged = {s: [{k: v.copy() for k, v in q.items()} for q in p[s]] for s in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted answer")


def self_test():
    """42 sequential seeds, bounded digest/feature sets, no files or traces."""
    shapes, output_shapes, layouts, signatures, codebooks = set(), set(), set(), set(), set()
    marker_columns, counts, sides, normalized = set(), set(), set(), set()
    reference_orders, occupied_slots, group_axes, neutrals = set(), set(), set(), set()
    pair_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                scene = _analyze(pair["input"])
                target = scene["target"]
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                shapes.add(pair["input"].shape)
                output_shapes.add(pair["output"].shape)
                layouts.add(tuple(scene["panels"]))
                codebooks.add(tuple(scene["code"][d] for d in _DIGITS))
                marker_columns.add(target["column"])
                counts.add(len(target["bars"]))
                sides.add(tuple(sorted({-1 if b[1] < target["column"] else 1 for b in target["bars"]})))
                reference_orders.add(scene["reference_ids"][0] < scene["reference_ids"][1])
                occupied_slots.add(scene["occupied_id"] < scene["blank_id"])
                b0, b1 = scene["boxes"]
                group_axes.add("stacked" if b0[0]+b0[2] <= b1[0] or b1[0]+b1[2] <= b0[0] else "beside")
                neutrals.add(scene["neutral"])
                roles = {scene["neutral"]: 0}
                norm = np.array([roles.setdefault(int(x), len(roles)) for x in pair["output"].flat], dtype=np.uint8)
                normalized.add((pair["output"].shape, norm.tobytes()))
                pair_count += 1
    assert len(signatures) == 42 and len(shapes) >= 15 and len(output_shapes) >= 8
    assert len(layouts) >= 150 and len(normalized) >= 180 and len(codebooks) >= 100
    assert len(marker_columns) >= 8 and counts == {1, 2, 3, 4, 5}
    assert sides == {(-1,), (1,), (-1, 1)} and len(neutrals) == 10
    assert reference_orders == occupied_slots == {False, True} and len(group_axes) == 2
    for nt, ns in ((2, 1), (10, 1), (10, 4), (7, 3)):
        validate_generated_puzzle(generate_puzzle(100+nt, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
    _diagnostics()
    return {"seeds": 42, "pairs": pair_count, "input_shapes": len(shapes),
            "output_shapes": len(output_shapes), "panel_layouts": len(layouts),
            "normalized_outputs": len(normalized), "codebooks": len(codebooks),
            "marker_columns": sorted(marker_columns), "target_bar_counts": sorted(counts)}
