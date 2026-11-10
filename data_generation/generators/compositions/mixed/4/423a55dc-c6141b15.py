"""Exchange a segment/glyph arrangement, then globally shear its new foreground.

No parent module, external data, generation metadata, or saved puzzle is used by
solve. Diagnostics are in-memory; running this file prints one compact report.
"""
from __future__ import annotations

from itertools import combinations
import hashlib
import json

import numpy as np


# All templates have odd tight bounding boxes, an occupied box center, and one
# 8-connected component. Copies are translated without changing orientation.
_PATTERNS = (
    ("1",),
    ("010", "111", "010"),
    ("101", "010", "101"),
    ("111", "010", "010"),
    ("010", "111", "011"),
    ("11111", "00100", "00100"),
    ("10000", "11111", "10000"),
    ("111", "010", "010", "010", "111"),
    ("010", "010", "111", "010", "010"),
    ("010", "110", "111", "001", "001"),
    ("11111", "10101", "10101", "10101", "11111"),
)
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
_ORIENTATIONS = ("horizontal", "vertical", "diagonal")


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(a, b):
    if not _integer(a) or not 2 <= a <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(b) or not 1 <= b <= 4 or a <= b:
        raise ValueError("num_test must be an integer in 1..4 below num_train")
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid cells must be integer colors in 0..9")
    return a


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = r + dr, c + dc
                    if q in remaining:
                        remaining.remove(q)
                        part.add(q)
                        stack.append(q)
        result.append(frozenset(part))
    return result


def _line(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    if not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        raise ValueError("edges must be horizontal, vertical, or 45 degrees")
    length = max(abs(dr), abs(dc))
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return frozenset((a[0] + k * sr, a[1] + k * sc) for k in range(length + 1))


def _figure(centers):
    return frozenset().union(*(_line(a, b) for a, b in combinations(centers, 2)))


def _stamp(centers, motif):
    return frozenset((r + dr, c + dc) for r, c in centers for dr, dc in motif)


def _apart(a, b):
    # A background gap under 8-connectivity, for unambiguous segmentation.
    return not any((r + dr, c + dc) in b for r, c in a
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _straight(cells):
    if len(cells) < 4:
        return None
    a, b = min(cells), max(cells)
    try:
        return (a, b) if _line(a, b) == cells else None
    except ValueError:
        return None


def _glyphs(cells):
    parts = _components(cells)
    if len(parts) not in (2, 3):
        return None
    centers, motifs = [], []
    for part in parts:
        rows, cols = zip(*part)
        if (min(rows) + max(rows)) % 2 or (min(cols) + max(cols)) % 2:
            return None
        r, c = (min(rows) + max(rows)) // 2, (min(cols) + max(cols)) // 2
        motif = frozenset((rr - r, cc - c) for rr, cc in part)
        if (0, 0) not in motif:
            return None
        centers.append((r, c))
        motifs.append(motif)
    if len(set(motifs)) != 1:
        return None
    try:
        _figure(centers)
    except ValueError:
        return None
    if len(centers) == 3:
        a, b, c = centers
        if (b[0]-a[0])*(c[1]-a[1]) == (c[0]-a[0])*(b[1]-a[1]):
            return None
        ds = sorted((a[0]-b[0])**2 + (a[1]-b[1])**2
                    for a, b in combinations(centers, 2))
        if not (ds[0] == ds[1] and ds[0] + ds[1] == ds[2]):
            return None
    return tuple(sorted(centers)), motifs[0]


def _analyze(input_grid):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    if len(values) != 3 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("exactly three colors with a unique dominant background required")
    bg = int(values[np.argmax(counts)])
    fg = [int(v) for v in values if v != bg]
    candidates = []
    for edge, glyph in (fg, fg[::-1]):
        line_cells = frozenset(map(tuple, np.argwhere(a == edge).tolist()))
        ends = _straight(line_cells)
        if ends is None:
            continue
        marks = frozenset(map(tuple, np.argwhere(a == glyph).tolist()))
        info = _glyphs(marks)
        if info is not None and _apart(line_cells, marks):
            centers, motif = info
            candidates.append(dict(bg=bg, edge=edge, glyph=glyph, ends=ends,
                                   centers=centers, motif=motif))
    if len(candidates) != 1:
        raise ValueError("segment and repeated-glyph roles must be unambiguous")
    return candidates[0]


def _render(shape, bg, edge, glyph, figure, marks):
    if not figure.isdisjoint(marks):
        raise ValueError("exchanged roles collide")
    if not all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in figure | marks):
        raise ValueError("complete structures must fit before shear")
    out = np.full(shape, bg, dtype=np.int8)
    for r, c in figure:
        out[r, c] = edge
    for r, c in marks:
        out[r, c] = glyph
    return out


def _exchange(shape, info):
    marks = _stamp(info["ends"], info["motif"])
    if len(_components(marks)) != 2:
        raise ValueError("endpoint glyph copies must remain separate before shear")
    return _render(shape, info["bg"], info["edge"], info["glyph"],
                   _figure(info["centers"]), marks)


def _shear(a, bg, anchor=None, rate=1):
    rows, cols = np.nonzero(a != bg)
    if not len(rows):
        return a.copy()
    if anchor is None:
        anchor = int(rows.max())
    target = cols - rate * (anchor - rows)
    keep = (target >= 0) & (target < a.shape[1])
    out = np.full(a.shape, bg, dtype=np.int8)
    out[rows[keep], target[keep]] = a[rows[keep], cols[keep]]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer roles, exchange arrangements, then shear using the exchanged bottom."""
    a = _grid(input_grid)
    info = _analyze(a)
    return _shear(_exchange(a.shape, info), info["bg"])


def _orientation(ends):
    a, b = ends
    return "horizontal" if a[0] == b[0] else "vertical" if a[1] == b[1] else "diagonal"


def _motif(rng, indices=None):
    index = int(rng.choice(list(indices) if indices is not None else len(_PATTERNS)))
    a = np.array([[int(c) for c in row] for row in _PATTERNS[index]])
    a = np.rot90(a, int(rng.integers(4)))
    if rng.random() < 0.5:
        a = a[:, ::-1]
    return frozenset((int(r)-a.shape[0]//2, int(c)-a.shape[1]//2)
                     for r, c in np.argwhere(a))


def _asymmetric(motif):
    return motif != frozenset((-r, -c) for r, c in motif)


def _place(rng, relative, motif, height, width):
    rows = [r + dr for r, c in relative for dr, dc in motif]
    cols = [c + dc for r, c in relative for dr, dc in motif]
    if max(rows) - min(rows) >= height or max(cols) - min(cols) >= width:
        return None
    r = int(rng.integers(-min(rows), height-max(rows)))
    c = int(rng.integers(-min(cols), width-max(cols)))
    return tuple((rr+r, cc+c) for rr, cc in relative)


def _layout(rng, count):
    length = int(rng.integers(4, 11))
    if count == 2:
        dr, dc = _DIRECTIONS[int(rng.integers(4))]
        return ((0, 0), (dr*length, dc*length))
    if rng.random() < 0.5:
        points = ((0, 0), (0, length), (length, 0))
    else:
        points = ((0, 0), (length, length), (length, -length))
    turns, flip = int(rng.integers(4)), bool(rng.integers(2))
    out = []
    for r, c in points:
        if flip:
            c = -c
        for _ in range(turns):
            r, c = c, -r
        out.append((r, c))
    return tuple(out)


def _component_shear(mid, bg):
    out = np.full(mid.shape, bg, dtype=np.int8)
    for part in _components(frozenset(map(tuple, np.argwhere(mid != bg).tolist()))):
        bottom = max(r for r, c in part)
        for r, c in part:
            target = c - (bottom-r)
            if target >= 0:
                out[r, target] = mid[r, c]
    return out


def _compressed_row_shear(mid, bg):
    occupied = np.flatnonzero(np.any(mid != bg, axis=1))
    out = np.full(mid.shape, bg, dtype=np.int8)
    for index, r in enumerate(occupied):
        distance = len(occupied) - index - 1
        for c in np.flatnonzero(mid[r] != bg):
            if c >= distance:
                out[r, c-distance] = mid[r, c]
    return out


def _counterfactuals(a, info, mid):
    bg = info["bg"]
    colorwise = np.full(mid.shape, bg, dtype=np.int8)
    for color in (info["edge"], info["glyph"]):
        one = np.where(mid == color, color, bg)
        shifted = _shear(one, bg)
        colorwise[shifted == color] = color
    return {
        "omit_A": mid,
        "omit_B": _shear(a, bg),
        "input_anchor": _shear(mid, bg, int(np.nonzero(a != bg)[0].max())),
        "canvas_anchor": _shear(mid, bg, a.shape[0]-1),
        "separate_anchors": _component_shear(mid, bg),
        "color_anchors": colorwise,
        "occupied_row_count": _compressed_row_shear(mid, bg),
        "rightward": _shear(mid, bg, rate=-1),
        "double_rate": _shear(mid, bg, rate=2),
    }


def _make_pair(rng, count, orientation, motif, clipped, palette,
               color_anchor_witness=False):
    bg, edge, glyph = palette
    for _ in range(2400):
        h, w = int(rng.integers(12, 26)), int(rng.integers(12, 28))
        centers = _place(rng, _layout(rng, count), motif, h, w)
        length = int(rng.integers(4, 16))
        dr, dc = {"horizontal": (0, 1), "vertical": (1, 0),
                  "diagonal": (1, int(rng.choice((-1, 1))))}[orientation]
        ends = _place(rng, ((0, 0), (dr*length, dc*length)), motif, h, w)
        if centers is None or ends is None:
            continue
        source = _line(*ends)
        original = _stamp(centers, motif)
        marks, figure = _stamp(ends, motif), _figure(centers)
        if not _apart(source, original) or len(_components(original)) != count:
            continue
        if not _apart(figure, marks) or len(_components(marks)) != 2:
            continue
        union = source | original | marks | figure
        minr, minc = min(r for r, c in union), min(c for r, c in union)
        spanr = max(r for r, c in union) - minr + 1
        spanc = max(c for r, c in union) - minc + 1
        # Global horizontal placement controls clipping without distorting any
        # geometry. The shear's origin is the exchanged foreground, not union.
        bottom = max(r for r, c in marks | figure)
        required_left = max(bottom-r-(c-minc) for r, c in marks | figure)
        required_left = max(0, required_left)
        if clipped:
            if required_left == 0:
                continue
            high = min(required_left-1, 30-spanc)
            if high < 0:
                continue
            left = int(rng.integers(max(0, high-5), high+1))
        else:
            if required_left + spanc > 30:
                continue
            left = int(rng.integers(required_left, min(required_left+5, 30-spanc)+1))
        top = int(rng.integers(0, min(4, 29-spanr)+1))
        height = spanr + top + int(rng.integers(1, min(5, 30-spanr-top)+1))
        width = spanc + left + int(rng.integers(0, min(5, 30-spanc-left)+1))
        def move(points):
            return tuple((r-minr+top, c-minc+left) for r, c in points)
        centers, ends = move(centers), move(ends)
        source, original = _line(*ends), _stamp(centers, motif)
        figure, marks = _figure(centers), _stamp(ends, motif)
        a = _render((height, width), bg, edge, glyph, source, original)
        info = dict(bg=bg, edge=edge, glyph=glyph, ends=ends, centers=centers, motif=motif)
        mid = _render(a.shape, bg, edge, glyph, figure, marks)
        out = _shear(mid, bg)
        # Preserve informative traces of both roles and both stamped copies.
        if any(np.count_nonzero(out == color) < max(2, np.count_nonzero(mid == color)//2)
               for color in (edge, glyph)):
            continue
        anchor = max(r for r, c in figure | marks)
        if any(not any(c >= anchor-r for r, c in _stamp((end,), motif)) for end in ends):
            continue
        if count == 3 and any(
            not any(c >= anchor-r for r, c in _line(x, y) - {x, y})
            for x, y in combinations(centers, 2)
        ):
            # Each side must leave a non-vertex witness after clipping, so a
            # two-edge path can never masquerade as the complete triangle.
            continue
        if np.count_nonzero(a == bg) <= a.size//2:
            continue
        if np.array_equal(out, a):
            continue
        alts = _counterfactuals(a, info, mid)
        required = ("omit_A", "omit_B", "canvas_anchor", "separate_anchors",
                    "occupied_row_count", "rightward", "double_rate")
        if any(np.array_equal(out, alts[name]) for name in required):
            continue
        if color_anchor_witness and np.array_equal(out, alts["color_anchors"]):
            continue
        if len(motif) > 1 and np.array_equal(out, alts["input_anchor"]):
            continue
        # Generation ground truth is constructed from explicit sampled geometry;
        # the solver re-discovers it from pixels here as a construction check.
        if not np.array_equal(solve(a), out):
            raise AssertionError("input parsing disagrees with constructed geometry")
        return {"input": a, "output": out}
    raise RuntimeError("bounded geometric placement search exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    directions = list(rng.permutation(_ORIENTATIONS))
    feature = _motif(rng, (3, 4, 5, 6, 9))
    base = tuple(int(c) for c in rng.choice(10, 3, replace=False))
    variable_bg = bool(rng.integers(2))
    # Cross mandatory count, clipping, direction, and motif choices independently.
    counts, clips = [2, 3], [False, True]
    rng.shuffle(counts)
    rng.shuffle(clips)
    motifs = [feature, _motif(rng, (0, 1, 2, 7, 8, 10))]
    rng.shuffle(motifs)
    specs = [(counts[i], str(directions[i]), motifs[i], clips[i]) for i in range(2)]
    for _ in range(num_train-2):
        specs.append((int(rng.integers(2, 4)), str(rng.choice(directions[:2])),
                      _motif(rng), bool(rng.integers(2))))
    rng.shuffle(specs)
    for i in range(num_test):
        motif = feature if rng.random() < 0.55 else _motif(rng, (0, 1, 2, 7, 8, 10))
        orientation = str(directions[2] if i == 0 else rng.choice(directions))
        specs.append((int(rng.integers(2, 4)), orientation, motif, bool(rng.integers(2))))
    puzzle, seen = {"train": [], "test": []}, set()
    for i, (count, orientation, motif, clipped) in enumerate(specs):
        if i == 0 or rng.random() < 0.25:
            palette = base
        elif i == 1 or rng.random() < 0.4:
            palette = (base[0], base[2], base[1])
        else:
            bg = int(rng.integers(10)) if variable_bg else base[0]
            colors = rng.choice([c for c in range(10) if c != bg], 2, replace=False)
            palette = (bg, int(colors[0]), int(colors[1]))
        # Ensure the first two examples do not encode a fixed foreground-color key.
        if i == 1 and palette == base:
            palette = (base[0], base[2], base[1])
        for _ in range(12):
            pair = _make_pair(rng, count, orientation, motif, clipped, palette,
                              color_anchor_witness=(i == 0))
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in seen:
                seen.add(key)
                puzzle["train" if i < num_train else "test"].append(pair)
                break
        else:
            raise RuntimeError("bounded distinct-input search exhausted")
    return puzzle


def _facts(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    a, out = _grid(pair["input"]), _grid(pair["output"])
    assert a.shape == out.shape
    info = _analyze(a)
    bg, edge, glyph = info["bg"], info["edge"], info["glyph"]
    mid = _exchange(a.shape, info)
    assert _apart(_figure(info["centers"]), _stamp(info["ends"], info["motif"]))
    expected = _shear(mid, bg)
    assert np.array_equal(expected, out)
    rows, cols = np.nonzero(mid != bg)
    bottom = int(rows.max())
    assert np.array_equal(mid[bottom], out[bottom])
    lost = int(np.count_nonzero(mid != bg) - np.count_nonzero(out != bg))
    assert lost == sum(int(c < bottom-r) for r, c in zip(rows, cols))
    assert all(np.count_nonzero(out == color) >= max(2, np.count_nonzero(mid == color)//2)
               for color in (edge, glyph))
    assert all(any(c >= bottom-r for r, c in _stamp((end,), info["motif"]))
               for end in info["ends"])
    if len(info["centers"]) == 3:
        assert all(any(c >= bottom-r for r, c in _line(x, y) - {x, y})
                   for x, y in combinations(info["centers"], 2))
    alternatives = _counterfactuals(a, info, mid)
    defeated = {name for name, alt in alternatives.items() if not np.array_equal(alt, out)}
    assert {"omit_A", "omit_B", "canvas_anchor", "separate_anchors",
            "occupied_row_count", "rightward", "double_rate"} <= defeated
    if len(info["motif"]) > 1:
        assert "input_anchor" in defeated
    return dict(info, lost=lost, bottom=bottom, defeated=defeated,
                shape=a.shape, span=int(rows.max()-rows.min()+1))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, facts = set(), {}
    for split in ("train", "test"):
        facts[split] = []
        for pair in puzzle[split]:
            fact = _facts(pair)
            a = np.asarray(pair["input"])
            key = a.shape, a.tobytes()
            assert key not in seen
            seen.add(key)
            facts[split].append(fact)
    train = facts["train"]
    assert {len(f["centers"]) for f in train} == {2, 3}
    assert {f["lost"] > 0 for f in train} == {False, True}
    train_directions = {_orientation(f["ends"]) for f in train}
    assert len(train_directions) == 2
    assert _orientation(facts["test"][0]["ends"]) not in train_directions
    assert len({(f["bg"], f["edge"], f["glyph"]) for f in train}) >= 2
    assert any(_asymmetric(f["motif"]) for f in train)
    assert any("color_anchors" in f["defeated"] for f in train)
    assert all(f["motif"] in {g["motif"] for g in train}
               for f in facts["test"] if _asymmetric(f["motif"]))


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds, bounded memory, all explicit count choices, and ablation checks."""
    shapes, motifs, palettes, layouts, digests = set(), set(), set(), set(), set()
    directions = {s: set() for s in ("train", "test")}
    counts = {s: set() for s in ("train", "test")}
    clipping = {s: set() for s in ("train", "test")}
    anchor_roles = set()
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = _digest(puzzle)
        assert digest == _digest(generate_puzzle(seed))
        digests.add(digest)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _facts(pair)
                a, out = pair["input"], pair["output"]
                mid = _exchange(a.shape, f)
                # A literal cell loop checks the vectorized shear and clipping.
                literal = np.full(a.shape, f["bg"], dtype=np.int8)
                for r in range(a.shape[0]):
                    for c in range(a.shape[1]):
                        if mid[r, c] != f["bg"] and c-(f["bottom"]-r) >= 0:
                            literal[r, c-(f["bottom"]-r)] = mid[r, c]
                assert np.array_equal(literal, out)
                assert np.array_equal(solve(a.tolist()), out)
                shapes.add(a.shape)
                motifs.add(f["motif"])
                palettes.add((f["bg"], f["edge"], f["glyph"]))
                layout = (len(f["centers"]), _orientation(f["ends"]),
                          tuple((r-f["centers"][0][0], c-f["centers"][0][1])
                                for r, c in f["centers"]),
                          tuple((r-f["ends"][0][0], c-f["ends"][0][1]) for r, c in f["ends"]))
                layouts.add(layout)
                directions[split].add(_orientation(f["ends"]))
                counts[split].add(len(f["centers"]))
                clipping[split].add(f["lost"] > 0)
                figure_bottom = max(r for r, c in _figure(f["centers"]))
                glyph_bottom = max(r for r, c in _stamp(f["ends"], f["motif"]))
                anchor_roles.add("edge" if figure_bottom > glyph_bottom else
                                 "glyph" if glyph_bottom > figure_bottom else "tie")
                pairs += 1
    assert len(digests) == 42 and len(shapes) >= 35 and len(motifs) >= 15
    assert len(palettes) >= 30 and len(layouts) >= 100
    assert all(directions[s] == set(_ORIENTATIONS) for s in directions)
    assert all(counts[s] == {2, 3} and clipping[s] == {False, True} for s in counts)
    assert {"edge", "glyph"} <= anchor_roles
    legal_count_calls = 0
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(9000+a*10+b, a, b))
            legal_count_calls += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(digests),
                shapes=len(shapes), motifs=len(motifs), palettes=len(palettes),
                layouts=len(layouts), legal_count_calls=legal_count_calls,
                anchor_roles=sorted(anchor_roles))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
