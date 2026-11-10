"""Clockwise ring compaction followed by clipped empty-neighborhood marking.

One nine-point motif, one canvas, fixed highlight color 3. No filesystem access,
parent imports, learned conventions, or latent generation state are used by solve.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


HIGHLIGHT = 3


def _integer(value, name, low, high=None):
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
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid of colors 0..9")
    return a


def _ring(cy, cx, ry, rx):
    """UL, UR, LR, LL: clockwise starting at the upper left."""
    return ((cy - ry, cx - rx), (cy - ry, cx + rx),
            (cy + ry, cx + rx), (cy + ry, cx - rx))


def _destinations(cy, cx):
    return ((cy - 1, cx - 1), (cy - 1, cx), (cy - 1, cx + 1),
            (cy, cx + 1), (cy + 1, cx + 1), (cy + 1, cx),
            (cy + 1, cx - 1), (cy, cx - 1), (cy, cx))


def _inspect(input_grid):
    """Recover and verify all geometry solely from the observed nine points."""
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if counts.max() <= a.size // 2 or HIGHLIGHT in colors:
        raise ValueError("need a majority background and no input color 3")
    rr, cc = np.nonzero(a != bg)
    if len(rr) != 9:
        raise ValueError("need exactly nine foreground points")
    rows = sorted(set(int(r) for r in rr))
    cols = sorted(set(int(c) for c in cc))
    if len(rows) != 5 or len(cols) != 5:
        raise ValueError("need two concentric four-corner rings and a center")
    cy, cx = rows[2], cols[2]
    oy, ox, iy, ix = cy - rows[0], cx - cols[0], cy - rows[1], cx - cols[1]
    if not (0 < iy < oy and 0 < ix < ox):
        raise ValueError("rings must be strictly nested")
    if (rows != [cy - oy, cy - iy, cy, cy + iy, cy + oy]
            or cols != [cx - ox, cx - ix, cx, cx + ix, cx + ox]):
        raise ValueError("rings must have one common integer center")
    positions = _ring(cy, cx, oy, ox) + _ring(cy, cx, iy, ix) + ((cy, cx),)
    if set(zip(rr.tolist(), cc.tolist())) != set(positions):
        raise ValueError("foreground does not form the required motif")
    points = tuple(int(a[r, c]) for r, c in positions)
    geom = (oy, ox, iy, ix, cy - oy, a.shape[0] - 1 - cy - oy,
            cx - ox, a.shape[1] - 1 - cx - ox)
    return a, bg, (cy, cx), geom, points


def _empty_neighborhoods(a, bg):
    """Off-grid cells do not block a clipped 3-by-3 neighborhood."""
    h, w = a.shape
    padded = np.pad(a != bg, 1, constant_values=False)
    blocked = np.zeros((h, w), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            blocked |= padded[dr:dr + h, dc:dc + w]
    return ~blocked


def _compact(a, bg, center, points):
    cy, cx = center
    out = np.full(a.shape, bg, dtype=np.int8)
    order = (0, 4, 1, 5, 2, 6, 3, 7, 8)
    for dest, source in zip(_destinations(cy, cx), order):
        out[dest] = points[source]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, bg, center, _, points = _inspect(input_grid)
    out = _compact(a, bg, center, points)
    # Compute every decision simultaneously on the relocated foreground.
    mask = _empty_neighborhoods(out, bg)
    out[mask] = HIGHLIGHT
    return out


def _palette(rng, bg, mode):
    p = [int(c) for c in rng.permutation([c for c in range(10) if c not in (bg, 3)])]
    if mode == "distinct_ring":
        return tuple(p + [p[int(rng.integers(8))]])
    if mode == "diagonal":
        a, b, c, d, center = p[:5]
        return (a, b, a, b, c, d, c, d, center)
    if mode == "reuse":
        # Cross-ring equality and a repeated center, with contrasting points.
        return (p[0], p[1], p[2], p[3], p[0], p[4], p[1], p[5],
                p[int(rng.integers(4))])
    n = int(rng.integers(4, 9))
    points = p[:n] + [p[int(rng.integers(n))] for _ in range(9 - n)]
    return tuple(int(v) for v in rng.permutation(points))


def _geometry(rng, index, training):
    if training and index == 0:
        # A simple square with both a tight edge and a thick blank margin.
        margins = [0, int(rng.integers(1, 5)), int(rng.integers(5)), 4]
        margins = tuple(int(x) for x in np.roll(margins, int(rng.integers(4))))
        return (2, 2, 1, 1, *margins)
    if training and index == 1:
        oy = int(rng.integers(3, 7))
        ox = int(rng.choice([x for x in range(3, 7) if x != oy]))
        iy, ix = int(rng.integers(1, oy)), int(rng.integers(1, ox))
    elif training:
        oy = int(rng.integers(2, 7))
        ox = oy if rng.random() < .45 else int(rng.integers(2, 7))
        iy = int(rng.integers(1, oy))
        ix = iy if ox == oy and rng.random() < .5 else int(rng.integers(1, ox))
    else:
        oy = int(rng.integers(7, 11))
        ox = oy if rng.random() < .4 else int(rng.integers(3, 11))
        if rng.random() < .5:
            oy, ox = ox, oy
        iy, ix = int(rng.integers(1, oy)), int(rng.integers(1, ox))
    margins = []
    for radius in (oy, ox):
        budget = min(9, 30 - (2 * radius + 1))
        total = int(rng.integers(0, budget + 1))
        first = int(rng.integers(total + 1))
        margins.extend((first, total - first))
    if oy == ox == 2 and not any(margins):
        margins[1] = int(rng.integers(1, 5))
    return (oy, ox, iy, ix, *margins)


def _make_pair(geometry, bg, points):
    oy, ox, iy, ix, top, bottom, left, right = geometry
    cy, cx = top + oy, left + ox
    h, w = 2 * oy + 1 + top + bottom, 2 * ox + 1 + left + right
    inp = np.full((h, w), bg, dtype=np.int8)
    positions = _ring(cy, cx, oy, ox) + _ring(cy, cx, iy, ix) + ((cy, cx),)
    for position, color in zip(positions, points):
        inp[position] = color
    # Construct labels analytically, separately from solve's neighborhood scan.
    # A filled 3x3 block excludes precisely a 5x5 square from highlighting.
    out = np.full((h, w), HIGHLIGHT, dtype=np.int8)
    out[cy - 2:cy + 3, cx - 2:cx + 3] = bg
    outer, inner = points[:4], points[4:8]
    out[cy - 1:cy + 2, cx - 1:cx + 2] = (
        (outer[0], inner[0], outer[1]),
        (inner[3], points[8], inner[1]),
        (outer[3], inner[2], outer[2]))
    return {"input": inp, "output": out}


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen = set()
    backgrounds = [c for c in range(10) if c != HIGHLIGHT]
    first_bg = int(rng.choice(backgrounds))
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            training = split == "train"
            for _ in range(32):
                if training and index == 0:
                    bg, mode = first_bg, "distinct_ring"
                elif training and index == 1:
                    bg = int(rng.choice([c for c in backgrounds if c != first_bg]))
                    mode = "diagonal"
                else:
                    bg = int(rng.choice(backgrounds))
                    mode = ("distinct_ring", "diagonal", "reuse", "partial")[int(rng.integers(4))]
                pair = _make_pair(_geometry(rng, index, training), bg, _palette(rng, bg, mode))
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema, observable geometry, causal ablations and teaching cases."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, train_features = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, bg, center, geometry, points = _inspect(pair["input"])
            out = _grid(pair["output"])
            assert a.shape == out.shape
            assert _key(a) not in seen
            seen.add(_key(a))
            cy, cx = center
            compact = _compact(a, bg, center, points)
            # Direct distances to observed final foreground independently check
            # every cell, including diagonal fringes and canvas boundaries.
            rr, cc = np.indices(a.shape)
            distance = np.full(a.shape, 100, dtype=int)
            for r, c in np.argwhere(compact != bg):
                distance = np.minimum(distance, np.maximum(abs(rr - r), abs(cc - c)))
            highlight = distance >= 2
            assert np.array_equal(out == HIGHLIGHT, highlight)
            assert np.array_equal(out[~highlight], compact[~highlight])
            assert Counter(int(c) for c in a[a != bg]) == Counter(
                int(c) for c in out[(out != bg) & (out != HIGHLIGHT)])
            assert out[cy, cx] == a[cy, cx]
            assert np.sum((out != bg) & (out != HIGHLIGHT)) == 9
            assert np.any(highlight) and np.any(out == bg)
            assert np.any(highlight[0]) or np.any(highlight[-1])
            assert out[cy - 2, cx - 2] == bg  # diagonal neighbors block
            assert np.array_equal(solve(a), out)
            # Remove A: a ring-only output loses all highlight cells.
            assert not np.array_equal(compact, out)
            # Remove B: its original points obstruct different empty windows.
            raw_highlight = _empty_neighborhoods(a, bg)
            assert np.any(raw_highlight != highlight)
            no_b = a.copy()
            no_b[raw_highlight] = HIGHLIGHT
            assert not np.array_equal(no_b, out)
            # Applying A before moving the points also yields a wrong mask.
            assert np.any(highlight & ~raw_highlight)
            if split == "train":
                assert max(geometry[:2]) <= 6
                train_features.append((bg, geometry, points))
            else:
                assert max(geometry[:2]) >= 7
    bg0, g0, p0 = train_features[0]
    bg1, g1, p1 = train_features[1]
    assert g0[:4] == (2, 2, 1, 1) and len(set(p0[:8])) == 8
    assert bg0 != bg1 and g1[0] != g1[1]
    assert p1[8] not in p1[:8]  # center-copy role is separately visible


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; retain small summaries, never full multi-seed samples."""
    signatures, shapes, geometries, backgrounds, equalities = set(), set(), set(), set(), set()
    counts = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, bg, _, g, points = _inspect(pair["input"])
                shapes.add(a.shape)
                geometries.add(g)
                backgrounds.add(bg)
                normalized = {}
                equalities.add(tuple(normalized.setdefault(c, len(normalized)) for c in points))
                counts["pairs"] += 1
                counts["rectangular"] += g[0] != g[1] or g[2] != g[3]
                counts["wide_outer_gap"] += g[0] - g[2] > 1 or g[1] - g[3] > 1
                counts["wide_inner_gap"] += g[2] > 1 or g[3] > 1
                counts["uneven_margins"] += len(set(g[4:])) > 1
                counts["full_axis"] += 30 in a.shape
                counts["cross_ring_reuse"] += bool(set(points[:4]) & set(points[4:8]))
        # A color relabeling that fixes the family highlight must commute with
        # the entire transformation; this also exercises zero as foreground.
        palette = [c for c in range(10) if c != HIGHLIGHT]
        relabel = np.arange(10, dtype=np.int8)
        relabel[palette] = np.random.default_rng(seed + 1000).permutation(palette)
        pair = puzzle["test"][0]
        assert np.array_equal(solve(relabel[pair["input"]]), relabel[pair["output"]])
    assert len(signatures) == 42 and len(shapes) >= 60 and len(geometries) >= 160
    assert backgrounds == set(range(10)) - {HIGHLIGHT} and len(equalities) >= 25
    assert min(counts[k] for k in ("rectangular", "wide_outer_gap", "wide_inner_gap",
                                   "uneven_margins", "cross_ring_reuse")) >= 20
    assert counts["full_axis"] >= 1
    # Every legal explicit count combination, streamed one puzzle at a time.
    legal_counts = 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(10000 + 10 * nt + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
            legal_counts += 1
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {nt}/{nq}")
    # Valid family members beyond the sampled radius range, including a tight
    # 5x5 canvas with no highlight and a 30x30 almost-full-span motif.
    for geometry in ((2, 2, 1, 1, 0, 0, 0, 0), (14, 14, 13, 1, 0, 1, 1, 0)):
        pair = _make_pair(geometry, 0, (1, 2, 4, 5, 6, 7, 8, 9, 6))
        before = pair["input"].copy()
        assert np.array_equal(solve(before), pair["output"])
        assert np.array_equal(before, pair["input"])
    return {"seeds": 42, "pairs": counts["pairs"], "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "geometries": len(geometries),
            "backgrounds": len(backgrounds), "point_equality_patterns": len(equalities),
            "legal_count_combinations": legal_counts, "features": dict(counts),
            "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
