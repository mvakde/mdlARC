"""Mirror loom: reflected oscillating beams form knots at color intersections.

Only NumPy and the standard library are required.  The input frame identifies
the knot color; its empty interior identifies the background.  No puzzle-level
parameters are needed by solve().
"""

import json
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(7 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 7..30 rectangle of ARC color integers")
    return a


def _parse(value):
    a = _grid(value)
    h, w = a.shape
    frame, background = int(a[0, 0]), int(a[1, 1])
    if frame == background or not np.all(a[1:-1, 1:-1] == background):
        raise ValueError("the input must have a uniform empty interior")
    if any(int(a[r, c]) != frame for r, c in ((0, w-1), (h-1, 0), (h-1, w-1))):
        raise ValueError("the four corners must identify one frame color")
    observed, tracks = {}, {}
    for axis, length in ((0, w), (1, h)):
        for p in range(1, length - 1):
            locations = ((0, p), (h-1, p)) if axis == 0 else ((p, 0), (p, w-1))
            for r, c in locations:
                color = int(a[r, c])
                if color == background:
                    raise ValueError("background cannot replace the frame")
                if color == frame:
                    continue
                key = (axis, min(p, length - 1 - p))
                if key in tracks and tracks[key] != color:
                    raise ValueError("one reflected source orbit has conflicting colors")
                tracks[key] = color
                observed[r, c] = color
    if {axis for axis, _ in tracks} != {0, 1}:
        raise ValueError("both vertical and horizontal emitters are required")
    if len(set(observed.values())) < 2:
        raise ValueError("at least two thread colors are required")
    return a, background, frame, observed, tracks


def _reflect_sources(observed, shape, rows=True, columns=True):
    h, w = shape
    sources = {}
    for (r, c), color in observed.items():
        for rr in ({r, h-1-r} if rows else {r}):
            for cc in ({c, w-1-c} if columns else {c}):
                if (rr, cc) in sources and sources[rr, cc] != color:
                    raise ValueError("incompatible reflected sources")
                sources[rr, cc] = color
    return sources


def _render(a, background, frame, sources, *, oscillate=True,
            halfway=False, count_copies=False):
    """Placement-based rendering; counterfactual options are for diagnostics."""
    h, w = a.shape
    hits = np.zeros((10, h, w), dtype=bool)
    copies = np.zeros((h, w), dtype=np.int16)
    for (r, c), color in sources.items():
        vertical = r in (0, h-1)
        inward = h if vertical else w
        lateral = w if vertical else h
        p = c if vertical else r
        for distance in range(1, inward - 1):
            if halfway and distance > (inward - 1) / 2:
                continue
            offsets = (-1, 1) if oscillate and distance % 2 else (0,)
            for offset in offsets:
                q = p + offset
                if not 1 <= q < lateral - 1:
                    continue  # a beam never paints the frame
                if vertical:
                    rr, cc = (distance if r == 0 else h-1-distance), q
                else:
                    rr, cc = q, (distance if c == 0 else w-1-distance)
                hits[color, rr, cc] = True
                copies[rr, cc] += 1
    counts = hits.sum(axis=0)
    out = np.full(a.shape, background, dtype=np.int8)
    single = counts == 1
    out[single] = hits.argmax(axis=0)[single]
    out[(copies if count_copies else counts) >= 2] = frame
    out[[0, h-1], :] = frame
    out[:, [0, w-1]] = frame
    for (r, c), color in sources.items():
        out[r, c] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete mirror-related sources and draw their intersecting waves."""
    a, bg, frame, observed, _ = _parse(input_grid)
    sources = _reflect_sources(observed, a.shape)
    return _render(a, bg, frame, sources)


def _ablation(a, mode):
    a, bg, frame, observed, _ = _parse(a)
    sources = _reflect_sources(observed, a.shape,
                               rows=mode not in ("no_B", "columns_only"),
                               columns=mode not in ("no_B", "rows_only"))
    return _render(a, bg, frame, sources, oscillate=mode != "no_A",
                   halfway=mode == "halfway", count_copies=mode == "all_overlap")


def _formula(a):
    """Cellwise audit using lane distances, separately from beam placement.

    A vertical lane covers a cell iff its transverse distance is equal to
    the parity of distance from at least one of the two emitting edges.
    Reflection also supplies the transverse mirror lane.  Transpose for
    horizontal lanes.  Taking a union per color makes same-color overlap inert.
    """
    a, bg, frame, _, tracks = _parse(a)
    h, w = a.shape
    rr, cc = np.indices(a.shape)
    masks = np.zeros((10, h, w), dtype=bool)
    border = np.full(a.shape, frame, dtype=np.int8)
    for (axis, p), color in tracks.items():
        extent = w if axis == 0 else h
        for q in {p, extent-1-p}:
            if axis == 0:
                transverse = np.abs(cc - q)
                masks[color] |= ((transverse == rr % 2)
                                 | (transverse == (h-1-rr) % 2))
                border[0, q] = border[h-1, q] = color
            else:
                transverse = np.abs(rr - q)
                masks[color] |= ((transverse == cc % 2)
                                 | (transverse == (w-1-cc) % 2))
                border[q, 0] = border[q, w-1] = color
    counts = masks.sum(axis=0)
    expected = np.full(a.shape, bg, dtype=np.int8)
    singles = counts == 1
    expected[singles] = masks.argmax(axis=0)[singles]
    expected[counts >= 2] = frame
    expected[[0, h-1], :] = border[[0, h-1], :]
    expected[:, [0, w-1]] = border[:, [0, w-1]]
    return expected, counts


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(int(v) for v in a.flat)


def _counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10),
                                (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _length(rng, parity, band=None):
    low, high = band or (7, 30)
    return int(rng.choice([n for n in range(low, high+1) if n % 2 == parity]))


def _place_orbit(a, axis, p, color, rng, mode):
    h, w = a.shape
    positions = ([(0, p), (0, w-1-p), (h-1, p), (h-1, w-1-p)]
                 if axis == 0 else
                 [(p, 0), (h-1-p, 0), (p, w-1), (h-1-p, w-1)])
    positions = list(dict.fromkeys(positions))
    if mode == "corner":
        # Rotate later: the two active sides here meet at the top-left corner.
        side = positions[:len(positions)//2]
        chosen = [side[int(rng.integers(len(side)))]]
    else:
        lo = 2 if mode == "partial" else 1
        hi = len(positions) if mode == "partial" else min(3, len(positions))
        count = int(rng.integers(min(lo, hi), hi + 1))
        chosen = [positions[int(i)] for i in rng.choice(len(positions), count, replace=False)]
    for r, c in chosen:
        a[r, c] = color


def _sample_pair(rng, parity, palette, witness=False, is_test=False):
    # The first sparse witness stays readable; subsequent examples vary size
    # independently of density and may be shorter, wider, or taller.
    band = (9, 17) if witness else None
    h, w = (_length(rng, parity[0], band), _length(rng, parity[1], band))
    bg, frame = palette[:2]
    colors = palette[2:]
    a = np.full((h, w), bg, dtype=np.int8)
    a[[0, h-1], :] = frame
    a[:, [0, w-1]] = frame
    mode = "corner" if witness else str(rng.choice(("corner", "scattered", "partial"),
                                                   p=(0.40, 0.40, 0.20)))
    lane_positions = []
    for axis, extent in ((0, w), (1, h)):
        # Each p is one orbit, not one distinct color. The center lane is legal.
        available = list(range(1, (extent-1)//2 + 1))
        maximum = min(len(available), 5 if is_test else 3)
        count = 1 if witness else int(rng.integers(1, maximum+1))
        style = str(rng.choice(("cluster", "spread", "random")))
        if witness:
            # Noncentral and well-separated from its reflection. Matching
            # parities on the two source edges guarantee visible color knots.
            available = list(range(1, (extent-1)//2 - 1))
            if axis == 1:
                matching = [p for p in available if p % 2 == lane_positions[0][0] % 2]
                available = matching or available
            positions = [int(rng.choice(available))]
        elif style == "cluster":
            start = int(rng.integers(0, len(available)-count+1))
            positions = available[start:start+count]
        elif style == "spread":
            indexes = np.linspace(0, len(available)-1, count, dtype=int)
            positions = [available[int(i)] for i in indexes]
        else:
            positions = sorted(int(p) for p in rng.choice(available, count, replace=False))
        lane_positions.append(positions)
        for index, p in enumerate(positions):
            # The first lane of each orientation supplies different colors;
            # other lanes may repeat either color or introduce more colors.
            color = colors[axis] if index == 0 else int(rng.choice(colors))
            _place_orbit(a, axis, p, color, rng, mode)
    turns = int(rng.integers(4))
    return np.rot90(a, turns).copy()


def _informative(a):
    expected, counts = _formula(a)
    interior = (slice(1, -1), slice(1, -1))
    _, bg, _, observed, _ = _parse(a)
    # Both parent mechanisms must affect interior cells, not just source beads.
    return (np.any(counts[interior] >= 2)
            and np.any((counts[interior] == 1))
            and np.any(expected[interior] == bg)
            and all(not np.array_equal(expected[interior], _ablation(a, mode)[interior])
                    for mode in ("no_A", "no_B"))
            and len(observed) < len(_reflect_sources(observed, a.shape)))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    fixed_palette = bool(rng.random() < 0.35)
    shared = [int(c) for c in rng.permutation(10)[:int(rng.integers(4, 8))]]
    # Anchor demonstrations establish an oscillating room, then the effect of
    # opposite phases on one even axis. Test combines evenness on both axes.
    parities = [(1, 1), (0, 1) if rng.random() < .5 else (1, 0)]
    parities += [((1, 1), (0, 1), (1, 0))[int(rng.integers(3))]
                 for _ in range(num_train-2)]
    parities += [(0, 0)]
    parities += [tuple(int(p) for p in rng.integers(0, 2, size=2))
                 for _ in range(num_test-1)]
    used = set()
    puzzle = {"train": [], "test": []}
    for i, parity in enumerate(parities):
        split = "train" if i < num_train else "test"
        palette = shared if fixed_palette else [int(c) for c in rng.permutation(10)[:int(rng.integers(4, 8))]]
        for _ in range(256):
            a = _sample_pair(rng, parity, palette, witness=i == 0, is_test=split == "test")
            if _key(a) in used or not _informative(a):
                continue
            out = solve(a)
            if i == 0 and any(np.array_equal(out[1:-1, 1:-1], _ablation(a, mode)[1:-1, 1:-1])
                              for mode in ("rows_only", "columns_only", "all_overlap")):
                continue
            if i == 1 and np.array_equal(out, _ablation(a, "halfway")):
                continue
            break
        else:
            raise RuntimeError("bounded sampling exhausted while finding an informative room")
        used.add(_key(a))
        puzzle[split].append({"input": a, "output": out})
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape
            assert _key(a) not in seen
            seen.add(_key(a))
            expected, hits = _formula(a)
            assert np.array_equal(out, expected), "beam/color-set formula disagrees"
            assert np.array_equal(out, out[::-1, :])
            assert np.array_equal(out, out[:, ::-1])
            _, bg, frame, observed, _ = _parse(a)
            assert all(out[r, c] == color for (r, c), color in observed.items())
            assert set(np.unique(out)) == set(np.unique(a))
            assert _informative(a), "each sampled room must exercise both parents"
            assert np.all((out[1:-1, 1:-1] == frame) == (hits[1:-1, 1:-1] >= 2))
            if split == "train":
                train.append((a, out))
    # These are evidence checks for generated puzzles, not solve restrictions.
    assert any(a.shape[0] % 2 and a.shape[1] % 2 for a, _ in train)
    assert any(a.shape[0] % 2 != a.shape[1] % 2 for a, _ in train)
    for mode in ("rows_only", "columns_only", "all_overlap", "halfway"):
        assert any(not np.array_equal(out[1:-1, 1:-1], _ablation(a, mode)[1:-1, 1:-1])
                   for a, out in train), f"training fails to distinguish {mode}"
    assert all(d % 2 == 0 for d in np.asarray(puzzle["test"][0]["input"]).shape)


def _signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded count, semantic, mutation, determinism, and 42-seed checks."""
    # Hand-computable 11x11 room: source color 2 at top column 2, color 3
    # at left row 2; their mirror lanes are 8. The five-cell intersections
    # are knots, while long isolated segments keep their own colors.
    a = np.zeros((11, 11), dtype=np.int8)
    a[[0, -1], :] = 8
    a[:, [0, -1]] = 8
    a[0, 2], a[2, 0] = 2, 3
    out = solve(a)
    assert out[2, 2] == 8 and out[1, 1] == 8 and out[3, 3] == 8
    assert out[4, 2] == 2 and out[2, 4] == 3 and out[4, 4] == 0
    assert out[10, 8] == 2 and out[8, 10] == 3
    assert np.array_equal(out, _formula(a)[0])
    # Opposite transverse parities on an odd room can miss each other.
    b = a.copy()
    b[2, 0], b[3, 0] = 8, 3
    assert not np.any(solve(b)[1:-1, 1:-1] == 8)
    # Existing same-colored reflected mates add no new effect.
    b = a.copy()
    b[10, 8] = 2
    assert np.array_equal(solve(a), solve(b))
    bad = a.copy()
    bad[10, 8] = 4
    try:
        solve(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting mirror orbit was accepted")

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_signature(p) for p in puzzles}) == 42
    assert _signature(puzzles[0]) == _signature(generate_puzzle(0))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000 + 10*nt + ne, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {nt}/{ne}")
    shapes, normalized, backgrounds, frames = set(), set(), set(), set()
    orbit_counts, source_counts, active_sides, palettes = set(), set(), set(), set()
    pair_total = repeated = boundary = sparse = 0
    for p in puzzles:
        validate_generated_puzzle(p)
        for pair in p["train"] + p["test"]:
            a, bg, frame, observed, tracks = _parse(pair["input"])
            h, w = a.shape
            pair_total += 1
            shapes.add(a.shape)
            backgrounds.add(bg)
            frames.add(frame)
            palettes.add(tuple(sorted(int(c) for c in np.unique(a))))
            orbit_counts.add(len(tracks))
            source_counts.add(len(observed))
            active_sides.add(sum((any(r == 0 for r, c in observed),
                                  any(r == h-1 for r, c in observed),
                                  any(c == 0 for r, c in observed),
                                  any(c == w-1 for r, c in observed))))
            role_map = {bg: 0, frame: 1}
            normalized.add((a.shape, tuple(role_map.setdefault(int(c), len(role_map)) for c in a.flat)))
            repeated += len(set(tracks.values())) < len(tracks)
            boundary += any(p == 1 for axis, p in tracks)
            sparse += len(tracks) == 2
            assert np.array_equal(solve(a), pair["output"])
    assert len(shapes) >= 60 and len(normalized) >= 180
    assert backgrounds == set(range(10)) and frames == set(range(10))
    assert len(palettes) >= 30 and len(orbit_counts) >= 6
    assert len(source_counts) >= 10 and active_sides == {2, 3, 4}
    assert repeated >= 60 and boundary >= 60 and sparse >= 42
    assert any(h == 30 or w == 30 for h, w in shapes)
    assert any(h == 7 or w == 7 for h, w in shapes)
    assert any(h >= 2*w for h, w in shapes) and any(w >= 2*h for h, w in shapes)
    # A damaged answer must fail the semantic validator.
    damaged = {s: [dict(pair) for pair in puzzles[0][s]] for s in ("train", "test")}
    damaged["train"][0]["output"] = damaged["train"][0]["output"].copy()
    damaged["train"][0]["output"][1, 1] = (int(damaged["train"][0]["output"][1, 1])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted answer passed validation")
    return {"seeds": 42, "pairs": pair_total, "shapes": len(shapes),
            "normalized_inputs": len(normalized), "palettes": len(palettes),
            "backgrounds": len(backgrounds), "frame_colors": len(frames),
            "orbit_counts": sorted(orbit_counts), "source_counts": sorted(source_counts),
            "active_side_counts": sorted(active_sides), "repeated_color_pairs": repeated,
            "near_frame_pairs": boundary, "sparse_pairs": sparse}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
