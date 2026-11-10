"""Reflected rays define the foreground for a simultaneous empty-window test.

Only Python's standard library and NumPy are used. Nothing is loaded from the
parents, from saved puzzles, or from generation state by the input-only solver.
The highlight literal is always 3; background is the input's unique majority.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


HIGHLIGHT = 3
DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
CHECK_SEEDS = (0, 832790071, 675734216, 531693745, 951250761, 810475753,
               143121795, 743125475, 153238314, 830524435, 575554054,
               575791732, 717750606, 526472272, 260039395, 425549014,
               758010272, 878055679, 358066377, 554018461, 390481660,
               239298244, 753903103, 1919484, 326519326, 942530531,
               382418987, 314945147, 775294821, 640449142, 636700313,
               734335172, 156033782, 916071694, 179934430, 868606248,
               285353954, 541546177, 529828281, 211011009, 747449532,
               652809246)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(n < 1 or n > 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _background(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] * 2 <= a.size or bg == HIGHLIGHT:
        raise ValueError("background must be a strict majority and cannot be 3")
    if np.any(a == HIGHLIGHT):
        raise ValueError("3 is reserved for output highlights")
    return bg


def _components(mask):
    """Deterministic four-connected components, including singleton pockets."""
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(tuple(sorted(cells)))
    return result


def _parse(a):
    bg = _background(a)
    sources, mirrors = [], []
    for color in sorted(int(v) for v in np.unique(a) if v != bg):
        for cells in _components(a == color):
            rows, cols = zip(*cells)
            r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
            if len(cells) == 3 and r1-r0 == c1-c0 == 1:
                missing = next(p for p in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
                               if p not in cells)
                elbow = (r0+r1-missing[0], c0+c1-missing[1])
                direction = (elbow[0]-missing[0], elbow[1]-missing[1])
                sources.append(dict(cells=cells, color=color, elbow=elbow,
                                    direction=direction))
            elif (len(cells) % 2 == 1 and (r0 == r1 or c0 == c1)
                  and len(cells) == max(r1-r0, c1-c0)+1):
                axes = (0, 1) if len(cells) == 1 else ((0,) if r0 == r1 else (1,))
                mirrors.append(dict(cells=cells, color=color, axes=axes,
                                    midpoint=((r0+r1)//2, (c0+c1)//2)))
            else:
                raise ValueError("each monochromatic component must be an L or an odd straight mirror")
    if not sources:
        raise ValueError("at least one L source is required")
    sources.sort(key=lambda s: s["elbow"])
    mirrors.sort(key=lambda m: m["midpoint"])
    return bg, sources, mirrors


def _trace(a):
    """Trace through immutable input objects; rays never act as new mirrors."""
    bg, sources, mirrors = _parse(a)
    h, w = a.shape
    events = {}
    for index, mirror in enumerate(mirrors):
        for axis in mirror["axes"]:
            for sign in (-1, 1):
                p = list(mirror["midpoint"])
                p[axis] -= sign
                events.setdefault(tuple(p), []).append((axis, sign, index))
    completed = a.copy()
    painted, used, routes = {}, set(), []
    for source in sources:
        dr, dc = source["direction"]
        r, c = source["elbow"]
        r, c = r+dr, c+dc
        color = source["color"]
        visited, path, contacts = set(), [], []
        while 0 <= r < h and 0 <= c < w:
            state = (r, c, dr, dc)
            if state in visited:
                raise ValueError("nonterminating ray")
            visited.add(state)
            if a[r, c] != bg:
                raise ValueError("ray strikes an object rather than a midpoint face")
            hits = [(axis, index) for axis, sign, index in events.get((r, c), ())
                    if (dr, dc)[axis] == sign]
            if len(hits) > 1:
                raise ValueError("ambiguous midpoint contact")
            if hits:
                axis, index = hits[0]
                old_color = color
                color = mirrors[index]["color"]
                contacts.append((index, axis, (r, c), old_color, color))
                used.add(index)
                dr, dc = (-dr, dc) if axis == 0 else (dr, -dc)
            if (r, c) in painted and painted[r, c] != color:
                raise ValueError("different-colored rays demand the same cell")
            painted[r, c] = color
            completed[r, c] = color
            path.append((r, c))
            r, c = r+dr, c+dc
        routes.append(dict(elbow=source["elbow"], path=tuple(path),
                           contacts=tuple(contacts), final_direction=(dr, dc)))
    if used != set(range(len(mirrors))):
        raise ValueError("every mirror must be reached by a ray")
    return completed, dict(background=bg, sources=sources, mirrors=mirrors, routes=routes)


def _centers(foreground):
    """False padding implements clipping: outside the grid is not an obstacle."""
    h, w = foreground.shape
    padded = np.pad(foreground, 1, constant_values=False)
    near = np.zeros((h, w), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            near |= padded[dy:dy+h, dx:dx+w]
    return ~near


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    completed, info = _trace(a)
    # Freeze the full foreground before recoloring any centers. The new 3s
    # never enter the neighborhood predicate and cannot inhibit each other.
    mask = _centers(completed != info["background"])
    answer = completed.copy()
    answer[mask] = HIGHLIGHT
    return answer


def _source_cells(elbow, direction):
    r, c = elbow
    dr, dc = direction
    return {(r, c), (r-dr, c), (r, c-dc)}


def _mirror_cells(midpoint, axis, length):
    r, c = midpoint
    return {(r, c+k) if axis == 0 else (r+k, c)
            for k in range(-(length//2), length//2+1)}


def _place(a, cells, color, bg):
    h, w = a.shape
    if any(not (0 <= r < h and 0 <= c < w) or a[r, c] != bg for r, c in cells):
        return None
    for r, c in cells:
        for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if (p not in cells and 0 <= p[0] < h and 0 <= p[1] < w
                    and a[p] == color):
                return None
    trial = a.copy()
    for p in cells:
        trial[p] = color
    return trial


def _launch(rng, a, palette, direction=None):
    bg = _background(a)
    h, w = a.shape
    for _ in range(48):
        dr, dc = direction if direction is not None else DIRECTIONS[int(rng.integers(4))]
        r, c = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
        if rng.random() < .45:
            if rng.integers(2):
                r = 1 if dr > 0 else h-2
            else:
                c = 1 if dc > 0 else w-2
        if not (0 <= r+3*dr < h and 0 <= c+3*dc < w):
            continue
        trial = _place(a, _source_cells((r, c), (dr, dc)), int(rng.choice(palette)), bg)
        if trial is None:
            continue
        try:
            _trace(trial)
        except ValueError:
            continue
        return trial, (r, c)
    return None


def _extend(rng, a, elbow, palette, axis=None, length=None, color_mode=None):
    _, before = _trace(a)
    route = next(r for r in before["routes"] if r["elbow"] == elbow)
    source = next(s for s in before["sources"] if s["elbow"] == elbow)
    dr, dc = route["final_direction"]
    anchor = route["contacts"][-1][2] if route["contacts"] else elbow
    incoming = route["contacts"][-1][4] if route["contacts"] else source["color"]
    h, w = a.shape
    distance = min(h-1-anchor[0] if dr > 0 else anchor[0],
                   w-1-anchor[1] if dc > 0 else anchor[1])
    if distance < 2:
        return None
    options = [(t, ax) for t in range(2, distance+1)
               for ax in ((axis,) if axis is not None else (0, 1))]
    rng.shuffle(options)
    for t, ax in options[:48]:
        p = (anchor[0]+t*dr, anchor[1]+t*dc)
        reflected = (-dr, dc) if ax == 0 else (dr, -dc)
        if not (0 <= p[0]+2*reflected[0] < h and 0 <= p[1]+2*reflected[1] < w):
            continue
        mid = (p[0]+(dr if ax == 0 else 0), p[1]+(dc if ax == 1 else 0))
        n = int(rng.choice([1, 3, 5, 7])) if length is None else length
        choices = list(palette)
        if color_mode == "same":
            color = incoming
        else:
            if color_mode == "change":
                choices = [c for c in choices if c != incoming]
            if not choices:
                continue
            color = int(rng.choice(choices))
        trial = _place(a, _mirror_cells(mid, ax, n), color, before["background"])
        if trial is None:
            continue
        try:
            _, after = _trace(trial)
        except ValueError:
            continue
        new = next(r for r in after["routes"] if r["elbow"] == elbow)
        old_events = [(before["mirrors"][i]["midpoint"], k, p0)
                      for i, k, p0, _, _ in route["contacts"]]
        new_events = [(after["mirrors"][i]["midpoint"], k, p0)
                      for i, k, p0, _, _ in new["contacts"]]
        if new_events == old_events + [(mid, ax, p)]:
            return trial
    return None


def _palette(rng, bg):
    available = [v for v in range(10) if v not in (bg, HIGHLIGHT)]
    rng.shuffle(available)
    return available[:int(rng.integers(2, 7))]


def _parallel_demo(rng, bg, axis):
    """A movable/rotatable diagnostic corridor, not a different rule.

    Parallel rays six columns apart leave diagonal single-cell empty centers.
    This directly refutes the inherited B-v2 size threshold and a four-neighbor
    test. One ray also undergoes a changed-color bar reflection.
    """
    h, w = int(rng.integers(16, 29)), int(rng.integers(18, 31))
    a = np.full((h, w), bg, dtype=np.int8)
    palette = _palette(rng, bg)
    first, second = palette[:2]
    r, c = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    flight = int(rng.integers(5, min(12, h-r-2, w-c-3)+1))
    length = int(rng.choice([3, 5, 7]))
    # Both incoming rays run down-right. At their later actual shared cell,
    # the reflected first ray and the second ray have the same color.
    for cells, color in ((_source_cells((r, c), (1, 1)), first),
                         (_source_cells((r, c+6), (1, 1)), second),
                         (_mirror_cells((r+flight+1, c+flight), 0, length), second)):
        a = _place(a, cells, color, bg)
        if a is None:
            return None
    if axis == 1:
        a = a.T.copy()
    if rng.integers(2):
        a = a[::-1].copy()
    if rng.integers(2):
        a = a[:, ::-1].copy()
    try:
        _trace(a)
    except ValueError:
        return None
    return a


def _random_scene(rng, bg, teaching_axis=None, test=False):
    h, w = (int(rng.integers(14 if teaching_axis is not None else 10, 31))
            for _ in range(2))
    a = np.full((h, w), bg, dtype=np.int8)
    palette = _palette(rng, bg)
    first = _launch(rng, a, palette)
    if first is None:
        return None
    a, elbow = first
    if teaching_axis is not None:
        a = _extend(rng, a, elbow, palette, teaching_axis,
                    int(rng.choice([3, 5, 7])), "same")
        if a is None:
            return None
        a = _extend(rng, a, elbow, palette, length=1, color_mode="change")
        if a is None:
            return None
    else:
        bounces = int(rng.integers(2, 5)) if test else int(rng.integers(0, 5))
        for _ in range(bounces):
            trial = _extend(rng, a, elbow, palette)
            if trial is None:
                if test and _max_bounces(a) < 2:
                    return None
                break
            a = trial
    target = int(rng.integers(2, 7)) if test else int(rng.integers(1, 6))
    for _ in range(target-1):
        launch = _launch(rng, a, palette)
        if launch is None:
            break
        a, extra = launch
        for _ in range(int(rng.integers(0, 3))):
            if rng.random() < .45:
                break
            trial = _extend(rng, a, extra, palette)
            if trial is None:
                break
            a = trial
    if test:
        _, info = _trace(a)
        if len(info["sources"]) < 2 or _max_bounces(a) < 2:
            return None
    return a


def _max_bounces(a):
    return max(len(r["contacts"]) for r in _trace(a)[1]["routes"])


def _features(a):
    completed, info = _trace(a)
    bg = info["background"]
    mask = _centers(completed != bg)
    original_mask = _centers(a != bg)
    lost = original_mask & ~mask & (completed == bg)
    f = completed != bg
    pad = np.pad(f, 1, constant_values=False)
    h, w = a.shape
    axial_clear = ~(f | pad[:h, 1:w+1] | pad[2:h+2, 1:w+1]
                    | pad[1:h+1, :w] | pad[1:h+1, 2:w+2])
    contacts = [c for r in info["routes"] for c in r["contacts"]]
    sizes = [len(c) for c in _components(mask)]
    return dict(shape=a.shape, background=bg, source_count=len(info["sources"]),
                directions=tuple(s["direction"] for s in info["sources"]),
                mirror_lengths=tuple(len(m["cells"]) for m in info["mirrors"]),
                bar_axes=tuple(sorted({axis for index, axis, _, _, _ in contacts
                                       if len(info["mirrors"][index]["cells"]) > 1})),
                axes=tuple(c[1] for c in contacts),
                changed_contacts=sum(c[3] != c[4] for c in contacts),
                unchanged_contacts=sum(c[3] == c[4] for c in contacts),
                bounces=tuple(len(r["contacts"]) for r in info["routes"]),
                ray_cells=int(np.count_nonzero(completed != a)),
                lost_centers=int(lost.sum()), highlight_cells=int(mask.sum()),
                diagonal_fringe=int(np.count_nonzero(axial_clear & ~mask)),
                edge_centers=int(mask[0].sum()+mask[-1].sum()+mask[:, 0].sum()+mask[:, -1].sum()),
                small_components=sum(n < 5 for n in sizes),
                highlight_components=len(sizes),
                palette=tuple(int(v) for v in np.unique(a) if v != bg))


def _usable(a):
    f = _features(a)
    return (f["ray_cells"] >= 4 and f["lost_centers"] >= 2
            and f["highlight_cells"] >= 5 and f["diagonal_fringe"] > 0
            and f["edge_centers"] > 0)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must lie in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    backgrounds = [v for v in range(10) if v != HIGHLIGHT]
    rng.shuffle(backgrounds)
    axis = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    first_shape, first_directions = None, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(128):
                if split == "train" and index == 0:
                    a = _parallel_demo(rng, backgrounds[0], axis)
                else:
                    bg = backgrounds[1] if split == "train" and index == 1 else int(rng.choice(backgrounds))
                    a = _random_scene(rng, bg,
                                      1-axis if split == "train" and index == 1 else None,
                                      test=split == "test")
                if a is None or _key(a) in seen or not _usable(a):
                    continue
                f = _features(a)
                if split == "train" and index == 0 and not f["small_components"]:
                    continue
                if split == "train" and index == 1:
                    if a.shape == first_shape or len(first_directions | set(f["directions"])) < 2:
                        continue
                break
            else:
                raise RuntimeError("bounded scene construction exhausted")
            seen.add(_key(a))
            puzzle[split].append({"input": a, "output": solve(a)})
            if split == "train" and index == 0:
                first_shape, first_directions = a.shape, set(f["directions"])
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema, exact semantics, and evidence for both causal mechanisms."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= tr <= 10 and 1 <= te <= 4 and tr > te
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape and _key(a) not in seen
            seen.add(_key(a))
            completed, info = _trace(a)
            bg = info["background"]
            # Deliberately use literal clipped slices here rather than the
            # solver's padded mask. This checks every selected/unselected cell.
            expected = completed.copy()
            h, w = a.shape
            for r in range(h):
                for c in range(w):
                    if np.all(completed[max(0, r-1):min(h, r+2),
                                        max(0, c-1):min(w, c+2)] == bg):
                        expected[r, c] = HIGHLIGHT
            assert np.array_equal(out, expected), "wrong ray or neighborhood output"
            assert np.array_equal(out[a != bg], a[a != bg]), "changed an original object"
            assert np.array_equal(out[completed != bg], completed[completed != bg])
            assert np.any(out == bg) and np.any(out == HIGHLIGHT)
            assert set(np.unique(out)) == set(np.unique(a)) | {HIGHLIGHT}
            f = _features(a)
            assert _usable(a), "both stages must have visible diagnostic effects"
            if split == "train":
                training.append(f)
            else:
                assert f["source_count"] >= 2 and max(f["bounces"]) >= 2
    assert {axis for f in training for axis in f["bar_axes"]} == {0, 1}
    assert len({d for f in training for d in f["directions"]}) >= 2
    assert any(1 in f["mirror_lengths"] for f in training)
    assert any(f["changed_contacts"] for f in training)
    assert any(f["unchanged_contacts"] for f in training)
    assert any(f["small_components"] for f in training)
    assert len({f["background"] for f in training}) >= 2
    assert len({f["shape"] for f in training}) >= 2


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _regressions():
    # An explicit whole-composition expectation, including clipping and a
    # background center suppressed by a ray that was absent from the input.
    small = np.zeros((7, 7), dtype=np.int8)
    small[0, 1] = small[1, 0] = small[1, 1] = 1
    expected_small = np.array([[int(v) for v in row] for row in (
        "0103333", "1100333", "0010033", "3001003",
        "3300100", "3330010", "3333001")], dtype=np.int8)
    untouched = small.copy()
    assert np.array_equal(solve(small), expected_small)
    assert np.array_equal(small, untouched), "solve mutated its input"
    assert _centers(small != 0)[2, 3] and expected_small[2, 3] == 0
    assert expected_small[0, 6] == HIGHLIGHT
    # Explicit first reflection and contact-color expectations, with a 7-bar.
    a = np.zeros((12, 13), dtype=np.int8)
    for p in ((0, 2), (1, 1), (1, 2)):
        a[p] = 1
    a[7, 4:11] = 2
    expected = a.copy()
    for p in ((2, 3), (3, 4), (4, 5), (5, 6)):
        expected[p] = 1
    for p in ((6, 7), (5, 8), (4, 9), (3, 10), (2, 11), (1, 12)):
        expected[p] = 2
    assert np.array_equal(_trace(a)[0], expected)
    for k in range(4):
        assert np.array_equal(_trace(np.rot90(a, k))[0], np.rot90(expected, k))
    # Direct neighborhood diagnostics: corner clipping and diagonal exclusion.
    f = np.zeros((5, 5), dtype=bool)
    f[2, 2] = True
    mask = _centers(f)
    expected_mask = np.ones((5, 5), dtype=bool)
    expected_mask[1:4, 1:4] = False
    assert np.array_equal(mask, expected_mask)
    assert mask[0, 0] and not mask[1, 1]
    # The clear clipped window centered at (0,0) includes (1,1), but (1,1)
    # cannot itself be highlighted: its window contains the obstacle (2,2).
    # Thus a union-of-windows answer differs from this center-only expectation.
    # Simultaneity also allows all adjacent cells of a clear canvas to qualify.
    assert np.all(_centers(np.zeros((2, 3), dtype=bool)))
    # Color-aware components allow adjacent differently colored L sources.
    touching = np.zeros((12, 12), dtype=np.int8)
    touching[0, 1] = touching[1, 0] = touching[1, 1] = 1
    touching[0, 3] = touching[1, 2] = touching[1, 3] = 2
    assert len(_trace(touching)[1]["sources"]) == 2
    ambiguous = touching.copy()
    ambiguous[ambiguous == 2] = 1
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous same-colored touching Ls accepted")
    # Actual shared cells permit equal colors and reject conflicting colors.
    crossing = np.zeros((11, 11), dtype=np.int8)
    crossing[0, 1] = crossing[1, 0] = crossing[1, 1] = 1
    crossing[0, 9] = crossing[1, 9] = crossing[1, 10] = 1
    assert _trace(crossing)[0][5, 5] == 1
    for p in ((0, 9), (1, 9), (1, 10)):
        crossing[p] = 2
    try:
        solve(crossing)
    except ValueError:
        pass
    else:
        raise AssertionError("incompatible ray superposition accepted")


def self_test():
    """Bounded: 42 host seeds, no saved grids or growing logs; return metrics."""
    _regressions()
    signatures, shapes, normalized, backgrounds, palettes = set(), set(), set(), set(), set()
    sources, bounces, lengths, directions, axes = set(), set(), set(), set(), set()
    stats = Counter()
    for seed in CHECK_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                f = _features(a)
                assert np.array_equal(solve(a.copy()), out)
                shapes.add(a.shape)
                backgrounds.add(f["background"])
                palettes.add(f["palette"])
                sources.add(f["source_count"])
                bounces.update(f["bounces"])
                lengths.update(f["mirror_lengths"])
                directions.update(f["directions"])
                axes.update(f["axes"])
                mapping = {f["background"]: 0}
                encoded = bytes(mapping.setdefault(int(v), len(mapping)) for v in a.flat)
                normalized.add((a.shape, encoded))
                stats["pairs"] += 1
                stats["small_pocket_pairs"] += f["small_components"] > 0
                stats["multi_highlight_pairs"] += f["highlight_components"] > 1
                stats["ray_necessity_pairs"] += f["lost_centers"] > 0
                stats["boundary_diagnostic_pairs"] += f["edge_centers"] > 0
                stats["diagonal_diagnostic_pairs"] += f["diagonal_fringe"] > 0
                stats["full_30_axis_pairs"] += 30 in a.shape
                if split == "test":
                    stats["test_chain_multi_source_pairs"] += f["source_count"] >= 2 and max(f["bounces"]) >= 2
    assert len(signatures) == 42 and len(shapes) >= 60 and len(normalized) >= 190
    assert len(backgrounds) == 9 and len(palettes) >= 50
    assert sources >= {1, 2, 3, 4} and bounces >= {0, 1, 2, 3}
    assert lengths >= {1, 3, 5, 7} and directions == set(DIRECTIONS) and axes == {0, 1}
    assert stats["small_pocket_pairs"] >= 42 and stats["full_30_axis_pairs"] > 0
    assert stats["ray_necessity_pairs"] == stats["pairs"]
    assert _digest(generate_puzzle(0)) == _digest(generate_puzzle(0))
    # Every legal explicit count combination. Calls remain one puzzle at a time.
    count_cases = 0
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            p = generate_puzzle(17+tr*5+te, tr, te)
            assert len(p["train"]) == tr and len(p["test"]) == te
            count_cases += 1
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), grid_shapes=len(shapes),
                normalized_inputs=len(normalized), backgrounds=sorted(backgrounds),
                source_counts=sorted(sources), bounce_counts=sorted(bounces),
                mirror_lengths=sorted(lengths), legal_count_combinations=count_cases,
                **dict(stats))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
