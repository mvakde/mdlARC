"""Inspection-window comb repair: exact crop registration selects a local ruler.

Only NumPy and the standard library are used. Generation writes no files and
retains no traces. The solver reads both framed objects from the input itself.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


HOST_SEEDS = (0, 802027149, 649846994, 527615597, 41545319, 831236985,
              302413562, 923364153, 246407138, 114261700, 528113835,
              992665829, 230939710, 903839079, 137943127, 171456695,
              870322485, 634300277, 105215881, 427347551, 328830047,
              813453907, 973868766, 506187168, 16865281, 414849058,
              142316037, 619445913, 843159062, 236197836, 44866085,
              682887533, 233144665, 243228995, 725006785, 583073534,
              604655600, 971050250, 102591392, 349095152, 76956656,
              485631588)


def _d4(a, index):
    """0..3 rotations; 4..7 horizontal mirror followed by rotations."""
    return np.rot90(np.fliplr(a) if index >= 4 else a, index % 4).copy()


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _canonical(a):
    return min(_key(_d4(a, i)) for i in range(8))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = remaining.pop()
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    cells.append(q)
        result.append(cells)
    return result


def _panels(a):
    """Read the two equal-color hollow frames; size identifies the full view."""
    outside = int(a[0, 0])
    if not (np.all(a[0] == outside) and np.all(a[-1] == outside)
            and np.all(a[:, 0] == outside) and np.all(a[:, -1] == outside)):
        raise ValueError("the outer border must be uniform empty canvas")
    interpretations = []
    for color in np.unique(a):
        if color == outside:
            continue
        parts = _components(a == color)
        if len(parts) != 2:
            continue
        boxes = []
        for cells in parts:
            rr, cc = zip(*cells)
            r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
            h, w = r1 - r0 + 1, c1 - c0 + 1
            perimeter = {(r, c) for r in range(r0, r1 + 1)
                         for c in range(c0, c1 + 1)
                         if r in (r0, r1) or c in (c0, c1)}
            if min(h, w) < 5 or set(cells) != perimeter:
                break
            panel = a[r0 + 1:r1, c0 + 1:c1].copy()
            if len(np.unique(panel)) != 2 or np.any(panel == outside):
                break
            boxes.append((r0, r1, c0, c1, panel))
        if len(boxes) != 2:
            continue
        occupied = np.zeros(a.shape, dtype=bool)
        for r0, r1, c0, c1, _ in boxes:
            if occupied[r0:r1 + 1, c0:c1 + 1].any():
                break
            occupied[r0:r1 + 1, c0:c1 + 1] = True
        else:
            if np.all(a[~occupied] == outside):
                panels = sorted((b[4] for b in boxes), key=lambda p: p.size)
                if panels[0].size < panels[1].size:
                    interpretations.append((panels[1], panels[0]))
    if len(interpretations) != 1:
        raise ValueError("expected exactly two unambiguous framed views")
    return interpretations[0]


def _comb(mask):
    """Parse separated parallel strokes and their unique common anchor side."""
    cells = np.argwhere(mask == 1)
    candidates = []
    for transverse in (0, 1):
        axial = 1 - transverse
        keys = sorted(set(int(p[transverse]) for p in cells))
        n = len(keys)
        if n < 5 or n % 2 == 0 or any(b - a < 2 for a, b in zip(keys, keys[1:])):
            continue
        groups = [cells[cells[:, transverse] == k] for k in keys]
        groups = [g[np.argsort(g[:, axial])] for g in groups]
        if any(not np.all(np.diff(g[:, axial]) == 1) for g in groups):
            continue
        lo = [int(g[0, axial]) for g in groups]
        hi = [int(g[-1, axial]) for g in groups]
        if (len(set(lo)) == 1) == (len(set(hi)) == 1):
            continue
        sign = 1 if len(set(lo)) == 1 else -1
        if sign == -1:
            groups = [g[::-1] for g in groups]
        direction = np.zeros(2, dtype=int)
        direction[axial] = sign
        candidates.append(dict(transverse=transverse, axial=axial,
                               keys=tuple(keys), groups=groups,
                               lengths=tuple(len(g) for g in groups),
                               direction=direction))
    if len(candidates) != 1:
        raise ValueError("reference is not an unambiguous comb")
    return candidates[0]


def _reference_mask(panel):
    colors, counts = np.unique(panel, return_counts=True)
    if len(colors) != 2 or counts[0] == counts[1]:
        raise ValueError("reference needs a sparse stroke role")
    mask = (panel == colors[int(np.argmin(counts))]).astype(np.int8)
    _comb(mask)
    return mask


def _locations(full, patch):
    h, w = patch.shape
    if h > full.shape[0] or w > full.shape[1]:
        return []
    windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
    return [tuple(map(int, p)) for p in np.argwhere(
        np.all(windows == patch, axis=(-2, -1)))]


def _register(reference, patch):
    colors = tuple(int(c) for c in np.unique(patch))
    if len(colors) != 2:
        raise ValueError("the crop must show both roles")
    matches = []
    for orientation in range(8):
        full = _d4(reference, orientation)
        for ink in colors:
            binary = (patch == ink).astype(np.int8)
            for r, c in _locations(full, binary):
                matches.append((full, r, c, colors[1] if ink == colors[0] else colors[0],
                                ink, orientation))
    if len(matches) != 1:
        raise ValueError(f"crop requires unique orientation, roles and offset; got {len(matches)}")
    return matches[0]


def _selection(full, r, c, shape):
    a = _comb(full)
    low = (r, c)[a["transverse"]]
    high = low + shape[a["transverse"]]
    selected = [i for i, k in enumerate(a["keys"]) if low <= k < high]
    if not (3 <= len(selected) <= len(a["keys"]) - 2 and len(selected) % 2):
        raise ValueError("crop must select an odd proper subarray with at least three strokes")
    for i in selected:
        g = a["groups"][i]
        visible = ((r <= g[:, 0]) & (g[:, 0] < r + shape[0])
                   & (c <= g[:, 1]) & (g[:, 1] < c + shape[1]))
        if not visible.any():
            raise ValueError("each selected stroke must be visible in the crop")
    return a, selected


def _repair(full, analysis, selected, target_index=None, scope=None):
    target_index = selected[len(selected) // 2] if target_index is None else target_index
    target = analysis["lengths"][target_index]
    result = full.copy()
    for i in selected if scope is None else scope:
        g = analysis["groups"][i]
        result[g[:, 0], g[:, 1]] = 0
        line = g[0] + np.arange(target)[:, None] * analysis["direction"]
        if np.any(line < 0) or np.any(line[:, 0] >= full.shape[0]) or np.any(line[:, 1] >= full.shape[1]):
            raise ValueError("repaired strokes must fit the full reference")
        result[line[:, 0], line[:, 1]] = 1
    return result


def _decode(input_grid):
    a = _grid(input_grid)
    panel, patch = _panels(a)
    reference = _reference_mask(panel)
    full, r, c, background, ink, orientation = _register(reference, patch)
    analysis, selected = _selection(full, r, c, patch.shape)
    repaired = _repair(full, analysis, selected)
    output = np.where(repaired, ink, background).astype(np.int8)
    return output, dict(reference=reference, full=full, patch=patch,
                        r=r, c=c, background=background, ink=ink,
                        orientation=orientation, analysis=analysis,
                        selected=selected, repaired=repaired)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer entirely from the two views present in this input."""
    return _decode(input_grid)[0]


def _make_reference(rng):
    """A procedural asymmetric comb, including two useful demonstration windows."""
    n = int(rng.choice((7, 9, 11)))
    width = int(rng.integers(10, 15))
    target = int(rng.integers(5, 9))
    other = int(rng.choice([v for v in range(5, width) if v != target]))
    lengths = rng.integers(2, width + 1, size=n).tolist()
    lengths[:5] = [int(rng.integers(2, target)) for _ in range(5)]
    lengths[0] = min(lengths[0], target - 2)
    lengths[2] = target
    long_index = int(rng.choice((1, 3)))
    lengths[long_index] = int(rng.integers(target + 1, width + 1))
    # A second, differently sized ruler with its anchor and tip hidden in a crop.
    lengths[-3] = int(rng.integers(2, min(target, other - 1)))
    lengths[-2] = other
    lengths[-1] = width
    if lengths[n // 2] == target:
        lengths[n // 2] = other
    gaps = [2] * (n - 1)
    gaps[int(rng.choice((0, 3)))] += int(rng.choice((3, 5))) if n < 11 else 3
    gaps[-2 + int(rng.integers(2))] = 3
    # The uniquely uneven last triple helps identify even heavily clipped views.
    budget = 25 - sum(gaps)
    for i in rng.permutation(list(range(4, n - 3))):
        if budget >= 2 and rng.random() < .65:
            gaps[int(i)] += 2
            budget -= 2
    rows = [0] + np.cumsum(gaps).tolist()
    full = np.zeros((rows[-1] + 1, width), dtype=np.int8)
    for row, length in zip(rows, lengths):
        full[row, :length] = 1
    assert len({_key(_d4(full, i)) for i in range(8)}) == 8
    return full


def _reserved_crop(reference, kind):
    """Three small views checked before attempting any scene construction."""
    a = _comb(reference)
    rows, lengths = a["keys"], a["lengths"]
    if kind == "clipped":
        r, end, c = rows[-3], reference.shape[0], 1
        width = max(3, lengths[-3])
    else:
        r, end = 0, rows[4] + 1
        c = 1 if kind == "query" else 0
        width = max(3, min(lengths[:5])) if c else lengths[2] + 1
    return reference[r:end, c:c + width].copy(), (r, c)


def _checked_reference(rng):
    for attempt in range(33):
        reference = _make_reference(rng)
        if attempt == 32:
            # A large gap at the first edge and the unique three-cell gap at
            # the last edge give identifying views even if sampling was unlucky.
            a = _comb(reference)
            gaps = [2] * (len(a["keys"]) - 1)
            gaps[0], gaps[-1] = 5, 3
            rows = [0] + np.cumsum(gaps).tolist()
            reference = np.zeros((rows[-1] + 1, reference.shape[1]), dtype=np.int8)
            for row, length in zip(rows, a["lengths"]):
                reference[row, :length] = 1
        try:
            for kind in ("witness", "clipped", "query"):
                patch, _ = _reserved_crop(reference, kind)
                _register(reference, patch)
        except ValueError:
            continue
        return reference
    raise AssertionError("structured identifying-view fallback failed")


def _crop(rng, reference, kind):
    a = _comb(reference)
    rows, lengths = a["keys"], a["lengths"]
    n = len(rows)
    if kind == "witness":
        start, count, c = 0, 5, 0
        width = lengths[2] + 1
    elif kind == "clipped":
        start, count = n - 3, 3
        short, target = lengths[-3], lengths[-2]
        c = int(rng.integers(1, short))
        # At least one tail is visible; the actual central tip is outside.
        low = max(3, short - c + 1)
        high = min(7, target - c - 1)
        if low > high:
            raise ValueError("no fitting clipped ruler view")
        width = int(rng.integers(low, high + 1))
    elif kind == "query":
        eligible = [(s, k) for k in range(5, n, 2) for s in range(n - k + 1)
                    if lengths[s + k // 2] >= 5
                    and min(lengths[s:s + k]) <= lengths[s + k // 2] - 2]
        start, count = eligible[int(rng.integers(len(eligible)))]
        shortest = min(lengths[start:start + count])
        target = lengths[start + count // 2]
        c = int(rng.integers(1, shortest))
        low, high = max(3, shortest - c + 1), min(7, target - c - 1)
        if high < low:
            raise ValueError("no fitting clipped subarray")
        width = int(rng.integers(low, high + 1))
    else:
        count = int(rng.choice(list(range(3, n, 2))))
        start = int(rng.integers(n - count + 1))
        shortest = min(lengths[start:start + count])
        c = int(rng.integers(shortest))
        high = min(reference.shape[1] - c, 23 - reference.shape[1])
        if high < 3:
            raise ValueError("no fitting viewport width")
        width = int(rng.integers(3, high + 1))
    first, last = start, start + count - 1
    low = 0 if first == 0 else rows[first - 1] + 1
    high = reference.shape[0] if last == n - 1 else rows[last + 1]
    r0 = int(rng.integers(low, rows[first] + 1))
    r1 = int(rng.integers(rows[last] + 1, high + 1))
    return reference[r0:r1, c:c + width].copy(), (r0, c)


def _frame(panel, color):
    result = np.full((panel.shape[0] + 2, panel.shape[1] + 2), color, dtype=np.int8)
    result[1:-1, 1:-1] = panel
    return result


def _scene(rng, full, patch, first_palette):
    outside, border = map(int, rng.choice(10, 2, replace=False))
    available = [c for c in range(10) if c not in (outside, border)]
    full_colors = list(map(int, rng.choice(available, 2, replace=False)))
    if first_palette:
        patch_colors = full_colors[::-1]
    else:
        patch_colors = list(map(int, rng.choice(available, 2, replace=False)))
        if patch_colors == full_colors:
            patch_colors.reverse()
    panels = [_frame(np.where(full, full_colors[1], full_colors[0]), border),
              _frame(np.where(patch, patch_colors[1], patch_colors[0]), border)]
    if bool(rng.integers(2)):
        panels.reverse()
    max_gap = min(3, 28 - sum(p.shape[1] for p in panels))
    if max_gap < 1:
        raise ValueError("panels do not fit")
    gap = int(rng.integers(1, max_gap + 1))
    min_h = max(p.shape[0] for p in panels) + 2
    min_w = sum(p.shape[1] for p in panels) + gap + 2
    h = int(rng.integers(min_h, 31))
    w = int(rng.integers(min_w, 31))
    canvas = np.full((h, w), outside, dtype=np.int8)
    col = int(rng.integers(1, w - min_w + 2))
    for p in panels:
        row = int(rng.integers(1, h - p.shape[0]))
        canvas[row:row + p.shape[0], col:col + p.shape[1]] = p
        col += p.shape[1] + gap
    return canvas, patch_colors


def _pair(rng, reference, kind, seen):
    # At most 240 small in-memory attempts; no persistent retry payloads.
    for attempt in range(240):
        try:
            if attempt >= 200:
                patch, (r, c) = _reserved_crop(reference, kind)
            else:
                patch, (r, c) = _crop(rng, reference, kind)
            if kind == "witness":
                orientation = int(rng.choice((4, 6)))  # demonstrate a reflection
            elif kind == "clipped":
                orientation = int(rng.choice((1, 3)))  # demonstrate a quarter turn
            else:
                orientation = int(rng.integers(8))
            oriented_patch = _d4(patch, orientation)
            if oriented_patch.shape[1] + reference.shape[1] > 23:
                continue
            match = _register(reference, oriented_patch)
            full, rr, cc, bg, ink, _ = match
            a, selected = _selection(full, rr, cc, oriented_patch.shape)
            repaired = _repair(full, a, selected)
            # Construct the label directly in the sampled horizontal layout.
            # Registration and the general repair routine are checks, rather
            # than the source of the returned label.
            rows = np.flatnonzero(reference.any(axis=1))
            active = rows[(r <= rows) & (rows < r + patch.shape[0])]
            target = int(reference[active[len(active) // 2]].sum())
            drawn = reference.copy()
            for row in active:
                drawn[row] = 0
                drawn[row, :target] = 1
            drawn = _d4(drawn, orientation)
            assert np.array_equal(drawn, repaired)
            if np.array_equal(repaired, full):
                continue
            all_repaired = _repair(full, a, selected, scope=range(len(a["keys"])))
            if np.array_equal(all_repaired, repaired):
                continue
            canvas, colors = _scene(rng, reference, oriented_patch, kind == "witness")
            output = np.where(drawn, colors[1], colors[0]).astype(np.int8)
            # Match's binary roles must be the role mapping used when drawing.
            assert (bg, ink) == (0, 1)
            global_orientation = int(rng.integers(8))
            canvas, output = _d4(canvas, global_orientation), _d4(output, global_orientation)
            if _key(canvas) in seen:
                continue
            f = _features(canvas)
            if kind == "clipped" and not f["hidden_tip"]:
                continue
            if kind == "free" and f["selected"] >= 5 and f["hidden_tip"] and f["hidden_anchor"]:
                continue
            if kind == "query" and not (f["selected"] >= 5 and f["hidden_tip"] and f["hidden_anchor"]):
                continue
            seen.add(_key(canvas))
            return {"input": canvas, "output": output}
        except ValueError:
            continue
    raise RuntimeError(f"bounded sampling exhausted for {kind}")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    reference = _checked_reference(rng)
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(nt):
        kind = "witness" if i == 0 else "clipped" if i == 1 else "free"
        puzzle["train"].append(_pair(rng, reference, kind, seen))
    rng.shuffle(puzzle["train"])
    for _ in range(nq):
        # Transfer clipping from a demonstrated three-tooth view to a larger
        # selected subarray, with the real ruler visibly present in the full view.
        puzzle["test"].append(_pair(rng, reference, "query", seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(input_grid):
    _, d = _decode(input_grid)
    a, selected = d["analysis"], d["selected"]
    middle = selected[len(selected) // 2]
    lengths = [a["lengths"][i] for i in selected]
    keys = np.array([a["keys"][i] for i in selected])
    distances = abs(2 * keys - keys[0] - keys[-1])
    nearest = np.flatnonzero(distances == distances.min())
    visible = []
    for i in selected:
        g = a["groups"][i]
        inside = ((d["r"] <= g[:, 0]) & (g[:, 0] < d["r"] + d["patch"].shape[0])
                  & (d["c"] <= g[:, 1]) & (g[:, 1] < d["c"] + d["patch"].shape[1]))
        visible.append(int(inside.sum()))
    g = a["groups"][middle]
    tip, anchor = g[-1], g[0]
    def inside(p):
        return bool(d["r"] <= p[0] < d["r"] + d["patch"].shape[0]
                    and d["c"] <= p[1] < d["c"] + d["patch"].shape[1])
    return dict(count=len(a["keys"]), selected=len(selected), target=len(g),
                lengths=tuple(lengths), gaps=tuple(np.diff(keys).tolist()),
                extension=any(v < len(g) for v in lengths),
                truncation=any(v > len(g) for v in lengths),
                nonmedian=sorted(lengths)[len(lengths) // 2] != len(g),
                geometric_witness=bool(len(nearest) == 1 and selected[int(nearest[0])] != middle
                                       and lengths[int(nearest[0])] != len(g)),
                global_witness=a["lengths"][len(a["keys"]) // 2] != len(g),
                hidden_tip=not inside(tip), hidden_anchor=not inside(anchor),
                visible_lengths=tuple(visible),
                clipped_target=visible[len(selected) // 2] != len(g),
                shape=tuple(np.asarray(input_grid).shape), output_shape=tuple(d["full"].shape),
                patch_shape=tuple(d["patch"].shape), orientation=d["orientation"],
                palette=tuple(map(int, np.unique(input_grid))))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, references, features = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            expected, d = _decode(inp)
            assert np.array_equal(expected, out), "incorrect registered local repair"
            references.add(_canonical(d["reference"]))
            a, selected = d["analysis"], d["selected"]
            target = a["lengths"][selected[len(selected) // 2]]
            assert not np.array_equal(d["full"], d["repaired"])
            # Check scope, intact anchors, retained prefixes and corrected tails.
            for i, g in enumerate(a["groups"]):
                if i not in selected:
                    assert np.all(d["repaired"][g[:, 0], g[:, 1]] == 1)
                else:
                    assert np.all(d["repaired"][g[:target, 0], g[:target, 1]] == 1)
                    if len(g) > target:
                        assert np.all(d["repaired"][g[target:, 0], g[target:, 1]] == 0)
                assert d["repaired"][tuple(g[0])] == 1
            # Omitting B gives the unmodified registered completion, and fails.
            assert not np.array_equal(d["full"], d["repaired"])
            # Omitting A's scope and normalizing the entire comb also fails:
            # the selected reference is used, but unselected teeth get changed.
            all_repaired = _repair(d["full"], a, selected, scope=range(len(a["keys"])))
            assert not np.array_equal(all_repaired, d["repaired"])
            f = _features(inp)
            features[split].append(f)
    assert len(references) == 1, "one shared reference per puzzle"
    train = features["train"]
    assert len({f["target"] for f in train}) >= 2, "demonstrate variable target lengths"
    assert len({f["selected"] for f in train}) >= 2, "demonstrate variable selected counts"
    assert any(f["nonmedian"] and f["geometric_witness"] and f["extension"]
               and f["truncation"] and f["global_witness"] for f in train)
    assert any(f["hidden_tip"] and f["hidden_anchor"] and f["clipped_target"] for f in train)
    assert all(not (f["selected"] >= 5 and f["hidden_tip"] and f["hidden_anchor"]) for f in train)
    assert all(f["selected"] >= 5 and f["hidden_tip"] and f["hidden_anchor"]
               for f in features["test"])


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                a = np.asarray(pair[name], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded: 42 streamed puzzles, count cases, mutations and equivariance.

    No files, whole-corpus grids, logs, caches or execution traces are saved.
    Returns compact-serializable descriptive measurements, not external approval.
    """
    signatures, references, shapes, patches, targets, counts = (set() for _ in range(6))
    orientations, directions, behaviors = set(), set(), set()
    totals = dict(pairs=0, hidden_anchor_and_tip=0, nonmedian=0,
                  geometric_witness=0, global_witness=0, selected_scope_checks=0)
    rng = np.random.default_rng(20261022)
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        if seed == 0:
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _, d = _decode(pair["input"])
                f = _features(pair["input"])
                references.add(_canonical(d["reference"]))
                shapes.add(f["shape"])
                patches.add(f["patch_shape"])
                targets.add(f["target"])
                counts.add((f["count"], f["selected"]))
                orientations.add(f["orientation"])
                directions.add(tuple(d["analysis"]["direction"].tolist()))
                behaviors.add((f["extension"], f["truncation"]))
                totals["pairs"] += 1
                totals["hidden_anchor_and_tip"] += int(f["hidden_anchor"] and f["hidden_tip"])
                for name in ("nonmedian", "geometric_witness", "global_witness"):
                    totals[name] += int(f[name])
                # A scope ablation with the correct selected ruler length.
                wrong = _repair(d["full"], d["analysis"], d["selected"],
                                scope=range(f["count"]))
                assert not np.array_equal(wrong, d["repaired"])
                totals["selected_scope_checks"] += 1
        pair = puzzle["test"][0]
        # The visible input alone fixes all geometry and literal color roles.
        palette = rng.permutation(10)
        assert np.array_equal(solve(palette[pair["input"]]), palette[pair["output"]])
        for i in range(8):
            assert np.array_equal(solve(_d4(pair["input"], i)), _d4(pair["output"], i))
    assert len(signatures) == len(references) == 42
    assert len(shapes) >= 30 and len(patches) >= 20 and len(targets) >= 6
    assert {n for n, _ in counts} == {7, 9, 11}
    assert {3, 5, 7} <= {n for _, n in counts}
    assert orientations == set(range(8)) and len(directions) == 4
    assert (True, True) in behaviors
    # Exercise every legal count; the host repeats this independently.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(87123 + nt * 10 + nq, nt, nq)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = generate_puzzle(0, 2, 1)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged answer accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), reference_classes=len(references),
                input_shapes=len(shapes), crop_shapes=len(patches), targets=sorted(targets),
                count_combinations=sorted(counts), relative_orientations=sorted(orientations),
                directed_axes=sorted(directions), correction_behaviors=sorted(behaviors),
                totals=totals, independent_solver=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
