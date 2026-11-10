"""Framed analogies whose demonstrated relation is exact equality-run collapse.

Only standard-library modules and NumPy are used. Generation constructs a
canonical answer and stretches it; solve reads only the supplied scene.
No generated puzzles or execution traces are written by this module.
"""
from __future__ import annotations

import hashlib
import json
from itertools import combinations, product

import numpy as np


SELF_TEST_SEEDS = (
    0, 813611879, 578424150, 789753921, 445247324, 257092526, 783771242,
    284230604, 384372038, 534616914, 388219876, 137382822, 415899422,
    230279000, 579899926, 238047315, 890747112, 64447068, 554943070,
    708693145, 683782513, 587376894, 75749007, 8004063, 300663248,
    886127234, 894224462, 968170118, 425199842, 978142646, 819378,
    464448393, 963392890, 257130571, 565684667, 996631530, 429261136,
    675844020, 766017067, 380456630, 254153033, 308515598,
)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _digest(a):
    a = np.asarray(a, dtype=np.int8)
    return hashlib.sha256(bytes(a.shape) + a.tobytes()).hexdigest()


def _keep(a, axis):
    vectors = a if axis == 0 else a.T
    return np.r_[True, np.any(vectors[1:] != vectors[:-1], axis=1)]


def _collapse(a, axes=(0, 1)):
    """Masks are calculated on the original interior, independently per axis."""
    rows = _keep(a, 0) if 0 in axes else np.ones(a.shape[0], dtype=bool)
    cols = _keep(a, 1) if 1 in axes else np.ones(a.shape[1], dtype=bool)
    return a[rows][:, cols].copy()


def _run_lengths(a, axis):
    starts = np.flatnonzero(_keep(a, axis))
    return tuple(map(int, np.diff(np.r_[starts, a.shape[axis]])))


def _separated_repeat(a, axis):
    vectors = a if axis == 0 else a.T
    return any(np.array_equal(vectors[i], vectors[j])
               for i in range(len(vectors)) for j in range(i + 2, len(vectors)))


def _inside(a, b):
    r, c, h, w = a[:4]
    y, x, bh, bw = b[:4]
    return y < r and x < c and r + h < y + bh and c + w < x + bw


def _disjoint(a, b):
    r, c, h, w = a[:4]
    y, x, bh, bw = b[:4]
    return r + h <= y or y + bh <= r or c + w <= x or x + bw <= c


def _outline(a, box, color):
    r, c, h, w = box[:4]
    a[r, c:c+w] = a[r+h-1, c:c+w] = color
    a[r:r+h, c] = a[r:r+h, c+w-1] = color


def _interior(a, box):
    r, c, h, w = box[:4]
    return a[r+1:r+h-1, c+1:c+w-1].copy()


def _outline_components(a):
    """Recognize isolated one-cell rectangular outlines by connected pixels.

    Frame colors are disjoint from contents and wallpaper, but their literal
    values are unknown. Nested content outlines are allowed and filtered later.
    """
    h, w = a.shape
    seen = np.zeros(a.shape, dtype=bool)
    found = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(a[r, c])
            seen[r, c] = True
            stack = [(r, c)]
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if (0 <= yy < h and 0 <= xx < w and not seen[yy, xx]
                            and a[yy, xx] == color):
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            top = min(p[0] for p in cells)
            left = min(p[1] for p in cells)
            bottom = max(p[0] for p in cells)
            right = max(p[1] for p in cells)
            bh, bw = bottom-top+1, right-left+1
            if (bh >= 4 and bw >= 4 and len(cells) == 2*(bh+bw)-4
                    and all(y in (top, bottom) or x in (left, right) for y, x in cells)):
                found.append((top, left, bh, bw, color))
    return found


def _infer_reference(reference):
    """The complete pair must select both axes over either single-axis rule."""
    hypotheses = []
    for i in (0, 1):
        source, partner = reference[i], reference[1-i]
        for axes in ((0,), (1,), (0, 1)):
            if source.shape != partner.shape and np.array_equal(_collapse(source, axes), partner):
                hypotheses.append((i, axes))
    if len(hypotheses) != 1 or hypotheses[0][1] != (0, 1):
        raise ValueError("complete pair must uniquely demonstrate two-axis run collapse")
    return hypotheses[0][0]


def _read_scene(a):
    loops = _outline_components(a)
    groups = []
    for outer in loops:
        contained = [p for p in loops if _inside(p, outer)]
        direct = [p for p in contained if not any(_inside(p, q) for q in contained)]
        if (len(direct) == 2 and direct[0][4] == direct[1][4]
                and direct[0][4] != outer[4] and _disjoint(*direct)):
            groups.append((outer, sorted(direct)))
    scenes = {}
    for (box1, panels1), (box2, panels2) in combinations(groups, 2):
        if (box1[4] != box2[4] or panels1[0][4] != panels2[0][4]
                or not _disjoint(box1, box2)):
            continue
        boxes, panels = [box1, box2], [panels1, panels2]
        frame_colors = {box1[4], panels1[0][4]}
        interiors = [[_interior(a, p) for p in cluster] for cluster in panels]
        if any(frame_colors & set(map(int, np.unique(p))) for cluster in interiors for p in cluster):
            continue
        blanks = [(g, s) for g in (0, 1) for s in (0, 1)
                  if np.all(interiors[g][s] == interiors[g][s][0, 0])]
        if len(blanks) != 1:
            continue
        target_group, missing_slot = blanks[0]
        reference_group = 1-target_group
        reference = interiors[reference_group]
        occupied = interiors[target_group][1-missing_slot]
        blank = interiors[target_group][missing_slot]
        try:
            source_slot = _infer_reference(reference)
        except ValueError:
            continue
        answer = _collapse(occupied)
        if answer.shape != blank.shape or answer.shape == occupied.shape:
            continue
        signature = tuple(sorted(p for cluster in panels for p in cluster))
        scenes[signature] = dict(
            boxes=boxes, panels=panels, reference=reference,
            reference_source=reference[source_slot], reference_result=reference[1-source_slot],
            reference_source_slot=source_slot, reference_group=reference_group,
            target_group=target_group, missing_slot=missing_slot, occupied=occupied,
            blank=blank, neutral=int(blank[0, 0]), answer=answer,
        )
    if len(scenes) != 1:
        raise ValueError(f"expected one supported four-panel analogy, found {len(scenes)}")
    return next(iter(scenes.values()))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the demonstrated contraction and reconstruct only the blank crop."""
    scene = _read_scene(_grid(input_grid))
    # _read_scene checks the reference relation and the selected blank's size.
    return scene["answer"].copy()


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _kernel(rng, shape, kind, colors):
    h, w = shape
    if kind == 0:
        # ABA recurrences on both axes; not globally unique row/column removal.
        motif = np.array([[0, 1], [2, 0]], dtype=np.int8)
        motif = rng.permutation(colors)[motif]
        a = motif[np.arange(h)[:, None] % 2, np.arange(w)[None, :] % 2]
    elif kind == 1:
        # A compact line weave, including literal mixed or coherent crossings.
        a = np.zeros(shape, dtype=np.int8)
        horizontal = rng.integers(1, colors, size=h//2)
        vertical = rng.integers(1, colors, size=w//2)
        for j, color in enumerate(vertical):
            a[:, 2*j+1] = color
        for i, color in enumerate(horizontal):
            a[2*i+1, :] = color
        mode = int(rng.integers(5))
        bits = rng.integers(2, size=(h//2, w//2))
        if mode < 2:
            bits[:] = mode
        elif mode == 2:
            bits[:] = bits[:, :1]
        elif mode == 3:
            bits[:] = bits[:1, :]
        for i in range(h//2):
            for j in range(w//2):
                a[2*i+1, 2*j+1] = vertical[j] if bits[i, j] else horizontal[i]
    else:
        # Irregular canonical mosaics stretch patterned bands too.
        for _ in range(24):
            a = rng.integers(colors, size=shape, dtype=np.int8)
            if _collapse(a).shape == shape:
                break
        else:
            rr, cc = np.indices(shape)
            a = ((rr + cc) % colors).astype(np.int8)
    assert _collapse(a).shape == a.shape and len(np.unique(a)) > 1
    return a.astype(np.int8)


def _shape(rng, kind, maximum):
    if kind == 1:
        return tuple(int(rng.choice([3, 5])) for _ in range(2))
    return tuple(int(rng.integers(3, maximum+1)) for _ in range(2))


def _stretch(rng, length, total):
    extra = total-length
    runs = np.ones(length, dtype=int)
    if extra:
        mode = int(rng.integers(3))
        weights = np.ones(length)
        if mode:
            index = int(rng.choice([0, length-1])) if mode == 1 else int(rng.integers(1, length-1))
            weights[index] = 6
        runs += rng.multinomial(extra, weights/weights.sum())
        # Unequal runs make literal resizing an inadequate explanation.
        if np.all(runs == runs[0]):
            runs[0] -= 1
            runs[1] += 1
    return runs


def _expand(rng, kernel, totals):
    runs = [_stretch(rng, n, total) for n, total in zip(kernel.shape, totals)]
    return np.repeat(np.repeat(kernel, runs[0], axis=0), runs[1], axis=1)


def _anchor_expand(rng, kernel):
    """Make run equality distinguishable from several ordinary resize rules.

    A middle-heavy ABA axis defeats start-aligned sampling; the other axis
    puts its one-cell B off center, defeating center and endpoint sampling.
    These properties survive the scene's rotations and reflections.
    """
    runs = [np.array([1, int(rng.integers(3, 8)), 1]),
            np.array([int(rng.integers(4, 7)), 1, int(rng.integers(1, 3))])]
    if rng.integers(2):
        runs.reverse()
    for axis in (0, 1):
        if rng.integers(2):
            runs[axis] = runs[axis][::-1]
    return np.repeat(np.repeat(kernel, runs[0], axis=0), runs[1], axis=1)


def _resize_guesses(source, shape):
    """Three explicit shortcut hypotheses, used only by validation/diagnostics."""
    for mode in ("start", "center", "endpoints"):
        indices = []
        for size, target in zip(source.shape, shape):
            if mode == "start":
                positions = np.arange(target)*size//target
            elif mode == "center":
                positions = (2*np.arange(target)+1)*size//(2*target)
            else:
                positions = np.linspace(0, size-1, target).astype(int)
            indices.append(positions)
        yield source[np.ix_(*indices)]


def _group_size(contents, axis):
    sizes = [(a.shape[0]+2, a.shape[1]+2) for a in contents]
    if axis == 0:  # horizontal pair
        return max(s[0] for s in sizes)+2, sum(s[1] for s in sizes)+3
    return sum(s[0] for s in sizes)+3, max(s[1] for s in sizes)+2


def _render(rng, contents, palette):
    """Pack independently oriented, unequal enclosures without rejection."""
    choices = []
    for axis1, axis2, pack in product((0, 1), repeat=3):
        sizes = [_group_size(contents[i], axis) for i, axis in enumerate((axis1, axis2))]
        h = sum(s[0] for s in sizes)+1 if pack == 0 else max(s[0] for s in sizes)
        w = max(s[1] for s in sizes) if pack == 0 else sum(s[1] for s in sizes)+1
        if h <= 30 and w <= 30:
            choices.append(((axis1, axis2), pack, sizes))
    if not choices:
        raise AssertionError("bounded panel sizes should always admit a layout")
    axes, pack, sizes = choices[int(rng.integers(len(choices)))]
    sizes = [list(s) for s in sizes]
    for dim in (0, 1):
        for g in (0, 1):
            available = (30-sum(s[dim] for s in sizes)-1 if dim == pack
                         else 30-sizes[g][dim])
            sizes[g][dim] += int(rng.integers(min(2, available)+1))
    minimum = [sum(s[d] for s in sizes)+1 if d == pack else max(s[d] for s in sizes)
               for d in (0, 1)]
    shape = tuple(int(rng.integers(n, 31)) for n in minimum)
    along = rng.multinomial(shape[pack]-minimum[pack], [1/3]*3)
    starts = [int(along[0]), int(along[0])+sizes[0][pack]+1+int(along[1])]
    boxes = []
    for g in (0, 1):
        cross = int(rng.integers(shape[1-pack]-sizes[g][1-pack]+1))
        r, c = (starts[g], cross) if pack == 0 else (cross, starts[g])
        boxes.append((r, c, sizes[g][0], sizes[g][1]))
    # Ground colors are separate from data and structural colors. Blocks and
    # stripes can repeat, but never create another isolated rectangular loop.
    rr, cc = np.indices(shape)
    mode = int(rng.integers(4))
    if mode == 0:
        ground = np.zeros(shape, dtype=int)
    else:
        rh, cw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        ground = ((rr//rh if mode != 2 else 0) + (cc//cw if mode != 3 else 0)) % 2
        ground = np.broadcast_to(ground, shape)
    a = np.where(ground, palette[9], palette[0]).astype(np.int8)
    for g, (r, c, gh, gw) in enumerate(boxes):
        # The same ground field continues inside enclosures; only borders group.
        _outline(a, boxes[g], int(palette[2]))
        dims = [(p.shape[0]+2, p.shape[1]+2) for p in contents[g]]
        axis = axes[g]
        # Along-axis coordinates are columns for a horizontal local pair.
        dim = 1-axis
        span = (gh, gw)[dim]
        remaining = span-sum(d[dim] for d in dims)-2
        gap = int(rng.integers(1, remaining+1))
        start = int(rng.integers(1, span-sum(d[dim] for d in dims)-gap))
        along_positions = [start, start+dims[0][dim]+gap]
        for slot, interior in enumerate(contents[g]):
            ph, pw = dims[slot]
            cross_span = (gh, gw)[1-dim]
            cross_size = dims[slot][1-dim]
            perpendicular = int(rng.integers(1, cross_span-cross_size))
            pr, pc = ((perpendicular, along_positions[slot]) if axis == 0
                      else (along_positions[slot], perpendicular))
            box = (r+pr, c+pc, ph, pw)
            _outline(a, box, int(palette[1]))
            a[r+pr+1:r+pr+ph-1, c+pc+1:c+pc+pw-1] = interior
    return a


def _make_pair(rng, split, index, flags):
    palette = rng.permutation(10).astype(np.int8)
    color_count = int(rng.integers(3, 7))
    data = palette[3:3+color_count]
    ref_kind = int(rng.integers(3))
    ref_core = _kernel(rng, _shape(rng, ref_kind, 5), ref_kind, color_count)
    ref_totals = tuple(int(rng.integers(n+1, 9)) for n in ref_core.shape)
    ref_source = _expand(rng, ref_core, ref_totals)
    if split == "train" and index < 2:
        kind = index  # recurrent weave then line weave
    else:
        kind = int(rng.integers(3))
    target_shape = (3, 3) if split == "train" and index == 0 else _shape(rng, kind, 6 if split == "test" else 5)
    target_core = _kernel(rng, target_shape, kind, color_count)
    if np.array_equal(target_core, ref_core):
        target_core = (target_core+1) % color_count
    if split == "test":
        totals = [int(rng.integers(n+1, 13)) for n in target_core.shape]
        large_axis = int(rng.integers(2))
        totals[large_axis] = int(rng.integers(10, 13))
        if rng.random() < .2:
            totals[1-large_axis] = target_core.shape[1-large_axis]
    else:
        totals = [int(rng.integers(n+1, 10)) for n in target_core.shape]
        if index == 1 or (index > 1 and rng.random() < .25):
            axis = int(rng.integers(2))
            totals[axis] = target_core.shape[axis]
    target_source = (_anchor_expand(rng, target_core)
                     if split == "train" and index == 0
                     else _expand(rng, target_core, totals))
    reference = [data[ref_source], data[ref_core]]
    target = [data[target_source], np.full(target_core.shape, data[0], dtype=np.int8)]
    # The first two demonstrations contrast each positional choice. Later
    # choices are independent; no position supplies transformation direction.
    bits = [flags[j] ^ bool(index) if split == "train" and index < 2
            else bool(rng.integers(2)) for j in range(3)]
    if bits[0]:
        reference.reverse()
    if bits[1]:
        target.reverse()
    contents = [reference, target] if not bits[2] else [target, reference]
    a = _render(rng, contents, palette)
    output = data[target_core].astype(np.int8)
    rotation = int(rng.integers(4))
    return {"input": np.rot90(a, rotation).copy(), "output": np.rot90(output, rotation).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    flags = tuple(bool(rng.integers(2)) for _ in range(3))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(24):
                pair = _make_pair(rng, split, index, flags)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    return puzzle


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    a, output = _grid(pair["input"]), _grid(pair["output"])
    scene = _read_scene(a)
    assert np.array_equal(scene["answer"], output), "incorrect missing companion"
    assert output.shape == scene["blank"].shape
    assert not np.array_equal(output, scene["reference_result"]), "reference-copy shortcut"
    for source, canonical in ((scene["reference_source"], scene["reference_result"]),
                              (scene["occupied"], output)):
        runs = [_run_lengths(source, axis) for axis in (0, 1)]
        rebuilt = np.repeat(np.repeat(canonical, runs[0], axis=0), runs[1], axis=1)
        assert np.array_equal(rebuilt, source), "lossless equality-run reconstruction failed"
        assert _collapse(canonical).shape == canonical.shape
        assert set(map(int, np.unique(source))) == set(map(int, np.unique(canonical)))
    # A whole-canvas contraction retains structural content rather than answering.
    assert not np.array_equal(_collapse(a), output)
    scene["row_runs"] = _run_lengths(scene["occupied"], 0)
    scene["column_runs"] = _run_lengths(scene["occupied"], 1)
    return scene


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            scene = _analyze_pair(pair)
            key = _key(pair["input"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            analyses[split].append(scene)
    train = analyses["train"]
    assert any(all(s["occupied"].shape[axis] > s["answer"].shape[axis]
                       and _separated_repeat(s["answer"], axis) for axis in (0, 1)) for s in train)
    assert any(sum(s["occupied"].shape[axis] > s["answer"].shape[axis] for axis in (0, 1)) == 1
               for s in train), "training must show an unchanged axis"
    assert any(all(not np.array_equal(guess, s["answer"])
                       for guess in _resize_guesses(s["occupied"], s["answer"].shape))
               for s in train), "training must distinguish run collapse from resizing"
    assert all(max(s["occupied"].shape) <= 9 for s in train)
    assert all(max(s["occupied"].shape) >= 10 for s in analyses["test"])
    return None


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _diagnostics():
    core = np.array([[1, 2, 1], [3, 4, 3], [1, 2, 1]], dtype=np.int8)
    stretched = np.repeat(np.repeat(core, [3, 1, 2], axis=0), [1, 3, 2], axis=1)
    assert np.array_equal(_collapse(stretched), core)
    assert not np.array_equal(stretched[::2, ::2], core), "uniform-resize counterexample"
    assert _collapse(stretched, (0,)).shape == (3, 6)
    assert _collapse(stretched, (1,)).shape == (6, 3)
    assert len(np.unique(core, axis=0)) < core.shape[0]
    assert np.unique(core, axis=1).shape[1] < core.shape[1]
    for transform in (lambda x: x.T, lambda x: x[::-1], lambda x: x[:, ::-1]):
        assert np.array_equal(_collapse(transform(stretched)), transform(core))
    assert _infer_reference([core, stretched]) == 1
    assert _infer_reference([stretched, core]) == 0
    pair = generate_puzzle(0, 2, 1)["train"][0]
    scene = _analyze_pair(pair)
    for axes in ((0,), (1,)):
        assert not np.array_equal(_collapse(scene["occupied"], axes), pair["output"])
    for transform in (lambda x: x.T, lambda x: x[::-1], lambda x: x[:, ::-1]):
        assert np.array_equal(solve(transform(pair["input"])), transform(pair["output"]))
        changed = transform(scene["occupied"])
        expected = transform(scene["answer"])
        assert all(not np.array_equal(guess, expected)
                   for guess in _resize_guesses(changed, expected.shape))
    # All three choices of member/group order are semantically irrelevant.
    target = (core+2) % 5
    target_source = np.repeat(np.repeat(target, [2, 1, 1], axis=0), [1, 2, 1], axis=1)
    rng = np.random.default_rng(4321)
    for reverse_reference, reverse_target, reverse_groups in product((False, True), repeat=3):
        reference = [core, stretched]
        incomplete = [target_source, np.zeros_like(target)]
        if reverse_reference:
            reference.reverse()
        if reverse_target:
            incomplete.reverse()
        groups = [reference, incomplete]
        if reverse_groups:
            groups.reverse()
        # Distinct frame, group and wallpaper colors; contents use 0..4.
        palette = np.array([8, 6, 7, 0, 1, 2, 3, 4, 5, 9], dtype=np.int8)
        rendered = _render(rng, groups, palette)
        assert np.array_equal(solve(rendered), target)
    # Removing the informative reference cannot yield an accepted scene.
    broken = pair["input"].copy()
    for box in scene["panels"][scene["reference_group"]]:
        r, c, h, w = box[:4]
        broken[r+1:r+h-1, c+1:c+w-1] = scene["neutral"]
    try:
        solve(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("missing reference accepted")
    original = generate_puzzle(17, 2, 1)
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]]
               for s in ("train", "test")}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong output accepted")


def self_test():
    """Bounded, file-free checks on all 42 requested seeds and legal counts."""
    shapes, output_shapes, layouts, structures, run_patterns, colors = (set() for _ in range(6))
    puzzle_hashes = set()
    one_axis = both_axes = recurrent = pairs = 0
    for seed in SELF_TEST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzle_hashes.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                scene = _analyze_pair(pair)
                pairs += 1
                shapes.add(pair["input"].shape)
                output_shapes.add(pair["output"].shape)
                layouts.add(tuple(p[:4] for cluster in scene["panels"] for p in cluster))
                run_patterns.add((scene["row_runs"], scene["column_runs"]))
                mapping = {}
                normalized = np.array([mapping.setdefault(int(x), len(mapping))
                                       for x in pair["output"].flat], dtype=np.int8)
                structures.add((pair["output"].shape, normalized.tobytes()))
                colors.add(scene["neutral"])
                shrinking = sum(scene["occupied"].shape[i] > pair["output"].shape[i] for i in (0, 1))
                one_axis += shrinking == 1
                both_axes += shrinking == 2
                recurrent += all(_separated_repeat(pair["output"], i) for i in (0, 1))
    assert len(puzzle_hashes) == 42 and len(colors) >= 8
    assert len(shapes) >= 20 and len(output_shapes) >= 8 and len(layouts) >= 150
    assert len(structures) >= 50 and len(run_patterns) >= 100
    assert one_axis >= 42 and both_axes >= 84 and recurrent >= 42
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(123456, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
            validate_generated_puzzle(puzzle)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _puzzle_digest(generate_puzzle(0)) == _puzzle_digest(generate_puzzle(0))
    assert _puzzle_digest(generate_puzzle(0)) != _puzzle_digest(generate_puzzle(1))
    _diagnostics()
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(puzzle_hashes),
                input_shapes=len(shapes), output_shapes=len(output_shapes),
                panel_layouts=len(layouts), normalized_outputs=len(structures),
                run_patterns=len(run_patterns), blank_colors=len(colors),
                one_axis_targets=one_axis, both_axis_targets=both_axes,
                separated_repeat_targets=recurrent, legal_count_combinations=30,
                independent_solver=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
