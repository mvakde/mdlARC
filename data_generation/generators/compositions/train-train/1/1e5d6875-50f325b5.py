"""Constellations of missing corners: negative-space stencil selection of L echoes.

Only Python's standard library and NumPy are used. No parent code or saved pairs
are loaded. Generation metadata is never needed by solve().
"""
from __future__ import annotations

import numbers
import numpy as np


CORNERS = frozenset(((0, 0), (0, 1), (1, 0), (1, 1)))
FREE_COLORS = (0, 3, 4, 5, 6, 7, 8, 9)


def _norm(points):
    points = tuple((int(r), int(c)) for r, c in points)
    r0, c0 = min(r for r, _ in points), min(c for _, c in points)
    return tuple(sorted((r - r0, c - c0) for r, c in points))


def _variants(points, mirrors=True):
    result = set()
    for flip in range(2 if mirrors else 1):
        p = [(r, -c if flip else c) for r, c in points]
        for _ in range(4):
            result.add(_norm(p))
            p = [(c, -r) for r, c in p]
    return tuple(sorted(result))


def _family(points):
    return min(_variants(points))


def _matches(stencil, landmarks):
    """Complete foreground matches in a point cloud; gaps are unconstrained."""
    landmarks = set(landmarks)
    found = set()
    for variant in _variants(stencil):
        vr, vc = variant[0]
        for r, c in sorted(landmarks):
            dr, dc = r - vr, c - vc
            cells = frozenset((y + dr, x + dc) for y, x in variant)
            if cells <= landmarks:
                found.add(cells)
    return found


def _union(matches):
    return set().union(*matches) if matches else set()


def _source(obj):
    (gr, gc), missing, _ = obj
    mr, mc = divmod(missing, 2)
    return {(gr + r - mr, gc + c - mc) for r, c in CORNERS
            if (r, c) != (mr, mc)}


def _echo(obj, direction=None):
    _, missing, color = obj
    mr, mc = divmod(missing, 2)
    sign = (1 if color == 1 else -1) if direction is None else direction
    dy, dx = sign * (2 * mr - 1), sign * (2 * mc - 1)
    return {(r + dy, c + dx) for r, c in _source(obj)}


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=False)


def _decode(value):
    inp = _grid(value)
    colors, counts = np.unique(inp, return_counts=True)
    biggest = int(np.argmax(counts))
    if sum(counts == counts[biggest]) != 1:
        raise ValueError("background must be uniquely most frequent")
    bg = int(colors[biggest])
    marker_candidates = set(map(int, colors)) - {bg, 1, 2}
    if bg in (1, 2) or len(marker_candidates) != 1:
        raise ValueError("expected one background, one exemplar color, and source roles 1/2")
    marker = marker_candidates.pop()
    exemplar = {tuple(map(int, p)) for p in np.argwhere(inp == marker)}
    if len(exemplar) < 3:
        raise ValueError("exemplar needs at least three points")
    objects, gaps = [], set()
    for color in (1, 2):
        remaining = {tuple(map(int, p)) for p in np.argwhere(inp == color)}
        while remaining:
            start = min(remaining)
            remaining.remove(start)
            component, stack = {start}, [start]
            while stack:
                r, c = stack.pop()
                for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if p in remaining:
                        remaining.remove(p)
                        component.add(p)
                        stack.append(p)
            r0 = min(r for r, _ in component)
            c0 = min(c for _, c in component)
            relative = {(r-r0, c-c0) for r, c in component}
            if len(component) != 3 or not relative < CORNERS:
                raise ValueError("each source component must be a three-cell L")
            mr, mc = next(iter(CORNERS - relative))
            gap = r0 + mr, c0 + mc
            if inp[gap] != bg or gap in gaps:
                raise ValueError("L gaps must be distinct empty cells")
            gaps.add(gap)
            objects.append((gap, 2*mr+mc, color))
    if not objects:
        raise ValueError("no source Ls")
    return inp, bg, marker, _norm(exemplar), objects


def _paint(inp, bg, marker, objects, active, direction=None):
    out = inp.copy()
    for obj in objects:
        if obj[0] in active:
            for p in _echo(obj, direction):
                if _inside(p, inp.shape) and inp[p] == bg:
                    out[p] = marker
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the local palette, L landmarks and all stencil memberships from input."""
    inp, bg, marker, stencil, objects = _decode(input_grid)
    active = _union(_matches(stencil, [obj[0] for obj in objects]))
    return _paint(inp, bg, marker, objects, active)


def _compatible(obj, objects, shape, protected):
    cells = _source(obj)
    gap = obj[0]
    echo = _echo(obj)
    if (not all(_inside(p, shape) for p in cells | {gap})
            or (cells | {gap} | echo) & protected):
        return False
    for old in objects:
        old_cells = _source(old)
        if gap == old[0] or gap in old_cells or old[0] in cells:
            return False
        if any(abs(r-y) + abs(c-x) <= 1 for r, c in cells for y, x in old_cells):
            return False
        # Construction keeps even inactive hypothetical echoes off source ink.
        if cells & _echo(old) or old_cells & echo:
            return False
    return True


def _objects_at(rng, gaps, objects, shape, protected, clipped=False):
    """Assign independent L orientations around prescribed empty landmarks."""
    trial = list(objects)
    new_gaps = sorted(set(gaps) - {o[0] for o in objects})
    rng.shuffle(new_gaps)
    clip_gap = None
    if clipped:
        edge = [p for p in gaps if p[0] in (0, shape[0]-1)
                or p[1] in (0, shape[1]-1)]
        if not edge:
            return None
        clip_gap = edge[int(rng.integers(len(edge)))]
        if clip_gap in new_gaps:
            new_gaps.remove(clip_gap)
            new_gaps.insert(0, clip_gap)
    for index, gap in enumerate(new_gaps):
        options = [(gap, missing, role) for missing in range(4) for role in (1, 2)]
        rng.shuffle(options)
        if gap == clip_gap:
            options = [o for o in options if o[2] == 1
                       and 0 < sum(_inside(p, shape) for p in _echo(o)) < 3]
        for obj in options:
            if _compatible(obj, trial, shape, protected):
                trial.append(obj)
                break
        else:
            return None
    return trial


def _anchors(rng, variant, shape, objects, edge=False, overlap=False):
    bh, bw = max(r for r, _ in variant)+1, max(c for _, c in variant)+1
    if overlap:
        points = {o[0] for o in objects}
        candidates = {(r-y, c-x) for r, c in points for y, x in variant}
        candidates = [p for p in candidates if 0 <= p[0] <= shape[0]-bh
                      and 0 <= p[1] <= shape[1]-bw]
    else:
        candidates = [(r, c) for r in range(shape[0]-bh+1)
                      for c in range(shape[1]-bw+1)]
    if edge:
        candidates = [(r, c) for r, c in candidates
                      if r in (0, shape[0]-bh) or c in (0, shape[1]-bw)]
    candidates.sort()
    rng.shuffle(candidates)
    return candidates


def _add_group(rng, stencil, objects, shape, protected, variants=None,
               clipped=False, overlap=False):
    choices = list(variants if variants is not None else _variants(stencil))
    rng.shuffle(choices)
    existing = {o[0] for o in objects}
    for variant in choices:
        for ar, ac in _anchors(rng, variant, shape, objects, clipped, overlap)[:160]:
            gaps = {(ar+r, ac+c) for r, c in variant}
            if overlap and not (gaps & existing and gaps - existing):
                continue
            if not overlap and gaps & existing:
                continue
            for _ in range(3):
                trial = _objects_at(rng, gaps, objects, shape, protected, clipped)
                if trial is not None:
                    return trial, frozenset(gaps)
    return None


def _sample_stencil(rng, chiral=False):
    for _ in range(200):
        nr, nc = int(rng.integers(2, 4)), int(rng.integers(2, 5))
        # Irregular axis spacing and independently chosen sites give many shapes,
        # including gapped, symmetric and chiral constellations.
        ys = np.r_[0, np.cumsum(rng.integers(3, 5, size=nr-1))]
        xs = np.r_[0, np.cumsum(rng.integers(3, 5, size=nc-1))]
        if max(xs) > 10:
            continue
        sites = [(int(r), int(c)) for r in ys for c in xs]
        k = int(rng.integers(4, min(6, len(sites))+1))
        chosen = [sites[int(i)] for i in rng.choice(len(sites), k, replace=False)]
        p = _norm(chosen)
        if max(r for r, _ in p) < 3 or max(c for _, c in p) < 3:
            continue
        if chiral and len(_variants(p)) == len(_variants(p, False)):
            continue
        return _variants(p)[int(rng.integers(len(_variants(p))))]
    raise RuntimeError("stencil sampling exhausted")


def _extra_in_hole(rng, stencil, matched, objects, shape, protected):
    r0, r1 = min(r for r, _ in matched), max(r for r, _ in matched)
    c0, c1 = min(c for _, c in matched), max(c for _, c in matched)
    holes = [(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)
             if (r, c) not in matched]
    rng.shuffle(holes)
    before = _matches(stencil, [o[0] for o in objects])
    for gap in holes:
        trial = _objects_at(rng, [gap], objects, shape, protected)
        if trial is not None and len(trial) > len(objects):
            if _matches(stencil, [o[0] for o in trial]) == before:
                return trial
    return None


def _add_partial(rng, stencil, objects, shape, protected):
    before = _matches(stencil, [o[0] for o in objects])
    existing = {o[0] for o in objects}
    choices = list(_variants(stencil))
    rng.shuffle(choices)
    for variant in choices:
        for ar, ac in _anchors(rng, variant, shape, objects)[:75]:
            full = [(ar+r, ac+c) for r, c in variant]
            if set(full) & existing:
                continue
            omit = int(rng.integers(len(full)))
            partial = full[:omit] + full[omit+1:]
            trial = _objects_at(rng, partial, objects, shape, protected)
            if trial is not None and _matches(stencil, [o[0] for o in trial]) == before:
                return trial
    return None


def _strict_active(stencil, gaps):
    """Wrong alternative: require all non-stencil sites in a match box empty."""
    strict = []
    for match in _matches(stencil, gaps):
        r0, r1 = min(r for r, _ in match), max(r for r, _ in match)
        c0, c1 = min(c for _, c in match), max(c for _, c in match)
        inside = {p for p in gaps if r0 <= p[0] <= r1 and c0 <= p[1] <= c1}
        if inside == set(match):
            strict.append(match)
    return _union(strict)


def _alternatives(inp, bg, marker, stencil, objects):
    gaps = {o[0] for o in objects}
    active = _union(_matches(stencil, gaps))
    tops = [(o[0][0]-o[1]//2, o[0][1]-o[1]%2) for o in objects]
    top_active = _union(_matches(stencil, tops))
    corner_wrong = {o[0] for o, top in zip(objects, tops) if top in top_active}
    gap_only = inp.copy()
    for p in active:
        gap_only[p] = marker
    return {
        "all_Ls": _paint(inp, bg, marker, objects, gaps),
        "gap_marks_only": gap_only,
        "all_toward": _paint(inp, bg, marker, objects, active, 1),
        "all_away": _paint(inp, bg, marker, objects, active, -1),
        "box_origins": _paint(inp, bg, marker, objects, corner_wrong),
        "empty_holes_required": _paint(inp, bg, marker, objects,
                                        _strict_active(stencil, gaps)),
    }


def _make_pair(rng, stencil, marker, bg, index, copy_count):
    # First two demonstrations are pedagogic witnesses; all later scenes sample
    # the same rule freely. Tests have a new shape, not a hidden extra operation.
    required_mirror = index == 0
    required_rotation = index == 1
    for _ in range(90):
        shape = (int(rng.integers(20, 31)), int(rng.integers(20, 31)))
        bh, bw = max(r for r, _ in stencil)+1, max(c for _, c in stencil)+1
        er = int(rng.integers(1, shape[0]-bh))
        ec = int(rng.integers(1, shape[1]-bw))
        protected = {(r, c) for r in range(er-1, er+bh+1)
                     for c in range(ec-1, ec+bw+1)}
        rotations = set(_variants(stencil, False))
        variants = _variants(stencil)
        if required_mirror:
            variants = tuple(v for v in variants if v not in rotations)
        elif required_rotation:
            variants = tuple(v for v in variants if v in rotations and v != stencil)
        if not variants:
            raise RuntimeError("teaching stencil lacks a needed orientation")
        placed = _add_group(rng, stencil, [], shape, protected, variants,
                            clipped=(index == 0 or (index > 1 and rng.random() < .3)))
        if placed is None:
            continue
        objects, first_match = placed
        failed = False
        for _copy in range(copy_count-1):
            result = _add_group(rng, stencil, objects, shape, protected,
                                overlap=bool(rng.random() < .45))
            if result is None:
                result = _add_group(rng, stencil, objects, shape, protected)
            if result is None:
                failed = True
                break
            objects, _ = result
        if failed:
            continue
        with_hole = _extra_in_hole(rng, stencil, first_match, objects, shape, protected)
        if with_hole is None:
            continue
        objects = with_hole
        partial = _add_partial(rng, stencil, objects, shape, protected)
        if partial is None:
            continue
        objects = partial
        wanted = int(rng.integers(1, 5))
        matches_before = _matches(stencil, [o[0] for o in objects])
        for _noise in range(wanted):
            for _try in range(50):
                gap = (int(rng.integers(shape[0])), int(rng.integers(shape[1])))
                trial = _objects_at(rng, [gap], objects, shape, protected)
                if (trial is not None and len(trial) > len(objects)
                        and _matches(stencil, [o[0] for o in trial]) == matches_before):
                    objects = trial
                    break
        inp = np.full(shape, bg, dtype=np.int8)
        for r, c in stencil:
            inp[er+r, ec+c] = marker
        for obj in objects:
            for p in _source(obj):
                inp[p] = obj[2]
        active = _union(matches_before)
        selected = [o for o in objects if o[0] in active]
        if {o[2] for o in selected} != {1, 2}:
            continue
        if len({o[1] for o in selected}) < 2:
            continue
        out = _paint(inp, bg, marker, objects, active)
        alternatives = _alternatives(inp, bg, marker, stencil, objects)
        # Each scene distinguishes the six principal shortcut rules.
        if any(np.array_equal(out, a) for a in alternatives.values()):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("could not build a teaching-quality scene")


def _check_counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    marker = int(rng.choice(FREE_COLORS))
    backgrounds = [c for c in FREE_COLORS if c != marker]
    rng.shuffle(backgrounds)
    puzzle, seen, train_families = {"train": [], "test": []}, set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            pedagogic_index = i if split == "train" else num_train+i
            for _retry in range(30):
                stencil = _sample_stencil(rng, chiral=(split == "train" and i < 2))
                if split == "test" and _family(stencil) in train_families:
                    continue
                # Copy count is not a train/test code. The first demonstration
                # has one complete match to make its reflection and hole clear.
                count_copies = 1 if pedagogic_index == 0 else int(rng.integers(1, 4))
                pair = _make_pair(rng, stencil, marker,
                                  backgrounds[pedagogic_index % len(backgrounds)],
                                  pedagogic_index, count_copies)
                key = _key(pair["input"])
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == "train":
                    train_families.add(_family(stencil))
                break
            else:
                raise RuntimeError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _pair_info(pair):
    inp, bg, marker, stencil, objects = _decode(pair["input"])
    out = _grid(pair["output"])
    assert out.shape == inp.shape
    gaps = {o[0] for o in objects}
    matches = _matches(stencil, gaps)
    active = _union(matches)
    assert 4 <= len(stencil) <= 6 and len(matches) >= 1
    assert {o[2] for o in objects if o[0] in active} == {1, 2}
    assert gaps - active, "need inactive source Ls"
    assert len({o[1] for o in objects if o[0] in active}) >= 2
    # Check protection and layout from actual returned pixels, not placement plans.
    assert np.array_equal(out[inp != bg], inp[inp != bg])
    assert np.all(out[inp != out] == marker)
    source_cells = set().union(*(_source(o) for o in objects))
    exemplar_cells = {tuple(map(int, p)) for p in np.argwhere(inp == marker)}
    for o in objects:
        assert not (_echo(o) & (source_cells | exemplar_cells))
    assert np.array_equal(out, solve(inp))
    # A direct cell-by-cell formula checks the drawing arithmetic independently
    # of _echo and _paint. This is generator-owned validation, not blind solving.
    direct = inp.copy()
    for gap, missing, role in objects:
        if gap not in active:
            continue
        mr, mc = divmod(missing, 2)
        s = 1 if role == 1 else -1
        for r in range(2):
            for c in range(2):
                if (r, c) == (mr, mc):
                    continue
                p = gap[0]+r-mr+s*(2*mr-1), gap[1]+c-mc+s*(2*mc-1)
                if _inside(p, inp.shape) and inp[p] == bg:
                    direct[p] = marker
    assert np.array_equal(direct, out)
    for name, alternative in _alternatives(inp, bg, marker, stencil, objects).items():
        assert not np.array_equal(alternative, out), f"scene does not refute {name}"
    rotations = set(_variants(stencil, False))
    partial = False
    for variant in _variants(stencil):
        for r, c in sorted(gaps):
            # Align every possible occupied site, including cases where site 0
            # is the omitted one in a deliberately incomplete copy.
            for vr, vc in variant:
                cells = {(r+y-vr, c+x-vc) for y, x in variant}
                present = cells & gaps
                if (len(present) == len(stencil)-1 and present.isdisjoint(active)
                        and all(_inside(p, inp.shape) for p in cells)):
                    partial = True
                    break
            if partial:
                break
        if partial:
            break
    assert partial, "need a near-complete inactive constellation"
    return {
        "marker": marker, "background": bg, "family": _family(stencil),
        "matches": len(matches), "objects": len(objects),
        "active": len(active), "points": len(stencil),
        "mirror": any(_norm(m) not in rotations for m in matches),
        "rotation": any(_norm(m) in rotations and _norm(m) != stencil for m in matches),
        "clipped": any(0 < sum(_inside(p, inp.shape) for p in _echo(o)) < 3
                       for o in objects if o[0] in active),
        "shared": sum(map(len, matches)) > len(active),
        "orientations": {(o[2], o[1]) for o in objects if o[0] in active},
    }


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, info = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _key(_grid(pair["input"]))
            assert key not in seen
            seen.add(key)
            info[split].append(_pair_info(pair))
    all_info = info["train"] + info["test"]
    assert len({f["marker"] for f in all_info}) == 1
    assert len({f["background"] for f in all_info}) >= 2
    assert any(f["mirror"] for f in info["train"])
    assert any(f["rotation"] for f in info["train"])
    assert any(f["clipped"] for f in info["train"])
    trained = {f["family"] for f in info["train"]}
    assert all(f["family"] not in trained for f in info["test"])


def self_test():
    """Bounded 42-seed diversity, semantic witnesses, replay and count checks."""
    signatures, dimensions, families, palettes = set(), set(), set(), set()
    counts, points, orientations = set(), set(), set()
    shared = mirrors = clips = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(tuple(_key(x[k]) for s in ("train", "test")
                             for x in p[s] for k in ("input", "output")))
        for split in ("train", "test"):
            for pair in p[split]:
                f = _pair_info(pair)
                dimensions.add(np.asarray(pair["input"]).shape)
                families.add(f["family"])
                palettes.add((f["marker"], f["background"]))
                counts.add(f["matches"])
                points.add(f["points"])
                orientations.update(f["orientations"])
                shared += f["shared"]
                mirrors += f["mirror"]
                clips += f["clipped"]
    assert len(signatures) == 42 and len(dimensions) >= 45
    assert len(families) >= 70 and len(palettes) >= 25
    assert {1, 2, 3} <= counts and points == {4, 5, 6}
    assert len(orientations) == 8 and shared and mirrors and clips
    a, b = generate_puzzle(2026, 2, 1), generate_puzzle(2026, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s])
               for k in ("input", "output"))
    witness = a["train"][0]
    before = _key(witness["input"])
    for flip in (False, True):
        source = np.fliplr(witness["input"]) if flip else witness["input"]
        target = np.fliplr(witness["output"]) if flip else witness["output"]
        for turns in range(4):
            assert np.array_equal(solve(np.rot90(source, turns)), np.rot90(target, turns))
    # Reassign locally recognizable palette roles without altering blue/red.
    inp, bg, marker, _, _ = _decode(witness["input"])
    changed_in, changed_out = inp.copy(), witness["output"].copy()
    changed_in[inp == bg], changed_in[inp == marker] = marker, bg
    changed_out[witness["output"] == bg] = marker
    changed_out[witness["output"] == marker] = bg
    assert np.array_equal(solve(changed_in), changed_out)
    assert _key(witness["input"]) == before, "solve modified its input"
    for nt, ne in ((10, 4), (5, 3)):
        p = generate_puzzle(100+nt, nt, ne)
        assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {nt}/{ne}")
    return {"seeds": 42, "pairs": 210, "grid_shapes": len(dimensions),
            "stencil_families": len(families), "palettes": len(palettes),
            "shared_membership_pairs": shared, "mirror_pairs": mirrors,
            "clipped_pairs": clips, "match_counts": sorted(counts)}
