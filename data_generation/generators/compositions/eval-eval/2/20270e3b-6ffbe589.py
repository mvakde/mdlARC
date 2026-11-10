"""Collision-powered assembly: registered coincidences turn merged color masks.

Only standard-library and NumPy dependencies. The public solver reads pixels;
generation constructs its answer separately with array rotations.
"""
from itertools import permutations
import json

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(a, b):
    if not _integer(a) or not 2 <= a <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(b) or not 1 <= b <= 4 or a <= b:
        raise ValueError("num_test must be in 1..4 and less than num_train")
    return int(a), int(b)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(cells, diagonal=True):
    left, groups = set(cells), []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or not (r and c))]
    while left:
        p = min(left)
        left.remove(p)
        group, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _box(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _shift(cells, dr, dc):
    return {(r + dr, c + dc) for r, c in cells}


def _rotate(p, n, k):
    r, c = p
    for _ in range(k % 4):
        r, c = c, n - 1 - r
    return r, c


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a


def _registrations(value):
    """Enumerate the geometric grammar, never using outputs or generation state."""
    grid = _grid(value)
    vals, nums = np.unique(grid, return_counts=True)
    if len(vals) < 4 or np.sum(nums == nums.max()) != 1:
        return []
    bg = int(vals[nums.argmax()])
    occupied = set(map(tuple, np.argwhere(grid != bg).tolist()))
    fragments = _components(occupied)
    if len(fragments) != 2:
        return []
    result = []
    for marker in map(int, vals):
        if marker == bg:
            continue
        bars = _components(set(map(tuple, np.argwhere(grid == marker).tolist())), False)
        if len(bars) != 2 or len(bars[0]) != len(bars[1]) or not 2 <= len(bars[0]) <= 5:
            continue
        length = len(bars[0])
        for axis in (0, 1):
            # axis=0: normal row, tangent column; axis=1: the reverse.
            if any(len({p[axis] for p in bar}) != 1 or
                   max(p[1-axis] for p in bar) - min(p[1-axis] for p in bar) + 1 != length
                   for bar in bars):
                continue
            records = []
            for part in fragments:
                own = [bar for bar in bars if bar <= part]
                if len(own) != 1:
                    break
                bar = own[0]
                normal = min(p[axis] for p in bar)
                tangent = min(p[1-axis] for p in bar)
                fg = part - bar
                oriented = {(p[axis] - normal, p[1-axis] - tangent) for p in fg}
                neg = any((-1, c) in oriented for c in range(length))
                pos = any((1, c) in oriented for c in range(length))
                if neg == pos:
                    break
                records.append((neg, fg, bar, normal, tangent))
            if len(records) != 2 or records[0][0] == records[1][0]:
                continue
            records.sort(key=lambda x: not x[0])
            anchor, mover = records
            dn, dt = anchor[3] - 1 - mover[3], anchor[4] - mover[4]
            dr, dc = (dn, dt) if axis == 0 else (dt, dn)
            palette = tuple(int(v) for v in vals if v not in (bg, marker))
            a = {color: {p for p in anchor[1] if grid[p] == color} for color in palette}
            b = {color: _shift({p for p in mover[1] if grid[p] == color}, dr, dc)
                 for color in palette}
            if any(a[x] & b[y] for x in palette for y in palette if x != y):
                continue
            union = set().union(*a.values(), *b.values())
            top, left, bottom, right = _box(union)
            if bottom - top != right - left or bottom - top > 29:
                continue
            a = {color: _shift(mask, -top, -left) for color, mask in a.items()}
            b = {color: _shift(mask, -top, -left) for color, mask in b.items()}
            result.append(dict(background=bg, marker=marker, palette=palette,
                               a=a, b=b, size=bottom-top+1, axis=axis,
                               length=length, delta=(dr, dc),
                               initial=[anchor[1], mover[1]], bars=[anchor[2], mover[2]]))
    return result


def _parse(value):
    candidates = _registrations(value)
    if len(candidates) != 1:
        raise ValueError("input must have one unambiguous registration")
    return candidates[0]


def _bbox_alignment_differs(info):
    tangent = 1-info["axis"]
    minima = [min(p[tangent] for p in part | bar)
              for part,bar in zip(info["initial"],info["bars"])]
    return info["delta"][tangent] != minima[0]-minima[1]


def _predict(info, count_mode="overlap", direction=1, blend="union"):
    n, bg = info["size"], info["background"]
    out = np.full((n, n), bg, dtype=np.int8)
    for color in info["palette"]:
        a, b = info["a"][color], info["b"][color]
        modes = {"overlap": len(a & b), "union_area": len(a | b),
                 "sum_area": len(a) + len(b), "anchor_area": len(a),
                 "mover_area": len(b), "zero": 0, "one": 1,
                 "guide_length": info["length"], "contact_boolean": int(bool(a & b))}
        k = modes[count_mode]
        cells = (a | b) if blend == "union" else (a ^ b)
        for p in sorted(cells):
            q = _rotate(p, n, direction * k)
            if out[q] != bg:
                return None
            out[q] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Translate by guides, count equal-color collisions, union, and turn masks."""
    result = _predict(_parse(input_grid))
    if result is None:
        raise ValueError("rotated layers conflict")
    return result


def _orbits(n):
    left, result = {(r, c) for r in range(n) for c in range(n)}, []
    while left:
        p = min(left)
        orbit = {_rotate(p, n, k) for k in range(4)}
        left -= orbit
        result.append(sorted(orbit))
    return result


def _source_geometry(rng, n, s, length, offset, layer_count, wrap_required=False):
    """Compose silhouettes and a jagged partition, independent of literal colors."""
    yy, xx = np.indices((n, n))
    style = int(rng.integers(6))
    thickness = int(rng.integers(1, 3))
    middle = (n - 1) / 2
    if style == 0:  # Outlines, with crossbars.
        base = np.minimum.reduce([yy, xx, n-1-yy, n-1-xx]) < thickness
    elif style == 1:  # Filled mass with several rectangular cavities/notches.
        base = np.ones((n, n), dtype=bool)
        for _ in range(int(rng.integers(1, 5))):
            r, c = map(int, rng.integers(0, n-2, 2))
            h, w = map(int, rng.integers(1, max(2, n//3), 2))
            base[r:r+h, c:c+w] = False
    elif style == 2:  # Uneven cross with teeth.
        base = (np.abs(xx-middle) <= thickness) | (np.abs(yy-middle) <= thickness)
        for r in range(n):
            if rng.random() < .5:
                lo, hi = sorted(map(int, rng.integers(0, n, 2)))
                base[r, lo:hi+1] = True
    elif style == 3:  # Chamfered/diamond silhouette.
        base = np.abs(yy-middle) + np.abs(xx-middle) <= middle + thickness
    elif style == 4:  # Branched, variable-length strokes.
        base = np.zeros((n, n), dtype=bool)
        spine = int(rng.integers(n//3, 2*n//3+1))
        for r in range(n):
            lo = int(rng.integers(spine+1))
            hi = int(rng.integers(spine, n))
            base[r, lo:hi+1] = True
        base[:, spine] = True
    else:  # Nested interrupted outlines.
        depth = np.minimum.reduce([yy, xx, n-1-yy, n-1-xx])
        base = depth % int(rng.integers(2, 4)) == 0
        base[:, n//2] = True
    # The middle band keeps both jagged halves connected and backs the bars.
    base[max(0, s-3):min(n, s+3), :] = True
    base[:, n//2] = True
    # One complete extremal orbit makes the crop square before/after every turn.
    for p in {_rotate((0, n//2), n, k) for k in range(4)}:
        base[p] = True
    boundary = rng.integers(s-3, s+2, size=n)
    boundary[offset:offset+length] = s-1
    # Let one piece wrap around the other at either edge. Their tangent bbox
    # origins then differ: simply lining up bounding boxes cannot replace bars.
    wrap = int(rng.integers(1 if wrap_required else 0,4))
    side_owner = bool(rng.integers(2))
    if wrap in (1,3):
        width = int(rng.integers(1,min(3,offset)+1))
        boundary[:width] = n-1 if side_owner else -1
    if wrap in (2,3):
        width = int(rng.integers(1,min(3,n-offset-length)+1))
        boundary[-width:] = -1 if side_owner else n-1
    a = set(map(tuple, np.argwhere(base & (yy <= boundary[None, :])).tolist()))
    b = set(map(tuple, np.argwhere(base & (yy > boundary[None, :])).tolist()))
    ga = {(s, c) for c in range(offset, offset+length)}
    gb = {(s-1, c) for c in range(offset, offset+length)}
    if len(_components(a | ga)) != 1 or len(_components(b | gb)) != 1:
        return None
    # Rotation-invariant territories avoid color conflicts, with no overwrite rule.
    orbits = _orbits(n)
    geometry = int(rng.integers(3))
    jitter = rng.random(len(orbits))
    if geometry == 0:
        score = [min(r, c, n-1-r, n-1-c) for r, c in (o[0] for o in orbits)]
    elif geometry == 1:
        ar, ac = rng.uniform(0, middle, 2)
        anchors = [(ar, ac), (ac, n-1-ar), (n-1-ar, n-1-ac), (n-1-ac, ar)]
        score = [min((r-u)**2+(c-v)**2 for u, v in anchors) for r, c in (o[0] for o in orbits)]
    else:
        score = rng.random(len(orbits)) * n
    order = np.argsort(np.asarray(score) + jitter * .8)
    # Modest size differences, rather than forcing one dominant outer color.
    allocation = np.full(layer_count, 2, dtype=int)
    allocation += rng.multinomial(len(orbits)-2*layer_count, rng.dirichlet(np.full(layer_count, 3.)))
    owner, cursor = {}, 0
    for color_index in rng.permutation(layer_count):
        for idx in order[cursor:cursor+allocation[color_index]]:
            for p in orbits[int(idx)]:
                owner[p] = int(color_index)
        cursor += int(allocation[color_index])
    # Candidate duplicates touch the opposite mask and cannot overwrite a guide.
    near_a = {(r+dr, c+dc) for r, c in a for dr in (-1,0,1) for dc in (-1,0,1)}
    near_b = {(r+dr, c+dc) for r, c in b for dr in (-1,0,1) for dc in (-1,0,1)}
    candidates = (a & (near_b - gb)) | (b & (near_a - ga))
    capacities = [sorted(p for p in candidates if owner[p] == k) for k in range(layer_count)]
    return a, b, ga, gb, owner, capacities


def _stage(rng, colored_a, colored_b, ga, gb, bg, marker, stage):
    groups = []
    for colors, guide in ((colored_a, ga), (colored_b, gb)):
        top, left, bottom, right = _box(set(colors) | guide)
        groups.append(({(r-top, c-left): v for (r,c),v in colors.items()},
                       _shift(guide, -top, -left), bottom-top+1, right-left+1))
    if rng.integers(2):
        groups.reverse()
    ac, ag, ah, aw = groups[0]
    bc, bguide, bh, bw = groups[1]
    gap = int(rng.integers(1, 5))
    if stage == "horizontal":
        dr, dc = int(rng.integers(-bh+1, ah)), aw+gap
    elif stage == "vertical":
        dr, dc = ah+gap, int(rng.integers(-bw+1, aw))
    else:
        dr, dc = ah+gap, aw+gap
    cells = dict(ac)
    cells.update({(r+dr,c+dc):v for (r,c),v in bc.items()})
    guides = ag | _shift(bguide, dr, dc)
    top, left, bottom, right = _box(set(cells) | guides)
    h, w = bottom-top+1, right-left+1
    if h > 30 or w > 30:
        return None
    # Independent zero/positive margins, conditioned only on finite canvas fit.
    pads = [int(rng.integers(0, 4)) for _ in range(4)]
    pads[0] = min(pads[0], 30-h)
    pads[1] = min(pads[1], 30-h-pads[0])
    pads[2] = min(pads[2], 30-w)
    pads[3] = min(pads[3], 30-w-pads[2])
    grid = np.full((h+pads[0]+pads[1], w+pads[2]+pads[3]), bg, dtype=np.int8)
    for (r,c),v in cells.items():
        grid[r-top+pads[0], c-left+pads[2]] = v
    for r,c in guides:
        grid[r-top+pads[0], c-left+pads[2]] = marker
    vals, nums = np.unique(grid, return_counts=True)
    if int(vals[nums.argmax()]) != bg or np.sum(nums == nums.max()) != 1:
        return None
    return grid


def _make_pair(rng, witness, length, stage, cap, long_required, used, reject_models=()):
    for attempt in range(450):
        layers = 3 if witness in (0,1) else int(rng.integers(2,5))
        n = int(rng.integers(9,14))
        s = int(rng.integers(3, n-2))
        offset = int(rng.integers(1, n-length))
        data = _source_geometry(rng, n, s, length, offset, layers, witness == 0)
        if data is None:
            continue
        a, b, ga, gb, owner, capacities = data
        if witness in (0,1):
            target = (1,2,0) if witness == 0 else (3,4,0)
            options = [p for p in permutations(target) if all(p[k] <= len(capacities[k]) for k in range(layers))]
            if not options:
                continue
            counts = list(options[int(rng.integers(len(options)))])
        else:
            counts = [int(rng.integers(0, min(cap,len(v))+1)) for v in capacities]
            if long_required:
                eligible = [k for k,v in enumerate(capacities) if len(v) >= 5]
                if not eligible:
                    continue
                k = int(rng.choice(eligible))
                counts[k] = int(rng.integers(5, min(cap,len(capacities[k]))+1))
        if not any(k % 4 for k in counts):
            continue
        for k, count in enumerate(counts):
            for index in rng.choice(len(capacities[k]), count, replace=False):
                p = capacities[k][int(index)]
                if p in a:
                    b.add(p)
                else:
                    a.add(p)
        palette = list(map(int, rng.choice(10, layers+2, replace=False)))
        bg, marker, colors = palette[0], palette[1], palette[2:]
        ca = {p: colors[owner[p]] for p in a}
        cb = {p: colors[owner[p]] for p in b}
        source = np.full((n,n), bg, dtype=np.int8)
        for p in a | b:
            source[p] = colors[owner[p]]
        if any(not np.any(source == c) for c in colors):
            continue
        if witness in (0,1) and any(any(np.array_equal(source == c, np.rot90(source == c,k))
                                      for k in (1,2,3)) for c in colors):
            continue
        output = np.full((n,n), bg, dtype=np.int8)
        for k,c in enumerate(colors):
            mask = np.rot90(source == c, -counts[k])
            if np.any((output != bg) & mask):
                raise AssertionError("orbit territories failed")
            output[mask] = c
        if np.array_equal(output, source):
            continue
        inp = _stage(rng, ca, cb, ga, gb, bg, marker, stage)
        if inp is None:
            continue
        rotation = int(rng.integers(4))
        inp, output = np.rot90(inp,rotation).copy(), np.rot90(output,rotation).copy()
        if _key(inp) in used:
            continue
        try:
            parsed = _parse(inp)
        except ValueError:
            continue
        if not np.array_equal(_predict(parsed), output):
            raise AssertionError("construction and pixel registration disagree")
        # Every pair makes both counted turning and registration observable.
        if np.array_equal(_predict(parsed,"zero"), output):
            continue
        if witness == 0 and not _bbox_alignment_differs(parsed):
            continue
        if any(np.array_equal(_predict(parsed,*model),output) for model in reject_models):
            continue
        return {"input": inp, "output": output}
    raise RuntimeError("bounded collision geometry sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    ntrain, ntest = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(3))
    puzzle, used = {"train": [], "test": []}, set()
    for split, number in (("train",ntrain),("test",ntest)):
        for index in range(number):
            witness = index if split == "train" and index < 2 else -1
            if regime == 1:
                length = (2 if index % 2 == 0 else 4) if split == "train" else 3
            else:
                length = int(rng.integers(2,6))
            if regime == 2:
                stage = str(rng.choice(["horizontal","vertical"])) if split == "train" else "diagonal"
            else:
                stage = str(rng.choice(["horizontal","vertical","diagonal"]))
            cap = 4 if regime == 0 and split == "train" else 8
            reject_models = []
            if witness == 1:
                first = puzzle["train"][0]
                first_info = _parse(first["input"])
                reject_models = [model for model in _ALTERNATIVES
                                 if np.array_equal(_predict(first_info,*model),first["output"])]
            pair = _make_pair(rng,witness,length,stage,cap,
                              regime == 0 and split == "test",used,reject_models)
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


_ALTERNATIVES = [(mode,direction,blend)
                 for mode in ("overlap","union_area","sum_area","anchor_area","mover_area",
                              "zero","one","guide_length","contact_boolean")
                 for direction in (1,-1) for blend in ("union","xor")
                 if (mode,direction,blend) != ("overlap",1,"union")]


def _analyze(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _parse(inp)
    expected = _predict(info)
    assert expected is not None and np.array_equal(out,expected), "wrong registration/count/turn/union"
    assert out.shape == (info["size"],info["size"])
    n, bg = info["size"], info["background"]
    orbit_owners = {}
    for color in info["palette"]:
        a,b = info["a"][color],info["b"][color]
        assert np.sum(out == color) == len(a)+len(b)-len(a & b), "union area not conserved"
        for p in a | b:
            orbit = min(_rotate(p,n,k) for k in range(4))
            assert orbit_owners.setdefault(orbit,color) == color, "color territories conflict"
    assert not np.any(out == info["marker"])
    assert all(np.any(out[edge] != bg) for edge in (0,-1))
    assert np.any(out[:,0] != bg) and np.any(out[:,-1] != bg)
    assert not np.array_equal(out,_predict(info,"zero")), "no visible counted turn"
    # Omitting A leaves the two staged pieces separate: there are no collisions.
    staged = set.union(*info["initial"])
    top,left,bottom,right = _box(staged)
    unregistered = np.full((bottom-top+1,right-left+1),bg,dtype=np.int8)
    for r,c in staged:
        unregistered[r-top,c-left] = inp[r,c]
    assert not np.array_equal(out,unregistered), "registration has no effect"
    info["bbox_alignment_differs"] = _bbox_alignment_differs(info)
    info["counts"] = {c:len(info["a"][c] & info["b"][c]) for c in info["palette"]}
    return info


def _staging(info):
    boxes = [_box(part | bar) for part,bar in zip(info["initial"],info["bars"])]
    a,b = boxes
    x = a[3] < b[1] or b[3] < a[1]
    y = a[2] < b[0] or b[2] < a[0]
    return "diagonal" if x and y else "axial"


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    _counts(len(puzzle["train"]),len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            key = _key(_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _analyze(pair)
            assert 2 <= len(info["palette"]) <= 4
            if split == "train":
                training.append(info)
    # The teaching scaffold is checked from actual pixels, not sampling labels.
    assert sorted(training[0]["counts"].values()) == [0,1,2]
    assert sorted(training[1]["counts"].values()) == [0,3,4]
    assert training[0]["bbox_alignment_differs"], "guides must defeat bbox alignment"
    for info in training[:2]:
        n = info["size"]
        for color in info["palette"]:
            cells = info["a"][color] | info["b"][color]
            assert all({_rotate(p,n,k) for p in cells} != cells for k in (1,2,3))
        # Changing a duplicate into a single occurrence leaves the union intact
        # but must change a turn. This tests the new interaction specifically.
        color = next(c for c,k in info["counts"].items() if k > 0)
        changed = dict(info)
        changed["a"] = {c:set(mask) for c,mask in info["a"].items()}
        changed["a"][color].remove(min(info["a"][color] & info["b"][color]))
        assert changed["a"][color] | info["b"][color] == info["a"][color] | info["b"][color]
        assert not np.array_equal(_predict(changed),_predict(info))
    for alternative in _ALTERNATIVES:
        survivors = all(np.array_equal(_predict(info,*alternative),pair["output"])
                        for info,pair in zip(training,puzzle["train"]))
        assert not survivors, "training leaves a count/direction/blend shortcut"


def self_test():
    """Bounded semantic, negative, interface and 42-seed variation checks."""
    shapes, sizes, palettes, structures, counts, axes, lengths = set(),set(),set(),set(),set(),set(),set()
    layers, fingerprints, rejected, regimes = set(),set(),set(),set()
    stages, foreground_areas, collision_patterns = set(),set(),set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprint = tuple(_key(p["input"]) for s in puzzle for p in puzzle[s])
        fingerprints.add(fingerprint)
        infos = {s:[_analyze(p) for p in puzzle[s]] for s in puzzle}
        train_counts = [k for i in infos["train"] for k in i["counts"].values()]
        test_counts = [k for i in infos["test"] for k in i["counts"].values()]
        if max(train_counts) <= 4 and max(test_counts) >= 5:
            regimes.add("collision_extrapolation")
        if {i["length"] for i in infos["train"]} == {2,4} and {i["length"] for i in infos["test"]} == {3}:
            regimes.add("guide_interpolation")
        if all(_staging(i) == "axial" for i in infos["train"]) and all(_staging(i) == "diagonal" for i in infos["test"]):
            regimes.add("staging_transfer")
        for split in puzzle:
            for pair in puzzle[split]:
                info = _analyze(pair)
                assert np.array_equal(solve(pair["input"].copy()),pair["output"])
                shapes.add(pair["input"].shape)
                sizes.add(info["size"])
                axes.add(info["axis"])
                lengths.add(info["length"])
                palettes.add((info["background"],info["marker"],info["palette"]))
                counts.update(info["counts"].values())
                layers.add(len(info["palette"]))
                stages.add(_staging(info))
                collision_patterns.add(tuple(sorted(info["counts"].values())))
                foreground_areas.add(sum(len(info["a"][c] | info["b"][c]) for c in info["palette"]))
                # Binary masks strip literal palette and staging differences.
                structures.add((info["size"],tuple(sorted(set().union(*info["a"].values()))),
                                tuple(sorted(set().union(*info["b"].values())))))
                for model in _ALTERNATIVES:
                    if not np.array_equal(_predict(info,*model),pair["output"]):
                        rejected.add(model)
        # One-cell output corruption must fail meaningful semantic validation.
        bad = {"input":puzzle["test"][0]["input"].copy(),
               "output":puzzle["test"][0]["output"].copy()}
        bad["output"][0,0] = (int(bad["output"][0,0])+1)%10
        try:
            _analyze(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupt output accepted")
    assert len(fingerprints) == 42 and len(shapes) >= 40
    assert sizes == set(range(9,14)) and axes == {0,1} and lengths == {2,3,4,5}
    assert layers == {2,3,4} and counts >= set(range(9))
    assert len(palettes) >= 100 and len(structures) >= 150
    assert len(rejected) == len(_ALTERNATIVES)
    assert regimes == {"collision_extrapolation","guide_interpolation","staging_transfer"}
    assert stages == {"axial","diagonal"} and len(foreground_areas) >= 50
    assert len(collision_patterns) >= 40
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            p = generate_puzzle(600+a*10+b,a,b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1),(4,None)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first,second = generate_puzzle(123,2,1),generate_puzzle(123,2,1)
    assert all(np.array_equal(p[k],q[k]) for s in first for p,q in zip(first[s],second[s])
               for k in ("input","output"))
    return {"seeds":42,"input_shapes":len(shapes),"binary_fragment_structures":len(structures),
            "palettes":len(palettes),"square_sizes":sorted(sizes),"collision_counts":sorted(counts),
            "collision_patterns":len(collision_patterns),"foreground_areas":len(foreground_areas),
            "train_test_regimes":sorted(regimes)}


if __name__ == "__main__":
    print(json.dumps(self_test(),separators=(",",":")))
