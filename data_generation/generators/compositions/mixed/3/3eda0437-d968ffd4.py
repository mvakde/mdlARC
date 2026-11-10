"""Largest blank window controls opposed-pad, full-grid slice painting.

Standard library + NumPy only. No file access, retained examples, or generator
state is used by solve(). Coordinates in internal gap masks are always
(transverse row, axial column); vertical inputs are transposed for analysis.
"""
from __future__ import annotations

from collections import Counter, deque
import hashlib
import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} outside allowed range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a


def _box(a, color):
    points = np.argwhere(a == color)
    r0, c0 = map(int, points.min(axis=0))
    r1, c1 = (int(v) + 1 for v in points.max(axis=0))
    return (r0, r1, c0, c1) if len(points) == (r1-r0)*(c1-c0) else None


def _rectangles(mask):
    """Maximal-width runs for every >=2-row span; sufficient for area maxima.

    A non-maximal-width rectangle cannot be a global area winner because its
    horizontal extension is strictly larger and remains eligible. Bitsets keep
    the at-most-30-column search small without saving execution traces.
    """
    h, w = mask.shape
    rows = [sum(1 << int(c) for c in np.flatnonzero(row)) for row in mask]
    found = []
    for top in range(h - 1):
        common = (1 << w) - 1
        for bottom in range(top, h):
            common &= rows[bottom]
            if not common:
                break
            if bottom == top:
                continue
            bits = common
            while bits:
                left = (bits & -bits).bit_length() - 1
                shifted = bits >> left
                width = (shifted ^ (shifted + 1)).bit_length() - 1
                bits &= ~(((1 << width) - 1) << left)
                if width >= 2:
                    height = bottom - top + 1
                    found.append((height * width, top, left, height, width))
    return found


def _winner(mask):
    candidates = _rectangles(mask)
    if not candidates:
        raise ValueError("no two-dimensional blank rectangle")
    area = max(x[0] for x in candidates)
    winners = [x[1:] for x in candidates if x[0] == area]
    if len(winners) != 1:
        raise ValueError("largest blank rectangle is tied")
    return winners[0], candidates


def _analyze(input_grid):
    a = _grid(input_grid)
    background = int(a[0, 0])
    if not (np.all(a[0] == background) and np.all(a[-1] == background)
            and np.all(a[:, 0] == background) and np.all(a[:, -1] == background)):
        raise ValueError("input must have a uniform background outer rim")
    colors = [int(c) for c in np.unique(a) if c != background]
    if len(colors) != 3:
        raise ValueError("expected two pad colors and one obstacle/separator color")
    boxes = {c: _box(a, c) for c in colors}
    pads = [c for c in colors if boxes[c] is not None]
    if len(pads) != 2:
        raise ValueError("exactly two non-background colors must form solid rectangles")
    separator = next(c for c in colors if c not in pads)
    x, y = (boxes[c] for c in pads)
    horizontal = max(x[0], y[0]) < min(x[1], y[1])
    vertical = max(x[2], y[2]) < min(x[3], y[3])
    if horizontal and (x[3] + 2 <= y[2] or y[3] + 2 <= x[2]):
        axis, normal = 1, a
    elif vertical and (x[1] + 2 <= y[0] or y[1] + 2 <= x[0]):
        axis, normal = 0, a.T
    else:
        raise ValueError("pads must oppose along exactly one axis with a gap >=2")
    normal_boxes = {c: _box(normal, c) for c in pads}
    near, far = sorted(pads, key=lambda c: normal_boxes[c][2])
    start, stop = normal_boxes[near][3], normal_boxes[far][2]
    _, obstacle_cols = np.nonzero(normal == separator)
    if not np.all((obstacle_cols >= start) & (obstacle_cols < stop)):
        raise ValueError("all obstacles must lie strictly between the pad faces")
    mask = normal[:, start:stop] == background
    rect, candidates = _winner(mask)
    return {"axis": axis, "normal": normal, "background": background,
            "separator": separator, "near": near, "far": far,
            "boxes": (normal_boxes[near], normal_boxes[far]),
            "start": start, "stop": stop, "mask": mask, "rect": rect,
            "candidates": candidates}


def _render(normal, start, left, length, near, far, separator):
    out = normal.copy()
    first, last = start + left, start + left + length
    half = length // 2
    out[:, first:first + half] = near
    out[:, last - half:last] = far
    if length % 2:
        out[:, first + half] = separator
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, search scope, winner and painted slices from the input."""
    info = _analyze(input_grid)
    _, left, _, length = info["rect"]
    out = _render(info["normal"], info["start"], left, length,
                  info["near"], info["far"], info["separator"])
    return out if info["axis"] == 1 else out.T.copy()


def _features(info):
    mask = info["mask"]
    top, left, height, length = info["rect"]
    area = height * length
    others = [x for x in info["candidates"]
              if not (top <= x[1] and left <= x[2]
                      and x[1] + x[3] <= top + height
                      and x[2] + x[4] <= left + length)]
    overlap = [x for x in others
               if max(top, x[1]) < min(top + height, x[1] + x[3])
               and max(left, x[2]) < min(left + length, x[2] + x[4])]
    # Component size is a diagnostic only: it never selects the answer.
    todo, component = deque([(top, left)]), {(top, left)}
    while todo:
        r, c = todo.popleft()
        for nr, nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]
                    and mask[nr,nc] and (nr,nc) not in component):
                component.add((nr,nc))
                todo.append((nr,nc))
    longest_line = 0
    for rows in (mask, mask.T):
        for row in rows:
            run = 0
            for value in row:
                run = run + 1 if value else 0
                longest_line = max(run, longest_line)
    return {"area": area, "length": length, "height": height,
            "parity": length % 2, "competitors": len(others),
            "runner_up": max((x[0] for x in others), default=0),
            "longer_loser": any(x[4] > length for x in others),
            "broader_loser": any(x[3] > height for x in others),
            "overlap": bool(overlap), "connected_extra": len(component) > area,
            "component_interval_larger": max(c for _,c in component)-min(c for _,c in component)+1 > length,
            "long_line": longest_line > area,
            "partial": length < mask.shape[1],
            "offcenter": 2 * left + length != mask.shape[1],
            "overwrite": bool(np.any(~mask[:,left:left+length])),
            "transverse_edge": top == 0 or top + height == mask.shape[0],
            "axial_edge": left == 0 or left + length == mask.shape[1]}


def _carve_ranked(rng, parity, maximum_gap, extra):
    # A wider, shorter winner competes with a narrow but longer rectangle.
    # The ever-present single-row rim is longer than the winner's area.
    shapes = [(h,w) for h in range(3,6) for w in range(3,7)
              if w % 2 == parity and h*w < maximum_gap
              and (h*w-1)//2 > w]
    h, w = shapes[int(rng.integers(len(shapes)))]
    gap = int(rng.integers(h*w+1, maximum_gap+1))
    rival_width = int(rng.integers(w+1, (h*w-1)//2+1))
    depths = [h,2] + [2] * extra
    spacing = [int(rng.integers(1,3)) for _ in range(len(depths)-1)]
    margin0, margin1 = int(rng.integers(2,4)), int(rng.integers(2,4))
    cross = sum(depths) + sum(spacing) + margin0 + margin1
    mask = np.zeros((cross,gap), dtype=bool)
    mask[0] = mask[-1] = True
    boxes, r = [], margin0
    for index, d in enumerate(depths):
        width = w if index == 0 else rival_width if index == 1 else int(
            rng.integers(2, min(gap, (h*w-1)//d)+1))
        c = int(rng.integers(0,gap-width+1))
        mask[r:r+d,c:c+width] = True
        boxes.append((r,c,d,width))
        if index < len(spacing):
            r += d + spacing[index]
    top, left, _, _ = boxes[0]
    # One-cell appendages produce overlapping eligible alternatives, while
    # leaving the planted rectangle larger than their stretched subrectangles.
    choices = []
    if top > 1:
        choices.append((top-1,left,1,w-1))
    if left > 0:
        choices.append((top,left-1,h-1,1))
    if left+w < gap:
        choices.append((top+1,left+w,h-1,1))
    rng.shuffle(choices)
    for r,c,dh,dw in choices[:int(rng.integers(1,len(choices)+1))]:
        mask[r:r+dh,c:c+dw] = True
    # Connect pockets by a narrow bent corridor in some cases. This changes
    # connected components without replacing rectangle-area selection.
    if rng.random() < 0.8:
        for a,b in zip(boxes, boxes[1:]):
            ca = int(rng.integers(a[1],a[1]+a[3]))
            cb = int(rng.integers(b[1],b[1]+b[3]))
            middle = a[0]+a[2]
            mask[middle:b[0]+1,cb] = True
            mask[middle,min(ca,cb):max(ca,cb)+1] = True
    # Small background holes and rotations of the transverse arrangement
    # remove a fixed top/bottom winner convention.
    for _ in range(int(rng.integers(0,5))):
        mask[int(rng.integers(1,cross-1)),int(rng.integers(gap))] = True
    if bool(rng.integers(2)):
        mask = mask[::-1].copy()
    if bool(rng.integers(2)):
        mask = mask[:,::-1].copy()
    return mask


def _carve_other(rng, profile, maximum_gap):
    gap = int(rng.integers(7,maximum_gap+1))
    cross = int(rng.integers(6,27))
    if profile == "open":
        mask = rng.random((cross,gap)) > float(rng.uniform(.12,.40))
        for _ in range(int(rng.integers(1,5))):
            if bool(rng.integers(2)):
                row = int(rng.integers(1,cross-1))
                c = int(rng.integers(gap-2))
                mask[row,c:min(gap,c+int(rng.integers(2,8)))] = False
            else:
                col = int(rng.integers(gap))
                r = int(rng.integers(1,cross-2))
                mask[r:min(cross-1,r+int(rng.integers(2,8))),col] = False
    else:
        mask = np.zeros((cross,gap),dtype=bool)
        count = 1 if profile == "small" else int(rng.integers(3,8))
        for _ in range(count):
            h = 2 if profile == "small" else int(rng.integers(2,min(10,cross-2)+1))
            w = int(rng.integers(2,4)) if profile == "small" else int(rng.integers(2,min(14,gap)+1))
            r, c = int(rng.integers(1,cross-h)), int(rng.integers(gap-w+1))
            mask[r:r+h,c:c+w] = True
        if profile != "small":
            mask |= rng.random((cross,gap)) < float(rng.uniform(.03,.12))
    mask[0] = mask[-1] = True
    return mask


def _make_pair(rng, axis, parity, palette, profile, extra=0, strong=False):
    background, near, far, separator = palette
    for _ in range(192):
        ends = [int(rng.integers(1,3)) for _ in range(2)]
        depths = [int(rng.integers(1,4)) for _ in range(2)]
        maximum_gap = min(24,30-sum(ends)-sum(depths))
        mask = (_carve_ranked(rng,parity,maximum_gap,extra) if profile == "ranked"
                else _carve_other(rng,profile,maximum_gap))
        cross, gap = mask.shape
        if cross > 30:
            continue
        try:
            rect, candidates = _winner(mask)
        except ValueError:
            continue
        if rect[3] % 2 != parity:
            continue
        # The provisional analysis has no palette dependence.
        features = _features({"mask":mask,"rect":rect,"candidates":candidates})
        if not (features["partial"] and features["overwrite"]):
            continue
        if strong and not (features["longer_loser"] and features["long_line"]
                           and features["overlap"] and features["offcenter"]
                           and features["component_interval_larger"]):
            continue
        normal = np.full((cross,gap+sum(ends)+sum(depths)),background,dtype=np.int8)
        start = ends[0]+depths[0]
        stop = start+gap
        normal[:,start:stop] = np.where(mask,background,separator)
        span0 = int(rng.integers(2,cross-1))
        r0 = int(rng.integers(1,cross-span0))
        span1, r1 = span0, r0
        if rng.random() < .4:
            span1 = max(2,min(cross-2,span0+int(rng.integers(-2,3))))
            r1 = max(1,min(cross-span1-1,r0+int(rng.integers(-1,2))))
            if max(r0,r1) >= min(r0+span0,r1+span1):
                span1, r1 = span0, r0
        normal[r0:r0+span0,ends[0]:start] = near
        normal[r1:r1+span1,stop:stop+depths[1]] = far
        inp = normal if axis == 1 else normal.T.copy()
        try:
            info = _analyze(inp)
        except ValueError:
            continue
        assert info["rect"] == rect
        out = _render(normal,start,rect[1],rect[3],near,far,separator)
        return {"input":inp,"output":out if axis == 1 else out.T.copy()}
    raise RuntimeError(f"bounded sampling exhausted: {profile}/{parity}/{strong}")


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _shape(info):
    return info["rect"][2:]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed,"seed",0)
    nt = _integer(num_train,"num_train",2,10)
    ns = _integer(num_test,"num_test",1,4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    base = [int(c) for c in rng.choice(10,4,replace=False)]
    stable = rng.random() < .25
    first_axis, first_parity = int(rng.integers(2)), int(rng.integers(2))
    train_plan = [(first_axis,first_parity),(1-first_axis,1-first_parity)]
    remaining = [(first_axis,1-first_parity),(1-first_axis,first_parity)]
    rng.shuffle(remaining)
    train_plan += remaining[:max(0,min(nt,4)-2)]
    train_plan += [(int(rng.integers(2)),int(rng.integers(2))) for _ in range(max(0,nt-4))]
    puzzle, seen, trained_shapes = {"train":[],"test":[]}, set(), set()
    for split,count in (("train",nt),("test",ns)):
        for index in range(count):
            plan = train_plan[index] if split == "train" else train_plan[int(rng.integers(nt))]
            if split == "train" and index == 0:
                palette = base
            elif split == "train" and index == 1:
                palette = [base[1],base[3],base[0],base[2]]
            else:
                palette = [int(c) for c in rng.permutation(base)] if stable else [
                    int(c) for c in rng.choice(10,4,replace=False)]
            anchor = split == "train" and index < 2
            if anchor:
                profile, extra = "ranked",0
            elif split == "test":
                profile = ("ranked","open","cavities","small")[int(rng.choice(4,p=[.4,.3,.25,.05]))]
                extra = int(rng.integers(1,3)) if profile == "ranked" else 0
            else:
                profile = ("open","cavities","small","ranked")[int(rng.choice(4,p=[.35,.35,.15,.15]))]
                extra = int(rng.integers(2)) if profile == "ranked" else 0
            for attempt in range(96):
                # A finite small/ranked profile might have all its target
                # shapes already taught in a ten-demonstration puzzle. Broaden
                # the geometry rather than retrying an impossible holdout.
                if split == "test" and attempt == 12:
                    profile, extra = "cavities",0
                if split == "test" and attempt == 36:
                    profile, extra = "open",0
                pair = _make_pair(rng,*plan,palette,profile,extra,strong=anchor)
                key = _key(pair["input"])
                shape = _shape(_analyze(pair["input"]))
                if key in seen or (split == "test" and shape in trained_shapes):
                    continue
                seen.add(key)
                if split == "train":
                    trained_shapes.add(shape)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded distinct/held-out-shape sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in ("train","test"))
    nt, ns = len(puzzle["train"]),len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, analyses, features = set(), {"train":[],"test":[]}, {"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            inp,out = _grid(pair["input"]),_grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            old, new = info["normal"],out if info["axis"] == 1 else out.T
            t,l,h,w = info["rect"]
            assert h >= 2 and w >= 2
            assert np.all(info["mask"][t:t+h,l:l+w])
            first, last, half = info["start"]+l,info["start"]+l+w,w//2
            # Check the entire output column by column, including overwritten
            # obstacles, both parities, untouched end zones and original pads.
            for col in range(old.shape[1]):
                expected = (info["near"] if first <= col < first+half else
                            info["far"] if last-half <= col < last else
                            info["separator"] if w%2 and col == first+half else None)
                if expected is None:
                    assert np.array_equal(new[:,col],old[:,col])
                else:
                    assert np.all(new[:,col] == expected)
            for color,box in zip((info["near"],info["far"]),info["boxes"]):
                r0,r1,c0,c1 = box
                assert np.all(new[r0:r1,c0:c1] == color)
            f = _features(info)
            assert f["partial"] and f["overwrite"]
            assert not np.array_equal(inp,out)
            # Necessity: omitting A paints the entire gap, not this interval.
            without_a = _render(old,info["start"],0,info["stop"]-info["start"],
                                info["near"],info["far"],info["separator"])
            assert not np.array_equal(new,without_a)
            # Omitting B gives A's uniform rectangle highlight; also reject
            # restricting the three-color split to the rectangle footprint.
            without_b = old.copy()
            without_b[t:t+h,first:last] = info["separator"]
            assert not np.array_equal(new,without_b)
            clipped = old.copy()
            clipped[t:t+h,first:last] = new[t:t+h,first:last]
            assert not np.array_equal(new,clipped)
            analyses[split].append(info)
            features[split].append(f)
    train = analyses["train"]
    cases = {(i["axis"],i["rect"][3]%2) for i in train}
    assert {i["axis"] for i in train} == {0,1}
    assert {i["rect"][3]%2 for i in train} == {0,1}
    assert len(cases) >= min(nt,4)
    assert train[1]["background"] == train[0]["near"]
    for f in features["train"][:2]:
        assert all(f[k] for k in ("longer_loser","long_line","overlap","offcenter","component_interval_larger"))
    for info in analyses["test"]:
        assert _shape(info) not in {_shape(x) for x in train}
        assert (info["axis"],info["rect"][3]%2) in cases


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train","test"):
        for pair in puzzle[split]:
            for role in ("input","output"):
                a = np.asarray(pair[role],dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded streaming checks: 42 seeds, all counts, and equivariances.

    Returns small descriptive measurements; saves nothing. These are
    generator-owned semantic checks, not independent approval.
    """
    stats = Counter()
    shapes, targets, structures, signatures, test_cases = set(),set(),set(),set(),set()
    role_colors = {k:set() for k in ("background","near","far","separator")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_digest(puzzle))
        for split in ("train","test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                f = _features(info)
                stats["pairs"] += 1
                for name,value in f.items():
                    if isinstance(value,bool):
                        stats[name] += value
                shapes.add(pair["input"].shape)
                targets.add(_shape(info))
                structures.add((info["axis"],info["mask"].shape,info["mask"].tobytes()))
                for role in role_colors:
                    role_colors[role].add(info[role])
                stats["odd" if f["parity"] else "even"] += 1
                if split == "test":
                    test_cases.add((info["axis"],f["parity"]))
                before = pair["input"].copy()
                assert np.array_equal(solve(before),pair["output"])
                assert np.array_equal(before,pair["input"])
        # Spatial and palette equivariances test recognition and endpoint
        # reversal, including different transpose coordinates in the search.
        pair = puzzle["test"][0]
        for transform in (np.transpose,np.fliplr,np.flipud,lambda a:np.rot90(a)):
            assert np.array_equal(solve(transform(pair["input"])),transform(pair["output"]))
        permutation = np.random.default_rng(10000+seed).permutation(10)
        assert np.array_equal(solve(permutation[pair["input"]]),permutation[pair["output"]])
    assert len(signatures) == 42 and len(shapes) >= 40 and len(targets) >= 20
    assert len(structures) >= 180 and len(test_cases) == 4
    assert all(len(s) >= 9 for s in role_colors.values())
    assert stats["overlap"] >= 84 and stats["longer_loser"] >= 84
    assert stats["long_line"] >= 84 and stats["connected_extra"] >= 84
    assert stats["transverse_edge"] >= 5 and stats["axial_edge"] >= 10
    assert _puzzle_digest(generate_puzzle(7)) == _puzzle_digest(generate_puzzle(7))
    legal = 0
    for nt in range(2,11):
        for ns in range(1,min(4,nt-1)+1):
            p = generate_puzzle(9001,nt,ns)
            assert (len(p["train"]),len(p["test"])) == (nt,ns)
            legal += 1
    for nt,ns in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),
                  (4.0,1),(4,1.5),(None,1),(4,None),(-1,1),(4,-1)):
        try:
            generate_puzzle(0,nt,ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Kernel witnesses: 3x3 wins over overlapping 2x4, and a 1x12
    # background run is ignored despite exceeding both rectangles in area.
    witness = np.zeros((7,12),dtype=bool)
    witness[0] = True
    witness[2:5,3:6] = True
    witness[2:4,6] = True
    assert _winner(witness)[0] == (2,3,3,3)
    tied = np.zeros((6,8),dtype=bool)
    tied[0:2,0:2] = tied[4:6,6:8] = True
    try:
        _winner(tied)
    except ValueError:
        pass
    else:
        raise AssertionError("area ties must be rejected")
    return {"seeds":42,"pairs":stats["pairs"],"legal_count_combinations":legal,
            "distinct_puzzles":len(signatures),"input_shapes":len(shapes),
            "target_shapes":len(targets),"structures":len(structures),
            "test_axis_parities":sorted(test_cases),
            "colors_per_role":{k:len(v) for k,v in role_colors.items()},
            "features":dict(sorted(stats.items()))}
