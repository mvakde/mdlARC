"""Docked aperture gates select the finite boundaries of a diagonal weave.

Only Python's standard library and NumPy are used. Generation and all diagnostics
are bounded and in memory; importing this module writes nothing.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import numpy as np


HOST_SEEDS = (0, 698505480, 938992490, 36200407, 484535723, 655496839,
              810529614, 272249957, 96571447, 355718544, 970285121, 884094249,
              642317967, 807889085, 639575820, 18813526, 573996496, 651864765,
              714756968, 351886946, 537938058, 364129607, 453249237, 591235219,
              24859668, 262145288, 13268848, 812826251, 599022400, 143913651,
              523883693, 726991101, 524897740, 260343692, 209970806, 533790518,
              560569341, 47610579, 195326279, 366865674, 965667971, 716698967)
ALTERNATIVES = ("all_strokes", "open_column", "all_holes", "local_wall", "endpoints",
                "consecutive", "pixel_centers", "unbounded")


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(name + " is out of range")
    return value


def _counts(num_train, num_test):
    a = _integer(num_train, "num_train", 2, 10)
    b = _integer(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must be greater than num_test")
    return a, b


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "invalid ARC color")
    return grid


def _key(grid):
    grid = np.asarray(grid)
    return grid.shape, grid.astype(np.int8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    # Sorted starts ensure reproducibility independent of hash iteration order.
    for start in sorted(remaining):
        if start not in remaining:
            continue
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for neighbor in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        result.append(cells)
    return result


def _profile(mask):
    """Four-connected enclosure; the entire component defines its front."""
    h, w = mask.shape
    holes, active, decoy, enclosed, local = [], set(), set(), set(), set()
    for region in _components(~mask):
        if any(r in (0, h-1) or c in (0, w-1) for r, c in region):
            continue
        holes.append(region)
        enclosed.update(region)
        for r in {r for r, c in region}:
            local_end = int(np.flatnonzero(mask[r])[-1])
            hole_end = max(c for rr, c in region if rr == r)
            if hole_end == local_end-1:
                local.add(r)
        contacts = {r for r, c in region if c == w-2 and mask[r, w-1]}
        active.update(contacts)
        if not contacts:
            decoy.update(r for r, c in region)
    naive = {r for r in range(h) if not mask[r, w-2] and mask[r, w-1]}
    open_rows = {r for r in naive if (r, w-2) not in enclosed}
    return dict(holes=holes, active=active, decoy=decoy, naive=naive,
                open=open_rows, closed={r for r, c in enclosed}, local=local)


def _objects(work, background):
    objects = []
    for cells in _components(work != background):
        rr, cc = np.array(cells).T
        r0, r1, c0, c1 = map(int, (rr.min(), rr.max(), cc.min(), cc.max()))
        _require(r1 > r0 and c1 > c0 and len(cells) >= 4,
                 "work objects must be substantial four-connected shapes")
        colors = {int(work[r, c]) for r, c in cells}
        _require(len(colors) == 1, "a work object must be monochrome")
        mask = np.zeros((r1-r0+1, c1-c0+1), dtype=bool)
        mask[rr-r0, cc-c0] = True
        objects.append(dict(cells=cells, row=r0, bottom=r1, right=c1,
                            color=colors.pop(), mask=mask, profile=_profile(mask)))
    objects.sort(key=lambda o: o["row"])
    _require(bool(objects), "missing gate objects")
    _require(all(b["row"] > a["bottom"] + 1 for a, b in zip(objects, objects[1:])),
             "object row bands must have a blank separator")
    return objects


def _strokes(band, background):
    h, b = band.shape
    cells = [(int(r), int(c)+1) for r, c in np.argwhere(band != background)]
    candidates = []
    for slope in (1, -1):
        lines = {}
        for r, depth in cells:
            lines.setdefault(r-slope*depth, []).append((r, depth))
        strokes, valid = [], bool(lines)
        for q, pixels in sorted(lines.items()):
            pixels.sort(key=lambda p: p[1])
            depths = [d for r, d in pixels]
            if (not 0 <= q < h or len(pixels) < 2 or
                    depths != list(range(depths[0], depths[-1]+1))):
                valid = False
                break
            colors = {int(band[r, d-1]) for r, d in pixels}
            if len(colors) != 1:
                valid = False
                break
            times = [r+slope*d for r, d in pixels]
            strokes.append(dict(q=q, lo=min(times), hi=max(times),
                                start=depths[0], end=depths[-1],
                                color=colors.pop(), length=len(pixels)))
        if valid and all(b["q"]-a["q"] >= 2 for a, b in zip(strokes, strokes[1:])):
            candidates.append((slope, strokes))
    _require(len(candidates) == 1, "receiver must have one recognizable diagonal slope")
    return candidates[0]


def _parse(input_grid):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    _require(len(colors) == 4 and counts.max() > grid.size/2,
             "input needs four roles and a majority background")
    background = int(colors[np.argmax(counts)])
    candidates = []
    for turn in range(4):
        g = np.rot90(grid, turn)
        h, w = g.shape
        for col in range(2, w-2):
            rail = int(g[0, col])
            if rail == background or not np.all(g[:, col] == rail):
                continue
            if np.count_nonzero(g == rail) != h:
                continue
            left, right = g[:, :col], g[:, col+1:]
            wall_colors = set(map(int, np.unique(left))) - {background}
            stroke_colors = set(map(int, np.unique(right))) - {background}
            if len(wall_colors) != 1 or len(stroke_colors) != 1 or wall_colors == stroke_colors:
                continue
            try:
                objects = _objects(left, background)
                slope, strokes = _strokes(right, background)
            except AssertionError:
                continue
            gates = {o["row"]+r for o in objects for r in o["profile"]["active"]}
            candidates.append(dict(grid=g, turn=turn, bg=background, rail=rail,
                                   col=col, objects=objects, slope=slope,
                                   strokes=strokes, gates=gates))
    _require(len(candidates) == 1, "rail and object side must be uniquely recognizable")
    return candidates[0]


def _selected(info, alternative=""):
    gates = info["gates"]
    if alternative in ("open_column", "all_holes", "local_wall"):
        feature = {"open_column": "naive", "all_holes": "closed", "local_wall": "local"}[alternative]
        gates = {o["row"]+r for o in info["objects"] for r in o["profile"][feature]}
    if alternative == "all_strokes":
        return info["strokes"]
    if alternative == "endpoints":
        return [s for s in info["strokes"]
                if s["q"]+info["slope"]*s["start"] in gates]
    return [s for s in info["strokes"] if s["q"] in gates]


def _coordinates(info):
    h, w = info["grid"].shape
    rows, depths = np.indices((h, w-info["col"]-1))
    depths = depths+1
    return rows-info["slope"]*depths, rows+info["slope"]*depths


def _fill(info, alternative=""):
    """Label construction: a sweep, including scanlines between pixel centers."""
    q, t = _coordinates(info)
    selected = _selected(info, alternative)
    mask = np.zeros(q.shape, dtype=bool)
    if alternative == "consecutive":
        for a, b in zip(selected, selected[1:]):
            mask |= ((t >= max(a["lo"], b["lo"])) & (t <= min(a["hi"], b["hi"]))
                     & (q > a["q"]) & (q < b["q"]))
    elif alternative == "unbounded":
        if len(selected) >= 2:
            mask = (q > selected[0]["q"]) & (q < selected[-1]["q"])
    else:
        for level in range(int(t.min()), int(t.max())+1):
            active = [s["q"] for s in selected if s["lo"] <= level <= s["hi"]
                      and (alternative != "pixel_centers" or (level-s["q"]) % 2 == 0)]
            if len(active) >= 2:
                mask |= (t == level) & (q > min(active)) & (q < max(active))
    return mask & (info["grid"][:, info["col"]+1:] == info["bg"])


def _pairwise_fill(info):
    """Validation cross-check: union every finite pair's overlap parallelogram."""
    q, t = _coordinates(info)
    mask = np.zeros(q.shape, dtype=bool)
    for a, b in itertools.combinations(_selected(info), 2):
        mask |= ((q > min(a["q"], b["q"])) & (q < max(a["q"], b["q"]))
                 & (t >= max(a["lo"], b["lo"])) & (t <= min(a["hi"], b["hi"])))
    return mask & (info["grid"][:, info["col"]+1:] == info["bg"])


def _output(info, mask=None):
    result = info["grid"].copy()
    col = info["col"]
    result[:, :col] = info["bg"]
    for obj in info["objects"]:
        shift = col-1-obj["right"]
        for r, c in obj["cells"]:
            result[r, c+shift] = obj["color"]
    if mask is None:
        mask = _fill(info)
    result[:, col+1:][mask] = info["rail"]
    return np.rot90(result, -info["turn"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, aperture, stroke, and output cell from this input."""
    return _output(_parse(input_grid))


def _differences(info):
    correct = _fill(info)
    return {name: int(np.count_nonzero(correct != _fill(info, name)))
            for name in ALTERNATIVES}


def _make_gate(rng, rooms, width, *, simple=False, force_recessed=False):
    top = 1 if simple else int(rng.integers(1, 3))
    bottom = 1 if simple else int(rng.integers(1, 3))
    gaps = [1 if simple else int(rng.integers(1, 3)) for _ in rooms[:-1]]
    height = top+bottom+sum(length for kind, length in rooms)+sum(gaps)
    mask = np.ones((height, width), dtype=bool)
    row = top
    for index, (kind, length) in enumerate(rooms):
        if kind == "open":
            mask[row:row+length, :-1] = False
        elif kind != "solid":
            end = width-2 if kind == "active" else width-3
            if kind == "decoy" and not simple and width >= 6 and rng.random() < .35:
                end -= 1
            back = 1 if simple else int(rng.integers(1, end+1))
            for r in range(row, row+length):
                edge = back if simple or rng.random() < .5 else int(rng.integers(1, end+1))
                mask[r, edge:end+1] = False
            if kind == "decoy" and (simple or force_recessed or rng.random() < .65):
                # A thin local wall remains; the global leading extreme lies
                # farther forward on the object's connecting crossbars.
                mask[row:row+length, end+2:] = False
        row += length
        if index < len(gaps):
            row += gaps[index]
    if not simple:
        # These two corners cannot change enclosure or disconnect the mask.
        for r in (0, height-1):
            if rng.random() < .45:
                mask[r, 0] = False
    _require(len(_components(mask)) == 1, "builder disconnected an object")
    return mask


def _spaced_subset(rng, values, count, chosen=()):
    result = list(chosen)
    for value in rng.permutation(sorted(set(values))).tolist():
        if len(result) >= count:
            break
        if all(abs(value-old) >= 2 for old in result):
            result.append(int(value))
    return sorted(result)


def _canonical_candidate(rng, purpose, palette):
    """Jointly build gates and the strokes they do and do not enable."""
    witness = purpose in ("open", "decoy")
    if witness:
        rooms = [("active", int(rng.integers(5, 8))),
                 (purpose, int(rng.integers(1, 4)))]
        if rng.random() < .2:
            rooms.append(("active", int(rng.integers(1, 3))))
    elif purpose == "test":
        rooms = [("active", int(rng.integers(3, 6))),
                 ("open", int(rng.integers(1, 4))),
                 ("decoy", int(rng.integers(1, 4)))]
        if rng.random() < .3:
            rooms.append(("active", int(rng.integers(1, 4))))
    elif rng.random() < .12:
        rooms = [("open", int(rng.integers(1, 4))),
                 ("decoy", int(rng.integers(1, 4)))]
    else:
        inactive = str(rng.choice(["open", "decoy", "solid"]))
        rooms = [("active", int(rng.integers(2, 7))),
                 (inactive, int(rng.integers(1, 4)))]
        if rng.random() < .35:
            rooms.append(("active", int(rng.integers(1, 4))))
    rng.shuffle(rooms)
    width = int(rng.integers(4, 10))
    masks = [_make_gate(rng, rooms, width, force_recessed=purpose == "decoy")]
    # Extra bodies include small solids, single/multiple ports and open shapes.
    for _ in range(int(rng.integers(0, 3))):
        kind = str(rng.choice(["solid", "active", "open", "decoy"]))
        if kind == "solid":
            extra = np.ones((int(rng.integers(2, 5)), int(rng.integers(2, 7))), bool)
        else:
            extra = _make_gate(rng, [(kind, int(rng.integers(1, 3)))],
                               int(rng.integers(4, 8)), simple=True)
        if sum(m.shape[0]+1 for m in masks)+extra.shape[0] <= 21:
            masks.append(extra)
    rng.shuffle(masks)
    used = sum(m.shape[0] for m in masks)+len(masks)-1
    if used > 23:
        raise ValueError("source objects leave too little receiver headroom")
    height = int(rng.integers(max(21, used+5), 31))
    band = int(rng.integers(10, 17))
    widest = max(m.shape[1] for m in masks)
    rail_col = int(rng.integers(widest+1, min(13, 29-band)+1))
    bg, wall, rail, marker = palette
    grid = np.full((height, rail_col+1+band), bg, dtype=np.int8)
    grid[:, rail_col] = rail
    # Source blank space varies independently of object ranks and categories.
    padding = rng.multinomial(height-5-used, np.full(len(masks)+1, 1/(len(masks)+1)))
    row = int(padding[0])
    objects = []
    for index, mask in enumerate(masks):
        available = rail_col-mask.shape[1]
        shift = 0 if rng.random() < .25 else int(rng.integers(1, available+1))
        if witness and index == 0 and shift == 0:
            shift = 1
        start = rail_col-mask.shape[1]-shift
        view = grid[row:row+mask.shape[0], start:start+mask.shape[1]]
        view[mask] = wall
        objects.append((row, mask, _profile(mask)))
        row += mask.shape[0]+1+int(padding[index+1])
    gates = {r0+r for r0, mask, p in objects for r in p["active"]}
    if purpose == "open":
        long_holes = [(r0, hole) for r0, mask, p in objects for hole in p["holes"]
                      if {r for r, c in hole} <= p["active"]
                      and len({r for r, c in hole}) >= 5]
        r0, hole = long_holes[int(rng.integers(len(long_holes)))]
        rows = sorted({r0+r for r, c in hole})
        base = int(rng.integers(rows[0], rows[-1]-3))
        enabled = [base, base+2, base+4]
    else:
        enabled = _spaced_subset(rng, gates, int(rng.integers(2, 6)))
    requested = ["open"] if purpose == "open" else ["decoy"] if purpose == "decoy" else (
        ["open", "decoy"] if purpose == "test" else [])
    all_q = enabled[:]
    for feature in requested:
        options = [r0+r for r0, mask, p in objects for r in p[feature]
                   if all(abs(r0+r-q) >= 2 for q in all_q)]
        if not options:
            raise ValueError("missing a separated disabled boundary")
        all_q.append(int(rng.choice(options)))
    candidates = [r for r in range(height-5) if r not in gates]
    all_q = _spaced_subset(rng, candidates, len(all_q)+int(rng.integers(0, 3)), all_q)
    if len(all_q) < 2 or (purpose != "random" and len(enabled) < 2):
        raise ValueError("insufficient gate or stroke rows")
    max_depth = {q: min(band, height-1-q) for q in all_q}
    low = max(q+2 for q in all_q)+4
    high = min(q+2*max_depth[q] for q in all_q)-4
    if high < low:
        raise ValueError("no continuous overlap with endpoint freedom")
    core = int(rng.integers(low, high+1))
    specs = {}
    for q in all_q:
        lower, upper = q+2, q+2*max_depth[q]
        if purpose == "open" and q == enabled[1]:
            lo = core-1
            lo += (q-lo) % 2
            hi = lo+2*int(rng.integers(1, 3))
        else:
            before = int(rng.integers(4, 9)) if purpose == "open" and q in enabled else int(rng.integers(0, 9))
            after = int(rng.integers(6, 10)) if purpose == "open" and q in enabled else int(rng.integers(1, 10))
            lo = max(lower, core-before)
            lo += (q-lo) % 2
            hi = min(upper, core+after)
            hi -= (hi-q) % 2
            if hi <= lo:
                hi = min(upper, lo+2)
                lo = max(lower, hi-2)
        specs[q] = (lo, hi)
    # Some ordinary examples have a chain of overlaps with no common interval.
    if purpose != "open" and len(enabled) >= 3 and high-low >= 6 and rng.random() < .4:
        order = list(map(int, rng.permutation(enabled)))
        windows = [(core-7, core-1), (core-4, core+4), (core+1, core+7)]
        for q, (lo, hi) in zip(order[:3], windows):
            lo = max(q+2, lo)
            lo += (q-lo) % 2
            hi = min(q+2*max_depth[q], hi)
            hi -= (hi-q) % 2
            if hi > lo:
                specs[q] = (lo, hi)
    for q, (lo, hi) in specs.items():
        if hi <= lo:
            raise ValueError("stroke must have at least two pixels")
        for t in range(lo, hi+1, 2):
            r, depth = (t+q)//2, (t-q)//2
            grid[r, rail_col+depth] = marker
    return grid


def _fallback(purpose, palette, index):
    """Sixteen distinct canvas shapes: at most thirteen can be occupied."""
    bg, wall, rail, marker = palette
    h, band, col = 26+index//4, 12+index % 4, 9
    grid = np.full((h, col+1+band), bg, dtype=np.int8)
    grid[:, col] = rail
    kind = "decoy" if purpose == "decoy" else "open"
    rooms = [("active", 5), (kind, 1)]
    if purpose == "test":
        rooms.append(("decoy", 1))
    mask = _make_gate(np.random.default_rng(0), rooms, 5, simple=True)
    grid[1:1+mask.shape[0], 1:6][mask] = wall
    specs = [(2, 1, 10), (4, 5, 6), (6, 1, 8), (8, 2, 7)]
    if purpose == "test":
        specs.append((10, 1, 6))
    for q, start, end in specs:
        for depth in range(start, end+1):
            grid[q+depth, col+depth] = marker
    return grid


def _make_pair(rng, purpose, palette, flip, turn, seen):
    for attempt in range(112):
        try:
            grid = (_canonical_candidate(rng, purpose, palette) if attempt < 96
                    else _fallback(purpose, palette, attempt-96))
            if flip:
                grid = np.flipud(grid)
            grid = np.rot90(grid, turn).copy()
            if _key(grid) in seen:
                continue
            info = _parse(grid)
            if purpose != "random":
                differences = _differences(info)
                # Keep the essential receiver change visible in a large scene.
                # This is a sampling preference, not a solver special case.
                minimum_fill = {"open": 8, "decoy": 4, "test": 8}[purpose]
                if int(_fill(info).sum()) < minimum_fill or not differences["all_strokes"]:
                    continue
                if purpose == "open" and not all(differences[k] for k in
                        ("open_column", "consecutive", "pixel_centers", "endpoints", "unbounded")):
                    continue
                if purpose == "decoy" and not all(differences[k] for k in ("all_holes", "local_wall")):
                    continue
            return {"input": grid, "output": _output(info)}
        except (ValueError, AssertionError):
            continue
    raise RuntimeError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    a, b = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    base_palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    varying_palette = bool(rng.random() < .35)
    base_turn = int(rng.integers(4))
    other_turn = (base_turn+int(rng.integers(1, 4))) % 4
    turns = [base_turn, other_turn] if rng.random() < .35 else [base_turn]
    base_flip = bool(rng.integers(2))
    flips = [base_flip, not base_flip] if rng.random() < .3 else [base_flip]
    puzzle, seen = {"train": [], "test": []}, set()
    purposes = ["open", "decoy"]+["random"]*(a-2)
    rng.shuffle(purposes)
    for split, count in (("train", a), ("test", b)):
        for index in range(count):
            # Every orientation/slope used at test has a training demonstration.
            turn = turns[index % len(turns)] if split == "train" else int(rng.choice(turns))
            flip = flips[index % len(flips)] if split == "train" else bool(rng.choice(flips))
            palette = (tuple(map(int, rng.choice(10, 4, replace=False)))
                       if varying_palette else base_palette)
            purpose = purposes[index] if split == "train" else "test"
            pair = _make_pair(rng, purpose, palette, flip, turn, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "invalid split")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, differences = set(), {"train": [], "test": []}
    teaching = {"motion": False, "active": False, "open": False, "decoy": False}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions must be preserved")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _parse(inp)
            expected = _output(info, _pairwise_fill(info))
            _require(np.array_equal(expected, out), "incorrect motion, gate selection or finite weave")
            _require(np.array_equal(_fill(info), _pairwise_fill(info)), "sweep/pairwise disagreement")
            canonical_out = np.rot90(out, info["turn"])
            col = info["col"]
            _require(np.array_equal(canonical_out[:, col], info["grid"][:, col]), "rail moved")
            band = info["grid"][:, col+1:]
            occupied = band != info["bg"]
            _require(np.array_equal(canonical_out[:, col+1:][occupied], band[occupied]),
                     "an existing diagonal stroke was overwritten")
            _require(set(map(int, np.unique(out))) <= set(map(int, np.unique(inp))),
                     "output introduced an unencoded color")
            for obj in info["objects"]:
                for hole in obj["profile"]["holes"]:
                    rows = {r for r, c in hole}
                    contact = {r for r, c in hole if c == obj["mask"].shape[1]-2
                               and obj["mask"][r, -1]}
                    _require(not contact or rows == contact, "partial-contact cavity outside intended domain")
                if split == "train":
                    teaching["motion"] |= obj["right"] < col-1
                    for feature in ("active", "open", "decoy"):
                        teaching[feature] |= bool(obj["profile"][feature])
            differences[split].append(_differences(info))
    _require(all(teaching.values()), "training lacks motion or one of the three aperture categories")
    for name in ALTERNATIVES:
        if any(d[name] for d in differences["test"]):
            _require(any(d[name] for d in differences["train"]), "untaught alternative: "+name)
    _require(any(d["all_strokes"] for d in differences["train"]), "parent A is inessential")
    _require(any(d["unbounded"] for d in differences["train"]), "parent B endpoints are inessential")


def _features(pair):
    info = _parse(pair["input"])
    selected = _selected(info)
    profiles = [o["profile"] for o in info["objects"]]
    mask = _fill(info)
    roles = [info["bg"], info["rail"], info["objects"][0]["color"], info["strokes"][0]["color"]]
    normalized = np.zeros(info["grid"].shape, dtype=np.int8)
    for index, color in enumerate(roles):
        normalized[info["grid"] == color] = index
    signature = hashlib.sha256(bytes(normalized.shape)+normalized.tobytes()).hexdigest()
    return dict(shape=list(np.asarray(pair["input"]).shape), structure=signature,
                objects=len(profiles), strokes=len(info["strokes"]), enabled=len(selected),
                holes=sum(len(p["holes"]) for p in profiles),
                open_rows=sum(len(p["open"]) for p in profiles),
                decoy_rows=sum(len(p["decoy"]) for p in profiles),
                active_rows=len(info["gates"]), slope=info["slope"], turn=info["turn"],
                shifts=[info["col"]-1-o["right"] for o in info["objects"]],
                depths=[s["start"] for s in info["strokes"]],
                lengths=[s["length"] for s in info["strokes"]],
                palette=roles, fill_cells=int(mask.sum()),
                staggered=len(selected) >= 3 and max(s["lo"] for s in selected) > min(s["hi"] for s in selected),
                compound=any(p["active"] and p["open"] and p["decoy"] for p in profiles),
                alternatives=_differences(info))


def self_test():
    """42 seeds, finite fixtures, all legal counts and corruption tests; no I/O."""
    structures, shapes, palettes, puzzles = set(), set(), set(), set()
    features = []
    witness_counts = {k: 0 for k in ALTERNATIVES}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprint = tuple(_key(p[k]) for split in ("train", "test")
                            for p in puzzle[split] for k in ("input", "output"))
        puzzles.add(fingerprint)
        again = generate_puzzle(seed)
        _require(fingerprint == tuple(_key(p[k]) for split in ("train", "test")
                                     for p in again[split] for k in ("input", "output")),
                 "seed is not reproducible")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                original = np.asarray(pair["input"]).copy()
                _require(np.array_equal(solve(original), pair["output"]), "input-only solve mismatch")
                _require(np.array_equal(original, pair["input"]), "solve modified its input")
                f = _features(pair)
                features.append(f)
                structures.add(f["structure"])
                shapes.add(tuple(f["shape"]))
                palettes.add(tuple(f["palette"]))
                for name in ALTERNATIVES:
                    witness_counts[name] += f["alternatives"][name] > 0
        _require(all(_features(p)["compound"] for p in puzzle["test"]),
                 "test must recombine the three aperture categories in one object")
        _require(not any(_features(p)["compound"] for p in puzzle["train"]),
                 "composition shift disappeared")
    _require(len(puzzles) == 42 and len(structures) >= 160, "structural diversity collapsed")
    _require(len(shapes) >= 30 and len(palettes) >= 25, "shape or palette diversity collapsed")
    _require({f["slope"] for f in features} == {-1, 1}, "missing diagonal slope")
    _require({f["turn"] for f in features} == set(range(4)), "missing docking direction")
    _require(len({f["strokes"] for f in features}) >= 5, "stroke counts collapsed")
    _require({1, 2, 3} <= {f["objects"] for f in features}, "object counts collapsed")
    _require(any(0 in f["shifts"] for f in features), "already-docked objects absent")
    _require(any(f["staggered"] for f in features), "staggered interval combinations absent")
    _require(all(n >= 20 for n in witness_counts.values()), "an inherited distinction is ineffective")
    calls = 0
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(17+a*7+b, a, b)
            validate_generated_puzzle(puzzle)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b), "explicit counts ignored")
            calls += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Hand-calculated finite-interval fixture. These checks do not create
    # the asserted cells with solve or the label constructor.
    info = _parse(_fallback("open", (0, 3, 8, 5), 0))
    _require(info["gates"] == {2, 3, 4, 5, 6}, "hand-calculated active aperture regression")
    _require({o["row"]+r for o in info["objects"] for r in o["profile"]["open"]} == {8},
             "open C must not become an enclosed aperture")
    recessed = _parse(_fallback("decoy", (0, 3, 8, 5), 0))
    _require(recessed["gates"] == {2, 3, 4, 5, 6} and
             {o["row"]+r for o in recessed["objects"] for r in o["profile"]["decoy"]} == {8},
             "a thin local wall must not substitute for the global leading extreme")
    mask = _fill(info)
    # q=4,t=20: r=12,d=8, between long boundaries after the middle one ended.
    _require(mask[12, 7] and not _fill(info, "consecutive")[12, 7], "inactive interior boundary regression")
    # q=3,t=17: r=10,d=7, between the pixel centers of both outer strokes.
    _require(mask[10, 6] and not _fill(info, "pixel_centers")[10, 6], "continuous interval regression")
    # q=4,t=26 lies beyond both long strokes' last longitudinal coordinate 22.
    _require(not mask[15, 10] and _fill(info, "unbounded")[15, 10], "endpoint stopping regression")
    for purpose in ("open", "decoy", "test"):
        keys = set()
        for index in range(16):
            fixture = _fallback(purpose, (0, 3, 8, 5), index)
            f = _parse(fixture)
            keys.add(_key(fixture))
            _require(np.array_equal(_fill(f), _pairwise_fill(f)), "fallback geometry mismatch")
        _require(len(keys) == 16, "fallback uniqueness orbit collapsed")
    bad = generate_puzzle(0)
    bad["test"][0]["output"] = bad["test"][0]["output"].copy()
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupt label")
    return dict(seeds=42, pairs=len(features), distinct_puzzles=len(puzzles),
                normalized_structures=len(structures), shapes=len(shapes), palettes=len(palettes),
                legal_count_combinations=calls, alternative_witness_pairs=witness_counts,
                object_counts=sorted({f["objects"] for f in features}),
                stroke_counts=sorted({f["strokes"] for f in features}),
                staggered_pairs=sum(f["staggered"] for f in features),
                max_grid_side=max(max(f["shape"]) for f in features),
                semantic_verification="authored checks, not independent approval")


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
