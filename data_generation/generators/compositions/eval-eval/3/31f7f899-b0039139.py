"""Compressed ribbons: sort repeated stencil recipes into fixed colored tabs.

Only the input is needed by solve. All sampling preferences are geometry choices,
never hidden conventions. See task_spec.md for the supported input language.
"""
from itertools import combinations, permutations

import numpy as np


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "color bounds")
    return a


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask, diagonal=False):
    todo = set(map(tuple, np.argwhere(mask)))
    groups = []
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    while todo:
        start = min(todo)
        todo.remove(start)
        group, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    group.append(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _crop(mask):
    pts = np.argwhere(mask)
    _require(len(pts) > 0, "empty object")
    lo, hi = pts.min(axis=0), pts.max(axis=0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1]].copy(), tuple(map(int, lo))


def _asymmetric(mask):
    return all(not np.array_equal(mask, np.rot90(mask[:, ::-1] if k >= 4 else mask, k % 4))
               for k in range(1, 8))


def _decode(input_grid):
    """Read the carrier, tabs, cropped motifs and counters from cells alone."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(all(int(a[r, c]) == bg for r, c in
                 ((0, 0), (0, -1), (-1, 0), (-1, -1))), "background corners")
    rows = np.flatnonzero(np.all(a != bg, axis=1))
    cols = np.flatnonzero(np.all(a != bg, axis=0))
    _require(len(rows) + len(cols) == 1, "unique carrier required")
    vertical = bool(len(cols))
    g = a.T if vertical else a
    r = int(cols[0] if vertical else rows[0])
    _require(0 < r < g.shape[0] - 1, "carrier must have two sides")
    line = g[r]
    counts = np.bincount(line, minlength=10)
    carrier = int(np.argmax(counts))
    _require(np.count_nonzero(counts == counts[carrier]) == 1, "carrier exposure")
    _require(carrier != bg and counts[carrier] >= 3, "carrier role")
    _require(np.count_nonzero(g == carrier) == counts[carrier], "carrier off its line")
    rebuilt = np.full(g.shape, bg, dtype=np.int8)
    rebuilt[r] = line
    slots = []
    x = 0
    while x < len(line):
        color = int(line[x])
        end = x + 1
        while end < len(line) and int(line[end]) == color:
            end += 1
        if color != carrier:
            _require(color != bg, "background tab")
            upper, lower = g[:r, x:end], g[r+1:, x:end]
            _require(set(map(int, np.unique(upper))) <= {bg, color}, "foreign stencil ink")
            _require(set(map(int, np.unique(lower))) <= {bg, color}, "foreign counter ink")
            mask, offset = _crop(upper == color)
            _require(len(_components(mask)) == 1, "stencil must be connected")
            counter = lower == color
            objects = _components(counter)
            n = len(objects)
            _require(n > 0 and n == len(_components(counter, True)), "counter connectivity")
            length = n * (mask.shape[0] + 1) - 1
            _require(length < r, "expanded ribbon needs a background margin")
            slots.append({"start": x, "width": end-x, "color": color,
                          "mask": mask, "count": n, "length": length,
                          "offset": offset, "counter": counter,
                          "areas": tuple(len(o) for o in objects)})
            rebuilt[:r, x:end] = upper
            rebuilt[r+1:, x:end] = lower
        x = end
    _require(2 <= len(slots) <= 5, "tab count")
    _require(len({s["color"] for s in slots}) >= 2, "need distinct tab identities")
    _require(slots[0]["start"] > 0 and slots[-1]["start"] + slots[-1]["width"] < len(line),
             "exposed carrier ends")
    _require(all(b["start"] > s["start"] + s["width"] for s, b in zip(slots, slots[1:])),
             "tabs need carrier gaps")
    _require(np.array_equal(rebuilt, g), "unexplained ink outside tab lanes")
    _require(bool(np.all(g[0] == bg) and np.all(g[-1] == bg)), "outer background margins")
    _require(all(s["mask"].shape[1] <= min(t["width"] for t in slots) for s in slots),
             "a recipe does not fit every destination")
    return {"grid": g, "vertical": vertical, "background": bg, "carrier": carrier,
            "row": r, "slots": slots}


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _decode(input_grid)
    g, r, bg = d["grid"], d["row"], d["background"]
    out = np.full(g.shape, bg, dtype=np.int8)
    out[r] = g[r]
    ordered = sorted(d["slots"], key=lambda s: s["length"])
    for dest, source in zip(d["slots"], ordered):
        mask = source["mask"]
        h, w = mask.shape
        top = r - source["length"]
        # Modular indexing deliberately differs from construction concatenation.
        for y in range(source["length"]):
            local = y % (h + 1)
            if local < h:
                for x in range(w):
                    if mask[local, x]:
                        out[top+y, dest["start"]+x] = dest["color"]
    return out.T.copy() if d["vertical"] else out


def _mask(rng, h, w):
    """Prune a rectangle without losing connectivity or its tight bounding box."""
    mask = np.ones((h, w), dtype=bool)
    if min(h, w) == 1 or rng.random() < .14:
        return mask
    target = int(rng.integers(h+w-1, h*w))
    for k in rng.permutation(h*w):
        if int(mask.sum()) <= target:
            break
        y, x = divmod(int(k), w)
        mask[y, x] = False
        crop, _ = _crop(mask)
        if crop.shape != (h, w) or len(_components(mask)) != 1:
            mask[y, x] = True
    return mask


def _grow_area(rng, area, width):
    h = (area + width - 1) // width
    mask = np.zeros((h, width), dtype=bool)
    frontier = {(int(rng.integers(h)), int(rng.integers(width)))}
    for _ in range(area):
        candidates = sorted(frontier)
        p = candidates[int(rng.integers(len(candidates)))]
        frontier.remove(p)
        mask[p] = True
        y, x = p
        for q in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if 0 <= q[0] < h and 0 <= q[1] < width and not mask[q]:
                frontier.add(q)
    return _crop(mask)[0]


def _recipe(mask, n):
    return {"mask": mask, "count": int(n), "length": int(n*(mask.shape[0]+1)-1)}


def _ranks(recipes, mode="length"):
    if mode == "height":
        values = [s["mask"].shape[0] for s in recipes]
    elif mode == "count":
        values = [s["count"] for s in recipes]
    elif mode == "ink":
        values = [s["count"]*int(s["mask"].sum()) for s in recipes]
    else:
        values = [s["length"] for s in recipes]
    return sorted(range(len(recipes)), key=lambda i: values[i])


def _witness_one(rng):
    params = [(h, n) for h in range(1, 5) for n in range(1, 6) if n*(h+1)-1 <= 12]
    choices = []
    for triple in combinations(params, 3):
        hs, ns = zip(*triple)
        ls = [n*(h+1)-1 for h, n in triple]
        rank = sorted(range(3), key=lambda i: ls[i])
        if (len(set(hs)) == len(set(ns)) == len(set(ls)) == 3
                and rank != sorted(range(3), key=lambda i: hs[i])
                and rank != sorted(range(3), key=lambda i: ns[i])):
            choices.append(triple)
    for _ in range(80):
        triple = choices[int(rng.integers(len(choices)))]
        rs = [_recipe(_mask(rng, h, int(rng.integers(1, 4))), n) for h, n in triple]
        inversion = any(a["length"] < b["length"] and
                        a["count"]*int(a["mask"].sum()) > b["count"]*int(b["mask"].sum())
                        for a in rs for b in rs)
        if inversion and any(_asymmetric(s["mask"]) for s in rs):
            break
    else:
        rs = [_recipe(np.ones((4, 1), bool), 2), _recipe(np.ones((1, 3), bool), 4),
              _recipe(np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], bool), 3)]
    order = _ranks(rs)
    orders = [p for p in permutations(range(3)) if list(p) != order and list(p)[::-1] != order]
    p = orders[int(rng.integers(len(orders)))]
    return [rs[i] for i in p]


def _witness_tie(rng):
    params = [(h, n) for h in range(1, 5) for n in range(1, 6) if n*(h+1)-1 <= 12]
    ties = [(a, b) for a, b in combinations(params, 2) if a[1]*(a[0]+1) == b[1]*(b[0]+1)]
    chosen = [ties[int(i)] for i in rng.choice(len(ties), 2, replace=False)]
    chosen.sort(key=lambda pair: pair[0][1]*(pair[0][0]+1), reverse=True)
    # Two moved groups with opposite count orders establish stability rather than
    # either ascending-count or descending-count tie breaking.
    ascending_first = bool(rng.integers(2))
    groups = [sorted(pair, key=lambda p: p[1], reverse=(i == 0) != ascending_first)
              for i, pair in enumerate(chosen)]
    rs = [_recipe(_mask(rng, h, 3), n) for group in groups for h, n in group]
    if not any(_asymmetric(s["mask"]) for s in rs):
        i = next(i for i, s in enumerate(rs) if s["mask"].shape[0] >= 3)
        h = rs[i]["mask"].shape[0]
        m = np.zeros((h, 3), dtype=bool)
        m[:, 0] = m[0, :] = True
        m[-1, 1] = True
        rs[i] = _recipe(m, rs[i]["count"])
    return rs


def _ordinary_recipes(rng, count, cap, common_width, min_length=0, wide=False):
    params = [(h, n) for h in range(1, 6) for n in range(1, 8)
              if n*(h+1)-1 <= cap and n*(h+1)-1 + 2*n <= 27]
    for _ in range(100):
        selected = [params[int(rng.integers(len(params)))] for _ in range(count)]
        if min_length:
            large = [p for p in params if p[1]*(p[0]+1)-1 >= min_length]
            selected[int(rng.integers(count))] = large[int(rng.integers(len(large)))]
        lengths = [n*(h+1)-1 for h, n in selected]
        if max(lengths) + 2*max(n for h, n in selected) <= 27:
            break
    else:
        # Any maximal-length recipe can coexist with low-count recipes.
        p = min([p for p in params if p[1]*(p[0]+1)-1 >= min_length], key=lambda p: p[1])
        selected = [p] + [(1, 1)]*(count-1)
        rng.shuffle(selected)
    rs = [_recipe(_mask(rng, h, int(rng.integers(1, common_width+1))), n) for h, n in selected]
    if wide:
        i = int(rng.integers(count))
        h, n = selected[i]
        rs[i] = _recipe(_mask(rng, h, common_width), n)
    return rs


def _make_pair(rng, recipes, vertical, palette, witness=False, minimum_width=1):
    bg, carrier, *accents = palette
    nslots = len(recipes)
    minimum_width = max(minimum_width, max(s["mask"].shape[1] for s in recipes))
    colors = list(map(int, rng.choice(accents, nslots, replace=False)))
    if rng.random() < .35 and nslots >= 3:
        colors[-1] = colors[0]
    widths = list(map(int, rng.integers(minimum_width, min(6, minimum_width+3)+1, nslots)))

    def exposure():
        return max(nslots+1, 3, max(sum(w for w, c in zip(widths, colors) if c == k)
                                         for k in set(colors))+1)

    while sum(widths) + exposure() > 30:
        eligible = [i for i, w in enumerate(widths) if w > minimum_width]
        _require(bool(eligible), "tab widths cannot fit")
        widths[max(eligible, key=lambda i: widths[i])] -= 1
    width = int(rng.integers(sum(widths)+exposure(), 31))
    gaps = np.ones(nslots+1, dtype=int)
    gaps += rng.multinomial(width-sum(widths)-nslots-1, [1/(nslots+1)]*(nslots+1))
    starts, pos = [], int(gaps[0])
    for i, w in enumerate(widths):
        starts.append(pos)
        pos += w + int(gaps[i+1])
    longest = max(s["length"] for s in recipes)
    lower_budget = 26-longest  # two lower margins, carrier, one upper margin
    same_area = 2*max(s["count"] for s in recipes) if witness else None
    collections = []
    for s, w in zip(recipes, widths):
        n = s["count"]
        if witness:
            areas = [1]*n
            for _ in range(same_area-n):
                areas[int(rng.integers(n))] += 1
            objects = [_grow_area(rng, area, w) for area in areas]
        else:
            heights = [1]*n
            extra = lower_budget-(2*n-1)
            for _ in range(int(rng.integers(extra+1))):
                eligible = [i for i, h in enumerate(heights) if h < 4]
                if not eligible:
                    break
                heights[int(rng.choice(eligible))] += 1
            objects = [_mask(rng, h, int(rng.integers(1, w+1))) for h in heights]
        rng.shuffle(objects)
        _require(sum(o.shape[0] for o in objects)+n-1 <= lower_budget, "counter budget")
        collections.append(objects)
    lower_min = max(sum(o.shape[0] for o in objects)+len(objects)-1 for objects in collections)
    height_min = longest + lower_min + 4
    height = int(rng.integers(height_min, 31))
    row = int(rng.integers(longest+1, height-lower_min-2))
    inp = np.full((height, width), bg, dtype=np.int8)
    inp[row] = carrier
    for s, x, w, color, objects in zip(recipes, starts, widths, colors, collections):
        inp[row, x:x+w] = color
        mask = s["mask"]
        h, mw = mask.shape
        y = int(rng.integers(1, row-h+1))
        left = x+int(rng.integers(w-mw+1))
        inp[y:y+h, left:left+mw][mask] = color
        # Empty rows guarantee equal four/eight-neighbor counts, regardless of shape.
        total = sum(o.shape[0] for o in objects)+len(objects)-1
        slack = height-row-1-total-2
        extras = rng.multinomial(slack, [1/(len(objects)+1)]*(len(objects)+1))
        y = row+2+int(extras[0])
        for i, obj in enumerate(objects):
            oh, ow = obj.shape
            left = x+int(rng.integers(w-ow+1))
            inp[y:y+oh, left:left+ow][obj] = color
            y += oh+1+int(extras[i+1])
    out = np.full_like(inp, bg)
    out[row] = inp[row]
    for dest, source in zip(range(nslots), sorted(recipes, key=lambda s: s["length"])):
        tile = np.where(source["mask"], colors[dest], bg).astype(np.int8)
        pieces = []
        for k in range(source["count"]):
            if k:
                pieces.append(np.full((1, tile.shape[1]), bg, dtype=np.int8))
            pieces.append(tile)
        ribbon = np.concatenate(pieces, axis=0)
        out[row-ribbon.shape[0]:row, starts[dest]:starts[dest]+tile.shape[1]] = ribbon
    if vertical:
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}


def _counts(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(name + " outside its legal range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, "num_train", 2, 10), _counts(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palettes = [list(map(int, rng.permutation(10))) for _ in range(3)]
    axes = int(rng.integers(3))  # horizontal, vertical, or demonstrated mixture
    train_axes = ([False, True] + [bool(rng.integers(2)) for _ in range(nt-2)]
                  if axes == 2 else [bool(axes)]*nt)
    shift = str(rng.choice(["stations", "length", "width"]))
    puzzle, seen = {"train": [], "test": []}, set()
    train_recipes = []
    for i in range(nt):
        if i == 0:
            rs = _witness_one(rng)
        elif i == 1:
            rs = _witness_tie(rng)
        else:
            rs = _ordinary_recipes(rng, int(rng.integers(2, 5)), 12, int(rng.integers(1, 4)))
        pair = _make_pair(rng, rs, train_axes[i], palettes[i % 3], i < 2,
                          minimum_width=3 if i < 2 else 1)
        puzzle["train"].append(pair)
        train_recipes.extend(rs)
        seen.add(_key(pair["input"]))
    for _ in range(nq):
        vertical = bool(rng.choice(train_axes))
        count = 5 if shift == "stations" else int(rng.integers(2, 6))
        mw = 4 if shift == "width" else int(rng.integers(1, 5))
        rs = _ordinary_recipes(rng, count, 18 if shift == "length" else 12, mw,
                               min_length=max(s["length"] for s in train_recipes)+1
                               if shift == "length" else 0, wide=shift == "width")
        pair = _make_pair(rng, rs, vertical, palettes[int(rng.integers(3))], minimum_width=mw)
        puzzle["test"].append(pair)
    # Finite color permutations resolve any duplicate without changing geometry or roles.
    seen = set()
    for split in ("train", "test"):
        for i, pair in enumerate(puzzle[split]):
            if _key(pair["input"]) in seen:
                original = pair
                found = False
                for bg, carrier in permutations(range(10), 2):
                    old = _decode(original["input"])
                    old_colors = [old["background"], old["carrier"]] + sorted(
                        {s["color"] for s in old["slots"]})
                    new_colors = [bg, carrier] + [c for c in range(10) if c not in (bg, carrier)]
                    mapping = np.arange(10, dtype=np.int8)
                    for a, b in zip(old_colors, new_colors):
                        mapping[a] = b
                    candidate = {k: mapping[v] for k, v in original.items()}
                    if _key(candidate["input"]) not in seen:
                        pair, found = candidate, True
                        puzzle[split][i] = pair
                        break
                _require(found, "duplicate fallback exhausted")
            seen.add(_key(pair["input"]))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    d = _decode(pair["input"])
    out = _grid(pair["output"])
    _require(out.shape == np.asarray(pair["input"]).shape, "canvas changed")
    canonical = out.T if d["vertical"] else out
    expected = np.full(canonical.shape, d["background"], dtype=np.int8)
    r = d["row"]
    expected[r] = d["grid"][r]
    sources = _ranks(d["slots"])
    # Check each donor copy explicitly, without solve or construction concatenation.
    for dest, j in zip(d["slots"], sources):
        s = d["slots"][j]
        h, w = s["mask"].shape
        for k in range(s["count"]):
            top = r-s["length"]+k*(h+1)
            for y, x in np.argwhere(s["mask"]):
                expected[top+int(y), dest["start"]+int(x)] = dest["color"]
    _require(np.array_equal(canonical, expected), "wrong recipe assignment, stamp or clearing")
    return d


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, data = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            d = _check_pair(pair)
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            data[split].append(d)
    train = data["train"]
    _require(any(_ranks(d["slots"]) != list(range(len(d["slots"]))) for d in train),
             "training must require reassignment (A ablation)")
    for alternative in ("height", "count", "ink"):
        _require(any(_ranks(d["slots"]) != _ranks(d["slots"], alternative) for d in train),
                 "shortcut not distinguished: " + alternative)
    _require(any(any(a["length"] < b["length"] and
                     a["count"]*int(a["mask"].sum()) > b["count"]*int(b["mask"].sum())
                     for a in d["slots"] for b in d["slots"]) for d in train),
             "need strict ink-area inversion, not accidental tie order")
    _require(any(_ranks(d["slots"]) != list(reversed(range(len(d["slots"])))) for d in train),
             "reversal shortcut")
    _require(any(any(s["count"] > 1 and sum(s["areas"]) > s["count"] for s in d["slots"])
                 for d in train), "component and repetition evidence (B ablation)")
    _require(any(any(sum(a["areas"]) == sum(b["areas"]) and a["count"] != b["count"]
                     for a, b in combinations(d["slots"], 2)) for d in train), "area versus count")
    _require(any(any(not s["mask"].all() and _asymmetric(s["mask"]) for s in d["slots"])
                 for d in train), "binary stencil and orientation evidence")
    _require(any(any(s["offset"] != (0, 0) or s["mask"].shape[1] != s["width"]
                     for s in d["slots"]) for d in train), "crop evidence")
    tie_directions = set()
    for d in train:
        rs, order = d["slots"], _ranks(d["slots"])
        for a, b in combinations(range(len(rs)), 2):
            if (rs[a]["length"] == rs[b]["length"] and
                    (not np.array_equal(rs[a]["mask"], rs[b]["mask"]) or rs[a]["count"] != rs[b]["count"]) and
                    order.index(a) != a and order.index(b) != b):
                tie_directions.add(1 if rs[a]["count"] < rs[b]["count"] else -1)
    _require(tie_directions == {-1, 1}, "moved ties must teach both count orders")
    axes = {d["vertical"] for d in train}
    train_n = max(len(d["slots"]) for d in train)
    train_l = max(s["length"] for d in train for s in d["slots"])
    train_w = max(s["mask"].shape[1] for d in train for s in d["slots"])
    for d in data["test"]:
        _require(d["vertical"] in axes, "undemonstrated test axis")
        _require(len(d["slots"]) > train_n or max(s["length"] for s in d["slots"]) > train_l
                 or max(s["mask"].shape[1] for s in d["slots"]) > train_w, "missing structural test shift")


def _puzzle_key(p):
    return tuple(tuple((_key(x["input"]), _key(x["output"])) for x in p[s]) for s in ("train", "test"))


def _ablation(d, omit_a=False, omit_b=False):
    """Diagnostic counterfactuals only; never used to construct correct labels."""
    rs = d["slots"]
    order = list(range(len(rs))) if omit_a else _ranks(rs, "height" if omit_b else "length")
    out = np.full(d["grid"].shape, d["background"], dtype=np.int8)
    out[d["row"]] = d["grid"][d["row"]]
    for dest, j in zip(rs, order):
        mask = rs[j]["mask"]
        n = 1 if omit_b else rs[j]["count"]
        h, w = mask.shape
        top = d["row"] - (n*(h+1)-1)
        for k in range(n):
            for y, x in np.argwhere(mask):
                out[top+k*(h+1)+int(y), dest["start"]+int(x)] = dest["color"]
    return out.T.copy() if d["vertical"] else out


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "puzzle diversity")
    shapes, structures, motifs, widths, counts, heights, axes, shifts = set(), set(), set(), set(), set(), set(), set(), set()
    repeated = mixed = 0
    for p in puzzles:
        ds = [_decode(x["input"]) for x in p["train"]]
        for option in ("omit_a", "omit_b"):
            _require(any(not np.array_equal(_ablation(d, **{option: True}), pair["output"])
                         for d, pair in zip(ds, p["train"])), "parent ablation had no effect")
        tn = max(len(d["slots"]) for d in ds)
        tl = max(s["length"] for d in ds for s in d["slots"])
        tw = max(s["mask"].shape[1] for d in ds for s in d["slots"])
        for split in ("train", "test"):
            for pair in p[split]:
                d = _decode(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve disagreement")
                shapes.add(pair["input"].shape)
                slots = d["slots"]
                widths.update(s["width"] for s in slots)
                counts.update(s["count"] for s in slots)
                heights.update(s["mask"].shape[0] for s in slots)
                axes.add(d["vertical"])
                motifs.update(_key(s["mask"]) for s in slots)
                structures.add((d["grid"].shape, d["row"], tuple(
                    (s["start"], s["width"], _key(s["mask"]), s["count"], s["offset"], _key(s["counter"])) for s in slots)))
                repeated += len({s["color"] for s in slots}) < len(slots)
                mixed += len({s["width"] for s in slots}) > 1
                if split == "test":
                    if len(slots) > tn:
                        shifts.add("stations")
                    if max(s["length"] for s in slots) > tl:
                        shifts.add("length")
                    if max(s["mask"].shape[1] for s in slots) > tw:
                        shifts.add("width")
    _require(len(shapes) >= 35 and len(structures) >= 180 and len(motifs) >= 55, "structural diversity")
    _require(axes == {False, True} and shifts == {"stations", "length", "width"}, "regime diversity")
    _require(counts >= {1, 2, 3, 4, 5, 6} and heights == {1, 2, 3, 4, 5}, "recipe diversity")
    _require(widths >= {1, 2, 3, 4, 5, 6} and repeated >= 15 and mixed >= 40, "tab diversity")
    # The seven-component limit uses the entire height budget. Exercise it
    # explicitly instead of requiring this rare event in a finite random sweep.
    extreme = [_recipe(np.ones((1, 1), bool), 7), _recipe(np.ones((5, 4), bool), 1)]
    for vertical in (False, True):
        pair = _make_pair(np.random.default_rng(8901), extreme, vertical,
                          list(range(10)), minimum_width=4)
        _check_pair(pair)
        _require(np.array_equal(solve(pair["input"]), pair["output"]), "boundary recipe")
    for seed in (0, 7, 41):
        _require(_puzzle_key(puzzles[seed]) == _puzzle_key(generate_puzzle(seed)), "nondeterminism")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(9000+a*10+b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "ignored counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    corrupt = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in ("train", "test")}
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "input_shapes": len(shapes), "structures": len(structures),
            "binary_motifs": len(motifs), "counter_counts": sorted(counts),
            "motif_heights": sorted(heights), "tab_widths": sorted(widths),
            "repeated_color_pairs": repeated, "mixed_width_pairs": mixed,
            "test_shifts": sorted(shifts)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
