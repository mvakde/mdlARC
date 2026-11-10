"""Ranked stencils: infer a docking/painting recipe, then match two queues.

Only NumPy and the standard library are used. No files are read or written.
The pixel parser is also used by the validator; it is not an independent oracle.
"""
from __future__ import annotations

import hashlib
from itertools import product

import numpy as np


MODES = ("interior", "rim", "square")
ANCHORS = tuple(product(range(3), repeat=2))  # top/center/bottom x left/center/right


def _integer(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must lie in {low}..{high}")
    return value


def _counts(a, b):
    a = _integer(a, 2, 10, "num_train")
    b = _integer(b, 1, 4, "num_test")
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _array(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "integer colors required"
    assert np.all((0 <= a) & (a <= 9)), "color range"
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
        rr, cc = zip(*cells)
        components.append((int(min(rr)), int(min(cc)),
                           int(max(rr)-min(rr)+1), int(max(cc)-min(cc)+1)))
    return components


def _tile(size, color, bg, kind):
    a = np.full((size, size), color, dtype=np.int8)
    if kind == "H":
        a[1:-1, 1:-1] = bg
    return a


def _tiles(a, bg):
    objects = []
    for r, c, h, w in _components(a != bg):
        assert h == w and h >= 3, "square tile required"
        t = a[r:r+h, c:c+w]
        color = int(t[0, 0])
        assert color != bg
        kind = "S" if np.all(t == color) else "H"
        assert np.array_equal(t, _tile(h, color, bg, kind)), "malformed tile"
        # Distinct footprints need a background gap, even diagonally.
        r0, c0 = max(0, r-1), max(0, c-1)
        halo = a[r0:min(a.shape[0], r+h+1), c0:min(a.shape[1], c+w+1)].copy()
        halo[r-r0:r-r0+h, c-c0:c-c0+w] = bg
        assert np.all(halo == bg), "touching tiles"
        objects.append(dict(r=r, c=c, size=h, color=color, kind=kind))
    assert objects
    for kind in ("H", "S"):
        rows = [o["r"] for o in objects if o["kind"] == kind]
        assert len(rows) == len(set(rows)), "within-class top-row tie"
    return sorted(objects, key=lambda o: (o["r"], o["c"]))


def _offset(solid_size, hollow_size, anchor):
    d = solid_size - hollow_size
    return (0, d // 2, d)[anchor]


def _stencil(s, h, mode, anchor):
    """Rasterize the translated stencil in carrier coordinates; clip naturally."""
    rr, cc = np.indices((s, s))
    r = rr - _offset(s, h, anchor[0])
    c = cc - _offset(s, h, anchor[1])
    full = (r >= 0) & (r < h) & (c >= 0) & (c < h)
    inner = (r > 0) & (r < h-1) & (c > 0) & (c < h-1)
    return inner if mode == "interior" else full & ~inner if mode == "rim" else full


def _paint(s, h, solid_color, hollow_color, mode, anchor):
    out = np.full((s, s), solid_color, dtype=np.int8)
    out[_stencil(s, h, mode, anchor)] = hollow_color
    return out


def _target_output(objects, bg, mode, anchor, *, order="vertical"):
    queues = [[o for o in objects if o["kind"] == k] for k in ("H", "S")]
    for q in queues:
        q.sort(key=lambda o: (o["r"], o["c"]))
    if order == "horizontal":
        queues[0].sort(key=lambda o: (o["c"], o["r"]))
        queues[1].sort(key=lambda o: (o["c"], o["r"]))
    elif order == "reverse":
        queues[0].reverse()
    elif order == "nearest":
        # A common incorrect shortcut: reuse the nearest hollow for each solid.
        queues[0] = [min(queues[0], key=lambda h: (abs(h["r"]-s["r"])+abs(h["c"]-s["c"]),
                                                h["r"], h["c"])) for s in queues[1]]
    elif order == "greedy":
        rows = [[None, None]]
        for o in objects:
            col = int(o["kind"] == "S")
            if rows[-1][col] is not None:
                rows.append([None, None])
            rows[-1][col] = o
        matches = {id(s): h for h, s in rows if s is not None}
        queues[0] = [matches.get(id(s)) for s in queues[1]]
    hollows, solids = queues
    assert hollows and solids
    r0, c0 = min(s["r"] for s in solids), min(s["c"] for s in solids)
    r1 = max(s["r"]+s["size"] for s in solids)
    c1 = max(s["c"]+s["size"] for s in solids)
    out = np.full((r1-r0, c1-c0), bg, dtype=np.int8)
    for i, s in enumerate(solids):
        tile = np.full((s["size"], s["size"]), s["color"], dtype=np.int8)
        if i < len(hollows) and hollows[i] is not None:
            h = hollows[i]
            tile = _paint(s["size"], h["size"], s["color"], h["color"], mode, anchor)
        r, c = s["r"]-r0, s["c"]-c0
        out[r:r+s["size"], c:c+s["size"]] = tile
    return out


def _analyze(input_grid):
    a = _array(input_grid)
    bg = int(a[0, 0])
    dividers = [r for r in range(a.shape[0]) if a[r, 0] != bg and np.all(a[r] == a[r, 0])]
    assert len(dividers) == 1, "one full-width reference divider required"
    divider = dividers[0]
    frame = int(a[divider, 0])
    boxes = _components(a[:divider] == frame)
    assert len(boxes) == 2, "two reference panels required"
    sources, results = [], []
    covered = np.zeros(a[:divider].shape, dtype=bool)
    for r, c, h, w in boxes:
        t = a[r:r+h, c:c+w]
        assert h >= 9 and w >= 9
        rim = np.zeros(t.shape, dtype=bool)
        rim[[0, -1], :] = True
        rim[:, [0, -1]] = True
        assert np.all(t[rim] == frame) and not np.any(t[~rim] == frame)
        covered[r:r+h, c:c+w] = True
        inside = t[1:-1, 1:-1]
        comps = _components(inside != bg)
        if len(comps) == 2:
            objects = _tiles(inside, bg)
            assert sorted(o["kind"] for o in objects) == ["H", "S"]
            sources.append(objects)
        elif len(comps) == 1:
            y, x, hh, ww = comps[0]
            assert hh == ww
            result = inside[y:y+hh, x:x+ww]
            assert np.all(result != bg) and len(np.unique(result)) == 2
            results.append((result.copy(), (r+1+y, c+1+x, hh)))
        else:
            raise AssertionError("reference panel contents")
    assert np.all(a[:divider][~covered] == bg), "extra reference-area content"
    assert len(sources) == len(results) == 1
    h = next(o for o in sources[0] if o["kind"] == "H")
    s = next(o for o in sources[0] if o["kind"] == "S")
    assert h["color"] != s["color"] and h["size"] <= s["size"]-2
    result, result_box = results[0]
    assert result.shape == (s["size"], s["size"])
    recipes = [(m, anchor) for m in MODES for anchor in ANCHORS
               if np.array_equal(result, _paint(s["size"], h["size"], s["color"], h["color"], m, anchor))]
    assert len(recipes) == 1, "reference must determine one recipe"
    objects = _tiles(a[divider+1:], bg)
    assert {o["kind"] for o in objects} == {"H", "S"}
    assert not np.any(a[divider+1:] == frame), "reserved frame color in target"
    mode, anchor = recipes[0]
    return dict(bg=bg, frame=frame, divider=divider, objects=objects,
                mode=mode, anchor=anchor, ref_h=h, ref_s=s, result_box=result_box)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the entire rule, palette, dimensions and rank correspondences locally."""
    a = _analyze(input_grid)
    return _target_output(a["objects"], a["bg"], a["mode"], a["anchor"])


def _reference_twin(grid, mode, anchor):
    a = _analyze(grid)
    s, h = a["ref_s"], a["ref_h"]
    out = grid.copy()
    r, c, n = a["result_box"]
    out[r:r+n, c:c+n] = _paint(n, h["size"], s["color"], h["color"], mode, anchor)
    return out


def _place(rng, sizes, kinds, height, width):
    """Feasible column initialization followed by unrestricted legal relocation."""
    columns = []
    for kind in ("H", "S"):
        ids = [i for i, k in enumerate(kinds) if k == kind]
        rng.shuffle(ids)
        columns.extend(ids[i:i+2] for i in range(0, len(ids), 2))
    rng.shuffle(columns)
    widths = [max(sizes[i] for i in col) for col in columns]
    required = sum(widths)+len(columns)-1
    assert required <= width
    gaps = rng.multinomial(width-required, np.full(len(columns)+1, 1/(len(columns)+1)))
    positions = [None]*len(sizes)
    used_rows = {"H": set(), "S": set()}
    left = int(gaps[0])
    for col, cw, gap in zip(columns, widths, gaps[1:]):
        span = sum(sizes[i] for i in col)+len(col)-1
        choices = []
        for start in range(height-span+1):
            rows, top = [], start
            for i in col:
                rows.append(top)
                top += sizes[i]+1
            if not used_rows[kinds[col[0]]].intersection(rows):
                choices.append(rows)
        assert choices, "column initialization has no distinct top rows"
        rows = choices[int(rng.integers(len(choices)))]
        for i, top in zip(col, rows):
            positions[i] = (top, left+int(rng.integers(cw-sizes[i]+1)))
            used_rows[kinds[i]].add(top)
        left += cw+1+int(gap)
    for _ in range(2):
        for idx in rng.permutation(len(sizes)):
            s = sizes[idx]
            rr, cc = np.indices((height-s+1, width-s+1))
            legal = np.ones(rr.shape, bool)
            for j, (r, c) in enumerate(positions):
                if j == idx:
                    continue
                t = sizes[j]
                legal &= (rr+s < r) | (r+t < rr) | (cc+s < c) | (c+t < cc)
                if kinds[j] == kinds[idx]:
                    legal &= rr != r
            sites = np.argwhere(legal)
            assert len(sites)
            positions[idx] = tuple(map(int, sites[int(rng.integers(len(sites)))]))
    return positions


def _make_target(rng, height, width, bg, frame, nh, ns, test):
    kinds = ["H"]*nh+["S"]*ns
    sizes = [int(rng.integers(5, 7) if test else rng.integers(3, 5)) for _ in range(nh)]
    sizes += [int(rng.integers(4, 8) if test else rng.integers(5, 7)) for _ in range(ns)]
    positions = _place(rng, sizes, kinds, height, width)
    palette = list(map(int, rng.permutation([c for c in range(10) if c not in (bg, frame)])))
    objects = [dict(r=r, c=c, size=s, kind=k, color=-1)
               for (r, c), s, k in zip(positions, sizes, kinds)]
    queues = [sorted([o for o in objects if o["kind"] == k], key=lambda o:o["r"]) for k in ("H", "S")]
    repeat = not test and rng.random() < .25
    for i, h in enumerate(queues[0]):
        h["color"] = palette[i % (2 if repeat else len(palette))]
    for i, s in enumerate(queues[1]):
        choices = [c for c in palette if i >= nh or c != queues[0][i]["color"]]
        s["color"] = int(rng.choice(choices))
    grid = np.full((height, width), bg, dtype=np.int8)
    for o in objects:
        r, c, s = o["r"], o["c"], o["size"]
        grid[r:r+s, c:c+s] = _tile(s, o["color"], bg, o["kind"])
    return grid


def _outline(a, r, c, h, w, color):
    a[r, c:c+w] = a[r+h-1, c:c+w] = color
    a[r:r+h, c] = a[r:r+h, c+w-1] = color


def _scene(rng, nh, ns, test, mode, anchor):
    bg, frame, hc, sc = map(int, rng.choice(10, 4, replace=False))
    rh = int(rng.integers(3, 5))
    rs = rh+3  # Odd slack visibly demonstrates the top/left rounding convention.
    ph, sw, ow = rs+4, rh+rs+5, rs+4
    width = int(rng.integers(max(29, sw+ow+3), 31))
    divider = ph+2
    height = 30-divider-1
    target = _make_target(rng, height, width, bg, frame, nh, ns, test)
    grid = np.full((30, width), bg, dtype=np.int8)
    grid[divider] = frame
    grid[divider+1:] = target
    source_left = bool(rng.integers(2))
    total = sw+ow+1
    start = int(rng.integers(width-total+1))
    source_c = start if source_left else start+ow+1
    result_c = start+sw+1 if source_left else start
    _outline(grid, 1, source_c, ph, sw, frame)
    _outline(grid, 1, result_c, ph, ow, frame)
    # Object order within the source panel carries no semantic direction.
    hollow_left = bool(rng.integers(2))
    hcol = source_c+2 if hollow_left else source_c+rs+3
    scol = source_c+rh+3 if hollow_left else source_c+2
    hrow = 3+int(rng.integers(rs-rh+1))
    grid[hrow:hrow+rh, hcol:hcol+rh] = _tile(rh, hc, bg, "H")
    grid[3:3+rs, scol:scol+rs] = sc
    grid[3:3+rs, result_c+2:result_c+2+rs] = _paint(rs, rh, sc, hc, mode, anchor)
    return grid


def _witnesses(a):
    correct = _target_output(a["objects"], a["bg"], a["mode"], a["anchor"])
    return {order: not np.array_equal(correct, _target_output(a["objects"], a["bg"], a["mode"], a["anchor"], order=order))
            for order in ("horizontal", "reverse", "nearest", "greedy")}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    modes = [MODES[int(i)] for i in rng.permutation(3)]
    anchors = [ANCHORS[int(rng.integers(9))] for _ in range(2)]
    first_counts = (2, 3) if rng.integers(2) else (3, 2)
    puzzle = {"train": [], "test": []}
    # Every puzzle, even with two demonstrations, has exact target-only twins
    # and concrete witnesses against geometry-only/local-greedy pairing.
    for _ in range(96):
        first = _scene(rng, *first_counts, False, modes[0], anchors[0])
        a = _analyze(first)
        if not all(_witnesses(a).values()):
            continue
        second = _reference_twin(first, modes[1], anchors[1])
        if np.array_equal(solve(first), solve(second)):
            continue
        break
    else:
        raise RuntimeError("bounded rank-witness sampling exhausted")
    for grid in (first, second):
        puzzle["train"].append({"input": grid, "output": solve(grid)})
    seen = {_key(first), _key(second)}
    for split, count in (("train", num_train-2), ("test", num_test)):
        for index in range(count):
            for _ in range(64):
                if split == "train" and index == 0:
                    nh, ns = first_counts[::-1]
                elif split == "train" and index == 1:
                    nh = ns = int(rng.integers(1, 5))
                else:
                    nh, ns = map(int, rng.integers(1 if split == "train" else 2, 5, size=2))
                    if split == "test" and num_train == 2:
                        nh, ns = sorted((nh, ns), reverse=first_counts[0] > first_counts[1])
                mode = str(rng.choice(MODES))
                anchor = ANCHORS[int(rng.integers(9))]
                grid = _scene(rng, nh, ns, split == "test", mode, anchor)
                a = _analyze(grid)
                hs = [o for o in a["objects"] if o["kind"] == "H"]
                ss = [o for o in a["objects"] if o["kind"] == "S"]
                # Avoid invisible matched stencils in sampled evidence. The
                # solver still supports zero-intersection clipped stencils.
                if any(not np.any(_stencil(s["size"], h["size"], mode, anchor)) for h, s in zip(hs, ss)):
                    continue
                if _key(grid) in seen:
                    continue
                seen.add(_key(grid))
                puzzle[split].append({"input": grid, "output": solve(grid)})
                break
            else:
                raise RuntimeError("bounded unique-pair sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            a = _analyze(inp)
            expected = _target_output(a["objects"], a["bg"], a["mode"], a["anchor"])
            assert np.array_equal(expected, out), "incorrect stencil transfer"
            solids = [o for o in a["objects"] if o["kind"] == "S"]
            assert np.count_nonzero(out != a["bg"]) == sum(o["size"]**2 for o in solids), "carrier area"
            assert np.all(np.any(out != a["bg"], axis=1)[[0, -1]])
            assert np.all(np.any(out != a["bg"], axis=0)[[0, -1]])
            analyses[split].append(a)
    for order in ("horizontal", "reverse", "nearest", "greedy"):
        assert any(_witnesses(a)[order] for a in analyses["train"]), f"missing {order} counterexample"
    # Read twin evidence from pixels, without generation metadata or labels.
    twins = False
    for i, a in enumerate(analyses["train"]):
        for j, b in enumerate(analyses["train"][:i]):
            ta = np.asarray(puzzle["train"][i]["input"])[a["divider"]+1:]
            tb = np.asarray(puzzle["train"][j]["input"])[b["divider"]+1:]
            if a["bg"] == b["bg"] and np.array_equal(ta, tb):
                twins |= not np.array_equal(puzzle["train"][i]["output"], puzzle["train"][j]["output"])
    assert twins, "missing reference-dependent target twins"
    train_sizes = {o["size"] for a in analyses["train"] for o in a["objects"] if o["kind"] == "H"}
    assert train_sizes <= {3, 4}
    assert all(any(o["kind"] == "H" and o["size"] > max(train_sizes) for o in a["objects"])
               for a in analyses["test"]), "missing stencil-scale shift"


def self_test():
    """42 streamed puzzles, no persistent samples, bounded finite diagnostics."""
    recipes, layouts, shapes, counts, palettes, normalized, signatures = (set() for _ in range(7))
    clipped = repeated = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        signature = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                a = _analyze(pair["input"])
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(solve(inp.copy()), out)
                recipes.add((a["mode"], a["anchor"]))
                shapes.add(out.shape)
                counts.add(tuple(sum(o["kind"] == k for o in a["objects"]) for k in ("H", "S")))
                palettes.add((a["bg"], a["frame"]))
                layouts.add(tuple((o["r"], o["c"], o["size"], o["kind"]) for o in a["objects"]))
                roles = {a["bg"]: 0}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in out.flat)
                normalized.add((out.shape, norm))
                hs = [o for o in a["objects"] if o["kind"] == "H"]
                ss = [o for o in a["objects"] if o["kind"] == "S"]
                clipped += any(h["size"] > s["size"] for h, s in zip(hs, ss))
                repeated += len({o["color"] for o in a["objects"]}) < len(a["objects"])
                signature.update(inp.tobytes())
                signature.update(out.tobytes())
        signatures.add(signature.digest())
    assert len(signatures) == 42
    assert recipes == set(product(MODES, ANCHORS)), "missing recipe diversity"
    assert len(layouts) >= 150 and len(shapes) >= 50 and len(counts) >= 10
    assert len(palettes) >= 40 and len(normalized) >= 150
    assert clipped >= 10 and repeated >= 20
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(9341, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    p = generate_puzzle(123)
    q = generate_puzzle(123)
    assert all(_key(x[k]) == _key(y[k]) for s in p for x, y in zip(p[s], q[s]) for k in ("input", "output"))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Negative label control.
    p["test"][0]["output"] = p["test"][0]["output"].copy()
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged output accepted")
    # Exhaustive small clipping/centering check against literal stencil points.
    for s, h, mode, anchor in product(range(3, 8), range(3, 9), MODES, ANCHORS):
        literal = np.zeros((s, s), bool)
        dr, dc = (_offset(s, h, x) for x in anchor)
        for r, c in product(range(h), repeat=2):
            interior = 0 < r < h-1 and 0 < c < h-1
            selected = interior if mode == "interior" else not interior if mode == "rim" else True
            if selected and 0 <= r+dr < s and 0 <= c+dc < s:
                literal[r+dr, c+dc] = True
        assert np.array_equal(literal, _stencil(s, h, mode, anchor))
    return {"seeds": 42, "pairs": 210, "recipes": len(recipes), "layouts": len(layouts),
            "output_shapes": len(shapes), "count_patterns": len(counts),
            "normalized_outputs": len(normalized), "clipped_pairs": clipped,
            "repeated_color_pairs": repeated}
