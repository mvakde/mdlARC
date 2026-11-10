import hashlib

import numpy as np


_SOURCE_INPUT_HASHES = {
    "60406099c821d3aec360c47bd31a49ceeb409549608d8aaeb1f4633e845b5cb4",
    "1d5007d6f711806a6da66939464f0318083dc65dba62dab270b0ef0c1067de08",
    "83feed0a15755c85c1a8403349f905d3d85655d6ddb40620ddd84541dde23c48",
    "297795e4c7937776b197ca9b034574880a666a7ac9c3051594dbc8d7c837dd60",
}


def _validate_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return int(seed)


def _count_value(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    num_train = _count_value(num_train, "num_train", 2, 10)
    num_test = _count_value(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        lo = max(2, num_test + 1)
        num_train = lo + seed % (11 - lo)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _contains(outer, inner):
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _separate(a, b):
    return a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1]


def _border(rect):
    r0, c0, r1, c1 = rect
    return {(r0, c) for c in range(c0, c1 + 1)} | {(r1, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0 + 1, r1)} | {(r, c1) for r in range(r0 + 1, r1)}


def _components(grid, background):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == background:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = set()
            while stack:
                y, x = stack.pop()
                cells.add((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and int(grid[yy, xx]) == color:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            result.append((color, cells, (min(rows), min(cols), max(rows), max(cols))))
    return result


def _analyze_input(grid):
    assert grid.ndim == 2
    h, w = grid.shape
    bg = int(grid[0, 0])
    assert int(grid[0, -1]) == int(grid[-1, 0]) == int(grid[-1, -1]) == bg
    components = _components(grid, bg)
    frames = []
    motifs = []
    for color, cells, rect in components:
        r0, c0, r1, c1 = rect
        rh, rw = r1 - r0 + 1, c1 - c0 + 1
        if rh >= 5 and rw >= 5 and cells == _border(rect):
            frames.append({"color": color, "rect": rect})
        else:
            assert rh <= 3 and rw <= 3
            assert cells == {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
            motifs.append({"color": color, "rect": rect, "cells": cells})
    assert frames
    for i, frame in enumerate(frames):
        rect = frame["rect"]
        containers = [j for j, other in enumerate(frames) if i != j and _contains(other["rect"], rect)]
        if containers:
            parent = min(containers, key=lambda j: (frames[j]["rect"][2] - frames[j]["rect"][0]) * (frames[j]["rect"][3] - frames[j]["rect"][1]))
            a = frames[parent]["rect"]
            assert rect[0] - a[0] >= 2 and rect[1] - a[1] >= 2 and a[2] - rect[2] >= 2 and a[3] - rect[3] >= 2
            frame["parent"] = parent
        else:
            frame["parent"] = None
        for j in range(i):
            other = frames[j]["rect"]
            assert _contains(rect, other) or _contains(other, rect) or _separate(rect, other)
    roots = [i for i, frame in enumerate(frames) if frame["parent"] is None]
    def root_of(i):
        depth = 1
        while frames[i]["parent"] is not None:
            i = frames[i]["parent"]
            depth += 1
            assert depth <= 4
        return i, depth
    for i, frame in enumerate(frames):
        frame["root"], frame["depth"] = root_of(i)
    for motif in motifs:
        rect = motif["rect"]
        containers = [i for i, frame in enumerate(frames) if _contains(frame["rect"], rect)]
        for frame in frames:
            a = frame["rect"]
            assert _contains(a, rect) or _separate(a, rect)
        if containers:
            parent = max(containers, key=lambda i: frames[i]["depth"])
            motif["root"] = frames[parent]["root"]
            motif["level"] = frames[parent]["depth"] + 1
            assert len(containers) == frames[parent]["depth"]
        else:
            motif["root"] = None
            motif["level"] = 0
    child_counts = [0] * len(frames)
    for frame in frames:
        if frame["parent"] is not None:
            child_counts[frame["parent"]] += 1
    recursive = False
    for frame in frames:
        if frame["depth"] >= 3 and frame["color"] != frames[frame["root"]]["color"] and frames[frame["parent"]]["color"] != frames[frame["root"]]["color"]:
            recursive = True
    for motif in motifs:
        if motif["level"] >= 3:
            root_color = frames[motif["root"]]["color"]
            if motif["color"] != root_color:
                recursive = True
    return {"background": bg, "frames": frames, "roots": roots, "motifs": motifs, "child_counts": child_counts, "recursive": recursive,
            "max_level": max([f["depth"] for f in frames] + [m["level"] for m in motifs]),
            "branch": any(n >= 2 for n in child_counts),
            "empty_roots": sum(child_counts[i] == 0 and not any(m["root"] == i for m in motifs) for i in roots),
            "outside": sum(m["root"] is None for m in motifs)}


def _expected_output(grid, analysis):
    result = grid.copy()
    bg = analysis["background"]
    for i in analysis["roots"]:
        frame = analysis["frames"][i]
        r0, c0, r1, c1 = frame["rect"]
        src = grid[r0 + 1:r1, c0 + 1:c1]
        dst = result[r0 + 1:r1, c0 + 1:c1]
        dst[src != bg] = frame["color"]
    return result


def _choice(rng, palette, forbidden=()):
    options = [c for c in palette if c not in forbidden]
    assert options
    return int(options[int(rng.integers(len(options)))])


def _draw_frame(grid, rect, color, bg):
    for r, c in _border(rect):
        assert int(grid[r, c]) == bg
        grid[r, c] = color


def _inner_rect(rng, outer, minimum):
    r0, c0, r1, c1 = outer
    maxh = r1 - r0 - 3
    maxw = c1 - c0 - 3
    assert min(maxh, maxw) >= minimum
    hh = int(rng.integers(minimum, maxh + 1))
    ww = int(rng.integers(minimum, maxw + 1))
    rr = int(rng.integers(r0 + 2, r1 - hh))
    cc = int(rng.integers(c0 + 2, c1 - ww))
    return (rr, cc, rr + hh - 1, cc + ww - 1)


def _place_motif(grid, parent, forbidden, rng, palette, bg, root_color, required=False):
    shapes = [(1, 1), (1, 2), (1, 3), (2, 1), (3, 1), (2, 2), (2, 3), (3, 2)]
    order = rng.permutation(len(shapes))
    if required:
        order = [i for i in order if shapes[i] != (1, 1)] + [0]
    r0, c0, r1, c1 = parent
    for idx in order:
        hh, ww = shapes[int(idx)]
        candidates = []
        for r in range(r0 + 2, r1 - hh):
            for c in range(c0 + 2, c1 - ww):
                box = (r, c, r + hh - 1, c + ww - 1)
                if any(not _separate(box, x) for x in forbidden):
                    continue
                if np.all(grid[r - 1:r + hh + 1, c - 1:c + ww + 1] == bg):
                    candidates.append((r, c))
        if candidates:
            r, c = candidates[int(rng.integers(len(candidates)))]
            grid[r:r + hh, c:c + ww] = _choice(rng, palette, (root_color,))
            return True
    return False


def _draw_chain(grid, root, depth, rng, palette, bg, root_color, split):
    parent = root
    parent_color = root_color
    chain = [root]
    for level in range(2, depth + 1):
        child = _inner_rect(rng, parent, 5 + 4 * (depth - level))
        color = _choice(rng, palette, (parent_color, root_color) if level == 2 else (parent_color,))
        _draw_frame(grid, child, color, bg)
        chain.append(child)
        parent, parent_color = child, color
    if depth == 2 or split == "test":
        assert _place_motif(grid, chain[-1], [], rng, palette, bg, root_color, required=True)
    for outer, inner in zip(chain, chain[1:]):
        if rng.random() < 0.65:
            _place_motif(grid, outer, [inner], rng, palette, bg, root_color)


def _draw_branch(grid, root, rng, palette, bg, root_color):
    r0, c0, r1, c1 = root
    horizontal = (c1 - c0) >= (r1 - r0) if rng.random() < 0.65 else (c1 - c0) < (r1 - r0)
    if horizontal:
        lo, hi = c0 + 2, c1 - 2
        assert hi - lo + 1 >= 11
        cut = int(rng.integers(lo + 5, hi - 4))
        slots = [(r0 + 2, lo, r1 - 2, cut - 1), (r0 + 2, cut + 1, r1 - 2, hi)]
    else:
        lo, hi = r0 + 2, r1 - 2
        assert hi - lo + 1 >= 11
        cut = int(rng.integers(lo + 5, hi - 4))
        slots = [(lo, c0 + 2, cut - 1, c1 - 2), (cut + 1, c0 + 2, hi, c1 - 2)]
    children = []
    for slot in slots:
        sr0, sc0, sr1, sc1 = slot
        hh = int(rng.integers(5, sr1 - sr0 + 2))
        ww = int(rng.integers(5, sc1 - sc0 + 2))
        rr = int(rng.integers(sr0, sr1 - hh + 2))
        cc = int(rng.integers(sc0, sc1 - ww + 2))
        child = (rr, cc, rr + hh - 1, cc + ww - 1)
        _draw_frame(grid, child, _choice(rng, palette, (root_color,)), bg)
        children.append(child)
    for child in children:
        if rng.random() < 0.85:
            _place_motif(grid, child, [], rng, palette, bg, root_color, required=False)
    if rng.random() < 0.75:
        _place_motif(grid, root, children, rng, palette, bg, root_color)


def _root_specs(rng, split, pair_index):
    n = int(rng.integers(1, 4))
    if split == "train" and pair_index == 0:
        mode = "chain"
        depth = 2
    elif split == "train" and pair_index == 1:
        mode = "branch"
        depth = 2
    else:
        mode = "chain" if rng.random() < 0.68 else "branch"
        depth = int(rng.integers(2, 4)) if mode == "chain" else 2
    extra = 9 if n == 1 else 6 if n == 2 else 4
    if mode == "chain":
        minimum = 5 + 4 * (depth - 1)
        h = minimum + int(rng.integers(extra + 1))
        w = minimum + int(rng.integers(extra + 1))
    else:
        h, w = (11, 15) if rng.random() < 0.5 else (15, 11)
        h += int(rng.integers(extra + 1))
        w += int(rng.integers(extra + 1))
    specs = [(mode, depth, h, w)]
    for _ in range(n - 1):
        mode2 = ("empty", "motif", "chain")[int(rng.choice(3, p=[0.3, 0.4, 0.3]))]
        base = 9 if mode2 == "chain" else 5
        specs.append((mode2, 2 if mode2 == "chain" else 1,
                      base + int(rng.integers(0, 4 if n == 3 else 6)),
                      base + int(rng.integers(0, 4 if n == 3 else 6))))
    return specs


def _pack(rng, specs):
    for _ in range(80):
        rects = []
        for _, _, h, w in specs:
            candidates = [(r, c, r + h - 1, c + w - 1) for r in range(1, 30 - h) for c in range(1, 30 - w)]
            rng.shuffle(candidates)
            chosen = next((rect for rect in candidates if all(_separate(rect, other) for other in rects)), None)
            if chosen is None:
                break
            rects.append(chosen)
        if len(rects) == len(specs):
            return rects
    raise ValueError("root packing failed")


def _outside_marks(grid, rects, rng, palette, bg):
    count = int(rng.choice(4, p=[0.3, 0.4, 0.22, 0.08]))
    shapes = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2)]
    rr0 = max(1, min(x[0] for x in rects) - 3)
    cc0 = max(1, min(x[1] for x in rects) - 3)
    rr1 = min(28, max(x[2] for x in rects) + 3)
    cc1 = min(28, max(x[3] for x in rects) + 3)
    for _ in range(count):
        candidates = []
        hh, ww = shapes[int(rng.integers(len(shapes)))]
        for r in range(rr0, rr1 - hh + 2):
            for c in range(cc0, cc1 - ww + 2):
                box = (r, c, r + hh - 1, c + ww - 1)
                if all(_separate(box, x) for x in rects) and np.all(grid[r - 1:r + hh + 1, c - 1:c + ww + 1] == bg):
                    candidates.append((r, c))
        if candidates:
            r, c = candidates[int(rng.integers(len(candidates)))]
            grid[r:r + hh, c:c + ww] = _choice(rng, palette)


def _make_pair(rng, split, pair_index, params):
    bg = params["background"] if rng.random() < 0.75 else int(rng.integers(10))
    palette = [c for c in params["palette"] if c != bg]
    if len(palette) < 3:
        palette = [c for c in range(10) if c != bg]
    if split == "test" and rng.random() < 0.35:
        unseen = [c for c in range(10) if c != bg and c not in palette]
        if unseen:
            palette.append(int(rng.choice(unseen)))
    specs = _root_specs(rng, split, pair_index)
    rects = _pack(rng, specs)
    grid = np.full((30, 30), bg, dtype=np.int8)
    for (mode, depth, _, _), rect in zip(specs, rects):
        color = _choice(rng, palette)
        _draw_frame(grid, rect, color, bg)
        if mode == "chain":
            _draw_chain(grid, rect, depth, rng, palette, bg, color, split)
        elif mode == "branch":
            _draw_branch(grid, rect, rng, palette, bg, color)
        elif mode == "motif":
            _place_motif(grid, rect, [], rng, palette, bg, color, required=False)
    _outside_marks(grid, rects, rng, palette, bg)
    ys, xs = np.nonzero(grid != bg)
    top = max(0, int(ys.min()) - int(rng.integers(1, 4)))
    left = max(0, int(xs.min()) - int(rng.integers(1, 4)))
    bottom = min(30, int(ys.max()) + 1 + int(rng.integers(1, 4)))
    right = min(30, int(xs.max()) + 1 + int(rng.integers(1, 4)))
    grid = grid[top:bottom, left:right].copy()
    analysis = _analyze_input(grid)
    return grid, _expected_output(grid, analysis)


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _source_hash(grid):
    h, w = grid.shape
    return hashlib.sha256(bytes((h, w)) + bytes(int(v) for v in grid.flat)).hexdigest()


def _puzzle_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _validate_seed(seed)
    ntrain, ntest = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    background = int(rng.integers(10))
    colors = [c for c in range(10) if c != background]
    rng.shuffle(colors)
    palette = colors[:int(rng.integers(3, 9))]
    params = {"background": background, "palette": palette}
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            for _ in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                try:
                    inp, out = _make_pair(pair_rng, split, index, params)
                except (AssertionError, ValueError):
                    continue
                key = _grid_key(inp)
                if key not in seen and _source_hash(inp) not in _SOURCE_INPUT_HASHES:
                    seen.add(key)
                    puzzle[split].append({"input": inp, "output": out})
                    break
            else:
                raise ValueError(f"could not construct pair {split} {index}")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_analyses = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer) and np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            assert _source_hash(inp) not in _SOURCE_INPUT_HASHES
            analysis = _analyze_input(inp)
            assert np.array_equal(out, _expected_output(inp, analysis))
            assert np.any(inp != out)
            if split == "train":
                assert analysis["max_level"] <= 3
                train_analyses.append(analysis)
            else:
                assert analysis["max_level"] <= 4
    assert any(a["recursive"] for a in train_analyses)
    assert any(a["branch"] for a in train_analyses)


def self_test() -> None:
    batch = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in batch}) >= 4
    assert len({len(p["test"]) for p in batch}) >= 3
    assert all(_puzzle_equal(p, generate_puzzle(seed)) for seed, p in enumerate(batch[:4]))
    assert not _puzzle_equal(batch[0], batch[1])
    for train, test in ((2, 1), (10, 1), (10, 4), (7, 3)):
        p = generate_puzzle(1300 + train + test, train, test)
        assert (len(p["train"]), len(p["test"])) == (train, test)
    for kwargs in ({"num_test": 1}, {"num_test": 4}, {"num_train": 2}, {"num_train": 10}):
        generate_puzzle(1400, **kwargs)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}):
        try:
            generate_puzzle(17, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {kwargs}")
    analyses = [_analyze_input(pair["input"]) for p in batch for split in ("train", "test") for pair in p[split]]
    assert {len(a["roots"]) for a in analyses} >= {1, 2, 3}
    assert {a["max_level"] for a in analyses} >= {3, 4}
    assert any(a["empty_roots"] for a in analyses)
    assert any(a["outside"] == 0 for a in analyses) and any(a["outside"] >= 2 for a in analyses)
    assert len({pair["input"].shape for p in batch for split in ("train", "test") for pair in p[split]}) >= 35
    assert len({tuple((len(_analyze_input(x["input"])["roots"]), x["input"].shape) for x in p["train"] + p["test"]) for p in batch}) >= 35
    assert {a["background"] for a in analyses} == set(range(10))
    assert {a["outside"] for a in analyses} >= {0, 1, 2, 3}
    assert any(a["branch"] and a["outside"] >= 1 for a in analyses)
    assert any(a["empty_roots"] and a["branch"] for a in analyses)
    assert any(a["max_level"] == 4 and len(a["roots"]) >= 2 for a in analyses)
    shapes = {(m["rect"][2] - m["rect"][0] + 1, m["rect"][3] - m["rect"][1] + 1) for a in analyses for m in a["motifs"]}
    assert shapes >= {(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2)}
    gaps = {d for a in analyses for f in a["frames"] if f["parent"] is not None
            for d in (f["rect"][0] - a["frames"][f["parent"]]["rect"][0],
                      f["rect"][1] - a["frames"][f["parent"]]["rect"][1])}
    assert len(gaps) >= 5
    assert any(any(_analyze_input(x["input"])["background"] != _analyze_input(y["input"])["background"] for x in p["train"] for y in p["test"]) for p in batch)
    assert any(len({_analyze_input(x["input"])["outside"] for x in p["train"] + p["test"]}) >= 2 for p in batch)
    assert any(len({len(_analyze_input(x["input"])["roots"]) for x in p["train"] + p["test"]}) >= 2 for p in batch)
    assert any(any(a["frames"][root]["color"] == child["color"] and child["root"] != root
                   for root in a["roots"] for child in a["frames"] if child["parent"] is not None) for a in analyses)
    assert any(m["root"] is not None and
               (abs((m["rect"][0] + m["rect"][2]) - (a["frames"][m["root"]]["rect"][0] + a["frames"][m["root"]]["rect"][2])) >= 4 or
                abs((m["rect"][1] + m["rect"][3]) - (a["frames"][m["root"]]["rect"][1] + a["frames"][m["root"]]["rect"][3])) >= 4)
               for a in analyses for m in a["motifs"])
    larger = smaller = novel_root = 0
    for p in batch:
        max_train_area = max(x["input"].size for x in p["train"])
        train_root_colors = {a["frames"][i]["color"] for x in p["train"] for a in [_analyze_input(x["input"])] for i in a["roots"]}
        for x in p["test"]:
            a = _analyze_input(x["input"])
            larger += x["input"].size > max_train_area
            smaller += x["input"].size <= max_train_area
            novel_root += any(a["frames"][i]["color"] not in train_root_colors for i in a["roots"])
    assert larger and smaller and novel_root
