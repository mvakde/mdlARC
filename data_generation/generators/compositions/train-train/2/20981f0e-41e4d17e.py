"""ARC family: global frame rails constrain local centering one axis at a time.

No parent imports, stored examples, puzzle-specific color maps or solver state.
"""
from __future__ import annotations

import numpy as np


_CELL_SIZES = (5, 7, 9)
_TEST_SIZES = ((7, 5), (5, 7), (7, 7), (9, 5), (5, 9),
               (9, 7), (7, 9), (9, 9))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and
             np.all((a >= 0) & (a <= 9)), "grid colors must be integers 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _connected(mask):
    points = {tuple(map(int, p)) for p in np.argwhere(mask)}
    if not points:
        return False
    seen = {min(points)}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if p in points and p not in seen:
                seen.add(p)
                stack.append(p)
    return seen == points


def _hollow(bitmap):
    h, w = bitmap.shape
    if h != w or h < 3 or h % 2 == 0:
        return False
    expected = np.ones((h, w), dtype=bool)
    expected[1:-1, 1:-1] = False
    return np.array_equal(bitmap, expected)


def _analyze(grid):
    """Recover all geometry and color roles solely from the visible input."""
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    _require(int(counts.max()) * 2 > grid.size, "background must be a majority")
    candidates = []
    for color in map(int, colors):
        if color == background:
            continue
        points = np.argwhere(grid == color)
        rows = tuple(map(int, np.unique(points[:, 0])))
        cols = tuple(map(int, np.unique(points[:, 1])))
        if len(rows) < 2 or len(cols) < 2:
            continue
        dr = {b - a for a, b in zip(rows, rows[1:])}
        dc = {b - a for a, b in zip(cols, cols[1:])}
        if len(dr) != 1 or len(dc) != 1:
            continue
        ch, cw = next(iter(dr)) - 1, next(iter(dc)) - 1
        if ch not in _CELL_SIZES or cw not in _CELL_SIZES:
            continue
        if len(points) == len(rows) * len(cols):
            candidates.append((color, rows, cols, ch, cw))
    _require(len(candidates) == 1, "one complete guide lattice is required")
    guide, rows, cols, ch, cw = candidates[0]
    _require(rows[0] <= 3 and cols[0] <= 3 and
             grid.shape[0] - rows[-1] - 1 <= 3 and
             grid.shape[1] - cols[-1] - 1 <= 3, "outer margins must be 0..3")
    covered = grid == guide
    motifs, frames, empty = [], [], []
    for i, r in enumerate(rows[:-1]):
        for j, c in enumerate(cols[:-1]):
            top, left = r + 1, c + 1
            block = grid[top:top + ch, left:left + cw]
            covered[top:top + ch, left:left + cw] = True
            values = set(map(int, np.unique(block))) - {background}
            _require(len(values) <= 1 and guide not in values,
                     "each cell has at most one monochrome object")
            if not values:
                empty.append((i, j))
                continue
            color = values.pop()
            mask = block == color
            _require(_connected(mask), "cell motif must be connected")
            points = np.argwhere(mask)
            lo, hi = points.min(axis=0), points.max(axis=0) + 1
            a, b = map(int, lo)
            h, w = map(int, hi - lo)
            _require(h % 2 == w % 2 == 1 and h <= ch - 2 and w <= cw - 2
                     and max(h, w) >= 3, "object bounds must be odd and leave slack")
            bitmap = mask[a:a + h, b:b + w].copy()
            record = {"cell": (i, j), "top": top + a, "left": left + b,
                      "color": color, "bitmap": bitmap}
            if _hollow(bitmap):
                _require((a, b) == ((ch - h) // 2, (cw - w) // 2),
                         "hollow emitters must be centered")
                frames.append(record)
            else:
                motifs.append(record)
    _require(np.all((grid == background) | covered),
             "foreground outside complete cells or guide dots")
    _require(frames and motifs, "at least one frame and one loose motif are required")
    frame_colors = {f["color"] for f in frames}
    _require(len(frame_colors) == 1, "frames share one rail color")
    frame_color = frame_colors.pop()
    _require(all(m["color"] != frame_color for m in motifs),
             "motif colors must differ from frame color")
    return {"background": background, "guide": guide, "frame_color": frame_color,
            "rows": rows, "cols": cols, "cell_size": (ch, cw),
            "frames": frames, "motifs": motifs, "empty": empty}


def _rails(shape, data):
    mask = np.zeros(shape, dtype=bool)
    for f in data["frames"]:
        side = f["bitmap"].shape[0]
        mask[f["top"] + side // 2, :] = True
        mask[:, f["left"] + side // 2] = True
    return mask


def _destination(motif, data, mode="constrained"):
    i, j = motif["cell"]
    h, w = motif["bitmap"].shape
    ch, cw = data["cell_size"]
    active_rows = {f["cell"][0] for f in data["frames"]}
    active_cols = {f["cell"][1] for f in data["frames"]}
    vertical = mode == "all" or (mode == "constrained" and i in active_rows)
    horizontal = mode == "all" or (mode == "constrained" and j in active_cols)
    top = data["rows"][i] + 1 + (ch - h) // 2 if vertical else motif["top"]
    left = data["cols"][j] + 1 + (cw - w) // 2 if horizontal else motif["left"]
    return top, left


def _moved_scene(grid, data, mode="constrained"):
    result = grid.copy()
    for m in data["motifs"]:
        h, w = m["bitmap"].shape
        region = result[m["top"]:m["top"] + h, m["left"]:m["left"] + w]
        region[m["bitmap"]] = data["background"]
    for m in data["motifs"]:
        r, c = _destination(m, data, mode)
        h, w = m["bitmap"].shape
        result[r:r + h, c:c + w][m["bitmap"]] = m["color"]
    return result


def _render(grid, data, mode="constrained"):
    result = _moved_scene(grid, data, mode)
    result[_rails(grid.shape, data) & (result == data["background"])] = data["frame_color"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    return _render(grid, _analyze(grid))


def _bitmap(rng, height, width, asymmetric=False):
    """Connected tight box, with hollow perimeters reserved for emitters."""
    if asymmetric and height > 1 and width > 1:
        result = np.zeros((height, width), dtype=bool)
        result[:, 0] = result[-1, :] = True
    else:
        for _ in range(40):
            result = np.zeros((height, width), dtype=bool)
            if rng.random() < 0.25:
                result[:] = True
            else:
                r = c = 0
                result[r, c] = True
                steps = [0] * (height - 1) + [1] * (width - 1)
                for step in rng.permutation(steps):
                    r += int(step == 0)
                    c += int(step == 1)
                    result[r, c] = True
                for _ in range(int(rng.integers(0, height * width // 2 + 1))):
                    points = np.argwhere(result)
                    r, c = map(int, points[int(rng.integers(len(points)))])
                    choices = [(r + dr, c + dc) for dr, dc in
                               ((-1, 0), (1, 0), (0, -1), (0, 1))
                               if 0 <= r + dr < height and 0 <= c + dc < width]
                    if choices:
                        rr, cc = choices[int(rng.integers(len(choices)))]
                        result[rr, cc] = True
            if not _hollow(result):
                break
        else:
            raise RuntimeError("could not sample a non-frame motif")
    if rng.integers(2):
        result = result[::-1, :]
    if rng.integers(2):
        result = result[:, ::-1]
    return result.copy()


def _layouts(ch, cw, minimum=2):
    return [(r, c) for r in range(1, 5) for c in range(1, 5)
            if r * c >= minimum and r * (ch + 1) + 1 <= 30
            and c * (cw + 1) + 1 <= 30]


def _offset(rng, slack, condition="any"):
    center = slack // 2
    if condition == "center":
        return center
    choices = [i for i in range(slack + 1)
               if condition != "off" or i != center]
    return choices[int(rng.integers(len(choices)))]


def _axis(rng, count, cell_size):
    minimum = count * (cell_size + 1) + 1
    origin = int(rng.integers(min(3, 30 - minimum) + 1))
    tail = int(rng.integers(min(3, 30 - minimum - origin) + 1))
    coordinates = tuple(origin + k * (cell_size + 1) for k in range(count + 1))
    return coordinates, minimum + origin + tail


def _make_pair(rng, mode, excluded_roles=()):
    if mode == "intro":
        ch = cw = 5
        nr, nc = 2, 3
        frame_cells = {(0, 0)}
        empty_cells = {(0, 2)}
        cells = [(i, j) for i in range(nr) for j in range(nc)]
        occupied = set(cells) - frame_cells - empty_cells
        large_cell = None
    elif mode == "interaction":
        ch, cw = ((7, 5), (5, 7), (7, 7))[int(rng.integers(3))]
        nr = nc = 3
        frame_cells = {(0, 0), (1, 1)}
        empty_cells = {(1, 2)}
        cells = [(i, j) for i in range(nr) for j in range(nc)]
        occupied = set(cells) - frame_cells - empty_cells
        large_cell = None
    else:
        if mode == "test":
            ch, cw = _TEST_SIZES[int(rng.integers(len(_TEST_SIZES)))]
        else:
            ch, cw = (int(rng.choice(_CELL_SIZES)), int(rng.choice(_CELL_SIZES)))
        layouts = _layouts(ch, cw, 4 if mode == "test" else 2)
        nr, nc = layouts[int(rng.integers(len(layouts)))]
        cells = [(i, j) for i in range(nr) for j in range(nc)]
        order = [cells[int(k)] for k in rng.permutation(len(cells))]
        frame_count = int(rng.integers(1, min(4, len(cells) - 1) + 1))
        frame_cells = set(order[:frame_count])
        available = sorted(set(cells) - frame_cells)
        active_rows = {i for i, _ in frame_cells}
        active_cols = {j for _, j in frame_cells}
        reached = [(i, j) for i, j in available if i in active_rows or j in active_cols]
        forced = reached[int(rng.integers(len(reached)))]
        density = float(rng.uniform(0.5, 0.95))
        occupied = {cell for cell in available if rng.random() < density} | {forced}
        large_cell = forced if mode == "test" else None
        empty_cells = set(available) - occupied

    rows, height = _axis(rng, nr, ch)
    cols, width = _axis(rng, nc, cw)
    available_colors = [c for c in range(10) if c not in excluded_roles]
    palette = [int(c) for c in rng.permutation(available_colors)]
    background, guide, frame_color = palette[:3]
    motif_palette = palette[3:6]
    color_count = 1 if mode == "intro" else int(rng.integers(2, 4))
    grid = np.full((height, width), background, dtype=np.int8)
    grid[np.ix_(rows, cols)] = guide
    frames, motifs = [], []
    for i, j in sorted(frame_cells):
        sides = list(range(3, min(ch, cw) - 1, 2))
        side = int(rng.choice(sides))
        bitmap = np.ones((side, side), dtype=bool)
        bitmap[1:-1, 1:-1] = False
        top = rows[i] + 1 + (ch - side) // 2
        left = cols[j] + 1 + (cw - side) // 2
        grid[top:top + side, left:left + side][bitmap] = frame_color
        frames.append({"cell": (i, j), "top": top, "left": left,
                       "color": frame_color, "bitmap": bitmap})
    for index, (i, j) in enumerate(sorted(occupied)):
        if (i, j) == large_cell:
            bounds = [(a, b) for a in range(1, ch, 2) for b in range(1, cw, 2)
                      if a <= ch - 2 and b <= cw - 2 and max(a, b) >= 5]
            h, w = bounds[int(rng.integers(len(bounds)))]
        elif mode == "intro" and (i, j) == (1, 2):
            h, w = ((1, 3), (3, 1))[int(rng.integers(2))]
        elif mode in ("intro", "interaction"):
            h = w = 3
        else:
            h, w = ((1, 3), (3, 1), (3, 3))[int(rng.integers(3))]
        asymmetric = (mode == "intro" and (i, j) == (0, 1)) or (
            mode == "interaction" and (i, j) == (0, 1))
        bitmap = _bitmap(rng, h, w, asymmetric)
        row_condition = col_condition = "any"
        if mode == "intro" and (i, j) in {(0, 1), (1, 0), (1, 1)}:
            row_condition = col_condition = "off"
        elif mode == "interaction":
            row_condition = col_condition = "off"
            if (i, j) == (0, 2):
                row_condition = "center"  # Already aligned on its only constrained axis.
            if (i, j) == (2, 0):
                col_condition = "center"
        elif mode in ("random", "test") and (i, j) == forced:
            row_condition = col_condition = "off"
        top = rows[i] + 1 + _offset(rng, ch - h, row_condition)
        left = cols[j] + 1 + _offset(rng, cw - w, col_condition)
        color = motif_palette[index % color_count]
        grid[top:top + h, left:left + w][bitmap] = color
        motifs.append({"cell": (i, j), "top": top, "left": left,
                       "color": color, "bitmap": bitmap})
    # Ground truth uses construction records; solve independently recovers these
    # records from the input and has no access to this local generation state.
    data = {"background": background, "guide": guide, "frame_color": frame_color,
            "rows": rows, "cols": cols, "cell_size": (ch, cw),
            "frames": frames, "motifs": motifs, "empty": sorted(empty_cells)}
    output = _render(grid, data)
    rotation = int(rng.integers(4))
    grid, output = np.rot90(grid, rotation), np.rot90(output, rotation)
    if rng.integers(2):
        grid, output = grid[:, ::-1], output[:, ::-1]
    return {"input": grid.copy(), "output": output.copy()}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train_count = _count(num_train, "num_train", 2, 10)
    test_count = _count(num_test, "num_test", 1, 4)
    _require(train_count > test_count, "num_train must exceed num_test")
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
             and int(seed) >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    puzzle = {"train": [], "test": []}
    seen = set()
    first_roles = ()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            mode = ("test" if split == "test" else
                    "intro" if index == 0 else "interaction" if index == 1 else "random")
            for _ in range(40):
                pair = _make_pair(rng, mode, first_roles if mode == "interaction" else ())
                key = _key(pair["input"])
                if key in seen:
                    continue
                # This is also a guard on any future sampler extension.
                data = _analyze(pair["input"])
                _require(np.array_equal(_render(pair["input"], data), pair["output"]),
                         "construction and visible-input interpretation disagree")
                seen.add(key)
                puzzle[split].append(pair)
                if mode == "intro":
                    first_roles = (data["background"], data["guide"], data["frame_color"])
                break
            else:
                raise RuntimeError("could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _motif_facts(data):
    active_rows = {f["cell"][0] for f in data["frames"]}
    active_cols = {f["cell"][1] for f in data["frames"]}
    ch, cw = data["cell_size"]
    facts = []
    for m in data["motifs"]:
        i, j = m["cell"]
        h, w = m["bitmap"].shape
        v, u = i in active_rows, j in active_cols
        r, c = _destination(m, data)
        centered_r = data["rows"][i] + 1 + (ch - h) // 2
        centered_c = data["cols"][j] + 1 + (cw - w) // 2
        facts.append({"class": (v, u), "moving": (r != m["top"], c != m["left"]),
                      "off": (m["top"] != centered_r, m["left"] != centered_c),
                      "bounds": (h, w), "bitmap": (h, w, m["bitmap"].tobytes())})
    return facts


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train/test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    a = _count(len(puzzle["train"]), "num_train", 2, 10)
    b = _count(len(puzzle["test"]), "num_test", 1, 4)
    _require(a > b, "num_train must exceed num_test")
    seen, train_data, train_facts = set(), [], []
    omit_a_diff = omit_b_diff = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input/output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "output dimensions must be unchanged")
            _require(_key(inp) not in seen, "inputs must be distinct")
            seen.add(_key(inp))
            d = _analyze(inp)
            facts = _motif_facts(d)
            _require(np.array_equal(_render(inp, d), out), "incorrect transformed output")
            _require(np.array_equal(out[inp == d["guide"]], inp[inp == d["guide"]]),
                     "guides must survive")
            for f in d["frames"]:
                h, w = f["bitmap"].shape
                _require(np.all(out[f["top"]:f["top"] + h,
                                    f["left"]:f["left"] + w][f["bitmap"]] == f["color"]),
                         "frame pixels must survive")
            for m, fact in zip(d["motifs"], facts):
                i, j = m["cell"]
                top, left = d["rows"][i] + 1, d["cols"][j] + 1
                ch, cw = d["cell_size"]
                r, c = _destination(m, d)
                h, w = m["bitmap"].shape
                expected = np.zeros((ch, cw), dtype=bool)
                expected[r - top:r - top + h, c - left:c - left + w] = m["bitmap"]
                _require(np.array_equal(out[top:top + ch, left:left + cw] == m["color"],
                                        expected), "each motif must keep its exact bitmap")
                if fact["class"] == (False, False):
                    _require(np.array_equal(out[top:top + ch, left:left + cw],
                                            inp[top:top + ch, left:left + cw]),
                             "inactive cells must be identical")
            for color in {m["color"] for m in d["motifs"]}:
                _require(np.count_nonzero(inp == color) == np.count_nonzero(out == color),
                         "motif color counts must be conserved")
            moved = _moved_scene(inp, d)
            rails = _rails(inp.shape, d)
            _require(np.all(out[rails & (moved == d["background"])] == d["frame_color"]),
                     "all unobstructed rail cells must be painted")
            _require(np.array_equal(out[~rails], moved[~rails]), "paint must stay on rails")
            _require(set(map(int, np.unique(out))) == set(map(int, np.unique(inp))),
                     "all palette roles must be visible in the input")
            if split == "train":
                train_data.append(d)
                train_facts.extend(facts)
                omit_a_diff |= not np.array_equal(_render(inp, d, "none"), out)
                omit_b_diff |= not np.array_equal(_render(inp, d, "all"), out)
                _require(all(max(f["bounds"]) <= 3 for f in facts),
                         "training uses small payload bounds")
            else:
                _require(any(max(f["bounds"]) >= 5 and any(f["class"])
                             and any(f["moving"]) for f in facts),
                         "test must move a held-out larger motif")
    _require({f["class"] for f in train_facts} ==
             {(False, False), (True, False), (False, True), (True, True)},
             "training must show all four constraint combinations")
    for cls in ((True, False), (False, True), (True, True)):
        _require(any(f["class"] == cls and all(f["off"]) and any(f["moving"])
                     for f in train_facts), "missing displaced active witness")
    _require(any(f["class"] == (False, False) and all(f["off"])
                 for f in train_facts), "missing off-center inactive witness")
    _require(any(any(f["class"]) and not any(f["moving"]) for f in train_facts),
             "missing already aligned active witness")
    _require(any(d["empty"] for d in train_data), "training must include empty cells")
    _require({len(d["frames"]) for d in train_data} >= {1, 2},
             "training must show single and multiple emitters")
    _require(len({d["cell_size"] for d in train_data}) >= 2,
             "training must expose changing cell geometry")
    for role in ("background", "guide", "frame_color"):
        _require(len({d[role] for d in train_data}) >= 2, "training roles must change color")
    _require(omit_a_diff and omit_b_diff, "both parent ablations must change a training answer")


def _equal(first, second):
    return all(len(first[s]) == len(second[s]) and
               all(np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
                   for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """Bounded semantic, diversity and metamorphic checks over 42 seeds."""
    shapes, layouts, cell_sizes, frame_sides = set(), set(), set(), set()
    structures, classes, bounds, palettes = set(), set(), set(), set()
    counts, empty_counts, offsets = set(), [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        _require(_equal(puzzle, generate_puzzle(seed)), "seed determinism failed")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                d = _analyze(inp)
                _require(np.array_equal(solve(inp), out), "input-only solving failed")
                shapes.add(inp.shape)
                layouts.add((len(d["rows"]) - 1, len(d["cols"]) - 1))
                cell_sizes.add(d["cell_size"])
                frame_sides.update(f["bitmap"].shape[0] for f in d["frames"])
                counts.add(len(d["frames"]))
                empty_counts.append(len(d["empty"]))
                palettes.add((d["background"], d["guide"], d["frame_color"]))
                for f in _motif_facts(d):
                    classes.add(f["class"])
                    bounds.add(f["bounds"])
                    offsets.add(f["off"])
                roles = {}
                structures.add(tuple(roles.setdefault(int(x), len(roles)) for x in inp.flat)
                               + inp.shape)
        # Whole-scene symmetries must commute with solving, including non-square cells.
        pair = puzzle["test"][0]
        for k in (1, 2, 3):
            _require(np.array_equal(solve(np.rot90(pair["input"], k)),
                                    np.rot90(pair["output"], k)), "rotation equivariance failed")
        _require(np.array_equal(solve(pair["input"][:, ::-1]), pair["output"][:, ::-1]),
                 "reflection equivariance failed")
        permutation = np.random.default_rng(seed + 1000).permutation(10)
        _require(np.array_equal(solve(permutation[pair["input"]]),
                                permutation[pair["output"]]), "palette equivariance failed")
    _require(len(shapes) >= 60 and len(structures) >= 180, "insufficient scene variation")
    _require(len(layouts) >= 10 and len(cell_sizes) == 9, "insufficient lattice variation")
    _require(frame_sides == {3, 5, 7} and counts == {1, 2, 3, 4},
             "insufficient emitter variation")
    _require(len(classes) == 4 and len(bounds) >= 10 and len(offsets) == 4,
             "insufficient motif variation")
    _require(max(empty_counts) >= 3 and len(palettes) >= 100, "insufficient occupancy/palette variation")
    # Exercise every legal count request with generation's explanatory witnesses intact.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(2000 + 10 * a + b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "count request failed")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {a}/{b}")
    _require(not _equal(generate_puzzle(701), generate_puzzle(702)),
             "different seeds must produce different puzzles")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, count boundaries, family invariants, equivariance)")
