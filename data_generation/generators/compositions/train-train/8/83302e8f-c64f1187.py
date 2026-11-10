"""Toggle circuits: wall-connected rooms share the parity of local stencils.

Only NumPy and the standard library are used.  The decoder does not call any
sampling code and reads every color role and stencil from the current input.
"""

from collections import deque
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(np.all((a >= 0) & (a <= 9)), "colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _mask(code):
    return np.array([(int(code) >> i) & 1 for i in range(4)], dtype=bool).reshape(2, 2)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _edges(rows, cols):
    return [(r * cols + c, rr * cols + cc)
            for r in range(rows) for c in range(cols)
            for rr, cc in ((r + 1, c), (r, c + 1))
            if rr < rows and cc < cols]


def _groups(n, edges):
    adjacency = [[] for _ in range(n)]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    seen, groups = set(), []
    for start in range(n):
        if start in seen:
            continue
        seen.add(start)
        group, queue = [], deque([start])
        while queue:
            u = queue.popleft()
            group.append(u)
            for v in adjacency[u]:
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        groups.append(sorted(group))
    return groups


def _read_board(grid, background):
    """Find the unique closed frame, then infer the even chamber dimensions."""
    frames = []
    for wall in (int(c) for c in np.unique(grid) if c != background):
        coordinates = np.argwhere(grid == wall)
        top, left = coordinates.min(axis=0)
        bottom, right = coordinates.max(axis=0)
        board = grid[top:bottom + 1, left:right + 1]
        if min(board.shape) < 7:
            continue
        if not (np.all(board[0] == wall) and np.all(board[-1] == wall)
                and np.all(board[:, 0] == wall) and np.all(board[:, -1] == wall)):
            continue
        frames.append((wall, int(top), int(left), board))
    _require(len(frames) == 1, "one unambiguous framed board required")
    wall, top, left, board = frames[0]
    interpretations = []
    for cell_h in (2, 4, 6):
        if (board.shape[0] - 1) % (cell_h + 1):
            continue
        rows = (board.shape[0] - 1) // (cell_h + 1)
        if rows < 2:
            continue
        for cell_w in (2, 4, 6):
            if (board.shape[1] - 1) % (cell_w + 1):
                continue
            cols = (board.shape[1] - 1) // (cell_w + 1)
            if cols < 2:
                continue
            line_r = np.arange(board.shape[0]) % (cell_h + 1) == 0
            line_c = np.arange(board.shape[1]) % (cell_w + 1) == 0
            line = line_r[:, None] | line_c[None, :]
            if not np.all((board[line] == wall) | (board[line] == background)):
                continue
            if not np.all(board[np.ix_(line_r, line_c)] == wall):
                continue  # Every lattice junction survives, including the frame.
            allowed = np.full(board.shape, background, dtype=np.int8)
            allowed[line] = board[line]
            labels = np.full(rows * cols, -1, dtype=np.int8)
            valid = True
            for r in range(rows):
                for c in range(cols):
                    rr, cc = (r + 1) * (cell_h + 1) - 1, (c + 1) * (cell_w + 1) - 1
                    value = int(board[rr, cc])
                    if value == wall:
                        valid = False
                    if value != background:
                        labels[r * cols + c] = value
                    allowed[rr, cc] = value
            if not valid or not np.array_equal(allowed, board):
                continue
            openings = []
            for u, v in _edges(rows, cols):
                r, c = divmod(u, cols)
                if v == u + 1:
                    segment = board[1 + r * (cell_h + 1):(r + 1) * (cell_h + 1),
                                    (c + 1) * (cell_w + 1)]
                else:
                    segment = board[(r + 1) * (cell_h + 1),
                                    1 + c * (cell_w + 1):(c + 1) * (cell_w + 1)]
                if np.any(segment == background):
                    openings.append((u, v))
            interpretations.append(dict(board=board, wall=wall, background=background,
                                        top=top, left=left, rows=rows, cols=cols,
                                        cell_h=cell_h, cell_w=cell_w, labels=labels,
                                        line=line, openings=openings,
                                        groups=_groups(rows * cols, openings)))
    _require(len(interpretations) == 1, "chamber lattice is not uniquely recoverable")
    return interpretations[0]


def _parse(input_grid):
    grid = _grid(input_grid)
    background = int(grid[0, 0])
    _require(np.all(grid[0] == background) and np.all(grid[-1] == background)
             and np.all(grid[:, 0] == background) and np.all(grid[:, -1] == background),
             "a background margin must surround the input")
    info = _read_board(grid, background)
    top, left = info["top"], info["left"]
    h, w = info["board"].shape
    outside = grid.copy()
    outside[top:top + h, left:left + w] = background
    used = set(int(x) for x in info["labels"] if x >= 0)
    interpretations = []
    for stroke in (int(c) for c in np.unique(outside) if c != background):
        selectors = [int(c) for c in np.unique(outside)
                     if c not in (background, stroke)]
        if not 2 <= len(selectors) <= 5 or not used.issubset(selectors):
            continue
        mapping, positions = {}, {}
        expected = np.full_like(grid, background)
        valid = True
        for selector in selectors:
            coordinates = np.argwhere(outside == selector)
            if len(coordinates) != 1:
                valid = False
                break
            r, c = (int(x) for x in coordinates[0])
            if r + 3 > grid.shape[0] or c + 3 > grid.shape[1]:
                valid = False
                break
            block = grid[r + 1:r + 3, c + 1:c + 3]
            if not np.all((block == stroke) | (block == background)):
                valid = False
                break
            code = sum(int(block.flat[i] == stroke) << i for i in range(4))
            if not code:
                valid = False
                break
            mapping[selector], positions[selector] = code, (r, c)
            expected[r, c] = selector
            expected[r + 1:r + 3, c + 1:c + 3][_mask(code)] = stroke
        if not valid:
            continue
        marker_rows = {r for r, c in positions.values()}
        marker_cols = {c for r, c in positions.values()}
        if len(marker_rows) == 1 and max(r + 2 for r, c in positions.values()) <= top - 3:
            layout = "top"
            coordinates = sorted(c for r, c in positions.values())
        elif len(marker_cols) == 1 and max(c + 2 for r, c in positions.values()) <= left - 3:
            layout = "left"
            coordinates = sorted(r for r, c in positions.values())
        else:
            continue
        if any(b - a not in (4, 5) for a, b in zip(coordinates, coordinates[1:])):
            continue
        if stroke == info["wall"] or info["wall"] in selectors:
            continue
        if not np.array_equal(outside, expected):
            continue
        interpretations.append(dict(info, stroke=stroke, mapping=mapping,
                                    positions=positions, layout=layout))
    _require(len(interpretations) == 1, "one complete local legend required")
    return interpretations[0]


def _render(info, masks_by_room):
    board = info["board"]
    out = np.full(board.shape, info["background"], dtype=np.int8)
    out[board == info["wall"]] = info["wall"]
    qh, qw = info["cell_h"] // 2, info["cell_w"] // 2
    for u, code in enumerate(masks_by_room):
        r, c = divmod(u, info["cols"])
        rr, cc = 1 + r * (info["cell_h"] + 1), 1 + c * (info["cell_w"] + 1)
        mask = np.repeat(np.repeat(_mask(code), qh, axis=0), qw, axis=1)
        out[rr:rr + info["cell_h"], cc:cc + info["cell_w"]][mask] = info["stroke"]
    return out


def _net_masks(info):
    masks = np.zeros(len(info["labels"]), dtype=np.int8)
    for group in info["groups"]:
        code = 0
        for u in group:
            selector = int(info["labels"][u])
            if selector >= 0:
                code ^= info["mapping"][selector]
        masks[group] = code
    return masks


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the current legend, combine occurrences by XOR, broadcast by room."""
    info = _parse(input_grid)
    return _render(info, _net_masks(info))


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= value <= high:
            raise ValueError(f"{name} must be in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _sample_graph(rng, rows, cols, witness, test, triple):
    n, possible = rows * cols, _edges(rows, cols)
    minimum = 5 if test else 3
    for _ in range(600):
        mode = int(rng.integers(4))
        p = float(rng.uniform(.16, .44 if witness else .65))
        protected = set(int(x) for x in rng.choice(n, 2, replace=False)) if witness else set()
        edges = []
        for u, v in possible:
            if u in protected or v in protected:
                continue
            probability = p
            if mode == 1:
                probability *= 1.6 if v == u + 1 else .35
            elif mode == 2:
                probability *= .35 if v == u + 1 else 1.6
            elif mode == 3:
                # A closed internal cut produces separate substantial circuits.
                cut = cols // 2
                if u % cols < cut <= v % cols and v == u + 1:
                    probability = 0
            if rng.random() < probability:
                edges.append((u, v))
        groups = _groups(n, edges)
        large = [g for g in groups if len(g) >= minimum]
        if not large:
            continue
        main = large[int(rng.integers(len(large)))]
        others = [g for g in groups if g is not main]
        duplicates = [g for g in others if len(g) >= (3 if triple else 2)]
        singles = [g for g in others if len(g) == 1]
        if witness and (not duplicates or len(singles) < 2):
            continue
        duplicate = duplicates[int(rng.integers(len(duplicates)))] if duplicates else None
        return edges, groups, main, duplicate, singles
    raise RuntimeError("could not sample informative chamber components")


def _sample_mapping(rng, keys, changed=None):
    pairs = [(a, b) for a in range(1, 16) for b in range(1, 16)
             if a & b and a & ~b and b & ~a
             and (changed is None or a != changed)]
    a, b = pairs[int(rng.integers(len(pairs)))]
    return dict(zip(keys, [a, b] + [int(rng.integers(1, 16)) for _ in keys[2:]]))


def _make_pair(rng, *, witness, test, transpose, triple=False,
               scale=None, anchor=None, changed=None):
    if scale is None:
        scale = (int(rng.integers(1, 4)), int(rng.integers(1, 4)))
    qh, qw = scale
    cell_h, cell_w = 2 * qh, 2 * qw
    rows = int(rng.integers(3, (21 // (cell_h + 1)) + 1))
    cols = int(rng.integers(3, (27 // (cell_w + 1)) + 1))
    # Framed board H=R*(interior+1)+1; leave space for the legend and margins.
    height, width = rows * (cell_h + 1) + 1, cols * (cell_w + 1) + 1
    colors = [int(x) for x in rng.permutation(10) if x != anchor]
    background, wall, stroke = colors[:3]
    key_count = int(rng.integers(2, 6))
    keys = ([anchor] + colors[3:3 + key_count - 1]) if anchor is not None else colors[3:3 + key_count]
    mapping = _sample_mapping(rng, keys, changed)
    edges, groups, main, duplicate, singles = _sample_graph(rng, rows, cols, witness, test, triple)
    labels = np.full(rows * cols, -1, dtype=np.int8)
    for group in groups:
        count = int(rng.integers(0, min(len(group), 6 if test else 3) + 1))
        if count:
            for u in rng.choice(group, count, replace=False):
                labels[int(u)] = int(rng.choice(keys))
    main_count = int(rng.integers(4, min(6, len(main) - 1) + 1)) if test else 2
    if test:
        main_positions = [int(x) for x in rng.choice(main, main_count, replace=False)]
    else:
        internal_edges = [(u, v) for u, v in edges if u in main and v in main]
        main_positions = list(internal_edges[int(rng.integers(len(internal_edges)))])
    labels[main] = -1
    for _ in range(100):
        values = [keys[0], keys[1]] + [int(rng.choice(keys)) for _ in range(main_count - 2)]
        net = 0
        for value in values:
            net ^= mapping[value]
        if net not in (0, 15):
            break
    else:
        raise RuntimeError("could not choose a nontrivial main circuit")
    for u, value in zip(main_positions, values):
        labels[u] = value
    if witness:
        labels[duplicate] = -1
        positions = rng.choice(duplicate, 3 if triple else 2, replace=False)
        labels[positions] = int(rng.choice(keys))
        labels[singles[0][0]] = keys[0]
        labels[singles[1][0]] = -1
    board = np.full((height, width), background, dtype=np.int8)
    line_r = np.arange(height) % (cell_h + 1) == 0
    line_c = np.arange(width) % (cell_w + 1) == 0
    board[line_r[:, None] | line_c[None, :]] = wall
    for u, v in edges:
        r, c = divmod(u, cols)
        length = cell_h if v == u + 1 else cell_w
        count = int(rng.integers(1, length + 1))
        if rng.random() < .7:
            start = int(rng.integers(length - count + 1))
            offsets = np.arange(start, start + count)
        else:
            offsets = rng.choice(length, count, replace=False)
        if v == u + 1:
            board[1 + r * (cell_h + 1) + offsets, (c + 1) * (cell_w + 1)] = background
        else:
            board[(r + 1) * (cell_h + 1), 1 + c * (cell_w + 1) + offsets] = background
    for u, value in enumerate(labels):
        if value >= 0:
            r, c = divmod(u, cols)
            board[(r + 1) * (cell_h + 1) - 1, (c + 1) * (cell_w + 1) - 1] = value
    # Ground truth is composed from the sampled graph, independently of parsing.
    generated = dict(board=board, background=background, wall=wall, stroke=stroke,
                     cell_h=cell_h, cell_w=cell_w, rows=rows, cols=cols,
                     labels=labels, groups=groups, mapping=mapping)
    out = _render(generated, _net_masks(generated))
    legend_top = int(rng.integers(1, min(2, 24 - height) + 1))
    gap = 2
    board_top = legend_top + 3 + gap
    board_left = int(rng.integers(1, min(2, 29 - width) + 1))
    legend_left = int(rng.integers(1, 3))
    pitch = int(rng.choice((4, 5)))
    legend_width = 3 + (key_count - 1) * pitch
    canvas_width = max(board_left + width, legend_left + legend_width) + 1
    canvas_height = board_top + height + 1
    inp = np.full((canvas_height, canvas_width), background, dtype=np.int8)
    inp[board_top:board_top + height, board_left:board_left + width] = board
    for index, key in enumerate(rng.permutation(keys)):
        r, c = legend_top, legend_left + index * pitch
        inp[r, c] = key
        inp[r + 1:r + 3, c + 1:c + 3][_mask(mapping[int(key)])] = stroke
    if transpose:
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    # Both presentation orientations and two scales are exposed even with two trains.
    layouts = [False, True]
    layouts += [bool(rng.integers(2)) for _ in range(num_train - 2)]
    scales = [(1, 1), (int(rng.integers(1, 4)), int(rng.integers(2, 4)))]
    first = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            transpose = layouts[index] if split == "train" else bool(rng.choice(layouts))
            scale = scales[index] if split == "train" and index < 2 else None
            anchor = changed = None
            if split == "train" and index == 1:
                anchor = min(int(s) for s in first["labels"] if s >= 0)
                changed = first["mapping"][anchor]
                if transpose:
                    changed = sum(int(x) << i for i, x in enumerate(_mask(changed).T.flat))
            for _ in range(30):
                pair = _make_pair(rng, witness=split == "train" and index < 2,
                                  test=split == "test", transpose=transpose, scale=scale,
                                  triple=split == "train" and index == 1,
                                  anchor=anchor, changed=changed)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train" and index == 0:
                first = _parse(pair["input"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    labels, mapping = info["labels"], info["mapping"]
    masks = _net_masks(info)
    duplicate = odd_return = mixed = receiver = singleton = blank = False
    max_sources = 0
    adjacency = [set() for _ in labels]
    for u, v in info["openings"]:
        adjacency[u].add(v)
        adjacency[v].add(u)
    one_hop = np.zeros(len(labels), dtype=np.int8)
    for u in range(len(labels)):
        for v in adjacency[u] | {u}:
            if labels[v] >= 0:
                one_hop[u] ^= mapping[int(labels[v])]
    for group in info["groups"]:
        values = [int(labels[u]) for u in group if labels[u] >= 0]
        max_sources = max(max_sources, len(values))
        duplicate |= len(values) == 2 and values[0] == values[1] and masks[group[0]] == 0
        odd_return |= (len(values) == 3 and len(set(values)) == 1
                       and masks[group[0]] == mapping[values[0]])
        mixed |= (len(values) == 2 and values[0] != values[1]
                  and bool(mapping[values[0]] & mapping[values[1]])
                  and masks[group[0]] not in (0, 15, mapping[values[0]], mapping[values[1]]))
        receiver |= masks[group[0]] != 0 and any(labels[u] < 0 for u in group)
        singleton |= len(group) == 1 and bool(values)
        blank |= len(group) == 1 and not values
    local = [mapping[int(s)] if s >= 0 else 0 for s in labels]
    no_legend = info.copy()
    no_legend["mapping"] = {key: 15 for key in mapping}
    return dict(duplicate=duplicate, odd_return=odd_return, mixed=mixed, receiver=receiver,
                singleton=singleton, blank=blank, max_sources=max_sources,
                needs_A=not np.array_equal(masks, local),
                needs_B=not np.array_equal(masks, _net_masks(no_legend)),
                needs_multihop=any(labels[u] < 0 and masks[u] != one_hop[u]
                                   for u in range(len(labels))))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "pair lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, parsed = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _parse(inp)
            _require(out.shape == info["board"].shape, "crop must include exact frame")
            board, bg, wall, stroke = (info[k] for k in ("board", "background", "wall", "stroke"))
            _require(set(np.unique(out)).issubset({bg, wall, stroke}), "output color roles")
            _require(np.array_equal(out[info["line"]], board[info["line"]]), "wall/gap preservation")
            # Check each logical quadrant by counting source occupancy modulo two.
            # This uses arithmetic parity, rather than the decoder's bitwise fold.
            for group in info["groups"]:
                sources = [info["mapping"][int(info["labels"][u])]
                           for u in group if info["labels"][u] >= 0]
                for u in group:
                    r, c = divmod(u, info["cols"])
                    qh, qw = info["cell_h"] // 2, info["cell_w"] // 2
                    rr, cc = 1 + r * (info["cell_h"] + 1), 1 + c * (info["cell_w"] + 1)
                    for bit in range(4):
                        br, bc = divmod(bit, 2)
                        value = stroke if sum((code >> bit) & 1 for code in sources) % 2 else bg
                        _require(np.all(out[rr + br * qh:rr + (br + 1) * qh,
                                            cc + bc * qw:cc + (bc + 1) * qw] == value),
                                 "component parity or quadrant scale is incorrect")
            features = _features(info)
            if split == "train":
                _require(features["max_sources"] <= 3, "training source-count regime")
            else:
                _require(features["max_sources"] >= 4 and features["receiver"], "test composition shift")
            parsed[split].append((info, features))
    for info, features in parsed["train"][:2]:
        _require(all(features[k] for k in ("mixed", "receiver", "singleton", "blank",
                                           "needs_A", "needs_B", "needs_multihop")),
                 "missing demonstration witness")
    _require(parsed["train"][0][1]["duplicate"], "two equal instructions must demonstrate cancellation")
    _require(parsed["train"][1][1]["odd_return"], "three equal instructions must demonstrate odd parity")
    first, second = (parsed["train"][i][0] for i in range(2))
    shared = (set(int(s) for s in first["labels"] if s >= 0)
              & set(int(s) for s in second["labels"] if s >= 0))
    _require(any(first["mapping"][s] != second["mapping"][s] for s in shared), "local mapping shift")
    _require({x[0]["layout"] for x in parsed["train"]} == {"top", "left"}, "layout demonstrations")
    return None


def _metamorphic_checks():
    """Input edits check that topology, lookup, parity, and role identity matter."""
    pair = generate_puzzle(0)["train"][0]
    inp, out = pair["input"], pair["output"]
    info = _parse(inp)
    _require(np.array_equal(solve(inp.T), out.T), "transpose covariance")
    permutation = np.random.default_rng(3401).permutation(10)
    _require(np.array_equal(solve(permutation[inp]), permutation[out]), "color-role covariance")
    top, left = info["top"], info["left"]
    height, width = info["board"].shape

    def erase_marker(grid, u):
        r, c = divmod(u, info["cols"])
        grid[top + (r + 1) * (info["cell_h"] + 1) - 1,
             left + (c + 1) * (info["cell_w"] + 1) - 1] = info["background"]

    edited = inp.copy()
    for group in info["groups"]:
        sources = [u for u in group if info["labels"][u] >= 0]
        if len(sources) == 2 and info["labels"][sources[0]] == info["labels"][sources[1]]:
            for u in sources:
                erase_marker(edited, u)
            break
    _require(np.array_equal(solve(edited), out), "a duplicate pair must cancel exactly")
    edited = inp.copy()
    for u, value in enumerate(info["labels"]):
        if value >= 0:
            erase_marker(edited, u)
    expected = np.full(out.shape, info["background"], dtype=np.int8)
    expected[info["board"] == info["wall"]] = info["wall"]
    _require(np.array_equal(solve(edited), expected), "an unmarked board has zero state")
    edited = inp.copy()
    board = edited[top:top + height, left:left + width]
    board[info["line"]] = info["wall"]
    local = [info["mapping"][int(s)] if s >= 0 else 0 for s in info["labels"]]
    _require(np.array_equal(solve(edited), _render(_parse(edited), local)), "closed-wall ablation")
    _require(not np.array_equal(solve(edited), out), "A ablation must change the answer")
    for group in info["groups"]:
        if len(group) == 1 and info["labels"][group[0]] >= 0:
            selector = int(info["labels"][group[0]])
            break
    r, c = info["positions"][selector]
    code = info["mapping"][selector]
    replacement = 1 if code != 1 else 2
    edited = inp.copy()
    block = edited[r + 1:r + 3, c + 1:c + 3]
    block[:] = info["background"]
    block[_mask(replacement)] = info["stroke"]
    _require(not np.array_equal(solve(edited), out), "changing a used legend mask must matter")
    # Cover more occurrences than the sampler's test bias, and many cyclic paths.
    edited = inp.copy()
    board = edited[top:top + height, left:left + width]
    board[info["line"]] = info["background"]
    line_r = np.arange(height) % (info["cell_h"] + 1) == 0
    line_c = np.arange(width) % (info["cell_w"] + 1) == 0
    board[np.ix_(line_r, line_c)] = info["wall"]
    board[0, :] = board[-1, :] = info["wall"]
    board[:, 0] = board[:, -1] = info["wall"]
    for u in range(len(info["labels"])):
        r, c = divmod(u, info["cols"])
        board[(r + 1) * (info["cell_h"] + 1) - 1,
              (c + 1) * (info["cell_w"] + 1) - 1] = selector
    parsed = _parse(edited)
    code = info["mapping"][selector] if len(info["labels"]) % 2 else 0
    _require(np.array_equal(solve(edited), _render(parsed, [code] * len(info["labels"]))),
             "cyclic circuits count sources rather than routes")
    # The solver grammar includes two-room axes, although generation starts at three.
    small = inp.copy()
    small[top:top + height, left:left + width] = info["background"]
    hh, ww = 2 * (info["cell_h"] + 1) + 1, 2 * (info["cell_w"] + 1) + 1
    cropped = info["board"][:hh, :ww].copy()
    cropped[0, :] = cropped[-1, :] = info["wall"]
    cropped[:, 0] = cropped[:, -1] = info["wall"]
    small[top:top + hh, left:left + ww] = cropped
    _require(solve(small).shape == (hh, ww), "two-room axes must remain supported")


def self_test():
    """Bounded semantic and diversity checks over 42 complete seeded puzzles."""
    shapes, geometries, palettes, masks, signatures = set(), set(), set(), set(), set()
    component_sizes, source_counts, key_counts, layouts = set(), set(), set(), set()
    multi_aperture = cycles = rectangles = empty_components = repeated_shapes = 0
    total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s])
        signatures.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                total += 1
                info = _parse(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "input-only solver")
                shapes.add(pair["input"].shape)
                geometries.add(tuple(info[k] for k in ("rows", "cols", "cell_h", "cell_w")))
                palettes.add(tuple(info[k] for k in ("background", "wall", "stroke")))
                masks.update(info["mapping"].values())
                key_counts.add(len(info["mapping"]))
                layouts.add(info["layout"])
                component_sizes.update(len(g) for g in info["groups"])
                source_counts.update(sum(info["labels"][u] >= 0 for u in g) for g in info["groups"])
                cycles += len(info["openings"]) > len(info["labels"]) - len(info["groups"])
                rectangles += info["cell_h"] != info["cell_w"]
                repeated_shapes += len(set(info["mapping"].values())) < len(info["mapping"])
                empty_components += any(_net_masks(info)[g[0]] == 0 for g in info["groups"])
                holes = info["line"] & (info["board"] == info["background"])
                multi_aperture += int(holes.sum()) > len(info["openings"])
    _require(len(signatures) == 42, "seed variation")
    _require(len(shapes) >= 40 and len(geometries) >= 40, "geometry variation")
    _require(len(palettes) >= 100 and masks == set(range(1, 16)), "palette/stencil variation")
    _require(key_counts == {2, 3, 4, 5} and layouts == {"top", "left"}, "legend variation")
    _require({0, 1, 2, 3, 4}.issubset(source_counts) and max(component_sizes) >= 12, "circuit variation")
    _require(min(multi_aperture, cycles, rectangles, empty_components, repeated_shapes) > 0,
             "missing compatible structural variations")
    _metamorphic_checks()
    a, b = generate_puzzle(1771, 10, 4), generate_puzzle(1771, 10, 4)
    _require(all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s])
                 for k in ("input", "output")), "determinism")
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            p = generate_puzzle(42000 + ntrain * 10 + ntest, ntrain, ntest)
            _require(len(p["train"]) == ntrain and len(p["test"]) == ntest, "explicit counts")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=total, input_shapes=len(shapes), geometries=len(geometries),
                role_triples=len(palettes), masks=sorted(masks), key_counts=sorted(key_counts),
                component_sizes=sorted(component_sizes), source_counts=sorted(int(x) for x in source_counts),
                multi_aperture_pairs=multi_aperture, cyclic_pairs=cycles,
                rectangular_chamber_pairs=rectangles, zero_net_pairs=empty_components,
                repeated_legend_shape_pairs=repeated_shapes)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
