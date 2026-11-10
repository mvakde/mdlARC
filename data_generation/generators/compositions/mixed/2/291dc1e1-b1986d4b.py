"""L-key slots serialize a pooled square inventory as occurrence-rank nests.

Only standard library and NumPy are used. No files are read or written here.
The two-row fast arm makes the key convention visible to an input-only solver.
"""

from __future__ import annotations

import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} outside the permitted range")
    return value


def _counts(num_train, num_test):
    a = _integer("num_train", num_train, 2, 10)
    b = _integer("num_test", num_test, 1, 4)
    _require(a > b, "num_train must exceed num_test")
    return a, b


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "invalid grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "grid must have integer cells")
    _require(bool(np.all((0 <= g) & (g <= 9))), "grid colors must be 0..9")
    return g.astype(np.int8, copy=False)


def _digest(g):
    a = np.asarray(g, dtype=np.uint8)
    return hashlib.sha256(bytes(a.shape) + a.tobytes()).hexdigest()


def _components(mask):
    """Four-connected components, returned as half-open boxes and cell counts."""
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for q in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        ys, xs = zip(*cells)
        result.append((int(min(ys)), int(max(ys))+1,
                       int(min(xs)), int(max(xs))+1, len(cells)))
    return result


def _read_key(grid):
    """Properly rotate the fast direction right; never reflect to normalize."""
    matches = []
    for rotation in range(4):
        normal = np.rot90(grid, rotation)
        if min(normal.shape) < 7:
            continue
        for up in (False, True):
            # Reflection here checks border geometry only. Payload is read from
            # normal, not aligned: the physical slow-axis handedness survives.
            aligned = normal[::-1] if up else normal
            origin, fast, slow, bg = map(int, (aligned[0, 0], aligned[0, 1],
                                              aligned[2, 0], aligned[-1, -1]))
            if len({origin, fast, slow, bg}) != 4:
                continue
            if (np.all(aligned[:2, 0] == origin)
                    and np.all(aligned[:2, 1:] == fast)
                    and np.all(aligned[2:, 0] == slow)
                    and np.all(aligned[2, 1:] == bg)
                    and np.all(aligned[-1, 1:] == bg)
                    and np.all(aligned[2:, -1] == bg)):
                matches.append((normal.copy(), up, rotation, bg, origin, fast, slow))
    _require(len(matches) == 1, "expected one L-key with a two-row fast arm")
    return matches[0]


def _analyze(input_grid):
    grid = _grid(input_grid)
    normal, up, rotation, bg, origin, fast, slow = _read_key(grid)
    mask = normal != bg
    mask[:, 0] = False
    if up:
        mask[-2:] = False
    else:
        mask[:2] = False
    boxes = _components(mask)
    _require(2 <= len(boxes) <= 6, "expected 2..6 carrier strips")
    lanes, strips, classes = {}, [], {}
    carrier = None
    for y0, y1, x0, x1, count in boxes:
        h, w = y1-y0, x1-x0
        _require(count == h*w and w > h >= 4, "strip must be a full elongated rectangle")
        _require(x0 >= 2 and x1 < normal.shape[1], "strip needs clear horizontal margins")
        _require((y0 >= 1 and y1 <= normal.shape[0]-3) if up
                 else (y0 >= 3 and y1 < normal.shape[0]), "strip touches key or boundary")
        block = normal[y0:y1, x0:x1]
        color = int(block[0, 0])
        if carrier is None:
            carrier = color
        _require(color == carrier and color not in (bg, origin, fast, slow), "ambiguous carrier color")
        _require(np.all(block[0] == color) and np.all(block[-1] == color)
                 and np.all(block[:, 0] == color) and np.all(block[:, -1] == color),
                 "carrier needs an intact one-cell rim")
        local = {}
        for payload_color in map(int, np.unique(block)):
            if payload_color == carrier:
                continue
            for a, b, c, d, area in _components(block == payload_color):
                side = b-a
                _require(side == d-c and 2 <= side <= 5 and area == side*side,
                         "payload must be a complete solid square, not a fragment")
                _require(a >= 1 and c >= 1 and b < h and d < w, "square crosses the carrier rim")
                _require(side not in local, "at most one square of a size per carrier")
                local[side] = payload_color
                if side in classes:
                    _require(classes[side][0] == payload_color, "same-size color conflict")
                    classes[side][1] += 1
                else:
                    classes[side] = [payload_color, 1]
        _require(local, "empty carrier")
        index = len(strips)
        strips.append({"box": (y0, y1, x0, x1), "local": local})
        lanes.setdefault((y0, y1), []).append(index)
    _require(2 <= len(classes) <= 4, "expected 2..4 square classes")
    _require(len({v[0] for v in classes.values()}) == len(classes), "size classes need different colors")
    _require(max(v[1] for v in classes.values()) == len(strips),
             "the maximum square multiplicity must equal the number of slots")
    lane_boxes = sorted(lanes)
    _require(1 <= len(lane_boxes) <= 4, "too many lanes")
    _require(all(b[0] > a[1] for a, b in zip(lane_boxes, lane_boxes[1:])),
             "lane bands must be aligned, disjoint and separated")
    groups = []
    for band in lane_boxes:
        members = sorted(lanes[band], key=lambda i: strips[i]["box"][2])
        _require(all(strips[j]["box"][2] > strips[i]["box"][3]
                     for i, j in zip(members, members[1:])), "carriers must be separated")
        groups.append(members)
    ranks, rank = {}, 1
    for group in (groups[::-1] if up else groups):
        for index in group:
            ranks[index] = rank
            rank += 1
    return {"normal": normal, "up": up, "rotation": rotation, "background": bg,
            "carrier": carrier, "guides": (origin, fast, slow), "strips": strips,
            "groups": groups, "ranks": ranks, "classes": classes,
            "rank_groups": [[ranks[i] for i in group] for group in groups]}


def _nest(classes, rank):
    eligible = sorted((s for s, (_, n) in classes.items() if n >= rank), reverse=True)
    _require(bool(eligible), "empty rank")
    tile = np.empty((eligible[0], eligible[0]), dtype=np.int8)
    for side in eligible:
        tile[:side, :side] = classes[side][0]
    return tile


def _pack(tile_groups, bg):
    heights = [max(t.shape[0] for t in group)+1 for group in tile_groups]
    widths = [sum(t.shape[1]+1 for t in group) for group in tile_groups]
    _require(sum(heights) <= 30 and max(widths) <= 30, "output exceeds ARC bounds")
    output = np.full((sum(heights), max(widths)), bg, dtype=np.int8)
    y = 0
    for group, height in zip(tile_groups, heights):
        x = 0
        for tile in group:
            side = tile.shape[0]
            output[y:y+side, x:x+side] = tile
            x += side+1
        y += height
    return output


def _render(classes, rank_groups, bg):
    return _pack([[_nest(classes, rank) for rank in group] for group in rank_groups], bg)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer solely from visible key, carriers and square components."""
    facts = _analyze(input_grid)
    return _render(facts["classes"], facts["rank_groups"], facts["background"])


def _partitions(n):
    if n == 0:
        yield ()
    else:
        for first in range(1, min(3, n)+1):
            for rest in _partitions(n-first):
                if len(rest) < 4:
                    yield (first,)+rest


def _spread(rng, total, n):
    return rng.multinomial(total, np.full(n, 1.0/n)).astype(int).tolist()


def _palette(rng):
    p = [int(c) for c in rng.permutation(10)]
    bg, carrier, origin, fast, slow = p[:5]
    payload_pool = [c for c in range(10) if c not in (bg, carrier)]
    rng.shuffle(payload_pool)
    return (bg, carrier, origin, fast, slow), dict(zip(range(2, 6), payload_pool[:4]))


def _sample_inventory(rng, witness):
    if witness is not None:
        n = int(rng.integers(3, 5))
        # Together the two minimum-count demonstrations expose all side lengths.
        sides = (2, 3, 4) if witness == 0 else (2, 3, 5)
        leader = 0 if witness == 0 else int(rng.integers(1, 3))
        counts = [1, int(rng.integers(2, n)), n]
        rng.shuffle(counts)
        j = counts.index(n)
        counts[j], counts[leader] = counts[leader], counts[j]
    else:
        n = int(rng.integers(2, 7))
        k = int(rng.choice((2, 3, 4), p=(.30, .50, .20)))
        sides = tuple(sorted(map(int, rng.choice(np.arange(2, 6), k, replace=False))))
        counts = [int(c) for c in rng.integers(1, n+1, size=k)]
        counts[int(rng.integers(k))] = n
    items = [[] for _ in range(n)]
    for side, count in zip(sides, counts):
        for index in rng.choice(n, count, replace=False):
            items[int(index)].append(side)
    return sides, counts, items


def _layout_options(items, witness):
    options = []
    for populations in _partitions(len(items)):
        if witness and (len(populations) < 2 or max(populations) < 2):
            continue
        groups, start, heights, widths = [], 0, [], []
        for count in populations:
            group = items[start:start+count]
            height = max(s for item in group for s in item)+2
            group_widths = [max(height+1, sum(item)+len(item)+1) for item in group]
            groups.append(list(range(start, start+count)))
            heights.append(height)
            widths.append(group_widths)
            start += count
        hmin = 2+sum(heights)+len(populations)+1
        wmin = 1+max(sum(ws)+len(ws)+1 for ws in widths)
        if hmin <= 30 and wmin <= 30:
            options.append((groups, heights, widths, hmin, wmin))
    return options


def _make_pair(rng, roles, colors, configuration, witness=None):
    bg, carrier, origin, fast, slow = roles
    rotation, up = configuration
    for _ in range(96):
        sides, counts, items = _sample_inventory(rng, witness)
        options = _layout_options(items, witness is not None)
        if not options:
            continue
        groups, heights, widths, hmin, wmin = options[int(rng.integers(len(options)))]
        h = hmin+int(rng.integers(31-hmin))
        w = wmin+int(rng.integers(31-wmin))
        grid = np.full((h, w), bg, dtype=np.int8)
        grid[:2, 0], grid[:2, 1:], grid[2:, 0] = origin, fast, slow
        vertical_gaps = [1+x for x in _spread(rng, h-hmin, len(groups)+1)]
        y = 2+vertical_gaps[0]
        for lane, (group, height, ws) in enumerate(zip(groups, heights, widths)):
            spare = w-1-(sum(ws)+len(ws)+1)
            allocations = _spread(rng, spare, 2*len(group)+1)
            gaps = [1+a for a in allocations[:len(group)+1]]
            lengths = [s+a for s, a in zip(ws, allocations[len(group)+1:])]
            x = 1+gaps[0]
            for j, (index, length) in enumerate(zip(group, lengths)):
                grid[y:y+height, x:x+length] = carrier
                sequence = items[index].copy()
                rng.shuffle(sequence)
                extra = length-(sum(sequence)+len(sequence)+1)
                inside_gaps = [1+a for a in _spread(rng, extra, len(sequence)+1)]
                sx = x+inside_gaps[0]
                for q, side in enumerate(sequence):
                    sy = y+int(rng.integers(1, height-side))
                    grid[sy:sy+side, sx:sx+side] = colors[side]
                    sx += side+inside_gaps[q+1]
                x += length+gaps[j+1]
            y += height+vertical_gaps[lane+1]
        # These transforms create scenes. The solver performs only proper
        # rotation and retains the resulting physical lane order in its output.
        if up:
            grid = grid[::-1]
        grid = np.rot90(grid, -rotation).copy()
        classes = {s: [colors[s], c] for s, c in zip(sides, counts)}
        rank_groups = [[i+1 for i in group] for group in groups]
        if up:
            rank_groups.reverse()
        output = _render(classes, rank_groups, bg)
        facts = _analyze(grid)
        _require(facts["classes"] == classes, "visible inventory disagrees with construction")
        if witness is not None:
            # Both pooling and within-lane reading must affect each evidence
            # pair, even when some adjacent ranks happen to be identical.
            if (np.array_equal(output, _local_output(facts))
                    or np.array_equal(output, _reverse_fast_output(facts))):
                continue
        return {"input": grid, "output": output}
    raise RuntimeError("bounded carrier packing attempts exhausted")


def _local_output(facts):
    """Counterfactual: independently nest just the squares on each carrier."""
    return _pack([[_nest({s: (c, 1) for s, c in facts["strips"][i]["local"].items()}, 1)
                   for i in group] for group in facts["groups"]], facts["background"])


def _without_key_output(facts):
    rank, groups = 1, []
    for group in facts["groups"]:
        groups.append(list(range(rank, rank+len(group))))
        rank += len(group)
    return _render(facts["classes"], groups, facts["background"])


def _reverse_fast_output(facts):
    """Counterfactual: visit each lane in the opposite fast direction."""
    return _render(facts["classes"], [g[::-1] for g in facts["rank_groups"]],
                   facts["background"])


def _without_threshold_output(facts):
    """Counterfactual: keep every size class alive at every output slot."""
    classes = {s: (c, len(facts["strips"])) for s, (c, _) in facts["classes"].items()}
    return _render(classes, facts["rank_groups"], facts["background"])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer("seed", seed, 0)
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    base_rotation = int(rng.integers(4))
    train_configs = [(base_rotation, False), ((base_rotation+1) % 4, True)]
    test_configs = [(base_rotation, True), ((base_rotation+1) % 4, False)]
    stable = bool(rng.random() < .30)
    shared_roles, shared_colors = _palette(rng)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            configuration = ((train_configs[index] if index < 2 else train_configs[int(rng.integers(2))])
                             if split == "train" else test_configs[index % 2])
            witness = index if split == "train" and index < 2 else None
            for _ in range(16):
                roles, colors = (shared_roles, shared_colors) if stable else _palette(rng)
                pair = _make_pair(rng, roles, colors, configuration, witness)
                key = _digest(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input attempts exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            key = _digest(grid)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            facts = _analyze(grid)
            expected = _render(facts["classes"], facts["rank_groups"], facts["background"])
            _require(np.array_equal(expected, output), "incorrect rank composite placement")
            # A second cellwise check verifies overwrite direction and all
            # separators/padding, using band membership rather than painting.
            y = 0
            for group in facts["rank_groups"]:
                sizes = [max(s for s, (_, c) in facts["classes"].items() if c >= r) for r in group]
                height = max(sizes)+1
                x = 0
                for rank, size in zip(group, sizes):
                    active = sorted(s for s, (_, c) in facts["classes"].items() if c >= rank)
                    for dy in range(size):
                        for dx in range(size):
                            side = next(s for s in active if s > max(dx, dy))
                            _require(output[y+dy, x+dx] == facts["classes"][side][0], "bad nested band")
                    _require(np.all(output[y:y+height, x+size] == facts["background"]), "missing separator")
                    _require(np.all(output[y+size:y+height, x:x+size] == facts["background"]), "bad bottom padding")
                    x += size+1
                _require(np.all(output[y:y+height, x:] == facts["background"]), "bad right padding")
                y += height
            (training if split == "train" else testing).append(facts)
    _require({f["up"] for f in training} == {False, True}, "training must demonstrate both lane directions")
    _require(len({f["rotation"] for f in training}) >= 2, "training must demonstrate fast-axis normalization")
    _require(len({tuple(sorted((s, c) for s, (_, c) in f["classes"].items()))
                  for f in training}) >= 2, "training inventories must differ")
    _require(any(not np.array_equal(_without_key_output(f),
                 _render(f["classes"], f["rank_groups"], f["background"])) for f in training),
             "omitting L-key lane order must change a demonstration")
    _require(any(min(c for _, c in f["classes"].values()) < len(f["strips"])
                 and len(f["classes"]) >= 3 for f in training), "missing exhaustion witness")
    _require(any(not np.array_equal(_without_threshold_output(f),
                 _render(f["classes"], f["rank_groups"], f["background"])) for f in training),
             "omitting occurrence thresholds must change a demonstration")
    _require(any(not np.array_equal(_reverse_fast_output(f),
                 _render(f["classes"], f["rank_groups"], f["background"])) for f in training),
             "training must distinguish within-lane direction")
    _require(any(not np.array_equal(_local_output(f),
                 _render(f["classes"], f["rank_groups"], f["background"])) for f in training),
             "missing cross-strip pooling witness")
    familiar = {(f["rotation"], f["up"]) for f in training}
    _require(all((f["rotation"], f["up"]) not in familiar for f in testing),
             "test must recombine demonstrated fast axis and lane direction")


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                digest.update(_digest(pair[name]).encode("ascii"))
    return digest.hexdigest()


def self_test():
    """Bounded in-memory suite; retain hashes/features, not 42 puzzle payloads."""
    # Hand-built regression: counts {2:3,3:2,4:1}, lanes [2,1]. The expected
    # digit grids are specified directly, independently of the tile painter.
    fixture = np.zeros((15, 25), dtype=np.int8)
    fixture[:2, 0], fixture[:2, 1:], fixture[2:, 0] = 7, 8, 6
    fixture[3:9, 2:15] = 9
    fixture[4:6, 3:5], fixture[4:7, 6:9], fixture[4:8, 10:14] = 1, 2, 3
    fixture[3:9, 16:24] = 9
    fixture[4:6, 17:19], fixture[4:7, 20:23] = 1, 2
    fixture[10:14, 2:7], fixture[11:13, 3:5] = 9, 1
    expected_down = np.array([list(map(int, row)) for row in (
        "112301120", "112301120", "222302220", "333300000", "000000000",
        "110000000", "110000000", "000000000")], dtype=np.int8)
    expected_up = np.array([list(map(int, row)) for row in (
        "1120110", "1120110", "2220000", "0000000", "1123000",
        "1123000", "2223000", "3333000", "0000000")], dtype=np.int8)
    # Move the key to the bottom without reflecting the carrier scaffold.
    up_fixture = np.pad(fixture, ((0, 2), (0, 0)), constant_values=0)
    up_fixture[:2] = 0
    up_fixture[:-2, 0] = 6
    up_fixture[-2:, 0], up_fixture[-2:, 1:] = 7, 8
    for grid, expected in ((fixture, expected_down), (up_fixture, expected_up)):
        for k in range(4):
            _require(np.array_equal(solve(np.rot90(grid, k)), expected), "hand-built regression failed")
    for damage in ("incomplete_square", "color_conflict", "broken_rim"):
        bad_input = fixture.copy()
        if damage == "incomplete_square":
            bad_input[4, 3] = 9
        elif damage == "color_conflict":
            bad_input[11:13, 3:5] = 5
        else:
            bad_input[3, 2] = 5
        try:
            solve(bad_input)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed family input accepted: " + damage)
    puzzles, shapes, profiles, layouts, palettes, configs = (set() for _ in range(6))
    lane_counts, slot_counts, class_counts, leaders = (set() for _ in range(4))
    ties = singleton = small_tail = middle_tail = large_tail = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _analyze(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve disagrees")
                # Whole-input proper rotation preserves normalized output.
                for k in range(1, 4):
                    _require(np.array_equal(solve(np.rot90(pair["input"], k)), pair["output"]),
                             "proper-rotation covariance failed")
                shapes.add((pair["input"].shape, pair["output"].shape))
                profile = tuple(sorted((s, c) for s, (_, c) in f["classes"].items()))
                profiles.add(profile)
                layouts.add((tuple(len(g) for g in f["groups"]),
                             tuple(s["box"] for s in f["strips"])))
                palettes.add((f["background"], f["carrier"], f["guides"]))
                configs.add((f["rotation"], f["up"]))
                lane_counts.add(len(f["groups"]))
                slot_counts.add(len(f["strips"]))
                class_counts.add(len(profile))
                maximum = max(c for _, c in profile)
                leading = [s for s, c in profile if c == maximum]
                leaders.update(leading)
                ties += len(leading) > 1
                singleton += any(c == 1 for _, c in profile)
                if len(leading) == 1:
                    small_tail += leading[0] == profile[0][0]
                    large_tail += leading[0] == profile[-1][0]
                    middle_tail += profile[0][0] < leading[0] < profile[-1][0]
                # Recolor all ten roles without changing any geometry.
                permutation = np.roll(np.arange(10, dtype=np.int8), seed % 9+1)
                _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]),
                         "solver depends on literal colors")
    _require(len(puzzles) == 42 and len(layouts) >= 150, "insufficient seed/layout variation")
    _require(len(shapes) >= 75 and len(profiles) >= 30 and len(palettes) >= 30,
             "insufficient geometric, inventory or palette variation")
    _require(len(configs) == 8 and len(lane_counts) >= 3 and len(slot_counts) >= 4,
             "insufficient key, lane or count variation")
    _require(class_counts == {2, 3, 4} and leaders == {2, 3, 4, 5}, "missing square-class variation")
    _require(min(ties, singleton, small_tail, middle_tail, large_tail) >= 5, "missing rank-tail cases")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1000+a*10+b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "count handling")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    p = generate_puzzle(123, 2, 1)
    _require(_puzzle_digest(p) == _puzzle_digest(generate_puzzle(123, 2, 1)), "not deterministic")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in ("train", "test")}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("damaged output accepted")
    # Ignore-threshold shortcut: every class at every rank must disagree.
    for pair in p["train"]:
        f = _analyze(pair["input"])
        _require(not np.array_equal(_without_threshold_output(f), pair["output"]),
                 "omitting occurrence threshold leaves the answer unchanged")
    return {"seeds": 42, "pairs": 210, "distinct_layouts": len(layouts),
            "shape_pairs": len(shapes), "inventories": len(profiles), "key_configurations": len(configs),
            "palette_roles": len(palettes), "lane_counts": sorted(lane_counts),
            "slot_counts": sorted(slot_counts), "class_counts": sorted(class_counts)}
