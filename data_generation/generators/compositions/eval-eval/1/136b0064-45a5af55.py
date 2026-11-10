"""A glyph-controlled thread on color-coded nested rectangular contours.

Only the input determines the rule, palette, topology, start, and program.
The four glyph masks are fixed family conventions inherited from parent A.
"""
from itertools import permutations
import numpy as np

# Normalized occupancy masks, independent of glyph ink.
MASKS = (
    np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
)
# CW three future cells; CCW two; inward two; CCW four.
STEPS = (3, -2, None, -4)
TRAIN_LAYOUTS = ((1, 4), (2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (1, 6))
TEST_LAYOUTS = ((3, 3), (2, 5), (3, 4), (4, 3), (4, 4),
                (2, 6), (3, 5), (5, 2), (5, 3))


def _integer(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its legal range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _ring(height, width, depth):
    """Clockwise order, starting at the ring's top left corner."""
    top, left = depth, depth
    bottom, right = height - 1 - depth, width - 1 - depth
    assert top < bottom and left < right
    return ([(top, c) for c in range(left, right + 1)]
            + [(r, right) for r in range(top + 1, bottom + 1)]
            + [(bottom, c) for c in range(right - 1, left - 1, -1)]
            + [(r, left) for r in range(bottom - 1, top, -1)])


def _read_input(value):
    """Parse visible roles and fixed glyphs; no generator state is consulted."""
    a = _grid(value)
    candidates = []
    for col in range(5, a.shape[1] - 5):
        if not np.all(a[:, col] == a[0, col]):
            continue
        pane = a[:, col + 1:]
        colors, counts = np.unique(pane, return_counts=True)
        bg = int(colors[np.argmax(counts)])
        marks = np.argwhere(pane != bg)
        if (marks.shape == (1, 2) and int(a[0, col]) != bg
                and int(pane[tuple(marks[0])]) != int(a[0, col])):
            candidates.append((col, bg, tuple(int(v) for v in marks[0])))
    if len(candidates) != 1:
        raise ValueError("input needs one full-height separator and one seeded pane")
    sep_col, bg, start = candidates[0]
    sep, mark = int(a[0, sep_col]), int(a[start[0], sep_col + 1 + start[1]])
    pane = a[:, sep_col + 1:].copy()
    height, width = pane.shape
    if min(start[0], start[1], height - 1 - start[0], width - 1 - start[1]) != 0:
        raise ValueError("seed must be on the outer contour")
    panel = a[:, :sep_col]
    blanks = np.flatnonzero(np.all(panel == bg, axis=0))
    if not len(blanks):
        raise ValueError("missing strip/program gap")
    strip_width = int(blanks[0])
    if not 1 <= strip_width <= 3:
        raise ValueError("strip width must be 1..3")
    present = np.flatnonzero(panel[:, 0] != bg)
    if not len(present) or not np.array_equal(present, np.arange(len(present))):
        raise ValueError("strip must start at the top and have no gaps")
    code = panel[:len(present), 0].copy()
    if (not 4 <= len(code) <= 11 or code[-1] != code[-2]
            or min(height, width) != 2 * (len(code) - 1)):
        raise ValueError("strip length and terminal duplication must fit the contour depths")
    if not np.all(panel[:len(code), :strip_width] == code[:, None]):
        raise ValueError("strip rows must be uniform")
    if set(int(v) for v in code) & {bg, sep, mark}:
        raise ValueError("strip colors cannot be structural colors")
    pg = panel[:, strip_width + 1:]
    if (pg.shape[1] + 1) % 4:
        raise ValueError("program lanes must be three wide with unit gaps")
    lanes = (pg.shape[1] + 1) // 4
    if not 1 <= lanes <= 5:
        raise ValueError("invalid lane count")
    used_rows = np.flatnonzero(np.any(pg != bg, axis=1))
    if not len(used_rows):
        raise ValueError("empty program")
    bands = (int(used_rows[-1]) + 2) // 4
    if not 2 <= bands <= 7:
        raise ValueError("invalid glyph band count")
    program, inks = [], []
    expected = np.full_like(panel, bg)
    expected[:len(code), :strip_width] = code[:, None]
    for lane in range(lanes):
        for band in range(bands):
            top, left = 4 * band, strip_width + 1 + 4 * lane
            block = panel[top:top + 3, left:left + 3]
            mask = block != bg
            matches = [i for i, m in enumerate(MASKS) if np.array_equal(mask, m)]
            colors = np.unique(block[mask])
            if len(matches) != 1 or len(colors) != 1 or int(colors[0]) in (sep, mark):
                raise ValueError("unknown or nonmonochromatic glyph")
            command, ink = matches[0], int(colors[0])
            program.append(command)
            inks.append(ink)
            expected[top:top + 3, left:left + 3][MASKS[command]] = ink
    if not np.array_equal(expected, panel):
        raise ValueError("unexpected panel cells or spacing")
    if 2 * program.count(2) > len(code) - 2:
        raise ValueError("inward command would leave the innermost contour")
    return {"pane": pane, "background": bg, "separator": sep, "marker": mark,
            "start": start, "code": code, "program": tuple(program),
            "inks": tuple(inks), "lanes": lanes, "bands": bands,
            "strip_width": strip_width}


def _trace(info, program=None, lengths=STEPS):
    """Return cells in temporal order, including revisits, plus command index."""
    height, width = info["pane"].shape
    r, c = info["start"]
    d = 0
    trace = []
    for index, command in enumerate(info["program"] if program is None else program):
        if command == 2:
            for _ in range(2):
                d += 1
                if d >= len(info["code"]) - 1:
                    raise ValueError("inward movement past the center")
                r = max(d, min(height - 1 - d, r))
                c = max(d, min(width - 1 - d, c))
                trace.append((r, c, d, index))
        else:
            ring = _ring(height, width, d)
            pos = ring.index((r, c))
            step = int(lengths[command])
            direction = 1 if step > 0 else -1
            for _ in range(abs(step)):
                pos = (pos + direction) % len(ring)
                r, c = ring[pos]
                trace.append((r, c, d, index))
    return trace


def _paint(info, program=None, color_mode="depth", lengths=STEPS):
    out = info["pane"].copy()
    for r, c, depth, index in _trace(info, program, lengths):
        if (r, c) == info["start"]:
            continue  # The start marker is a permanent landmark.
        color = (info["inks"][index] if color_mode == "glyph" else
                 info["code"][0] if color_mode == "outer" else
                 info["code"][::-1][depth] if color_mode == "reverse" else
                 info["code"][depth])
        out[r, c] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from only the input, using the family's fixed four-glyph language."""
    return _paint(_read_input(input_grid))


def _edge_step_check(info, out):
    """Validate topology with a local wall-following construction.

    This intentionally does not use the perimeter lists used by solve.
    """
    height, width = out.shape
    r, c = info["start"]
    d = 0
    visited = set()
    for command in info["program"]:
        if command == 2:
            for _ in range(2):
                d += 1
                r = int(np.clip(r, d, height - 1 - d))
                c = int(np.clip(c, d, width - 1 - d))
                visited.add((r, c))
        else:
            for _ in range(abs(STEPS[command])):
                top, bottom, left, right = d, height - 1 - d, d, width - 1 - d
                if STEPS[command] > 0:
                    if r == top and c < right:
                        c += 1
                    elif c == right and r < bottom:
                        r += 1
                    elif r == bottom and c > left:
                        c -= 1
                    else:
                        r -= 1
                else:
                    if r == top and c > left:
                        c -= 1
                    elif c == left and r < bottom:
                        r += 1
                    elif r == bottom and c < right:
                        c += 1
                    else:
                        r -= 1
                visited.add((r, c))
    expected = info["pane"].copy()
    for r, c in visited - {info["start"]}:
        depth = min(r, c, height - 1 - r, width - 1 - c)
        expected[r, c] = info["code"][depth]
    assert np.array_equal(out, expected), "local contour walk or depth coloring disagrees"
    return visited


def _alternatives(info):
    """Named plausible competing rules, used to require informative training."""
    p, b, l = info["program"], info["bands"], info["lanes"]
    variants = {
        "glyph_ink": _paint(info, color_mode="glyph"),
        "one_color": _paint(info, color_mode="outer"),
        "reverse_strip": _paint(info, color_mode="reverse"),
        "unit_strokes": _paint(info, lengths=(1, -1, None, -1)),
        "all_clockwise": _paint(info, lengths=(3, 2, None, 4)),
        "reverse_program": _paint(info, tuple(reversed(p))),
    }
    if l > 1:
        variants["row_major"] = _paint(info, tuple(p[lane * b + band]
                                                for band in range(b) for lane in range(l)))
        variants["reverse_lanes"] = _paint(info, tuple(p[lane * b + band]
                                                    for lane in range(l - 1, -1, -1)
                                                    for band in range(b)))
    variants["reverse_bands"] = _paint(info, tuple(p[lane * b + band]
                                                for lane in range(l)
                                                for band in range(b - 1, -1, -1)))
    height, width = info["pane"].shape
    frames = np.empty((height, width), dtype=np.int8)
    for r in range(height):
        for c in range(width):
            frames[r, c] = info["code"][min(r, c, height - 1 - r, width - 1 - c)]
    frames[info["start"]] = info["marker"]
    variants["fill_frames"] = frames
    # Keep the supplied canvas, but discard repeated strip layers; repeat the
    # last surviving color if there are fewer colors than geometric depths.
    compressed = info["code"][np.r_[True, info["code"][1:] != info["code"][:-1]]]
    changed = dict(info)
    changed["code"] = np.asarray([compressed[min(i, len(compressed) - 1)]
                                  for i in range(len(info["code"]))], dtype=np.int8)
    variants["collapse_runs"] = _paint(changed)
    return variants


def _corner_crossing(info):
    current = info["start"]
    previous_vector = None
    previous_command = -1
    for r, c, d, i in _trace(info):
        vector = r - current[0], c - current[1]
        if (i == previous_command and info["program"][i] != 2
                and previous_vector != vector):
            return True
        previous_vector, previous_command, current = vector, i, (r, c)
    return False


def _compose(code, program, inks, roles, start, strip_width, lanes, bands, h, w):
    bg, sep, mark = roles
    panel_width = strip_width + 1 + 4 * lanes - 1
    a = np.full((h, panel_width + 1 + w), bg, dtype=np.int8)
    a[:, panel_width] = sep
    a[:len(code), :strip_width] = np.asarray(code)[:, None]
    for index, command in enumerate(program):
        lane, band = divmod(index, bands)
        top, left = 4 * band, strip_width + 1 + 4 * lane
        a[top:top + 3, left:left + 3][MASKS[command]] = inks[index]
    a[start[0], panel_width + 1 + start[1]] = mark
    return a


def _sample_code(rng, depth_count, colors, mode=None):
    if mode == "new":
        # Guaranteed repeated depth and a color introduced at the center.
        prefix = [int(colors[0])] * 2
        prefix.extend(int(rng.choice(colors[:-1])) for _ in range(depth_count - 3))
        layers = prefix + [int(colors[-1])]
    elif mode == "return":
        # A genuinely nonadjacent return at the center, not a terminal run.
        layers = [int(colors[0]), int(colors[1])]
        layers.extend(int(rng.choice(colors)) for _ in range(depth_count - 3))
        if layers[-1] == int(colors[0]):
            layers[-1] = int(colors[1])
        layers.append(int(colors[0]))
    else:
        style = int(rng.integers(4))
        if style == 0:
            layers = [int(colors[i % len(colors)]) for i in range(depth_count)]
        elif style == 1:
            cuts = sorted(int(v) for v in rng.choice(np.arange(1, depth_count),
                          min(len(colors) - 1, depth_count - 1), replace=False))
            edges = [0] + cuts + [depth_count]
            layers = [int(colors[i]) for i in range(len(edges) - 1)
                      for _ in range(edges[i + 1] - edges[i])]
        else:
            layers = [int(rng.choice(colors))]
            for _ in range(depth_count - 1):
                layers.append(layers[-1] if rng.random() < 0.35 else int(rng.choice(colors)))
    return np.asarray(layers + [layers[-1]], dtype=np.int8)


def _make_pair(rng, layout, informative, previous_roles, seen):
    lanes, bands = layout
    length = lanes * bands
    strip_width = int(rng.integers(1, 4))
    panel_width = strip_width + 4 * lanes
    max_depth = min(10, (29 - panel_width) // 2)
    if informative == "new":
        depth_count = 3 if length == 4 else int(rng.choice([3, 5]))
    elif informative == "return":
        depth_count = int(rng.choice([3, 5, 7]))
        depth_count = min(depth_count, max_depth if max_depth % 2 else max_depth - 1)
    else:
        depth_count = int(rng.integers(3, max_depth + 1))
    assert 3 <= depth_count <= max_depth
    core_side = 2 * depth_count
    # The shorter canvas dimension is determined by the code. The other may
    # extend independently, giving an elongated central rectangle.
    min_height = max(core_side, 4 * bands - 1)
    max_width = 29 - panel_width
    choices = []
    if core_side >= 4 * bands - 1:
        choices.extend((core_side, w) for w in range(core_side, max_width + 1))
    choices.extend((h, core_side) for h in range(min_height, 31))
    shape_groups = [[shape for shape in choices if predicate(*shape)]
                    for predicate in (lambda h, w: h == w,
                                      lambda h, w: h < w,
                                      lambda h, w: h > w)]
    feasible_groups = [group for group in shape_groups if group]
    group = feasible_groups[int(rng.integers(len(feasible_groups)))]
    h, w = group[int(rng.integers(len(group)))]
    roles = tuple(int(v) for v in rng.choice(10, 3, replace=False))
    if previous_roles is not None and roles[0] == previous_roles[0]:
        available = [v for v in range(10) if v not in roles and v != previous_roles[0]]
        roles = (int(rng.choice(available)), roles[1], roles[2])
    available = [v for v in range(10) if v not in roles]
    # Glyph inks are deliberately independent of the shell palette.
    palette_size = (2 if informative == "new" else 3 if informative == "return"
                    else int(rng.integers(1, min(6, depth_count) + 1)))
    palette = rng.choice(available, palette_size, replace=False)
    code = _sample_code(rng, depth_count, palette, informative)
    inks_available = ([v for v in available if v not in set(int(x) for x in code)]
                      if informative or rng.random() < 0.5 else available)
    inks = [int(rng.choice(inks_available)) for _ in range(length)]
    # Informative pairs visit every depth, while free pairs can stop early.
    inward_count = ((depth_count - 1) // 2 if informative else
                     int(rng.integers(0, (depth_count - 1) // 2 + 1)))
    if informative:
        assert inward_count + 3 <= length
    pool = [0, 1, 3] if informative else []
    pool += [2] * inward_count
    pool += [int(rng.choice([0, 1, 3])) for _ in range(length - len(pool))]
    # Finite candidates; if all were unlucky, deterministic completion below
    # searches programs and seeds without changing semantics.
    candidate_programs = [tuple(int(v) for v in rng.permutation(pool)) for _ in range(80)]
    if informative:
        candidate_programs.extend(sorted(set(permutations(pool))))
    starts = _ring(h, w, 0)
    start_order = [starts[int(i)] for i in rng.permutation(len(starts))]
    for program in candidate_programs:
        if informative and program[0] == 2:
            continue
        for start in start_order:
            a = _compose(code, program, inks, roles, start, strip_width, lanes, bands, h, w)
            if _key(a) in seen:
                continue
            info = _read_input(a)
            out = _paint(info)
            if informative:
                if not _corner_crossing(info):
                    continue
                alternatives = _alternatives(info)
                # Every informative example rejects the alternatives it can
                # show. Return examples need not reject repeat compression.
                required = set(alternatives) - ({"collapse_runs"} if informative == "return" else set())
                if any(np.array_equal(out, alternatives[name]) for name in required):
                    continue
                if len(set(out.flat) - {roles[0], roles[2]}) < 2:
                    continue
            seen.add(_key(a))
            return {"input": a, "output": out}, roles
    raise AssertionError("finite informative construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer("seed", seed, 0)
    nt = _integer("num_train", num_train, 2, 10)
    nq = _integer("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen, roles = set(), None
    first_layout = (2, 2)  # Shortest complete vocabulary, four commands.
    second_layout = ((2, 3), (3, 2))[int(rng.integers(2))]
    for i in range(nt):
        layout = first_layout if i == 0 else second_layout if i == 1 else TRAIN_LAYOUTS[int(rng.integers(len(TRAIN_LAYOUTS)))]
        pair, roles = _make_pair(rng, layout, "new" if i == 0 else "return" if i == 1 else None,
                                 roles, seen)
        puzzle["train"].append(pair)
    for _ in range(nq):
        layout = TEST_LAYOUTS[int(rng.integers(len(TEST_LAYOUTS)))]
        pair, roles = _make_pair(rng, layout, None, roles, seen)
        puzzle["test"].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check returned grids, topology, evidence, and the declared split shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    rejected = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _read_input(inp)
            assert out.shape == info["pane"].shape
            visited = _edge_step_check(info, out)
            assert out[info["start"]] == info["marker"]
            assert set(int(v) for v in out.flat) <= set(int(v) for v in info["code"]) | {info["background"], info["marker"]}
            assert len(visited) < out.size - 1, "generated thread must leave empty cells"
            infos[split].append(info)
            if split == "train":
                rejected.update(name for name, alt in _alternatives(info).items()
                                if not np.array_equal(out, alt))
    training = infos["train"]
    assert sum(set(i["program"]) == {0, 1, 2, 3} for i in training) >= 2
    assert len({len(i["program"]) for i in training}) >= 2
    assert len({i["background"] for i in training}) >= 2
    assert any(i["code"][-1] not in i["code"][:-2] for i in training)
    assert any(i["code"][-1] in i["code"][:-3] and i["code"][-3] != i["code"][-1]
               for i in training)
    assert sum(_corner_crossing(i) for i in training) >= 2
    expected_rejected = {"glyph_ink", "one_color", "reverse_strip", "unit_strokes",
                         "all_clockwise", "reverse_program", "row_major", "reverse_lanes",
                         "reverse_bands", "fill_frames", "collapse_runs"}
    assert expected_rejected <= rejected, "training leaves a named shortcut indistinguishable"
    assert min(len(i["program"]) for i in infos["test"]) > max(len(i["program"]) for i in training)


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded 42-seed checks and topology/count/mutation regressions."""
    puzzles = [generate_puzzle(s) for s in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    shapes, structures, lengths, lanes, codes, starts, colors, depths = (set() for _ in range(8))
    features = {"square": 0, "wide": 0, "tall": 0, "radial_corner": 0,
                "revisit": 0, "no_inward": 0, "repeated_layer": 0, "full_depth": 0}
    for puzzle in puzzles:
        for pair in puzzle["train"] + puzzle["test"]:
            info = _read_input(pair["input"])
            assert np.array_equal(solve(pair["input"]), pair["output"])
            h, w = info["pane"].shape
            shapes.add((pair["input"].shape, (h, w)))
            lengths.add(len(info["program"]))
            lanes.add(info["lanes"])
            starts.add(info["start"])
            depths.add(len(info["code"]) - 1)
            colors.update(int(v) for v in np.asarray(pair["input"]).flat)
            norm = {}
            code = tuple(norm.setdefault(int(v), len(norm)) for v in info["code"])
            codes.add(code)
            structures.add((h, w, info["start"], info["program"], code, info["lanes"], info["strip_width"]))
            features["square" if h == w else "wide" if w > h else "tall"] += 1
            trace = _trace(info)
            cells = [(r, c) for r, c, d, i in trace]
            features["revisit"] += len(set(cells)) < len(cells) or info["start"] in cells
            features["no_inward"] += 2 not in info["program"]
            features["full_depth"] += 2 * info["program"].count(2) == len(info["code"]) - 2
            features["repeated_layer"] += np.any(info["code"][1:-1] == info["code"][:-2])
            prev = info["start"]
            for r, c, d, i in trace:
                if info["program"][i] == 2 and r != prev[0] and c != prev[1]:
                    features["radial_corner"] += 1
                prev = r, c
    assert len(shapes) >= 65 and len(structures) >= 200
    assert len(codes) >= 35 and len(starts) >= 35
    assert lanes == {1, 2, 3, 4, 5}
    assert {4, 6, 8, 9, 10, 12, 15, 16} <= lengths
    assert len(depths) >= 5 and colors == set(range(10))
    assert all(value >= 5 for value in features.values()), features
    # A hand-calculated contour walk, with an adjacent repeated strip layer.
    fixture = _compose([1, 1, 2, 2], (0, 2, 3, 1), [7] * 4,
                       (0, 9, 8), (0, 0), 1, 2, 2, 8, 6)
    expected = np.array([[8, 1, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0],
                         [0, 0, 2, 2, 0, 0], [0, 0, 2, 0, 0, 0],
                         [0, 0, 2, 2, 0, 0], [0, 0, 2, 2, 0, 0],
                         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=np.int8)
    assert np.array_equal(solve(fixture), expected)
    # Strip width has no meaning; literal palette permutations commute with
    # solving. These checks exercise input-only role recognition.
    wider_strip = _compose([1, 1, 2, 2], (0, 2, 3, 1), [7] * 4,
                           (0, 9, 8), (0, 0), 3, 2, 2, 8, 6)
    assert np.array_equal(solve(wider_strip), expected)
    matching_ink = _compose([1, 1, 2, 2], (0, 2, 3, 1), [1, 2, 1, 2],
                            (0, 9, 8), (0, 0), 1, 2, 2, 8, 6)
    assert np.array_equal(solve(matching_ink), expected)
    palette_map = np.array([4, 8, 0, 9, 3, 1, 7, 2, 5, 6], dtype=np.int8)
    assert np.array_equal(solve(palette_map[fixture]), palette_map[expected])
    revisit = _compose([1, 1, 2, 2], (0, 1, 3, 0), [7] * 4,
                       (0, 9, 8), (0, 0), 1, 2, 2, 8, 6)
    revisit_info = _read_input(revisit)
    assert _trace(revisit_info)[-1][:2] == (0, 0)
    assert solve(revisit)[0, 0] == 8
    # Largest lane/band layout permitted by the specification: this exercises
    # 35 commands, well beyond the sampled test programs, on a 27-by-30 input.
    long_program = (0, 1, 3) * 11 + (2, 0)
    long_input = _compose([1, 2, 3, 3], long_program, [7] * 35,
                          (0, 9, 8), (0, 0), 3, 5, 7, 27, 6)
    assert long_input.shape == (27, 30)
    long_info = _read_input(long_input)
    _edge_step_check(long_info, solve(long_input))
    a = generate_puzzle(98765, 10, 4)
    assert _puzzle_key(a) == _puzzle_key(generate_puzzle(98765, 10, 4))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts were accepted")
    # Local boundary formulas and perimeter ordering agree at every cell of
    # small and extreme rectangular rings, including a 2-by-long center.
    for h, w in ((6, 6), (6, 20), (30, 6), (18, 18), (30, 20)):
        for d in range(min(h, w) // 2):
            ring = _ring(h, w, d)
            assert len(ring) == len(set(ring)) == 2 * (h + w - 4 * d) - 4
            assert all(abs(r[0] - s[0]) + abs(r[1] - s[1]) == 1
                       for r, s in zip(ring, ring[1:] + ring[:1]))
    broken = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in a[s]]
              for s in ("train", "test")}
    info = _read_input(broken["test"][0]["input"])
    r, c, _, _ = next(t for t in _trace(info) if t[:2] != info["start"])
    broken["test"][0]["output"][r, c] = info["background"]
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("damaged thread was accepted")
    return {"seeds": 42, "pairs": sum(len(p[s]) for p in puzzles for s in ("train", "test")),
            "shapes": len(shapes), "structures": len(structures), "normalized_codes": len(codes),
            "depth_counts": sorted(depths), "program_lengths": sorted(lengths),
            "lane_counts": sorted(lanes), "features": {k: int(v) for k, v in features.items()}}


if __name__ == "__main__":
    print(self_test())
