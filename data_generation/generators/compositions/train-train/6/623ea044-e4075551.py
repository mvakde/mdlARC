"""ARC family: diagonal rays escape through their matching colored wall.

The solver traces reflected rays. Generation and validation instead use an
unfolded-rectangle formula, providing two implementations of the declared rule.
Only the input determines background, walls, seeds, beam color, and exit wall.
"""

from itertools import combinations

import numpy as np


SIDES = ("top", "bottom", "left", "right")
DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
HOST_SEEDS = (
    0, 347889688, 555037985, 547988795, 365968846, 516337641, 203815780,
    295778371, 740936445, 905716247, 202760854, 428595195, 497774701,
    537363404, 66348888, 721180765, 879869333, 957564929, 424398722,
    436371297, 591750671, 759922376, 459106301, 543328824, 624566602,
    267617656, 46696781, 331414014, 90282605, 988921131, 411932051,
    282478209, 305718781, 183636655, 219639229, 859177771, 515774210,
    730717852, 700706749, 472424188, 113724794, 151077694,
)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return grid


def _isolated(positions):
    return all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1
               for a, b in combinations(positions, 2))


def _analyze(input_grid):
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    positions = [tuple(map(int, p)) for p in np.argwhere(grid != background)]
    if not 5 <= len(positions) <= 7 or not _isolated(positions):
        raise ValueError("expected four isolated wall markers and one to three seeds")
    if int(np.max(counts)) <= grid.size // 2:
        raise ValueError("background must be the unique majority")
    walls = {}
    for side, axis, extreme in (("top", 0, min), ("bottom", 0, max),
                                ("left", 1, min), ("right", 1, max)):
        coordinate = extreme(p[axis] for p in positions)
        matches = [p for p in positions if p[axis] == coordinate]
        if len(matches) != 1:
            raise ValueError("each directional extreme must be unique")
        walls[side] = matches[0]
    if len(set(walls.values())) != 4:
        raise ValueError("the four extrema must be four different markers")
    seeds = tuple(p for p in positions if p not in walls.values())
    t, b, l, r = walls["top"][0], walls["bottom"][0], walls["left"][1], walls["right"][1]
    if b - t < 6 or r - l < 6:
        raise ValueError("wall rectangle must be at least 7 by 7")
    if any(not (t + 2 <= y <= b - 2 and l + 2 <= x <= r - 2) for y, x in seeds):
        raise ValueError("seeds must be at least two cells inside the walls")
    wall_colors = {side: int(grid[p]) for side, p in walls.items()}
    if len(set(wall_colors.values())) != 4:
        raise ValueError("wall colors must be distinct")
    seed_colors = {int(grid[p]) for p in seeds}
    if len(seed_colors) != 1:
        raise ValueError("all interior seeds must have the same color")
    beam = next(iter(seed_colors))
    matching = [side for side in SIDES if wall_colors[side] == beam]
    if len(matching) != 1:
        raise ValueError("seed color must match exactly one wall")
    return {"shape": grid.shape, "background": background, "walls": walls,
            "colors": wall_colors, "seeds": seeds, "beam": beam,
            "gate": matching[0], "bounds": (t, b, l, r),
            "size": (b - t + 1, r - l + 1)}


def _frame(grid, info):
    output = grid.copy()
    t, b, l, r = info["bounds"]
    colors = info["colors"]
    # Like parent B, horizontal edges own the four corner cells.
    output[t, l:r + 1] = colors["top"]
    output[b, l:r + 1] = colors["bottom"]
    output[t + 1:b, l] = colors["left"]
    output[t + 1:b, r] = colors["right"]
    return output


def _on_gate(row, col, bounds, gate):
    t, b, l, r = bounds
    return {"top": row == t, "bottom": row == b,
            "left": col == l, "right": col == r}[gate]


def _trace_mask(info):
    """Stepwise reflection, used only by the public input-only solver."""
    t, b, l, r = info["bounds"]
    mask = np.zeros(info["shape"], dtype=bool)
    for sy, sx in info["seeds"]:
        mask[sy, sx] = True
        for dy, dx in DIRECTIONS:
            row, col = sy, sx
            # Reaching a chosen side takes at most two traversals of its axis.
            for _ in range(2 * max(b - t, r - l) + 1):
                row, col = row + dy, col + dx
                if t < row < b and l < col < r:
                    mask[row, col] = True
                if _on_gate(row, col, info["bounds"], info["gate"]):
                    break
                if row in (t, b):
                    dy = -dy
                if col in (l, r):
                    dx = -dx
            else:
                raise AssertionError("a diagonal ray did not reach its absorbing wall")
    return mask


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the rectangle and matching gate, then trace four rays per seed."""
    grid = _grid(input_grid)
    info = _analyze(grid)
    output = _frame(grid, info)
    output[_trace_mask(info)] = info["beam"]
    return output


def _fold(coordinates, extent):
    return extent - np.abs(coordinates % (2 * extent) - extent)


def _unfolded_mask(info, gate=None, seeds=None, contact_limit=None):
    """Analytic construction: straight rays in repeated mirrored rectangles."""
    gate = info["gate"] if gate is None else gate
    seeds = info["seeds"] if seeds is None else seeds
    t, b, l, r = info["bounds"]
    height, width = b - t, r - l
    mask = np.zeros(info["shape"], dtype=bool)
    for sy, sx in seeds:
        for dy, dx in DIRECTIONS:
            if gate == "top":
                length = sy - t if dy == -1 else b - sy + height
            elif gate == "bottom":
                length = b - sy if dy == 1 else sy - t + height
            elif gate == "left":
                length = sx - l if dx == -1 else r - sx + width
            else:
                length = r - sx if dx == 1 else sx - l + width
            steps = np.arange(length + 1, dtype=int)
            rows = t + _fold(sy - t + dy * steps, height)
            cols = l + _fold(sx - l + dx * steps, width)
            if contact_limit is not None:
                contacts = np.flatnonzero((rows == t) | (rows == b) | (cols == l) | (cols == r))
                if len(contacts) >= contact_limit:
                    end = int(contacts[contact_limit - 1]) + 1
                    rows, cols = rows[:end], cols[:end]
            inside = (rows > t) & (rows < b) & (cols > l) & (cols < r)
            mask[rows[inside], cols[inside]] = True
    return mask


def _straight_mask(info):
    """Counterfactual: draw each X only as far as its first wall."""
    t, b, l, r = info["bounds"]
    rows, cols = np.indices(info["shape"])
    mask = np.zeros(info["shape"], dtype=bool)
    for sy, sx in info["seeds"]:
        mask |= np.abs(rows - sy) == np.abs(cols - sx)
    return mask & (rows > t) & (rows < b) & (cols > l) & (cols < r)


def _evidence(info):
    mask = _unfolded_mask(info)
    straight = _straight_mask(info)
    alternatives = [_unfolded_mask(info, gate=gate) for gate in SIDES]
    one_bounce = _unfolded_mask(info, contact_limit=2)
    # A fully reflecting room contains the paths for every choice of exit.
    # Requiring extra cells in that union makes the absorbing gate observable.
    exit_union = np.logical_or.reduce(alternatives)
    per_seed = [_unfolded_mask(info, seeds=(seed,)) for seed in info["seeds"]]
    exclusive = []
    source_cells = np.zeros(info["shape"], dtype=bool)
    for p in info["seeds"]:
        source_cells[p] = True
    for index, one in enumerate(per_seed):
        others = np.zeros(info["shape"], dtype=bool)
        for j, other in enumerate(per_seed):
            if j != index:
                others |= other
        exclusive.append(int(np.count_nonzero(one & ~others & ~source_cells)))
    t, b, l, r = info["bounds"]
    interior_area = (b - t - 1) * (r - l - 1)
    return {"mask": mask, "reflection_gain": int(np.count_nonzero(mask & ~straight)),
            "gate_gain": int(np.count_nonzero(exit_union & ~mask)),
            "multibounce_gain": int(np.count_nonzero(mask & ~one_bounce)),
            "gate_disagreements": tuple(int(np.count_nonzero(mask ^ a)) for a in alternatives),
            "exclusive": exclusive,
            "blank": interior_area - int(np.count_nonzero(mask)),
            "area": interior_area}


def _teaching_evidence(infos):
    evidence = [_evidence(info) for info in infos]
    return (len({info["beam"] for info in infos}) >= 2
            and max(e["multibounce_gain"] for e in evidence) >= 3
            and all(any(e["gate_disagreements"][i] for e in evidence) for i in range(4)))


def _shape(rng, aspect, count):
    low = 10 if count == 3 else 8
    if aspect == "square":
        side = int(rng.integers(low, 25))
        return side, side
    if aspect in ("wide", "tall"):
        short = int(rng.integers(low, 22))
        long = int(rng.integers(short + 3, 25))
        return (short, long) if aspect == "wide" else (long, short)
    return tuple(map(int, rng.integers(low, 25, size=2)))


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _sample_pair(rng, count, gate, aspect, fixed_background, seen):
    for _ in range(700):
        rh, rw = _shape(rng, aspect, count)
        mt = int(rng.integers(0, min(4, 30 - rh) + 1))
        mb = int(rng.integers(0, min(4, 30 - rh - mt) + 1))
        ml = int(rng.integers(0, min(4, 30 - rw) + 1))
        mr = int(rng.integers(0, min(4, 30 - rw - ml) + 1))
        t, b, l, r = mt, mt + rh - 1, ml, ml + rw - 1
        palette = [int(x) for x in rng.permutation(10)]
        background = palette[0] if fixed_background is None else fixed_background
        colors = [c for c in palette if c != background][:4]
        beam = colors[SIDES.index(gate)]
        wall_positions = ((t, int(rng.integers(l + 1, r))),
                          (b, int(rng.integers(l + 1, r))),
                          (int(rng.integers(t + 1, b)), l),
                          (int(rng.integers(t + 1, b)), r))
        available = [(y, x) for y in range(t + 2, b - 1)
                     for x in range(l + 2, r - 1)]
        rng.shuffle(available)
        positions = list(wall_positions)
        if not _isolated(positions):
            continue
        seeds = []
        for candidate in available:
            if _isolated(positions + [candidate]):
                positions.append(candidate)
                seeds.append(candidate)
                if len(seeds) == count:
                    break
        if len(seeds) != count:
            continue
        grid = np.full((rh + mt + mb, rw + ml + mr), background, dtype=np.int8)
        for position, color in zip(wall_positions, colors):
            grid[position] = color
        for position in seeds:
            grid[position] = beam
        if _key(grid) in seen:
            continue
        info = _analyze(grid)
        evidence = _evidence(info)
        # A square is a useful symmetry case: its full four-ray union is
        # independent of the chosen exit. Core training uses nonsquares.
        if (evidence["reflection_gain"] < 3
                or (rh != rw and evidence["gate_gain"] < 3)
                or min(evidence["exclusive"]) < 2
                or evidence["blank"] < max(5, evidence["area"] // 5)):
            continue
        output = _frame(grid, info)
        output[evidence["mask"]] = beam
        return {"input": grid, "output": output}
    raise RuntimeError("could not sample a sufficiently informative ray room")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    # This affects sampling, never decoding: the same rule covers both shifts.
    shift = "extra_sources" if int(rng.integers(2)) == 0 else "aspect_swap"
    train_aspect = ("wide", "tall")[int(rng.integers(2))]
    test_aspect = "tall" if train_aspect == "wide" else "wide"
    fixed_background = int(rng.integers(10)) if rng.random() < .3 else None
    first_gate = SIDES[int(rng.integers(4))]
    perpendicular = ("left", "right") if first_gate in ("top", "bottom") else ("top", "bottom")
    second_gate = perpendicular[int(rng.integers(2))]
    plans = [(1, first_gate, True), (2, second_gate, True)]
    max_train_count = 2 if shift == "extra_sources" else 3
    for _ in range(num_train - 2):
        plans.append((int(rng.integers(1, max_train_count + 1)), SIDES[int(rng.integers(4))], False))
    rng.shuffle(plans)
    # Occasionally a valid set still permits a constant exit or a one-bounce
    # shortcut. Resample the teaching set rather than introducing a rule exception.
    for _ in range(64):
        seen = set()
        puzzle = {"train": [], "test": []}
        for count, gate, core in plans:
            aspect = train_aspect if shift == "aspect_swap" else (
                ("wide", "tall")[int(rng.integers(2))] if core else
                ("square", "wide", "tall", "free")[int(rng.integers(4))])
            pair = _sample_pair(rng, count, gate, aspect, fixed_background, seen)
            puzzle["train"].append(pair)
            seen.add(_key(pair["input"]))
        if _teaching_evidence([_analyze(pair["input"]) for pair in puzzle["train"]]):
            break
    else:
        raise RuntimeError("could not construct an informative teaching set")
    for _ in range(num_test):
        count = 3 if shift == "extra_sources" else int(rng.integers(1, 4))
        aspect = test_aspect if shift == "aspect_swap" else (
            "square", "wide", "tall", "free")[int(rng.integers(4))]
        gate = SIDES[int(rng.integers(4))]
        pair = _sample_pair(rng, count, gate, aspect, fixed_background, seen)
        puzzle["test"].append(pair)
        seen.add(_key(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check returned data, semantic invariants, evidence, and the split shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            evidence = _evidence(info)
            expected = _frame(inp, info)
            expected[evidence["mask"]] = info["beam"]
            assert np.array_equal(expected, out), "unfolded ray rule mismatch"
            assert np.array_equal(inp[inp != info["background"]],
                                  out[inp != info["background"]]), "input marker changed"
            assert set(map(int, np.unique(inp))) == set(map(int, np.unique(out)))
            t, b, l, r = info["bounds"]
            rows, cols = np.indices(inp.shape)
            outside = (rows < t) | (rows > b) | (cols < l) | (cols > r)
            assert np.array_equal(inp[outside], out[outside])
            interior = out[t + 1:b, l + 1:r]
            assert set(map(int, np.unique(interior))) == {info["background"], info["beam"]}
            assert evidence["reflection_gain"] >= 3, "reflections are not visible"
            assert info["size"][0] == info["size"][1] or evidence["gate_gain"] >= 3, "gate absorption is not visible"
            assert min(evidence["exclusive"]) >= 2, "a seed is redundant"
            assert evidence["blank"] >= max(5, evidence["area"] // 5)
            infos[split].append(info)
    train = infos["train"]
    assert {1, 2}.issubset({len(info["seeds"]) for info in train})
    gates = {info["gate"] for info in train}
    assert gates.intersection(("top", "bottom")) and gates.intersection(("left", "right"))
    informative_gates = {info["gate"] for info in train if _evidence(info)["gate_gain"] >= 3}
    assert informative_gates.intersection(("top", "bottom")) and informative_gates.intersection(("left", "right"))
    assert _teaching_evidence(train), "training permits a fixed exit, fixed beam color, or one-bounce shortcut"
    extra_sources = (max(len(a["seeds"]) for a in train) == 2
                     and all(len(a["seeds"]) == 3 for a in infos["test"]))
    train_aspects = {np.sign(a["size"][0] - a["size"][1]) for a in train}
    test_aspects = {np.sign(a["size"][0] - a["size"][1]) for a in infos["test"]}
    aspect_swap = len(train_aspects) == len(test_aspects) == 1 and train_aspects == {-v for v in test_aspects} and 0 not in train_aspects
    assert extra_sources or aspect_swap, "missing structural train/test shift"


def _puzzle_key(puzzle):
    return tuple(tuple((_key(pair["input"]), _key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def self_test():
    """Bounded 42-seed variation audit plus exact-count and solver checks."""
    signatures, shapes, layouts = set(), set(), set()
    wall_colors, beams, backgrounds = set(), set(), set()
    source_counts = {"train": set(), "test": set()}
    gates = {"train": set(), "test": set()}
    aspects = {"train": set(), "test": set()}
    edge = square = even = overlap = extra_shifts = aspect_shifts = 0
    minimum_reflection_gain = minimum_gate_gain = 10**6
    minimum_teaching_multibounce_gain = 10**6
    gate_sensitive_pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        signatures.add(_puzzle_key(puzzle))
        train_infos = [_analyze(pair["input"]) for pair in puzzle["train"]]
        test_infos = [_analyze(pair["input"]) for pair in puzzle["test"]]
        minimum_teaching_multibounce_gain = min(minimum_teaching_multibounce_gain,
                                              max(_evidence(a)["multibounce_gain"] for a in train_infos))
        extra_shifts += (max(len(a["seeds"]) for a in train_infos) == 2
                         and all(len(a["seeds"]) == 3 for a in test_infos))
        aspect_shifts += (len({np.sign(a["size"][0] - a["size"][1]) for a in train_infos + test_infos}) > 1
                          and len({np.sign(a["size"][0] - a["size"][1]) for a in train_infos}) == 1)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                evidence = _evidence(info)
                minimum_reflection_gain = min(minimum_reflection_gain, evidence["reflection_gain"])
                if evidence["gate_gain"]:
                    minimum_gate_gain = min(minimum_gate_gain, evidence["gate_gain"])
                    gate_sensitive_pairs += 1
                shapes.add(info["shape"])
                layouts.add((info["shape"], info["bounds"], tuple(info["walls"].values()), info["seeds"]))
                wall_colors.update(info["colors"].values())
                beams.add(info["beam"])
                backgrounds.add(info["background"])
                source_counts[split].add(len(info["seeds"]))
                gates[split].add(info["gate"])
                h, w = info["size"]
                aspects[split].add(int(np.sign(h - w)))
                square += h == w
                even += h % 2 == 0 or w % 2 == 0
                t, b, l, r = info["bounds"]
                gh, gw = info["shape"]
                edge += t == 0 or l == 0 or b == gh - 1 or r == gw - 1
                per_seed = [_unfolded_mask(info, seeds=(s,)) for s in info["seeds"]]
                overlap += any(np.any(a & b) for a, b in combinations(per_seed, 2))
    assert len(signatures) == 42
    assert len(shapes) >= 70 and len(layouts) >= 200
    assert len(wall_colors) == len(beams) == len(backgrounds) == 10
    assert all(source_counts[s] == {1, 2, 3} for s in source_counts)
    assert all(gates[s] == set(SIDES) for s in gates)
    assert all(aspects[s] == {-1, 0, 1} for s in aspects)
    assert min(edge, square, even, overlap, extra_shifts, aspect_shifts) > 0
    # Every legal explicit count is exercised; illegal types and counts fail.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * nt + ns, nt, ns)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Corner and coincident-path behavior: formula vs step tracing, including
    # cases that the informative-example rejection sampler need not retain.
    fixture = {"shape": (13, 15), "bounds": (1, 11, 2, 12), "seeds": ((4, 5),)}
    for gate in SIDES:
        fixture["gate"] = gate
        assert np.array_equal(_trace_mask(fixture), _unfolded_mask(fixture))
    return {"seeds": 42, "pairs": 210, "distinct_shapes": len(shapes),
            "distinct_layouts": len(layouts), "minimum_reflection_gain": minimum_reflection_gain,
            "minimum_gate_gain": minimum_gate_gain, "edge_pairs": edge,
            "gate_sensitive_pairs": gate_sensitive_pairs,
            "minimum_teaching_multibounce_gain": minimum_teaching_multibounce_gain,
            "overlapping_seed_pairs": overlap, "extra_source_shifts": int(extra_shifts),
            "aspect_shift_candidates": int(aspect_shifts)}


if __name__ == "__main__":
    print(self_test())
