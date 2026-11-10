"""Zipper spools: open frames feed whole cyclic words into a shared staircase.

Only NumPy and the standard library are used. Parsing and solving read pixels;
no generation records, seeds, examples, or parent modules enter solve().
"""

import hashlib
import itertools
import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 218978431, 786375531, 556614469, 902028510, 257096371,
              125050323, 973020228, 452290075, 300747273, 543998609,
              363212678, 203033935, 568895200, 924237592, 574997459,
              433785546, 422499259, 325685293, 919275748, 80849959,
              60124863, 93599518, 271837408, 918797550, 21111027,
              850225495, 304747248, 305435044, 229029404, 757614214,
              954859027, 243694786, 759131124, 840507726, 652189450,
              831011164, 726499714, 60685021, 103446022, 114386826, 914139140)
SHORTCUTS = ("unit_chunks", "reset_phase", "long_word_leads",
             "short_word_leads", "row_major_leads", "reverse_word",
             "primitive_chunks", "prefix_cycle", "straight")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= x <= 30 for x in g.shape), "grid size")
    _require(np.issubdtype(g.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= g) & (g <= 9))), "color outside 0..9")
    return g.astype(np.int8, copy=False)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _add(p, d, n=1):
    return p[0] + n * d[0], p[1] + n * d[1]


def _key(g):
    return g.shape, np.ascontiguousarray(g, dtype=np.int8).tobytes()


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        cells, stack = {first}, [first]
        while stack:
            p = stack.pop()
            for d in DIRECTIONS:
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _geometry(gap, direction, length, thickness):
    """The gap is the OUTER mouth; the word is read from the closed end."""
    t = thickness
    side = (-direction[1], direction[0])
    footprint = {_add(_add(gap, direction, s), side, z)
                 for s in range(-(length + 2*t - 1), 1)
                 for z in range(-t, t+1)}
    word_cells = [_add(gap, direction, -(t + length - 1) + j)
                  for j in range(length)]
    tunnel = [_add(gap, direction, s) for s in range(1-t, 1)]
    frame = footprint - set(word_cells) - set(tunnel)
    return footprint, frame, word_cells, tunnel


def _parse(input_grid):
    g = _grid(input_grid)
    counts = np.bincount(g.ravel(), minlength=10)
    background = int(np.argmax(counts))
    _require(int(counts[background]) > g.size // 2, "background must exceed half the input")
    foreground = set(zip(*np.nonzero(g != background)))
    interpretations = []
    for structure in sorted(int(c) for c in np.unique(g) if c != background):
        spools, claimed = [], set()
        for component in _components(g == structure):
            rows, cols = zip(*component)
            top, bottom, left, right = min(rows), max(rows), min(cols), max(cols)
            h, w = bottom-top+1, right-left+1
            choices = []
            for t in (1, 2):
                for d in DIRECTIONS:
                    longitudinal, transverse = (h, w) if d[0] else (w, h)
                    length = longitudinal - 2*t
                    if transverse != 2*t+1 or not 1 <= length <= 5:
                        continue
                    gap = ((top, (left+right)//2) if d == (-1, 0) else
                           (bottom, (left+right)//2) if d == (1, 0) else
                           ((top+bottom)//2, left) if d == (0, -1) else
                           ((top+bottom)//2, right))
                    footprint, frame, word_cells, tunnel = _geometry(gap, d, length, t)
                    if frame != component or not all(_inside(p, g.shape) for p in footprint):
                        continue
                    if any(int(g[p]) != background for p in tunnel):
                        continue
                    word = tuple(int(g[p]) for p in word_cells)
                    if any(c in (background, structure) for c in word):
                        continue
                    choices.append(dict(gap=gap, direction=d, word=word, thickness=t,
                                        footprint=footprint, frame=frame,
                                        word_cells=word_cells, tunnel=tunnel))
            if len(choices) != 1:
                break
            spool = choices[0]
            used = spool["frame"] | set(spool["word_cells"])
            if claimed & used:
                break
            claimed |= used
            spools.append(spool)
        else:
            if 2 <= len(spools) <= 4 and claimed == foreground:
                spools.sort(key=lambda s: s["gap"])
                footprints = [s["footprint"] for s in spools]
                if any(a & b for a, b in itertools.combinations(footprints, 2)):
                    continue
                interpretations.append((background, structure, spools))
    _require(len(interpretations) == 1, "missing or ambiguous open frames")
    background, structure, spools = interpretations[0]
    return g, background, structure, spools


def _ray(spool, shape):
    points = []
    p = spool["gap"]
    while _inside(p, shape):
        points.append(p)
        p = _add(p, spool["direction"])
    return points


def _encounter(spools, rays):
    events = []
    for i, j in itertools.combinations(range(len(spools)), 2):
        common = set(rays[i]) & set(rays[j])
        if not common:
            continue
        u, v = spools[i]["direction"], spools[j]["direction"]
        _require(u[0]*v[0] + u[1]*v[1] == 0 and len(common) == 1,
                 "only perpendicular single-cell encounters are supported")
        cell = next(iter(common))
        a, b = rays[i].index(cell), rays[j].index(cell)
        _require(a > 0 and b > 0 and a != b, "encounter needs unequal positive approaches")
        events.append((i, j, cell, a, b))
    _require(len(events) == 1, "exactly one spatial encounter is required")
    return events[0]


def _period(word, whole=True):
    return next(p for p in range(1, len(word)+1)
                if (not whole or len(word) % p == 0)
                and all(c == word[k % p] for k, c in enumerate(word)))


def _render(input_grid, variant="normal"):
    g, background, structure, spools = _parse(input_grid)
    rays = [_ray(s, g.shape) for s in spools]
    i, j, cell, a, b = _encounter(spools, rays)
    lengths = [len(s["word"]) for s in spools]
    words = [s["word"] for s in spools]
    chunks = lengths.copy()
    if variant == "reverse_word":
        words = [w[::-1] for w in words]
    elif variant == "primitive_chunks":
        chunks = [_period(w) for w in words]
    elif variant == "prefix_cycle":
        words = [w[:_period(w, False)] for w in words]
        chunks = [len(w) for w in words]
    elif variant == "unit_chunks":
        chunks = [1] * len(spools)
    output = g.copy()
    writes = {}

    def put(p, owner, index, permissive=False):
        if int(g[p]) != background:
            if permissive:
                return
            raise ValueError("thread hits a frame or its stored word")
        if p in writes and not permissive:
            raise ValueError("thread overlap")
        color = words[owner][index % len(words[owner])]
        writes[p] = (owner, index, color)
        output[p] = color

    # The opening is a tunnel for thick frames. Its cells consume symbols too.
    for n, s in enumerate(spools):
        for k, p in enumerate(s["tunnel"][:-1]):
            put(p, n, k)
    if variant == "straight":
        for n, (s, points) in enumerate(zip(spools, rays)):
            for k, p in enumerate(points):
                put(p, n, s["thickness"]-1+k, permissive=True)
        return output, dict(spools=spools, background=background, structure=structure,
                            encounter=(i, j, cell, a, b), writes=writes)

    for n, (s, points) in enumerate(zip(spools, rays)):
        stop = a if n == i else b if n == j else len(points)
        for k, p in enumerate(points[:stop]):
            put(p, n, s["thickness"]-1+k)
    leader, follower = (i, j) if a < b else (j, i)
    if variant == "long_word_leads" and lengths[i] != lengths[j]:
        leader = max((i, j), key=lambda n: lengths[n])
    elif variant == "short_word_leads" and lengths[i] != lengths[j]:
        leader = min((i, j), key=lambda n: lengths[n])
    elif variant == "row_major_leads":
        leader = min((i, j), key=lambda n: spools[n]["gap"])
    follower = j if leader == i else i
    counters = {i: spools[i]["thickness"]-1+a,
                j: spools[j]["thickness"]-1+b}
    if variant == "reset_phase":
        counters = {i: 0, j: 0}
    start_phases = {n: counters[n] % lengths[n] for n in (i, j)}
    p, first, zipper = cell, True, []
    # Both headings are perpendicular cardinal vectors, so the staircase is
    # monotone and exits within h+w cells. No saved execution trace is needed.
    for turn in range(2 * sum(g.shape) + 2):
        owner = (leader, follower)[turn % 2]
        for _ in range(chunks[owner]):
            if first:
                first = False
            else:
                p = _add(p, spools[owner]["direction"])
            if not _inside(p, g.shape):
                return output, dict(spools=spools, background=background, structure=structure,
                                    encounter=(i, j, cell, a, b), leader=leader,
                                    start_phases=start_phases, writes=writes, zipper=zipper,
                                    clipped_turn=(owner, counters[owner] % lengths[owner]))
            put(p, owner, counters[owner])
            zipper.append((p, owner, counters[owner]))
            counters[owner] += 1
    raise ValueError("zipper did not exit")


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover frames, words, directions, meeting, leader, phases and zipper."""
    return _render(input_grid)[0]


def _make_input(shape, specs, background, structure):
    g = np.full(shape, background, dtype=np.int8)
    occupied = set()
    for s in specs:
        footprint, frame, cells, tunnel = _geometry(
            s["gap"], s["direction"], len(s["word"]), s["thickness"])
        _require(all(_inside(p, shape) for p in footprint), "frame outside canvas")
        collar = {_add(p, d) for p in footprint
                  for d in itertools.product((-1, 0, 1), repeat=2)}
        _require(not occupied & collar, "frames need an empty separating collar")
        occupied |= footprint
        for p in frame:
            g[p] = structure
        for p, c in zip(cells, s["word"]):
            g[p] = c
    return g


def _random_word(rng, length, palette):
    if length == 1:
        return (int(rng.choice(palette)),)
    a, b = (int(x) for x in rng.choice(palette, 2, replace=False))
    style = int(rng.integers(4))
    if style == 0:
        return tuple((a, b)[k % 2] for k in range(length))
    if style == 1:
        return (a,) * (length-1) + (b,)
    if style == 2:
        return (a, b) + (a,) * (length-2)
    values = [int(x) for x in rng.choice(palette, length)]
    values[0], values[-1] = a, b
    return tuple(values)


def _try_extra(rng, shape, specs, background, structure, palette, limit):
    for _ in range(60):
        t = int(rng.choice((1, 1, 2)))
        length = int(rng.integers(1, limit+1))
        d = DIRECTIONS[int(rng.integers(4))]
        gap = (int(rng.integers(shape[0])), int(rng.integers(shape[1])))
        if rng.random() < .35:
            # Short boundary emissions are useful negative interaction controls.
            gap = ((0 if d[0] == -1 else shape[0]-1, gap[1]) if d[0] else
                   (gap[0], 0 if d[1] == -1 else shape[1]-1))
        word = _random_word(rng, length, palette)
        extra = dict(gap=gap, direction=d, word=word, thickness=t)
        try:
            candidate = _make_input(shape, specs+[extra], background, structure)
            _render(candidate)
            return specs+[extra]
        except ValueError:
            continue
    if len(specs) == 2:
        # Canonical main scenes leave the southwest corner free. This short
        # outward control guarantees that the second anchor teaches isolation,
        # even if randomized placement used all of its bounded attempts.
        extra = dict(gap=(shape[0]-1, 1), direction=(1, 0),
                     word=(int(rng.choice(palette)),), thickness=1)
        try:
            _render(_make_input(shape, specs+[extra], background, structure))
            return specs+[extra]
        except ValueError:
            pass
    return specs


def _make_pair(rng, kind, orientation):
    colors = [int(c) for c in rng.permutation(10)]
    background, structure = colors[:2]
    # Anchor spools have disjoint palettes so that leader/phase mistakes cannot
    # disappear merely because two suppliers happened to write the same color.
    a0, a1, b0, b1 = colors[2:6]
    if kind == "whole_word":
        wa, wb = (a0, a1, a0), (b0, b1)
    elif kind == "stored_length":
        wa, wb = (a0, a1, a0, a1), (b0, b1)
    else:
        la, lb = (int(x) for x in rng.integers(1, (6 if kind == "test" else 5), 2))
        if kind == "test":
            if rng.integers(2):
                la = 5
            else:
                lb = 5
        # Random examples can share motif colors, unlike the teaching anchors.
        palette = colors[2:int(rng.integers(5, 9))]
        wa, wb = _random_word(rng, la, palette), _random_word(rng, lb, palette)
    for _ in range(80):
        ta, tb = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        if kind == "whole_word":
            tb = 3-ta  # Show both tunnel thicknesses even with only two demos.
            choices = [v for v in range(4, 8) if (ta-1+v) % 3 == 1]
            da, db = int(rng.choice(choices)), int(rng.integers(8, 11))
        elif kind == "stored_length":
            choices = [v for v in range(4, 7) if (tb-1+v) % 2 == 1]
            da, db = int(rng.integers(8, 11)), int(rng.choice(choices))
        else:
            da, db = (int(x) for x in rng.integers(4, 11, 2))
            if da == db:
                continue
        row = int(rng.integers(7, 11))
        col = da + len(wa) + 2*ta-1 + int(rng.integers(4))
        h = row + db + len(wb) + 2*tb + int(rng.integers(4))
        w = col + int(rng.integers(7, 12))
        if max(h, w) > 30:
            continue
        specs = [dict(gap=(row, col-da), direction=(0, 1), word=wa, thickness=ta),
                 dict(gap=(row+db, col), direction=(-1, 0), word=wb, thickness=tb)]
        try:
            g = _make_input((h, w), specs, background, structure)
            _render(g)
        except ValueError:
            continue
        target = (0 if kind == "whole_word" else
                  int(rng.integers(1, 3)) if kind == "stored_length" else
                  int(rng.integers(3)))
        for _ in range(target):
            specs = _try_extra(rng, (h, w), specs, background, structure,
                               colors[2:int(rng.integers(5, 9))],
                               5 if kind == "test" else 4)
        g = _make_input((h, w), specs, background, structure)
        rotation, reflection = orientation
        g = np.rot90(g, rotation)
        if reflection:
            g = np.fliplr(g)
        g = np.ascontiguousarray(g)
        return {"input": g, "output": solve(g)}
    # A guaranteed main geometry, with no extra emitters, bounds packing cost.
    ta, tb = (1, 2) if kind == "whole_word" else (1, 1)
    da, db = ((4, 9) if kind != "stored_length" else (9, 5))
    row, col = 9, da+len(wa)+2*ta+1
    shape = (row+db+len(wb)+2*tb+1, col+9)
    specs = [dict(gap=(row, col-da), direction=(0, 1), word=wa, thickness=ta),
             dict(gap=(row+db, col), direction=(-1, 0), word=wb, thickness=tb)]
    if kind == "stored_length":
        specs.append(dict(gap=(shape[0]-1, 1), direction=(1, 0),
                          word=(a0,), thickness=1))
    g = _make_input(shape, specs, background, structure)
    g = np.rot90(g, orientation[0])
    if orientation[1]:
        g = np.fliplr(g)
    g = np.ascontiguousarray(g)
    return {"input": g, "output": solve(g)}


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " outside legal range")
    return int(value)


def _unique(pair, used):
    if _key(pair["input"]) not in used:
        return pair
    colors = [int(c) for c in np.unique(pair["input"])]
    # At most 13 prior inputs. The first 14 injective remappings suffice, since
    # every remapping produces a different visible input. No unbounded retries.
    for values in itertools.islice(itertools.permutations(range(10), len(colors)), 14):
        lookup = np.arange(10, dtype=np.int8)
        lookup[colors] = values
        candidate = {k: lookup[g] for k, g in pair.items()}
        if _key(candidate["input"]) not in used:
            return candidate
    raise AssertionError("finite uniqueness repair exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    orientation = (int(rng.integers(4)), bool(rng.integers(2)))
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for k in range(count):
            kind = ("whole_word" if k == 0 else "stored_length" if k == 1 else "extra") if split == "train" else "test"
            transform = (orientation if split == "train" and k < 2 else
                         ((orientation[0]+int(rng.integers(1, 4))) % 4,
                          bool(rng.integers(2))))
            pair = _unique(_make_pair(rng, kind, transform), used)
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _variant_matches(pair, variant):
    try:
        return np.array_equal(_render(pair["input"], variant)[0], pair["output"])
    except ValueError:
        return False


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split must be a list")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "pair counts")
    used, eliminated = set(), set()
    training_lengths, test_lengths, training_thicknesses = set(), set(), set()
    train_isolation, test_isolation = False, False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            g, out = _grid(pair["input"]), _grid(pair["output"])
            _require(g.shape == out.shape, "shape changed")
            key = _key(g)
            _require(key not in used, "duplicate input")
            used.add(key)
            expected, info = _render(g)
            _require(np.array_equal(expected, out), "wrong word, phase or zipper route")
            bg = info["background"]
            _require(np.array_equal(g[g != bg], out[g != bg]), "stored word or frame changed")
            _require(bool(np.any(g != out)), "identity example")
            i, j, cell, a, b = info["encounter"]
            _require(a != b and len(info["zipper"]) >= 4, "ambiguous or invisible splice")
            # One source symbol per new cell, including the cells inside thick
            # openings; source counters increase only when that source is used.
            by_owner = [[] for _ in info["spools"]]
            for p, (owner, index, color) in info["writes"].items():
                word = info["spools"][owner]["word"]
                _require(color == word[index % len(word)] and int(out[p]) == color,
                         "word conservation")
                by_owner[owner].append(index)
            for indices in by_owner:
                _require(sorted(indices) == list(range(len(indices))), "lost or reset symbol clock")
            lengths = {len(s["word"]) for s in info["spools"]}
            if split == "train":
                training_lengths |= lengths
                training_thicknesses.update(s["thickness"] for s in info["spools"])
                train_isolation |= len(info["spools"]) > 2
                for variant in SHORTCUTS:
                    if not _variant_matches(pair, variant):
                        eliminated.add(variant)
            else:
                test_lengths |= lengths
                test_isolation |= len(info["spools"]) > 2
    _require(set(SHORTCUTS) <= eliminated, "demonstrations leave a named shortcut unresolved")
    _require({2, 3, 4} <= training_lengths, "demonstrate whole words and stored lengths")
    _require(training_thicknesses == {1, 2}, "demonstrate both tunnel thicknesses")
    _require(not test_isolation or train_isolation, "teach isolated emission before testing it")
    _require(5 in test_lengths and 5 not in training_lengths, "missing word-length test shift")
    return None


def family_features(pair):
    """Small descriptive record reconstructed from the actual returned input."""
    out, info = _render(pair["input"])
    i, j, cell, a, b = info["encounter"]
    spools = info["spools"]
    return dict(shape=list(out.shape), spools=len(spools),
                lengths=sorted(len(s["word"]) for s in spools),
                thicknesses=sorted({s["thickness"] for s in spools}),
                directions=sorted({s["direction"] for s in spools}),
                approaches=sorted((a, b)), meeting=list(cell),
                leading_length=len(spools[info["leader"]]["word"]),
                phases=sorted(info["start_phases"].values()),
                zipper_cells=len(info["zipper"]), changed=int(np.count_nonzero(out != pair["input"])),
                background=info["background"], structure=info["structure"],
                shared_colors=bool(set(spools[i]["word"]) & set(spools[j]["word"])))


def _fixture():
    # E word ABA, distance 4; N word CD, distance 8. At the meeting the E
    # clock is 4, hence B leads. Its chunk is BAA, followed by CD northward.
    specs = [dict(gap=(8, 6), direction=(0, 1), word=(1, 2, 1), thickness=1),
             dict(gap=(16, 10), direction=(-1, 0), word=(3, 4), thickness=1)]
    g = _make_input((22, 22), specs, 0, 8)
    expected = g.copy()
    for c, v in zip(range(6, 10), (1, 2, 1, 1)):
        expected[8, c] = v
    for r, v in zip(range(16, 8, -1), (3, 4, 3, 4, 3, 4, 3, 4)):
        expected[r, 10] = v
    # Explicit digit-path expectation, independent of the routing loop.
    path = ((8, 10, 2), (8, 11, 1), (8, 12, 1), (7, 12, 3), (6, 12, 4),
            (6, 13, 2), (6, 14, 1), (6, 15, 1), (5, 15, 3), (4, 15, 4),
            (4, 16, 2), (4, 17, 1), (4, 18, 1), (3, 18, 3), (2, 18, 4),
            (2, 19, 2), (2, 20, 1), (2, 21, 1), (1, 21, 3), (0, 21, 4))
    for r, c, v in path:
        expected[r, c] = v
    return g, expected


def self_test():
    g, expected = _fixture()
    assert np.array_equal(solve(g), expected), "explicit zipper/phase fixture"
    # The rule commutes with all rotations/reflections and injective recoloring.
    for rotation in range(4):
        for reflection in (False, True):
            a, b = np.rot90(g, rotation), np.rot90(expected, rotation)
            if reflection:
                a, b = np.fliplr(a), np.fliplr(b)
            assert np.array_equal(solve(a), b)
    lookup = np.array([9, 6, 7, 5, 2, 0, 4, 8, 1, 3], dtype=np.int8)
    assert np.array_equal(solve(lookup[g]), lookup[expected])
    thick = _make_input((22, 22), [
        dict(gap=(8, 6), direction=(0, 1), word=(1, 2, 1), thickness=2),
        dict(gap=(16, 10), direction=(-1, 0), word=(3, 4), thickness=2)], 0, 8)
    thick_out = solve(thick)
    for p, color in (((8, 5), 1), ((8, 6), 2), ((17, 10), 3), ((16, 10), 4),
                     ((8, 10), 1), ((8, 11), 1), ((8, 12), 2),
                     ((7, 12), 4), ((6, 12), 3)):
        assert int(thick_out[p]) == color, "thick tunnel consumes symbols"
    assert _period((1, 2, 1)) == 3 and _period((1, 2, 1, 2)) == 2
    shapes, backgrounds, structures, counts, headings, layouts = set(), set(), set(), set(), set(), set()
    thicknesses, lengths, phases, approaches, leading_lengths = set(), set(), set(), set(), set()
    signatures, shared, clipped = set(), 0, set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = family_features(pair)
                shapes.add(tuple(f["shape"]))
                backgrounds.add(f["background"])
                structures.add(f["structure"])
                counts.add(f["spools"])
                headings.update(tuple(d) for d in f["directions"])
                thicknesses.update(f["thicknesses"])
                lengths.update(f["lengths"])
                phases.update(f["phases"])
                approaches.add(tuple(f["approaches"]))
                leading_lengths.add(f["leading_length"])
                shared += f["shared_colors"]
                _, info = _render(pair["input"])
                clipped.add(info["clipped_turn"][1] == 0)
                # Ignore palette and whole-grid symmetries when counting layouts.
                forms = []
                bg, st = info["background"], info["structure"]
                normalized = np.where(pair["input"] == bg, 0,
                                      np.where(pair["input"] == st, 1, 2)).astype(np.int8)
                for k in range(4):
                    x = np.rot90(normalized, k)
                    forms.extend(((x.shape, x.tobytes()), (x.shape, np.fliplr(x).tobytes())))
                layouts.add(min(forms))
                digest.update(str(pair["input"].shape).encode())
                digest.update(pair["input"].tobytes())
                digest.update(pair["output"].tobytes())
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42
    assert len(shapes) >= 45 and len(layouts) >= 160
    assert len(backgrounds) == len(structures) == 10
    assert counts == {2, 3, 4} and headings == set(DIRECTIONS)
    assert thicknesses == {1, 2} and lengths == {1, 2, 3, 4, 5}
    assert phases >= {0, 1, 2, 3, 4} and len(approaches) >= 15
    assert leading_lengths >= {1, 2, 3, 4, 5} and shared > 20
    assert clipped == {False, True}
    for nt, ne in ((2, 1), (10, 1), (10, 4), (5, 4)):
        a = generate_puzzle(9187, nt, ne)
        b = generate_puzzle(9187, nt, ne)
        assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
                   for k in ("input", "output"))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupt = generate_puzzle(0, 2, 1)
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes),
            "layouts_ignoring_palette_and_symmetry": len(layouts),
            "spool_counts": sorted(counts), "word_lengths": sorted(lengths),
            "approach_pairs": len(approaches), "shared_palette_pairs": shared,
            "semantic_verification": "author-owned fixtures, invariants and shortcut checks"}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
