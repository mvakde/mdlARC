"""Clockwork stencils: spatial phase sampling supplies shape-conditioned votes.

Only standard-library code and NumPy are used. The solver reconstructs the
emitter, whole signal word, sensor objects, and ballots from the input alone.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import itertools

import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _norm(cells):
    cells = tuple(cells)
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


@lru_cache(maxsize=4096)
def _variants(shape, reflections=True):
    answers = set()
    for sign in ((1, -1) if reflections else (1,)):
        current = _norm((r, sign*c) for r, c in shape)
        for _ in range(4):
            answers.add(current)
            current = _norm((c, -r) for r, c in current)
    return tuple(sorted(answers))


def _key(cells, equivalence="free"):
    s = _norm(cells)
    if equivalence == "area":
        return len(s)
    if equivalence == "box":
        return tuple(sorted(_size(s)))
    if equivalence == "translation":
        return s
    return min(_variants(s, equivalence != "rotation"))


def _size(shape):
    return max(r for r, c in shape)+1, max(c for r, c in shape)+1


def _bbox(cells):
    rs, cs = zip(*cells)
    return min(rs), min(cs), max(rs)+1, max(cs)+1


def _hole_free(shape):
    h, w = _size(shape)
    mask = np.ones((h+2, w+2), dtype=bool)
    for r, c in shape:
        mask[r+1, c+1] = False
    return len(_components(mask)) == 1


@lru_cache(maxsize=1)
def _pool():
    # Enumerate free polyominoes, rather than copying a parent's vocabulary.
    level, pool = {((0, 0),)}, set()
    for area in range(2, 8):
        next_level = set()
        for s in sorted(level):
            occupied = set(s)
            frontier = sorted({(r+dr, c+dc) for r, c in s for dr, dc in DIRS} - occupied)
            for p in frontier:
                t = _key((*s, p))
                if max(_size(t)) <= 4:
                    next_level.add(t)
        level = next_level
        if area >= 3:
            pool.update(s for s in level if _hole_free(s))
    return tuple(sorted(pool))


def _distance(point, box, direction):
    r, c = point
    t, l, b, right = box
    return (t-r, r-b+1, l-c, c-right+1)[direction]


def _seed_point(box, direction, distance):
    t, l, b, right = box
    middle_r, middle_c = (t+b-1)//2, (l+right-1)//2
    return ((t-distance, middle_c), (b-1+distance, middle_c),
            (middle_r, l-distance), (middle_r, right-1+distance))[direction]


def _inside(point, shape):
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


def _decode(input_grid):
    g = _grid(input_grid)
    candidates = []
    for color in map(int, np.unique(g)):
        for cells in _components(g == color):
            if len(cells) < 16:
                continue
            t, l, b, right = _bbox(cells)
            patch = g[t:b, l:right]
            other = set(map(int, np.unique(patch))) - {color}
            if (min(patch.shape) >= 5 and len(other) == 1
                    and np.all(patch[0] == color) and np.all(patch[-1] == color)
                    and np.all(patch[:, 0] == color) and np.all(patch[:, -1] == color)
                    and np.count_nonzero(g == color) == len(cells)):
                candidates.append((color, other.pop(), (t, l, b, right)))
    assert len(candidates) == 1, "one unique intact rectangular stencil is required"
    carrier, background, box = candidates[0]
    t, l, b, right = box
    patch = g[t:b, l:right]
    assert all(n % 2 for n in patch.shape), "odd stencil dimensions give a unique face center"
    cavities = _components(patch == background)
    assert cavities and len(_components(patch == carrier)) == 1
    assert all(3 <= len(c) <= 12 and max(_size(_norm(c))) <= 6 for c in cavities)
    faces = []
    for direction in range(4):
        if direction < 2:
            row = t-1 if direction == 0 else b
            face = [(row, c) for c in range(l, right)]
        else:
            col = l-1 if direction == 2 else right
            face = [(r, col) for r in range(t, b)]
        visible = [p for p in face if _inside(p, g.shape) and g[p] != background]
        if visible:
            assert visible == [_seed_point(box, direction, 1)], "signal must be centered and one cell wide"
            faces.append(direction)
    assert len(faces) == 1, "one emitting face is required"
    direction = faces[0]
    word, seed_cells = [], []
    distance = 1
    while _inside(_seed_point(box, direction, distance), g.shape):
        p = _seed_point(box, direction, distance)
        if g[p] == background:
            break
        word.append(int(g[p]))
        seed_cells.append(p)
        distance += 1
    assert word and _inside(_seed_point(box, direction, len(word)+1), g.shape)
    assert carrier not in word and background not in word
    exterior = g.copy()
    exterior[t:b, l:right] = background
    for p in seed_cells:
        exterior[p] = background
    tokens, roles = [], set()
    for cells in _components(exterior != background):
        counts = Counter(int(g[p]) for p in cells)
        assert 3 <= len(cells) <= 12 and len(counts) == 2, "token must have a body and one sensor"
        assert max(_size(_norm(cells))) <= 6
        sensors = [c for c, n in counts.items() if n == 1]
        assert len(sensors) == 1
        sensor_color = sensors[0]
        body_color = next(c for c in counts if c != sensor_color)
        assert not set(counts).intersection({carrier, background, *word})
        sensor = next(p for p in cells if int(g[p]) == sensor_color)
        d = _distance(sensor, box, direction)
        assert d > len(word), "sensors must lie beyond the complete signal word"
        assert all(_distance(p, box, direction) > 0 for p in cells)
        roles.add((body_color, sensor_color))
        tokens.append(dict(cells=cells, sensor=sensor, distance=d,
                           shape=_key(cells), vote=word[(d-1) % len(word)]))
    assert tokens and len(roles) == 1, "shared local body/sensor roles are required"
    # An empty cell collar makes whole objects unambiguous even across colors.
    owner = {p: i for i, token in enumerate(tokens) for p in token["cells"]}
    protected = set(seed_cells) | {(r, c) for r in range(t, b) for c in range(l, right)}
    for p, i in owner.items():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                assert owner.get((p[0]+dr, p[1]+dc), i) == i, "tokens touch"
                assert (p[0]+dr, p[1]+dc) not in protected, "token touches emitter or signal"
    shapes = {_key(c) for c in cavities}
    assert {token["shape"] for token in tokens} == shapes, "every shape must have complete local support"
    return dict(grid=g, patch=patch.copy(), box=box, carrier=carrier,
                background=background, direction=direction, word=tuple(word),
                seed_cells=tuple(seed_cells), cavities=cavities, tokens=tokens,
                roles=next(iter(roles)))


def _render_votes(data, *, equivalence="free", sampling="sensor", first_only=False,
                  prefix_period=False, global_vote=False):
    word = data["word"]
    if prefix_period:
        p = next(p for p in range(1, len(word)+1)
                 if all(v == word[i % p] for i, v in enumerate(word)))
        word = word[:p]
    ballots = defaultdict(Counter)
    for token in data["tokens"]:
        point = token["sensor"]
        if sampling == "center":
            t, l, b, right = _bbox(token["cells"])
            point = (t+b-1)//2, (l+right-1)//2
        elif sampling == "first":
            point = min(token["cells"])
        d = _distance(point, data["box"], data["direction"])
        vote = word[0] if first_only else word[(d-1) % len(word)]
        key = 0 if global_vote else _key(token["cells"], equivalence)
        ballots[key][vote] += 1
    winners = {}
    for key, votes in ballots.items():
        best = max(votes.values())
        colors = [c for c, n in votes.items() if n == best]
        assert len(colors) == 1, "plurality must be unique; ties have no rule"
        winners[key] = colors[0]
    output = data["patch"].copy()
    for cells in data["cavities"]:
        key = 0 if global_vote else _key(cells, equivalence)
        assert key in winners, "shape lacks support under this convention"
        for p in cells:
            output[p] = winners[key]
    return output, dict(ballots), winners


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the answer solely from this input's geometry and signal."""
    return _render_votes(_decode(input_grid))[0]


def _fails(pair, **alternative):
    try:
        answer = _render_votes(_decode(pair["input"]), **alternative)[0]
    except (AssertionError, KeyError):
        return True
    return not np.array_equal(answer, pair["output"])


def _block(available, cells):
    h, w = available.shape
    for r, c in cells:
        available[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = False


def _positions(available, shape):
    sh, sw = _size(shape)
    nr, nc = available.shape[0]-sh+1, available.shape[1]-sw+1
    if nr <= 0 or nc <= 0:
        return np.empty((0, 2), dtype=int)
    valid = np.ones((nr, nc), dtype=bool)
    for r, c in shape:
        valid &= available[r:r+nr, c:c+nc]
    return np.argwhere(valid)


def _pick(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _place_shape(rng, available, variants):
    for i in rng.permutation(len(variants)):
        variant = variants[int(i)]
        positions = _positions(available, variant)
        if len(positions):
            r, c = map(int, _pick(rng, positions))
            cells = tuple((r+dr, c+dc) for dr, dc in variant)
            _block(available, cells)
            return variant, cells
    return None


def _place_token(rng, available, variants, color, word, face_col, prefix_opposite=False):
    for vi in rng.permutation(len(variants)):
        variant = variants[int(vi)]
        origins = _positions(available, variant)
        if not len(origins):
            continue
        for si in rng.permutation(len(variant)):
            sr, sc = variant[int(si)]
            distances = origins[:, 1]+sc-face_col
            valid = np.array([word[(int(d)-1) % len(word)] == color for d in distances])
            if prefix_opposite:
                # ABA's deliberately wrong short-prefix AB rule predicts the
                # opposite color at every winning anchor sensor.
                valid &= np.array([word[(int(d)-1) % 2] != color for d in distances])
            choices = origins[valid]
            if not len(choices):
                continue
            r, c = map(int, _pick(rng, choices))
            cells = tuple((r+dr, c+dc) for dr, dc in variant)
            sensor = r+sr, c+sc
            _block(available, cells)
            return cells, sensor
    return None


def _make_pair(rng, shapes, mode, previous_winner=None, max_period=5):
    """Sample one scene; build labels from planned ballots, not from solve()."""
    colors = list(map(int, rng.permutation(10)))
    bg, carrier, body, sensor_color = colors[:4]
    signal_colors = colors[4:]
    if mode == "word" and signal_colors[1] == previous_winner:
        signal_colors[1], signal_colors[2] = signal_colors[2], signal_colors[1]
    if mode == "plurality":
        word = tuple(signal_colors[:3])
    elif mode == "word":
        word = (signal_colors[0], signal_colors[1], signal_colors[0])
    else:
        length = int(rng.integers(1, max_period+1))
        k = int(rng.integers(1, min(4, length)+1))
        word = tuple(signal_colors[int(i)] for i in rng.integers(k, size=length))
        # Use every selected color; repeated slices and constant words remain legal.
        word = tuple(signal_colors[i] if i < k else word[i] for i in range(length))
    palette = list(dict.fromkeys(word))
    plans, winners = {}, {}
    for i, shape in enumerate(shapes):
        winner = palette[i % len(palette)] if mode == "plurality" else _pick(rng, palette)
        if mode == "word":
            winner = palette[1] if i == 0 else palette[0]
        if mode == "plurality" and i == 0:
            counts = (3, 2, 2)
        elif mode in ("plurality", "word"):
            counts = (2, 1)
        else:
            counts = _pick(rng, ((1,), (2,), (2, 1), (3, 1), (3, 2), (3, 2, 1)))
            counts = counts[:len(palette)]
        losers = [c for c in palette if c != winner]
        rng.shuffle(losers)
        plans[shape] = list(zip([winner]+losers, counts))
        winners[shape] = winner
    # Keep physical packing bounded; optional rich ballots never crowd out a class.
    total = sum(sum(n for c, n in v) for v in plans.values())
    if total > 14:
        for s in reversed(shapes):
            if total <= 14:
                break
            total -= sum(n for c, n in plans[s])-1
            plans[s] = [(winners[s], 1)]
    specs = []
    for i, shape in enumerate(shapes):
        for color, count in plans[shape]:
            variants = _variants(shape)
            if mode == "plurality" and i == 0:
                reflected = _norm((r, -c) for r, c in shape)
                variants = _variants(reflected, False) if color == winners[shape] else _variants(shape, False)
            specs.extend((shape, color, variants, mode == "word" and i == 0
                          and color == winners[shape]) for _ in range(count))
    # Independent cavity multiplicities, not a copy of the exterior multiset.
    repeats = [1+int(rng.integers(3)) for s in shapes]
    for attempt in range(48):
        h = int(rng.integers(22, 31)) if len(specs) >= 8 else int(rng.integers(17, 31))
        w = int(rng.integers(26, 31))
        sh, sw = int(_pick(rng, (9, 11, 13, 15, 17))), int(_pick(rng, (7, 9, 11)))
        x = int(rng.integers(3))
        if attempt >= 16:
            h, w, sh, sw, x = 30, 30, 25, 9, 0
        y = int(rng.integers(h-sh+1))
        free = np.ones((sh, sw), dtype=bool)
        free[[0, -1], :] = False
        free[:, [0, -1]] = False
        cavities = []
        failed = False
        for i in sorted(range(len(shapes)), key=lambda i: len(shapes[i]), reverse=True):
            s = shapes[i]
            choices = (s,) if mode in ("plurality", "word") else _variants(s)
            placed = _place_shape(rng, free, choices)
            if placed is None:
                failed = True
                break
            cavities.append((s, placed[1]))
        if failed:
            continue
        for i, s in enumerate(shapes):
            for _ in range(repeats[i]-1):
                placed = _place_shape(rng, free, _variants(s))
                if placed is not None:
                    cavities.append((s, placed[1]))
        patch = np.full((sh, sw), carrier, dtype=np.int8)
        output = patch.copy()
        for s, cells in cavities:
            for p in cells:
                patch[p], output[p] = bg, winners[s]
        if len(_components(patch == carrier)) != 1:
            continue
        face_col = x+sw-1
        visible_word = word
        if mode == "random" and len(word) <= 2 and rng.random() < .3:
            visible_word = word*2
        start = face_col+len(visible_word)+2
        available = np.ones((h, w), dtype=bool)
        available[:, :start] = False
        external = []
        order = [specs[int(i)] for i in rng.permutation(len(specs))]
        order.sort(key=lambda s: len(s[0]), reverse=True)
        for s, color, variants, opposite in order:
            placed = _place_token(rng, available, variants, color, word, face_col, opposite)
            if placed is None:
                failed = True
                break
            external.append((s, placed[0], placed[1]))
        if failed:
            continue
        grid = np.full((h, w), bg, dtype=np.int8)
        grid[y:y+sh, x:x+sw] = patch
        for j, c in enumerate(visible_word, 1):
            grid[y+sh//2, face_col+j] = c
        for s, cells, sensor in external:
            for p in cells:
                grid[p] = body
            grid[sensor] = sensor_color
        pair = {"input": grid, "output": output}
        turns = int(rng.integers(4))
        mirror = bool(rng.integers(2))
        pair = {k: np.ascontiguousarray(np.rot90(np.fliplr(v) if mirror else v, turns))
                for k, v in pair.items()}
        # Inspect witnesses after orientation: a rounded bounding-box center is
        # not rotation-covariant when the box has an even dimension.
        if mode == "word" and not _fails(pair, sampling="center"):
            continue
        if mode == "word" and not _fails(pair, prefix_period=True):
            continue
        return pair
    raise RuntimeError("bounded scene packing exhausted")


def _array_key(grid):
    return grid.shape, grid.tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_array_key(np.asarray(p["input"])), _array_key(np.asarray(p["output"])))
                       for p in puzzle[s]) for s in ("train", "test"))


def _counts(a, b):
    for name, value, lo, hi in (("num_train", a, 2, 10), ("num_test", b, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return int(a), int(b)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    pool = _pool()
    anchors = [s for s in pool if 4 <= len(s) <= 6
               and _norm((r, -c) for r, c in s) not in _variants(s, False)
               and any(t != s and len(t) == len(s) and sorted(_size(t)) == sorted(_size(s)) for t in pool)]
    anchor = _pick(rng, anchors)
    companion = _pick(rng, [s for s in pool if s != anchor and len(s) == len(anchor)
                           and sorted(_size(s)) == sorted(_size(anchor))])
    puzzle, seen, taught = {"train": [], "test": []}, set(), set()
    prior = None
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            mode = "plurality" if split == "train" and i == 0 else "word" if split == "train" and i == 1 else "random"
            if mode != "random":
                shapes = [anchor, companion]
            else:
                k = int(rng.integers(1, 5))
                shapes = [_pick(rng, pool)]
                if split == "test":
                    shapes = [_pick(rng, [s for s in pool if s not in taught])]
                shapes += list(_pick_many(rng, [s for s in pool if s not in shapes], k-1))
            for _ in range(8):
                pair = _make_pair(rng, shapes, mode, prior)
                key = _array_key(pair["input"])
                if key not in seen:
                    break
            else:
                # Finite color permutation repair; there are far more bijections
                # than the maximum 14 excluded inputs, without changing the rule.
                for permutation in itertools.permutations(range(10)):
                    mapping = np.array(permutation, dtype=np.int8)
                    candidate = {k: mapping[v] for k, v in pair.items()}
                    if _array_key(candidate["input"]) not in seen:
                        pair = candidate
                        break
            seen.add(_array_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train":
                taught.update(shapes)
                if i == 0:
                    prior = _render_votes(_decode(pair["input"]))[2][anchor]
    validate_generated_puzzle(puzzle)
    return puzzle


def _pick_many(rng, sequence, n):
    return [sequence[int(i)] for i in rng.choice(len(sequence), n, replace=False)]


def validate_generated_puzzle(puzzle):
    """Check schema, reconstructed ballots, protected carrier, and teaching evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            key = _array_key(g)
            assert key not in seen, "duplicate input"
            seen.add(key)
            data = _decode(g)
            expected, ballots, winners = _render_votes(data)
            assert np.array_equal(expected, out), "wrong phase, vote, fill, or crop"
            assert np.all(out[data["patch"] == data["carrier"]] == data["carrier"])
            assert all(sum(v.values()) >= 1 for v in ballots.values())
            assert all(c in data["word"] for c in winners.values())
            data.update(ballots=ballots, winners=winners)
            analyses[split].append(data)
    train = puzzle["train"]
    for alternative in (dict(first_only=True), dict(global_vote=True), dict(equivalence="area"),
                        dict(equivalence="box"), dict(equivalence="rotation"),
                        dict(sampling="center"), dict(prefix_period=True)):
        assert any(_fails(pair, **alternative) for pair in train), ("missing teaching witness", alternative)
    assert any(len(v) >= 3 and max(v.values())*2 <= sum(v.values())
               for data in analyses["train"] for v in data["ballots"].values()), "teach plurality, not strict majority"
    history = defaultdict(set)
    for data in analyses["train"]:
        for s, c in data["winners"].items():
            history[s].add(c)
    assert any(len(colors) > 1 for colors in history.values()), "reject fixed shape-color lookup"
    known = set(history)
    assert all(set(data["ballots"])-known for data in analyses["test"]), "test must include a new supported geometry"


def family_features(pair):
    data = _decode(pair["input"])
    _, ballots, winners = _render_votes(data)
    return dict(input_shape=list(np.asarray(pair["input"]).shape),
                output_shape=list(np.asarray(pair["output"]).shape),
                direction=data["direction"], word=list(data["word"]),
                period=next(p for p in range(1, len(data["word"])+1)
                            if len(data["word"]) % p == 0 and all(v == data["word"][i % p]
                            for i, v in enumerate(data["word"]))),
                classes=len(ballots), tokens=len(data["tokens"]), cavities=len(data["cavities"]),
                ballots=sorted([sorted(v.values(), reverse=True) for v in ballots.values()]),
                areas=sorted({len(s) for s in ballots}), roles=[data["background"], data["carrier"], *data["roles"]],
                sensor_distances=sorted({t["distance"] for t in data["tokens"]}),
                sensor_offsets=sorted({_norm_offset(t) for t in data["tokens"]}),
                winners=len(set(winners.values())))


def _norm_offset(token):
    t, l, b, right = _bbox(token["cells"])
    return token["sensor"][0]-t, token["sensor"][1]-l


def _sensor_counterfactual(pair):
    """Move only a sensor within its shape; a 3-2-2 ballot changes to 2-3-2.

    Expected labels are edited from the original crop's affected cavity class,
    without using solve() to construct the edited answer.
    """
    data = _decode(pair["input"])
    _, ballots, winners = _render_votes(data)
    for token in data["tokens"]:
        shape, old_color = token["shape"], token["vote"]
        if sorted(ballots[shape].values()) != [2, 2, 3] or old_color != winners[shape]:
            continue
        for point in token["cells"]:
            d = _distance(point, data["box"], data["direction"])
            new_color = data["word"][(d-1) % len(data["word"])]
            if new_color == old_color or ballots[shape].get(new_color) != 2:
                continue
            changed = data["grid"].copy()
            body, sensor = data["roles"]
            changed[token["sensor"]] = body
            changed[point] = sensor
            expected = np.asarray(pair["output"]).copy()
            for cavity in data["cavities"]:
                if _key(cavity) == shape:
                    for p in cavity:
                        expected[p] = new_color
            assert not np.array_equal(expected, pair["output"])
            assert np.array_equal(solve(changed), expected)
            assert {_key(c) for c in _decode(changed)["cavities"]} == set(ballots)
            return {"input": changed, "output": expected}
    raise AssertionError("no answer-changing sensor move in the plurality demonstration")


def self_test():
    """Bounded 42-seed diversity and counterfactual checks; no external evidence writes."""
    # Exercise specified geometry beyond the sampler's catalog: an eleven-cell
    # L, a six-slice signal, and a sensor in the middle of its long arm.
    larger = np.full((30, 30), 0, dtype=np.int8)
    larger[:17, :9] = 1
    large_shape = tuple([(r, 0) for r in range(6)] + [(5, c) for c in range(1, 6)])
    expected = np.ones((17, 9), dtype=np.int8)
    for r, c in large_shape:
        larger[r+3, c+1] = 0
        expected[r+3, c+1] = 3
    larger[8, 9:15] = [3, 4, 3, 5, 4, 5]
    for top, sensor_col in ((0, 21), (8, 21), (16, 19)):
        for r, c in large_shape:
            larger[top+r, 18+c] = 2
        larger[top+5, sensor_col] = 9
    for turns in range(4):
        for mirror in (False, True):
            transform = lambda a: np.rot90(np.fliplr(a) if mirror else a, turns)
            assert np.array_equal(solve(transform(larger)), transform(expected))
    puzzles, features, signatures, vocab = [], [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        signatures.add(_puzzle_key(puzzle))
        _sensor_counterfactual(puzzle["train"][0])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair["input"]), pair["output"])
                features.append(family_features(pair))
                vocab.update(_render_votes(_decode(pair["input"]))[1])
    assert len(signatures) == 42
    assert len({tuple(f["input_shape"]) for f in features}) >= 30
    assert len({tuple(f["output_shape"]) for f in features}) >= 20
    assert {f["direction"] for f in features} == set(range(4))
    assert {f["classes"] for f in features} == {1, 2, 3, 4}
    assert {f["period"] for f in features} >= {1, 2, 3, 4, 5}
    assert len(vocab) >= 40
    assert len({f["tokens"] for f in features}) >= 9
    assert len({f["cavities"] for f in features}) >= 7
    assert len({tuple(f["roles"]) for f in features}) >= 60
    assert any(f["cavities"] > f["classes"] for f in features)
    assert any(f["ballots"] == [[1]] for f in features)
    assert len({tuple(p) for f in features for p in f["sensor_offsets"]}) >= 10
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(19231, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, True), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
              for s in ("train", "test")}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted carrier accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, all legal counts, family invariants")
