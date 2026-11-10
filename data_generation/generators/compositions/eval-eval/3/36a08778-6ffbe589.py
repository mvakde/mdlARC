"""Torque board: branching rays turn the color layers that obstruct them.

Only standard-library modules and NumPy are used. No parent generator is imported.
All interpretation, propagation and rotation state is reconstructed from the input.
"""
from collections import Counter
import json

import numpy as np


HOST_SEEDS = (0, 557591789, 544582612, 715147460, 649620035, 509483748,
              216132869, 43046904, 821803454, 412355312, 396647109,
              841384267, 368038165, 710845687, 559957709, 220936336,
              340556156, 403432692, 227378297, 977875248, 326223757,
              788561954, 641451644, 143547342, 646016223, 310971774,
              391160062, 843086450, 943358146, 246218535, 418224399,
              65078768, 529564643, 322564635, 238620397, 356679849,
              686675437, 492347995, 906301674, 405988623, 884420194,
              852433844)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grids required")
    _require(np.all((0 <= a) & (a <= 9)), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _orbit(r, c, n):
    result = []
    for _ in range(4):
        if (r, c) not in result:
            result.append((r, c))
        r, c = c, n - 1 - r
    return tuple(result)


def _orbits(n):
    remaining = {(r, c) for r in range(n) for c in range(n)}
    result = []
    while remaining:
        orbit = _orbit(*min(remaining), n)
        result.append(orbit)
        remaining.difference_update(orbit)
    return result


def _parse(value):
    a = _grid(value)
    _require(a.shape[0] == a.shape[1] and a.shape[0] >= 7, "square rotation board required")
    vals, amounts = np.unique(a, return_counts=True)
    _require(len(vals) >= 4 and np.sum(amounts == amounts.max()) == 1, "visible background and two obstacle colors required")
    bg = int(vals[amounts.argmax()])
    choices = []
    n = len(a)
    for orientation in range(4):
        b = np.rot90(a, orientation)
        for guide in map(int, vals):
            if guide == bg:
                continue
            entries = list(map(int, np.flatnonzero(b[0] == guide)))
            if not entries or any(y-x < 2 for x, y in zip(entries, entries[1:])):
                continue
            mask = np.zeros(b.shape, bool)
            lengths = []
            good = True
            for c in entries:
                rows = np.flatnonzero(b[:, c] == guide)
                if not 2 <= len(rows) < n or not np.array_equal(rows, np.arange(len(rows))):
                    good = False
                    break
                mask[:len(rows), c] = True
                lengths.append(len(rows))
            if good and np.array_equal(mask, b == guide):
                choices.append((orientation, guide, entries, lengths, b.copy()))
    _require(len(choices) == 1, "boundary stubs must identify one guide and direction")
    orientation, guide, entries, lengths, board = choices[0]
    colors = tuple(int(c) for c in vals if c not in (bg, guide))
    # The orbit closures are input-visible compatibility conditions, not hidden
    # layer territories needed by the solver. They prevent *any* turn conflict.
    ownership = {}
    for color in (guide,) + colors:
        for r, c in np.argwhere(board == color):
            for p in _orbit(int(r), int(c), n):
                _require(p not in ownership or ownership[p] == color,
                         "different color layers share a rotation orbit")
                ownership[p] = color
    return {"grid": board, "background": bg, "guide": guide, "colors": colors,
            "orientation": orientation, "entries": entries, "stub_lengths": lengths}


def _runs(row, bg):
    """Occupied transverse runs may contain several colors."""
    runs = []
    c = 0
    while c < len(row):
        if row[c] == bg:
            c += 1
            continue
        left = c
        while c + 1 < len(row) and row[c+1] != bg:
            c += 1
        runs.append((left, c))
        c += 1
    return runs


def _rotate_layers(board, bg, charges):
    result = np.full(board.shape, bg, np.int8)
    occupied = np.zeros(board.shape, bool)
    for color in map(int, np.unique(board)):
        if color == bg:
            continue
        mask = np.rot90(board == color, -int(charges.get(color, 0)) % 4)
        _require(not np.any(mask & occupied), "rotated layer overlap")
        result[mask] = color
        occupied |= mask
    return result


def _simulate(info, shortcut=None):
    """Synchronous forward scan; wrong-rule variants are used only for QA."""
    source = info["grid"]
    n, bg, guide = len(source), info["background"], info["guide"]
    board = source.copy()
    board[board == guide] = bg
    trace = source == guide
    active = set(info["entries"])
    totals = Counter()
    events = []
    snapshots = []
    for r in range(1, n):
        row_runs = _runs(board[r], bg)
        hit_runs = [(a, b) for a, b in row_runs if any(a <= c <= b for c in active)]
        following = {c for c in active if board[r, c] == bg}
        for c in following:
            trace[r, c] = True
        charges = Counter()
        for a, b in (row_runs if shortcut == "all_runs" else hit_runs):
            hits = sum(a <= c <= b for c in active)
            amounts = Counter(map(int, board[r, a:b+1]))
            if shortcut == "one_per_run":
                amounts = Counter({color: 1 for color in amounts})
            elif shortcut == "per_ray":
                amounts = Counter({color: amount * hits for color, amount in amounts.items()})
            charges.update(amounts)
            events.append({"row": r, "left": a, "right": b, "hits": hits,
                           "amounts": dict(amounts)})
            if shortcut != "no_branch":
                for c in range(max(0, a-1), min(n, b+2)):
                    if board[r-1, c] == bg:
                        trace[r-1, c] = True
                for c in (a-1, b+1):
                    if 0 <= c < n:
                        following.add(c)
                        trace[r, c] = True
        totals.update(charges)
        if charges and shortcut not in ("frozen", "no_rotation"):
            if shortcut == "counterclockwise":
                applied = {c: -v for c, v in charges.items()}
            elif shortcut == "global_charge":
                applied = {c: sum(charges.values()) for c in info["colors"]}
            else:
                applied = charges
            board = _rotate_layers(board, bg, applied)
        snapshots.append((r, tuple(sorted(active)), tuple(hit_runs)))
        active = following
    if shortcut == "frozen":
        board = _rotate_layers(board, bg, totals)
    out = np.full(source.shape, bg, np.int8)
    out[trace] = guide
    occupied = board != bg
    out[occupied] = board[occupied]
    _require(np.all(out[source == guide] == guide), "guide stubs overwritten")
    return out, {"charges": dict(totals), "events": events,
                 "trace": trace, "final_board": board, "snapshots": snapshots}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Interpret one input only; clockwise torque and chronology are fixed."""
    info = _parse(input_grid)
    out, _ = _simulate(info)
    return np.ascontiguousarray(np.rot90(out, -info["orientation"]), dtype=np.int8)


SHORTCUTS = ("no_rotation", "no_branch", "frozen", "one_per_run", "per_ray",
             "counterclockwise", "global_charge", "all_runs")


def _distinguished(info):
    correct, _ = _simulate(info)
    return {s for s in SHORTCUTS if not np.array_equal(_simulate(info, s)[0], correct)}


def _stubs(rng, n, count=None):
    if count is None:
        count = int(rng.integers(1, min(5, (n+1)//2)))
    candidates = list(map(int, rng.permutation(n)))
    entries = []
    for c in candidates:
        if all(abs(c-x) >= 2 for x in entries):
            entries.append(c)
            if len(entries) == count:
                break
    lengths = [int(rng.integers(2, 5)) for _ in entries]
    return sorted(zip(entries, lengths))


def _intro(rng):
    """One late mixed-color run visibly supplies raw charges 1, 2, 3.

    Its entire orbital closure is upstream of the impact, so this explanation
    is unobscured by a later cascade. Satellites show whole-layer movement.
    Two entries hit the same run, proving that it is counted once.
    """
    n = int(rng.choice([13, 14, 15, 16, 17, 18, 19]))
    a = np.zeros((n, n), np.int8)
    row = n-3
    left = int(rng.integers(3, n-8))
    entries = [left+1, left+4]
    a[:2, entries] = 1
    reserved = {p for r, c in np.argwhere(a == 1) for p in _orbit(int(r), int(c), n)}
    owner = {}
    for c, color in zip(range(left, left+6), [2, 3, 3, 4, 4, 4]):
        orbit = _orbit(row, c, n)
        if set(orbit) & reserved or any(p in owner and owner[p] != color for p in orbit):
            return None
        owner.update({p: color for p in orbit})
        a[row, c] = color
    # Place an unstruck satellite for each controlled color, within the safe
    # central square and away from the incoming columns and impact row.
    for color in (2, 3, 4):
        candidates = [p for p, k in owner.items() if k == color and a[p] == 0
                      and p[0] != row and p[1] not in entries]
        if not candidates:
            return None
        p = candidates[int(rng.integers(len(candidates)))]
        a[p] = color
    # A separate asymmetric uncharged layer makes the zero-count default visible.
    candidates = [o for o in _orbits(n) if len(o) == 4 and all(
        2 <= r <= row and 2 <= c <= row and p not in owner and p not in reserved
        for p in o for r, c in [p])]
    rng.shuffle(candidates)
    for orbit in candidates:
        spots = [p for p in orbit if p[0] != row and p[1] not in entries]
        if spots:
            a[spots[int(rng.integers(len(spots)))]] = 5
            break
    else:
        return None
    return a


def _random_board(rng, n, richness):
    count = int(rng.integers(2, 6))
    a = np.zeros((n, n), np.int8)
    for c, length in _stubs(rng, n):
        a[:length, c] = 1
    reserved = {p for r, c in np.argwhere(a == 1) for p in _orbit(int(r), int(c), n)}
    orbits = [o for o in _orbits(n) if not set(o) & reserved]
    lookup = {p: i for i, o in enumerate(orbits) for p in o}
    owner = {}
    # Spatial motifs claim complete orbits but occupy chosen phases only.
    # Ownership is just a construction device for collision-free rotations.
    for color in range(2, count+2):
        available = [i for i, o in enumerate(orbits) if i not in owner and len(o) == 4]
        if not available:
            return None
        anchor_id = available[int(rng.integers(len(available)))]
        anchor = orbits[anchor_id][int(rng.integers(4))]
        ar, ac = anchor
        kind = int(rng.integers(5))
        length = int(rng.integers(1, min(n, 8 if richness else 5)))
        if kind == 0:
            points = [(ar, ac+j) for j in range(-length//2, length-length//2)]
        elif kind == 1:
            points = [(ar+j, ac) for j in range(-length//2, length-length//2)]
        elif kind == 2:
            points = [(ar, ac+j) for j in range(length)] + [(ar+j, ac) for j in range(length)]
        elif kind == 3:
            points = [(ar+j//2, ac+j%2) for j in range(length)]
        else:
            points = [(ar+int(rng.integers(-2, 3)), ac+int(rng.integers(-2, 3))) for _ in range(length)]
        points.insert(0, anchor)
        for p in points:
            i = lookup.get(p)
            if i is not None and owner.get(i, color) == color:
                owner[i] = color
                a[p] = color
    # Supplement motifs with bands, islands, and symmetric groups. Choosing
    # spatially neighboring orbits retains recognizable geometry at higher fill.
    density = float(rng.uniform(.05, .24 if richness else .14))
    budget = int(n*n*density)
    free = [i for i in range(len(orbits)) if i not in owner]
    rng.shuffle(free)
    for i in free[:budget//2]:
        color = int(rng.integers(2, count+2))
        owner[i] = color
        orbit = orbits[i]
        phase = int(rng.integers(len(orbit)))
        mode = int(rng.integers(5))
        positions = [orbit[phase]]
        if mode == 1 and len(orbit) == 4:
            positions.append(orbit[(phase+2)%4])
        elif mode == 2:
            positions = list(orbit)
        for p in positions:
            a[p] = color
    # Symmetric layers remain actuators even when a particular turn is visually
    # invariant. Complete only same-color owned orbits, preserving compatibility.
    for color in range(2, count+2):
        mode = int(rng.choice(3, p=[.70, .18, .12]))
        mask = a == color
        if mode == 1:
            a[mask | np.rot90(mask, 2)] = color
        elif mode == 2:
            a[mask | np.rot90(mask) | np.rot90(mask, 2) | np.rot90(mask, 3)] = color
    # An optional central singleton is a distinct, fully invariant whole layer.
    if n % 2 and a[n//2, n//2] == 0 and rng.random() < .08:
        color = int(rng.integers(2, count+2))
        a[a == color] = 0
        a[n//2, n//2] = color
    if np.count_nonzero(a == 0) <= max(np.count_nonzero(a == c) for c in range(1, count+2)):
        return None
    return a


def _palette(rng, a):
    values = list(map(int, np.unique(a)))
    actual = rng.choice(10, len(values), replace=False)
    mapping = np.arange(10, dtype=np.int8)
    mapping[values] = actual
    return mapping[a]


def _fallback_boards(mode, curriculum):
    """Finite constructive reserve with more candidates than a legal puzzle.

    These are parameterized versions of the local witness motifs, not parent
    pairs. They keep semantic and curriculum requirements after rejection.
    """
    if mode == "intro":
        a = np.zeros((17, 17), np.int8)
        a[:2, [4, 7]] = 1
        a[14, 3:9] = [2, 3, 3, 4, 4, 4]
        a[3, 2], a[4, 2], a[6, 2], a[8, 9] = 2, 3, 4, 5
        yield a
    elif mode == "test" and curriculum == "cycle":
        for n in (11, 13, 15, 17, 19, 21):
            a = np.zeros((n, n), np.int8)
            a[:2, 5] = 1
            a[n-3, 3:8] = 2
            a[n//2, n//2] = 3
            yield a
    else:
        sizes = (21, 23, 25, 27, 29) if mode == "test" and curriculum == "scale" else (9, 11, 13, 15, 17, 19)
        for n in sizes:
            offset = (n-9)//2
            a = np.zeros((n, n), np.int8)
            a[:2, offset+4] = 1
            a[offset+2, offset+4] = 2
            a[offset+3, offset+6] = 2
            a[offset+6, offset+3] = 3
            a[offset+8, offset+6] = 4
            yield a


def _make_pair(rng, mode, curriculum, seen, taught, attempts=400, orientations=(0, 1, 2, 3)):
    for attempt in range(attempts):
        if mode == "intro":
            raw = _intro(rng)
        else:
            if mode == "test" and curriculum == "scale":
                n = int(rng.integers(21, 30))
            else:
                n = int(rng.integers(9, 22 if mode == "test" else 20))
            raw = _random_board(rng, n, richness=mode in ("cascade", "test") or rng.random() < .5)
        if raw is None:
            continue
        # A whole-board orientation is chosen independently for every pair.
        orientation = int(rng.choice(orientations))
        a = np.ascontiguousarray(np.rot90(_palette(rng, raw), -orientation))
        if _key(a) in seen:
            continue
        try:
            info = _parse(a)
        except AssertionError:
            continue
        out, stats = _simulate(info)
        if not stats["events"]:
            continue
        if np.count_nonzero(out == info["guide"]) <= np.count_nonzero(a == info["guide"]):
            continue
        distinguished = _distinguished(info)
        if mode == "intro":
            if any(any(np.array_equal(info["grid"] == c, np.rot90(info["grid"] == c, k))
                       for k in (1, 2, 3)) for c in stats["charges"]):
                continue
            if not {"no_rotation", "one_per_run", "per_ray", "counterclockwise", "global_charge"} <= distinguished:
                continue
            if len(stats["events"]) != 1 or sorted(stats["charges"].values()) != [1, 2, 3]:
                continue
        if mode == "cascade":
            if not {"no_branch", "frozen", "all_runs"} <= distinguished:
                continue
            # Require the feedback and branching effects to change actual
            # obstacle masks, not merely the number of visible guide pixels.
            if any(np.array_equal(_simulate(info, s)[1]["final_board"], stats["final_board"])
                   for s in ("no_branch", "frozen")):
                continue
            if len(stats["events"]) < 3:
                continue
        if mode == "test":
            if not distinguished <= taught:
                continue
            if curriculum == "recombine" and len(stats["events"]) < 4:
                continue
            if curriculum == "cycle" and max(stats["charges"].values(), default=0) < 5:
                continue
        pair = {"input": a, "output": np.ascontiguousarray(np.rot90(out, -info["orientation"]))}
        return pair
    # Forty palette/direction variants of each reserve board guarantee a fresh
    # input when at most fourteen legal pairs exist. No probabilistic retry.
    for raw in _fallback_boards(mode, curriculum):
        for orientation in orientations:
            for offset in range(10):
                a = np.ascontiguousarray(np.rot90((raw+offset) % 10, -orientation))
                if _key(a) in seen:
                    continue
                info = _parse(a)
                out, stats = _simulate(info)
                if mode == "test":
                    _require(_distinguished(info) <= taught, "reserve teaching coverage")
                return {"input": a, "output": np.ascontiguousarray(np.rot90(out, -orientation))}
    raise AssertionError("legal puzzle exhausted the finite uniqueness reserve")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test) or not 2 <= num_train <= 10
            or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(int(seed))
    curriculum = str(rng.choice(["scale", "recombine", "cycle"]))
    held_out_direction = int(rng.integers(4))
    train_directions = list(map(int, rng.permutation([d for d in range(4) if d != held_out_direction])))
    puzzle = {"train": [], "test": []}
    seen, taught = set(), set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            mode = ("intro" if index == 0 else "cascade" if index == 1 else "ordinary") if split == "train" else "test"
            directions = (held_out_direction,) if split == "test" else (train_directions[index],) if index < 2 else tuple(train_directions)
            pair = _make_pair(rng, mode, curriculum, seen, taught, orientations=directions)
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
            if split == "train":
                taught |= _distinguished(_parse(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def _coordinate_rotation(source, bg, charges):
    """Net layer conservation check, using coordinates instead of np.rot90."""
    n = len(source)
    result = np.full(source.shape, bg, np.int8)
    for r, c in np.argwhere(source != bg):
        color = int(source[r, c])
        y, x = int(r), int(c)
        for _ in range(charges.get(color, 0) % 4):
            y, x = x, n-1-y
        _require(result[y, x] == bg, "coordinate rotation conflict")
        result[y, x] = color
    return result


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, taught = set(), set()
    informative = set()
    training_directions = set()
    feedback = branching = uncharged = multihit = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            a, o = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == o.shape, "board extent must be preserved")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            info = _parse(a)
            expected, stats = _simulate(info)
            actual = np.rot90(o, info["orientation"])
            _require(np.array_equal(actual, expected), "incorrect torque/trace output")
            bg, guide = info["background"], info["guide"]
            source = info["grid"].copy()
            source[source == guide] = bg
            coordinate = _coordinate_rotation(source, bg, stats["charges"])
            _require(np.array_equal(coordinate, stats["final_board"]), "net rotation disagreement")
            for color in info["colors"]:
                _require(np.count_nonzero(a == color) == np.count_nonzero(o == color), "layer area changed")
            _require(np.all(o[a == guide] == guide), "source guide changed")
            _require(np.count_nonzero(o == guide) > np.count_nonzero(a == guide), "trace must visibly extend")
            distinguished = _distinguished(info)
            if split == "train":
                training_directions.add(info["orientation"])
                taught |= distinguished
                frozen = _simulate(info, "frozen")[1]["final_board"]
                stopped = _simulate(info, "no_branch")[1]["final_board"]
                feedback |= not np.array_equal(frozen, stats["final_board"])
                branching |= not np.array_equal(stopped, stats["final_board"])
                multihit |= any(e["hits"] >= 2 for e in stats["events"])
                for color in info["colors"]:
                    mask = info["grid"] == color
                    asymmetric = all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))
                    if asymmetric and color not in stats["charges"]:
                        uncharged = True
                    if asymmetric:
                        informative.update(e["amounts"].get(color, 0) for e in stats["events"])
            else:
                _require(info["orientation"] not in training_directions, "test entry side must be held out")
                _require(distinguished <= taught, "test-changing shortcut lacks a training counterexample")
    _require(set(SHORTCUTS) <= taught, "training leaves a listed shortcut unresolved")
    _require({1, 2, 3} <= informative, "asymmetric unary examples 1,2,3 required")
    _require(len(training_directions) >= 2, "two entry directions must be demonstrated")
    _require(feedback and branching and multihit and uncharged, "missing causal/whole-layer training evidence")


def self_test():
    """Bounded 42-seed structural, semantic-invariant and mutation checks."""
    # Hand-worked feedback example: the first singleton turns a distant color-2
    # satellite onto the right branch at depth 6. Both branches then collide,
    # share their inner flank, and color 2 receives its second quarter-turn.
    literal_input = ["000010000", "000010000", "000020000", "000000200",
                     "000000000", "000000000", "000300000", "000000000", "000000000"]
    literal_output = ["000010000", "000111000", "000101000", "003101000",
                      "000101000", "002111100", "001020100", "001010100", "001010100"]
    decode = lambda rows: np.array([[int(c) for c in row] for row in rows], np.int8)
    fixture = decode(literal_input)
    _require(np.array_equal(solve(fixture), decode(literal_output)), "hand-worked chronology regression")
    fi = _parse(fixture)
    _, fs = _simulate(fi)
    _require(fs["charges"] == {2: 2, 3: 1}, "feedback charge regression")
    _require(not np.array_equal(_simulate(fi, "frozen")[0], decode(literal_output)), "feedback fixture lost its distinction")
    # Four struck cells turn a mask through a full cycle. The uncharged central
    # singleton stays fixed while the guide still branches around the run.
    cycle = np.zeros((9, 9), np.int8)
    cycle[:2, 4] = 1
    cycle[2, 2:6] = 2
    cycle[4, 4] = 3
    ci = _parse(cycle)
    co, cs = _simulate(ci)
    _require(cs["charges"] == {2: 4} and np.array_equal(co == 2, cycle == 2), "full-cycle torque")
    # Exercise family members outside the ordinary sampler's size/layer range.
    # A solver is not allowed to depend on the generated motif or palette count.
    for n, length in ((7, 2), (8, 2), (30, 6)):
        extended = np.zeros((n, n), np.int8)
        extended[:length, n//2] = 1
        reserved = {p for r, c in np.argwhere(extended == 1) for p in _orbit(int(r), int(c), n)}
        available = [o for o in _orbits(n) if not set(o) & reserved]
        for color, orbit in zip(range(2, 10), available):
            extended[orbit[0]] = color
        ei = _parse(extended)
        eo, es = _simulate(ei)
        _require(len(ei["colors"]) == 8 and solve(extended).shape == (n, n), "full-palette grammar")
        original = extended.copy()
        original[original == 1] = 0
        _require(np.array_equal(_coordinate_rotation(original, 0, es["charges"]), es["final_board"]), "extended family net rotation")
    puzzles = [generate_puzzle(seed) for seed in HOST_SEEDS]
    shapes, directions, backgrounds, layers, entries, structures = set(), set(), set(), set(), set(), set()
    features = Counter()
    symmetries = Counter()
    residues, amounts, event_counts = set(), set(), set()
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                info = _parse(a)
                _, stats = _simulate(info)
                shapes.add(a.shape)
                directions.add(info["orientation"])
                backgrounds.add(info["background"])
                layers.add(len(info["colors"]))
                entries.add(len(info["entries"]))
                for color in info["colors"]:
                    mask = info["grid"] == color
                    symmetry = "quarter" if np.array_equal(mask, np.rot90(mask)) else "half" if np.array_equal(mask, np.rot90(mask, 2)) else "asymmetric"
                    symmetries[symmetry] += 1
                event_counts.add(len(stats["events"]))
                amounts.update(v for e in stats["events"] for v in e["amounts"].values())
                residues.update(v % 4 for v in stats["charges"].values())
                roles = {info["background"]: 0, info["guide"]: 1}
                normalized = tuple(roles.setdefault(int(c), len(roles)) for c in info["grid"].flat)
                structures.add((a.shape, normalized))
                features["simultaneous"] += any(sum(e["row"] == r for e in stats["events"]) > 1 for r in range(len(a)))
                features["mixed_run"] += any(len(e["amounts"]) > 1 for e in stats["events"])
                features["clipped"] += any(e["left"] == 0 or e["right"] == len(a)-1 for e in stats["events"])
                features["repeated_color"] += any(sum(c in e["amounts"] for e in stats["events"]) >= 2 for c in info["colors"])
                features["singleton"] += any(e["left"] == e["right"] for e in stats["events"])
                features["large_charge"] += max(stats["charges"].values(), default=0) >= 5
    _require(len(puzzles) == 42 and len(structures) >= 180, "42-seed structural diversity")
    _require(len(shapes) >= 15 and directions == set(range(4)), "size/direction diversity")
    _require(backgrounds == set(range(10)) and layers == {2, 3, 4, 5}, "palette/layer diversity")
    _require({1, 2, 3, 4} <= entries and len(event_counts) >= 8, "entry/cascade diversity")
    _require(residues == {0, 1, 2, 3} and {1, 2, 3, 4} <= amounts, "torque diversity")
    _require(all(v > 0 for v in features.values()) and len(features) == 6, "dynamic feature coverage")
    _require(set(symmetries) == {"asymmetric", "half", "quarter"}, "layer symmetry coverage")
    for curriculum in ("scale", "recombine", "cycle"):
        for nt, nq in ((2, 1), (10, 4)):
            reserve = {"train": [], "test": []}
            seen, taught = set(), set()
            rng = np.random.default_rng(123)
            for split, count in (("train", nt), ("test", nq)):
                for i in range(count):
                    mode = ("intro" if i == 0 else "cascade" if i == 1 else "ordinary") if split == "train" else "test"
                    reserve_directions = (0,) if split == "test" else (i+1,) if i < 2 else (1, 2, 3)
                    pair = _make_pair(rng, mode, curriculum, seen, taught, attempts=0, orientations=reserve_directions)
                    reserve[split].append(pair)
                    seen.add(_key(pair["input"]))
                    if split == "train":
                        taught |= _distinguished(_parse(pair["input"]))
            validate_generated_puzzle(reserve)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            generate_puzzle(0, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _require(all(_key(p["input"]) == _key(q["input"]) and _key(p["output"]) == _key(q["output"])
                 for s in ("train", "test") for p, q in zip(puzzles[0][s], generate_puzzle(0)[s])), "determinism")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ("train", "test")}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": sum(len(p[s]) for p in puzzles for s in p),
            "board_sizes": sorted(h for h, w in shapes), "directions": sorted(directions),
            "backgrounds": sorted(backgrounds), "layer_counts": sorted(layers),
            "entry_counts": sorted(entries), "color_normalized_structures": len(structures),
            "event_counts": sorted(event_counts), "raw_charges": sorted(amounts),
            "net_residues": sorted(residues), "features": dict(features),
            "layer_symmetries": dict(symmetries),
            "semantic_verification": "generator-owned invariants and counterfactuals only"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
