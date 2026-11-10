"""Motion echoes: compare repeated objects, then transfer their motions to a pen.

Only NumPy and the standard library are used. No examples, random state, or
generation metadata are consulted by solve(). Nothing is written on import,
generation, validation, or self-test.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import hashlib

import numpy as np


SEED_COLOR = 9
_DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)


def _integer(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid size"
    assert np.issubdtype(a.dtype, np.integer), "integer cells required"
    assert np.all((0 <= a) & (a <= 9)), "color range"
    return a


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _connected(points):
    remaining = set(map(tuple, points))
    reached = {remaining.pop()}
    stack = list(reached)
    while stack:
        r, c = stack.pop()
        for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if p in remaining:
                remaining.remove(p)
                reached.add(p)
                stack.append(p)
    return not remaining


def _inside(p, shape):
    return 1 <= p[0] < shape[0] - 1 and 1 <= p[1] < shape[1] - 1


def _line(position, command):
    dr, dc, _ = command
    length = max(abs(dr), abs(dc))
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return [(position[0] + t * sr, position[1] + t * sc)
            for t in range(1, length + 1)]


def _draw(shape, background, start, commands, *, anchored=False):
    """Paint start-exclusive segments; later paint wins, but the seed survives."""
    out = np.full(shape, background, dtype=np.int8)
    out[start] = SEED_COLOR
    position = start
    footprints = []
    for command in commands:
        points = _line(start if anchored else position, command)
        assert points and all(_inside(p, shape) for p in points), "pen leaves tile"
        for p in points:
            if p != start:
                out[p] = command[2]
        position = points[-1]
        footprints.append(points)
    return out, footprints


def _parse(input_grid):
    """Recover the lattice, consensus, and translated objects from this input."""
    grid = _grid(input_grid)
    lattice = int(grid[0, 0])
    rows = np.flatnonzero(np.all(grid == lattice, axis=1))
    cols = np.flatnonzero(np.all(grid == lattice, axis=0))
    assert len(rows) >= 2 and len(cols) >= 4, "framed tile lattice required"
    assert rows[0] == cols[0] == 0
    assert rows[-1] == grid.shape[0] - 1 and cols[-1] == grid.shape[1] - 1
    assert np.all(np.diff(rows) == rows[1]) and np.all(np.diff(cols) == cols[1])
    ph, pw = int(rows[1] - 1), int(cols[1] - 1)
    assert 4 <= ph <= 8 and 5 <= pw <= 8
    bands, copies = len(rows) - 1, len(cols) - 1
    assert 1 <= bands <= 5 and 3 <= copies <= 4
    panels = [[grid[r + 1:s, c + 1:d] for c, d in zip(cols, cols[1:])]
              for r, s in zip(rows, rows[1:])]
    counts = Counter(_key(p) for row in panels for p in row)
    high = max(counts.values())
    winners = [k for k, count in counts.items() if count == high]
    assert len(winners) == 1, "unique consensus required"
    master = next(p for row in panels for p in row if _key(p) == winners[0])
    background = int(master[0, 0])
    colors = set(map(int, np.unique(master)))
    assert len(colors) == 4 and background != SEED_COLOR and lattice not in colors
    assert SEED_COLOR in colors
    inks = tuple(sorted(colors - {background, SEED_COLOR}))
    marks = np.argwhere(master == SEED_COLOR)
    assert len(marks) == 1
    start = tuple(map(int, marks[0]))
    assert _inside(start, master.shape)
    for color in inks:
        points = np.argwhere(master == color)
        assert 1 <= len(points) <= 4 and _connected(points), "one small object per ink"
    commands, selected, source_overlap = [], [], []
    for row in panels:
        for panel in row:
            assert set(map(int, np.unique(panel))) == colors
            assert np.all(panel[[0, -1], :] == background)
            assert np.all(panel[:, [0, -1]] == background)
            assert np.array_equal(panel == SEED_COLOR, master == SEED_COLOR)
        exceptions = [c for c, p in enumerate(row) if not np.array_equal(p, master)]
        assert len(exceptions) == 1, "one visible exception per band"
        selected.append(exceptions[0])
        variant = row[exceptions[0]]
        moved = [color for color in inks
                 if not np.array_equal(variant == color, master == color)]
        assert len(moved) == 1, "exactly one ink object translates"
        color = moved[0]
        old, new = np.argwhere(master == color), np.argwhere(variant == color)
        assert len(old) == len(new), "object size must be preserved"
        delta = new.min(axis=0) - old.min(axis=0)
        dr, dc = map(int, delta)
        assert (dr or dc) and (dr == 0 or dc == 0 or abs(dr) == abs(dc))
        assert {tuple(p + delta) for p in old} == set(map(tuple, new)), "rigid translation"
        # Full colored bitmap equality checks the vacated cells too. In
        # particular, overlapping old/new shapes are not mistaken for singletons.
        expected = master.copy()
        expected[master == color] = background
        for p in new:
            expected[tuple(p)] = color
        assert np.array_equal(variant, expected)
        commands.append((dr, dc, color))
        source_overlap.append(bool(set(map(tuple, old)) & set(map(tuple, new))))
    return dict(shape=(ph, pw), bands=bands, copies=copies, lattice=lattice,
                background=background, inks=inks, start=start, master=master,
                panels=panels, commands=commands, selected=selected,
                source_overlap=source_overlap)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution for every member of the specified family."""
    info = _parse(input_grid)
    return _draw(info["shape"], info["background"], info["start"], info["commands"])[0]


def _translate(master, command):
    dr, dc, color = command
    background = int(master[0, 0])
    result = master.copy()
    result[master == color] = background
    for p in np.argwhere(master == color):
        q = (int(p[0]) + dr, int(p[1]) + dc)
        assert _inside(q, master.shape) and result[q] == background
        result[q] = color
    return result


def _moves(master, inks):
    """Enumerate at most 2*8*6 candidate displacements, retaining legal ones."""
    background = int(master[0, 0])
    moves = []
    for color in inks:
        old = np.argwhere(master == color)
        cleared = master.copy()
        cleared[master == color] = background
        for sr, sc in _DIRECTIONS:
            for length in range(1, max(master.shape) - 2):
                shifted = old + np.array((sr * length, sc * length))
                if all(_inside(p, master.shape) and cleared[tuple(p)] == background
                       for p in shifted):
                    moves.append((sr * length, sc * length, color))
    return moves


def _bent(a, b):
    return a[0] * b[1] != a[1] * b[0]


def _overlapping_move(master, command):
    old = set(map(tuple, np.argwhere(master == command[2])))
    shifted = {(r + command[0], c + command[1]) for r, c in old}
    return bool(old & shifted)


def _path(rng, master, start, moves, length, mode):
    ordered = [moves[int(i)] for i in rng.permutation(len(moves))]

    def fits(position, command):
        points = _line(position, command)
        return points if all(_inside(p, master.shape) for p in points) else None

    if mode == "simple":
        # Two differently colored, nonintersecting segments make chaining and
        # direction transfer visible without any overpainting ambiguity.
        for a in ordered:
            p = fits(start, a)
            if p is None:
                continue
            for b in ordered:
                if a[2] == b[2] or not _bent(a, b):
                    continue
                if bool(a[0] and a[1]) == bool(b[0] and b[1]):
                    continue
                if not (_overlapping_move(master, a) or _overlapping_move(master, b)):
                    continue
                q = fits(p[-1], b)
                if q is not None and not (set(q) & (set(p) | {start})):
                    return [a, b]
        return None
    if mode == "return":
        # Retrace a length>=2 segment in the other color, preserve the seed,
        # then turn. This demonstrates the same crossing rule used in tests.
        for a in ordered:
            if max(abs(a[0]), abs(a[1])) < 2:
                continue
            p = fits(start, a)
            if p is None:
                continue
            for b in ordered:
                if b[:2] != (-a[0], -a[1]) or b[2] == a[2]:
                    continue
                for c in ordered:
                    if _bent(a, c) and fits(start, c) is not None:
                        return [a, b, c]
        return None
    # General programs: bounded depth-first search. Repeats, partial retraces,
    # crossings, diagonal moves, and repeated variants are all permitted.
    failed, nodes = set(), 0

    def visit(position, prefix, used, first, bent):
        nonlocal nodes
        nodes += 1
        if nodes > 2048:
            return None
        if len(prefix) == length:
            return prefix if len(used) == 2 and bent else None
        state = (position, length - len(prefix), tuple(sorted(used)), first, bent)
        if state in failed:
            return None
        offset = int(rng.integers(len(ordered))) if ordered else 0
        choices = ordered[offset:] + ordered[:offset]
        for cmd in choices:
            points = fits(position, cmd)
            if points is None:
                continue
            answer = visit(points[-1], prefix + [cmd], used | {cmd[2]},
                           first or cmd[:2], bent or (first is not None and _bent(first, cmd)))
            if answer is not None:
                return answer
        failed.add(state)
        return None

    return visit(start, [], set(), None, False)


def _master(rng, shape, palette, multi):
    background, _, *inks = palette
    ph, pw = shape
    master = np.full(shape, background, dtype=np.int8)
    start = (int(rng.integers(1, ph - 1)), int(rng.integers(1, pw - 1)))
    master[start] = SEED_COLOR
    for index, color in enumerate(inks):
        choices = _SHAPES[1:] if multi and index == 0 else _SHAPES
        motif = choices[int(rng.integers(len(choices)))]
        mh = max(p[0] for p in motif) + 1
        mw = max(p[1] for p in motif) + 1
        placements = [(r, c) for r in range(1, ph - mh)
                      for c in range(1, pw - mw)
                      if all(master[r + dr, c + dc] == background for dr, dc in motif)]
        if not placements:
            return None
        r, c = placements[int(rng.integers(len(placements)))]
        for dr, dc in motif:
            master[r + dr, c + dc] = color
    return master, start


def _fallback(shape, palette, mode):
    """A constructive safety net after the finite random proposals are exhausted."""
    bg, _, a, b = palette
    ph, pw = shape
    master = np.full(shape, bg, dtype=np.int8)
    master[1, 1:3] = a
    if mode == "return":
        master[2, pw - 2] = b
        start = (3, 1)
    else:
        master[ph - 2, pw - 2] = b
        start = (ph - 2, 1)
    master[start] = SEED_COLOR
    return master, start


def _make_pair(rng, shape, bands, copies, palette, mode):
    master = start = commands = None
    for _ in range(48):
        proposed = _master(rng, shape, palette, mode != "general")
        if proposed is None:
            continue
        master, start = proposed
        commands = _path(rng, master, start, _moves(master, palette[2:]), bands, mode)
        if commands is not None:
            break
    if commands is None:
        master, start = _fallback(shape, palette, mode)
        commands = _path(rng, master, start, _moves(master, palette[2:]), bands, mode)
        assert commands is not None, "constructive fallback must be feasible"
    columns = [int(v) for v in rng.integers(copies, size=bands)]
    if mode == "simple":
        columns[1] = (columns[0] + int(rng.integers(1, copies))) % copies
    elif mode == "return":
        columns[1] = columns[0]
    ph, pw = shape
    grid = np.full((bands * (ph + 1) + 1, copies * (pw + 1) + 1),
                   palette[1], dtype=np.int8)
    for band, command in enumerate(commands):
        variant = _translate(master, command)
        for col in range(copies):
            r, c = 1 + band * (ph + 1), 1 + col * (pw + 1)
            grid[r:r + ph, c:c + pw] = variant if col == columns[band] else master
    # Ground truth starts from the construction's actual motion list; solve()
    # must recover that list independently from colored panel comparisons.
    output = _draw(shape, palette[0], start, commands)[0]
    return {"input": grid, "output": output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer("seed", seed, 0)
    nt = _integer("num_train", num_train, 2, 10)
    ne = _integer("num_test", num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    # Different unordered palettes guarantee distinct inputs, even in a rare
    # structural collision, without retries or retained duplicate puzzles.
    palettes = tuple(combinations(range(9), 4))
    palette_indices = rng.choice(len(palettes), nt + ne, replace=False)
    puzzle = {"train": [], "test": []}
    for index in range(nt + ne):
        palette = tuple(map(int, rng.permutation(palettes[int(palette_indices[index])])))
        if index == 0:
            mode, bands = "simple", 2
            shape = (int(rng.integers(5, 9)), int(rng.integers(5, 9)))
        elif index == 1:
            mode, bands = "return", 3
            shape = (int(rng.integers(6, 9)), int(rng.integers(6, 9)))
        else:
            mode = "general"
            bands = int(rng.integers(2, 4)) if index < nt else int(rng.integers(4, 6))
            max_height = min(8, 29 // bands - 1)
            shape = (int(rng.integers(4, max_height + 1)), int(rng.integers(5, 9)))
        maximum = min(4, 29 // (shape[1] + 1))
        copies = int(rng.integers(3, maximum + 1))
        puzzle["train" if index < nt else "test"].append(
            _make_pair(rng, shape, bands, copies, palette, mode))
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative(info, kind, column=0):
    commands = info["commands"]
    if kind == "fixed_column":
        commands = [cmd for cmd, selected in zip(commands, info["selected"]) if selected == column]
    elif kind == "reverse":
        commands = commands[::-1]
    try:
        return _draw(info["shape"], info["background"], info["start"], commands,
                     anchored=kind == "anchored")[0]
    except AssertionError:
        # An alternative that leaves the visible tile is already contradicted.
        return None


def _different(a, b):
    return a is None or not np.array_equal(a, b)


def _paint_features(info):
    _, footprints = _draw(info["shape"], info["background"], info["start"], info["commands"])
    earlier, overwrite, revisit = {}, False, False
    for cmd, points in zip(info["commands"], footprints):
        for p in points:
            if p == info["start"]:
                revisit = True
            else:
                overwrite |= p in earlier and earlier[p] != cmd[2]
                earlier[p] = cmd[2]
    return overwrite, revisit


def validate_generated_puzzle(puzzle):
    """Validate actual bitmaps, motion semantics, and demonstration witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _parse(inp)
            assert out.shape == info["shape"]
            assert np.array_equal(solve(inp), out), "motion drawing mismatch"
            assert np.count_nonzero(out == SEED_COLOR) == 1
            assert out[info["start"]] == SEED_COLOR
            assert np.all(out[[0, -1], :] == info["background"])
            assert np.all(out[:, [0, -1]] == info["background"])
            # Every variant has the same color histogram as the common tile:
            # ink counts cannot replace the full bitmap comparison.
            histogram = np.bincount(info["master"].ravel(), minlength=10)
            assert all(np.array_equal(np.bincount(p.ravel(), minlength=10), histogram)
                       for row in info["panels"] for p in row)
            infos[split].append(info)
    train = infos["train"]
    assert any(_different(_alternative(i, "anchored"), solve(p["input"]))
               for i, p in zip(train, puzzle["train"])), "demonstrate endpoint chaining"
    assert any(_different(_alternative(i, "reverse"), solve(p["input"]))
               for i, p in zip(train, puzzle["train"])), "demonstrate band order"
    assert any(all(_different(_alternative(i, "fixed_column", c), solve(p["input"]))
                   for c in range(i["copies"]))
               for i, p in zip(train, puzzle["train"])), "refute every fixed column"
    assert any(all(_paint_features(i)) for i in train), "demonstrate overwrite and seed return"
    assert any(any(a == b for a, b in zip(i["selected"], i["selected"][1:]))
               for i in train), "repeated exception columns are allowed"
    assert all({cmd[2] for cmd in i["commands"]} == set(i["inks"]) for i in train[:2])
    assert all(any(np.count_nonzero(i["master"] == c) > 1 for c in i["inks"])
               for i in train[:2]), "demonstrate whole-object motion"
    assert any(any(i["source_overlap"]) for i in train), "demonstrate overlapping translations"
    assert any(dr and dc for i in train for dr, dc, _ in i["commands"]), "demonstrate diagonals"
    assert min(i["bands"] for i in infos["test"]) > max(i["bands"] for i in train)


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed diversity, count, mutation, and geometric checks.

    Streaming measurements retain only hashes/feature sets, never a corpus.
    Returned diagnostics are JSON-compatible; this function writes no files.
    """
    digests, input_shapes, tile_shapes, token_shapes = set(), set(), set(), set()
    directions, lengths, bands_seen, copies_seen = set(), set(), set(), set()
    source_overlap = pair_count = multi = singleton = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digests.add(_puzzle_digest(puzzle))
        for pair in puzzle["train"] + puzzle["test"]:
            pair_count += 1
            info = _parse(pair["input"])
            input_shapes.add(pair["input"].shape)
            tile_shapes.add(info["shape"])
            bands_seen.add(info["bands"])
            copies_seen.add(info["copies"])
            source_overlap += sum(info["source_overlap"])
            for dr, dc, _ in info["commands"]:
                directions.add((int(np.sign(dr)), int(np.sign(dc))))
                lengths.add(max(abs(dr), abs(dc)))
            for color in info["inks"]:
                cells = np.argwhere(info["master"] == color)
                token_shapes.add(tuple(map(tuple, cells - cells.min(axis=0))))
                singleton += int(len(cells) == 1)
                multi += int(len(cells) > 1)
    assert len(digests) == 42
    assert len(input_shapes) >= 20 and len(tile_shapes) >= 15
    assert bands_seen == {2, 3, 4, 5} and copies_seen == {3, 4}
    assert directions == set(_DIRECTIONS) and {1, 2, 3} <= lengths
    assert len(token_shapes) >= 8 and source_overlap >= 10 and multi and singleton
    assert _puzzle_digest(generate_puzzle(12345, 7, 3)) == _puzzle_digest(generate_puzzle(12345, 7, 3))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(1, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Exercise fallback geometry directly, including the smallest general tile.
    rng = np.random.default_rng(101)
    for shape, mode, count in (((5, 5), "simple", 2), ((6, 6), "return", 3),
                               ((4, 5), "general", 5)):
        master, start = _fallback(shape, (0, 1, 2, 3), mode)
        assert _path(rng, master, start, _moves(master, (2, 3)), count, mode) is not None
    base = generate_puzzle(0)
    for damage in ("output", "missing", "two_exceptions", "deformed"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in base[s]] for s in base}
        pair = bad["train"][0]
        info = _parse(pair["input"])
        ph, pw = info["shape"]
        col = info["selected"][0]
        left = 1 + col * (pw + 1)
        if damage == "output":
            pair["output"][info["start"]] = info["background"]
        elif damage == "missing":
            pair["input"][1:ph + 1, left:left + pw] = info["master"]
        elif damage == "two_exceptions":
            target = 1 + ((col + 1) % info["copies"]) * (pw + 1)
            pair["input"][1:ph + 1, target:target + pw] = info["panels"][0][col]
        else:
            color = info["commands"][0][2]
            panel = pair["input"][1:ph + 1, left:left + pw]
            point = tuple(np.argwhere(panel == color)[0])
            panel[point] = info["background"]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError(f"mutation accepted: {damage}")
    return dict(seeds=42, pairs=pair_count, distinct_puzzles=len(digests),
                input_shapes=len(input_shapes), tile_shapes=len(tile_shapes),
                token_shapes=len(token_shapes), directions=sorted(directions),
                lengths=sorted(lengths), bands=sorted(bands_seen), copies=sorted(copies_seen),
                overlapping_source_translations=source_overlap,
                singleton_tokens=singleton, multicell_tokens=multi,
                legal_count_combinations=30, mutations_rejected=4)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
