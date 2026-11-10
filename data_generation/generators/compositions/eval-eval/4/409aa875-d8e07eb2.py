"""Addressed interference: exact token lookup launches toroidal lattice shots.

Only standard-library modules and NumPy are used. No parent module is imported.
All solver conventions are visible in each input; generation metadata is unused.
"""
from collections import Counter
from functools import lru_cache
import itertools

import numpy as np


DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
CARDINAL = tuple(d for d in DIRECTIONS if 0 in d)
HOST_SEEDS = (0, 807700291, 408371993, 93971790, 970940972, 1829004,
              492430625, 318004595, 814831469, 39440363, 356836454,
              181108533, 643530624, 422701078, 611494011, 942396158,
              762320248, 804218792, 623253152, 429678667, 245571150,
              611046092, 329993021, 768472221, 449561399, 801305870,
              273247212, 451696553, 547761944, 38532843, 420712679,
              949809138, 378862285, 743032880, 289420375, 394228428,
              456515744, 393983621, 93291028, 531961932, 809438488, 525220624)


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _orient(grid, transform):
    return np.rot90(np.fliplr(grid) if transform >= 4 else grid,
                    transform % 4).copy()


def _unorient(grid, transform):
    result = np.rot90(grid, -transform % 4)
    return np.fliplr(result).copy() if transform >= 4 else result.copy()


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))
    return grid.astype(np.int8, copy=True)


def _arrow_mask(direction):
    dr, dc = direction
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    mask = np.zeros((3, 3), dtype=bool)
    for r, c in offsets:
        mask[1 + r, 1 + c] = True
    return mask


@lru_cache(maxsize=4)
def _mask_bank(gh, gw):
    """Centered masks, unique even after trimming; connectivity is irrelevant."""
    bank = {}
    for code in range(1, 1 << (gh * gw)):
        mask = np.array([(code >> i) & 1 for i in range(gh * gw)], bool).reshape(gh, gw)
        rr, cc = np.where(mask)
        crop = mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
        key = _key(crop)
        if key not in bank:
            centered = np.zeros((gh, gw), bool)
            r, c = (gh - len(crop)) // 2, (gw - crop.shape[1]) // 2
            centered[r:r + len(crop), c:c + crop.shape[1]] = crop
            bank[key] = centered
    return tuple(bank.values())


def _line(counts, rows, cols, *, any_hit=False, diagonal=False):
    good = (lambda p: counts.get(p, 0) > 0) if any_hit else (lambda p: counts.get(p, 0) == 1)
    lines = [[(r, c) for c in range(cols)] for r in range(rows)]
    lines += [[(r, c) for r in range(rows)] for c in range(cols)]
    if diagonal and rows == cols:
        lines += [[(i, i) for i in range(rows)], [(i, cols - 1 - i) for i in range(rows)]]
    return any(all(good(p) for p in line) for line in lines)


def _counts(info, commands=None, *, move=True, wrap=True, lookup="exact"):
    commands = info["commands"] if commands is None else commands
    counts = Counter()
    for index, (origin, direction, _) in enumerate(commands):
        if lookup != "exact":
            token = info["tokens"][origin]
            if lookup == "shape":
                candidates = [p for p, other in info["tokens"].items()
                              if np.array_equal(token != info["bg"], other != info["bg"])]
            elif lookup == "color":
                ink = int(token[token != info["bg"]][0])
                candidates = [p for p, other in info["tokens"].items()
                              if int(other[other != info["bg"]][0]) == ink]
            elif lookup == "ordinal":
                candidates = [divmod(index % (info["rows"] * info["cols"]), info["cols"])]
            else:
                raise ValueError("unknown counterfactual")
            origin = min(candidates)
        r, c = origin
        if move:
            r += direction[0]
            c += direction[1]
        if wrap:
            r, c = r % info["rows"], c % info["cols"]
        if 0 <= r < info["rows"] and 0 <= c < info["cols"]:
            counts[r, c] += 1
    return counts


def _render(info, counts, *, any_hit=False, diagonal=False):
    out = info["grid"].copy()
    for (r, c), count in counts.items():
        rs = slice(info["top"] + 1 + r * info["sh"], info["top"] + 1 + (r + 1) * info["sh"])
        cs = slice(info["left"] + 1 + c * info["sw"], info["left"] + 1 + (c + 1) * info["sw"])
        block = out[rs, cs]
        block[block == info["bg"]] = info["single"] if count == 1 else info["collision"]
    win = _line(counts, info["rows"], info["cols"], any_hit=any_hit, diagonal=diagonal)
    out[-1] = info["single"] if win else info["collision"]
    return out


def _parse_canonical(grid):
    bg, single, collision = int(grid[0, 0]), int(grid[0, 1]), int(grid[0, 3])
    assert len({bg, single, collision}) == 3 and grid[0, 4] == collision
    expected = np.full(grid.shape, bg, np.int8)
    expected[0, 1], expected[0, 3:5] = single, collision
    assert np.array_equal(expected[0], grid[0])
    full = [r for r in range(1, len(grid)) if grid[r, 0] != bg and np.all(grid[r] == grid[r, 0])]
    assert len(full) == 1 and full[0] in (5, 9)
    separator = full[0]
    divider = int(grid[separator, 0])
    assert divider not in (bg, single, collision)
    expected[separator] = divider
    rr, cc = np.where(grid[separator + 1:] == divider)
    assert len(rr)
    top, bottom = int(rr.min()) + separator + 1, int(rr.max()) + separator + 1
    left, right = int(cc.min()), int(cc.max())
    assert top == separator + 1 and bottom == len(grid) - 2
    assert 1 <= left < right < grid.shape[1] - 1
    height, width = bottom - top - 1, right - left - 1
    row_choices = [(r, h) for r in (3, 4) for h in (2, 3) if r * (h + 2) == height]
    col_choices = [(c, w) for c in (3, 4) for w in (2, 3) if c * (w + 2) == width]
    assert len(row_choices) == len(col_choices) == 1
    rows, gh = row_choices[0]
    cols, gw = col_choices[0]
    sh, sw = gh + 2, gw + 2
    expected[top, left:right + 1] = divider
    expected[bottom, left:right + 1] = divider
    expected[top:bottom + 1, left] = divider
    expected[top:bottom + 1, right] = divider
    tokens, lookup, trimmed = {}, {}, set()
    for r, c in itertools.product(range(rows), range(cols)):
        rs = slice(top + 2 + r * sh, top + 2 + r * sh + gh)
        cs = slice(left + 2 + c * sw, left + 2 + c * sw + gw)
        token = grid[rs, cs].copy()
        inks = np.unique(token[token != bg])
        assert len(inks) == 1 and int(inks[0]) not in (single, collision, divider)
        mask = token != bg
        tr, tc = np.where(mask)
        crop = token[tr.min():tr.max() + 1, tc.min():tc.max() + 1]
        assert tr.min() == (gh - crop.shape[0]) // 2
        assert tc.min() == (gw - crop.shape[1]) // 2
        assert _key(token) not in lookup and _key(crop) not in trimmed
        lookup[_key(token)] = r, c
        trimmed.add(_key(crop))
        tokens[r, c] = token
        expected[rs, cs] = token
    card_width = gw + 5
    card_cols = (grid.shape[1] - 2) // card_width
    assert 1 <= card_cols <= 3
    commands = []
    for cr in range((separator - 1) // 4):
        for cc in range(card_cols):
            r, c = 1 + cr * 4, 1 + cc * card_width
            block = grid[r:r + 4, c:c + card_width]
            if np.all(block == bg):
                continue
            token = grid[r + (3 - gh) // 2:r + (3 - gh) // 2 + gh, c:c + gw]
            assert _key(token) in lookup
            arrow = grid[r:r + 3, c + gw + 1:c + gw + 4]
            assert np.all((arrow == bg) | (arrow == single))
            directions = [d for d in DIRECTIONS if np.array_equal(arrow == single, _arrow_mask(d))]
            assert len(directions) == 1
            commands.append((lookup[_key(token)], directions[0], (r, c)))
            expected[r + (3 - gh) // 2:r + (3 - gh) // 2 + gh, c:c + gw] = token
            expected[r:r + 3, c + gw + 1:c + gw + 4] = arrow
    assert 1 <= len(commands) <= 6
    assert len({(p, d) for p, d, _ in commands}) == len(commands), "duplicate instruction"
    assert np.array_equal(grid, expected), "unexpected ink, padding, frame or nonblank status"
    return dict(grid=grid, bg=bg, single=single, collision=collision, divider=divider,
                rows=rows, cols=cols, gh=gh, gw=gw, sh=sh, sw=sw,
                top=top, left=left, bottom=bottom, right=right,
                separator=separator, tokens=tokens, commands=commands)


def _parse(grid):
    grid = _grid(grid)
    choices = []
    if min(grid.shape) < 6:
        raise AssertionError("not an addressed-interference scene")
    for transform in range(8):
        try:
            info = _parse_canonical(_orient(grid, transform))
        except (AssertionError, IndexError):
            continue
        info["transform"] = transform
        choices.append(info)
    assert len(choices) == 1, "scene orientation must be uniquely visible"
    return choices[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read only this input, project every command, and render simultaneous hits."""
    info = _parse(input_grid)
    return _unorient(_render(info, _counts(info)), info["transform"])


def _target_plan(rng, rows, cols, kind, axis):
    index = int(rng.integers(rows if axis == "row" else cols))
    line = [(index, c) for c in range(cols)] if axis == "row" else [(r, index) for r in range(rows)]
    outside = [p for p in itertools.product(range(rows), range(cols)) if p not in line]
    if kind == "win":
        return Counter({**{p: 1 for p in line}, _choose(rng, outside): 2})
    if kind == "blocked":
        plan = Counter({p: 1 for p in line})
        plan[_choose(rng, line)] = 3
        return plan
    if kind == "diagonal":
        # Also valid on rectangular boards: a partial diagonal never wins.
        flip = bool(rng.integers(2))
        return Counter({(i, cols - 1 - i if flip else i): 1 for i in range(min(rows, cols))})
    if kind == "cross":
        r, c = int(rng.integers(rows)), int(rng.integers(cols))
        return Counter({p: 1 for p in set([(r, j) for j in range(cols)] + [(i, c) for i in range(rows)])})
    if kind == "double":
        sites = list(map(int, rng.choice(rows * cols, 2, replace=False)))
        return Counter({divmod(sites[0], cols): 2, divmod(sites[1], cols): 3})
    if kind == "sparse":
        size = int(rng.integers(1, min(rows, cols)))
        return Counter({divmod(int(p), cols): 1 for p in rng.choice(rows * cols, size, replace=False)})
    if kind == "scatter":
        for _ in range(64):
            size = int(rng.integers(3, 7))
            plan = Counter(divmod(int(p), cols) for p in rng.choice(rows * cols, size, replace=True))
            if not _line(plan, rows, cols):
                return plan
        return Counter({(0, 0): 2, (rows - 1, cols - 1): 2})
    raise ValueError("unknown plan")


def _commands(rng, plan, rows, cols, pool, require_wrap):
    result = []
    for target, multiplicity in plan.items():
        for i in rng.choice(len(pool), multiplicity, replace=False):
            direction = pool[int(i)]
            origin = ((target[0] - direction[0]) % rows, (target[1] - direction[1]) % cols)
            result.append((origin, direction))
    rng.shuffle(result)
    if require_wrap and not any(not (0 <= p[0] + d[0] < rows and 0 <= p[1] + d[1] < cols) for p, d in result):
        return None
    return result


def _board(rng, rows, cols, gh, gw, roles, referenced):
    bg, _, _, _ = roles
    available = [c for c in range(10) if c not in roles]
    inks = list(map(int, rng.choice(available, int(rng.integers(2, 7)), replace=False)))
    masks = _mask_bank(gh, gw)
    # Assign a non-first referenced cell a shape alias and a color alias earlier
    # in row-major board order. This supplies visible exact-identity distractors.
    anchor = max(referenced)
    earlier = [p for p in itertools.product(range(rows), range(cols)) if p < anchor]
    if len(earlier) < 2:
        return None
    alias_shape, alias_color = earlier[:2]
    first = int(rng.integers(len(masks)))
    second = (first + int(rng.integers(1, len(masks)))) % len(masks)
    assigned = {anchor: (first, inks[0]), alias_shape: (first, inks[1]), alias_color: (second, inks[0])}
    used = set(assigned.values())
    vocabulary = [(i, ink) for i in range(len(masks)) for ink in inks if (i, ink) not in used]
    rng.shuffle(vocabulary)
    tokens = {}
    for p in itertools.product(range(rows), range(cols)):
        i, ink = assigned[p] if p in assigned else vocabulary.pop()
        token = np.full((gh, gw), bg, np.int8)
        token[masks[i]] = ink
        tokens[p] = token
    return tokens


def _make_input(rng, shape, roles, tokens, commands):
    rows, cols, gh, gw = shape
    bg, single, collision, divider = roles
    sh, sw, card_width = gh + 2, gw + 2, gw + 5
    minimum_cols = (len(commands) + 1) // 2
    card_cols = int(rng.integers(max(1, minimum_cols), 4))
    card_rows = (len(commands) + card_cols - 1) // card_cols
    separator = 1 + 4 * card_rows
    top = separator + 1
    bh, bw = rows * sh + 2, cols * sw + 2
    width = max(bw + 2, card_cols * card_width + 2) + int(rng.integers(3))
    grid = np.full((top + bh + 1, width), bg, np.int8)
    assert max(grid.shape) <= 30
    grid[0, 1], grid[0, 3:5] = single, collision
    grid[separator] = divider
    left = int(rng.integers(1, width - bw))
    bottom, right = top + bh - 1, left + bw - 1
    grid[top, left:right + 1] = divider
    grid[bottom, left:right + 1] = divider
    grid[top:bottom + 1, left] = divider
    grid[top:bottom + 1, right] = divider
    for (r, c), token in tokens.items():
        grid[top + 2 + r * sh:top + 2 + r * sh + gh, left + 2 + c * sw:left + 2 + c * sw + gw] = token
    occupied = list(map(int, rng.choice(card_rows * card_cols, len(commands), replace=False)))
    for slot, (origin, direction) in zip(occupied, commands):
        r, c = 1 + (slot // card_cols) * 4, 1 + (slot % card_cols) * card_width
        grid[r + (3 - gh) // 2:r + (3 - gh) // 2 + gh, c:c + gw] = tokens[origin]
        block = grid[r:r + 3, c + gw + 1:c + gw + 4]
        block[_arrow_mask(direction)] = single
    return grid


def _pair(rng, shape, roles, kind, axis, pool, require_wrap, orientation, essential):
    rows, cols, gh, gw = shape
    for _ in range(160):
        plan = _target_plan(rng, rows, cols, kind, axis)
        if sum(plan.values()) > 6:
            continue
        commands = _commands(rng, plan, rows, cols, pool, require_wrap)
        if commands is None:
            continue
        tokens = _board(rng, rows, cols, gh, gw, roles, {p for p, _ in commands})
        if tokens is None:
            continue
        inp = _make_input(rng, shape, roles, tokens, commands)
        info = _parse_canonical(inp)
        counts = _counts(info)
        assert counts == plan
        out = _render(info, counts)
        # Essential examples provide actual answer differences for ablations.
        if essential and any(np.array_equal(out, _render(info, _counts(info, **variant)))
                             for variant in ({"move": False}, {"lookup": "shape"},
                                             {"lookup": "color"}, {"wrap": False})):
            continue
        return {"input": _orient(inp, orientation), "output": _orient(out, orientation)}
    raise RuntimeError("bounded construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " outside allowed range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 4, replace=False)))
    shapes = [(r, c, h, w) for r, c in itertools.product((3, 4), repeat=2)
              for h, w in itertools.product((2, 3), repeat=2) if r * (h + 2) <= 16]
    shape = _choose(rng, shapes)
    orientation = int(rng.integers(8))
    axis = _choose(rng, ("row", "col"))
    opposite = "col" if axis == "row" else "row"
    schedule = [("win", axis, CARDINAL, True), ("blocked", axis, DIRECTIONS, True)]
    if nt >= 3:
        schedule.append(("win", opposite, DIRECTIONS, True))
    if nt >= 4:
        schedule.append(("diagonal", axis, DIRECTIONS, False))
    while len(schedule) < nt:
        kind = _choose(rng, ("win", "blocked", "scatter", "sparse", "double", "cross"))
        if kind == "cross" and shape[0] + shape[1] - 1 > 6:
            kind = "scatter"
        schedule.append((kind, _choose(rng, (axis, opposite)), DIRECTIONS, False))
    test_specs = []
    for _ in range(nq):
        kinds = (("win", "blocked", "scatter", "double", "sparse") if nt >= 4
                 else ("win", "blocked", "double", "sparse"))
        kind = _choose(rng, kinds)
        test_specs.append((kind, _choose(rng, (axis, opposite)) if nt >= 3 else axis, DIRECTIONS, False))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, specs in (("train", schedule), ("test", test_specs)):
        for kind, local_axis, pool, essential in specs:
            for _ in range(12):
                pair = _pair(rng, shape, roles, kind, local_axis, pool, essential,
                             orientation, essential)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    counts = _counts(info)
    wrap = sum(not (0 <= p[0] + d[0] < info["rows"] and 0 <= p[1] + d[1] < info["cols"])
               for p, d, _ in info["commands"])
    row_win = any(all(counts[r, c] == 1 for c in range(info["cols"])) for r in range(info["rows"]))
    col_win = any(all(counts[r, c] == 1 for r in range(info["rows"])) for c in range(info["cols"]))
    diagonals = [{(i, i) for i in range(min(info["rows"], info["cols"]))},
                 {(i, info["cols"] - 1 - i) for i in range(min(info["rows"], info["cols"]))}]
    diagonal_only = set(counts) in diagonals and all(n == 1 for n in counts.values())
    return {"shape": (info["rows"], info["cols"], info["gh"], info["gw"]),
            "commands": len(info["commands"]), "directions": tuple(sorted({d for _, d, _ in info["commands"]})),
            "wraps": wrap, "collisions": sum(n > 1 for n in counts.values()),
            "max_arrivals": max(counts.values()), "win": _line(counts, info["rows"], info["cols"]),
            "axes": tuple(axis for axis, yes in (("row", row_win), ("col", col_win)) if yes),
            "diagonal_only": diagonal_only,
            "blocked": _line(counts, info["rows"], info["cols"], any_hit=True) and not _line(counts, info["rows"], info["cols"]),
            "repeat_origin": len({p for p, _, _ in info["commands"]}) < len(info["commands"]),
            "arrivals": tuple(sorted(counts.items())), "tokens": tuple(_key(t) for t in info["tokens"].values())}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, conventions, features, infos = set(), set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            conventions.add(tuple(info[k] for k in ("bg", "single", "collision", "divider")))
            expected = _render(info, _counts(info))
            assert np.array_equal(_orient(out, info["transform"]), expected), "incorrect simultaneous output"
            assert not np.array_equal(inp, out)
            # Ink, all cards, the legend, scaffold and untouched slots are copied.
            canon_out = _orient(out, info["transform"])
            assert np.array_equal(canon_out[:info["top"]], info["grid"][:info["top"]])
            for (r, c), token in info["tokens"].items():
                block = canon_out[info["top"] + 2 + r * info["sh"]:info["top"] + 2 + r * info["sh"] + info["gh"],
                                  info["left"] + 2 + c * info["sw"]:info["left"] + 2 + c * info["sw"] + info["gw"]]
                assert np.array_equal(block[token != info["bg"]], token[token != info["bg"]])
            infos.append(info)
            features.append(_features(info))
    assert len(conventions) == 1
    train = features[:len(puzzle["train"])]
    assert any(f["win"] for f in train) and any(f["blocked"] for f in train)
    assert any(f["max_arrivals"] >= 3 for f in train)
    assert any(f["wraps"] for f in train)
    for variant in ({"move": False}, {"lookup": "shape"}, {"lookup": "color"},
                    {"lookup": "ordinal"}, {"wrap": False}):
        assert any(not np.array_equal(_render(i, _counts(i)), _render(i, _counts(i, **variant)))
                   for i in infos[:len(train)]), "missing counterexample: " + str(variant)
    for f in features[len(train):]:
        if f["win"]:
            assert any(t["win"] for t in train)
        assert set(f["axes"]) <= {a for t in train for a in t["axes"]}, "winning axis not taught"
        if f["diagonal_only"]:
            assert any(t["diagonal_only"] for t in train), "diagonal failure not taught"


def _semantic_regressions():
    """Concrete input fixtures with declared logical answers, not sampled plans."""
    roles = (0, 1, 2, 3)
    masks = _mask_bank(2, 2)
    vocabulary = [(0, 4), (0, 5), (1, 4)] + [(i, 6) for i in range(6)]
    tokens = {}
    for index, (mask, ink) in enumerate(vocabulary):
        token = np.zeros((2, 2), np.int8)
        token[masks[mask]] = ink
        tokens[divmod(index, 3)] = token

    commands = [((2, 0), (1, 0)), ((0, 0), (0, 1)), ((0, 1), (0, 1))]
    def fixture(instructions):
        inp = _make_input(np.random.default_rng(2026), (3, 3, 2, 2), roles, tokens, instructions)
        return inp, _parse_canonical(inp), solve(inp)

    inp, info, out = fixture(commands)
    assert _counts(info) == Counter({(0, 0): 1, (0, 1): 1, (0, 2): 1})
    assert np.all(out[-1] == 1)
    # Every padded slot in row 0 has its background painted; other rows don't.
    for r, c in itertools.product(range(3), repeat=2):
        rs = slice(info["top"] + 1 + r * 4, info["top"] + 1 + (r + 1) * 4)
        cs = slice(info["left"] + 1 + c * 4, info["left"] + 1 + (c + 1) * 4)
        assert np.all(out[rs, cs][inp[rs, cs] == 0] == (1 if r == 0 else 0))
    assert not np.array_equal(out, _render(info, _counts(info, move=False)))
    assert not np.array_equal(out, _render(info, _counts(info, wrap=False)))
    assert not np.array_equal(out, _render(info, _counts(info, lookup="shape")))
    assert not np.array_equal(out, _render(info, _counts(info, lookup="color")))

    _, jam_info, jam_out = fixture(commands + [((1, 1), (-1, 0))])
    assert _counts(jam_info)[0, 1] == 2 and np.all(jam_out[-1] == 2)
    assert np.all(_render(jam_info, _counts(jam_info), any_hit=True)[-1] == 1)
    _, triple_info, triple_out = fixture(commands + [((1, 1), (-1, 0)), ((1, 0), (-1, 1))])
    assert _counts(triple_info)[0, 1] == 3 and np.all(triple_out[-1] == 2)
    assert np.all(jam_out[jam_info["top"] + 1, jam_info["left"] + 5:jam_info["left"] + 9] == 2)
    assert np.all(triple_out[triple_info["top"] + 1, triple_info["left"] + 5:triple_info["left"] + 9] == 2)

    _, extra_info, extra_out = fixture(commands + [((1, 0), (0, 1)), ((1, 2), (0, -1))])
    assert _counts(extra_info)[1, 1] == 2 and np.all(extra_out[-1] == 1)
    _, reordered_info, reordered = fixture(commands[::-1])
    assert np.array_equal(reordered[reordered_info["top"]:], out[info["top"]:])
    # Move an address's board glyph without changing any command-card pixels.
    moved = inp.copy()
    a = (slice(info["top"] + 10, info["top"] + 12), slice(info["left"] + 2, info["left"] + 4))
    b = (slice(info["top"] + 6, info["top"] + 8), slice(info["left"] + 2, info["left"] + 4))
    moved[a], moved[b] = inp[b], inp[a]
    assert np.array_equal(moved[:info["top"]], inp[:info["top"]])
    assert np.all(solve(moved)[-1] == 2)
    _, diag_info, diag_out = fixture([((2, 2), (1, 1)), ((1, 0), (0, 1)), ((2, 1), (0, 1))])
    assert set(_counts(diag_info)) == {(0, 0), (1, 1), (2, 2)}
    assert np.all(diag_out[-1] == 2)

    # A valid larger-glyph board is supported even though ordinary generation
    # reserves height for two command rows and therefore omits this geometry.
    large = {}
    for index, mask in enumerate(_mask_bank(3, 3)[:16]):
        token = np.zeros((3, 3), np.int8)
        token[mask] = 4
        large[divmod(index, 4)] = token
    large_inp = _make_input(np.random.default_rng(7), (4, 4, 3, 3), roles, large, [((3, 3), (1, 0))])
    large_info = _parse_canonical(large_inp)
    assert _counts(large_info) == Counter({(0, 3): 1})
    assert np.all(solve(large_inp)[-1] == 2)


def self_test():
    """Bounded 42-seed semantic diversity checks, symmetries and damage tests."""
    _semantic_regressions()
    records, keys, palettes, inputs, token_maps, plans = [], set(), set(), set(), set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        repeated = generate_puzzle(seed)
        signature = tuple((_key(p["input"]), _key(p["output"])) for s in puzzle for p in puzzle[s])
        assert signature == tuple((_key(p["input"]), _key(p["output"])) for s in repeated for p in repeated[s])
        keys.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"].copy()
                assert np.array_equal(solve(inp), pair["output"])
                assert np.array_equal(inp, pair["input"]), "solver mutated input"
                info = _parse(inp)
                record = _features(info)
                records.append(record)
                palettes.add(tuple(info[k] for k in ("bg", "single", "collision", "divider")))
                inputs.add(inp.shape)
                token_maps.add(record["tokens"])
                plans.add((record["shape"][:2], record["arrivals"]))
                assert _counts(info) == _counts(info, info["commands"][::-1])
    assert len(keys) == 42 and len(palettes) >= 25
    assert len({f["shape"] for f in records}) == 12
    assert len(inputs) >= 20 and len(token_maps) >= 180 and len(plans) >= 100
    assert {d for f in records for d in f["directions"]} == set(DIRECTIONS)
    assert {f["commands"] for f in records} >= {1, 2, 3, 4, 5, 6}
    assert any(f["collisions"] >= 2 for f in records) and any(f["repeat_origin"] for f in records)
    assert any(f["win"] for f in records) and any(f["blocked"] for f in records)
    baseline = generate_puzzle(0)
    for t in range(8):
        transformed = {s: [{k: _orient(v, t) for k, v in p.items()} for p in baseline[s]] for s in baseline}
        validate_generated_puzzle(transformed)
    # All legal counts, rather than relying on default-count diversity.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(42 + nt * 10 + nq, nt, nq)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for defect in ("status", "ink", "duplicate"):
        broken = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in baseline[s]] for s in baseline}
        if defect == "duplicate":
            broken["train"][1] = {k: v.copy() for k, v in broken["train"][0].items()}
        else:
            pair = broken["test"][0]
            info = _parse(pair["input"])
            out = _orient(pair["output"], info["transform"])
            if defect == "status":
                out[-1, 0] = info["bg"]
            else:
                token = info["tokens"][0, 0]
                r, c = np.argwhere(token != info["bg"])[0]
                out[info["top"] + 2 + r, info["left"] + 2 + c] = info["bg"]
            pair["output"] = _unorient(out, info["transform"])
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("accepted " + defect)
    return {"seeds": len(HOST_SEEDS), "pairs": len(records), "board_geometries": len({f["shape"] for f in records}),
            "grid_shapes": len(inputs), "token_maps": len(token_maps), "arrival_patterns": len(plans),
            "role_palettes": len(palettes), "directions": len(DIRECTIONS)}
