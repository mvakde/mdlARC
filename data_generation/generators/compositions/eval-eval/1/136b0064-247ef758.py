"""Cut-and-print stroke programs. Only the standard library and NumPy are used.

Frame-colored beads are local command origins. A complete color address both
prints and cuts the pending path. The solver never reads generator parameters.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


# A command's first pixel is immediately below its bead. Further pixels follow
# this direction; the last pixel becomes the chained endpoint.
CATALOG = tuple((dr, dc, n) for dr, dc in
                ((0, -1), (0, 1), (1, 0), (1, -1), (1, 1)) for n in (2, 3))


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _counts(rng, num_train, num_test):
    # Explicit defaults are 4/1; None also supports the local inspection tool.
    a = None if num_train is None else _integer(num_train, "num_train", 2, 10)
    b = None if num_test is None else _integer(num_test, "num_test", 1, 4)
    if b is None:
        b = int(rng.integers(1, min(4, a - 1) + 1 if a is not None else 5))
    if a is None:
        a = int(rng.integers(b + 1, 11))
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _pixels(command):
    dr, dc, n = command
    return tuple((1 + dr * k, dc * k) for k in range(n))


def _read(input_grid):
    """Identify frame, beads, miniature strokes, and paired coordinates."""
    inp = _grid(input_grid)
    h, w = inp.shape
    candidates = []
    for d in range(5, w - 8):
        if not np.all(inp[:, d] == inp[0, d]):
            continue
        field = inp[:, d + 1:]
        frame = int(field[0, 0])
        if not all(int(field[r, c]) == frame
                   for r, c in ((0, -1), (-1, 0), (-1, -1))):
            continue
        interior = field[1:-1, 1:-1]
        if not interior.size or not np.all(interior == interior[0, 0]):
            continue
        bg, divider = int(interior[0, 0]), int(inp[0, d])
        if len({bg, divider, frame}) != 3:
            continue
        stage = inp[:, :d]
        beads = np.argwhere(stage == frame)
        if len(beads) < 2 or np.any(stage == divider):
            continue
        ys = sorted(set(map(int, beads[:, 0])))
        xs = sorted(set(map(int, beads[:, 1])))
        if (ys[0] != 0 or xs[0] != 2 or len(ys) < 2
                or len(beads) != len(xs) * len(ys)
                or d != xs[-1] + 3 or ys[-1] + 4 > h):
            continue
        if (any(b - a not in (5, 6) for a, b in zip(ys, ys[1:]))
                or any(b - a not in (6, 7) for a, b in zip(xs, xs[1:]))):
            continue
        expected = np.full_like(stage, bg)
        tokens = []
        valid = True
        for x in xs:
            for y in ys:
                tile = stage[y:y + 4, x - 2:x + 3]
                ink = set(map(int, np.unique(tile))) - {bg, frame}
                if len(ink) != 1 or int(tile[0, 2]) != frame:
                    valid = False
                    break
                color = ink.pop()
                positions = set((int(r), int(c) - 2)
                                for r, c in np.argwhere(tile == color))
                matches = [cmd for cmd in CATALOG if positions == set(_pixels(cmd))]
                if len(matches) != 1:
                    valid = False
                    break
                cmd = matches[0]
                expected[y, x] = frame
                for dy, dx in _pixels(cmd):
                    expected[y + dy, x + dx] = color
                tokens.append((cmd, color))
            if not valid:
                break
        if not valid or not np.array_equal(expected, stage):
            continue
        if (not np.array_equal(field[0, 1:-1], field[-1, 1:-1])
                or not np.array_equal(field[1:-1, 0], field[1:-1, -1])):
            continue
        border = np.concatenate((field[0], field[-1], field[:, 0], field[:, -1]))
        if np.any((border == bg) | (border == divider)):
            continue
        rows, cols = {}, {}
        for r in range(1, h - 1):
            c = int(field[r, 0])
            if c != frame:
                rows.setdefault(c, []).append(r)
        for c in range(1, field.shape[1] - 1):
            color = int(field[0, c])
            if color != frame:
                cols.setdefault(color, []).append(c)
        candidates.append(dict(inp=inp, d=d, bg=bg, frame=frame,
                               divider=divider, field=field.copy(), tokens=tokens,
                               bands=len(ys), lanes=len(xs), rows=rows, cols=cols))
    if len(candidates) != 1:
        raise ValueError("absent or ambiguous program/frame layout")
    return candidates[0]


def _interpret(info, variant="correct"):
    """Input-only state machine; points are relative to the *final endpoint*."""
    tokens = list(info["tokens"])
    if variant == "row_major":
        b, l = info["bands"], info["lanes"]
        tokens = [tokens[x * b + y] for y in range(b) for x in range(l)]
    pending = {(0, 0)}
    er = ec = 0
    prints = []
    for index, (command, color) in enumerate(tokens):
        active = bool(info["rows"].get(color) and info["cols"].get(color))
        if variant == "skip_inactive" and not active:
            continue
        if variant == "each_glyph":
            pending, er, ec = {(0, 0)}, 0, 0
        for dy, dx in _pixels(command):
            pending.add((er + dy, ec + dx))
        dy, dx = _pixels(command)[-1]
        er, ec = er + dy, ec + dx
        if active:
            offsets = {(r - er, c - ec) for r, c in pending}
            if variant == "box_fill":
                offsets = {(r, c) for r in range(min(r for r, c in offsets), 1)
                           for c in range(min(c for r, c in offsets),
                                          max(c for r, c in offsets) + 1)}
            if variant == "bbox_anchor":
                # A tempting inherited B convention, rounded down for even boxes.
                cr = (min(r for r, c in offsets) + max(r for r, c in offsets)) // 2
                cc = (min(c for r, c in offsets) + max(c for r, c in offsets)) // 2
                offsets = {(r - cr, c - cc) for r, c in offsets}
            prints.append((color, offsets, index))
            if variant != "no_cut":
                pending, er, ec = {(0, 0)}, 0, 0
    if variant == "correct" and pending != {(0, 0)}:
        raise ValueError("the last command must have a complete address")
    return prints


def _paint(info, prints, variant="correct", strict=True):
    out = info["field"].copy()
    if variant == "reverse_priority":
        prints = list(reversed(prints))
    elif variant == "destination_priority":
        prints = sorted(prints, key=lambda t: (min(info["rows"][t[0]]),
                                               min(info["cols"][t[0]]), t[2]))
    occupied = set()
    for color, offsets, _ in prints:
        rows, cols = info["rows"][color], info["cols"][color]
        centers = list(zip(rows, cols)) if variant == "zip_axes" else [
            (r, c) for r in rows for c in cols]
        if variant == "single_copy":
            centers = centers[:1]
        for r, c in centers:
            for dy, dx in offsets:
                y, x = r + dy, c + dx
                if not (1 <= y < out.shape[0] - 1 and 1 <= x < out.shape[1] - 1):
                    if strict:
                        raise ValueError("printed path crosses the frame")
                    continue
                if (y, x) not in occupied:
                    out[y, x] = color
                    occupied.add((y, x))
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read(input_grid)
    return _paint(info, _interpret(info))


def _wrong(info, variant):
    interpret_variant = variant if variant in {
        "row_major", "skip_inactive", "each_glyph", "no_cut", "box_fill", "bbox_anchor"
    } else "correct"
    return _paint(info, _interpret(info, interpret_variant), variant, strict=False)


SHORTCUTS = ("row_major", "skip_inactive", "each_glyph", "no_cut", "box_fill",
             "bbox_anchor", "reverse_priority", "destination_priority", "zip_axes",
             "single_copy")


def _parts(rng, total):
    parts = []
    while total:
        n = int(rng.integers(1, min(3, total) + 1))
        parts.append(n)
        total -= n
    return parts


def _allocate_axis(rng, domains, desired, fixed):
    """Small finite matching problem: one border position cannot have two colors."""
    result = {c: list(fixed.get(c, [])) for c in domains}
    used = {v for values in result.values() for v in values}
    if len(used) != sum(map(len, result.values())):
        return None
    if any(v not in domains[c] for c in result for v in result[c]):
        return None
    order = sorted(domains, key=lambda c: (len(domains[c]), -desired[c]))

    def visit(index):
        if index == len(order):
            return True
        color = order[index]
        need = desired[color] - len(result[color])
        options = [v for v in domains[color] if v not in used]
        rng.shuffle(options)
        # Border dimensions <=30 and requests <=3. Search combinations lazily.
        from itertools import combinations
        for extra in combinations(options, need):
            result[color].extend(extra)
            used.update(extra)
            if visit(index + 1):
                return True
            used.difference_update(extra)
            if extra:
                del result[color][-len(extra):]
        return False

    return {c: sorted(v) for c, v in result.items()} if visit(0) else None


def _construct(rng, commands, parts, layout, witness=False, repeat_color=False):
    """Construct labels from records and answers by reverse-order painting.

    This does not use solve, _read, _interpret, or _paint to create labels.
    A failed placement returns None, without writing files or retaining grids.
    """
    lanes, bands = layout
    xgap = int(rng.integers(1, 3)) if lanes <= 2 else 1
    ygap = int(rng.integers(1, 3))
    sw = 5 * lanes + xgap * (lanes - 1)
    panel_h = 4 * bands + ygap * (bands - 1)
    h_low = max(panel_h, 18 if witness else 13)
    h = int(rng.integers(h_low, min(30, max(h_low + 3, 24)) + 1))
    fw_max = 29 - sw
    fw_min = min(fw_max, 14 if lanes <= 2 else 11)
    fw = int(rng.integers(fw_min, fw_max + 1))
    palette = list(map(int, rng.permutation(10)))
    bg, divider, frame = palette[:3]
    available = palette[3:]
    group_count = len(parts)
    active_count = 3 if witness else int(rng.integers(1, min(3, group_count) + 1))
    active_colors = available[:active_count]
    end_colors = [active_colors[j % active_count] for j in range(group_count)]
    if not witness:
        rng.shuffle(end_colors)
    if repeat_color and group_count >= 2:
        end_colors[-1] = end_colors[0]
        active_colors = list(dict.fromkeys(end_colors))
    inactive_count = sum(parts) - len(parts)
    rest = [c for c in available if c not in active_colors]
    n_inactive = min(3 if witness else int(rng.integers(1, 4)), inactive_count)
    inactive_colors = rest[:n_inactive]
    states = dict(zip(inactive_colors, ("N", "R", "C") if witness else
                      [str(rng.choice(["N", "R", "C"])) for _ in inactive_colors]))
    tokens, stamps, inactive_index = [], [], 0
    offset = 0
    for group_index, length in enumerate(parts):
        path = [(0, 0)]
        er = ec = 0
        for j in range(length):
            cmd = commands[offset]
            if j == length - 1:
                color = end_colors[group_index]
            else:
                color = inactive_colors[inactive_index % len(inactive_colors)]
                inactive_index += 1
            tokens.append((cmd, color))
            dr, dc, n = cmd
            # Deliberately separate construction arithmetic from the interpreter.
            for k in range(n):
                path.append((er + 1 + dr * k, ec + dc * k))
            er += 1 + dr * (n - 1)
            ec += dc * (n - 1)
            offset += 1
        mask = {(r - er, c - ec) for r, c in path}
        stamps.append((end_colors[group_index], mask, offset - 1))
    # Each occurrence of a color uses the same visible row/column sets. Domains
    # must fit the union of ALL that color's (possibly different) batch masks.
    masks = {c: set().union(*(s for color, s, _ in stamps if color == c))
             for c in active_colors}
    row_domains, col_domains = {}, {}
    for color, mask in masks.items():
        row_domains[color] = list(range(1 - min(r for r, c in mask), h - 1))
        col_domains[color] = list(range(1 - min(c for r, c in mask),
                                        fw - 1 - max(c for r, c in mask)))
    for color, state in states.items():
        if state == "R":
            row_domains[color] = list(range(1, h - 1))
        elif state == "C":
            col_domains[color] = list(range(1, fw - 1))
    requests = {}
    for color in active_colors:
        requests[color] = (int(rng.choice([1, 2, 3], p=[.45, .45, .10])),
                           int(rng.choice([1, 2, 3], p=[.45, .45, .10])))
    if witness:
        requests[active_colors[0]] = (2, 2)
        requests[active_colors[1]] = (1, 1)
        requests[active_colors[2]] = (1, 1)
    row_fixed, col_fixed = {}, {}
    if witness:
        # Earlier stencil A covers later stencil B's endpoint. B lands above A,
        # so sorting by target row gives the opposite overlap priority.
        first, second = stamps[:2]
        options = [(r, c) for r, c in first[1] if r < 0 and c != 0]
        rng.shuffle(options)
        for dy, dx in options:
            centers = [(r, c) for r in row_domains[first[0]]
                       for c in col_domains[first[0]]
                       if r + dy in row_domains[second[0]]
                       and c + dx in col_domains[second[0]]]
            if centers:
                r, c = centers[int(rng.integers(len(centers)))]
                row_fixed = {first[0]: [r], second[0]: [r + dy]}
                col_fixed = {first[0]: [c], second[0]: [c + dx]}
                break
        else:
            return None
    rows = _allocate_axis(rng, row_domains,
                          {c: requests[c][0] if c in requests else 1 for c in row_domains},
                          row_fixed)
    cols = _allocate_axis(rng, col_domains,
                          {c: requests[c][1] if c in requests else 1 for c in col_domains},
                          col_fixed)
    if rows is None or cols is None:
        return None
    # Optional complete address with no command, using at most one spare color.
    unused = [c for c in available if c not in active_colors + inactive_colors]
    if unused and rng.random() < .35:
        free_r = sorted(set(range(1, h - 1)) - {r for rs in rows.values() for r in rs})
        free_c = sorted(set(range(1, fw - 1)) - {c for cs in cols.values() for c in cs})
        if free_r and free_c:
            rows[unused[0]] = [int(rng.choice(free_r))]
            cols[unused[0]] = [int(rng.choice(free_c))]
    inp = np.full((h, sw + 1 + fw), bg, dtype=np.int8)
    inp[:, sw] = divider
    field = inp[:, sw + 1:]
    field[[0, -1], :] = frame
    field[:, [0, -1]] = frame
    for color, rs in rows.items():
        for r in rs:
            field[r, 0] = field[r, -1] = color
    for color, cs in cols.items():
        for c in cs:
            field[0, c] = field[-1, c] = color
    for index, (cmd, color) in enumerate(tokens):
        lane, band = divmod(index, bands)
        y, x = band * (4 + ygap), 2 + lane * (5 + xgap)
        inp[y, x] = frame
        dr, dc, n = cmd
        for k in range(n):
            inp[y + 1 + dr * k, x + dc * k] = color
    out = field.copy()
    # Reverse painting makes the earlier print win without an occupancy sentinel.
    for color, mask, _ in reversed(stamps):
        for r in rows[color]:
            for c in cols[color]:
                for dy, dx in mask:
                    out[r + dy, c + dx] = color
    return {"input": inp, "output": out}


def _make_pair(rng, split, index, commands=None):
    witness = split == "train" and index == 0
    fixed_commands = commands
    for attempt in range(96):
        if fixed_commands is not None:
            commands = fixed_commands
            layout = (2, len(commands) // 2)
        else:
            layouts = [(1, 2), (1, 3), (2, 2), (2, 3)] if split == "train" else [
                (2, 4), (3, 3), (3, 4)]
            layout = layouts[int(rng.integers(len(layouts)))]
            commands = [CATALOG[int(rng.integers(len(CATALOG)))]
                        for _ in range(layout[0] * layout[1])]
        if witness:
            choices = ((2, 2, 2), (2, 3, 1), (3, 1, 2), (3, 2, 1))
            parts = list(choices[int(rng.integers(len(choices)))])
        elif fixed_commands is not None:
            parts = [3, 1] if rng.random() < .5 else [1, 3]
        else:
            parts = _parts(rng, len(commands))
        pair = _construct(rng, commands, parts, layout, witness,
                          repeat_color=not witness and rng.random() < .35)
        if pair is None:
            continue
        info = _read(pair["input"])
        if witness and not all(not np.array_equal(pair["output"], _wrong(info, name))
                               for name in SHORTCUTS):
            continue
        return pair
    raise RuntimeError("bounded placement search exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    rng = np.random.default_rng(seed)
    nt, nq = _counts(rng, num_train, num_test)
    # Every miniature direction/length is shown in the first two demonstrations.
    # A horizontal first command ensures the overlap witness has sideways extent.
    first = int(rng.integers(4))
    order = [first] + [int(i) for i in rng.permutation(10) if int(i) != first]
    vocabulary = [CATALOG[i] for i in order]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            commands = (vocabulary[:6] if index == 0 else vocabulary[6:]) if (
                split == "train" and index < 2) else None
            pair = _make_pair(rng, split, index, commands)
            # Twenty distinct recolorings exceed thirteen prior inputs.
            # This fallback only repairs duplicates; it never changes semantics.
            original = pair
            original_info = _read(original["input"])
            for shift in range(20):
                mapping = np.arange(10, dtype=np.int8)
                if shift >= 10:
                    a, b = original_info["bg"], original_info["divider"]
                    mapping[a], mapping[b] = b, a
                mapping = (mapping + shift % 10) % 10
                pair = {k: mapping[v] for k, v in original.items()}
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise AssertionError("distinct recolorings exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, lengths, train_commands = set(), {"train": [], "test": []}, set()
    witnesses = {name: False for name in SHORTCUTS}
    training_states = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _read(inp)
            prints = _interpret(info)
            assert prints and out.shape == info["field"].shape
            assert np.array_equal(out, _paint(info, prints)), "wrong print/cut/priority"
            assert np.array_equal(out[[0, -1]], info["field"][[0, -1]])
            assert np.array_equal(out[:, [0, -1]], info["field"][:, [0, -1]])
            assert np.any(out[1:-1, 1:-1] != info["bg"])
            # Monotone endpoint chaining has one unique origin and no erased ink.
            for _, offsets, _ in prints:
                assert (0, 0) in offsets and max(r for r, c in offsets) == 0
                assert min(r for r, c in offsets) < 0
                assert len(offsets) >= 3
            lengths[split].append(len(info["tokens"]))
            if split == "train":
                train_commands.update(cmd for cmd, color in info["tokens"])
                for color in {color for cmd, color in info["tokens"]}:
                    r, c = bool(info["rows"].get(color)), bool(info["cols"].get(color))
                    training_states.add("A" if r and c else "R" if r else "C" if c else "N")
                for name in SHORTCUTS:
                    witnesses[name] |= not np.array_equal(out, _wrong(info, name))
    assert train_commands == set(CATALOG), "unseen miniature stroke type"
    assert training_states == {"A", "R", "C", "N"}, "incomplete-address evidence missing"
    assert all(witnesses.values()), f"under-demonstrated shortcuts: {witnesses}"
    assert min(lengths["test"]) > max(lengths["train"]), "longer-program shift missing"


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _hand_regression():
    """A literal small example, independent of random generation and its labels."""
    inp = np.zeros((12, 24), dtype=np.int8)
    inp[:, 11] = 8
    field = inp[:, 12:]
    field[[0, -1], :] = 9
    field[:, [0, -1]] = 9
    for r in (5, 9):
        field[r, 0] = field[r, -1] = 1
    for c in (4, 8):
        field[0, c] = field[-1, c] = 1
    field[4, 0] = field[4, -1] = 2
    field[0, 5] = field[-1, 5] = 2
    field[7, 0] = field[7, -1] = 3  # Incomplete address: still a real command.
    for y, x, color, pixels in (
        (0, 2, 4, ((1, 0), (1, 1))),
        (5, 2, 1, ((1, 0), (2, 0))),
        (0, 8, 3, ((1, 0), (1, -1), (1, -2))),
        (5, 8, 2, ((1, 0), (2, 1))),
    ):
        inp[y, x] = 9
        for r, c in pixels:
            inp[y + r, x + c] = color
    # Manually enumerated placed pixels, including the original path origins.
    # Color 1 has a 2x2 Cartesian address. Color 2 loses (3,4) to earlier ink.
    expected = field.copy()
    for r, c in ((2, 3), (3, 3), (3, 4), (4, 4), (5, 4),
                 (2, 7), (3, 7), (3, 8), (4, 8), (5, 8),
                 (6, 3), (7, 3), (7, 4), (8, 4), (9, 4),
                 (6, 7), (7, 7), (7, 8), (8, 8), (9, 8)):
        expected[r, c] = 1
    for r, c in ((1, 6), (2, 6), (2, 5), (2, 4), (4, 5)):
        expected[r, c] = 2
    before = inp.copy()
    assert np.array_equal(solve(inp), expected)
    assert np.array_equal(inp, before)
    return inp, expected


def self_test():
    """Bounded, streaming checks; no files and no saved execution traces."""
    _hand_regression()
    shapes, programs, stencils, roles, products, dimensions = set(), set(), set(), set(), set(), set()
    features = Counter()
    signatures = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_fingerprint(p))
        for split in ("train", "test"):
            for pair in p[split]:
                info = _read(pair["input"])
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                shapes.add(pair["input"].shape)
                programs.add(tuple(cmd for cmd, color in info["tokens"]))
                roles.add((info["bg"], info["divider"], info["frame"]))
                dimensions.add((info["lanes"], info["bands"]))
                prints = _interpret(info)
                colors = [c for c, mask, index in prints]
                features["repeated_print_color"] += len(colors) != len(set(colors))
                features["black_ink"] += any(c == 0 for cmd, c in info["tokens"])
                for color, mask, index in prints:
                    stencils.add(tuple(sorted(mask)))
                    products.add((len(info["rows"][color]), len(info["cols"][color])))
                    features["sideways"] += any(c != 0 for r, c in mask)
                    features["transparent"] += len(mask) < (
                        (1 - min(r for r, c in mask)) *
                        (1 + max(c for r, c in mask) - min(c for r, c in mask)))
    assert len(signatures) == 42 and len(programs) > 100
    assert len(shapes) >= 30 and len(stencils) >= 100 and len(roles) >= 100
    assert {(1, 2), (2, 2), (2, 3), (2, 4), (3, 3), (3, 4)} <= dimensions
    assert {(1, 1), (2, 1), (1, 2), (2, 2)} <= products
    assert all(features[name] >= 10 for name in
               ("repeated_print_color", "black_ink", "sideways", "transparent"))
    # Determinism, global recoloring equivariance, and no in-place input mutation.
    p = generate_puzzle(343770293, 2, 1)
    assert _fingerprint(p) == _fingerprint(generate_puzzle(343770293, 2, 1))
    permutation = np.array([7, 0, 9, 3, 1, 4, 8, 2, 6, 5], dtype=np.int8)
    for pair in p["train"] + p["test"]:
        before = pair["input"].copy()
        assert np.array_equal(solve(permutation[before]), permutation[pair["output"]])
        assert np.array_equal(before, pair["input"])
    # Every legal explicit count combination. Only one puzzle is retained.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            q = generate_puzzle(1000 + 10 * nt + nq, nt, nq)
            assert len(q["train"]) == nt and len(q["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (4, 1.5)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Validator must reject the semantic alternatives, not just malformed arrays.
    for name in SHORTCUTS:
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]]
               for s in ("train", "test")}
        pair = bad["train"][0]
        pair["output"] = _wrong(_read(pair["input"]), name)
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"accepted {name} mutation")
    return {"seeds": 42, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "programs": len(programs),
            "stencils": len(stencils), "role_assignments": len(roles),
            "layouts": sorted(dimensions), "coordinate_products": sorted(products),
            "features": dict(features)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
