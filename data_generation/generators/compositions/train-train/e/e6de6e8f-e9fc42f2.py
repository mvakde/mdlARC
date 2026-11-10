"""Colored seams on a path jig. Standard library and NumPy only."""
from collections import Counter
import json

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= s <= 30 for s in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(a, bg):
    remaining = {(int(r), int(c)) for r, c in zip(*np.nonzero(a != bg))}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], {}
        while stack:
            r, c = stack.pop()
            cells[r, c] = int(a[r, c])
            for dr, dc in DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _render(cells, bg):
    assert cells
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    h = max(r for r, _ in cells) - r0 + 1
    w = max(c for _, c in cells) - c0 + 1
    assert h <= 30 and w <= 30
    a = np.full((h, w), bg, dtype=np.int8)
    for (r, c), color in cells.items():
        a[r - r0, c - c0] = color
    return a


def _stroke_targets(commands):
    r = c = 0
    result = []
    for command in commands:
        first = r + 1, c
        if command == "S":
            r += 2
        elif command in ("L", "R"):
            r += 1
            c += 1 if command == "R" else -1
        else:
            raise AssertionError("unknown command")
        result.append((first, (r, c)))
    return result


def _parse_tag(cells, bg, body):
    a = _render(cells, bg)
    assert a.shape[0] == 4 and a.shape[1] >= 7
    assert np.all(a[0] == body) and np.all(a[-1] == body)
    assert np.all(a[:, 0] == body) and np.all(a[:, -1] == body)
    inside = a[1:3, 1:-1]
    assert not np.any(inside == body)
    active = np.any(inside != bg, axis=0)
    assert active[0] and active[-1]
    commands, colors, gaps = [], [], []
    c = 0
    while c < inside.shape[1]:
        start = c
        while c < inside.shape[1] and active[c]:
            c += 1
        glyph = inside[:, start:c]
        foreground = set(int(x) for x in glyph[glyph != bg])
        assert len(foreground) == 1
        color = foreground.pop()
        mask = glyph != bg
        if mask.shape == (2, 1) and mask.all():
            command = "S"
        elif mask.shape == (2, 2) and np.array_equal(mask, [[1, 0], [1, 1]]):
            command = "R"
        elif mask.shape == (2, 2) and np.array_equal(mask, [[0, 1], [1, 1]]):
            command = "L"
        else:
            raise AssertionError("invalid tag glyph")
        commands.append(command)
        colors.append(color)
        if c < inside.shape[1]:
            start = c
            while c < inside.shape[1] and not active[c]:
                c += 1
            assert c < inside.shape[1]
            gaps.append(c - start)
    assert 3 <= len(commands) <= 7 and len(set(colors)) == len(colors)
    return tuple(commands), tuple(colors), tuple(gaps)


def _analyze(input_grid):
    """Reconstruct an arbitrary valid member from its input alone."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    assert all(int(a[r, c]) == bg for r, c in
               ((0, 0), (0, a.shape[1] - 1), (a.shape[0] - 1, 0),
                (a.shape[0] - 1, a.shape[1] - 1)))
    histogram = Counter(int(x) for x in a[a != bg])
    assert histogram
    body, count = histogram.most_common(1)[0]
    assert sum(v == count for v in histogram.values()) == 1
    roots = [c for c, v in histogram.items() if c != body and v == 1]
    assert len(roots) == 1
    origin = roots[0]
    components = _components(a, bg)
    tags = [i for i, cells in enumerate(components)
            if any(n > 1 for c, n in Counter(cells.values()).items() if c != body)]
    assert len(tags) == 1
    tag = components.pop(tags[0])
    commands, colors, gaps = _parse_tag(tag, bg, body)
    assert origin not in colors and len(components) == len(commands) + 1
    occurrences = {color: [] for color in colors}
    root_piece = None
    root_pos = None
    for i, cells in enumerate(components):
        assert sum(c == body for c in cells.values()) >= 2
        for q, color in cells.items():
            if color == origin:
                assert root_piece is None
                root_piece, root_pos = i, q
            elif color != body:
                assert color in occurrences
                occurrences[color].append((i, q))
    assert root_piece is not None
    for entries in occurrences.values():
        assert len(entries) == 2 and entries[0][0] != entries[1][0]
    targets = _stroke_targets(commands)
    assignments = {i: [] for i in range(len(components))}
    assignments[root_piece].append((root_pos, (0, 0)))
    order = [root_piece]
    current = root_piece
    for color, (first, second) in zip(colors, targets):
        entries = occurrences[color]
        here = [item for item in entries if item[0] == current]
        assert len(here) == 1
        other = next(item for item in entries if item[0] != current)
        assignments[current].append((here[0][1], first))
        assignments[other[0]].append((other[1], second))
        current = other[0]
        assert current not in order
        order.append(current)
    assert len(order) == len(components)
    placed, all_cells = [], {}
    for i, cells in enumerate(components):
        links = assignments[i]
        p, q = links[0]
        delta = q[0] - p[0], q[1] - p[1]
        assert all((v[0] - u[0], v[1] - u[1]) == delta for u, v in links)
        port_count = sum(v not in (body, origin) for v in cells.values())
        assert port_count == (1 if i in (order[0], order[-1]) else 2)
        moved = {(r + delta[0], c + delta[1]): v for (r, c), v in cells.items()}
        assert not (all_cells.keys() & moved.keys())
        all_cells.update(moved)
        placed.append(moved)
    expected = _render(all_cells, bg)
    # These are boundary mating cells, not two markers buried inside a piece.
    for color, (p, q) in zip(colors, targets):
        assert all_cells[p] == all_cells[q] == color
        owners = [i for i, cells in enumerate(placed) if p in cells or q in cells]
        assert len(owners) == 2
        assert abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1
    return {"output": expected, "bg": bg, "body": body, "origin": origin,
            "commands": commands, "colors": colors, "gaps": gaps,
            "pieces": components, "placed": placed, "order": order,
            "targets": targets, "tag": tag, "cells": all_cells}


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _analyze(input_grid)["output"]


def _alternative(placed, terminal, last_targets, bg, expected):
    """Witness that B's matching/translation constraints alone are insufficient."""
    first, second = last_targets
    rest = {q: v for i, cells in enumerate(placed) if i != terminal for q, v in cells.items()}
    actual = second[0] - first[0], second[1] - first[1]
    for direction in ((0, 1), (1, 0), (0, -1)):
        if direction == actual:
            continue
        delta = first[0] + direction[0] - second[0], first[1] + direction[1] - second[1]
        shifted = {(r + delta[0], c + delta[1]): v
                   for (r, c), v in placed[terminal].items()}
        if rest.keys() & shifted.keys():
            continue
        union = dict(rest)
        union.update(shifted)
        try:
            alternative = _render(union, bg)
        except AssertionError:
            continue
        if not np.array_equal(alternative, expected):
            return direction, alternative
    return None


def _tag(commands, colors, rng, bg, body):
    gaps = [int(x) for x in rng.integers(1, 4, size=len(commands) - 1)]
    token_width = sum(1 if c == "S" else 2 for c in commands)
    while token_width + sum(gaps) + 2 > 28:
        choices = [i for i, gap in enumerate(gaps) if gap > 1]
        gaps[int(rng.choice(choices))] -= 1
    a = np.full((4, token_width + sum(gaps) + 2), bg, dtype=np.int8)
    a[0] = a[-1] = body
    a[:, 0] = a[:, -1] = body
    col = 1
    for i, (command, color) in enumerate(zip(commands, colors)):
        if command == "S":
            a[1:3, col] = color
            col += 1
        else:
            a[2, col:col + 2] = color
            a[1, col + (command == "L")] = color
            col += 2
        if i < len(gaps):
            col += gaps[i]
    return {(int(r), int(c)): int(a[r, c]) for r, c in zip(*np.nonzero(a != bg))}


def _grow(commands, colors, origin, body, bg, rng):
    targets = _stroke_targets(commands)
    pieces = [{} for _ in range(len(commands) + 1)]
    pieces[0][0, 0] = origin
    for i, (color, (first, second)) in enumerate(zip(colors, targets)):
        pieces[i][first] = color
        pieces[i + 1][second] = color
    occupied = {q for cells in pieces for q in cells}
    first, second = targets[-1]
    actual = second[0] - first[0], second[1] - first[1]
    alternate_dirs = [d for d in ((0, 1), (1, 0), (0, -1)) if d != actual and
                      (first[0] + d[0], first[1] + d[1]) not in occupied]
    if not alternate_dirs:
        return None
    d = alternate_dirs[int(rng.integers(len(alternate_dirs)))]
    reserved = {(first[0] + d[0], first[1] + d[1]),
                (second[0] - d[0], second[1] - d[1])} - occupied
    # Grow masks around fixed ports, never alter or recolor the skeleton.
    for index in rng.permutation(len(pieces)):
        index = int(index)
        cells = pieces[index]
        rlo, rhi = min(r for r, _ in cells) - 3, max(r for r, _ in cells) + 3
        clo, chi = min(c for _, c in cells) - 3, max(c for _, c in cells) + 3
        amount = int(rng.integers(2, 6 if index == len(pieces) - 1 else 11))
        for _ in range(amount):
            choices = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in DIRS
                              if rlo <= r + dr <= rhi and clo <= c + dc <= chi}
                             - occupied - reserved)
            if not choices:
                return None
            q = choices[int(rng.integers(len(choices)))]
            cells[q] = body
            occupied.add(q)
    union = {q: v for cells in pieces for q, v in cells.items()}
    out = _render(union, bg)
    if _alternative(pieces, len(pieces) - 1, targets[-1], bg, out) is None:
        return None
    return pieces, out


def _scatter(tag, pieces, rng, bg):
    objects = [tag] + pieces
    local = []
    for cells in objects:
        r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
        local.append({(r - r0, c - c0): v for (r, c), v in cells.items()})
    tag_width = max(c for _, c in local[0]) + 1
    area = sum(len(cells) for cells in local)
    base_h = max(14, int(np.sqrt(area * 2.6)) + int(rng.integers(0, 5)))
    base_w = max(tag_width + 2, int(np.sqrt(area * 2.8)) + int(rng.integers(0, 6)))
    for attempt in range(24):
        h = min(30, base_h + attempt // 4)
        w = min(30, base_w + attempt // 4)
        occupied, placements = set(), {}
        corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
        order = [0] + [int(x) for x in rng.permutation(np.arange(1, len(local)))]
        for i in order:
            cells = local[i]
            ph = max(r for r, _ in cells) + 1
            pw = max(c for _, c in cells) + 1
            if ph > h or pw > w:
                break
            options = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
            rng.shuffle(options)
            for r, c in options:
                shifted = {(r + dr, c + dc) for dr, dc in cells}
                if shifted & corners or any((x + dr, y + dc) in occupied
                                            for x, y in shifted for dr, dc in ((0, 0),) + DIRS):
                    continue
                occupied.update(shifted)
                placements[i] = r, c
                break
            else:
                break
        if len(placements) == len(local):
            a = np.full((h, w), bg, dtype=np.int8)
            for i, (r, c) in placements.items():
                for (dr, dc), v in local[i].items():
                    a[r + dr, c + dc] = v
            return a
    return None


def _commands(rng, length, anchor=None):
    if anchor == 0:
        values = list("SLR")
    elif anchor == 1:
        values = list("SSLR") + [str(rng.choice(["L", "R"]))]
    else:
        mode = int(rng.integers(6))
        if mode == 0:
            values = ["S"] * length
        elif mode == 1:
            values = [str(rng.choice(["L", "R"]))] * length
        elif mode == 2:
            hand = str(rng.choice(["L", "R"]))
            values = [str(rng.choice(["S", hand])) for _ in range(length)]
        elif mode == 3:
            values = ["L" if i % 2 else "R" for i in range(length)]
            return tuple(values)
        else:
            values = [str(x) for x in rng.choice(["S", "L", "R"], size=length)]
    rng.shuffle(values)
    return tuple(values)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    palette = [int(x) for x in rng.permutation(10)]
    bg, body, origin = palette[:3]
    puzzle, seen, train_lengths = {"train": [], "test": []}, set(), []
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train":
                length = 3 if index == 0 else 5 if index == 1 else int(rng.integers(3, 7))
                train_lengths.append(length)
            else:
                length = max(train_lengths) + 1 if index == 0 else int(rng.integers(5, 8))
            anchor = index if split == "train" and index < 2 else None
            commands = _commands(rng, length, anchor)
            colors = tuple(int(x) for x in rng.choice(palette[3:], size=length, replace=False))
            for _ in range(160):
                grown = _grow(commands, colors, origin, body, bg, rng)
                if grown is None:
                    continue
                pieces, out = grown
                inp = _scatter(_tag(commands, colors, rng, bg, body), pieces, rng, bg)
                if inp is None or _key(inp) in seen:
                    continue
                seen.add(_key(inp))
                puzzle[split].append({"input": inp, "output": out})
                break
            else:
                raise RuntimeError("could not grow and scatter a valid distinct assembly")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, roles, lengths, symbols = set(), set(), {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            assert np.array_equal(out, info["output"])
            roles.add((info["bg"], info["body"], info["origin"]))
            lengths[split].append(len(info["commands"]))
            if split == "train":
                symbols.update(info["commands"])
            else:
                assert set(info["commands"]) <= symbols
            source_counts = Counter(v for cells in info["pieces"] for v in cells.values())
            output_counts = Counter(int(v) for v in out[out != info["bg"]])
            assert source_counts == output_counts
            assert output_counts[info["origin"]] == 1
            assert all(output_counts[color] == 2 for color in info["colors"])
            assert _alternative(info["placed"], info["order"][-1], info["targets"][-1],
                                info["bg"], out) is not None
            without_bodies = np.array(out, copy=True)
            without_bodies[without_bodies == info["body"]] = info["bg"]
            assert not np.array_equal(without_bodies, out)
    assert len(roles) == 1 and symbols == {"S", "L", "R"}
    assert len(set(lengths["train"])) >= 2
    assert lengths["test"][0] == max(lengths["train"]) + 1


def _normalized(a):
    roles = {}
    return a.shape, tuple(roles.setdefault(int(v), len(roles)) for v in a.flat)


def self_test():
    signatures, roles, shapes, structures, masks, programs, gaps = set(), set(), set(), set(), set(), set(), set()
    modes, alternate_axes = set(), set()
    irregular = edge = tag_internal = beyond_skeleton = counterfactuals = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                info = _analyze(a)
                assert np.array_equal(solve(a), out)
                roles.add((info["bg"], info["body"], info["origin"]))
                shapes.add((a.shape, out.shape))
                structures.add(_normalized(out))
                programs.add(info["commands"])
                gaps.update(info["gaps"])
                modes.add(tuple(sorted(set(info["commands"]))))
                alternative = _alternative(info["placed"], info["order"][-1], info["targets"][-1], info["bg"], out)
                alternate_axes.add(0 if alternative[0][0] else 1)
                edge += any(np.any(side != info["bg"]) for side in (a[0], a[-1], a[:, 0], a[:, -1]))
                tag_internal += min(r for r, _ in info["tag"]) > 0 and max(r for r, _ in info["tag"]) < a.shape[0] - 1
                skeleton = {q: v for q, v in info["cells"].items() if v != info["body"]}
                beyond_skeleton += _render(skeleton, info["bg"]).shape != out.shape
                for piece in info["pieces"]:
                    local = _render(piece, info["bg"])
                    mask = local != info["bg"]
                    masks.add((mask.shape, mask.tobytes()))
                    irregular += not mask.all()
        # Keep every rigid piece unchanged; change just the last instruction.
        # Its alternative docking must be solvable under the very same rule.
        first = _analyze(puzzle["train"][0]["input"])
        direction, changed_output = _alternative(
            first["placed"], first["order"][-1], first["targets"][-1],
            first["bg"], first["output"])
        changed_commands = list(first["commands"])
        changed_commands[-1] = {(0, 1): "R", (1, 0): "S", (0, -1): "L"}[direction]
        crng = np.random.default_rng(9000 + seed)
        changed_input = _scatter(_tag(changed_commands, first["colors"], crng,
                                     first["bg"], first["body"]),
                                 first["pieces"], crng, first["bg"])
        assert changed_input is not None
        assert np.array_equal(solve(changed_input), changed_output)
        assert not np.array_equal(changed_output, first["output"])
        counterfactuals += 1
    assert len(signatures) == 42 and len(roles) >= 30
    assert len(shapes) >= 80 and len(structures) >= 180 and len(masks) >= 200
    assert len(programs) >= 70 and gaps == {1, 2, 3}
    assert {("S",), ("L",), ("R",), ("L", "R", "S")} <= modes
    assert alternate_axes == {0, 1}
    assert irregular >= 500 and edge >= 10 and tag_internal >= 100 and beyond_skeleton >= 100
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (0, 93841):
        left, right = generate_puzzle(seed), generate_puzzle(seed)
        assert all(_key(a[k]) == _key(b[k]) for s in ("train", "test")
                   for a, b in zip(left[s], right[s]) for k in ("input", "output"))
    # Explicitly refute count-only and handedness-only simplifications.
    assert _stroke_targets("SLR") != _stroke_targets("SRL")
    assert _stroke_targets("RSL") != _stroke_targets("RLS")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "palettes": len(roles),
            "shape_pairs": len(shapes), "normalized_outputs": len(structures),
            "piece_masks": len(masks), "programs": len(programs),
            "irregular_pieces": irregular, "edge_touching_inputs": edge,
            "interior_tags": tag_internal, "crops_extended_by_bodies": beyond_skeleton,
            "alternate_docking_axes": sorted(alternate_axes),
            "changed_glyph_same_piece_checks": counterfactuals}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
