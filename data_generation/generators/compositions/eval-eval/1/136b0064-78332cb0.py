"""Branching brush relay: exact frontier signatures dispatch colored glyphs.

No parent generator, training examples, seed, or stored state enters solve().
All sampling and checks are bounded; this module writes no files.
"""

from itertools import combinations, permutations
import hashlib
import numpy as np


GLYPHS = {
    "L": ("010", "110", "010"),
    "R": ("010", "011", "010"),
    "F": ("010", "111", "101"),
}
MASKS = {name: np.array([[int(c) for c in row] for row in rows], dtype=bool)
         for name, rows in GLYPHS.items()}
CARD_HEIGHT = 6


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "integer grid required"
    assert np.all((0 <= a) & (a <= 9)), "grid palette"
    return a


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _advance(state, command):
    if command == "L":
        return tuple(c - 1 for c in state)
    if command == "R":
        return tuple(c + 1 for c in state)
    assert command == "F"
    return tuple(sorted({c + d for c in state for d in (-1, 1)}))


def _rung(state, command):
    offsets = {"L": (-1, 0), "R": (0, 1), "F": (-1, 0, 1)}[command]
    return sorted({c + d for c in state for d in offsets})


def _sequence_output(height, width, bg, marker, initial, commands, colors):
    """Construction-side rendering, also used for explicit shortcut hypotheses."""
    out = np.full((height, width), bg, dtype=np.int8)
    out[0, list(initial)] = marker
    state = tuple(initial)
    if 2 * len(commands) >= height:
        return None
    for i, (command, color) in enumerate(zip(commands, colors)):
        following = _advance(state, command)
        if following[0] < 0 or following[-1] >= width:
            return None
        out[2 * i + 1, _rung(state, command)] = color
        out[2 * i + 2, list(following)] = color
        state = following
    return out


def _parse(input_grid):
    a = _grid(input_grid)
    h, total_w = a.shape
    candidates = []
    for cut in range(6, total_w - 4):
        divider = int(a[0, cut])
        if not np.all(a[:, cut] == divider):
            continue
        pane = a[:, cut + 1:]
        if h < 7 or not np.all(pane[1:] == pane[1, 0]):
            continue
        bg = int(pane[1, 0])
        width = pane.shape[1]
        initial = tuple(map(int, np.flatnonzero(pane[0] != bg)))
        if not initial or bg == divider:
            continue
        markers = np.unique(pane[0, list(initial)])
        if len(markers) != 1 or int(markers[0]) in (bg, divider):
            continue
        marker = int(markers[0])
        if (cut + 1) % (width + 1):
            continue
        cols = (cut + 1) // (width + 1)
        if not 1 <= cols <= 3 or not 6 <= width <= 9:
            continue
        cards, bad = [], False
        glyph_left = (width - 3) // 2
        panel = a[:, :cut]
        for row in range((h + 1) // 7):
            for col in range(cols):
                top, left = 7 * row, (width + 1) * col
                block = panel[top:top + CARD_HEIGHT, left:left + width]
                if np.all(block == bg):
                    continue
                tag = tuple(map(int, np.flatnonzero(block[0] != bg)))
                glyph = block[2:5, glyph_left:glyph_left + 3]
                ink = np.unique(glyph[glyph != bg])
                names = [name for name, mask in MASKS.items()
                         if np.array_equal(glyph != bg, mask)]
                if (not tag or not np.all(block[0, list(tag)] == marker)
                        or len(ink) != 1 or int(ink[0]) in (bg, divider, marker)
                        or len(names) != 1):
                    bad = True
                    break
                expected = np.full(block.shape, bg, dtype=np.int8)
                expected[0, list(tag)] = marker
                expected[2:5, glyph_left:glyph_left + 3][MASKS[names[0]]] = ink[0]
                if not np.array_equal(block, expected):
                    bad = True
                    break
                cards.append(dict(tag=tag, command=names[0], color=int(ink[0]),
                                  row=row, col=col))
            if bad:
                break
        if bad or not 3 <= len(cards) <= 6:
            continue
        # Reconstruct the entire left region: reject hidden data, stray ink,
        # malformed separators, and blank rows inside the card array.
        rows = max(card["row"] for card in cards) + 1
        if {c["row"] for c in cards} != set(range(rows)):
            continue
        expected = np.full(panel.shape, bg, dtype=np.int8)
        used_h = rows * 7 - 1
        for row in range(rows - 1):
            expected[7 * row + 6] = divider
        for col in range(cols - 1):
            expected[:used_h, (width + 1) * col + width] = divider
        for card in cards:
            top, left = 7 * card["row"], (width + 1) * card["col"]
            expected[top, left + np.array(card["tag"])] = marker
            block = expected[top + 2:top + 5, left + glyph_left:left + glyph_left + 3]
            block[MASKS[card["command"]]] = card["color"]
        if not np.array_equal(panel, expected):
            continue
        if len({c["tag"] for c in cards}) != len(cards):
            continue
        candidates.append(dict(pane=pane.copy(), background=bg, marker=marker,
                               divider=divider, cards=cards, layout=(rows, cols),
                               width=width, height=h, initial=initial))
    assert len(candidates) == 1, "one visible card field and seed pane required"
    return candidates[0]


def _execute(info):
    """Dispatch from the current frontier, drawing each pen separately."""
    cards = info["cards"]
    lookup = {card["tag"]: i for i, card in enumerate(cards)}
    out = info["pane"].copy()
    state, used, states, order = info["initial"], set(), [info["initial"]], []
    for step in range(len(cards)):
        assert state in lookup, "relay halted with unused cards"
        index = lookup[state]
        assert index not in used, "cyclic relay"
        card = cards[index]
        used.add(index)
        order.append(index)
        r = 2 * step + 1
        assert r + 1 < out.shape[0], "drawing exceeds pane height"
        next_tips = set()
        directions = (-1, 1) if card["command"] == "F" else (
            (-1,) if card["command"] == "L" else (1,))
        for c in state:
            out[r, c] = card["color"]
            for d in directions:
                tip = c + d
                assert 0 <= tip < out.shape[1], "drawing exceeds pane width"
                out[r, tip] = card["color"]
                out[r + 1, tip] = card["color"]
                next_tips.add(tip)
        state = tuple(sorted(next_tips))
        states.append(state)
    assert state not in lookup, "final frontier must be an unmatched terminal"
    return out, order, states


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _execute(_parse(input_grid))[0]


def _program(rng, width, n, bound_witness=False):
    """Finite depth-six search over at most 3^6 branches per starting mask."""
    sizes = list(map(int, rng.permutation((1, 2, 3))))
    for size in sizes:
        starts = list(combinations(range(width), size))
        rng.shuffle(starts)
        for initial in starts:
            def visit(states, commands):
                if len(commands) == n:
                    if set(commands) != {"L", "R", "F"}:
                        return None
                    tags = states[:-1]
                    if bound_witness and not (
                            len({s[0] for s in tags}) < n or
                            len({s[-1] for s in tags}) < n):
                        return None
                    if bound_witness and not _sortable_witness(states, commands):
                        return None
                    return states, commands
                for command in rng.permutation(("L", "R", "F")):
                    command = str(command)
                    following = _advance(states[-1], command)
                    if (following[0] < 0 or following[-1] >= width
                            or following in states):
                        continue
                    result = visit(states + [following], commands + [command])
                    if result is not None:
                        return result
                return None
            result = visit([initial], [])
            if result is not None:
                return result
    raise AssertionError("finite program space exhausted")


def _pack(cards, order, cols, width, height, bg, divider, marker, initial):
    rows = (len(cards) + cols - 1) // cols
    panel_w = cols * (width + 1) - 1
    a = np.full((height, panel_w + 1 + width), bg, dtype=np.int8)
    a[:, panel_w] = divider
    a[0, panel_w + 1 + np.array(initial)] = marker
    for row in range(rows - 1):
        a[7 * row + 6, :panel_w] = divider
    for col in range(cols - 1):
        a[:7 * rows - 1, (width + 1) * col + width] = divider
    glyph_left = (width - 3) // 2
    for slot, index in enumerate(order):
        row, col = divmod(slot, cols)
        top, left = row * 7, col * (width + 1)
        card = cards[index]
        a[top, left + np.array(card["tag"])] = marker
        block = a[top + 2:top + 5, left + glyph_left:left + glyph_left + 3]
        block[MASKS[card["command"]]] = card["color"]
    return a


def _shortcut(info, indices, replace_fork=False):
    sequence = [info["cards"][i] for i in indices]
    commands = [("R" if replace_fork and c["command"] == "F" else c["command"])
                for c in sequence]
    return _sequence_output(info["height"], info["width"], info["background"],
                            info["marker"], info["initial"], commands,
                            [c["color"] for c in sequence])


def _different(a, b):
    return a is None or not np.array_equal(a, b)


def _sorted_orders(cards):
    """Six simple ordering hypotheses, with spatial order breaking ties."""
    for feature in (lambda tag: tag[0], lambda tag: tag[-1], len):
        for reverse in (False, True):
            yield sorted(range(len(cards)), key=lambda i: feature(cards[i]["tag"]),
                         reverse=reverse)


def _sortable_witness(states, commands):
    """A four-card monochrome demonstration can refute all six tag sorts."""
    intended = tuple(commands)
    for spatial in permutations(range(len(commands))):
        if tuple(commands[i] for i in spatial) == intended:
            continue
        cards = [dict(tag=states[i]) for i in spatial]
        if all(tuple(commands[spatial[i]] for i in order) != intended
               for order in _sorted_orders(cards)):
            return True
    return False


def _make_pair(rng, n, witness, color_mode, previous_roles):
    width = int(rng.integers(7 if n == 6 else 6, 10))
    states, commands = _program(rng, width, n, bound_witness=witness)
    palette = list(map(int, rng.permutation(10)))
    bg = next(c for c in palette if previous_roles is None or c != previous_roles[0])
    divider = next(c for c in palette if c != bg and
                   (previous_roles is None or c != previous_roles[1]))
    marker = next(c for c in palette if c not in (bg, divider))
    inks = [c for c in palette if c not in (bg, divider, marker)]
    if color_mode == "distinct":
        colors = inks[:n]
    elif color_mode == "shared":
        colors = [inks[0]] * n
    else:
        size = int(rng.integers(2, min(4, n) + 1))
        colors = [inks[int(rng.integers(size))] for _ in range(n)]
    cards = [dict(tag=states[i], command=commands[i], color=colors[i]) for i in range(n)]
    layouts = [cols for cols in range(1, 4)
               if ((n + cols - 1) // cols) * 7 - 1 <= 30
               and (cols + 1) * width + cols <= 30]
    rng.shuffle(layouts)
    margin = int(rng.integers(3))
    # Pick a shuffle that disproves both common spatial traversals. This is a
    # finite permutation space (at most 720), not a retry loop.
    orders = list(permutations(range(n)))
    rng.shuffle(orders)
    chosen = None
    for cols in layouts:
        rows = (n + cols - 1) // cols
        height = max(rows * 7 - 1, 2 * n + 1) + margin
        out = _sequence_output(height, width, bg, marker, states[0], commands, colors)
        assert out is not None
        for order in orders:
            col_major = sorted(range(n), key=lambda s: (s % cols, s // cols))
            alternatives = [order, tuple(order[s] for s in col_major)]
            if witness:
                spatial_cards = [cards[i] for i in order]
                alternatives.extend(tuple(order[s] for s in alt)
                                    for alt in _sorted_orders(spatial_cards))
            if all(_different(_sequence_output(height, width, bg, marker, states[0],
                                              [commands[i] for i in alt],
                                              [colors[i] for i in alt]), out)
                   for alt in alternatives):
                chosen = order
                break
        if chosen is not None:
            break
    assert chosen is not None
    inp = _pack(cards, chosen, cols, width, height, bg, divider, marker, states[0])
    return {"input": inp, "output": out}, (bg, divider)


def _fresh(pair, seen):
    if _key(pair["input"]) not in seen:
        return pair
    # Ninety different visible background/divider assignments exceed the
    # maximum thirteen previous inputs. All colors are mapped bijectively.
    info = _parse(pair["input"])
    old_roles = [info["background"], info["divider"]]
    old = old_roles + [c for c in range(10) if c not in old_roles]
    for bg in range(10):
        for divider in range(10):
            if bg == divider:
                continue
            new = [bg, divider] + [c for c in range(10) if c not in (bg, divider)]
            mapping = np.empty(10, dtype=np.int8)
            mapping[old] = new
            candidate = {k: mapping[v] for k, v in pair.items()}
            if _key(candidate["input"]) not in seen:
                return candidate
    raise AssertionError("finite palette fallback exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen, previous = {"train": [], "test": []}, set(), None
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            if split == "train":
                n = 3 if i == 0 else 4 if i == 1 else int(rng.integers(3, 5))
                mode = ("distinct" if i == 0 else "shared" if i == 1 else
                        str(rng.choice(("distinct", "shared", "partial"))))
            else:
                n = int(rng.integers(5, 7))
                mode = str(rng.choice(("shared", "partial")))
            pair, previous = _make_pair(rng, n, split == "train" and i == 1, mode, previous)
            pair = _fresh(pair, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _parse(inp)
            predicted, order, states = _execute(info)
            assert np.array_equal(out, predicted), "incorrect relay trace"
            assert out.shape == info["pane"].shape
            assert set(order) == set(range(len(info["cards"])))
            assert len(set(states)) == len(states), "frontier repetition"
            assert {c["command"] for c in info["cards"]} == {"L", "R", "F"}
            assert np.array_equal(out[0], info["pane"][0]), "seed preservation"
            assert np.all(out[2 * len(order) + 1:] == info["background"]), "blank margins"
            # Both inherited mechanisms must affect the answer, pair by pair.
            assert _different(_shortcut(info, order, replace_fork=True), out), "fork ablation"
            assert _different(_shortcut(info, range(len(order))), out), "raster shortcut"
            col_order = sorted(range(len(order)), key=lambda i: (info["cards"][i]["col"], info["cards"][i]["row"]))
            assert _different(_shortcut(info, col_order), out), "column-major shortcut"
            info.update(order=order, states=states)
            records[split].append(info)
    assert all(3 <= len(i["cards"]) <= 4 for i in records["train"])
    assert all(5 <= len(i["cards"]) <= 6 for i in records["test"])
    first, second = records["train"][:2]
    assert len(first["cards"]) == 3 and len(second["cards"]) == 4
    assert len({c["color"] for c in first["cards"]}) == 3
    assert len({c["color"] for c in second["cards"]}) == 1
    tags = [c["tag"] for c in second["cards"]]
    assert len({t[0] for t in tags}) < len(tags) or len({t[-1] for t in tags}) < len(tags)
    true_output = _execute(second)[0]
    for order in _sorted_orders(second["cards"]):
        assert _different(_shortcut(second, order), true_output), "sorted-tag shortcut"
    assert any(len({len(c["tag"]) for c in i["cards"]}) < len(i["cards"])
               for i in records["train"]), "equal-cardinality port witness"


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, data = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(data)
    return digest.hexdigest()


def self_test():
    """42 seeds, streamed; no saved grids, logs, traces, or growing cache."""
    # Hand-calculated witness: two initial tips, a merging fork, then a shift.
    fixture_cards = [dict(tag=(2, 4), command="L", color=2),
                     dict(tag=(1, 3), command="F", color=3),
                     dict(tag=(0, 2, 4), command="R", color=4)]
    fixture = _pack(fixture_cards, (2, 0, 1), 2, 7, 13, 0, 9, 8, (2, 4))
    rows = ("0080800", "0222200", "0202000", "3333300", "3030300",
            "4444440", "0404040") + ("0000000",) * 6
    expected = np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
    assert np.array_equal(solve(fixture), expected)
    metrics = {name: set() for name in ("puzzles", "shapes", "layouts", "widths",
                                      "initial_masks", "programs", "tip_counts", "fork_counts")}
    merges = boundary_hits = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        metrics["puzzles"].add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                output, order, states = _execute(info)
                sequence = [info["cards"][i] for i in order]
                metrics["shapes"].add(pair["input"].shape)
                metrics["layouts"].add(info["layout"])
                metrics["widths"].add(info["width"])
                metrics["initial_masks"].add((info["width"], info["initial"]))
                metrics["programs"].add(tuple(c["command"] for c in sequence))
                metrics["tip_counts"].update(map(len, states))
                metrics["fork_counts"].add(sum(c["command"] == "F" for c in sequence))
                for state, card, following in zip(states, sequence, states[1:]):
                    if card["command"] == "F":
                        merges += len(following) < 2 * len(state)
                    boundary_hits += following[0] == 0 or following[-1] == info["width"] - 1
                assert np.array_equal(output, pair["output"])
        # A palette bijection cannot alter dispatch or the geometric rule.
        mapping = np.roll(np.arange(10, dtype=np.int8), seed % 9 + 1)
        pair = puzzle["test"][0]
        assert np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]])
        # A new spatial permutation leaves the trace identical.
        info = _parse(pair["input"])
        cards = info["cards"]
        shuffled = _pack(cards, list(reversed(range(len(cards)))), info["layout"][1],
                         info["width"], info["height"], info["background"],
                         info["divider"], info["marker"], info["initial"])
        assert np.array_equal(solve(shuffled), pair["output"])
    assert len(metrics["puzzles"]) == 42
    assert len(metrics["shapes"]) >= 25
    assert len(metrics["layouts"]) >= 5
    assert metrics["widths"] == {6, 7, 8, 9}
    assert len(metrics["initial_masks"]) >= 30
    assert len(metrics["programs"]) >= 15
    assert len(metrics["tip_counts"]) >= 5
    assert len(metrics["fork_counts"]) >= 2
    assert merges and boundary_hits
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(42, nt, nq)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A true output mutation must fail the semantic validator.
    damaged = generate_puzzle(100)
    out = damaged["test"][0]["output"].copy()
    info = _parse(damaged["test"][0]["input"])
    r, c = np.argwhere(out[1:] != info["background"])[0]
    out[int(r) + 1, int(c)] = info["background"]
    damaged["test"][0]["output"] = out
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("output mutation accepted")
    # Duplicate tags must be rejected even if glyphs differ.
    bad_input = generate_puzzle(101)["train"][0]["input"].copy()
    info = _parse(bad_input)
    a, b = info["cards"][:2]
    pane_left = bad_input.shape[1] - info["width"] - 1
    row, left = 7 * b["row"], (info["width"] + 1) * b["col"]
    bad_input[row, left:left + info["width"]] = info["background"]
    bad_input[row, left + np.array(a["tag"])] = info["marker"]
    assert pane_left >= left + info["width"]
    try:
        solve(bad_input)
    except AssertionError:
        pass
    else:
        raise AssertionError("ambiguous tags accepted")
    return {"seeds": 42, "pairs": 210,
            **{name: len(values) for name, values in metrics.items()},
            "fork_merges": int(merges), "boundary_hits": int(boundary_hits)}
