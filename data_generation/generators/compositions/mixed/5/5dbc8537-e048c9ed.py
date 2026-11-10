"""Exact-cover horizontal bars, then squared-adjacency projection.

No parent generator is imported. The solver reconstructs panel roles, finite
inventory, unique tile transfer, and reference-column labels from input cells.
No file I/O, persistent cache, or execution traces are used by this module.
"""
from collections import Counter
import hashlib
import numbers

import numpy as np


_LENGTHS = (2, 3, 4, 5)
_DIGITS = {length: (length - 1) ** 2 % 10 for length in _LENGTHS}
_BOARD_COLORS = (0, 2, 3, 5, 7, 8)
_SIDES = ("left", "right", "top", "bottom")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and
             bool(np.all((0 <= a) & (a <= 9))), "grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=False)


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        _require(not isinstance(value, (bool, np.bool_)) and
                 isinstance(value, numbers.Integral) and lo <= value <= hi,
                 "counts must be integers: train 2..10, test 1..4")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(a):
    return (a.shape, a.tobytes())


def _runs(row, color):
    points = np.flatnonzero(row == color)
    if not len(points):
        return []
    cuts = np.flatnonzero(np.diff(points) > 1) + 1
    return [(int(s[0]), len(s)) for s in np.split(points, cuts)]


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        stack, cells = [(y, x)], []
        seen[y, x] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        yield cells


def _stencil(stencil, board, vacancy, marker):
    """Recognize slots and the reference without looking at the tray or output."""
    _require(board in _BOARD_COLORS, "board color would hide a digit")
    _require(np.count_nonzero(stencil == marker) == 1, "reference must be a singleton")
    mr, mc = map(int, np.argwhere(stencil == marker)[0])
    _require(np.all(stencil[:, mc][np.arange(stencil.shape[0]) != mr] == board),
             "reference column must initially be clear")
    slots = []
    for r, row in enumerate(stencil):
        runs = _runs(row, vacancy)
        if not runs:
            continue
        _require(r != mr and len(runs) == 1, "one slot per occupied row")
        c, length = runs[0]
        _require(length in _LENGTHS, "slot length must be 2..5")
        _require(c + length < mc or c > mc + 1, "one board cell must separate slot and reference column")
        _require(abs(r - mr) >= 2, "reference must be separated from bar rows")
        slots.append((r, c, length))
    _require(bool(slots), "stencil has no slots")
    _require(all(b[0] - a[0] >= 2 for a, b in zip(slots, slots[1:])),
             "bar rows must have a separating board row")
    return (mr, mc), slots


def _tiles(tray, vacancy, board):
    _require(not np.any(tray == board), "board color cannot occur in tray")
    _require(np.count_nonzero(tray == vacancy) > tray.size // 2,
             "tray background must be the strict majority")
    tiles = []
    for cells in _components(tray != vacancy):
        rows, columns = zip(*cells)
        r, c = min(rows), min(columns)
        length = len(cells)
        _require(min(rows) == max(rows) and max(columns) - c + 1 == length,
                 "tray objects must be horizontal bars")
        _require(2 <= length <= 9 and np.all(tray[r, c:c + length] == tray[r, c]),
                 "tray bars must be monochrome and length 2..9")
        tiles.append((length, int(tray[r, c])))
    _require(bool(tiles), "tray is empty")
    return tiles


def _exact_cover(slots, tiles, max_nodes=10000):
    """Exhaustively cover row intervals with finite bar copies.

    This is generic interval exact cover, including potential end-to-end tilings.
    A family member must have one rendered answer and unambiguous single-bar
    rows. Grouping identical copies removes permutation-only duplicate work.
    Bounds apply to CPU/RAM; nothing is written to disk.
    """
    inventory = Counter(tiles)
    groups = sorted(inventory)
    remaining = [inventory[g] for g in groups]
    # Short slots first makes exhausted inventories constrain longer slots.
    order = sorted(range(len(slots)), key=lambda i: (slots[i][2], slots[i][0]))
    paints = [None] * len(slots)
    assignments = [None] * len(slots)
    outcomes = {}
    nodes = 0

    def visit(slot_index, offset, pieces, pixels):
        nonlocal nodes
        nodes += 1
        _require(nodes <= max_nodes, "exact-cover search budget exceeded")
        if len(outcomes) >= 2:
            return
        if slot_index == len(order):
            rendering = tuple(paints)
            single = all(len(a) == 1 for a in assignments)
            if rendering not in outcomes:
                outcomes[rendering] = (tuple(assignments), single)
            elif not single:
                old, _ = outcomes[rendering]
                outcomes[rendering] = (old, False)
            return
        i = order[slot_index]
        target_length = slots[i][2]
        if offset == target_length:
            paints[i], assignments[i] = tuple(pixels), tuple(pieces)
            visit(slot_index + 1, 0, [], [])
            paints[i], assignments[i] = None, None
            return
        for g, (length, color) in enumerate(groups):
            if remaining[g] and offset + length <= target_length:
                remaining[g] -= 1
                visit(slot_index, offset + length, pieces + [(length, color)],
                      pixels + [color] * length)
                remaining[g] += 1
                if len(outcomes) >= 2:
                    break

    visit(0, 0, [], [])
    _require(len(outcomes) == 1, "cover must have exactly one rendered completion")
    rendering, (placements, single) = next(iter(outcomes.items()))
    _require(single, "cover must leave one unambiguous placed bar per row")
    return rendering, placements, nodes


def _panel_views(grid):
    for axis in (0, 1):
        for cut in range(1, grid.shape[axis]):
            a, b = (grid[:cut], grid[cut:]) if axis == 0 else (grid[:, :cut], grid[:, cut:])
            yield (_SIDES[2] if axis == 0 else _SIDES[0]), a, b
            yield (_SIDES[3] if axis == 0 else _SIDES[1]), b, a


def _analyze(input_grid):
    grid = _grid(input_grid)
    answers = []
    for side, stencil, tray in _panel_views(grid):
        colors, counts = np.unique(stencil, return_counts=True)
        if len(colors) != 3:
            continue
        singleton_colors = colors[counts == 1]
        if len(singleton_colors) != 1:
            continue
        marker = int(singleton_colors[0])
        other = [int(c) for c in colors if c != marker]
        for board, vacancy in (other, other[::-1]):
            if board not in _BOARD_COLORS or np.any(tray == board):
                continue
            try:
                reference, slots = _stencil(stencil, board, vacancy, marker)
                tiles = _tiles(tray, vacancy, board)
                rendering, placements, nodes = _exact_cover(slots, tiles)
            except ValueError as exc:
                if "budget" in str(exc):
                    # An unresolved interpretation is never silently discarded.
                    raise
                continue
            filled = stencil.copy()
            for (r, c, length), pixels in zip(slots, rendering):
                filled[r, c:c + length] = pixels
            output = filled.copy()
            for r, c, length in slots:
                # Count adjacencies on the actual transferred monochrome bar.
                bar = filled[r, c:c + length]
                _require(np.all(bar == bar[0]), "transferred bar is not monochrome")
                adjacencies = int(np.count_nonzero(bar[1:] == bar[:-1]))
                output[r, reference[1]] = adjacencies ** 2 % 10
            answers.append(dict(side=side, stencil=stencil, tray=tray,
                                board=board, vacancy=vacancy, marker=marker,
                                reference=reference, slots=slots, tiles=tiles,
                                placements=placements, filled=filled,
                                output=output, nodes=nodes))
    _require(len(answers) == 1, "input must identify exactly one stencil and cover")
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the filled, labeled stencil, using input cells alone."""
    return _analyze(input_grid)["output"].copy()


def _scatter(rng, tiles, height, width, vacancy, seam):
    """Independent source coordinates, one-cell moat, and a visible seam edge."""
    for _ in range(24):
        tray = np.full((height, width), vacancy, dtype=np.int8)
        occupied = np.zeros((height, width), dtype=bool)
        order = list(rng.permutation(len(tiles)))
        order.sort(key=lambda i: -tiles[i][0])
        for index, i in enumerate(order):
            length, color = tiles[i]
            legal = []
            for r in range(height):
                for c in range(width - length + 1):
                    if index == 0:
                        if seam == "left" and c != 0:
                            continue
                        if seam == "right" and c + length != width:
                            continue
                        if seam == "top" and r != 0:
                            continue
                        if seam == "bottom" and r != height - 1:
                            continue
                    if not np.any(occupied[max(0, r - 1):min(height, r + 2),
                                           max(0, c - 1):min(width, c + length + 1)]):
                        legal.append((r, c))
            if not legal:
                break
            r, c = legal[int(rng.integers(len(legal)))]
            tray[r, c:c + length] = color
            occupied[r, c:c + length] = True
        else:
            if np.count_nonzero(tray == vacancy) > tray.size // 2:
                return tray
    raise ValueError("tray packing failed")


def _make_pair(rng, lengths, side, *, decoy=None, avoid_three=None,
               avoid_roles=None, teaching=False):
    for _ in range(64):
        board = int(rng.choice(_BOARD_COLORS))
        vacancy = int(rng.choice([c for c in range(10) if c != board]))
        if (board, vacancy) == avoid_roles:
            continue
        palette = [c for c in range(10) if c not in (board, vacancy)]
        marker = int(rng.choice(palette))
        colors = {length: int(rng.choice(palette)) for length in set(lengths)}
        if teaching:
            # One visible color at multiple lengths disproves color-coded labels.
            common = int(rng.choice(palette))
            colors[2] = colors[3] = common
        if avoid_three is not None:
            colors[3] = int(rng.choice([c for c in palette if c != avoid_three]))
        specs = [(length, colors[length]) for length in lengths]
        tiles = list(specs)
        if decoy is not None:
            tiles.append((decoy, int(rng.choice(palette))))
        if rng.random() < .32:
            tiles.append((int(rng.integers(6, 10)), int(rng.choice(palette))))
        # A bar color may equal the reference; its singleton role is local.
        if rng.random() < .40:
            marker = colors[int(rng.choice(lengths))]
        specs = [specs[int(i)] for i in rng.permutation(len(specs))]
        n = len(specs)
        gaps = rng.integers(2, 4, size=n)
        active = [int(rng.integers(0, 3))]
        for gap in gaps:
            active.append(active[-1] + int(gap))
        h = active[-1] + 1 + int(rng.integers(0, 3))
        # At most eight bars + reference: 30 rows can always accommodate them.
        if h > (22 if side in ("top", "bottom") else 30):
            active = list(range(1, 2 * (n + 1), 2))
            h = active[-1] + 1 + int(rng.integers(0, 2))
        marker_slot = int(rng.integers(n + 1))
        mr = active[marker_slot]
        rows = active[:marker_slot] + active[marker_slot + 1:]
        mixed = teaching or rng.random() < .78
        if mixed:
            mc = int(rng.integers(6, 10))
            w = mc + 7 + int(rng.integers(0, 4))
            sides = [-1, 1] + [int(rng.choice((-1, 1))) for _ in range(n - 2)]
            if n == 1:
                sides = [int(rng.choice((-1, 1)))]
            rng.shuffle(sides)
        else:
            w = int(rng.integers(9, 16))
            right = bool(rng.integers(2))
            mc = int(rng.integers(0, 3)) if right else w - 1 - int(rng.integers(0, 3))
            sides = [1 if right else -1] * n
        stencil = np.full((h, w), board, dtype=np.int8)
        filled = stencil.copy()
        stencil[mr, mc] = filled[mr, mc] = marker
        expected = []
        for r, (length, color), direction in zip(rows, specs, sides):
            if direction == -1:
                c = int(rng.integers(0, mc - length))
            else:
                c = int(rng.integers(mc + 2, w - length + 1))
            stencil[r, c:c + length] = vacancy
            filled[r, c:c + length] = color
            expected.append((r, c, length))
        out = filled.copy()
        for r, _, length in expected:
            out[r, mc] = _DIGITS[length]
        try:
            if side in ("left", "right"):
                tw = int(rng.integers(max(tiles)[0] + 1, 31 - w))
                tray = _scatter(rng, tiles, h, tw, vacancy,
                                "left" if side == "left" else "right")
                inp = np.concatenate((stencil, tray) if side == "left" else (tray, stencil), axis=1)
            else:
                th = int(rng.integers(5, 31 - h))
                tray = _scatter(rng, tiles, th, w, vacancy,
                                "top" if side == "top" else "bottom")
                inp = np.concatenate((stencil, tray) if side == "top" else (tray, stencil), axis=0)
            a = _analyze(inp)
            _require(np.array_equal(a["output"], out), "reconstructed answer differs")
            return {"input": inp, "output": out}, colors, (board, vacancy)
        except ValueError:
            continue
    raise RuntimeError("bounded pair construction failed")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    panel_order = [str(s) for s in rng.permutation(_SIDES)]
    # Half the puzzles hold out the other panel axis; bars stay horizontal.
    hold_axis = bool(rng.integers(2))
    if hold_axis:
        train_sides = [panel_order[0], {"left": "right", "right": "left",
                        "top": "bottom", "bottom": "top"}[panel_order[0]]]
        test_sides = [s for s in _SIDES if s not in train_sides]
    else:
        train_sides = test_sides = panel_order
    puzzle = {"train": [], "test": []}
    seen, first_three, first_roles = set(), None, None
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if split == "train" and i == 0:
                lengths, decoy = [2, 3, 4, 5], None
            elif split == "train" and i == 1:
                lengths = [3, 3, 4, 5] + [int(rng.choice((3, 4))) for _ in range(int(rng.integers(0, 3)))]
                decoy = 2
            elif split == "train":
                n = int(rng.integers(1, 7))
                lengths = [int(rng.choice((2, 3, 4))) for _ in range(n)]
                if rng.random() < .75:
                    lengths[int(rng.integers(n))] = 5
                missing = [l for l in (2, 3, 4) if l not in lengths]
                decoy = int(rng.choice(missing)) if missing and rng.random() < .55 else None
            else:
                omitted = int(rng.choice((2, 3, 4)))
                available = [l for l in (2, 3, 4) if l != omitted]
                n = int(rng.integers(4, 9))
                lengths = [5, 5] + [int(rng.choice(available + [5])) for _ in range(n - 2)]
                # Ensure at least one non-five; repeated fives are a held-out multiplicity.
                lengths[2] = int(rng.choice(available))
                decoy = omitted if rng.random() < .75 else None
            candidates = train_sides if split == "train" else test_sides
            side = candidates[i % len(candidates)]
            for _ in range(8):
                pair, colors, roles = _make_pair(
                    rng, lengths, side, decoy=decoy,
                    avoid_three=first_three if split == "train" and i == 1 else None,
                    avoid_roles=first_roles if split == "train" and i == 1 else None,
                    teaching=split == "train" and i == 0)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
            if split == "train" and i == 0:
                first_three = colors[3]
                first_roles = roles
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, metas = set(), {"train": [], "test": []}
    taught, length_colors, color_lengths = {}, {}, {}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            a = _analyze(inp)
            _require(np.array_equal(out, a["output"]), "incorrect answer")
            _require(out.shape == a["stencil"].shape and out.size < inp.size, "output must crop to full stencil")
            mr, mc = a["reference"]
            _require(out[mr, mc] == a["marker"], "reference changed")
            slots = a["slots"]
            used = Counter()
            for (r, c, length), placement in zip(slots, a["placements"]):
                _require(len(placement) == 1, "one placed tile per row required")
                tile = placement[0]
                used[tile] += 1
                _require(tile[0] == length and np.all(out[r, c:c + length] == tile[1]), "tile transfer changed")
                _require(out[r, mc] == _DIGITS[length], "wrong squared-adjacency code")
                if split == "train":
                    _require(length not in taught or taught[length] == int(out[r, mc]), "contradictory teaching")
                    taught[length] = int(out[r, mc])
                    length_colors.setdefault(length, set()).add(tile[1])
                    color_lengths.setdefault(tile[1], set()).add(length)
                else:
                    _require(taught.get(length) == int(out[r, mc]), "test code not taught")
            _require(not (used - Counter(a["tiles"])), "physical tile reused")
            _require(np.count_nonzero(a["filled"] != a["stencil"]) == sum(s[2] for s in slots), "coverage is not exact")
            _require(np.count_nonzero(out != a["filled"]) == len(slots), "must write one visible code per row")
            changed = out != a["stencil"]
            allowed = a["stencil"] == a["vacancy"]
            for r, _, _ in slots:
                allowed[r, mc] = True
            _require(not np.any(changed & ~allowed), "unrelated stencil cell changed")
            # Concrete parent ablations: fill-only fails B; labels-only fails A.
            _require(not np.array_equal(out, a["filled"]), "B is dispensable")
            labels_only = a["stencil"].copy()
            for r, _, length in slots:
                labels_only[r, mc] = _DIGITS[length]
            _require(not np.array_equal(labels_only, out), "A is dispensable")
            a["unused"] = Counter(a["tiles"]) - used
            metas[split].append(a)
    _require(set(taught) == set(_LENGTHS), "training must teach all four length codes")
    _require(any(len(c) >= 2 for c in length_colors.values()), "teach fixed length with different colors")
    _require(any(len(l) >= 2 for l in color_lengths.values()), "teach fixed color with different lengths")
    _require(any(a["unused"] for a in metas["train"]), "training must demonstrate surplus")
    _require(any(len({-1 if c < a["reference"][1] else 1 for _, c, _ in a["slots"]}) == 2
                 for a in metas["train"]), "training must teach both sides of reference")
    _require(len({(a["board"], a["vacancy"]) for a in metas["train"]}) >= 2,
             "training must show local color roles")
    return None


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Stream 42 seeds; retain only digests and small diversity counters."""
    signatures, shapes, layouts, palettes, sides, board_colors = (set() for _ in range(6))
    bar_counts, markers, local_decoys = set(), set(), 0
    repeated, zero_payload, nonzero_board = False, False, False
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_puzzle_digest(p))
        for split in ("train", "test"):
            for pair in p[split]:
                before = pair["input"].copy()
                a = _analyze(before)
                assert np.array_equal(solve(before), pair["output"])
                assert np.array_equal(before, pair["input"]), "solve mutated input"
                shapes.add((pair["input"].shape, pair["output"].shape))
                layouts.add((a["side"], a["reference"], tuple(a["slots"])))
                palettes.add(tuple(map(int, np.unique(pair["input"]))))
                sides.add(a["side"])
                board_colors.add(a["board"])
                bar_counts.add(len(a["slots"]))
                markers.add(a["reference"])
                used = Counter(x[0] for x in a["placements"])
                unused = Counter(a["tiles"]) - used
                local_decoys += sum(count for (length, _), count in unused.items()
                                    if any(length <= s[2] for s in a["slots"]))
                repeated |= max(used.values()) >= 2
                zero_payload |= any(color == 0 for _, color in a["tiles"])
                nonzero_board |= a["board"] != 0
    assert len(signatures) == 42 and len(shapes) >= 50 and len(layouts) >= 150
    assert len(palettes) >= 20 and sides == set(_SIDES) and board_colors == set(_BOARD_COLORS)
    assert bar_counts >= {1, 2, 3, 4, 5, 6, 7, 8} and len(markers) >= 40
    assert repeated and zero_payload and nonzero_board and local_decoys >= 42
    assert _puzzle_digest(generate_puzzle(17)) == _puzzle_digest(generate_puzzle(17))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(8100 + nt * 10 + ne, nt, ne)
            validate_generated_puzzle(p)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Direct semantic probes: inventory exhaustion, ambiguous transfer, and
    # indistinguishable bar-boundary ambiguity must all be rejected.
    for slots, tiles in (([(0, 0, 4)], [(2, 7)]),
                         ([(0, 0, 2)], [(2, 7), (2, 8)]),
                         ([(0, 0, 4)], [(4, 7), (2, 7), (2, 7)])):
        try:
            _exact_cover(slots, tiles)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid or ambiguous cover accepted")
    paint, placements, _ = _exact_cover([(0, 0, 2), (2, 0, 2)], [(2, 7), (2, 7)])
    assert paint == ((7, 7), (7, 7)) and len(placements) == 2
