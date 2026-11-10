"""Chamber ballots: wall-defined reflection certifies one vote per room.

No parent code is imported. Construction keeps explicit room boxes; prediction
recovers rooms from the returned pixels and has no access to those boxes.
"""
from collections import Counter
import json

import numpy as np


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((0 <= g) & (g <= 9))):
        raise ValueError("expected an integer ARC grid")
    return g.astype(np.int8, copy=False)


def _key(grid):
    g = np.asarray(grid, dtype=np.int8)
    return g.shape, g.tobytes()


def _walls(shape, boxes):
    mask = np.zeros(shape, dtype=bool)
    for t, l, b, r in boxes:
        mask[t, l:r + 1] = mask[b, l:r + 1] = True
        mask[t:b + 1, l] = mask[t:b + 1, r] = True
    return mask


def _enclosed_rooms(mask):
    """Flood fill non-wall cells; padding distinguishes exterior at grid edges."""
    free = np.pad(~mask, 1, constant_values=True)
    seen = np.zeros(free.shape, dtype=bool)
    h, w = free.shape
    rooms = []
    for sy, sx in np.argwhere(free):
        sy, sx = int(sy), int(sx)
        if seen[sy, sx]:
            continue
        seen[sy, sx] = True
        stack, cells, outside = [(sy, sx)], [], False
        while stack:
            y, x = stack.pop()
            cells.append((y - 1, x - 1))
            outside |= y in (0, h - 1) or x in (0, w - 1)
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and free[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        if not outside:
            a = np.array(cells)
            t, l = map(int, a.min(axis=0))
            b, r = map(int, a.max(axis=0))
            assert len(cells) == (b - t + 1) * (r - l + 1), "rooms must be rectangles"
            rooms.append((t, l, b, r))
    return sorted(rooms)


def _flanks(mask):
    padded = np.pad(mask, 1)
    horizontal = padded[1:-1, :-2] & padded[1:-1, 2:]
    vertical = padded[:-2, 1:-1] & padded[2:, 1:-1]
    return horizontal | vertical


def _plurality(counts):
    assert counts, "no ballots"
    maximum = max(counts.values())
    winners = [color for color, count in counts.items() if count == maximum]
    assert len(winners) == 1, "tied elections are outside the family"
    return int(winners[0])


def _decode(input_grid):
    g = _grid(input_grid)
    frequencies = np.bincount(g.ravel().astype(int), minlength=10)
    bg = int(frequencies.argmax())
    assert np.count_nonzero(frequencies == frequencies[bg]) == 1, "ambiguous background"
    foreground = np.argwhere(g != bg)
    assert len(foreground)
    t, l = map(int, foreground.min(axis=0))
    b, r = map(int, foreground.max(axis=0))
    wall = int(g[t, l])
    assert wall != bg and g[t, r] == g[b, l] == g[b, r] == wall, "intact frame corners required"
    visible = g == wall
    # Gaps are marked by a non-background color, are one cell long, and have
    # visible collinear neighbors. Deliberate background openings are not filled.
    repaired = (~visible) & (g != bg) & _flanks(visible)
    mask = visible | repaired
    assert (mask[t, l:r + 1].all() and mask[b, l:r + 1].all()
            and mask[t:b + 1, l].all() and mask[t:b + 1, r].all())
    rooms = _enclosed_rooms(mask)
    assert rooms
    boxes = [(a - 1, c - 1, d + 1, e + 1) for a, c, d, e in rooms]
    assert np.array_equal(mask, _walls(g.shape, boxes)), "unexplained wall fragments"
    outside = np.ones_like(mask)
    outside[t:b + 1, l:r + 1] = False
    assert np.all(g[outside] == bg)
    if repaired.any():
        crack_colors = set(map(int, np.unique(g[repaired])))
        assert len(crack_colors) == 1
        crack = next(iter(crack_colors))
        assert np.array_equal(g == crack, repaired), "crack ink only occurs on walls"
    else:
        crack = None
    ballots, stats = Counter(), []
    for top, left, bottom, right in rooms:
        room = g[top:bottom + 1, left:right + 1]
        supported = (room == room[:, ::-1]) & (room != bg)
        colors = set(map(int, np.unique(room[supported])))
        assert wall not in colors and len(colors) <= 1, "a room cannot cast conflicting ballots"
        ballot = next(iter(colors)) if colors else None
        if ballot is not None:
            ballots[ballot] += 1
        raw = Counter(map(int, room[room != bg]))
        stats.append({"box": (top, left, bottom, right), "ballot": ballot,
                      "support": int(supported.sum()),
                      "unmatched": int(np.count_nonzero(room != bg) - supported.sum()),
                      "raw": raw, "axis2": left + right,
                      "axis_cells": int(np.count_nonzero(supported[:, room.shape[1] // 2]))
                      if room.shape[1] % 2 else 0})
    winner = _plurality(ballots)
    return {"background": bg, "wall": wall, "crack": crack, "mask": mask,
            "rooms": stats, "ballots": ballots, "winner": winner,
            "frame": (t, l, b, r), "repaired": repaired}


def _render(shape, background, wall, mask, rooms, winner):
    output = np.full(shape, background, dtype=np.int8)
    output[mask] = wall
    for t, l, b, r in rooms:
        output[t:b + 1, l:r + 1] = winner
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all roles, local axes, ballots, and the election from input only."""
    g = _grid(input_grid)
    a = _decode(g)
    return _render(g.shape, a["background"], a["wall"], a["mask"],
                   [s["box"] for s in a["rooms"]], a["winner"])


def _capacity(box):
    t, l, b, r = box
    return ((b - t) // 4) * ((r - l) // 5)


def _spans(rng, parts, span, minimum):
    gaps = np.full(parts, minimum, dtype=int)
    for _ in range(span - parts * minimum):
        gaps[int(rng.integers(parts))] += 1
    return [0] + list(map(int, np.cumsum(gaps)))


def _partition(rng, h, w, count, style):
    """Each cut divides one room, retaining T-junction endpoints and capacity."""
    outer = (0, 0, h - 1, w - 1)
    if _capacity(outer) < count:
        return None
    if style in ("strip", "grid"):
        factors = [(ny, count // ny) for ny in range(1, count + 1)
                   if count % ny == 0 and 4 * ny <= h - 1 and 5 * (count // ny) <= w - 1
                   and (style == "grid" or ny == 1 or ny == count)]
        if factors:
            ny, nx = factors[int(rng.integers(len(factors)))]
            ys, xs = _spans(rng, ny, h - 1, 4), _spans(rng, nx, w - 1, 5)
            return [(ys[i], xs[j], ys[i + 1], xs[j + 1]) for i in range(ny) for j in range(nx)]
        style = "partial"
    boxes, first_axis = [outer], None
    if style == "pinwheel":
        if count < 5 or h - 1 < 12 or w - 1 < 15:
            return None
        for _ in range(20):
            y1 = int(rng.integers(4, h - 8))
            y2 = int(rng.integers(y1 + 4, h - 4))
            x1 = int(rng.integers(5, w - 10))
            x2 = int(rng.integers(x1 + 5, w - 5))
            proposed = [(0, 0, y1, x2), (0, x2, y2, w - 1),
                        (y2, x1, h - 1, w - 1), (y1, 0, h - 1, x1),
                        (y1, x1, y2, x2)]
            if sum(map(_capacity, proposed)) >= count:
                boxes = proposed
                break
        else:
            return None
    for step in range(count - len(boxes)):
        choices = []
        for i, (t, l, b, r) in enumerate(boxes):
            for axis, start, end, minimum in ((0, t, b, 4), (1, l, r, 5)):
                if style == "partial" and step == 1 and axis == first_axis:
                    continue
                for cut in range(start + minimum, end - minimum + 1):
                    halves = ((t, l, cut, r), (cut, l, b, r)) if axis == 0 else ((t, l, b, cut), (t, cut, b, r))
                    capacity = sum(_capacity(x) for j, x in enumerate(boxes) if j != i)
                    if capacity + sum(map(_capacity, halves)) >= count:
                        choices.append((i, axis, halves))
        if not choices:
            return None
        i, axis, halves = choices[int(rng.integers(len(choices)))]
        first_axis = axis if step == 0 else first_axis
        boxes[i:i + 1] = halves
    if style == "partial" and not _has_partial(_walls((h, w), boxes)):
        return None
    return boxes


def _has_partial(mask):
    """A divider has a visible stop rather than extending across the whole frame."""
    ys, xs = np.nonzero(mask)
    t, b, l, r = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    for y in range(t + 1, b):
        if np.any(mask[y, l:r] & mask[y, l + 1:r + 1]) and not mask[y, l:r + 1].all():
            return True
    for x in range(l + 1, r):
        if np.any(mask[t:b, x] & mask[t + 1:b + 1, x]) and not mask[t:b + 1, x].all():
            return True
    return False


def _orbits(h, w):
    return [((y, x), (y, w - 1 - x)) for y in range(h) for x in range(w // 2)]


def _core(rng, h, w, tiny=False, axis_only=False, dense_min=0):
    core = np.zeros((h, w), dtype=bool)
    if axis_only:
        core[int(rng.integers(h)), w // 2] = True
        return core
    pairs = _orbits(h, w)
    if dense_min:
        amount = max(1, (dense_min + 1) // 2)
        for i in rng.permutation(len(pairs))[:amount]:
            for p in pairs[int(i)]:
                core[p] = True
        return core if core.sum() >= dense_min else None
    if tiny:
        for p in pairs[int(rng.integers(len(pairs)))]:
            core[p] = True
        return core
    style = int(rng.integers(4))
    if style == 0:  # Thin hollow contour.
        core[0] = core[-1] = True
        core[:, 0] = core[:, -1] = True
    elif style == 1:  # Changing row profiles, some gaps and narrow stems.
        for y in range(h):
            extent = int(rng.integers(w // 2 + 1))
            core[y, w // 2 - extent:(w + 1) // 2 + extent] = True
    else:  # Sparse or dense supported islands; connectivity is immaterial.
        probability = .12 if style == 2 else .55
        for pair in pairs:
            if rng.random() < probability:
                for p in pair:
                    core[p] = True
        if w % 2:
            core[:, w // 2] = rng.random(h) < .25
    if not core.any():
        for p in pairs[int(rng.integers(len(pairs)))]:
            core[p] = True
    return core


def _add_unmatched(rng, room, background, colors, amount):
    """Use at most one cell of each empty non-axis mirror orbit."""
    h, w = room.shape
    candidates = [pair for pair in _orbits(h, w)
                  if all(room[p] == background for p in pair)]
    rng.shuffle(candidates)
    for pair in candidates[:amount]:
        room[pair[int(rng.integers(2))]] = int(rng.choice(colors))
    return min(amount, len(candidates))


def _roles(rng, parameters, excluded_winner=None, winner_index=0):
    roles = list(parameters["palette"])
    if parameters["local_palette"]:
        roles = list(map(int, rng.permutation(10)[:len(roles)]))
    if excluded_winner is not None and roles[3 + winner_index] == excluded_winner:
        j = 3 + (winner_index + 1) % parameters["pigments"]
        roles[3 + winner_index], roles[j] = roles[j], roles[3 + winner_index]
    return roles[0], roles[1], roles[2], roles[3:]


def _damage(rng, grid, mask, crack, require):
    p = np.pad(mask.astype(int), 1)
    degree = p[1:-1, :-2] + p[1:-1, 2:] + p[:-2, 1:-1] + p[2:, 1:-1]
    candidates = np.argwhere(mask & _flanks(mask) & (degree == 2))
    rng.shuffle(candidates)
    amount = int(rng.integers(1 if require else 0, 7))
    selected = []
    for y, x in candidates:
        if len(selected) >= amount:
            break
        if all(abs(int(y) - yy) + abs(int(x) - xx) >= 3 for yy, xx in selected):
            grid[y, x] = crack
            selected.append((int(y), int(x)))
    assert not require or selected


def _make_pair(rng, parameters, count, witness, winner_index, excluded_winner=None):
    bg, wall, crack, pigments = _roles(rng, parameters, excluded_winner, winner_index)
    winning_color = pigments[winner_index]
    losing_color = pigments[(winner_index + 1) % len(pigments)]
    for attempt in range(180):
        # Capacities, not a regular tiling, constrain scale. Both split ranges overlap.
        h, w = int(rng.integers(14, 29)), int(rng.integers(15, 29))
        style = "partial" if witness == "counterfactual" else str(rng.choice(["partial", "partial", "grid", "strip", "recursive", "pinwheel"]))
        boxes = _partition(rng, h, w, count, style)
        if boxes is None:
            continue
        interiors = [(t + 1, l + 1, b - 1, r - 1) for t, l, b, r in boxes]
        largest = max(range(count), key=lambda i: (interiors[i][2] - interiors[i][0] + 1)
                      * (interiors[i][3] - interiors[i][1] + 1))
        if witness:
            # Teach odd and even room widths, with an axis-only ballot as well.
            odd_choices = [i for i, (_, l, _, r) in enumerate(interiors)
                           if (r - l + 1) % 2 and i != largest]
            if not odd_choices or not any((r - l + 1) % 2 == 0 for _, l, _, r in interiors):
                continue
            q = count - len(pigments)
            # One abstaining room; each minority pigment owns one real ballot.
            if q < 2:
                continue
            labels = [None] * count
            axis_room = odd_choices[int(rng.integers(len(odd_choices)))]
            labels[axis_room] = winning_color
            remaining = [i for i in range(count) if i not in (largest, axis_room)]
            rng.shuffle(remaining)
            other_colors = [c for c in pigments if c not in (winning_color, losing_color)]
            if witness == "counterfactual":
                labels[largest] = losing_color
                for i in remaining[:q - 1]:
                    labels[i] = winning_color
                for i, color in zip(remaining[q - 1:], other_colors):
                    labels[i] = color
            else:
                # The contrasting demonstration breaks "choose the rarest
                # pigment": now the real winner also owns the dense room.
                labels[largest] = winning_color
                for i in remaining[:q - 2]:
                    labels[i] = winning_color
                for i, color in zip(remaining[q - 2:], [losing_color] + other_colors):
                    labels[i] = color
        else:
            axis_room = None
            labels = None
            for _ in range(80):
                proposed = [None if rng.random() < .17 else int(rng.choice(pigments)) for _ in range(count)]
                votes = Counter(c for c in proposed if c is not None)
                if len(votes) >= 2 and votes.get(winning_color, 0) >= 2:
                    if all(votes[winning_color] > n for c, n in votes.items() if c != winning_color):
                        labels = proposed
                        break
            if labels is None:
                labels = [winning_color, winning_color, losing_color] + [None] * (count - 3)
        top_pad, bottom_pad = int(rng.integers(0, min(3, 30 - h) + 1)), 0
        bottom_pad = int(rng.integers(0, min(3, 30 - h - top_pad) + 1))
        left_pad = int(rng.integers(0, min(3, 30 - w) + 1))
        right_pad = int(rng.integers(0, min(3, 30 - w - left_pad) + 1))
        shape = (h + top_pad + bottom_pad, w + left_pad + right_pad)
        placed = [(t + top_pad, l + left_pad, b + top_pad, r + left_pad) for t, l, b, r in boxes]
        inner = [(t + 1, l + 1, b - 1, r - 1) for t, l, b, r in placed]
        mask = _walls(shape, placed)
        inp = np.full(shape, bg, dtype=np.int8)
        inp[mask] = wall
        valid = True
        for i, ((t, l, b, r), label) in enumerate(zip(inner, labels)):
            room = inp[t:b + 1, l:r + 1]
            if label is not None:
                # The two witnesses deliberately put their dense core on
                # opposite sides of the election, defeating both pixel counts
                # and the inverse shortcut of choosing the rarest pigment.
                core = _core(rng, *room.shape, tiny=witness and i != largest,
                             axis_only=witness and i == axis_room,
                             dense_min=2 * labels.count(winning_color) + 2 if witness and i == largest else 0)
                if core is None:
                    valid = False
                    break
                room[core] = label
            if witness and i != largest:
                amount = int(rng.integers(3, 9))
                noise_color = losing_color if witness == "counterfactual" else winning_color
                used = _add_unmatched(rng, room, bg, [noise_color], amount)
                if used <= (0 if label is None else int(core.sum())):
                    valid = False
                    break
            elif not witness and rng.random() < .78:
                amount = int(rng.integers(1, min(13, room.size // 3) + 1))
                _add_unmatched(rng, room, bg, pigments, amount)
        if not valid:
            continue
        _damage(rng, inp, mask, crack, require=witness)
        frequencies = np.bincount(inp.ravel().astype(int), minlength=10)
        if frequencies[bg] <= max(frequencies[c] for c in range(10) if c != bg):
            continue
        votes = Counter(c for c in labels if c is not None)
        out = _render(shape, bg, wall, mask, inner, _plurality(votes))
        analysis = _decode(inp)
        assert analysis["ballots"] == votes and np.array_equal(solve(inp), out)
        if witness == "counterfactual":
            alternatives = _alternatives(analysis)
            if alternatives["raw_room_winner"] == winning_color or alternatives["support_pixel_winner"] == winning_color:
                continue
        elif witness == "control":
            if any(c != winning_color for c in _alternatives(analysis).values()):
                continue
        return {"input": inp, "output": out}
    # A spacious construction still uses the same sampling rule and does not
    # change counts or labels. Failure is explicit rather than silently narrowing.
    raise RuntimeError("bounded room construction exhausted")


def _alternatives(a):
    raw_votes, support_pixels, total_pixels = Counter(), Counter(), Counter()
    for room in a["rooms"]:
        raw = room["raw"]
        if raw:
            top = max(raw.values())
            leaders = [c for c, n in raw.items() if n == top]
            if len(leaders) == 1:
                raw_votes[leaders[0]] += 1
        if room["ballot"] is not None:
            support_pixels[room["ballot"]] += room["support"]
        total_pixels.update(raw)
    def unique(counts):
        if not counts:
            return None
        leaders = [c for c, n in counts.items() if n == max(counts.values())]
        return leaders[0] if len(leaders) == 1 else None
    return {"raw_room_winner": unique(raw_votes),
            "support_pixel_winner": unique(support_pixels),
            "raw_pixel_winner": unique(total_pixels)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    pigments = int(rng.integers(2, 5))
    parameters = {"pigments": pigments, "palette": list(map(int, rng.permutation(10)[:3 + pigments])),
                  "local_palette": bool(rng.integers(2)), "ceiling": int(rng.integers(6, 9))}
    puzzle, seen = {"train": [], "test": []}, set()
    first_winner = None
    for split, number in (("train", nt), ("test", ne)):
        for i in range(number):
            witness = ("counterfactual" if i == 0 else "control") if split == "train" and i < 2 else None
            if witness:
                count = int(rng.integers(pigments + 2, parameters["ceiling"] + 1))
            elif split == "train":
                count = int(rng.integers(4, parameters["ceiling"] + 1))
            else:
                count = int(rng.integers(parameters["ceiling"] + 1, min(10, parameters["ceiling"] + 3) + 1))
            winner_index = i % pigments
            for _ in range(12):
                pair = _make_pair(rng, parameters, count, witness, winner_index,
                                  first_winner if split == "train" and i == 1 else None)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train" and i == 0:
                        first_winner = _decode(pair["input"])["winner"]
                    break
            else:
                raise RuntimeError("duplicate input after bounded retries")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, o = _grid(pair["input"]), _grid(pair["output"])
            assert g.shape == o.shape
            assert _key(g) not in seen
            seen.add(_key(g))
            a = _decode(g)
            assert 4 <= len(a["rooms"]) <= 10
            assert len(a["ballots"]) >= 2
            roles = {a["background"], a["wall"], a["crack"]}
            assert 2 <= len(set(map(int, np.unique(g))) - roles) <= 4
            gaps = [tuple(map(int, p)) for p in np.argwhere(a["repaired"])]
            for i, (y, x) in enumerate(gaps):
                assert all(abs(y - yy) + abs(x - xx) >= 3 for yy, xx in gaps[:i])
                assert sum(bool(a["mask"][yy, xx]) for yy, xx in
                           ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
                           if 0 <= yy < g.shape[0] and 0 <= xx < g.shape[1]) == 2
            for room in a["rooms"]:
                t, l, b, r = room["box"]
                assert b - t + 1 >= 3 and r - l + 1 >= 4
            expected = _render(g.shape, a["background"], a["wall"], a["mask"],
                               [s["box"] for s in a["rooms"]], a["winner"])
            assert np.array_equal(o, expected), "wrong consensus or wall recovery"
            assert np.array_equal(o[a["mask"]], np.full(a["mask"].sum(), a["wall"]))
            assert not np.array_equal(g, o)
            analyses[split].append(a)
    train = analyses["train"]
    assert len({a["winner"] for a in train}) >= 2, "literal winner shortcut"
    assert {s["axis2"] % 2 for a in train for s in a["rooms"]} == {0, 1}
    assert any(_has_partial(a["mask"]) for a in train), "partial-divider witness missing"
    assert any(a["repaired"].any() for a in train), "wall repair never taught"
    assert any(s["ballot"] is None and s["unmatched"] for a in train for s in a["rooms"])
    assert any(s["axis_cells"] for a in train for s in a["rooms"]), "self-support not taught"
    # Explicit omissions: without B, noisy room modes elect another pigment;
    # without A's one-room unit, even certified pixel totals elect another.
    for omission in ("raw_room_winner", "support_pixel_winner", "raw_pixel_winner"):
        assert any(_alternatives(a)[omission] is not None
                   and _alternatives(a)[omission] != a["winner"] for a in train), omission
    assert any(all(c == a["winner"] for c in _alternatives(a).values()) for a in train), "rare-pigment shortcut"
    assert min(len(a["rooms"]) for a in analyses["test"]) > max(len(a["rooms"]) for a in train)


def _euler_rooms(mask):
    """Connected wall graph's cycle rank, excluding filled unit-square faces."""
    edges = int((mask[:, :-1] & mask[:, 1:]).sum() + (mask[:-1] & mask[1:]).sum())
    squares = int((mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]).sum())
    return edges - int(mask.sum()) + 1 - squares


def self_test():
    """Bounded 42-seed check of topology, ballots, counterfactuals, and variation."""
    shapes, topology, votes, palettes, keys, room_sizes = set(), set(), set(), set(), set(), set()
    features = Counter()
    first = None
    for seed in range(42):
        p = generate_puzzle(seed)
        if seed == 0:
            first = p
        fingerprint = []
        for split in ("train", "test"):
            for pair in p[split]:
                g, out = pair["input"], pair["output"]
                a = _decode(g)
                assert _euler_rooms(a["mask"]) == len(a["rooms"])
                # Coordinate partner enumeration checks the vectorized support.
                tally = Counter()
                for s in a["rooms"]:
                    t, l, b, r = s["box"]
                    retained = [(y, x) for y in range(t, b + 1) for x in range(l, r + 1)
                                if g[y, x] != a["background"] and g[y, x] == g[y, l + r - x]]
                    assert len(retained) == s["support"]
                    colors = {int(g[y, x]) for y, x in retained}
                    if colors:
                        tally[next(iter(colors))] += 1
                    room_sizes.add((b - t + 1, r - l + 1))
                assert tally == a["ballots"]
                assert np.array_equal(solve(g.copy()), out)
                assert np.array_equal(solve(np.fliplr(g)), np.fliplr(out))
                assert np.array_equal(solve(np.flipud(g)), np.flipud(out))
                permutation = np.array([9, 5, 7, 2, 8, 0, 3, 6, 1, 4], dtype=np.int8)
                assert np.array_equal(solve(permutation[g]), permutation[out])
                shapes.add(g.shape)
                palettes.add((a["background"], a["wall"], tuple(sorted(a["ballots"]))))
                votes.add(tuple(sorted(a["ballots"].values())))
                # Preserve relative split endpoints but remove numeric spacing.
                t, l, b, r = a["frame"]
                boxes = [s["box"] for s in a["rooms"]]
                ys = sorted({v for box in boxes for v in (box[0] - 1, box[2] + 1)})
                xs = sorted({v for box in boxes for v in (box[1] - 1, box[3] + 1)})
                topology.add(tuple(sorted((ys.index(y - 1), xs.index(x - 1), ys.index(z + 1), xs.index(q + 1))
                                          for y, x, z, q in boxes)))
                features["partial"] += _has_partial(a["mask"])
                features["intact"] += not a["repaired"].any()
                features["edge"] += t == 0 or l == 0 or b == g.shape[0] - 1 or r == g.shape[1] - 1
                features["abstain"] += any(s["ballot"] is None for s in a["rooms"])
                features["nonmajority"] += 2 * a["ballots"][a["winner"]] <= sum(a["ballots"].values())
                fingerprint.append((_key(g), _key(out)))
        keys.add(tuple(fingerprint))
    assert len(keys) == 42 and len(shapes) >= 40 and len(topology) >= 50
    assert len(votes) >= 12 and len(palettes) >= 35 and len(room_sizes) >= 50
    assert all(features[k] >= 5 for k in ("partial", "intact", "edge", "abstain", "nonmajority"))
    repeat = generate_puzzle(0)
    assert all(_key(a[k]) == _key(b[k]) for s in first for a, b in zip(first[s], repeat[s]) for k in a)
    for i, (nt, ne) in enumerate(((2, 1), (10, 1), (10, 4))):
        validate_generated_puzzle(generate_puzzle(500 + i, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Reject a plausible but wrong minority fill, including on a small request.
    bad = generate_puzzle(901, 2, 1)
    a = _decode(bad["test"][0]["input"])
    t, l, _, _ = a["rooms"][0]["box"]
    bad["test"][0]["output"][t, l] = next(c for c in a["ballots"] if c != a["winner"])
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong fill accepted")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes), "topologies": len(topology),
            "vote_multisets": len(votes), "role_palettes": len(palettes),
            "room_sizes": len(room_sizes), "features": dict(features)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
