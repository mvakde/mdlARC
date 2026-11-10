"""Clockwork mosaic kits: local color rotations coupled to finite exact cover.

Only the standard library and NumPy are needed. No files are read or written by
generation, solving, validation, or self_test. See task_spec.md for the grammar.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


HOST_SEEDS = (0, 942744556, 872264304, 690984176, 730942509, 480602030,
              22798323, 336185769, 747991893, 497526474, 255218430, 28708083,
              361615421, 402268587, 429668876, 755762431, 500551875, 977785716,
              177563789, 219392472, 560430713, 776435657, 121266910, 795397181,
              786744075, 290580641, 336362378, 98481196, 940445682, 346475169,
              179626001, 661844816, 532538016, 830015882, 604587797, 261624811,
              265138389, 591410668, 438556386, 412905175, 991366543, 36295299)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("num_test must be in 1..4 and less than num_train")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, comp = [start], []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
        result.append(comp)
    return result


def _trim(a):
    rr, cc = np.where(a >= 0)
    return a[rr.min():rr.max()+1, cc.min():cc.max()+1].copy()


def _asymmetric(mask):
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _configure(cards, bg, counts, *, direction=1, arrays=False):
    """Rotate layers about the visible local centers; -1 means empty space.

    Construction uses array rotations. Solving uses direct coordinate updates.
    """
    result = []
    for card in cards:
        n = card.shape[0]
        target = np.full((n, n), -1, dtype=np.int8)
        for color in np.unique(card):
            color = int(color)
            if color == bg:
                continue
            turns = direction * counts.get(color, 0) % 4
            if arrays:
                coords = np.argwhere(np.rot90(card == color, -turns))
            else:
                coords = []
                for r, c in np.argwhere(card == color):
                    r, c = int(r), int(c)
                    for _ in range(turns):
                        r, c = c, n-1-r
                    coords.append((r, c))
            for r, c in coords:
                if target[r, c] != -1:
                    raise ValueError("Rotated parts overlap inside a kit")
                target[r, c] = color
        result.append(_trim(target))
    return result


def _parse(inp):
    """Recover frames, kit centers, stencil and unary bars from this input only."""
    vals, nums = np.unique(inp, return_counts=True)
    if np.sum(nums == nums.max()) != 1:
        raise ValueError("Background must be the unique most frequent color")
    bg = int(vals[nums.argmax()])
    frame_candidates = []
    for color in vals:
        color = int(color)
        if color == bg:
            continue
        boxes = []
        for comp in _components(inp == color):
            rows, cols = zip(*comp)
            r, c, h, w = min(rows), min(cols), max(rows)-min(rows)+1, max(cols)-min(cols)+1
            if h < 5 or w < 5 or len(comp) != 2*h+2*w-4:
                break
            border = np.zeros((h, w), dtype=bool)
            border[0] = border[-1] = True
            border[:, 0] = border[:, -1] = True
            if not np.array_equal(inp[r:r+h, c:c+w] == color, border):
                break
            boxes.append((r, c, h, w))
        else:
            if 3 <= len(boxes) <= 6:
                frame_candidates.append((color, boxes))
    if len(frame_candidates) != 1:
        raise ValueError("Exactly one structural frame color is required")
    frame, boxes = frame_candidates[0]
    boxes.sort(key=lambda b: -(b[2]-2)*(b[3]-2))
    board_box = boxes[0]
    if (boxes[0][2]-2)*(boxes[0][3]-2) <= (boxes[1][2]-2)*(boxes[1][3]-2):
        raise ValueError("The destination frame must be uniquely largest")
    framed = np.zeros(inp.shape, dtype=bool)
    for r, c, h, w in boxes:
        if np.any(framed[max(0, r-1):r+h+1, max(0, c-1):c+w+1]):
            raise ValueError("Frames must have a separating background gap")
        framed[r:r+h, c:c+w] = True
    r, c, h, w = board_box
    board = inp[r+1:r+h-1, c+1:c+w-1].copy()
    board_colors = [int(v) for v in np.unique(board) if v != bg]
    if len(board_colors) != 1 or board_colors[0] == frame or not np.any(board == bg):
        raise ValueError("Destination must be a two-color stencil")
    board_color = board_colors[0]
    cards = []
    for rr, cc, hh, ww in boxes[1:]:
        if hh != ww or not 5 <= hh <= 7:
            raise ValueError("Source kits need square interiors of size 3..5")
        card = inp[rr+1:rr+hh-1, cc+1:cc+ww-1].copy()
        palette = set(int(v) for v in np.unique(card)) - {bg}
        if len(palette) < 2 or frame in palette or board_color in palette or not np.any(card == bg):
            raise ValueError("Each kit needs empty space and at least two payload colors")
        # Every connected part of a color is a filled axis-aligned rectangle.
        for color in palette:
            for comp in _components(card == color):
                yy, xx = zip(*comp)
                if len(comp) != (max(yy)-min(yy)+1)*(max(xx)-min(xx)+1):
                    raise ValueError("A kit part is not rectangular")
        cards.append(card)
    palette = set(int(v) for card in cards for v in np.unique(card)) - {bg}
    if not 2 <= len(palette) <= 4:
        raise ValueError("A pair needs two through four payload colors")
    exterior = (~framed) & (inp != bg)
    counts, orientations = {}, {}
    for color in np.unique(inp[exterior]):
        color = int(color)
        if color not in palette:
            raise ValueError("Unmatched exterior counter")
        points = np.argwhere(exterior & (inp == color))
        yy, xx = points.T
        horizontal = np.ptp(yy) == 0 and np.ptp(xx)+1 == len(points)
        vertical = np.ptp(xx) == 0 and np.ptp(yy)+1 == len(points)
        if not (horizontal or vertical) or not 1 <= len(points) <= 9:
            raise ValueError("Each marked color needs one straight unary bar of length 1..9")
        for y, x in points:
            halo = inp[max(0, y-1):y+2, max(0, x-1):x+2]
            if np.any((halo != bg) & (halo != color)) or np.any(
                    framed[max(0, y-1):y+2, max(0, x-1):x+2]):
                raise ValueError("Counters must be detached from frames and other counters")
        counts[color] = len(points)
        orientations[color] = "single" if len(points) == 1 else "horizontal" if horizontal else "vertical"
    if not counts:
        raise ValueError("At least one unary counter is required")
    shapes = _configure(cards, bg, counts)
    if sum(np.count_nonzero(s >= 0) for s in shapes) != np.count_nonzero(board == bg):
        raise ValueError("Finite kit area must equal destination vacancy area")
    return dict(background=bg, frame=frame, board_color=board_color, board=board,
                board_box=board_box, boxes=boxes, cards=cards, counts=counts,
                orientations=orientations, shapes=shapes, mask=board == bg)


def _cover(mask, shapes, *, max_nodes=None, required=None, limit=2):
    """Enumerate rendered answers, grouping interchangeable copies.

    Empty cells inside a kit's bounding box reserve no space. They may cover
    board background or material from another kit. Only occupied cells matter.
    """
    coords = [tuple(p) for p in np.argwhere(mask).tolist()]
    if sum(int(np.count_nonzero(s >= 0)) for s in shapes) != len(coords):
        return {}, {"nodes": 0, "dead_ends": 0, "complete": True}
    index = {p: i for i, p in enumerate(coords)}
    groups, lookup = [], {}
    for shape in shapes:
        key = _key(shape)
        if key not in lookup:
            lookup[key] = len(groups)
            groups.append([shape, 0])
        groups[lookup[key]][1] += 1
    placements, by_cell = [], [[] for _ in coords]
    h, w = mask.shape
    for g, (shape, _) in enumerate(groups):
        sh, sw = shape.shape
        cells = [(int(y), int(x), int(shape[y, x])) for y, x in np.argwhere(shape >= 0)]
        for r in range(h-sh+1):
            for c in range(w-sw+1):
                if any((r+y, c+x) not in index for y, x, _ in cells):
                    continue
                if required is not None and any(required[r+y, c+x] != v for y, x, v in cells):
                    continue
                assignments = tuple((index[r+y, c+x], v) for y, x, v in cells)
                bits = sum(1 << i for i, _ in assignments)
                pid = len(placements)
                placements.append((g, bits, assignments, (r, c)))
                for i, _ in assignments:
                    by_cell[i].append(pid)
    full = (1 << len(coords))-1
    remaining = [n for _, n in groups]
    render = [-1]*len(coords)
    outcomes, path, dead = {}, [], set()
    stats = {"nodes": 0, "dead_ends": 0, "complete": True}

    def search(covered):
        stats["nodes"] += 1
        if max_nodes is not None and stats["nodes"] > max_nodes:
            stats["complete"] = False
            return False
        if covered == full:
            if any(remaining):
                return False
            outcomes.setdefault(tuple(render), tuple(path))
            return True
        state = covered, tuple(remaining)
        if state in dead:
            return False
        best = None
        for i in range(len(coords)):
            if covered & (1 << i):
                continue
            options = [p for p in by_cell[i]
                       if remaining[placements[p][0]] and not covered & placements[p][1]]
            if not options:
                dead.add(state)
                stats["dead_ends"] += 1
                return False
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
        found = False
        for pid in best:
            g, bits, assignments, pos = placements[pid]
            remaining[g] -= 1
            for i, v in assignments:
                render[i] = v
            path.append((g, pos, assignments))
            found = search(covered | bits) or found
            path.pop()
            for i, _ in assignments:
                render[i] = -1
            remaining[g] += 1
            if not stats["complete"] or len(outcomes) >= limit:
                return found
        if not found:
            dead.add(state)
        return found

    search(0)
    return outcomes, stats


def _answer(info, *, max_nodes=None):
    outcomes, stats = _cover(info["mask"], info["shapes"], max_nodes=max_nodes)
    if not stats["complete"]:
        raise ValueError("Unresolved exact-cover search")
    if len(outcomes) != 1:
        raise ValueError("Expected exactly one rendered packing")
    values, path = next(iter(outcomes.items()))
    result = info["board"].copy()
    result[info["mask"]] = values
    return result, path, stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only prediction: no seed, examples, generation metadata or cache."""
    return _answer(_parse(_grid(input_grid)))[0]


def _counterfactual_possible(info, answer, counts, *, direction=1):
    try:
        shapes = _configure(info["cards"], info["background"], counts, direction=direction)
    except ValueError:
        return False
    outcomes, _ = _cover(info["mask"], shapes, required=answer, limit=1)
    return bool(outcomes)


def _framed(inner, frame):
    h, w = inner.shape
    result = np.full((h+2, w+2), frame, dtype=np.int8)
    result[1:-1, 1:-1] = inner
    return result


def _starts(blocked, h, w):
    bh, bw = blocked.shape
    if h > bh or w > bw:
        return np.empty((0, 2), dtype=int)
    integral = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    return np.argwhere(sums == 0)


def _layout(rng, board, cards, bg, frame, counts, forced):
    objects = [_framed(board, frame)] + sorted(
        [_framed(card, frame) for card in cards], key=lambda a: -a.size)
    for _ in range(35):
        h, w = (int(v) for v in rng.integers(24, 31, size=2))
        inp = np.full((h, w), bg, dtype=np.int8)
        blocked = np.zeros((h, w), dtype=bool)
        good = True
        for obj in objects:
            oh, ow = obj.shape
            starts = _starts(blocked, oh, ow)
            if not len(starts):
                good = False
                break
            r, c = (int(v) for v in starts[int(rng.integers(len(starts)))])
            inp[r:r+oh, c:c+ow] = obj
            blocked[max(0, r-1):min(h, r+oh+1), max(0, c-1):min(w, c+ow+1)] = True
        if not good:
            continue
        for color in sorted(counts, key=lambda v: -counts[v]):
            length = counts[color]
            options = []
            for orientation in (0, 1):
                if color in forced and orientation != forced[color]:
                    continue
                oh, ow = (1, length) if orientation == 0 else (length, 1)
                starts = _starts(blocked, oh, ow)
                if len(starts):
                    options.append((oh, ow, starts))
            if not options:
                good = False
                break
            oh, ow, starts = options[int(rng.integers(len(options)))]
            r, c = (int(v) for v in starts[int(rng.integers(len(starts)))])
            inp[r:r+oh, c:c+ow] = color
            blocked[max(0, r-1):min(h, r+oh+1), max(0, c-1):min(w, c+ow+1)] = True
        if good:
            return inp
    return None


def _make_card(rng, n, bg, colors, counts, all_colors=False, informative=False):
    """Sample rectangular parts; never repair overlaps using overwrite priority."""
    for _ in range(100):
        selected = list(colors) if all_colors else [int(v) for v in rng.choice(
            colors, size=int(rng.integers(2, min(3, len(colors))+1)), replace=False)]
        rng.shuffle(selected)
        source = np.full((n, n), bg, dtype=np.int8)
        rotated_used = np.zeros((n, n), dtype=bool)
        good = True
        for color in selected:
            placed = False
            for _ in range(30):
                ph, pw = int(rng.integers(1, min(n-1, 3)+1)), int(rng.integers(1, min(n-1, 3)+1))
                r, c = int(rng.integers(n-ph+1)), int(rng.integers(n-pw+1))
                mask = np.zeros((n, n), dtype=bool)
                mask[r:r+ph, c:c+pw] = True
                if not informative and rng.random() < .24:
                    # A second detached rectangle belongs to the same color mask.
                    qh, qw = int(rng.integers(1, 3)), int(rng.integers(1, 3))
                    y, x = int(rng.integers(n-qh+1)), int(rng.integers(n-qw+1))
                    if not np.any(mask[max(0, y-1):y+qh+1, max(0, x-1):x+qw+1]):
                        mask[y:y+qh, x:x+qw] = True
                turned = np.rot90(mask, -counts.get(color, 0) % 4)
                if (np.any(source[mask] != bg) or np.any(rotated_used & turned)
                        or (informative and not _asymmetric(mask))):
                    continue
                source[mask] = color
                rotated_used |= turned
                placed = True
                break
            if not placed:
                good = False
                break
        area = int(np.count_nonzero(source != bg))
        if good and 3 <= area <= int(.80*n*n) and any(
                np.count_nonzero(source == color) >= 2 for color in selected):
            return source
    return None


def _plant(rng, shapes, height, width, board_color):
    occupied = np.zeros((height, width), dtype=bool)
    output = np.full((height, width), board_color, dtype=np.int8)
    boxes = []
    for shape in shapes:
        sh, sw = shape.shape
        cells = np.argwhere(shape >= 0)
        candidates, weights = [], []
        for r in range(height-sh+1):
            for c in range(width-sw+1):
                if any(occupied[r+y, c+x] for y, x in cells):
                    continue
                contacts = sum(0 <= r+y+dy < height and 0 <= c+x+dx < width
                               and occupied[r+y+dy, c+x+dx]
                               for y, x in cells for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)))
                if boxes and not contacts:
                    continue
                overlap = sum(max(0, min(r+sh, y+h)-max(r, y)) *
                              max(0, min(c+sw, x+w)-max(c, x)) for y, x, h, w in boxes)
                candidates.append((r, c))
                weights.append(1+contacts*contacts+2*overlap)
        if not candidates:
            return None
        probabilities = np.asarray(weights, dtype=float)
        probabilities /= probabilities.sum()
        r, c = candidates[int(rng.choice(len(candidates), p=probabilities))]
        for y, x in cells:
            occupied[r+y, c+x] = True
            output[r+y, c+x] = shape[y, x]
        boxes.append((r, c, sh, sw))
    return occupied, output


def _features(info, path, stats):
    coords = [tuple(p) for p in np.argwhere(info["mask"]).tolist()]
    owner = np.full(info["mask"].shape, -1, dtype=int)
    bboxes = []
    for kit, (_, _, assignments) in enumerate(path):
        points = [coords[i] for i, _ in assignments]
        for p in points:
            owner[p] = kit
        yy, xx = zip(*points)
        bboxes.append((min(yy), min(xx), max(yy)+1, max(xx)+1))
    fused = max(len({int(owner[p]) for p in comp}) for comp in _components(info["mask"]))
    interlocked = any(min(a[2], b[2]) > max(a[0], b[0]) and
                      min(a[3], b[3]) > max(a[1], b[1])
                      for i, a in enumerate(bboxes) for b in bboxes[i+1:])
    sizes = tuple(card.shape[0] for card in info["cards"])
    return dict(kits=len(path), sizes=sizes, board_shape=info["board"].shape,
                input_shape=None, colors=len(set(int(v) for s in info["shapes"] for v in s.flat if v >= 0)),
                parts=sum(len(_components(card == color)) for card in info["cards"]
                          for color in np.unique(card) if color != info["background"]),
                fused=fused, interlocked=interlocked, nodes=stats["nodes"],
                dead_ends=stats["dead_ends"], holes=int(info["mask"].sum()),
                repeated=len({_key(s) for s in info["shapes"]}) < len(info["shapes"]),
                multi_part_color=any(len(_components(card == color)) > 1
                                     for card in info["cards"] for color in np.unique(card)
                                     if color != info["background"]))


def _build_pair(rng, witness, long_training, testing):
    for _ in range(240):
        colors_all = [int(v) for v in rng.permutation(10)]
        bg, frame, board_color = colors_all[:3]
        color_count = 3 if witness >= 0 else int(rng.integers(2, 5))
        colors = colors_all[3:3+color_count]
        forced = {}
        if witness == 0:
            lengths = [1, 2, 2]
            flip = int(rng.integers(2))
            forced = {colors[1]: flip, colors[2]: 1-flip}
        elif witness == 1:
            lengths = [3, 0, 8 if long_training else 4]
        else:
            lengths = [int(v) for v in rng.integers(0, 10 if long_training or testing else 5, color_count)]
            if not any(lengths):
                lengths[0] = 1
        if testing:
            lengths[int(rng.integers(color_count))] = int(rng.integers(5, 10))
        counts = {c: n for c, n in zip(colors, lengths) if n}
        number = int(rng.integers(2, 4)) if witness == 0 else int(rng.integers(2, 6))
        cards = []
        for i in range(number):
            if i > 0 and witness < 0 and rng.random() < .18:
                cards.append(cards[int(rng.integers(len(cards)))].copy())
                continue
            n = int(rng.integers(4, 6)) if i == 0 and witness >= 0 else int(rng.integers(3, 6))
            card = _make_card(rng, n, bg, colors, counts,
                              all_colors=i == 0, informative=witness >= 0 and i == 0)
            if card is None:
                break
            cards.append(card)
        if len(cards) != number:
            continue
        shapes = _configure(cards, bg, counts, arrays=True)
        height, width = (int(v) for v in rng.integers(7, 13, 2))
        planted = _plant(rng, shapes, height, width, board_color)
        if planted is None:
            continue
        mask, expected = planted
        if mask.all():
            continue
        outcomes, stats = _cover(mask, shapes, max_nodes=6000)
        wanted = tuple(int(v) for v in expected[mask])
        if not stats["complete"] or len(outcomes) != 1 or wanted not in outcomes:
            continue
        board = np.full(mask.shape, board_color, dtype=np.int8)
        board[mask] = bg
        # Check pedagogical counterfactuals before spending work on canvas layout.
        temporary = dict(cards=cards, background=bg, mask=mask)
        if witness >= 0 and (_counterfactual_possible(temporary, expected, {}) or
                _counterfactual_possible(temporary, expected, counts, direction=-1)):
            continue
        if witness == 0 and any(_counterfactual_possible(
                temporary, expected, {c: k for c in colors}) for k in range(4)):
            # One shared turn of every layer cannot explain this demonstration.
            continue
        if witness == 1:
            absent_colors = [c for c in colors if c not in counts]
            if any(_counterfactual_possible(temporary, expected,
                       counts | {c: k for c in absent_colors}) for k in (1, 2, 3)):
                continue
        if testing:
            informative_long = False
            for color, length in counts.items():
                if length < 5 or not any(_asymmetric(card == color) for card in cards
                                         if np.any(card == color)):
                    continue
                changed_counts = dict(counts)
                changed_counts[color] = 1 if length % 4 == 0 else 0
                informative_long |= not _counterfactual_possible(temporary, expected, changed_counts)
            if not informative_long:
                continue
        inp = _layout(rng, board, cards, bg, frame, counts, forced)
        if inp is None:
            continue
        try:
            info = _parse(inp)
            answer, path, checked = _answer(info, max_nodes=6000)
        except ValueError:
            continue
        if not np.array_equal(answer, expected):
            continue
        features = _features(info, path, checked)
        if features["fused"] < 2:
            continue
        return {"input": inp, "output": expected.copy()}
    raise RuntimeError("Bounded construction exhausted; no ambiguous fallback is emitted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    long_training = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            witness = i if split == "train" and i < 2 else -1
            for _ in range(12):
                pair = _build_pair(rng, witness, long_training, split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Distinct-input construction exhausted")
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check visible grammar, exact cover, conservation and training witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, output = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "Repeated input"
            seen.add(_key(inp))
            info = _parse(inp)
            answer, path, stats = _answer(info)
            assert np.array_equal(answer, output), "Incorrect packing or crop"
            assert np.array_equal(output[~info["mask"]], info["board"][~info["mask"]])
            assert not np.any(output == info["background"])
            assert Counter(int(v) for card in info["cards"] for v in card.flat
                           if v != info["background"]) == Counter(int(v) for v in output[info["mask"]])
            assert len(path) == len(info["cards"]), "Every physical kit must be used once"
            features = _features(info, path, stats)
            assert features["fused"] >= 2, "Need a merged region combining kits"
            if split == "train":
                training.append((info, output))
    # Each of the two required demonstrations defeats literal copying and the
    # wrong rotation direction, even after allowing any legal translations.
    for info, output in training[:2]:
        assert not _counterfactual_possible(info, output, {}), "Rotation omission is a shortcut"
        assert not _counterfactual_possible(info, output, info["counts"], direction=-1)
    first, first_output = training[0]
    first_colors = set(int(v) for card in first["cards"] for v in np.unique(card)) - {first["background"]}
    assert not any(_counterfactual_possible(first, first_output, {c: k for c in first_colors})
                   for k in range(4)), "A whole-kit rotation explains the first demonstration"
    second, second_output = training[1]
    absent_colors = (set(int(v) for card in second["cards"] for v in np.unique(card))
                     - {second["background"]} - set(second["counts"]))
    assert absent_colors
    assert not any(_counterfactual_possible(second, second_output,
                       second["counts"] | {c: k for c in absent_colors}) for k in (1, 2, 3)), \
        "The second demonstration must distinguish the absent-counter default"
    for pair in puzzle["test"]:
        info, output = _parse(_grid(pair["input"])), _grid(pair["output"])
        informative_long = False
        for color, length in info["counts"].items():
            if length < 5 or not any(_asymmetric(card == color) for card in info["cards"]
                                     if np.any(card == color)):
                continue
            changed = dict(info["counts"])
            changed[color] = 1 if length % 4 == 0 else 0
            informative_long |= not _counterfactual_possible(info, output, changed)
        assert informative_long, "The long-counter test shift must change a consequential layer"
    informative = set()
    absent, cycle, orientation, independent = False, False, False, False
    for info, _ in training:
        seen_orientations = {}
        for card in info["cards"]:
            residues = set()
            for color in np.unique(card):
                color = int(color)
                if color == info["background"] or not _asymmetric(card == color):
                    continue
                count = info["counts"].get(color, 0)
                absent |= count == 0
                cycle |= count in (4, 8)
                if count:
                    informative.add(count % 4)
                    residues.add(count % 4)
                    seen_orientations.setdefault(count, set()).add(info["orientations"][color])
            independent |= len(residues) >= 2
        orientation |= any({"horizontal", "vertical"} <= v for v in seen_orientations.values())
    assert {1, 2, 3} <= informative and absent and cycle and orientation and independent
    assert nt > ne


def self_test():
    """42 fixed seeds, transient arrays only; no diagnostics accumulate on disk."""
    puzzles, rows = set(), []
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                info = _parse(pair["input"])
                _, path, stats = _answer(info)
                f = _features(info, path, stats)
                f.update(background=info["background"], frame=info["frame"],
                         board_color=info["board_color"], split=split,
                         lengths=tuple(info["counts"].values()),
                         orientations=tuple(info["orientations"].values()),
                         input_shape=pair["input"].shape)
                rows.append(f)
                digest.update(repr(pair["input"].shape).encode())
                digest.update(pair["input"].tobytes())
                digest.update(pair["output"].tobytes())
        puzzles.add(digest.hexdigest())
    assert len(puzzles) == 42
    assert {r["background"] for r in rows} == set(range(10))
    assert {n for r in rows for n in r["sizes"]} == {3, 4, 5}
    assert {r["kits"] for r in rows} == {2, 3, 4, 5}
    assert {r["colors"] for r in rows} == {2, 3, 4}
    assert len({r["board_shape"] for r in rows}) >= 20
    assert len({r["input_shape"] for r in rows}) >= 20
    assert sum(r["interlocked"] for r in rows) >= 30
    assert sum(r["multi_part_color"] for r in rows) >= 10
    assert any(r["repeated"] for r in rows)
    assert max(r["fused"] for r in rows) >= 4
    assert {n for r in rows if r["split"] == "test" for n in r["lengths"] if n >= 5} == set(range(5, 10))
    # Inventory, ambiguity and transparent space have explicit small regressions.
    a = np.array([[1, 2]], dtype=np.int8)
    assert not _cover(np.ones((1, 4), dtype=bool), [a])[0]
    assert len(_cover(np.ones((1, 4), dtype=bool), [a, a])[0]) == 1
    assert len(_cover(np.ones((1, 4), dtype=bool), [a, np.array([[2, 1]], dtype=np.int8)])[0]) == 2
    shape1 = np.array([[1, -1], [1, 2]], dtype=np.int8)
    shape2 = np.array([[3, 3], [-1, 4]], dtype=np.int8)
    assert _cover(np.ones((2, 3), dtype=bool), [shape1, shape2])[0]
    for a_count in range(2, 11):
        for b_count in range(1, min(4, a_count-1)+1):
            assert _counts(a_count, b_count) == (a_count, b_count)
    for a_count, b_count in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0, a_count, b_count)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    one, two = generate_puzzle(81, 2, 1), generate_puzzle(81, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in ("train", "test")
               for a, b in zip(one[s], two[s]) for k in ("input", "output"))
    # These transformations change layout, bar orientation, dimensions and
    # literal colors while preserving the visible rule and clockwise handedness.
    relabel = np.array([7, 3, 9, 0, 6, 1, 8, 4, 2, 5], dtype=np.int8)
    for pair in one["train"] + one["test"]:
        unchanged = pair["input"].copy()
        for turns in (1, 2, 3):
            assert np.array_equal(solve(np.rot90(pair["input"], turns)),
                                  np.rot90(pair["output"], turns))
        assert np.array_equal(solve(relabel[pair["input"]]), relabel[pair["output"]])
        assert np.array_equal(pair["input"], unchanged)
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in one[s]] for s in one}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Corrupt output was accepted")
    return {"seeds": 42, "pairs": len(rows), "distinct_puzzles": len(puzzles),
            "board_shapes": len({r["board_shape"] for r in rows}),
            "input_shapes": len({r["input_shape"] for r in rows}),
            "interlocked_pairs": sum(r["interlocked"] for r in rows),
            "repeated_kit_pairs": sum(r["repeated"] for r in rows),
            "multi_part_color_pairs": sum(r["multi_part_color"] for r in rows)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
