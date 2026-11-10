"""Interlocked contour keys: jointly decode a jigsaw and erosion-phase clues.

Only standard-library modules and NumPy are used. Generation and tests do not
write files, retain corpora, import parent code, or consult external state.
"""
from collections import deque
from itertools import permutations
import hashlib
import json

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=True)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        queue = [(sr, sc)]
        seen[sr, sc] = True
        for r, c in queue:
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if (0 <= rr < h and 0 <= cc < w and mask[rr, cc]
                        and not seen[rr, cc]):
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(np.asarray(queue, dtype=int))
    return result


def _mask(shape, cells):
    result = np.zeros(shape, dtype=bool)
    result[cells[:, 0], cells[:, 1]] = True
    return result


def _depth(mask, diagonal=True):
    """Iterative erosion: a boundary cell has depth zero, including at edges."""
    h, w = mask.shape
    current = mask.copy()
    result = np.full(mask.shape, -1, dtype=np.int16)
    level = 0
    directions = DIR8 if diagonal else DIR4
    while current.any():
        padded = np.pad(current, 1)
        interior = current.copy()
        for dr, dc in directions:
            interior &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        result[current & ~interior] = level
        current = interior
        level += 1
    return result


def _bfs_depth(mask):
    """A separate depth algorithm used for construction and validation."""
    h, w = mask.shape
    result = np.full(mask.shape, -1, dtype=np.int16)
    queue = deque()
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if any(not (0 <= r + dr < h and 0 <= c + dc < w
                        and mask[r + dr, c + dc]) for dr, dc in DIR8):
            result[r, c] = 0
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and result[rr, cc] < 0:
                result[rr, cc] = result[r, c] + 1
                queue.append((rr, cc))
    return result


def _dilate(mask):
    h, w = mask.shape
    padded = np.pad(mask, 1)
    return np.logical_or.reduce([padded[r:r + h, c:c + w]
                                 for r in range(3) for c in range(3)])


def _bbox(cells):
    low, high = cells.min(axis=0), cells.max(axis=0)
    return tuple(map(int, (low[0], high[0], low[1], high[1])))


def _scene(grid):
    """Find the enclosing wall, then rotate its sole exterior socket to the left.

    The socket's full cross span is explicitly delimited by its protruding wall
    cells, all of whose columns have positive length. No inferred maximal notch
    or unobserved attachment-site preference is used.
    """
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if np.count_nonzero(counts == counts[bg]) != 1:
        raise ValueError("background must be the unique most frequent input color")
    scenes = []
    for wall in range(10):
        if wall == bg or counts[wall] < 8:
            continue
        wall_mask = grid == wall
        wall_components = _components(wall_mask)
        if len(wall_components) != 1:
            continue
        wr0, wr1, wc0, wc1 = _bbox(wall_components[0])
        local_wall = np.pad(wall_mask[wr0:wr1 + 1, wc0:wc1 + 1], 1)
        enclosed = []
        # Search the candidate's bounding box, with an explicit exterior collar.
        # No generated chamber-size threshold is needed by the solving rule.
        for cells in _components(~local_wall):
            if np.all(cells > 0) and np.all(cells < np.asarray(local_wall.shape) - 1):
                enclosed.append(cells + np.asarray((wr0 - 1, wc0 - 1)))
        if len(enclosed) != 1:
            continue
        room = _mask(grid.shape, enclosed[0])
        for turns in range(4):
            a, m = np.rot90(grid, turns), np.rot90(room, turns)
            r0, r1, c0, c1 = _bbox(np.argwhere(m))
            r0, r1, c0, c1 = r0 - 1, r1 + 1, c0 - 1, c1 + 1
            rectangle = np.zeros_like(m)
            rectangle[r0:r1 + 1, c0:c1 + 1] = True
            if not np.array_equal(rectangle & ~m, rectangle & (a == wall)):
                continue
            root = np.argwhere((a == wall) & ~rectangle)
            if not len(root) or not np.all(root[:, 1] < c0):
                continue
            low, high = int(root[:, 0].min()), int(root[:, 0].max())
            if not r0 < low <= high < r1:
                continue
            profile = []
            good = True
            for r in range(low, high + 1):
                cols = np.sort(root[root[:, 0] == r, 1])
                if not len(cols) or not np.array_equal(cols, np.arange(cols[0], c0)):
                    good = False
                    break
                profile.append(len(cols))
            if not good or len(set(profile)) == 1:
                continue
            pieces = []
            exterior = ~rectangle & (a != bg) & (a != wall)
            occupied = a == wall
            for color in range(10):
                for cells in _components(exterior & (a == color)):
                    if np.any(_dilate(occupied)[cells[:, 0], cells[:, 1]]):
                        raise ValueError("input pieces must be separated by background")
                    occupied = occupied | _mask(a.shape, cells)
                    pieces.append((cells, color))
            if not 3 <= len(pieces) <= 5:
                raise ValueError("expected three to five detached pieces")
            cues = np.argwhere(m & (a != bg))
            if not len(cues):
                raise ValueError("at least one chamber phase clue is required")
            scenes.append({"grid": a, "room": m, "background": bg, "wall": wall,
                           "frame": (r0, r1, c0, c1), "root_start": low,
                           "root": tuple(profile), "pieces": pieces,
                           "cues": cues, "turns": turns, "depth": _depth(m)})
    if len(scenes) != 1:
        raise ValueError("expected one framed chamber with one jagged exterior root")
    return scenes[0]


def _describe(cells, width):
    cross = cells[:, 0] - cells[:, 0].min()
    outward = -cells[:, 1]
    if int(cross.max()) + 1 != width:
        raise ValueError("all pieces must span the socket width")
    lo, hi = [], []
    for r in range(width):
        values = np.sort(outward[cross == r])
        if not len(values) or not np.array_equal(values, np.arange(values[0], values[-1] + 1)):
            raise ValueError("pieces must be solid intervals along the key direction")
        lo.append(int(values[0]))
        hi.append(int(values[-1]) + 1)
    origin = min(lo)
    return (tuple(x - origin for x in lo), tuple(x - origin for x in hi),
            np.column_stack((cross, outward - origin)))


def _orders(info):
    pieces = info["pieces"]
    width = len(info["root"])
    descriptions = [_describe(cells, width) for cells, _ in pieces]
    c0 = info["frame"][2]
    geometric = []

    def extend(profile, order, placements):
        if len(order) == len(pieces):
            if len(set(profile)) == 1:
                geometric.append((tuple(order), placements))
            return
        offset = min(profile)
        normalized = tuple(v - offset for v in profile)
        for i, (lo, hi, local) in enumerate(descriptions):
            if i in order or lo != normalized:
                continue
            destination = np.column_stack((local[:, 0] + info["root_start"],
                                           c0 - 1 - offset - local[:, 1]))
            if np.any(destination < 0) or np.any(destination >= np.asarray(info["grid"].shape)):
                continue
            if np.any(info["grid"][destination[:, 0], destination[:, 1]] == info["wall"]):
                continue
            if np.array_equal(cells_key(destination), cells_key(pieces[i][0])):
                continue
            extend(tuple(offset + v for v in hi), order + [i], placements + [destination])

    extend(info["root"], [], [])
    valid = []
    for order, placements in geometric:
        palette = [pieces[i][1] for i in order]
        if all(palette[int(info["depth"][r, c]) % len(palette)] == info["grid"][r, c]
               for r, c in info["cues"]):
            valid.append((order, placements))
    return geometric, valid


def cells_key(cells):
    return np.asarray(sorted(map(tuple, cells)), dtype=int)


def _render(info, order, placements):
    a = info["grid"]
    answer = np.full_like(a, info["background"])
    answer[a == info["wall"]] = info["wall"]
    for i, cells in zip(order, placements):
        answer[cells[:, 0], cells[:, 1]] = info["pieces"][i][1]
    palette = np.asarray([info["pieces"][i][1] for i in order], dtype=np.int8)
    answer[info["room"]] = palette[info["depth"][info["room"]] % len(order)]
    return answer


def _decode(grid):
    info = _scene(grid)
    geometric, valid = _orders(info)
    answers = {}
    for order, placements in valid:
        answer = _render(info, order, placements)
        answers.setdefault(answer.tobytes(), (answer, order, placements))
    if len(answers) != 1:
        raise ValueError("shape constraints and contour clues must select one colored answer")
    answer, order, placements = next(iter(answers.values()))
    info.update(geometric=geometric, valid=valid, order=order, placements=placements)
    return np.rot90(answer, -info["turns"]).copy(), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer solely from the queried input, including its clues."""
    return _decode(_grid(input_grid))[0]


def _cuts(rng, width, count):
    # A finite pool avoids an unbounded duplicate-profile rejection loop.
    pool = [np.asarray([(code >> j) & 1 for j in range(width)], dtype=int)
            for code in range(1, (1 << width) - 1)]
    states = [pool[int(i)] for i in rng.choice(len(pool), size=3, replace=False)]
    r, s, t = states
    sequence = {3: [r, r, r], 4: [r, s, r, s], 5: [t, r, s, r, s]}[count]
    cuts = [sequence[0] + int(rng.integers(1, 3))]
    # Repeated transitions use exactly the same thickness, giving a real
    # two-way geometric ambiguity between differently colored equal masks.
    bonuses = {}
    for previous, following in zip(sequence, sequence[1:]):
        key = (tuple(previous), tuple(following))
        if key not in bonuses:
            bonuses[key] = int(rng.integers(0, 2))
        shift = max(int(np.max(previous + 1 - following)),
                    int(np.max(np.maximum(previous[:-1], previous[1:]) + 1
                               - np.minimum(following[:-1], following[1:]))))
        cuts.append(following + int(cuts[-1].min()) + shift + bonuses[key])
    cuts.append(np.full(width, int(cuts[-1].max()) + int(rng.integers(1, 3))))
    return cuts


def _chamber(rng, h, w, required_depth, notch_count):
    room = np.ones((h, w), dtype=bool)
    side = 2 * required_depth + 1
    reserve = np.zeros_like(room)
    rr, cc = (h - side) // 2, (w - side) // 2
    reserve[rr:rr + side, cc:cc + side] = True
    added = 0
    sides = list(map(int, rng.permutation(4)))
    for wall_side in sides:
        for _ in range(35):
            candidate = room.copy()
            extent = w if wall_side < 2 else h
            length = int(rng.integers(2, min(6, extent - 3)))
            start = int(rng.integers(1, extent - length))
            depth = int(rng.integers(1, 4))
            if wall_side == 0:
                candidate[:depth, start:start + length] = False
            elif wall_side == 1:
                candidate[-depth:, start:start + length] = False
            elif wall_side == 2:
                candidate[start:start + length, :depth] = False
            else:
                candidate[start:start + length, -depth:] = False
            if np.all(candidate[reserve]) and len(_components(candidate)) == 1:
                room = candidate
                added += 1
                break
        if added >= notch_count:
            break
    if added < notch_count:
        raise ValueError("could not place the requested notches around the deep core")
    return room


def _scatter(rng, base, frame, pieces, targets, bg):
    # Workspaces and attempts are in memory. Neither failures nor retries write.
    h, w = base.shape
    exterior = np.ones((h, w), dtype=bool)
    r0, r1, c0, c1 = frame
    exterior[r0:r1 + 1, c0:c1 + 1] = False
    for _ in range(24):
        a = base.copy()
        forbidden = _dilate(a != bg) | ~exterior
        order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i][0]))
        success = True
        for i in order:
            cells, color = pieces[i]
            local = cells - cells.min(axis=0)
            ph, pw = local.max(axis=0) + 1
            choices = np.ones((h - ph + 1, w - pw + 1), dtype=bool)
            for r, c in local:
                choices &= ~forbidden[r:r + choices.shape[0], c:c + choices.shape[1]]
            # Equal-mask tiles exchange two destinations. Neither starting
            # location may accidentally invalidate one of the two assemblies
            # through the requirement that every component actually moves.
            for target in targets:
                normalized = target - target.min(axis=0)
                if np.array_equal(cells_key(normalized), cells_key(local)):
                    choices[tuple(target.min(axis=0))] = False
            positions = np.argwhere(choices)
            if not len(positions):
                success = False
                break
            destination = local + positions[int(rng.integers(len(positions)))]
            a[destination[:, 0], destination[:, 1]] = color
            forbidden |= _dilate(_mask(a.shape, destination))
        if success:
            return a
    raise ValueError("not enough exterior scatter clearance")


def _make_pair(rng, count, late_clue, notch_count):
    width = int(rng.integers(3, 7))
    cuts = _cuts(rng, width, count)
    key_length = int(cuts[-1][0])
    ambiguous_phase = 1 if count == 5 else 0
    clue_depth = ambiguous_phase + (count if late_clue else 0)
    required = max(count, clue_depth)
    minimum_width = max(11, 2 * required + 1)
    maximum_width = 30 - key_length - 2
    if maximum_width < minimum_width:
        raise ValueError("key and chamber exceed the canvas width")
    iw = int(rng.integers(minimum_width, maximum_width + 1))
    ih = int(rng.integers(max(13, 2 * required + 5), 27))
    local_room = _chamber(rng, ih, iw, required, notch_count)
    fh, fw = ih + 2, iw + 2
    h = int(rng.integers(fh, 31))
    w = int(rng.integers(key_length + fw, 31))
    r0 = int(rng.integers(h - fh + 1))
    c0 = key_length + int(rng.integers(w - key_length - fw + 1))
    frame = (r0, r0 + fh - 1, c0, c0 + fw - 1)
    root_start = int(rng.integers(r0 + 2, r0 + fh - width - 1))
    colors = list(map(int, rng.permutation(10)))
    bg, wall = colors[:2]
    palette = colors[2:2 + count]
    # A safe, occasional nonadjacent reuse never equates the ambiguous pair.
    if count == 5 and rng.random() < .3:
        palette[-1] = palette[0]
    room = np.zeros((h, w), dtype=bool)
    room[r0 + 1:r0 + ih + 1, c0 + 1:c0 + iw + 1] = local_room
    structural = np.zeros_like(room)
    structural[r0:r0 + fh, c0:c0 + fw] = True
    structural[room] = False
    for r, length in enumerate(cuts[0]):
        structural[root_start + r, c0 - int(length):c0] = True
    base = np.full((h, w), bg, dtype=np.int8)
    base[structural] = wall
    depth = _bfs_depth(room)
    clue_options = np.argwhere(room & (depth == clue_depth))
    r, c = clue_options[int(rng.integers(len(clue_options)))]
    base[r, c] = palette[ambiguous_phase]
    if count >= 4 and rng.random() < .4:
        phase = count - 1
        options = np.argwhere(room & (depth % count == phase))
        r, c = options[int(rng.integers(len(options)))]
        base[r, c] = palette[phase]
    expected = base.copy()
    expected[room] = np.asarray(palette, dtype=np.int8)[depth[room] % count]
    pieces, targets = [], []
    for i, (lower, upper) in enumerate(zip(cuts, cuts[1:])):
        cells = np.asarray([(root_start + r, c0 - 1 - u)
                            for r in range(width)
                            for u in range(int(lower[r]), int(upper[r]))], dtype=int)
        assert len(_components(_mask((h, w), cells))) == 1
        assert np.all(expected[cells[:, 0], cells[:, 1]] == bg)
        expected[cells[:, 0], cells[:, 1]] = palette[i]
        pieces.append((cells, palette[i]))
        targets.append(cells)
    inp = _scatter(rng, base, frame, pieces, targets, bg)
    turns = int(rng.integers(4))
    inp, expected = np.rot90(inp, turns), np.rot90(expected, turns)
    if rng.integers(2):
        inp, expected = inp[::-1], expected[::-1]
    pair = {"input": inp.copy(), "output": expected.copy()}
    _validate_pair(pair)
    return pair


def _check_count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _check_count(num_train, "num_train", 2, 10)
    num_test = _check_count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            if split == "train" and index < 2:
                count, late, notches = 3 + index, bool(index), 1 + index
            else:
                count = int(rng.integers(3, 6))
                late = split == "test" or bool(rng.integers(2))
                notches = int(rng.integers(2 if split == "test" else 1, 4))
            for _ in range(120):
                try:
                    pair = _make_pair(rng, count, late, notches)
                except ValueError:
                    continue
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded construction exhausted; no silent fallback")
    validate_generated_puzzle(puzzle)
    return puzzle


def _contour_alternatives(info, palette):
    room, depth = info["room"], info["depth"]
    rr, cc = np.nonzero(room)
    box = np.minimum.reduce((rr - rr.min(), rr.max() - rr, cc - cc.min(), cc.max() - cc))
    p = np.asarray(palette)
    actual = p[depth[room] % len(p)]
    return (bool(np.any(actual != p[box % len(p)])),
            bool(np.any(actual != p[_depth(room, False)[room] % len(p)])))


def _validate_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    predicted, info = _decode(inp)
    assert np.array_equal(predicted, out), "label disagrees with input-derived joint constraints"
    a = info["grid"]
    canonical_out = np.rot90(out, info["turns"])
    assert np.array_equal(a == info["wall"], canonical_out == info["wall"])
    assert np.array_equal(info["depth"], _bfs_depth(info["room"]))
    palette = tuple(info["pieces"][i][1] for i in info["order"])
    assert all(_contour_alternatives(info, palette)), "need true nonrectangular eight-neighbor contours"
    assert int(info["depth"].max()) >= len(palette), "cycle must actually repeat"
    # Neither concept can be discarded. Geometry alone has two colored answers;
    # clues alone (allowing every tile-color order) also leave multiple answers.
    geometric_keys = {tuple(info["pieces"][i][1] for i in order)
                      for order, _ in info["geometric"]}
    assert len(geometric_keys) == 2, "the chamber must resolve a genuine jigsaw ambiguity"
    relaxed = {p for p in permutations(palette)
               if all(p[int(info["depth"][r, c]) % len(p)] == a[r, c]
                      for r, c in info["cues"])}
    assert len(relaxed) > 1, "clues must leave missing phases for the shape key to recover"
    bg = info["background"]
    outside = ~info["room"] & (a != info["wall"])
    # On the exterior the operation is exactly translation, with no color loss.
    assert np.array_equal(np.bincount(a[outside], minlength=10),
                          np.bincount(canonical_out[outside], minlength=10))
    destinations = np.zeros_like(info["room"])
    for i, target in zip(info["order"], info["placements"]):
        source, color = info["pieces"][i]
        assert len(_components(_mask(a.shape, source))) == 1
        assert np.array_equal(cells_key(source - source.min(axis=0)),
                              cells_key(target - target.min(axis=0)))
        assert not np.array_equal(cells_key(source), cells_key(target))
        assert not destinations[target[:, 0], target[:, 1]].any()
        destinations[target[:, 0], target[:, 1]] = True
        assert np.all(canonical_out[target[:, 0], target[:, 1]] == color)
    untouched = ~info["room"] & ~destinations & (a != info["wall"])
    assert np.all(canonical_out[untouched] == bg)
    assert np.array_equal(canonical_out[info["cues"][:, 0], info["cues"][:, 1]],
                          a[info["cues"][:, 0], info["cues"][:, 1]])
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _validate_pair(pair)
            inp = _grid(pair["input"])
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            analyses[split].append(info)
    assert {len(a["pieces"]) for a in analyses["train"]} >= {3, 4}
    assert any(np.any(a["depth"][a["cues"][:, 0], a["cues"][:, 1]] >= len(a["pieces"]))
               for a in analyses["train"]), "training must demonstrate wrapped clue phase"
    assert all(np.any(a["depth"][a["cues"][:, 0], a["cues"][:, 1]] >= len(a["pieces"]))
               for a in analyses["test"])


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(pair[role])
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded, streaming tests: 42 default puzzles, counts, and corruptions."""
    shapes, roles, roots, chambers, widths, counts, signatures, sides = (set() for _ in range(8))
    structures = set()
    total = reuse = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for pair in puzzle["train"] + puzzle["test"]:
            a = _validate_pair(pair)
            inp = _grid(pair["input"])
            shapes.add(inp.shape)
            roles.add((a["background"], a["wall"]))
            roots.add(a["root"])
            widths.add(len(a["root"]))
            counts.add(len(a["pieces"]))
            sides.add(a["turns"])
            chambers.add(hashlib.sha256(a["room"].tobytes() + bytes(a["room"].shape)).hexdigest())
            palette_map = {}
            normalized = bytes(palette_map.setdefault(int(c), len(palette_map)) for c in inp.flat)
            structures.add(hashlib.sha256(bytes(inp.shape) + normalized).hexdigest())
            reuse += len({c for _, c in a["pieces"]}) < len(a["pieces"])
            total += 1
    assert len(signatures) == 42 and len(structures) >= 200
    assert len(shapes) >= 30 and len(roles) >= 40 and len(roots) >= 25
    assert len(chambers) >= 180 and widths == {3, 4, 5, 6}
    assert counts == {3, 4, 5} and sides == {0, 1, 2, 3} and reuse > 0
    assert _digest(generate_puzzle(11)) == _digest(generate_puzzle(11))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(9000 + nt * 10 + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, None), ("4", 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    p = generate_puzzle(17, 2, 1)
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken["test"][0]["output"] = broken["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("identity-label corruption accepted")
    # Remove the clue that separates the equal-mask pieces: geometry alone is
    # deliberately ambiguous, and solve must reject rather than select a tie.
    info = _scene(p["train"][0]["input"])
    geometric, _ = _orders(info)
    actual = solve(info["grid"])
    old_word = tuple(info["grid"][r, c] for r, c in info["cues"])
    # With shapes fixed, changing the registration selects the other assembly.
    # This is a direct counterfactual for A's contribution, not a seed lookup.
    for order, placements in geometric:
        word = tuple(info["pieces"][i][1] for i in order)
        changed = info["grid"].copy()
        new_clues = tuple(word[int(info["depth"][r, c]) % len(word)] for r, c in info["cues"])
        if new_clues != old_word:
            changed[info["cues"][:, 0], info["cues"][:, 1]] = new_clues
            other = solve(changed)
            assert not np.array_equal(other, actual)
            assert np.array_equal(other, _render(info, order, placements))
            break
    else:
        raise AssertionError("changing registration did not expose the alternative assembly")
    damaged = info["grid"].copy()
    damaged[info["cues"][:, 0], info["cues"][:, 1]] = info["background"]
    try:
        solve(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("unregistered ambiguous key accepted")
    return {"seeds": 42, "pairs": total, "shapes": len(shapes), "roles": len(roles),
            "root_profiles": len(roots), "chambers": len(chambers),
            "normalized_inputs": len(structures), "piece_counts": sorted(counts),
            "socket_widths": sorted(widths), "reused_color_pairs": reuse,
            "semantic_verification": "generator-owned checks only"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
