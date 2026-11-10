"""Exchanged microtiles reveal checkpoint colors; paths organize their stamps.

No parent module, seed, example labels, or external files are used by solve.
Search is exact over induced orthogonal paths, including blank bends. Saved
artifacts are optional and bounded; generation and self_test do not write files.
"""
from __future__ import annotations

from functools import lru_cache
from heapq import heappop, heappush
import hashlib
import json

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 528155375, 416804983, 844775598, 303037069, 368412557,
              96619670, 486226368, 152934918, 156020899, 663267770,
              250153070, 266178268, 554000227, 287944414, 144120065,
              233770692, 380702636, 709245832, 239520659, 21991913,
              482103340, 914715659, 960351219, 789344391, 153852900,
              116527333, 231006407, 473341241, 258449564, 928659617,
              274729479, 678671009, 998111833, 314980509, 935670045,
              953645548, 849903291, 504293098, 954528817, 317580843,
              972988661)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _neighbors(shape):
    h, w = shape
    return [tuple((r + dr) * w + c + dc for dr, dc in _DIRS
                  if 0 <= r + dr < h and 0 <= c + dc < w)
            for r in range(h) for c in range(w)]


def _components(mask):
    near = _neighbors(mask.shape)
    todo = set(map(int, np.flatnonzero(mask)))
    result = []
    while todo:
        stack = [min(todo)]
        todo.remove(stack[0])
        part = []
        while stack:
            p = stack.pop()
            part.append(p)
            for q in near[p]:
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
        result.append(tuple(sorted(part)))
    return result


def _ordered(cells, near):
    cells = set(cells)
    if len(cells) == 1:
        return tuple(cells)
    degrees = {p: sum(q in cells for q in near[p]) for p in cells}
    ends = sorted(p for p in cells if degrees[p] == 1)
    if len(ends) != 2 or any(d not in (1, 2) for d in degrees.values()):
        raise ValueError("checkpoint fragment or completed path branches or cycles")
    path = [ends[0]]
    previous = None
    while path[-1] != ends[-1]:
        choices = [q for q in near[path[-1]] if q in cells and q != previous]
        if len(choices) != 1:
            raise ValueError("fragment is not a simple path")
        previous = path[-1]
        path.append(choices[0])
    if len(path) != len(cells):
        raise ValueError("fragment is disconnected")
    return tuple(path)


def _bends(path, width):
    directions = [(b // width - a // width, b % width - a % width)
                  for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(directions, directions[1:]))


def _route(board, color, background, mode="normal"):
    """Return (ordered cells, (added cells, bends), unique optimum).

    A* expands whole visible fragments and individual free cells. Every prefix
    is an induced path: a new cell must touch exactly its predecessor among
    used cells. All start fragments and both orientations are searched. An
    optimum never has a free endpoint, since trimming it improves length.

    The length lower bound is the nearest remaining fragment plus an MST on
    remaining fragments. Manhattan endpoint gaps ignore barriers and bends,
    so they cannot overestimate a valid completion's added-cell cost. There
    is no search cutoff or hidden length/bend hint. Modes other than normal
    are diagnostic competing rules, never used to define generated answers.
    """
    h, w = board.shape
    near = _neighbors(board.shape)
    parts = [_ordered(c, near) for c in _components(board == color)]
    if not parts:
        raise ValueError("route color is absent")
    if len(parts) == 1:
        return parts[0], (0, _bends(parts[0], w)), True
    values = board.ravel()
    endpoints = {p: i for i, part in enumerate(parts) for p in (part[0], part[-1])}
    k = len(parts)
    all_parts = (1 << k) - 1
    near_bits = [sum(1 << q for q in qs) for qs in near]

    def gap(a, b):
        return max(0, abs(a // w - b // w) + abs(a % w - b % w) - 1)

    edges = [[min(gap(a, b) for a in (pa[0], pa[-1])
                  for b in (pb[0], pb[-1])) for pb in parts] for pa in parts]

    @lru_cache(maxsize=None)
    def mst(mask):
        remaining = [i for i in range(k) if mask & (1 << i)]
        if len(remaining) < 2:
            return 0
        reached = {remaining.pop(0)}
        cost = 0
        while remaining:
            d, j = min((edges[i][j], j) for i in reached for j in remaining)
            reached.add(j)
            remaining.remove(j)
            cost += d
        return cost

    @lru_cache(maxsize=None)
    def lower(head, remaining):
        if not remaining:
            return 0
        return mst(remaining) + min(gap(head, p) for i, part in enumerate(parts)
                                    if remaining & (1 << i) for p in (part[0], part[-1]))

    def score(added, bends):
        if mode == "length":
            return added, 0
        if mode == "bend":
            return bends, added
        return added, bends

    def extend(path, visited, cells, added, bends, direction):
        for point in cells:
            if visited & (1 << point):
                return None
            if path:
                if (near_bits[point] & visited).bit_count() != 1 or path[-1] not in near[point]:
                    return None
                new_direction = (point // w - path[-1] // w,
                                 point % w - path[-1] % w)
                bends += direction is not None and direction != new_direction
                direction = new_direction
            added += int(values[point]) == background
            path += (point,)
            visited |= 1 << point
        return path, visited, added, int(bends), direction

    queue = []
    serial = 0

    def push(result, mask):
        nonlocal serial
        if result is None:
            return
        path, visited, added, bends, direction = result
        estimate = score(added + lower(path[-1], all_parts ^ mask), bends)
        heappush(queue, (*estimate, serial, mask, path, visited, added, bends, direction))
        serial += 1

    for i, part in enumerate(parts):
        for oriented in sorted({part, part[::-1]}):
            push(extend((), 0, oriented, 0, 0, None), 1 << i)
    best_score = None
    best_path = None
    best_cost = None
    while queue:
        first, second, _, mask, path, visited, added, bends, direction = heappop(queue)
        if best_score is not None and (first, second) > best_score:
            break
        if mask == all_parts:
            canonical = min(path, path[::-1])
            if best_path is None:
                best_path, best_score, best_cost = canonical, score(added, bends), (added, bends)
            elif canonical != best_path:
                return best_path, best_cost, False
            continue
        for point in near[path[-1]]:
            if visited & (1 << point):
                continue
            if int(values[point]) == background:
                cells, new_mask = (point,), mask
            elif int(values[point]) == color and point in endpoints:
                owner = endpoints[point]
                if mask & (1 << owner):
                    continue
                part = parts[owner]
                cells = part if point == part[0] else part[::-1]
                new_mask = mask | (1 << owner)
            else:
                continue
            push(extend(path, visited, cells, added, bends, direction), new_mask)
    if best_path is None:
        raise ValueError("no nonbranching checkpoint path exists")
    return best_path, best_cost, True


def _decode(input_grid):
    a = _grid(input_grid)
    frame = int(a[0, 0])
    candidates = []
    for n in (2, 3, 4):
        step = n + 1
        if (a.shape[0] - 1) % step or (a.shape[1] - 1) % step:
            continue
        if not (np.all(a[::step] == frame) and np.all(a[:, ::step] == frame)):
            continue
        h, w = ((d - 1) // step for d in a.shape)
        if min(h, w) < 3:
            continue
        tops = a[1::step, 1::step]
        grounds = set(map(int, np.unique(tops))) - {frame}
        if len(grounds) != 1:
            continue
        background = grounds.pop()
        board = np.full((h, w), background, dtype=np.int8)
        tiles = {}
        valid = True
        for r in range(h):
            for c in range(w):
                block = a[r * step + 1:r * step + n + 1, c * step + 1:c * step + n + 1]
                colors = set(map(int, np.unique(block)))
                if colors == {frame}:
                    board[r, c] = frame
                elif colors == {background}:
                    pass
                elif (len(colors) == 2 and background in colors and frame not in colors
                      and int(block[0, 0]) == background):
                    color = (colors - {background}).pop()
                    if color in tiles and not np.array_equal(tiles[color], block):
                        valid = False
                        break
                    tiles[color] = block.copy()
                    board[r, c] = color  # upper-left role AFTER exchanging this tile
                else:
                    valid = False
                    break
            if not valid:
                break
        if valid and tiles and np.any(board == background):
            candidates.append((a, n, background, frame, board, tiles))
    if len(candidates) != 1:
        raise ValueError("input must expose one unambiguous complete-tile lattice")
    return candidates[0]


def _all_routes(board, background, frame):
    result = {}
    used = set()
    for color in sorted(set(map(int, np.unique(board))) - {background, frame}):
        path, cost, unique = _route(board, color, background)
        if not unique:
            raise ValueError("equal shortest minimum-bend paths: invalid family member")
        if used.intersection(path):
            raise ValueError("individual optimal paths overlap: invalid family member")
        used.update(path)
        result[color] = (path, cost)
    return result


def _stamp(a, n, background, tiles, routes, exchange=True):
    out = a.copy()
    width = (a.shape[1] - 1) // (n + 1)
    for color, (path, _) in routes.items():
        tile = tiles[color]
        stamp = np.where(tile == background, color, background) if exchange else tile
        for p in path:
            r, c = divmod(p, width)
            out[r * (n + 1) + 1:r * (n + 1) + n + 1,
                c * (n + 1) + 1:c * (n + 1) + n + 1] = stamp
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, n, bg, frame, board, tiles = _decode(input_grid)
    routes = _all_routes(board, bg, frame)
    return _stamp(a, n, bg, tiles, routes)


def _tile(rng, n, bg, color):
    for _ in range(40):
        bits = rng.random((n, n)) < rng.uniform(.2, .8)
        bits[0, 0] = False
        if bits.any() and all(not np.array_equal(bits, np.rot90(bits, k)) for k in (1, 2, 3)):
            return np.where(bits, color, bg).astype(np.int8)
    bits = np.zeros((n, n), dtype=bool)
    bits[0, -1] = True
    return np.where(bits, color, bg).astype(np.int8)


def _encode(board, n, bg, frame, tiles):
    h, w = board.shape
    a = np.full((h * (n + 1) + 1, w * (n + 1) + 1), frame, dtype=np.int8)
    for r in range(h):
        for c in range(w):
            color = int(board[r, c])
            block = tiles[color] if color in tiles else np.full((n, n), color, dtype=np.int8)
            a[r * (n + 1) + 1:r * (n + 1) + n + 1,
              c * (n + 1) + 1:c * (n + 1) + n + 1] = block
    return a


def _transform(rng, board):
    # Transform macro geometry only; exemplar microtiles stay upright.
    board = np.rot90(board, int(rng.integers(4)))
    if rng.integers(2):
        board = np.fliplr(board)
    return board.copy()


def _witness_board(rng, kind, bg, frame, colors):
    x, y = colors[:2]
    if kind == "bend":
        h, w = int(rng.integers(5, 8)), int(rng.integers(5, 8))
        board = np.full((h, w), bg, dtype=np.int8)
        far = int(rng.integers(3, w))
        end = int(rng.integers(3, h - 1)) if h > 5 else 3
        board[0, :2] = x
        board[end - 1:end + 1, far] = x
        board[h - 1, 0] = y
        board[h - 1, int(rng.integers(2, w))] = y
    elif kind == "length":
        # A short six-bend weave beats the longer four-bend lower bypass.
        # The y singleton completes the second obstacle without being a wall.
        board = np.full((6, 7), bg, dtype=np.int8)
        board[2, :2] = x
        board[2, 5:] = x
        board[:3, 2] = frame
        board[2:4, 4] = frame
        board[4, 4] = y
    else:  # Guaranteed valid three-fragment scale-transfer fallback.
        board = np.full((5, 5), bg, dtype=np.int8)
        for r, c in ((0, 0), (0, 1), (1, 2), (2, 2), (2, 4), (3, 4), (4, 4)):
            board[r, c] = x
        board[4, 0] = y
        board[1, 4] = frame
    return _transform(rng, board)


def _walk(rng, shape, occupied, length):
    near = _neighbors(shape)
    free = [p for p in range(shape[0] * shape[1]) if p not in occupied]
    if not free:
        return None
    for _ in range(35):
        path = [free[int(rng.integers(len(free)))]]
        used = {path[0]}
        previous_delta = None
        while len(path) < length:
            choices = [q for q in near[path[-1]] if q not in occupied and q not in used
                       and sum(p in used for p in near[q]) == 1]
            if not choices:
                break
            straight = [q for q in choices if q - path[-1] == previous_delta]
            q = straight[0] if straight and rng.random() < .6 else choices[int(rng.integers(len(choices)))]
            previous_delta = q - path[-1]
            path.append(q)
            used.add(q)
        if len(path) == length:
            return tuple(path)
    return None


def _random_board(rng, n, bg, frame, colors, test):
    maximum = 29 // (n + 1)
    low = 4 if n == 4 else 5
    h, w = map(int, rng.integers(low, maximum + 1, size=2))
    board = np.full((h, w), bg, dtype=np.int8)
    occupied = set()
    count = int(rng.integers(1, (2 if n == 4 else 3) + 1))
    for i, color in enumerate(colors[:count]):
        minimum_length = 9 if test and i == 0 else 5
        maximum_exclusive = max(minimum_length + 1,
                                min(h * w // 2 + 1, 17 if test else 14))
        length = int(rng.integers(minimum_length, maximum_exclusive))
        path = _walk(rng, board.shape, occupied, length)
        if path is None:
            return None
        occupied.update(path)
        # Separated one-cell gaps produce 2..5 whole visible fragments. Blank
        # corners are allowed; uniqueness, not corner marking, is the criterion.
        gaps = []
        options = list(range(1, length - 1))
        rng.shuffle(options)
        wanted = int(rng.integers(2, min(4, (length - 1) // 2) + 1)) if test and i == 0 else int(rng.integers(1, 3))
        for j in options:
            if all(abs(j - old) > 1 for old in gaps):
                gaps.append(j)
                if len(gaps) == wanted:
                    break
        if len(gaps) != wanted:
            return None
        for j, p in enumerate(path):
            if j not in gaps:
                board.ravel()[p] = color
    # Barrier components can touch edges, be single cells, or form short bars.
    for _ in range(int(rng.integers(0, 5))):
        r, c = int(rng.integers(h)), int(rng.integers(w))
        dr, dc = _DIRS[int(rng.integers(4))]
        for j in range(int(rng.integers(1, 4))):
            rr, cc = r + dr * j, c + dc * j
            if 0 <= rr < h and 0 <= cc < w and rr * w + cc not in occupied:
                board[rr, cc] = frame
    if rng.random() < .5:
        free = [p for p in range(h * w) if p not in occupied and board.ravel()[p] == bg]
        if free:
            board.ravel()[free[int(rng.integers(len(free)))]] = colors[count]
    return board


def _make_pair(rng, n, bg, frame, colors, kind, test=False):
    for attempt in range(81):
        if kind in ("bend", "length"):
            board = _witness_board(rng, kind, bg, frame, colors)
        elif attempt == 80:
            board = _witness_board(rng, "fallback", bg, frame, colors)
        else:
            board = _random_board(rng, n, bg, frame, colors, test)
        if board is None:
            continue
        try:
            routes = _all_routes(board, bg, frame)
            if not any(cost[0] > 0 for _, cost in routes.values()):
                continue
            if kind == "bend":
                _, _, unique = _route(board, colors[0], bg, "length")
                if unique:
                    continue
            if kind == "length":
                _, alternative, _ = _route(board, colors[0], bg, "bend")
                normal = routes[colors[0]][1]
                if not (alternative[0] > normal[0] and alternative[1] < normal[1]):
                    continue
        except ValueError:
            continue
        tiles = {color: _tile(rng, n, bg, color) for color in routes}
        inp = _encode(board, n, bg, frame, tiles)
        out = _stamp(inp, n, bg, tiles, routes)
        return {"input": inp, "output": out}
    raise RuntimeError("bounded pair construction exhausted")


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _count("seed", seed, 0, 2**128 - 1)
    nt = _count("num_train", num_train, 2, 10)
    nq = _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    bg, frame = palette[:2]
    colors = palette[2:]
    first_n = int(rng.choice([2, 3]))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            kind = ("bend" if i == 0 else "length") if split == "train" and i < 2 else "random"
            n = (first_n if i == 0 else 5 - first_n) if split == "train" and i < 2 else int(rng.choice([2, 3]))
            if split == "test" and i == 0:
                n = 4  # Transfer a demonstrated lattice rule to a new tile scale.
            for _ in range(12):
                pair = _make_pair(rng, n, bg, frame, colors, kind, split == "test")
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_facts(pair):
    a, n, bg, frame, board, tiles = _decode(pair["input"])
    out = _grid(pair["output"])
    assert a.shape == out.shape
    routes = _all_routes(board, bg, frame)
    expected = _stamp(a, n, bg, tiles, routes)
    assert np.array_equal(expected, out), "output disagrees with decoded checkpoint paths"
    assert np.array_equal(out[::n + 1], a[::n + 1])
    assert np.array_equal(out[:, ::n + 1], a[:, ::n + 1])
    assert set(map(int, np.unique(out))) <= set(map(int, np.unique(a)))
    original = {c: (tuple(map(int, np.flatnonzero(board == c))), (0, 0)) for c in tiles}
    assert not np.array_equal(out, _stamp(a, n, bg, tiles, original)), "path completion is unnecessary"
    assert not np.array_equal(out, _stamp(a, n, bg, tiles, routes, False)), "tile exchange is unnecessary"
    active = 0
    fragments = []
    hidden = 0
    near = _neighbors(board.shape)
    for color, (path, cost) in routes.items():
        assert set(np.flatnonzero(board == color)) <= set(path)
        assert all(int(board.ravel()[p]) in (bg, color) for p in path)
        assert set(_ordered(path, near)) == set(path)
        assert cost == (sum(int(board.ravel()[p]) == bg for p in path), _bends(path, board.shape[1]))
        parts = len(_components(board == color))
        fragments.append(parts)
        active += parts > 1
        if parts == 1:
            assert cost[0] == 0, "single-component color extended"
        for j in range(1, len(path) - 1):
            if _bends(path[j - 1:j + 2], board.shape[1]) and board.ravel()[path[j]] == bg:
                hidden += 1
    return {"n": n, "macro_shape": tuple(board.shape), "groups": len(tiles), "active": active,
            "fragments": fragments, "added": sum(c[0] for _, c in routes.values()),
            "bends": [c[1] for _, c in routes.values()], "hidden_bends": hidden,
            "barriers": int(np.count_nonzero(board == frame)), "background": bg,
            "frame": frame, "patterns": [t.tobytes() for t in tiles.values()]}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen = set()
    roles = set()
    sizes = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a = _grid(pair["input"])
            key = a.shape, a.tobytes()
            assert key not in seen, "duplicate input"
            seen.add(key)
            f = _pair_facts(pair)
            roles.add((f["background"], f["frame"]))
            if split == "train":
                sizes.add(f["n"])
    assert len(roles) == 1, "carrier and lattice conventions must stay consistent"
    assert sizes == {2, 3}, "demonstrations must establish scale independence"
    for index, mode in ((0, "length"), (1, "bend")):
        _, _, bg, frame, board, tiles = _decode(puzzle["train"][index]["input"])
        normal = _all_routes(board, bg, frame)
        witnesses = 0
        for color in tiles:
            _, cost, unique = _route(board, color, bg, mode)
            if mode == "length":
                witnesses += not unique
            else:
                current = normal[color][1]
                witnesses += cost[0] > current[0] and cost[1] < current[1]
        assert witnesses, f"training does not distinguish {mode} optimization"
    first = _pair_facts(puzzle["train"][0])
    second = _pair_facts(puzzle["train"][1])
    assert first["active"] >= 2 and second["groups"] >= 2
    assert 1 in second["fragments"], "teach singleton no-extension"
    # Pixel-derived causal witnesses: removing an inert foreign group or the
    # solid barriers changes an active route, despite unchanged x checkpoints.
    _, _, bg, frame, board, tiles = _decode(puzzle["train"][1]["input"])
    normal = _all_routes(board, bg, frame)
    foreign_matters = barriers_matter = False
    for color, (path, _) in normal.items():
        if len(_components(board == color)) < 2:
            continue
        no_foreign = board.copy()
        no_foreign[(board != bg) & (board != frame) & (board != color)] = bg
        no_barriers = board.copy()
        no_barriers[board == frame] = bg
        foreign_matters |= set(_route(no_foreign, color, bg)[0]) != set(path)
        barriers_matter |= set(_route(no_barriers, color, bg)[0]) != set(path)
    assert foreign_matters and barriers_matter
    assert _pair_facts(puzzle["test"][0])["n"] == 4
    assert max(_pair_facts(puzzle["test"][0])["fragments"]) >= 3


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Stream 42 fixed seeds; retain only small sets and counters in RAM."""
    signatures, shapes, patterns, palettes = set(), set(), set(), set()
    active_counts, group_counts, bends, fragments, tile_sizes = set(), set(), set(), set(), set()
    hidden = barrier_pairs = zero_barrier_pairs = singleton_pairs = pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signature = _digest(puzzle)
        assert signature not in signatures
        signatures.add(signature)
        if seed in HOST_SEEDS[:2]:
            assert signature == _digest(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _pair_facts(pair)
                shapes.add(pair["input"].shape)
                patterns.update(f["patterns"])
                palettes.add((f["background"], f["frame"]))
                tile_sizes.add(f["n"])
                active_counts.add(f["active"])
                group_counts.add(f["groups"])
                bends.update(f["bends"])
                fragments.update(f["fragments"])
                hidden += f["hidden_bends"]
                barrier_pairs += f["barriers"] > 0
                zero_barrier_pairs += f["barriers"] == 0
                singleton_pairs += 1 in f["fragments"]
                pairs += 1
                copied = pair["input"].copy()
                before = copied.copy()
                assert np.array_equal(solve(copied), pair["output"])
                assert np.array_equal(copied, before), "solve mutated its input"
    assert tile_sizes == {2, 3, 4}
    assert {1, 2}.issubset(active_counts) and {1, 2, 3}.issubset(group_counts)
    assert {0, 1, 4, 6}.issubset(bends)
    assert {1, 2, 3, 4}.issubset(fragments)
    assert hidden > 42 and barrier_pairs > 42 and zero_barrier_pairs > 0
    assert singleton_pairs > 42 and len(shapes) >= 15 and len(patterns) > 100 and len(palettes) > 20
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(233770692, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    damaged = generate_puzzle(0, 2, 1)
    damaged["test"][0]["output"][1, 1] = (int(damaged["test"][0]["output"][1, 1]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    # A diagonal singleton pair has two equally good L paths: reject it.
    ambiguous = np.zeros((3, 3), dtype=np.int8)
    ambiguous[0, 0] = ambiguous[2, 2] = 2
    assert not _route(ambiguous, 2, 0)[2]
    crossing = np.zeros((5, 5), dtype=np.int8)
    crossing[2, 0] = crossing[2, 4] = 2
    crossing[0, 2] = crossing[4, 2] = 3
    try:
        _all_routes(crossing, 0, 9)
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping independent optima accepted")
    connected = np.zeros((4, 4), dtype=np.int8)
    connected[1, 1:3] = connected[2, 2] = 2
    path, cost, unique = _route(connected, 2, 0)
    assert unique and cost == (0, 1) and set(path) == {5, 6, 10}
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "tile_patterns": len(patterns),
            "palette_role_pairs": len(palettes), "tile_sizes": sorted(tile_sizes),
            "active_counts": sorted(active_counts), "group_counts": sorted(group_counts),
            "bend_counts": sorted(bends), "fragment_counts": sorted(fragments),
            "hidden_bends": hidden, "barrier_pairs": barrier_pairs,
            "zero_barrier_pairs": zero_barrier_pairs, "singleton_pairs": singleton_pairs,
            "independent_semantic_verification": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
