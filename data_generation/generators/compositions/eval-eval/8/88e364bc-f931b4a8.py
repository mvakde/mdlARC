"""Color-selective wind through a periodic screen.

Only NumPy and the standard library are used. Nothing is loaded from disk.
The two framed regions, glyph geometry and visible palette encode every role.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))
CARDINAL = tuple(d for d in DIRECTIONS if 0 in d)
REQUIRED_EVIDENCE = frozenset((
    "no_motion", "reverse", "one_step", "ignore_screen", "all_opaque",
    "single_tile", "stretch_tile", "phase_rows", "phase_columns",
    "corner_cut", "wrong_color",
))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(np.all((0 <= a) & (a <= 9)), "color bounds")
    return a


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    neighbors = DIRECTIONS if diagonal else CARDINAL
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in neighbors:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _bbox(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows) + 1, min(cols), max(cols) + 1


def _read(input_grid):
    """Parse only the input; no learned colors, example index or RNG state."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(np.all(a[0] == bg) and np.all(a[-1] == bg)
             and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg), "outer margin")
    interpretations = []
    for color in sorted(set(map(int, np.unique(a))) - {bg}):
        comps = _components(a == color)
        if len(comps) != 2:
            continue
        boxes = []
        for comp in comps:
            r0, r1, c0, c1 = _bbox(comp)
            h, w = r1 - r0, c1 - c0
            if min(h, w) < 4 or len(comp) != 2 * h + 2 * w - 4:
                break
            if not all(r in (r0, r1 - 1) or c in (c0, c1 - 1) for r, c in comp):
                break
            boxes.append((r0, r1, c0, c1))
        if len(boxes) == 2:
            interpretations.append((color, sorted(boxes, key=lambda b: (b[1]-b[0])*(b[3]-b[2]))))
    _require(len(interpretations) == 1, "unique common frame color required")
    frame, (small, large) = interpretations[0]

    def inside(box):
        r0, r1, c0, c1 = box
        return a[r0 + 1:r1 - 1, c0 + 1:c1 - 1]

    tile, board = inside(small), inside(large)
    _require(all(x < y for x, y in zip(tile.shape, board.shape)), "swatch must be smaller")
    _require(2 <= min(tile.shape) and max(tile.shape) <= 5, "tile sides must be 2..5")
    inks = tuple(sorted(set(map(int, np.unique(tile))) - {bg}))
    _require(2 <= len(inks) <= 3 and np.any(tile == bg), "perforated two/three-ink tile")
    _require(frame not in inks and np.all(tile != frame) and np.all(board != frame), "frame role reused")
    tips = set(map(int, np.unique(board))) - {bg} - set(inks)
    _require(len(tips) == 1, "head color must be absent from the swatch")
    tip = tips.pop()
    arrows = []
    for comp in _components(board != bg, diagonal=True):
        heads = [p for p in comp if board[p] == tip]
        _require(len(heads) == 1, "each separated stroke needs one head")
        head = heads[0]
        trails = set(comp) - {head}
        colors = {int(board[p]) for p in trails}
        _require(2 <= len(trails) <= 4 and len(colors) == 1, "two to four equal tail cells")
        ink = colors.pop()
        _require(ink in inks, "tail must select a swatch color")
        directions = [d for d in DIRECTIONS if trails == {
            (head[0] - j*d[0], head[1] - j*d[1]) for j in range(1, len(trails) + 1)}]
        _require(len(directions) == 1, "straight, uniquely oriented glyph required")
        arrows.append(dict(head=head, direction=directions[0], ink=ink,
                           length=len(trails), cells=tuple(sorted(comp))))
    _require(arrows and {x["ink"] for x in arrows} == set(inks), "every screen ink needs an arrow")
    outside = a.copy()
    for r0, r1, c0, c1 in (small, large):
        outside[r0:r1, c0:c1] = bg
    _require(np.all(outside == bg), "unexplained cells outside frames")
    h, w = board.shape
    terrain = tile[np.arange(h)[:, None] % tile.shape[0], np.arange(w)[None, :] % tile.shape[1]]
    for arrow in arrows:
        _require(terrain[arrow["head"]] != arrow["ink"], "head starts in its own wall")
    return dict(grid=a, background=bg, frame=frame, tip=tip, tile=tile,
                board=board, box=large, small_box=small, inks=inks, arrows=arrows, terrain=terrain)


def _trace(blocked, start, direction, corner=True):
    """Straight ray; the exterior acts as a wall of every color."""
    h, w = blocked.shape
    r, c = start
    dr, dc = direction
    path = [(r, c)]
    for _ in range(max(h, w)):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return path
        if blocked[nr, nc] or (corner and dr and dc and (blocked[nr, c] or blocked[r, nc])):
            return path
        r, c = nr, nc
        path.append((r, c))
    raise AssertionError("nonterminating ray")


def _paths(info, terrain=None, variant=None):
    terrain = info["terrain"] if terrain is None else terrain
    paths = []
    for arrow in info["arrows"]:
        ink, direction = arrow["ink"], arrow["direction"]
        if variant == "reverse":
            direction = tuple(-v for v in direction)
        if variant == "wrong_color":
            ink = info["inks"][(info["inks"].index(ink) + 1) % len(info["inks"])]
        blocked = terrain != info["background"] if variant == "all_opaque" else terrain == ink
        if variant == "ignore_screen":
            blocked = np.zeros(terrain.shape, bool)
        path = _trace(blocked, arrow["head"], direction, corner=variant != "corner_cut")
        if variant == "no_motion":
            path = path[:1]
        elif variant == "one_step":
            path = path[:2]
        paths.append(path)
    return paths


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read(input_grid)
    paths = _paths(info)
    ends = [p[-1] for p in paths]
    _require(len(set(ends)) == len(ends), "colliding final heads are outside this family")
    board = info["terrain"].copy()
    for end in ends:
        board[end] = info["tip"]
    out = info["grid"].copy()
    r0, r1, c0, c1 = info["box"]
    out[r0 + 1:r1 - 1, c0 + 1:c1 - 1] = board
    return out


def _evidence(info):
    correct = tuple(p[-1] for p in _paths(info))
    evidence = set()
    for name in REQUIRED_EVIDENCE:
        terrain = info["terrain"]
        tile, bg = info["tile"], info["background"]
        h, w = terrain.shape
        if name == "single_tile":
            terrain = np.full((h, w), bg, dtype=np.int8)
            terrain[:tile.shape[0], :tile.shape[1]] = tile
        elif name == "stretch_tile":
            terrain = tile[(np.arange(h)*tile.shape[0]//h)[:, None],
                           (np.arange(w)*tile.shape[1]//w)[None, :]]
        elif name == "phase_rows":
            terrain = tile[(np.arange(h)[:, None] + 1) % tile.shape[0],
                           np.arange(w)[None, :] % tile.shape[1]]
        elif name == "phase_columns":
            terrain = tile[np.arange(h)[:, None] % tile.shape[0],
                           (np.arange(w)[None, :] + 1) % tile.shape[1]]
        alternative = tuple(p[-1] for p in _paths(info, terrain, name))
        if alternative != correct:
            evidence.add(name)
    return evidence


def _tile(rng, inks, bg, need_corner=False):
    low = 3 if need_corner else 2
    h, w = map(int, rng.integers(low, 6, size=2))
    rr, cc = np.indices((h, w))
    style = int(rng.integers(5))
    density = float(rng.uniform(.18, .65))
    count = max(len(inks), min(h*w - 1, int(round(density*h*w))))
    mask = np.zeros((h, w), bool)
    if style == 0:
        mask.ravel()[rng.choice(h*w, count, replace=False)] = True
    elif style == 1:
        frontier = {(int(rng.integers(h)), int(rng.integers(w)))}
        cells = set()
        while len(cells) < count:
            choices = sorted(frontier)
            p = choices[int(rng.integers(len(choices)))]
            frontier.remove(p)
            cells.add(p)
            for dr, dc in CARDINAL:
                q = p[0] + dr, p[1] + dc
                if 0 <= q[0] < h and 0 <= q[1] < w and q not in cells:
                    frontier.add(q)
        for p in cells:
            mask[p] = True
    elif style == 2:
        axis = rr if rng.random() < .5 else cc
        period = int(rng.integers(2, max(h, w) + 1))
        mask = (axis + int(rng.integers(period))) % period == 0
    elif style == 3:
        mask = (rr + cc + int(rng.integers(2))) % 2 == 0
        mask &= rng.random((h, w)) < .85
    else:
        cy, cx = int(rng.integers(h)), int(rng.integers(w))
        mask = abs(rr - cy) + abs(cc - cx) <= int(rng.integers(1, max(h, w)))
        mask &= rng.random((h, w)) < .8
    # Repair random degeneracies without a fixed occupied corner or tip phase.
    while int(mask.sum()) < len(inks):
        mask.ravel()[int(rng.choice(np.flatnonzero(~mask)))] = True
    if mask.all():
        mask.ravel()[int(rng.integers(h*w))] = False
    tile = np.full((h, w), bg, dtype=np.int8)
    n = int(mask.sum())
    values = np.concatenate((np.array(inks, dtype=np.int8),
                             rng.choice(inks, n - len(inks))))
    rng.shuffle(values)
    tile[mask] = values
    return tile


def _candidates(terrain, inks, bg):
    h, w = terrain.shape
    candidates = []
    for ink in inks:
        blocked = terrain == ink
        for dr, dc in DIRECTIONS:
            for r in range(h):
                for c in range(w):
                    if blocked[r, c] or not (0 <= r-2*dr < h and 0 <= c-2*dc < w):
                        continue
                    path = _trace(blocked, (r, c), (dr, dc))
                    end = path[-1]
                    nr, nc = end[0] + dr, end[1] + dc
                    edge = not (0 <= nr < h and 0 <= nc < w)
                    features = set()
                    if len(path) > 1:
                        if edge:
                            features.add("edge")
                        else:
                            features.add("wall")
                        if not dr or not dc:
                            features.add("cardinal")
                        if terrain[end] not in (bg, ink):
                            features.add("other_landing")
                        if dr and dc and not edge and not blocked[nr, nc]:
                            features.add("corner")
                    candidates.append(dict(head=(r, c), direction=(dr, dc), ink=ink,
                                           path=path, features=features))
    return candidates


def _choose_arrows(rng, candidates, shape, inks, required):
    wanted = max(len(inks), int(rng.integers(2, 6)))
    selected, used_path, occupied, ends = [], set(), set(), []
    missing_features, missing_inks = set(required), set(inks)
    for slot in range(wanted):
        choices = []
        best = -1
        # The selected glyphs stay separate even under eight-neighbor grouping.
        for index in rng.permutation(len(candidates)):
            candidate = candidates[int(index)]
            if not used_path.isdisjoint(candidate["path"]):
                continue
            end = candidate["path"][-1]
            if any(max(abs(end[0]-r), abs(end[1]-c)) <= 1 for r, c in ends):
                continue
            r, c = candidate["head"]
            dr, dc = candidate["direction"]
            cells = {(r-j*dr, c-j*dc) for j in range(3)}
            if not occupied.isdisjoint(cells):
                continue
            gain = 10*len(candidate["features"] & missing_features) + 4*(candidate["ink"] in missing_inks)
            gain += int(len(candidate["path"]) > 1)
            if gain > best:
                best, choices = gain, [candidate]
            elif gain == best:
                choices.append(candidate)
        if not choices:
            break
        candidate = choices[int(rng.integers(len(choices)))].copy()
        r, c = candidate["head"]
        dr, dc = candidate["direction"]
        lengths = []
        for length in range(2, 5):
            cells = {(r-j*dr, c-j*dc) for j in range(length+1)}
            if (all(0 <= y < shape[0] and 0 <= x < shape[1] for y, x in cells)
                    and occupied.isdisjoint(cells)):
                lengths.append(length)
        length = int(rng.choice(lengths))
        cells = {(r-j*dr, c-j*dc) for j in range(length+1)}
        candidate.update(length=length, cells=tuple(sorted(cells)))
        selected.append(candidate)
        used_path.update(candidate["path"])
        ends.append(candidate["path"][-1])
        for y, x in cells:
            occupied.update((y+dy, x+dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        missing_features -= candidate["features"]
        missing_inks.discard(candidate["ink"])
    if missing_features or missing_inks or len(selected) < 2:
        return None
    return selected


def _pack(rng, board, tile, bg, frame):
    big = np.pad(board, 1, constant_values=frame)
    small = np.pad(tile, 1, constant_values=frame)
    horizontal = bool(rng.integers(2))
    if horizontal:
        max_gap = min(3, 28 - big.shape[1] - small.shape[1])
        gap = int(rng.integers(1, max_gap + 1))
        base_w, base_h = big.shape[1] + small.shape[1] + gap + 2, max(big.shape[0], small.shape[0]) + 2
    else:
        max_gap = min(3, 28 - big.shape[0] - small.shape[0])
        gap = int(rng.integers(1, max_gap + 1))
        base_h, base_w = big.shape[0] + small.shape[0] + gap + 2, max(big.shape[1], small.shape[1]) + 2
    h, w = int(rng.integers(base_h, min(30, base_h+4)+1)), int(rng.integers(base_w, min(30, base_w+4)+1))
    a = np.full((h, w), bg, dtype=np.int8)
    objects = [big, small] if rng.random() < .5 else [small, big]
    offset = int(rng.integers(1, (w-base_w+2) if horizontal else (h-base_h+2)))
    big_box = None
    for obj in objects:
        if horizontal:
            r, c = int(rng.integers(1, h-obj.shape[0])), offset
            offset += obj.shape[1] + gap
        else:
            r, c = offset, int(rng.integers(1, w-obj.shape[1]))
            offset += obj.shape[0] + gap
        a[r:r+obj.shape[0], c:c+obj.shape[1]] = obj
        if obj is big:
            big_box = (r, r+obj.shape[0], c, c+obj.shape[1])
    return a, big_box


def _sample_pair(rng, params, test=False, required=frozenset()):
    bg, frame, tip, palette = params
    # Finite attempts, one candidate pool at a time, no files or retained retries.
    for _ in range(40):
        k = int(rng.choice([2, 2, 3]))
        inks = tuple(map(int, rng.choice(palette, k, replace=False)))
        tile = _tile(rng, inks, bg, "corner" in required)
        h, w = map(int, rng.integers(9, 15, size=2))
        if test:
            if rng.random() < .5:
                h, w = int(rng.integers(15, 19)), int(rng.integers(9, 19))
            else:
                h, w = int(rng.integers(9, 19)), int(rng.integers(15, 19))
        terrain = np.tile(tile, ((h+tile.shape[0]-1)//tile.shape[0],
                                 (w+tile.shape[1]-1)//tile.shape[1]))[:h, :w]
        candidates = _candidates(terrain, inks, bg)
        arrows = _choose_arrows(rng, candidates, (h, w), inks, required)
        if arrows is None:
            continue
        board = np.full((h, w), bg, dtype=np.int8)
        output_board = terrain.copy()
        for arrow in arrows:
            for p in arrow["cells"]:
                board[p] = tip if p == arrow["head"] else arrow["ink"]
            output_board[arrow["path"][-1]] = tip
        a, box = _pack(rng, board, tile, bg, frame)
        b = a.copy()
        r0, r1, c0, c1 = box
        b[r0+1:r1-1, c0+1:c1-1] = output_board
        return {"input": a, "output": b}
    raise RuntimeError("bounded arrow placement exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.permutation(10)))
    params = (*palette[:3], palette[3:int(rng.integers(6, 11))])
    # Select two complementary, procedurally varied demonstrations, then shuffle.
    for _ in range(24):
        train = [_sample_pair(rng, params, required={"corner", "other_landing"}),
                 _sample_pair(rng, params, required={"edge", "cardinal"})]
        readings = [_read(p["input"]) for p in train]
        evidence = set().union(*(_evidence(x) for x in readings))
        if (REQUIRED_EVIDENCE <= evidence
                and readings[0]["tile"].shape != readings[1]["tile"].shape
                and readings[0]["board"].shape != readings[1]["board"].shape):
            break
    else:
        raise RuntimeError("bounded demonstration selection exhausted")
    puzzle, seen = {"train": train, "test": []}, {_key(p["input"]) for p in train}
    for split, count in (("train", nt), ("test", nq)):
        while len(puzzle[split]) < count:
            for _ in range(16):
                pair = _sample_pair(rng, params, test=split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness selection exhausted")
            seen.add(key)
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split types")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "pair counts")
    seen, conventions, evidence = set(), set(), set()
    train_shapes, test_shapes = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape and not np.array_equal(a, b), "shape or identity")
            key = _key(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _read(a)
            conventions.add((info["background"], info["frame"], info["tip"]))
            _require(np.array_equal(solve(a), b), "input-only solve disagrees")
            r0, r1, c0, c1 = info["box"]
            interior = b[r0+1:r1-1, c0+1:c1-1]
            fixed = np.ones(a.shape, bool)
            fixed[r0+1:r1-1, c0+1:c1-1] = False
            _require(np.array_equal(a[fixed], b[fixed]), "swatch, frames or margin changed")
            _require(np.count_nonzero(interior == info["tip"]) == len(info["arrows"]), "head conservation")
            _require(np.array_equal(interior[interior != info["tip"]],
                                    info["terrain"][interior != info["tip"]]), "screen damaged")
            paths = _paths(info)
            _require(any(len(p) > 1 for p in paths), "no moving head")
            used = set()
            for arrow, path in zip(info["arrows"], paths):
                _require(used.isdisjoint(path), "generated trajectories must be disjoint")
                used.update(path)
                blocked = info["terrain"] == arrow["ink"]
                _require(all(not blocked[p] for p in path), "head traversed its own wall")
                # Explicit maximality, including side cells for a diagonal step.
                r, c = path[-1]
                dr, dc = arrow["direction"]
                nr, nc = r+dr, c+dc
                if 0 <= nr < blocked.shape[0] and 0 <= nc < blocked.shape[1]:
                    _require(blocked[nr, nc] or (dr and dc and (blocked[nr, c] or blocked[r, nc])),
                             "head did not reach its first obstruction")
            if split == "train":
                evidence |= _evidence(info)
                train_shapes.append(info["board"].shape)
            else:
                test_shapes.append(info["board"].shape)
    _require(len(conventions) == 1, "puzzle color conventions changed")
    _require(REQUIRED_EVIDENCE <= evidence, "missing training evidence: " + str(REQUIRED_EVIDENCE-evidence))
    _require(all(any(t[axis] > max(s[axis] for s in train_shapes) for axis in (0, 1))
                 for t in test_shapes), "missing board-extent test shift")


def _normalized_digest(a):
    roles = {}
    values = bytes(roles.setdefault(int(x), len(roles)) for x in np.asarray(a).flat)
    return hashlib.sha256(bytes(a.shape) + values).hexdigest()


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.astype(np.uint8).tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; retain only small sets of feature hashes, never grids."""
    start = time.monotonic()
    signatures, normalized, roles = set(), set(), set()
    features = {s: {k: set() for k in ("boards", "tiles", "patterns", "directions", "arrows", "lengths", "sides")}
                for s in ("train", "test")}
    pairs = corner_pairs = other_landings = edge_stops = wall_stops = 0
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if first is None:
            first = puzzle
        signatures.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            fs = features[split]
            for pair in puzzle[split]:
                info = _read(pair["input"])
                pairs += 1
                normalized.add(_normalized_digest(pair["output"]))
                roles.add((info["background"], info["frame"], info["tip"]))
                fs["boards"].add(info["board"].shape)
                fs["tiles"].add(info["tile"].shape)
                fs["patterns"].add(_normalized_digest(info["tile"]))
                fs["arrows"].add(len(info["arrows"]))
                small, large = info["small_box"], info["box"]
                dy = small[0]+small[1]-large[0]-large[1]
                dx = small[2]+small[3]-large[2]-large[3]
                fs["sides"].add(("bottom" if dy > 0 else "top") if abs(dy) > abs(dx)
                                else ("right" if dx > 0 else "left"))
                corner_pairs += int("corner_cut" in _evidence(info))
                for arrow, path in zip(info["arrows"], _paths(info)):
                    fs["directions"].add(arrow["direction"])
                    fs["lengths"].add(arrow["length"])
                    end = path[-1]
                    other_landings += int(info["terrain"][end] not in (info["background"], arrow["ink"]))
                    nr, nc = end[0]+arrow["direction"][0], end[1]+arrow["direction"][1]
                    edge = not (0 <= nr < info["board"].shape[0] and 0 <= nc < info["board"].shape[1])
                    edge_stops += int(edge)
                    wall_stops += int(not edge)
    _require(len(signatures) == 42 and len(normalized) >= 200, "cross-seed structural variation")
    _require(len(roles) >= 35 and corner_pairs >= 42 and other_landings >= 42, "role/mechanism variation")
    for split, fs in features.items():
        _require(fs["directions"] == set(DIRECTIONS), f"all directions in {split}")
        _require(len(fs["boards"]) >= 25 and len(fs["tiles"]) >= 10, f"geometry variation in {split}")
        _require(len(fs["patterns"]) >= 35 and len(fs["arrows"]) >= 3, f"screen/count variation in {split}")
        _require(fs["lengths"] == {2, 3, 4} and len(fs["sides"]) == 4, f"glyph/layout variation in {split}")
    for nt, nq in ((2, 1), (10, 1), (10, 4), (6, 3)):
        p = generate_puzzle(1000 + nt*10 + nq, nt, nq)
        _require(_fingerprint(p) == _fingerprint(generate_puzzle(1000 + nt*10 + nq, nt, nq)), "determinism")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Actual corruption, rather than a second copy of an implementation check.
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in first[s]] for s in first}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "normalized_outputs": len(normalized), "role_assignments": len(roles),
            "corner_pairs": corner_pairs, "other_color_landings": other_landings,
            "edge_stops": edge_stops, "wall_stops": wall_stops,
            "diversity": {s: {k: len(v) for k, v in f.items()} for s, f in features.items()},
            "seconds": round(time.monotonic()-start, 3),
            "semantic_verification": "generator-owned invariants and counterfactuals only"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
