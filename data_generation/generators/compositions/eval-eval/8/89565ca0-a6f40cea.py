"""Room census lens: a global election by rooms, revealed through an occluder.

Only Python's standard library and NumPy are used. The public solver reads pixels;
the renderer computes labels from explicit room rectangles, not from that solver.
"""
from collections import Counter, deque
import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _check_counts(num_train, num_test):
    nt, ne = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne):
        raise ValueError("Require 2..10 train, 1..4 test, and num_train > num_test")
    return nt, ne


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _perimeter(shape, box):
    t, l, b, r = box
    mask = np.zeros(shape, dtype=bool)
    mask[t, l:r + 1] = mask[b, l:r + 1] = True
    mask[t:b + 1, l] = mask[t:b + 1, r] = True
    return mask


def _closed(shape, box):
    t, l, b, r = box
    mask = np.zeros(shape, dtype=bool)
    mask[t:b + 1, l:r + 1] = True
    return mask


def _wall_union(shape, boxes):
    mask = np.zeros(shape, dtype=bool)
    for box in boxes:
        mask |= _perimeter(shape, box)
    return mask


def _bridge(visible, lens):
    """Only original two-sided arms support a bridge, never a newly drawn cell."""
    t, l, b, r = lens
    mask = visible.copy()
    mask[t:b + 1, l:r + 1] = False
    horizontal = np.zeros_like(mask)
    vertical = np.zeros_like(mask)
    h, w = visible.shape
    if l >= 2 and r + 2 < w:
        for y in range(t, b + 1):
            if visible[y, l - 2:l].all() and visible[y, r + 1:r + 3].all():
                horizontal[y, l:r + 1] = True
    if t >= 2 and b + 2 < h:
        for x in range(l, r + 1):
            if visible[t - 2:t, x].all() and visible[b + 1:b + 3, x].all():
                vertical[t:b + 1, x] = True
    mask |= horizontal | vertical
    return mask, horizontal, vertical


def _regions(walls):
    """Four-connected bounded regions; zero labels the exterior, -1 walls."""
    h, w = walls.shape
    labels = np.full((h, w), -2, dtype=int)
    labels[walls] = -1
    rooms = []
    for y, x in np.argwhere(~walls):
        y, x = int(y), int(x)
        if labels[y, x] != -2:
            continue
        queue = deque([(y, x)])
        labels[y, x] = -3
        cells, outside = [], False
        while queue:
            yy, xx = queue.popleft()
            cells.append((yy, xx))
            outside |= yy in (0, h - 1) or xx in (0, w - 1)
            for a, c in ((yy - 1, xx), (yy + 1, xx), (yy, xx - 1), (yy, xx + 1)):
                if 0 <= a < h and 0 <= c < w and labels[a, c] == -2:
                    labels[a, c] = -3
                    queue.append((a, c))
        index = 0 if outside else len(rooms) + 1
        ys, xs = zip(*cells)
        labels[np.asarray(ys), np.asarray(xs)] = index
        if not outside:
            box = (min(ys), min(xs), max(ys), max(xs))
            rooms.append({"id": index, "box": box, "area": len(cells),
                          "rectangular": len(cells) == (box[2] - box[0] + 1) * (box[3] - box[1] + 1)})
    return labels, rooms


def _census(grid, labels, teams):
    scores = {}
    for color in teams:
        ids = set(map(int, labels[grid == color]))
        if any(i <= 0 for i in ids):
            raise ValueError("A marker lies outside a room or on a wall")
        scores[int(color)] = len(ids)
    return scores


def _maximum(scores):
    winners = [c for c, n in scores.items() if n == max(scores.values())]
    if len(winners) != 1:
        raise ValueError("The room election has no unique winner")
    return winners[0]


def _partial(walls, outer):
    t, l, b, r = outer
    hs = [y for y in range(t + 1, b) if np.any(walls[y, l:r] & walls[y, l + 1:r + 1])]
    vs = [x for x in range(l + 1, r) if np.any(walls[t:b, x] & walls[t + 1:b + 1, x])]
    return any(not walls[y, l:r + 1].all() for y in hs) or any(not walls[t:b + 1, x].all() for x in vs)


def _read(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    colors = list(map(int, np.unique(grid)))
    lenses = []
    for color in colors:
        if color == bg:
            continue
        points = np.argwhere(grid == color)
        t, l = map(int, points.min(axis=0))
        b, r = map(int, points.max(axis=0))
        if b - t < 2 or r - l < 2 or not np.array_equal(grid == color, _perimeter(grid.shape, (t, l, b, r))):
            continue
        interior = grid[t + 1:b, l + 1:r]
        fill = int(interior[0, 0])
        if fill not in (bg, color) and np.all(interior == fill):
            lenses.append(((t, l, b, r), color, fill))
    if len(lenses) != 1:
        raise ValueError("Expected one exclusive framed opaque lens")
    lens, frame, fill = lenses[0]
    hidden = _closed(grid.shape, lens)
    networks = []
    for color in colors:
        if color in (bg, frame, fill):
            continue
        visible = grid == color
        points = np.argwhere(visible)
        t, l = map(int, points.min(axis=0))
        b, r = map(int, points.max(axis=0))
        lt, ll, lb, lr = lens
        if not (0 < t < lt <= lb < b < grid.shape[0] - 1
                and 0 < l < ll <= lr < r < grid.shape[1] - 1):
            continue
        outer = (t, l, b, r)
        if not visible[_perimeter(grid.shape, outer)].all():
            continue
        walls, horizontal, vertical = _bridge(visible, lens)
        labels, rooms = _regions(walls)
        if not rooms or not all(room["rectangular"] for room in rooms):
            continue
        boundary_boxes = [(a - 1, c - 1, z + 1, d + 1) for a, c, z, d in (room["box"] for room in rooms)]
        if not np.array_equal(walls, _wall_union(grid.shape, boundary_boxes)):
            continue
        networks.append((color, outer, walls, labels, rooms, horizontal, vertical))
    if len(networks) != 1:
        raise ValueError("Expected one recoverable enclosing wall network")
    wall, outer, walls, labels, rooms, horizontal, vertical = networks[0]
    teams = [c for c in colors if c not in (bg, frame, fill, wall)]
    if not 2 <= len(teams) <= 4:
        raise ValueError("Expected 2..4 locally visible teams")
    if np.any((grid == fill) & ~hidden):
        raise ValueError("Lens fill must be exclusive to its interior")
    for room in rooms:
        room_mask = labels == room["id"]
        exposed = room_mask & ~hidden
        if not exposed.any():
            raise ValueError("Every room must retain exposed interior")
        owners = {int(c) for c in np.unique(grid[exposed]) if int(c) != bg}
        if len(owners) > 1 or not owners.issubset(teams):
            raise ValueError("A room can be claimed by at most one team")
        room["owner"] = next(iter(owners)) if owners else None
    scores = _census(grid, labels, teams)
    winner = _maximum(scores)
    t, l, b, r = lens
    output = np.full((b - t - 1, r - l - 1), fill, dtype=np.int8)
    crop_labels = labels[t + 1:b, l + 1:r]
    for room in rooms:
        if room["owner"] == winner:
            output[crop_labels == room["id"]] = winner
    output[walls[t + 1:b, l + 1:r]] = wall
    touched = {int(i) for i in np.unique(crop_labels) if i > 0}
    local = {c: sum(room["owner"] == c and room["id"] in touched for room in rooms) for c in teams}
    pixels = {c: int(np.count_nonzero(grid == c)) for c in teams}
    # The omission experiment treats the opaque patch as missing walls. It
    # allows several teams in a merged visible region, one vote per team there.
    visible_labels, _ = _regions(grid == wall)
    visible_scores = _census(grid, visible_labels, teams)
    return output, {"background": bg, "frame": frame, "fill": fill, "wall": wall,
                    "teams": teams, "lens": lens, "outer": outer, "walls": walls,
                    "labels": labels, "rooms": rooms, "scores": scores,
                    "winner": winner, "local": local, "pixels": pixels,
                    "visible_scores": visible_scores, "touched": touched,
                    "partial": _partial(walls, outer),
                    "hidden_horizontal": bool(horizontal[t + 1:b, l + 1:r].any()),
                    "hidden_vertical": bool(vertical[t + 1:b, l + 1:r].any())}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from local role geometry, wall arms, and visible room claims."""
    return _read(input_grid)[0]


def _split_options(box):
    t, l, b, r = box
    return ([(0, cut) for cut in range(t + 2, b - 1)]
            + [(1, cut) for cut in range(l + 2, r - 1)])


def _split(box, axis, cut):
    t, l, b, r = box
    return [(t, l, cut, r), (cut, l, b, r)] if axis == 0 else [(t, l, b, cut), (t, cut, b, r)]


def _offsets(start, end, number, rng):
    gaps = np.full(number, 2, dtype=int)
    for _ in range(end - start - 2 * number):
        gaps[int(rng.integers(number))] += 1
    return [start] + list(map(int, start + np.cumsum(gaps)))


def _partition(outer, count, style, rng):
    t, l, b, r = outer
    boxes = [outer]
    if style in ("strips", "grid", "mixed"):
        base = 4 if style == "mixed" else count
        factors = [(a, base // a) for a in range(1, base + 1) if base % a == 0
                   and 2 * a <= b - t and 2 * (base // a) <= r - l
                   and (style != "strips" or a == 1 or a == base)]
        if factors:
            nr, nc = factors[int(rng.integers(len(factors)))]
            ys, xs = _offsets(t, b, nr, rng), _offsets(l, r, nc, rng)
            boxes = [(ys[i], xs[j], ys[i + 1], xs[j + 1]) for i in range(nr) for j in range(nc)]
    while len(boxes) < count:
        options = [(i, axis, cut) for i, box in enumerate(boxes) for axis, cut in _split_options(box)]
        if not options:
            return None
        i, axis, cut = options[int(rng.integers(len(options)))]
        boxes[i:i + 1] = _split(boxes[i], axis, cut)
    return boxes


def _lens_info(shape, boxes, walls, lens):
    t, l, b, r = lens
    hidden = _closed(shape, lens)
    recovered, _, _ = _bridge(walls & ~hidden, lens)
    if not np.array_equal(recovered, walls) or not walls[t + 1:b, l + 1:r].any():
        return None
    available, touched = [], []
    for i, (a, c, z, d) in enumerate(boxes):
        room = np.zeros(shape, dtype=bool)
        room[a + 1:z, c + 1:d] = True
        sites = np.argwhere(room & ~hidden)
        if len(sites) < 1:
            return None
        available.append(sites)
        if room[t + 1:b, l + 1:r].any():
            touched.append(i)
    if len(touched) < 3 or len(boxes) - len(touched) < 2:
        return None
    return available, touched


def _geometry(rng, count, witness, preference):
    for _ in range(45):
        h, w = map(int, rng.integers(17 if count <= 10 else 21, 27, size=2))
        margins = list(map(int, rng.integers(1, 3, size=4)))
        if h + margins[0] + margins[2] > 30 or w + margins[1] + margins[3] > 30:
            continue
        shape = (h + margins[0] + margins[2], w + margins[1] + margins[3])
        outer = (margins[0], margins[1], margins[0] + h - 1, margins[1] + w - 1)
        style = preference if rng.random() < 0.4 else str(rng.choice(["recursive", "recursive", "mixed", "grid", "strips"]))
        boxes = _partition(outer, count, style, rng)
        if boxes is None:
            continue
        walls = _wall_union(shape, boxes)
        if witness == "global" and not _partial(walls, outer):
            continue
        ot, ol, ob, ore = outer
        for _ in range(160):
            # Choose lengths before placement, with varied small and wide lenses.
            lh = int(rng.integers(4, min(15, h - 3)))
            lw = int(rng.integers(4, min(15, w - 3)))
            t = int(rng.integers(ot + 2, ob - lh))
            l = int(rng.integers(ol + 2, ore - lw))
            lens = (t, l, t + lh - 1, l + lw - 1)
            info = _lens_info(shape, boxes, walls, lens)
            if info is not None:
                available, touched = info
                if witness and any(len(sites) < 3 for sites in available):
                    continue
                return shape, outer, boxes, walls, lens, available, touched
    # Safety geometry: four rooms meet a crossing lens; unequal strips on the
    # exposed right side provide additional rooms and staggered T-junctions.
    shape, outer = (28, 30), (1, 1, 26, 28)
    boxes = [(a, c, b, d) for a, b in ((1, 13), (13, 26)) for c, d in ((1, 9), (9, 19))]
    ys = _offsets(1, 26, count - 4, rng)
    if count == 6 and ys[1] == 13:
        ys[1] = 11
    boxes += [(a, 19, b, 28) for a, b in zip(ys, ys[1:])]
    walls = _wall_union(shape, boxes)
    lens = (int(rng.integers(7, 11)), int(rng.integers(4, 7)),
            int(rng.integers(16, 20)), int(rng.integers(12, 16)))
    info = _lens_info(shape, boxes, walls, lens)
    assert info is not None
    return shape, outer, boxes, walls, lens, *info


def _owners(rng, count, touched, teams, witness):
    hit, external = list(map(int, rng.permutation(touched))), [i for i in rng.permutation(count) if i not in touched]
    if witness:
        possibilities = []
        for win_count in range(3, min(5, count - 1) + 1):
            rival_count = win_count - 1
            if witness == "pixels":
                local_pairs = [(a, 1) for a in range(2, min(win_count, len(hit) - 1) + 1)]
            else:
                local_pairs = [(1, b) for b in range(2, min(rival_count, len(hit) - 1) + 1)]
            for wh, rh in local_pairs:
                if win_count - wh + rival_count - rh <= len(external):
                    possibilities.append((win_count, rival_count, wh, rh))
        if not possibilities:
            return None
        wc, rc, wh, rh = possibilities[int(rng.integers(len(possibilities)))]
        owned = [None] * count
        wids = hit[:wh] + external[:wc - wh]
        rids = hit[wh:wh + rh] + external[wc - wh:wc - wh + rc - rh]
        for i in wids:
            owned[i] = teams[0]
        for i in rids:
            owned[i] = teams[1]
        return owned
    for _ in range(100):
        values = rng.integers(0, len(teams) + 1, size=count)
        owned = [teams[int(v)] if v < len(teams) else None for v in values]
        scores = Counter(c for c in owned if c is not None)
        if len(scores) != len(teams) or max(scores.values()) < 2:
            continue
        winners = [c for c, n in scores.items() if n == max(scores.values())]
        if len(winners) != 1:
            continue
        hit_owners = {owned[i] for i in hit}
        if winners[0] in hit_owners and any(c is not None and c != winners[0] for c in hit_owners):
            return owned
    return None


def _render(rng, geometry, roles, owned, witness):
    shape, outer, boxes, walls, lens, available, touched = geometry
    bg, wall, frame, fill = roles[:4]
    grid = np.full(shape, bg, dtype=np.int8)
    grid[walls] = wall
    scores = Counter(c for c in owned if c is not None)
    winner = _maximum(scores)
    for owner, sites in zip(owned, available):
        if owner is None:
            continue
        if witness == "pixels":
            amount = 1 if owner == winner else int(rng.integers(3, 7))
        else:
            amount = int(rng.integers(1, 7))
        amount = min(amount, len(sites))
        for y, x in sites[rng.choice(len(sites), amount, replace=False)]:
            grid[y, x] = owner
    t, l, b, r = lens
    # Ground truth uses the explicit clean partition and room ownership, not
    # a reconstruction reader or a generated scene hidden from the solver.
    full_answer = np.full(shape, fill, dtype=np.int8)
    for box, owner in zip(boxes, owned):
        if owner == winner:
            a, c, z, d = box
            full_answer[a + 1:z, c + 1:d] = winner
    full_answer[walls] = wall
    output = full_answer[t + 1:b, l + 1:r].copy()
    grid[t:b + 1, l:r + 1] = fill
    grid[_perimeter(shape, lens)] = frame
    turns, reflection = int(rng.integers(4)), bool(rng.integers(2))
    grid, output = np.rot90(grid, turns), np.rot90(output, turns)
    if reflection:
        grid, output = np.fliplr(grid), np.fliplr(output)
    return {"input": grid.copy(), "output": output.copy()}


def _witnesses(scene):
    win = scene["winner"]
    pixel_max = max(scene["pixels"].values())
    pixel_winners = [c for c, n in scene["pixels"].items() if n == pixel_max]
    visible = scene["visible_scores"]
    no_b_unique = visible[win] > max(n for c, n in visible.items() if c != win)
    local = scene["local"]
    local_max = max(local.values())
    local_winners = [c for c, n in local.items() if n == local_max]
    has_win = local[win] > 0
    has_loser = any(n > 0 for c, n in local.items() if c != win)
    hidden = scene["hidden_horizontal"] or scene["hidden_vertical"]
    pixel = (len(pixel_winners) == 1 and pixel_winners[0] != win
             and visible != scene["scores"] and not no_b_unique)
    global_vote = len(local_winners) == 1 and local_winners[0] != win and scene["partial"]
    return {"pixels": bool(pixel and hidden and has_win and has_loser),
            "global": bool(global_vote and hidden and has_win and has_loser)}


def _make_pair(rng, count, witness, parameters):
    for _ in range(30):
        geometry = _geometry(rng, count, witness, parameters["preference"])
        k = 2 if witness else int(rng.integers(2, 5))
        roles = (parameters["roles"] if parameters["fixed_roles"]
                 else list(map(int, rng.permutation(10))))
        teams = list(map(int, rng.permutation(roles[4:4 + k])))
        owned = _owners(rng, count, geometry[-1], teams, witness)
        if owned is None:
            continue
        pair = _render(rng, geometry, roles, owned, witness)
        predicted, scene = _read(pair["input"])
        assert np.array_equal(predicted, pair["output"]), "Clean renderer and pixel solver disagree"
        assert len(scene["rooms"]) == count
        if witness is None or _witnesses(scene)[witness]:
            return pair
    raise RuntimeError("Could not construct a witnessed room election")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _check_counts(num_train, num_test)
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    parameters = {"roles": list(map(int, rng.permutation(10))),
                  "fixed_roles": bool(rng.random() < 0.25),
                  "preference": str(rng.choice(["recursive", "mixed", "grid", "strips"]))}
    puzzle, seen = {"train": [], "test": []}, set()
    for split, n in (("train", nt), ("test", ne)):
        for i in range(n):
            count = int(rng.integers(6, 11) if split == "train" else rng.integers(11, 16))
            witness = "pixels" if split == "train" and i == 0 else "global" if split == "train" and i == 1 else None
            for _ in range(12):
                pair = _make_pair(rng, count, witness, parameters)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Distinct inputs not found within bounded retries")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _euler_count(mask):
    # The connected wall-pixel graph has one cycle per bounded room, excluding
    # filled 2x2 unit faces (which have no interior cell).
    edges = int(np.count_nonzero(mask[:, :-1] & mask[:, 1:]))
    edges += int(np.count_nonzero(mask[:-1] & mask[1:]))
    unit_faces = int(np.count_nonzero(mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]))
    return edges - int(mask.sum()) + 1 - unit_faces


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, scenes = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            key = _key(grid)
            assert key not in seen, "Duplicate input"
            seen.add(key)
            prediction, scene = _read(grid)
            assert np.array_equal(output, prediction), "Wrong room-census reveal"
            assert len(scene["rooms"]) == _euler_count(scene["walls"]), "Topology count disagreement"
            assert scene["scores"][scene["winner"]] >= 2
            local = scene["local"]
            assert local[scene["winner"]] > 0 and any(n > 0 for c, n in local.items() if c != scene["winner"])
            assert scene["hidden_horizontal"] or scene["hidden_vertical"]
            assert {scene["wall"], scene["winner"], scene["fill"]}.issubset(set(map(int, np.unique(output))))
            scenes[split].append(scene)
    witnesses = [_witnesses(scene) for scene in scenes["train"]]
    assert any(w["pixels"] for w in witnesses), "Missing dot-total / no-completion counterexample"
    assert any(w["global"] for w in witnesses), "Missing global census / partial-divider counterexample"
    assert min(len(s["rooms"]) for s in scenes["test"]) > max(len(s["rooms"]) for s in scenes["train"]), "Missing room-count shift"


def self_test():
    """Bounded meaningful tests and a compact audit over 42 distinct seeds."""
    shapes, outputs, topologies, team_counts, room_counts, palettes = set(), set(), set(), set(), set(), set()
    pixels = globals_ = partials = crossings = unclaimed = 0
    area_counterexamples = component_counterexamples = 0
    digests = set()
    samples = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digests.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(_key(a[k]) == _key(b[k]) for s in puzzle for a, b in zip(puzzle[s], again[s]) for k in a)
            samples.extend(puzzle["train"] + puzzle["test"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _, scene = _read(pair["input"])
                shapes.add(pair["input"].shape)
                outputs.add(pair["output"].shape)
                team_counts.add(len(scene["teams"]))
                room_counts.add(len(scene["rooms"]))
                palettes.add((scene["background"], scene["wall"], scene["frame"], scene["fill"]))
                # Remove numeric spacing and color while retaining endpoints.
                boxes = [room["box"] for room in scene["rooms"]]
                ys = sorted({y for t, l, b, r in boxes for y in (t - 1, b + 1)})
                xs = sorted({x for t, l, b, r in boxes for x in (l - 1, r + 1)})
                norm = tuple(sorted((ys.index(t - 1), xs.index(l - 1), ys.index(b + 1), xs.index(r + 1)) for t, l, b, r in boxes))
                topologies.add(norm)
                evidence = _witnesses(scene)
                pixels += evidence["pixels"]
                globals_ += evidence["global"]
                if split == "train":
                    # Audit two other plausible substitutes for a room census.
                    areas = {c: sum(room["area"] for room in scene["rooms"]
                                    if room["owner"] == c) for c in scene["teams"]}
                    components = {c: len(_regions(pair["input"] != c)[1])
                                  for c in scene["teams"]}
                    for scores, kind in ((areas, "area"), (components, "components")):
                        leaders = [c for c, n in scores.items() if n == max(scores.values())]
                        contradicts = len(leaders) == 1 and leaders[0] != scene["winner"]
                        if kind == "area":
                            area_counterexamples += contradicts
                        else:
                            component_counterexamples += contradicts
                partials += scene["partial"]
                crossings += scene["hidden_horizontal"] and scene["hidden_vertical"]
                unclaimed += any(room["owner"] is None for room in scene["rooms"])
    assert len(digests) == 42
    assert len(shapes) >= 30 and len(outputs) >= 20 and len(topologies) >= 35
    assert team_counts == {2, 3, 4} and {6, 10, 11, 15}.issubset(room_counts)
    assert len(palettes) >= 25 and pixels >= 42 and globals_ >= 42
    assert partials >= 42 and crossings >= 5 and unclaimed >= 42
    assert area_counterexamples >= 10 and component_counterexamples >= 42
    # Color equivariance and all eight geometric symmetries use only pixels.
    permutation = np.array([9, 3, 7, 1, 8, 0, 5, 2, 6, 4], dtype=np.int8)
    for pair in samples[:6]:
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
        for turns in range(4):
            for flip in (False, True):
                grid, output = np.rot90(pair["input"], turns), np.rot90(pair["output"], turns)
                if flip:
                    grid, output = np.fliplr(grid), np.fliplr(output)
                assert np.array_equal(solve(grid), output)
    # A hand-built three-room plan is outside the generation count range and
    # establishes that the solver does not depend on that range or a seed.
    base = np.zeros((18, 24), dtype=int)
    boxes = [(1, 1, 16, 8), (1, 8, 16, 15), (1, 15, 16, 22)]
    walls = _wall_union(base.shape, boxes)
    base[walls] = 2
    base[3, 4] = base[3, 11] = 4
    base[3, 18] = 6
    grid = base.copy()
    lens = (6, 5, 12, 18)
    grid[6:13, 5:19] = 7
    grid[_perimeter(grid.shape, lens)] = 9
    expected = np.full((5, 12), 4, dtype=int)
    expected[:, 9:] = 7
    expected[:, 2] = expected[:, 9] = 2
    assert np.array_equal(solve(grid), expected)
    # A minimal 3x3 lens has a 1x1 answer. It can cover a winning
    # interior, a supported wall, or a losing interior under the same rule.
    for lens, value in (((6, 5, 8, 7), 4), ((6, 7, 8, 9), 2), ((6, 16, 8, 18), 7)):
        grid = base.copy()
        grid[_closed(grid.shape, lens)] = 7
        grid[_perimeter(grid.shape, lens)] = 9
        assert np.array_equal(solve(grid), np.array([[value]]))
    # Change only an external room's claim. The lens pixels and geometry
    # remain identical, yet the global election and the reveal both reverse.
    external_claim_flips = 0
    for seed in range(3):
        puzzle = generate_puzzle(seed, 2, 1)
        for pair in puzzle["train"]:
            _, scene = _read(pair["input"])
            if not _witnesses(scene)["global"]:
                continue
            winner = scene["winner"]
            rival = next(c for c in scene["teams"] if c != winner)
            assert scene["scores"][winner] == scene["scores"][rival] + 1
            room = next(r for r in scene["rooms"] if r["owner"] == winner
                        and r["id"] not in scene["touched"])
            changed = pair["input"].copy()
            claim = (scene["labels"] == room["id"]) & (changed == winner)
            changed[claim] = rival
            hidden = _closed(changed.shape, scene["lens"])
            assert np.array_equal(changed[hidden], pair["input"][hidden])
            prediction, modified = _read(changed)
            assert np.array_equal(modified["walls"], scene["walls"])
            assert modified["winner"] == rival
            assert not np.array_equal(prediction, pair["output"])
            external_claim_flips += 1
    assert external_claim_flips == 3
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(4242 + nt + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    bad = generate_puzzle(101, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Incorrect output accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(digests),
            "input_shapes": len(shapes), "output_shapes": len(outputs),
            "spacing_normalized_topologies": len(topologies), "team_counts": sorted(team_counts),
            "room_counts": sorted(room_counts), "role_palettes": len(palettes),
            "pixel_and_omitted_completion_witnesses": pixels,
            "global_and_partial_witnesses": globals_, "partial_plans": partials,
            "hidden_crossings": crossings, "plans_with_unclaimed_rooms": unclaimed,
            "training_area_counterexamples": area_counterexamples,
            "training_marker_component_counterexamples": component_counterexamples,
            "external_claim_flip_checks": external_claim_flips,
            "independent_semantic_approval": False}
