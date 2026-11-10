"""Blueprint archipelagos: complete topology, then transfer original pigments.

Only standard-library modules and NumPy are used. The predictor reads its input
afresh; construction plans never participate in inference.
"""
from __future__ import annotations

from functools import lru_cache
import itertools

import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
TILE_SHAPES = ((3, 3), (3, 4), (4, 3), (4, 4))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                   (num_test, "num_test", 1, 4)):
        _require(_integer(value) and low <= value <= high, name + " is outside its legal integer range")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= s <= 30 for s in grid.shape), "invalid grid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "invalid ARC color")
    return grid.astype(np.int8, copy=True)


def _key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def _normalized(grid):
    roles = {}
    return grid.shape, tuple(roles.setdefault(int(c), len(roles)) for c in grid.flat)


def _components(points):
    remaining, result = set(points), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in DIRECTIONS:
                neighbor = r + dr, c + dc
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    todo.append(neighbor)
        result.append(component)
    return result


def _mask_components(mask):
    active = {tuple(map(int, p)) for p in np.argwhere(mask)}
    components = _components(active)
    labels = np.full(mask.shape, -1, dtype=np.int8)
    for label, component in enumerate(components):
        for p in component:
            labels[p] = label
    return components, labels


def _origins(mask, lattice_shape, positions):
    h, w = mask.shape
    rows, cols = lattice_shape
    return [(r, c) for r in range(rows - h + 1) for c in range(cols - w + 1)
            if all(0 <= y-r < h and 0 <= x-c < w and mask[y-r, x-c]
                   for y, x in positions)]


def _ray(labels, start, direction):
    """Skip water and the source island; stop at the first OTHER island."""
    source = int(labels[start])
    r, c = start
    dr, dc = direction
    while True:
        r, c = r + dr, c + dc
        if not (0 <= r < labels.shape[0] and 0 <= c < labels.shape[1]):
            return None
        target = int(labels[r, c])
        if target >= 0 and target != source:
            return (r, c), target


def _arrow_candidates(mask):
    _, labels = _mask_components(mask)
    candidates = []
    for point in sorted(map(tuple, np.argwhere(mask))):
        point = tuple(map(int, point))
        for d, direction in enumerate(DIRECTIONS):
            hit = _ray(labels, point, direction)
            if hit is not None:
                targetpoint, target = hit
                candidates.append(dict(point=point, direction=d, source=int(labels[point]),
                                       target=target, targetpoint=targetpoint))
    return candidates


@lru_cache(maxsize=4)
def _t_shapes(shape):
    """Every translated cardinal T that fits completely inside a tile."""
    h, w = shape
    stencils = []
    for r in range(h):
        for c in range(w):
            for d, (dr, dc) in enumerate(DIRECTIONS):
                arms = ((r-dc, c+dr), (r+dc, c-dr))
                tip = r+dr, c+dc
                write = {(r, c), *arms, tip}
                if all(0 <= y < h and 0 <= x < w for y, x in write):
                    stencils.append(((r, c), d, arms, tip, write))
    return tuple(stencils)


def _read_arrow(tile, reserved):
    parses = []
    for center, d, arms, tip, write in _t_shapes(tile.shape):
        marker = int(tile[tip])
        if marker in reserved or any(int(tile[p]) != marker for p in arms):
            continue
        outside = [int(tile[p]) for p in np.ndindex(tile.shape) if p not in write]
        if len(set(outside)) != 1:
            continue
        host = outside[0]
        token = int(tile[center])
        if host in reserved or host == marker or token in reserved:
            continue
        if token == marker:
            payload, explicit = host, False
        else:
            payload, explicit = token, True
        parses.append(dict(center=center, direction=d, host=host, marker=marker,
                           payload=payload, explicit=explicit))
    _require(len(parses) == 1, "control tile must have exactly one T interpretation")
    return parses[0]


def _runs(indices):
    groups = []
    for value in map(int, indices):
        if groups and value == groups[-1][1] + 1:
            groups[-1] = groups[-1][0], value
        else:
            groups.append((value, value))
    return groups


def _lattice(grid):
    divider = int(grid[0, 0])
    rg = _runs(np.flatnonzero(np.all(grid == divider, axis=1)))
    cg = _runs(np.flatnonzero(np.all(grid == divider, axis=0)))
    _require(len(rg) >= 4 and len(cg) >= 4, "missing complete lattice dividers")
    _require(rg[0][0] == cg[0][0] == 0 and rg[-1][1] == grid.shape[0]-1
             and cg[-1][1] == grid.shape[1]-1, "missing outer divider frame")
    thicknesses = {b-a+1 for a, b in rg + cg}
    heights = {rg[i+1][0]-rg[i][1]-1 for i in range(len(rg)-1)}
    widths = {cg[i+1][0]-cg[i][1]-1 for i in range(len(cg)-1)}
    _require(len(thicknesses) == len(heights) == len(widths) == 1, "irregular lattice")
    t, h, w = thicknesses.pop(), heights.pop(), widths.pop()
    _require(t in (1, 2) and (h, w) in TILE_SHAPES, "unsupported tile geometry")
    _require(len(rg)-1 >= h and len(cg)-1 >= w, "stencil window cannot fit")
    tiles = {}
    for r in range(len(rg)-1):
        for c in range(len(cg)-1):
            y, x = rg[r][1]+1, cg[c][1]+1
            tile = grid[y:y+h, x:x+w].copy()
            _require(divider not in tile, "divider color occurs in a tile")
            tiles[r, c] = tile
    return (h, w, t, len(rg)-1, len(cg)-1), divider, tiles


def _infer(input_grid):
    grid = _grid(input_grid)
    geometry, divider, tiles = _lattice(grid)
    h, w, t, rows, cols = geometry
    records = {}
    for point, tile in tiles.items():
        records.setdefault(_key(tile), [tile, []])[1].append(point)
    ordinary, ordinary_positions = max(records.values(), key=lambda item: len(item[1]))
    _require(2 * len(ordinary_positions) > rows*cols, "ordinary pattern lacks a strict majority")
    ordinary_colors = set(map(int, np.unique(ordinary)))
    _require(len(ordinary_colors) == 2, "ordinary tile must use two colors")
    reserved = ordinary_colors | {divider}
    glyphs, arrows, originals, mask, background = [], [], {}, None, None
    for point, tile in tiles.items():
        if np.array_equal(tile, ordinary):
            continue
        colors = set(map(int, np.unique(tile)))
        if colors & ordinary_colors:
            shared, novel = colors & ordinary_colors, colors - ordinary_colors
            _require(len(shared) == len(novel) == 1, "invalid blueprint tile palette")
            bg, fill = next(iter(shared)), next(iter(novel))
            local_mask = tile == fill
            if mask is None:
                mask, background = local_mask.copy(), bg
            _require(bg == background and np.array_equal(local_mask, mask), "blueprint copies disagree")
            glyphs.append(point)
            originals[point] = fill
        else:
            arrow = _read_arrow(tile, reserved)
            arrow["point"] = point
            arrows.append(arrow)
            originals[point] = arrow["host"]
    _require(mask is not None and len(glyphs) >= 2 and len(arrows) >= 1, "insufficient blueprint or arrow evidence")
    marker_colors = {a["marker"] for a in arrows}
    _require(len(marker_colors) == 1, "arrows must share one marker color within a grid")
    marker = marker_colors.pop()
    _require(marker not in reserved and marker not in originals.values()
             and all(a["payload"] != marker for a in arrows), "marker and pigment roles overlap")
    components, local_labels = _mask_components(mask)
    _require(2 <= len(components) <= 5 and 4 <= int(mask.sum()) < h*w, "invalid archipelago stencil")
    observed = set(originals)
    origins = _origins(mask, (rows, cols), observed)
    _require(len(origins) == 1, "supplied positions do not uniquely register the blueprint")
    origin = origins[0]
    labels = np.full((rows, cols), -1, dtype=np.int8)
    labels[origin[0]:origin[0]+h, origin[1]:origin[1]+w] = local_labels
    active = {tuple(map(int, p)) for p in np.argwhere(labels >= 0)}
    _require(observed < active, "a strict incomplete archipelago is required")
    fills = [None] * len(components)
    for point, fill in originals.items():
        label = int(labels[point])
        _require(fills[label] in (None, fill), "one island has inconsistent original pigment clues")
        fills[label] = fill
    _require(all(c is not None for c in fills), "every island needs an original pigment clue")
    assignments = {}
    for arrow in arrows:
        point = arrow["point"]
        source = int(labels[point])
        hit = _ray(labels, point, DIRECTIONS[arrow["direction"]])
        _require(hit is not None, "arrow points to no other island")
        targetpoint, target = hit
        payload = arrow["payload"] if arrow["explicit"] else fills[source]
        _require(target not in assignments or assignments[target] == payload, "conflicting incoming pigments")
        assignments[target] = payload
        arrow.update(source=source, target=target, targetpoint=targetpoint)
    final_fills = [assignments.get(i, c) for i, c in enumerate(fills)]
    _require(final_fills != fills, "at least one island must change pigment")
    output = grid.copy()
    no_arrows = grid.copy()
    for point in active:
        r, c = point
        y, x = t+r*(h+t), t+c*(w+t)
        label = int(labels[point])
        output[y:y+h, x:x+w] = np.where(mask, final_fills[label], background)
        no_arrows[y:y+h, x:x+w] = np.where(mask, fills[label], background)
    pieces = [len(_components({p for p in observed if labels[p] == i}))
              for i in range(len(components))]
    topology = any(pieces[i] > 1 and final_fills[i] != fills[i] for i in range(len(components)))
    visible_labels = np.full(labels.shape, -1, dtype=np.int8)
    for point in observed:
        visible_labels[point] = labels[point]
    hidden = []
    for a in arrows:
        visible_hit = _ray(visible_labels, a["point"], DIRECTIONS[a["direction"]])
        hidden.append(visible_hit is None or visible_hit[1] != a["target"])
    hidden_hit = any(hidden)
    coupled = any(hide and pieces[a["target"]] > 1 and final_fills[a["target"]] != fills[a["target"]]
                  for a, hide in zip(arrows, hidden))
    simultaneous = any(not a["explicit"] and not b["explicit"]
                       and a["target"] == b["source"] and fills[a["source"]] != fills[b["source"]]
                       for a in arrows for b in arrows if a is not b)
    locality = any(fills[i] == fills[j] and final_fills[i] != final_fills[j]
                   for i in range(len(fills)) for j in range(i))
    info = dict(geometry=geometry, divider=divider, ordinary=ordinary, mask=mask,
                background=background, origin=origin, labels=labels, active=active,
                observed=observed, glyphs=glyphs, arrows=arrows, fills=fills,
                final_fills=final_fills, no_arrows=no_arrows, marker=marker,
                topology=topology, hidden_hit=hidden_hit, simultaneous=simultaneous,
                locality=locality, coupled=coupled, explicit=any(a["explicit"] for a in arrows),
                missing=len(active-observed), components=len(components))
    return output, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete the input-visible stencil topology and execute its pigment arrows."""
    return _infer(input_grid)[0]


def _asymmetric(mask):
    def cropped_key(array):
        y, x = np.nonzero(array)
        return _key(array[y.min():y.max()+1, x.min():x.max()+1])
    target = cropped_key(mask)
    alternatives = [np.rot90(mask, k) for k in (1, 2, 3)]
    alternatives += [np.fliplr(np.rot90(mask, k)) for k in range(4)]
    return all(cropped_key(m) != target for m in alternatives)


def _random_mask(rng, shape, maximum):
    # Arbitrary subsets dominate; structured bars and rings are also useful.
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    if rng.random() < .22:
        if rng.integers(2):
            mask[:, 0] = mask[:, -1] = True
            if rng.random() < .5:
                mask[int(rng.integers(h)), int(rng.integers(1, w-1))] = True
        else:
            mask[0, :] = mask[-1, :] = True
            if rng.random() < .5:
                mask[int(rng.integers(1, h-1)), int(rng.integers(w))] = True
    else:
        n = int(rng.integers(4, min(maximum, h*w-1)+1))
        mask.flat[rng.choice(h*w, n, replace=False)] = True
    if rng.integers(2):
        mask = mask[::-1]
    if rng.integers(2):
        mask = mask[:, ::-1]
    return mask.copy()


def _geometry(rng, shape=None):
    h, w = TILE_SHAPES[int(rng.integers(4))] if shape is None else shape
    t = int(rng.choice((1, 2), p=(.8, .2)))
    rows = int(rng.integers(h, (30-t)//(h+t)+1))
    cols = int(rng.integers(w, (30-t)//(w+t)+1))
    return h, w, t, rows, cols


def _choose_arrows(rng, mask, profile):
    candidates = _arrow_candidates(mask)
    if not candidates:
        return None
    candidates = [candidates[int(i)] for i in rng.permutation(len(candidates))]
    if profile == "topology":
        # Both arrows are implicit. Select reverse component edges at freely
        # sampled source cells rather than obliging the same two ray endpoints.
        pairs = [(a, b) for a in candidates for b in candidates
                 if a["source"] == b["target"] and a["target"] == b["source"]]
        if not pairs:
            return None
        a, b = pairs[int(rng.integers(len(pairs)))]
        return [dict(a, explicit=False), dict(b, explicit=False)]
    if profile == "payload":
        return [dict(candidates[0], explicit=True)]
    limit = min(4, int(mask.sum())-3)
    if profile == "combined" and limit < 2:
        return None
    desired = int(rng.integers(1, limit+1)) if profile == "free" else int(rng.integers(2, limit+1))
    selected, used, targets = [], set(), set()
    for candidate in candidates:
        # One incoming command per target keeps arbitrary payload choices
        # unambiguous. The solver also supports consistent duplicate commands.
        if candidate["point"] in used or candidate["target"] in targets:
            continue
        explicit = (len(selected) == 0 if profile == "combined" else bool(rng.random() < .4))
        selected.append(dict(candidate, explicit=explicit))
        used.add(candidate["point"])
        targets.add(candidate["target"])
        if len(selected) == desired:
            break
    if profile == "combined" and (len(selected) < 2 or all(a["explicit"] for a in selected)):
        return None
    return selected


def _fragment(rng, mask, lattice_shape, origin, arrows, profile, excluded=()):
    active = sorted(map(tuple, np.argwhere(mask)))
    active = [tuple(map(int, p)) for p in active]
    mandatory = {a["point"] for a in arrows}
    free = [p for p in active if p not in mandatory]
    if len(free) < 3:
        return None
    components, labels = _mask_components(mask)
    sizes = list(range(2, len(free)))
    rng.shuffle(sizes)
    for size in sizes:
        combinations = list(itertools.combinations(free, size))
        for idx in rng.permutation(len(combinations)):
            visible = mandatory | set(combinations[int(idx)])
            relative = tuple(sorted(visible))
            if relative in excluded or any(not (visible & comp) for comp in components):
                continue
            if profile == "topology":
                visible_labels = np.full(labels.shape, -1, dtype=np.int8)
                for point in visible:
                    visible_labels[point] = labels[point]
                coupled = False
                for a in arrows:
                    hit = _ray(visible_labels, a["point"], DIRECTIONS[a["direction"]])
                    if ((hit is None or hit[1] != a["target"])
                            and len(_components(visible & components[a["target"]])) > 1):
                        coupled = True
                if not coupled:
                    continue
            absolute = {(origin[0]+r, origin[1]+c) for r, c in visible}
            if _origins(mask, lattice_shape, absolute) == [origin]:
                return visible
    return None


def _paint(geometry, divider, ordinary, mask, background, labels, fills, visible, arrows, marker, rng):
    h, w, t, rows, cols = geometry
    inp = np.full((rows*(h+t)+t, cols*(w+t)+t), divider, dtype=np.int8)
    by_point = {a["point"]: a for a in arrows}
    for r in range(rows):
        for c in range(cols):
            point = r, c
            tile = ordinary
            if point in visible:
                fill = fills[int(labels[point])]
                if point in by_point:
                    arrow = by_point[point]
                    possible = [s for s in _t_shapes((h, w)) if s[1] == arrow["direction"]]
                    center, _, arms, tip, write = possible[int(rng.integers(len(possible)))]
                    tile = np.full((h, w), fill, dtype=np.int8)
                    for p in write:
                        tile[p] = marker
                    if arrow["explicit"]:
                        tile[center] = arrow["payload"]
                else:
                    tile = np.where(mask, fill, background).astype(np.int8)
            y, x = t+r*(h+t), t+c*(w+t)
            inp[y:y+h, x:x+w] = tile
    assignments = {a["target"]: a["payload"] for a in arrows}
    final = [assignments.get(i, color) for i, color in enumerate(fills)]
    out = inp.copy()
    for r, c in map(tuple, np.argwhere(labels >= 0)):
        y, x = t+int(r)*(h+t), t+int(c)*(w+t)
        out[y:y+h, x:x+w] = np.where(mask, final[int(labels[r, c])], background)
    return dict(input=inp, output=out)


def _make_pair(rng, profile, palette, familiar=None, forbidden=(), excluded=()):
    geometry = _geometry(rng, None if familiar is None else familiar.shape)
    h, w, t, rows, cols = geometry
    maximum = min(h*w-1, (rows*cols-1)//2)
    mask = _random_mask(rng, (h, w), maximum) if familiar is None else familiar.copy()
    if not 4 <= int(mask.sum()) <= maximum or _key(mask) in forbidden:
        return None
    components, local_labels = _mask_components(mask)
    if not 2 <= len(components) <= 5 or (profile == "payload" and not _asymmetric(mask)):
        return None
    arrows = _choose_arrows(rng, mask, profile)
    if not arrows:
        return None
    origin = int(rng.integers(rows-h+1)), int(rng.integers(cols-w+1))
    visible_local = _fragment(rng, mask, (rows, cols), origin, arrows, profile, excluded)
    if visible_local is None:
        return None
    divider, background, accent, marker, *pigments = palette
    fills = list(pigments[:len(components)])
    if profile == "payload":
        # Equal original colors + a local explicit change distinguish island
        # identity from a grid-wide color replacement even with two examples.
        fills[arrows[0]["target"]] = fills[arrows[0]["source"]]
    elif profile != "topology" and rng.random() < .3:
        a, b = rng.choice(len(fills), 2, replace=False)
        fills[int(a)] = fills[int(b)]
    for arrow in arrows:
        if arrow["explicit"]:
            choices = [c for c in pigments if c not in (fills[arrow["source"]], fills[arrow["target"]])]
            if profile == "payload":
                choices = [c for c in choices if c not in fills]
            if not choices:
                return None
            arrow["payload"] = int(rng.choice(choices))
        else:
            arrow["payload"] = fills[arrow["source"]]
    if not any(a["payload"] != fills[a["target"]] for a in arrows):
        return None
    if profile == "combined" and any(a["payload"] == fills[a["target"]] for a in arrows):
        return None
    labels = np.full((rows, cols), -1, dtype=np.int8)
    labels[origin[0]:origin[0]+h, origin[1]:origin[1]+w] = local_labels
    visible = {(origin[0]+r, origin[1]+c) for r, c in visible_local}
    for arrow in arrows:
        arrow["point"] = origin[0]+arrow["point"][0], origin[1]+arrow["point"][1]
    ordinary = np.where(rng.integers(2, size=(h, w)), accent, background).astype(np.int8)
    if len(np.unique(ordinary)) != 2 or not np.any(ordinary[~mask] != background):
        return None
    pair = _paint(geometry, divider, ordinary, mask, background, labels, fills, visible, arrows, marker, rng)
    predicted, info = _infer(pair["input"])
    _require(np.array_equal(predicted, pair["output"]), "construction and input reconstruction disagree")
    if profile == "topology" and not (info["coupled"] and info["simultaneous"]):
        return None
    if profile == "payload" and not (info["locality"] and info["explicit"]):
        return None
    return pair, info


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Reproducible puzzles with topology and payload contrasts in training."""
    _require(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    ntrain, ntest = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    stable_palette = bool(rng.random() < .3)
    base_palette = tuple(map(int, rng.permutation(10)))
    puzzle, seen, training = {"train": [], "test": []}, set(), []
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            profile = ("topology" if index == 0 else "payload" if index == 1 else "free") if split == "train" else "combined"
            familiar, excluded = None, ()
            if split == "test" and rng.random() < .45:
                choices = [f for f in training if int(f["mask"].sum()) >= 5]
                chosen = choices[int(rng.integers(len(choices)))]
                familiar = chosen["mask"].copy()
                excluded = {tuple(sorted((r-f["origin"][0], c-f["origin"][1]) for r, c in f["observed"]))
                            for f in training if _key(f["mask"]) == _key(familiar)}
            forbidden = {_key(f["mask"]) for f in training} if familiar is None else set()
            # Unseen test blueprints and distinct training blueprints use the
            # same arbitrary-mask process. Familiar tests get new fragments.
            result = None
            for attempt in range(1600):
                palette = base_palette if stable_palette else tuple(map(int, rng.permutation(10)))
                candidate = _make_pair(rng, profile, palette, familiar, forbidden, excluded)
                if candidate is None:
                    continue
                pair, info = candidate
                if _key(pair["input"]) in seen:
                    continue
                if split == "train" and index == 1 and _normalized(info["ordinary"]) == _normalized(training[0]["ordinary"]):
                    continue
                result = candidate
                break
            if result is None:
                raise RuntimeError("bounded archipelago construction exhausted")
            pair, info = result
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
            if split == "train":
                training.append(info)
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check input-visible semantics, preservation and teaching witnesses."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            expected, info = _infer(inp)
            _require(inp.shape == out.shape and np.array_equal(out, expected), "incorrect output")
            _require(not np.array_equal(out, info["no_arrows"]), "parent A has no observable contribution")
            h, w, t, rows, cols = info["geometry"]
            for r in range(rows):
                for c in range(cols):
                    y, x = t+r*(h+t), t+c*(w+t)
                    if (r, c) not in info["active"]:
                        _require(np.array_equal(inp[y:y+h, x:x+w], out[y:y+h, x:x+w]), "inactive tile changed")
            _require(np.array_equal(inp[inp == info["divider"]], out[inp == info["divider"]]), "divider changed")
            _require(info["missing"] > 0, "parent B has no completion to perform")
            _require(np.any(info["ordinary"][~info["mask"]] != info["background"]), "full-tile-copy witness missing")
            infos[split].append(info)
    train = infos["train"]
    _require(any(f["simultaneous"] and f["coupled"] for f in train), "missing topology/transfer demonstration")
    _require(any(f["explicit"] and f["locality"] for f in train), "missing local explicit-payload demonstration")
    _require(any(_asymmetric(f["mask"]) for f in train), "same-orientation copying is not distinguished")
    _require(len({_key(f["mask"]) for f in train}) >= 2, "training blueprint diversity missing")
    _require(len({_normalized(f["ordinary"]) for f in train}) >= 2, "training ordinary-pattern diversity missing")
    for f in infos["test"]:
        _require(f["explicit"] and any(not a["explicit"] for a in f["arrows"]), "test must recombine demonstrated payload modes")
        _require(all(a["payload"] != f["fills"][a["target"]] for a in f["arrows"]), "test command is a no-op")
        relatives = tuple(sorted((r-f["origin"][0], c-f["origin"][1]) for r, c in f["observed"]))
        _require(all(_key(old["mask"]) != _key(f["mask"]) or relatives != tuple(sorted(
            (r-old["origin"][0], c-old["origin"][1]) for r, c in old["observed"])) for old in train), "test fragment is not shifted")


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def _semantic_regressions():
    """Hand-constructed double bars with independently specified tile colors.

    Missing middle-right support bridges two clues. Both arrow rays initially
    see only water, so visible-only routing cannot execute the intended swap.
    This fixture does not call the construction planner or _paint.
    """
    mask = np.array(((1, 0, 1), (1, 0, 1), (1, 0, 1)), dtype=bool)
    ordinary = np.array(((0, 8, 0), (8, 0, 8), (0, 8, 0)), dtype=np.int8)
    inp = np.full((21, 21), 9, dtype=np.int8)

    def put(grid, point, tile):
        r, c = point
        grid[1+4*r:4+4*r, 1+4*c:4+4*c] = tile

    for point in np.ndindex((5, 5)):
        put(inp, point, ordinary)
    put(inp, (1, 1), np.where(mask, 2, 0))
    put(inp, (1, 3), np.where(mask, 7, 0))
    # Right-facing T on pigment 2; left-facing T on pigment 7.
    right = np.array(((2, 1, 2), (2, 1, 1), (2, 1, 2)), dtype=np.int8)
    left = np.array(((7, 1, 7), (1, 1, 7), (7, 1, 7)), dtype=np.int8)
    put(inp, (2, 1), right)
    put(inp, (3, 3), left)
    expected = inp.copy()
    for row in (1, 2, 3):
        put(expected, (row, 1), np.where(mask, 7, 0))
        put(expected, (row, 3), np.where(mask, 2, 0))
    predicted, f = _infer(inp)
    _require(np.array_equal(predicted, expected) and f["coupled"] and f["simultaneous"],
             "manual simultaneous/topology regression failed")

    # A literal junction 6 changes the right island to 6, while the left still
    # receives ORIGINAL 7 rather than the right island's updated 6.
    explicit = inp.copy()
    explicit[10, 6] = 6
    explicit_expected = expected.copy()
    for row in (1, 2, 3):
        put(explicit_expected, (row, 3), np.where(mask, 6, 0))
    _require(np.array_equal(solve(explicit), explicit_expected), "literal junction or original-color regression failed")

    # Reuse pigment 2 on both disconnected islands: recoloring the right must
    # preserve the left. This distinguishes component identity from color ID.
    repeated = explicit.copy()
    put(repeated, (1, 3), np.where(mask, 2, 0))
    put(repeated, (3, 3), np.where(left == 7, 2, left))
    repeated_expected = explicit_expected.copy()
    for row in (1, 2, 3):
        put(repeated_expected, (row, 1), np.where(mask, 2, 0))
    _require(np.array_equal(solve(repeated), repeated_expected), "repeated-color locality regression failed")

    consistent = inp.copy()
    put(consistent, (1, 1), right)
    put(consistent, (3, 1), np.where(mask, 2, 0))
    _require(np.array_equal(solve(consistent), expected), "consistent duplicate commands failed")
    conflicting = consistent.copy()
    conflicting[6, 6] = 6
    try:
        solve(conflicting)
    except ValueError as error:
        _require("conflicting incoming" in str(error), "conflict fixture failed for an unrelated reason")
    else:
        raise AssertionError("conflicting incoming arrows accepted")

    # Without the lower control clue both vertical translations fit equally.
    ambiguous = inp.copy()
    put(ambiguous, (3, 3), ordinary)
    try:
        solve(ambiguous)
    except ValueError as error:
        _require("uniquely register" in str(error), "ambiguous fixture failed for an unrelated reason")
    else:
        raise AssertionError("ambiguous blueprint translation accepted")
    inconsistent = inp.copy()
    put(inconsistent, (1, 3), np.where(mask, 6, 0))
    try:
        solve(inconsistent)
    except ValueError as error:
        _require("inconsistent original" in str(error), "inconsistent fixture failed for an unrelated reason")
    else:
        raise AssertionError("inconsistent island pigment accepted")
    return {"input": inp, "output": expected}


def self_test():
    """A bounded 42-seed semantic/diversity suite, plus adversarial checks."""
    _semantic_regressions()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records, shapes, masks, motifs, geometries, fingerprints = [], set(), set(), set(), set(), set()
    directions = {"train": set(), "test": set()}
    component_counts, missing_counts, arrow_counts, markers = set(), set(), set(), set()
    conditional, palettes, origin_edges = {}, set(), {"train": set(), "test": set()}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        fingerprints.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                input_copy = np.array(pair["input"], copy=True)
                predicted, f = _infer(input_copy)
                _require(np.array_equal(predicted, pair["output"]), "input-only solve disagrees")
                _require(np.array_equal(input_copy, pair["input"]), "solver mutated its input")
                records.append(f)
                shapes.add(pair["input"].shape)
                masks.add(_key(f["mask"]))
                motifs.add(_normalized(f["ordinary"]))
                geometries.add(f["geometry"])
                directions[split].update(a["direction"] for a in f["arrows"])
                component_counts.add(f["components"])
                missing_counts.add(f["missing"])
                arrow_counts.add(len(f["arrows"]))
                markers.add(f["marker"])
                palettes.add(tuple(map(int, np.unique(pair["input"]))))
                conditional.setdefault(_key(f["mask"]), set()).add(tuple(sorted(
                    (r-f["origin"][0], c-f["origin"][1]) for r, c in f["observed"])))
                h, w, t, rows, cols = f["geometry"]
                y, x = f["origin"]
                if y == 0:
                    origin_edges[split].add("top")
                if x == 0:
                    origin_edges[split].add("left")
                if y+h == rows:
                    origin_edges[split].add("bottom")
                if x+w == cols:
                    origin_edges[split].add("right")
                if 0 < y < rows-h and 0 < x < cols-w:
                    origin_edges[split].add("interior")
    _require(len(fingerprints) == 42, "seed diversity collapsed")
    _require(len(shapes) >= 15 and len(geometries) >= 25, "geometry diversity collapsed")
    _require(len(masks) >= 50 and len(motifs) >= 100, "structural diversity collapsed")
    _require({f["geometry"][:2] for f in records} == set(TILE_SHAPES), "tile shapes missing")
    _require({f["geometry"][2] for f in records} == {1, 2}, "divider thickness missing")
    _require(all(ds == set(range(4)) for ds in directions.values()) and markers == set(range(10)), "direction or role diversity missing")
    _require(all(edges == {"top", "left", "bottom", "right", "interior"}
                 for edges in origin_edges.values()), "split-dependent placement restriction")
    _require(sum(len(fragments) > 1 for fragments in conditional.values()) >= 10, "conditional fragment diversity missing")
    _require(len(component_counts) >= 3 and len(missing_counts) >= 4 and len(arrow_counts) >= 3, "count diversity missing")
    for name in ("topology", "hidden_hit", "simultaneous", "locality", "coupled"):
        _require(sum(f[name] for f in records) >= 42, "missing semantic witnesses: " + name)
    _require(_puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0)), "nondeterministic generation")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(4242+a*10+b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "explicit counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # The family must be equivariant under every palette permutation, including
    # changing the ordinary/divider/marker roles to zero or to former pigments.
    pair = puzzles[0]["train"][0]
    permutation = np.array((9, 3, 6, 0, 7, 1, 8, 4, 2, 5), dtype=np.int8)
    _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]), "literal-color shortcut")
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("damaged divider accepted")
    # Report measurements, not a claim of independent semantic approval.
    return dict(seeds=42, pairs=len(records), blueprints=len(masks), ordinary_patterns=len(motifs),
                canvas_shapes=len(shapes), geometries=len(geometries),
                input_palettes=len(palettes), blueprints_with_multiple_fragments=sum(len(fs) > 1 for fs in conditional.values()),
                component_counts=sorted(component_counts), arrow_counts=sorted(arrow_counts),
                missing_counts=sorted(missing_counts),
                witnesses={name: sum(bool(f[name]) for f in records)
                           for name in ("topology", "hidden_hit", "simultaneous", "locality", "coupled")})
