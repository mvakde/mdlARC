"""Ordered color-table rectangle completion, with source-faithful erasure.

Earlier table rows cover later ones, including their markers. Each ordinary
layer retains its own two Ls; a solid 2x2 layer recolors all four cells. Unused
rows do nothing and unlisted corner pairs are inert until covered. The source's
0 fill swatch clears to the current background, even when that background is
not 0. Its remapped identity is puzzle-wide and is visibly taught before use;
other literal colors, including black, retain their ordinary roles.

Pair geometry, backgrounds, mappings, local table positions and diagonals vary.
Regimes control the degree of variation, not the transformation. Validation
recovers the scene from public arrays and learns erasure from training only.
"""

import numpy as np


def _source_signature(rows):
    assert rows and all(len(row) == len(rows[0]) for row in rows)
    return (
        len(rows),
        len(rows[0]),
        tuple(int(cell) for row in rows for cell in row),
    )


_SOURCE_INPUT_SIGNATURES = {
    _source_signature((
        "0002200000", "0002000000", "0000110000", "0000100000",
        "0000001000", "0000011000", "0000000200", "0000002200",
        "1000000000", "2300000000",
    )),
    _source_signature((
        "0000000044", "0000000004", "0000011000", "0000010000",
        "0000000000", "0000000000", "0000000100", "0130401100",
        "0450440000", "0000000000",
    )),
    _source_signature((
        "33553333333333333", "33533335333333333", "33333355333333333",
        "33332233333333333", "33332344333333333", "33333343311332333",
        "33333333331322333", "33333333343333333", "33163333443333333",
        "33293333333333333", "33403331333333333", "33893331133333333",
    )),
    _source_signature((
        "000000000000000000", "000003300000000000",
        "000003000000000000", "000000000000000000",
        "000000003000000000", "000000033000000000",
        "000000000000440000", "000000000000400004",
        "000000660000000044", "000000600000000000",
        "000000000000000000", "330000000060007700",
        "670000000660007700", "440000000000000000",
        "760000000000000000",
    )),
    _source_signature((
        "9999449229999", "9999499299999", "9999999999999",
        "9119999999991", "9199999999911", "9999999999999",
        "9999559999995", "9999599999955", "9999999999999",
        "9999339999993", "9999399999933", "9999999999999",
        "9999999999999", "9999999999999", "9229999999999",
        "9599999999999", "9399999999999", "9149949929999",
        "9469449229999",
    )),
    _source_signature((
        "1111111111111111111", "1881117711111111111",
        "1881117111111111111", "1111111111111111111",
        "1111116611111111111", "1111116133111111111",
        "1111111131111612211", "1134111111316611211",
        "1168111113311111111", "1171111111111211111",
        "1182111171111221111", "1124111771111111111",
        "1111111111111111111",
    )),
}


def _grid_signature(grid):
    return (
        grid.shape[0],
        grid.shape[1],
        tuple(int(x) for x in grid.flat),
    )


def _pair_signature(pair):
    return (
        _grid_signature(pair["input"]),
        _grid_signature(pair["output"]),
    )


def _is_int(value):
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
    )


def _resolve_counts(seed, rng, num_train, num_test):
    if num_train is not None:
        if not _is_int(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)

    if num_test is not None:
        if not _is_int(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)

    if num_train is None and num_test is None:
        key = abs(int(seed))
        num_test = 1 + key % 4
        num_train = num_test + 1 + (key // 4) % (10 - num_test)
    elif num_train is None:
        low = max(2, num_test + 1)
        num_train = int(rng.integers(low, 11))
    elif num_test is None:
        high = min(4, num_train - 1)
        num_test = int(rng.integers(1, high + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _marker_cells(bounds, diagonal):
    r0, c0, r1, c1 = bounds

    if diagonal == 0:
        return {
            (r0, c0),
            (r0, c0 + 1),
            (r0 + 1, c0),
            (r1, c1),
            (r1, c1 - 1),
            (r1 - 1, c1),
        }

    return {
        (r0, c1),
        (r0, c1 - 1),
        (r0 + 1, c1),
        (r1, c0),
        (r1, c0 + 1),
        (r1 - 1, c0),
    }


def _bounds_overlap(a, b):
    return not (
        a[2] < b[0]
        or b[2] < a[0]
        or a[3] < b[1]
        or b[3] < a[1]
    )


def _recognize(coords):
    """Recover one object from visible cells, including separated thin Ls."""
    if len(coords) not in (4, 6):
        return None
    r0 = min(r for r, c in coords)
    c0 = min(c for r, c in coords)
    r1 = max(r for r, c in coords)
    c1 = max(c for r, c in coords)
    bounds = (r0, c0, r1, c1)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if len(coords) == 4:
        if h == w == 2:
            return dict(kind="block", bounds=bounds, markers=set(coords))
        return None
    if min(h, w) < 2 or (min(h, w) == 2 and max(h, w) < 4):
        return None
    # Written as local corner patterns, independently of the construction helper.
    patterns = (
        {(0, 0), (0, 1), (1, 0), (h-1, w-1), (h-1, w-2), (h-2, w-1)},
        {(0, w-1), (0, w-2), (1, w-1), (h-1, 0), (h-1, 1), (h-2, 0)},
    )
    local = {(r-r0, c-c0) for r, c in coords}
    for diagonal, pattern in enumerate(patterns):
        if local == pattern:
            return dict(kind="normal", bounds=bounds, markers=set(coords), diagonal=diagonal)
    return None


def _inside(r, c, bounds):
    return bounds[0] <= r <= bounds[2] and bounds[1] <= c <= bounds[3]


def _parse(grid):
    """Find a unique locally isolated vertical table and recover the whole scene."""
    h, w = grid.shape
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert np.sum(counts == counts[bg]) == 1, "ambiguous background"
    candidates = []
    for c in range(w-1):
        rows = np.flatnonzero(grid[:, c] != bg)
        runs = np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1)
        for run in runs:
            if not 2 <= len(run) <= 8:
                continue
            r0, r1 = int(run[0]), int(run[-1])
            cues = [int(v) for v in grid[r0:r1+1, c]]
            if len(set(cues)) != len(cues):
                continue
            table = (r0, c, r1, c+1)
            # A local one-cell moat, with no requirement on entire columns.
            if any(grid[r, cc] != bg
                   for r in range(max(0, r0-1), min(h, r1+2))
                   for cc in range(max(0, c-1), min(w, c+3))
                   if not _inside(r, cc, table)):
                continue
            scene = grid.copy()
            scene[r0:r1+1, c:c+2] = bg
            objects = {}
            valid = True
            for color in np.unique(scene):
                color = int(color)
                if color == bg:
                    continue
                coords = {tuple(map(int, rc)) for rc in np.argwhere(scene == color)}
                obj = _recognize(coords)
                if obj is None or _bounds_overlap(obj["bounds"], table):
                    valid = False
                    break
                obj["cue"] = color
                objects[color] = obj
            if not valid or not objects:
                continue
            fills = [int(v) for v in grid[r0:r1+1, c+1]]
            active, unused = [], []
            for rank, (cue, fill) in enumerate(zip(cues, fills)):
                if cue in objects:
                    active.append(dict(objects[cue], fill=fill, rank=rank))
                else:
                    unused.append(rank)
            unlisted = [o for cue, o in objects.items() if cue not in cues]
            if not active or any(o["kind"] != "normal" for o in unlisted):
                continue
            candidates.append(dict(background=bg, table=table, cues=cues, fills=fills,
                                   specs=active, unused=unused, unlisted=unlisted))
    assert len(candidates) == 1, "table or scene is ambiguous"
    return candidates[0]


def _paint(obj, r, c, bg, eraser):
    if obj["kind"] == "normal" and (r, c) in obj["markers"]:
        return obj["cue"]
    return bg if obj["fill"] == eraser else obj["fill"]


def _predict_parsed(grid, parsed, eraser):
    """Independent predictor: choose the first covering row at each cell.

    This never calls the generator's reverse-order rectangle compositor.
    None means a literal-swatch hypothesis, used while learning from training.
    """
    result = grid.copy()
    bg = parsed["background"]
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            for obj in parsed["specs"]:
                if _inside(r, c, obj["bounds"]):
                    result[r, c] = _paint(obj, r, c, bg, eraser)
                    break
    return result


def _learn(train_pairs):
    hypotheses = [None] + list(range(10))
    for pair in train_pairs:
        parsed = _parse(pair["input"])
        hypotheses = [e for e in hypotheses if np.array_equal(
            _predict_parsed(pair["input"], parsed, e), pair["output"])]
        assert hypotheses, "training pairs contradict the source rule"
    return hypotheses


def predict(train_pairs, test_input):
    """Only training pairs and a test input are available to this solver."""
    parsed = _parse(test_input)
    hypotheses = _learn(train_pairs)
    predictions = [_predict_parsed(test_input, parsed, e) for e in hypotheses]
    assert all(np.array_equal(predictions[0], p) for p in predictions[1:]), \
        "erasure convention needed by this input was not taught"
    return predictions[0]


def _compose(grid, specs, bg, eraser):
    output = grid.copy()
    for obj in reversed(specs):
        r0, c0, r1, c1 = obj["bounds"]
        output[r0:r1+1, c0:c1+1] = bg if obj["fill"] == eraser else obj["fill"]
        if obj["kind"] == "normal":
            for r, c in obj["markers"]:
                output[r, c] = obj["cue"]
    return output


def _witnesses(grid, parsed, eraser):
    """Visible semantic evidence, computed from reconstructed input objects."""
    bg, specs = parsed["background"], parsed["specs"]
    found = set()
    if parsed["unused"]:
        found.add("unused")
    visible_blocks = {o["cue"]: 0 for o in specs if o["kind"] == "block"}
    visible_unlisted = set()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            stack = [o for o in specs if _inside(r, c, o["bounds"])]
            if not stack:
                for obj in parsed["unlisted"]:
                    if _inside(r, c, obj["bounds"]) and (r, c) not in obj["markers"]:
                        if grid[r, c] == bg:
                            visible_unlisted.add(obj["cue"])
                continue
            top = stack[0]
            value = _paint(top, r, c, bg, eraser)
            if "occlusion" not in found and any(o["kind"] == "normal" and
                    (r, c) in o["markers"] and value != o["cue"] for o in specs):
                found.add("occlusion")
            if top["kind"] == "block" and value != top["cue"]:
                visible_blocks[top["cue"]] += 1
            if top["kind"] == "normal":
                if (r, c) in top["markers"]:
                    if top["cue"] != (bg if top["fill"] == eraser else top["fill"]):
                        found.add("retention")
                elif value != bg and value != grid[r, c]:
                    found.add("ordinary")
            if len(stack) >= 2:
                below = _paint(stack[1], r, c, bg, eraser)
                if value != below:
                    found.add("priority")
                    if value == bg and (r, c) not in top["markers"]:
                        found.add("clearing")
                        if top["fill"] == eraser and eraser != bg:
                            found.add("erase")
            # A visible changed swatch distinguishes erasure from literal paint.
            if top["fill"] == eraser and eraser != bg and value == bg:
                if top["kind"] == "block" or (r, c) not in top["markers"]:
                    found.add("erasure_role")
    if any(n == 4 for n in visible_blocks.values()):
        found.add("block")
    for obj in parsed["unlisted"]:
        if obj["cue"] in visible_unlisted and any(
                not any(_inside(r, c, o["bounds"]) for o in specs)
                for r, c in obj["markers"]):
            found.add("unlisted")
    return found


def _sample_bounds(rng, h, w, shape=None):
    shape = shape or str(rng.choice(["compact", "square", "horizontal", "vertical", "free"]))
    if (shape == "horizontal" and w < 4) or (shape == "vertical" and h < 4):
        shape = "compact"
    if shape == "square":
        hh = ww = int(rng.integers(3, min(h, w, 10)+1))
    elif shape == "compact":
        hh, ww = int(rng.integers(3, min(h, 7)+1)), int(rng.integers(3, min(w, 7)+1))
    elif shape == "horizontal":
        hh, ww = int(rng.integers(2, min(h, 4)+1)), int(rng.integers(4, w+1))
    elif shape == "vertical":
        hh, ww = int(rng.integers(4, h+1)), int(rng.integers(2, min(w, 4)+1))
    else:
        hh, ww = int(rng.integers(3, h+1)), int(rng.integers(3, w+1))
    r = int(rng.integers(0, h-hh+1))
    c = int(rng.integers(0, w-ww+1))
    return (r, c, r+hh-1, c+ww-1)


def _contains(a, b):
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _intervals(rng, extent, count):
    if extent < 3*count+1:
        return None
    sizes = [2] * count
    slack = extent - (3*count+1)
    for i in rng.permutation(count):
        if slack and rng.random() < .25:
            sizes[int(i)] += 1
            slack -= 1
    gaps = rng.multinomial(slack, np.full(count+1, 1/(count+1)))
    position, intervals = 1+int(gaps[0]), []
    for i, size in enumerate(sizes):
        intervals.append((position, position+size-1))
        position += size+1+int(gaps[i+1])
    return intervals


def _crossing_normals(rng, h, w, count):
    nv = 2 if count == 5 else int(rng.integers(1, count))
    columns, rows = _intervals(rng, w, nv), _intervals(rng, h, count-nv)
    if columns is None or rows is None:
        return None
    objects = []
    for c0, c1 in columns:
        r0 = int(rng.integers(rows[0][0]))
        r1 = int(rng.integers(rows[-1][1]+1, h))
        objects.append((r0, c0, r1, c1))
    for r0, r1 in rows:
        c0 = int(rng.integers(columns[0][0]))
        c1 = int(rng.integers(columns[-1][1]+1, w))
        objects.append((r0, c0, r1, c1))
    return [dict(kind="normal", bounds=b, markers=_marker_cells(b, d),
                 diagonal=d, inert=False)
            for b, d in zip(objects, map(int, rng.integers(2, size=count)))]


def _geometry(rng, h, w, normal_count, block_count, unlisted_count, family):
    objects, used = [], set()
    if family == "crisscross" and normal_count >= 2:
        objects = _crossing_normals(rng, h, w, normal_count)
        if objects is None:
            return None
        for obj in objects:
            if used & obj["markers"]:
                return None
            used.update(obj["markers"])
    for index in range(len(objects), normal_count + block_count + unlisted_count):
        block = normal_count <= index < normal_count + block_count
        inert = index >= normal_count + block_count
        placed = False
        for _ in range(100):
            if block:
                r, c = int(rng.integers(h-1)), int(rng.integers(w-1))
                bounds = (r, c, r+1, c+1)
            else:
                bounds = _sample_bounds(rng, h, w)
                if family == "nested" and 0 < index < normal_count and rng.random() < .7:
                    parent = objects[int(rng.integers(len(objects)))]["bounds"]
                    ph, pw = parent[2]-parent[0]+1, parent[3]-parent[1]+1
                    if ph >= 5 and pw >= 5:
                        bounds = _sample_bounds(rng, ph-2, pw-2)
                        bounds = (bounds[0]+parent[0]+1, bounds[1]+parent[1]+1,
                                  bounds[2]+parent[0]+1, bounds[3]+parent[1]+1)
            previous = [o["bounds"] for o in objects[:normal_count]]
            if not inert and not block and previous:
                intersects = [_bounds_overlap(bounds, b) for b in previous]
                if family == "disjoint" and any(intersects):
                    continue
                if family == "chain" and (not intersects[-1] or any(intersects[:-1])):
                    continue
                if family == "nested" and not any(
                        _contains(b, bounds) or _contains(bounds, b) for b in previous):
                    continue
            diagonal = int(rng.integers(2))
            if block:
                cells = {(r, c) for r in range(bounds[0], bounds[2]+1)
                         for c in range(bounds[1], bounds[3]+1)}
            else:
                cells = _marker_cells(bounds, diagonal)
            if cells & used or any(bounds == o["bounds"] for o in objects):
                continue
            if not block and _recognize(cells) is None:
                continue
            objects.append(dict(kind="block" if block else "normal", bounds=bounds,
                                markers=cells, diagonal=diagonal, inert=inert))
            used.update(cells)
            placed = True
            break
        if not placed:
            return None
    return objects


def _table_locations(h, w, length, objects):
    # The table may sit anywhere in free space, including an internal pocket.
    blocked = np.zeros((h, w), dtype=bool)
    for obj in objects:
        r0, c0, r1, c1 = obj["bounds"]
        blocked[max(0, r0-1):min(h, r1+2), max(0, c0-1):min(w, c1+2)] = True
    return [(r, c) for r in range(h-length+1) for c in range(w-1)
            if not blocked[r:r+length, c:c+2].any()]


def _pair_counts(rng, params, required):
    simple = params["regime"] == "gentle"
    normals = int(rng.integers(1, 4 if simple else 7))
    if "priority" in required or "erase" in required or "clearing" in required:
        normals = max(2, normals)
    if params["regime"] == "steady":
        normals = max(1, min(6, params["layers"] + int(rng.integers(-1, 2))))
        if required & {"priority", "erase", "clearing"}:
            normals = max(2, normals)
    blocks = int(rng.choice([0, 0, 0, 1, 1, 2])) if params["blocks"] else 0
    if "block" in required:
        blocks = max(1, blocks)
    if not params["incomplete"]:
        normals = max(normals, 2-blocks)
    normals = min(normals, 7-blocks)
    unused = int(rng.choice([0, 0, 0, 1, 1, 2, 3])) if params["incomplete"] else 0
    if "unused" in required:
        unused = max(1, unused)
    unused = min(unused, 8-normals-blocks)
    unused = max(unused, 2-normals-blocks)
    length = normals+blocks+unused
    unlisted = int(rng.choice([0, 0, 0, 1, 1, 2, 3])) if params["incomplete"] else 0
    if "unlisted" in required:
        unlisted = max(1, unlisted)
    unlisted = min(unlisted, 9-length)
    return normals, blocks, unused, unlisted


def _pair_size(rng, params, split):
    regime = params["regime"]
    if regime == "steady":
        h = int(np.clip(params["height"] + rng.integers(-2, 3), 8, 30))
        w = int(np.clip(params["width"] + rng.integers(-2, 3), 8, 30))
        return h, w
    if regime == "gentle":
        lo, hi = 8, 19
    elif regime == "progressive":
        lo, hi = (8, 21) if split == "train" else (12, 31)
    else:
        lo, hi = 8, 31
    return int(rng.integers(lo, hi)), int(rng.integers(lo, hi))


def _candidate(rng, params, split, required):
    bg = int(rng.integers(10))
    eraser = params["eraser"]
    if "erase" in required and bg == eraser:
        bg = (bg + int(rng.integers(1, 10))) % 10
    normals, blocks, unused, unlisted = _pair_counts(rng, params, required)
    length = normals+blocks+unused
    h, w = _pair_size(rng, params, split)
    if length > h:
        return None
    families = ["free", "free", "disjoint", "nested", "chain", "crisscross", "mixed"]
    if params["regime"] == "gentle":
        families = ["disjoint", "disjoint", "free", "nested", "mixed"]
    family = str(rng.choice(families))
    if family == "crisscross" and rng.random() < .5:
        normals = 5
        unused = min(unused, 8-normals-blocks)
        length = normals+blocks+unused
        unlisted = min(unlisted, 9-length)
    if required & {"priority", "erase", "clearing"} and family == "disjoint":
        family = str(rng.choice(["free", "nested", "chain", "mixed"]))
    objects = _geometry(rng, h, w, normals, blocks, unlisted, family)
    if objects is None:
        return None
    locations = _table_locations(h, w, length, objects)
    if not locations:
        return None
    tr, tc = locations[int(rng.integers(len(locations)))]
    colors = list(map(int, rng.permutation([c for c in range(10) if c != bg])))
    active = objects[:normals+blocks]
    inert = objects[normals+blocks:]
    for obj, cue in zip(active+inert, colors):
        obj["cue"] = cue
    spare = colors[len(active)+len(inert):len(active)+len(inert)+unused]
    rows = [dict(o) for o in active] + [dict(cue=c) for c in spare]
    rows = [rows[int(i)] for i in rng.permutation(len(rows))]
    # Fill choices have no fixed rank, orientation, or block relationship.
    paint_colors = params.get("paint_colors", [c for c in range(10) if params["erasure"] or c != eraser])
    for obj in rows:
        obj["fill"] = int(rng.choice(paint_colors))
        draw = rng.random()
        if draw < .12 and obj["cue"] in paint_colors:
            obj["fill"] = obj["cue"]
        elif draw < .24 and bg in paint_colors:
            obj["fill"] = bg
    if rng.random() < .4 and len(rows) >= 2:
        a, b = map(int, rng.choice(len(rows), 2, replace=False))
        rows[b]["fill"] = rows[a]["fill"]
    if required & {"erase", "clearing"}:
        overlapping = [(i, j) for i, a in enumerate(rows) if "bounds" in a
                       for j, b in enumerate(rows) if j > i and "bounds" in b
                       and _bounds_overlap(a["bounds"], b["bounds"])]
        if not overlapping:
            return None
        i, j = overlapping[int(rng.integers(len(overlapping)))]
        rows[i]["fill"] = eraser if "erase" in required else bg
        rows[j]["fill"] = int(rng.choice([c for c in paint_colors if c != bg and c != eraser]))
    grid = np.full((h, w), bg, dtype=np.int8)
    for obj in objects:
        for r, c in obj["markers"]:
            grid[r, c] = obj["cue"]
    for i, obj in enumerate(rows):
        grid[tr+i, tc:tc+2] = (obj["cue"], obj["fill"])
    specs = [obj for obj in rows if "bounds" in obj]
    output = _compose(grid, specs, bg, eraser)
    if np.array_equal(grid, output):
        return None
    try:
        parsed = _parse(grid)
    except AssertionError:
        return None
    if parsed["background"] != bg or not required <= _witnesses(grid, parsed, eraser):
        return None
    if not np.array_equal(output, _predict_parsed(grid, parsed, eraser)):
        return None
    return {"input": grid, "output": output}


def _fallback(rng, params, required):
    """A guaranteed readable demonstration after bounded spatial rejection.

    Used only as a last resort; colors, marker diagonals and ranks still vary.
    The upper crossing layer clears a contrasting lower layer. A separate block,
    unused row and unlisted pair demonstrate every optional obligation.
    """
    eraser = params["eraser"]
    allowed = params.get("paint_colors", list(range(10)))
    paint = int(rng.choice([c for c in allowed if c != eraser]))
    bg = int(rng.choice([c for c in range(10) if c not in (eraser, paint)]))
    colors = list(map(int, rng.permutation([c for c in range(10) if c != bg])))
    if colors[2] == paint:
        colors[2], colors[3] = colors[3], colors[2]
    bounds = [(3, 5, 9, 15), (6, 8, 14, 19)]
    if params["blocks"]:
        bounds.append((17, 14, 18, 15))
    objects = []
    for i, b in enumerate(bounds):
        block = i == 2
        cells = ({(r, c) for r in range(b[0], b[2]+1) for c in range(b[1], b[3]+1)}
                 if block else _marker_cells(b, int(rng.integers(2))))
        objects.append(dict(kind="block" if block else "normal", bounds=b,
                            markers=cells, cue=colors[i]))
    rows = objects + ([dict(cue=colors[3])] if params["incomplete"] else [])
    rows = [rows[int(i)] for i in rng.permutation(len(rows))]
    for obj in rows:
        obj["fill"] = paint
    normals = [o for o in rows if o.get("kind") == "normal"]
    normals[0]["fill"] = eraser if params["erasure"] else (bg if bg in allowed else paint)
    # This diagonal puts a lower marker inside the crossing upper rectangle.
    # The other diagonal can leave all six lower markers outside the overlap.
    normals[1]["markers"] = _marker_cells(normals[1]["bounds"], 0)
    grid = np.full((23, 24), bg, dtype=np.int8)
    for obj in objects:
        for r, c in obj["markers"]:
            grid[r, c] = obj["cue"]
    if params["incomplete"]:
        for r, c in _marker_cells((17, 5, 21, 10), int(rng.integers(2))):
            grid[r, c] = colors[4]
    for i, obj in enumerate(rows):
        grid[7+i, 1:3] = obj["cue"], obj["fill"]
    output = _compose(grid, [o for o in rows if "bounds" in o], bg, eraser)
    parsed = _parse(grid)
    assert required <= _witnesses(grid, parsed, eraser)
    return dict(input=grid, output=output)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    params = dict(eraser=int(rng.integers(10)), erasure=bool(rng.random() < .72),
                  blocks=bool(rng.random() < .72), incomplete=bool(rng.random() < .76),
                  regime=str(rng.choice(["gentle", "steady", "progressive", "mixed"])),
                  height=int(rng.integers(10, 29)), width=int(rng.integers(10, 29)),
                  layers=int(rng.integers(1, 6)))
    # Obligations go to independently chosen training slots, including the last.
    requirements = [set() for _ in range(num_train)]
    core = {"ordinary", "retention", "priority", "occlusion"}
    if params["erasure"]:
        core.add("erase")
    else:
        core.add("clearing")
    requirements[int(rng.integers(num_train))].update(core)
    if params["blocks"]:
        requirements[int(rng.integers(num_train))].add("block")
    if params["incomplete"]:
        requirements[int(rng.integers(num_train))].update({"unused", "unlisted"})
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        if split == "test":
            hypotheses = _learn(puzzle["train"])
            # Untaught hypothetical erasure colors cannot change a test answer.
            # With a demonstrated erasure role the hypothesis is already unique.
            if len(hypotheses) > 1:
                params["paint_colors"] = [c for c in range(10) if c not in hypotheses]
        for index in range(count):
            required = requirements[index] if split == "train" else set()
            pair = None
            for _ in range(160):
                candidate = _candidate(rng, params, split, required)
                if candidate is None:
                    continue
                signature = _grid_signature(candidate["input"])
                if signature not in seen and signature not in _SOURCE_INPUT_SIGNATURES:
                    pair = candidate
                    break
            if pair is None:
                pair = _fallback(rng, params, required)
                # A unique canvas margin color arrangement is not needed: there
                # are already many palettes. Retry boundedly if one repeats.
                for _ in range(20):
                    if _grid_signature(pair["input"]) not in seen:
                        break
                    pair = _fallback(rng, params, required)
            seen.add(_grid_signature(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, parsed_pairs = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair["input"].shape == pair["output"].shape
            assert not np.array_equal(pair["input"], pair["output"])
            key = _grid_signature(pair["input"])
            assert key not in seen and key not in _SOURCE_INPUT_SIGNATURES
            seen.add(key)
            parsed_pairs[split].append(_parse(pair["input"]))
    hypotheses = _learn(puzzle["train"])
    eraser = hypotheses[0]
    training_witnesses = set()
    for pair, parsed in zip(puzzle["train"], parsed_pairs["train"]):
        training_witnesses.update(_witnesses(pair["input"], parsed, eraser))
    assert {"ordinary", "retention", "priority", "occlusion"} <= training_witnesses
    assert any(not np.array_equal(pair["output"], _predict_parsed(pair["input"],
        dict(parsed, specs=list(reversed(parsed["specs"]))), eraser))
        for pair, parsed in zip(puzzle["train"], parsed_pairs["train"])), "priority direction is not visible"
    for pair, parsed in zip(puzzle["test"], parsed_pairs["test"]):
        # All training-consistent erasure hypotheses must yield the same answer.
        for hypothesis in hypotheses:
            expected = _predict_parsed(pair["input"], parsed, hypothesis)
            assert np.array_equal(expected, pair["output"]), "untaught or incorrect test output"
        needed = set()
        if any(o["kind"] == "block" for o in parsed["specs"]):
            needed.add("block")
        if parsed["unused"]:
            needed.add("unused")
        if parsed["unlisted"]:
            needed.add("unlisted")
        witness = _witnesses(pair["input"], parsed, eraser)
        if "clearing" in witness:
            needed.add("clearing")
        if "erasure_role" in witness:
            needed.add("erasure_role")
        if "occlusion" in witness:
            needed.add("occlusion")
        assert needed <= training_witnesses, "test case lacks visible training evidence"


def measure_pair(pair):
    """Review measurements recovered from public arrays, never generation state."""
    grid, output = pair["input"], pair["output"]
    p = _parse(grid)
    specs, bg = p["specs"], p["background"]
    normals = [o for o in specs if o["kind"] == "normal"]
    blocks = [o for o in specs if o["kind"] == "block"]
    edges = [(i, j) for i, a in enumerate(normals) for j, b in enumerate(normals)
             if i < j and _bounds_overlap(a["bounds"], b["bounds"])]
    degrees = [sum(i in edge for edge in edges) for i in range(len(normals))]
    reached, components = set(), 0
    for start in range(len(normals)):
        if start in reached:
            continue
        components += 1
        todo = [start]
        while todo:
            i = todo.pop()
            if i in reached:
                continue
            reached.add(i)
            todo += [b if a == i else a for a, b in edges if i in (a, b)]
    dims = [(o["bounds"][2]-o["bounds"][0]+1, o["bounds"][3]-o["bounds"][1]+1)
            for o in normals]
    common = bool(normals) and max(o["bounds"][0] for o in normals) <= min(o["bounds"][2] for o in normals) \
        and max(o["bounds"][1] for o in normals) <= min(o["bounds"][3] for o in normals)
    vertical = [i for i, (h, w) in enumerate(dims) if h >= 2*w]
    horizontal = [i for i, (h, w) in enumerate(dims) if w >= 2*h]
    cross = bool(vertical and horizontal) and all(
        _bounds_overlap(normals[i]["bounds"], normals[j]["bounds"])
        for i in vertical for j in horizontal)
    covered_markers = sum(output[r, c] != o["cue"] for o in normals for r, c in o["markers"])
    unlisted_covered = sum(output[r, c] != o["cue"] for o in p["unlisted"] for r, c in o["markers"])
    return dict(height=grid.shape[0], width=grid.shape[1], background=bg,
                table_position=list(p["table"][:2]), table_rows=len(p["cues"]),
                table_interior=p["table"][1] > 0 and p["table"][3] < grid.shape[1]-1,
                table_columns_reused=any(grid[r, c] != bg for r in range(grid.shape[0])
                    for c in (p["table"][1], p["table"][3]) if not _inside(r, c, p["table"])),
                active=len(specs), normals=len(normals), blocks=len(blocks),
                unused=len(p["unused"]), unlisted=len(p["unlisted"]),
                normal_sizes=[list(size) for size in dims],
                unlisted_bounds=[list(o["bounds"]) for o in p["unlisted"]],
                diagonals=sorted({o["diagonal"] for o in normals}),
                thin=sum(min(h, w) == 2 for h, w in dims),
                thin_horizontal=sum(h == 2 for h, w in dims),
                thin_vertical=sum(w == 2 for h, w in dims),
                squares=sum(h == w for h, w in dims),
                overlap_edges=len(edges), components=components, common_intersection=common,
                disjoint=len(normals) >= 2 and not edges,
                chain=len(normals) >= 3 and components == 1 and max(degrees) <= 2 and len(edges) == len(normals)-1,
                mixed_isolated=bool(edges) and 0 in degrees,
                disjoint_groups=bool(edges) and components > 1,
                partial_overlap=any(not _contains(normals[i]["bounds"], normals[j]["bounds"]) and
                    not _contains(normals[j]["bounds"], normals[i]["bounds"]) for i, j in edges),
                nested=any(_contains(normals[i]["bounds"], normals[j]["bounds"]) or
                           _contains(normals[j]["bounds"], normals[i]["bounds"]) for i, j in edges),
                offset_nested=any((_contains(normals[i]["bounds"], normals[j]["bounds"]) or
                    _contains(normals[j]["bounds"], normals[i]["bounds"])) and
                    (normals[i]["bounds"][0]+normals[i]["bounds"][2], normals[i]["bounds"][1]+normals[i]["bounds"][3]) !=
                    (normals[j]["bounds"][0]+normals[j]["bounds"][2], normals[j]["bounds"][1]+normals[j]["bounds"][3])
                    for i, j in edges),
                crisscross=cross,
                two_vertical_three_horizontal=cross and len(vertical) == 2 and len(horizontal) == 3 and not blocks,
                edge_objects=sum(o["bounds"][0] == 0 or o["bounds"][1] == 0 or
                                 o["bounds"][2] == grid.shape[0]-1 or o["bounds"][3] == grid.shape[1]-1 for o in specs),
                block_ranks=[o["rank"] for o in blocks],
                background_ranks=[o["rank"] for o in specs if o["fill"] == bg],
                horizontal_above_vertical=any(normals[i]["rank"] < normals[j]["rank"] and
                    _bounds_overlap(normals[i]["bounds"], normals[j]["bounds"]) for i in horizontal for j in vertical),
                small_behind_large=any(normals[i]["rank"] > normals[j]["rank"] and
                    dims[i][0]*dims[i][1] < dims[j][0]*dims[j][1] and
                    _bounds_overlap(normals[i]["bounds"], normals[j]["bounds"])
                    for i in range(len(normals)) for j in range(len(normals)) if i != j),
                self_fills=sum(o["cue"] == o["fill"] for o in specs),
                other_cue_fills=sum(o["cue"] != o["fill"] and o["fill"] in p["cues"] for o in specs),
                shared_fills=len(p["fills"])-len(set(p["fills"])),
                covered_markers=int(covered_markers), unlisted_covered=int(unlisted_covered))


def self_test() -> None:
    validate_generated_puzzle(generate_puzzle(1288, 2, 1))
    for eraser in range(10):
        for erasure in (False, True):
            for blocks in (False, True):
                for incomplete in (False, True):
                    params = dict(eraser=eraser, erasure=erasure, blocks=blocks, incomplete=incomplete)
                    required = {"ordinary", "retention", "priority", "occlusion",
                                "erase" if erasure else "clearing"}
                    if blocks:
                        required.add("block")
                    if incomplete:
                        required.update({"unused", "unlisted"})
                    pair = _fallback(np.random.default_rng(83+eraser), params, required)
                    parsed = _parse(pair["input"])
                    assert required <= _witnesses(pair["input"], parsed, eraser)
                    assert np.array_equal(_predict_parsed(pair["input"], parsed, eraser), pair["output"])
    a = generate_puzzle(314159, 7, 3)
    b = generate_puzzle(314159, 7, 3)
    assert all(_pair_signature(x) == _pair_signature(y) for s in a for x, y in zip(a[s], b[s]))
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(271828, *counts)
        validate_generated_puzzle(p)
        for split, count in zip(("train", "test"), counts):
            assert count is None or len(p[split]) == count
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    records, train_counts, test_counts, signatures = [], set(), set(), set()
    varying = dict(background=0, table_position=0, diagonals=0, active=0, table_rows=0)
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        current = []
        # Freeze the inferred convention using training only before testing.
        inferred = _learn(p["train"])
        for split in ("train", "test"):
            for pair in p[split]:
                key = _grid_signature(pair["input"])
                assert key not in signatures
                signatures.add(key)
                parsed = _parse(pair["input"])
                assert all(np.array_equal(_predict_parsed(pair["input"], parsed, e), pair["output"])
                           for e in inferred)
                m = measure_pair(pair)
                m["split"] = split
                records.append(m)
                current.append(m)
        for name in varying:
            varying[name] += len({str(m[name]) for m in current}) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert all(n >= 25 for n in varying.values()), varying
    assert len({(m["height"], m["width"]) for m in records}) >= 75
    assert {m["background"] for m in records} == set(range(10))
    for name in ("thin_horizontal", "thin_vertical", "squares", "unused", "unlisted", "edge_objects",
                 "disjoint", "chain", "nested", "mixed_isolated", "crisscross",
                 "horizontal_above_vertical", "small_behind_large", "covered_markers", "unlisted_covered",
                 "self_fills", "other_cue_fills", "shared_fills"):
        assert sum(bool(m[name]) for m in records) >= 3, name
    assert {m["blocks"] for m in records} >= {0, 1, 2}
    assert {m["unused"] for m in records} >= {0, 1, 2, 3}
    assert {m["unlisted"] for m in records} >= {0, 1, 2, 3}
    assert len({r for m in records for r in m["block_ranks"]}) >= 4
    assert len({r for m in records for r in m["background_ranks"]}) >= 4
    assert any(m["height"] <= 10 and m["width"] <= 15 for m in records)
    assert any(max(m["height"], m["width"]) == 30 for m in records)
    assert any(len(m["diagonals"]) == 2 and m["thin"] and m["unused"] and m["overlap_edges"] for m in records)
    for split in ("train", "test"):
        subset = [m for m in records if m["split"] == split]
        for name in ("thin_horizontal", "thin_vertical", "unused", "unlisted", "table_interior",
                     "table_columns_reused", "offset_nested", "partial_overlap", "disjoint_groups"):
            assert any(m[name] for m in subset), (split, name)
        assert {m["blocks"] for m in subset} >= {0, 1, 2}
        assert {m["unused"] for m in subset} >= {0, 1, 2, 3}
        assert {m["unlisted"] for m in subset} >= {0, 1, 2, 3}
        assert min(m["height"]*m["width"] for m in subset) <= 150
        assert max(max(m["height"], m["width"]) for m in subset) == 30
    assert any(m["two_vertical_three_horizontal"] for m in records)
    _source_regressions()


# Frozen source regressions: training alone identifies the erasure convention.
_SOURCE_REGRESSION = (
    (
        ('0002200000', '0002000000', '0000110000', '0000100000', '0000001000', '0000011000', '0000000200', '0000002200', '1000000000', '2300000000'),
        ('0002233300', '0002333300', '0003110300', '0003100300', '0003001300', '0003011300', '0003333200', '0003332200', '1000000000', '2300000000'),
    ),
    (
        ('0000000044', '0000000004', '0000011000', '0000010000', '0000000000', '0000000000', '0000000100', '0130401100', '0450440000', '0000000000'),
        ('0000555544', '0000555554', '0000511355', '0000513355', '0000533355', '0000533355', '0000533155', '0130431155', '0450445555', '0000000000'),
    ),
    (
        ('33553333333333333', '33533335333333333', '33333355333333333', '33332233333333333', '33332344333333333', '33333343311332333', '33333333331322333', '33333333343333333', '33163333443333333', '33293333333333333', '33403331333333333', '33893331133333333'),
        ('33553333333333333', '33533335333333333', '33333355333333333', '33332299999999333', '33332999999999333', '33339996611992333', '33339996661922333', '33333336666333333', '33163336666333333', '33293336666333333', '33403331666333333', '33893331166333333'),
    ),
    (
        ('000000000000000000', '000003300000000000', '000003000000000000', '000000000000000000', '000000003000000000', '000000033000000000', '000000000000440000', '000000000000400004', '000000660000000044', '000000600000000000', '000000000000000000', '330000000060007700', '670000000660007700', '440000000000000000', '760000000000000000'),
        ('000000000000000000', '000003333000000000', '000003333000000000', '000003333000000000', '000003333000000000', '000003333000000000', '000000000000444444', '000000000000444444', '000000667770444444', '000000677770000000', '000000777770000000', '330000777760006600', '670000777660006600', '440000000000000000', '760000000000000000'),
    ),
    (
        ('9999449229999', '9999499299999', '9999999999999', '9119999999991', '9199999999911', '9999999999999', '9999559999995', '9999599999955', '9999999999999', '9999339999993', '9999399999933', '9999999999999', '9999999999999', '9999999999999', '9229999999999', '9599999999999', '9399999999999', '9149949929999', '9469449229999'),
        ('9999449229999', '9999469229999', '9999669229999', '9114444224441', '9144444224411', '9999669229999', '9999559229995', '9999599229955', '9999669229999', '9999339229993', '9999399229933', '9999669229999', '9999669229999', '9999669229999', '9229669229999', '9599669229999', '9399669229999', '9149649229999', '9469449229999'),
    ),
    (
        ('1111111111111111111', '1881117711111111111', '1881117111111111111', '1111111111111111111', '1111116611111111111', '1111116133111111111', '1111111131111612211', '1134111111316611211', '1168111113311111111', '1171111111111211111', '1182111171111221111', '1124111771111111111', '1111111111111111111'),
        ('1111111111111111111', '1221117711111111111', '1221117111111111111', '1111111111111111111', '1111116688888811111', '1111116833488811111', '1111118834488642211', '1134118844386644211', '1168111143311444411', '1171111111111244411', '1182111171111224411', '1124111771111111111', '1111111111111111111'),
    ),
)


def _source_regressions():
    pairs = [dict(input=np.array([[int(c) for c in row] for row in inp], dtype=np.int8),
                  output=np.array([[int(c) for c in row] for row in out], dtype=np.int8))
             for inp, out in _SOURCE_REGRESSION]
    assert _learn(pairs[:4]) == [0]
    for pair in pairs:
        parsed = _parse(pair["input"])
        assert np.array_equal(_predict_parsed(pair["input"], parsed, 0), pair["output"])
    third = pairs[2]
    literal = _predict_parsed(third["input"], _parse(third["input"]), None)
    assert set(map(tuple, np.argwhere(literal != third["output"]))) == {(7, 6), (8, 6)}
    assert third["output"][7, 6] == third["output"][8, 6] == 3
    assert measure_pair(pairs[4])["thin"] == 5
    assert measure_pair(third)["unused"] == measure_pair(third)["unlisted"] == 1
    # Test inputs only enter prediction; their labels are compared afterward.
    for pair in pairs[4:]:
        prediction = predict(pairs[:4], pair["input"].copy())
        assert np.array_equal(prediction, pair["output"])
    # Thin 2x3 solid blocks are not two separate L fragments.
    assert _recognize({(r, c) for r in range(2) for c in range(3)}) is None
    assert _recognize(_marker_cells((0, 0, 1, 3), 0)) is not None
    assert _recognize(_marker_cells((0, 0, 3, 1), 1)) is not None
