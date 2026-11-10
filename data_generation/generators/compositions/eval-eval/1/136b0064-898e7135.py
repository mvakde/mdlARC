"""Keyhole loom: fit ported macro-pieces, then execute a miniature mold program.

The decoder receives only the input. Colors 1 and 2 are entry and exit ports;
all other roles are recovered locally. Generation uses no parent code or data.
"""
from __future__ import annotations

from collections import Counter
from math import isqrt
import numpy as np

ENTRY, EXIT = 1, 2
_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    arr = np.asarray(value)
    if arr.ndim != 2 or not all(1 <= d <= 30 for d in arr.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(arr.dtype, np.integer) or np.any((arr < 0) | (arr > 9)):
        raise ValueError("grid must contain integer colors 0..9")
    return arr.astype(np.int8, copy=True)


def _key(arr):
    arr = np.asarray(arr, dtype=np.int8)
    return arr.shape, arr.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, pending = [], [start]
        while pending:
            r, c = pending.pop()
            cells.append((r, c))
            for dr, dc in _NEIGHBORS:
                nxt = r + dr, c + dc
                if nxt in remaining:
                    remaining.remove(nxt)
                    pending.append(nxt)
        groups.append(tuple(sorted(cells)))
    return groups


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells) + 1,
            min(c for r, c in cells), max(c for r, c in cells) + 1)


def _normal(cells):
    r0, _, c0, _ = _bounds(cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _size(shape):
    return max(r for r, c in shape) + 1, max(c for r, c in shape) + 1


def _poses(shape, entry, exit_point, reflections=True):
    """Transform the whole decorated object, including both ports, under D4."""
    result = set()
    for mirror in range(2 if reflections else 1):
        for turns in range(4):
            def transform(point):
                r, c = point
                if mirror:
                    c = -c
                for _ in range(turns):
                    r, c = c, -r
                return r, c
            cells = [transform(p) for p in shape]
            r0, _, c0, _ = _bounds(cells)
            def shift(p):
                r, c = transform(p)
                return r-r0, c-c0
            result.add((tuple(sorted((r-r0, c-c0) for r, c in cells)),
                        shift(entry), shift(exit_point)))
    return tuple(sorted(result))


def _canonical(shape):
    return min(p[0] for p in _poses(shape, shape[0], shape[-1]))


def _safe_poses(shape, entry, exit_point):
    poses = _poses(shape, entry, exit_point)
    endings = {}
    for mask, a, b in poses:
        endings.setdefault((mask, a), set()).add(b)
    return tuple(p for p in poses if len(endings[p[:2]]) == 1)


def _positive_poses(shape, entry, exit_point):
    return tuple(p for p in _safe_poses(shape, entry, exit_point)
                 if p[1][0] == 0 and p[2][0] == _size(p[0])[0]-1
                 and _size(p[0])[0] <= 3 and _size(p[0])[1] <= 4)


def _find_panels(inp, bg):
    programs, panes = [], []
    for structural in range(10):
        if structural in (bg, ENTRY, EXIT):
            continue
        for comp in _components(inp == structural):
            r0, r1, c0, c1 = _bounds(comp)
            crop = inp[r0:r1, c0:c1]
            if min(crop.shape) < 3:
                continue
            if not (np.all(crop[0] == structural) and np.all(crop[-1] == structural)
                    and np.all(crop[:, 0] == structural) and np.all(crop[:, -1] == structural)):
                continue
            if not np.all(np.isin(crop, (bg, structural, ENTRY, EXIT))):
                continue
            interior = crop[1:-1, 1:-1]
            mark = np.argwhere(interior == EXIT)
            if (np.all(np.isin(interior, (bg, EXIT))) and mark.shape == (1, 2)
                    and mark[0, 0] == 0):
                panes.append((structural, (r0, r1, c0, c1), interior.copy(), int(mark[0, 1])))
            if np.any(crop == EXIT):
                continue
            holes = _components(crop != structural)
            if not 2 <= len(holes) <= 12:
                continue
            molds = []
            for cells in holes:
                pins = [p for p in cells if crop[p] == ENTRY]
                if len(pins) != 1 or not 3 <= len(cells) <= 8:
                    break
                hr0, hr1, hc0, hc1 = _bounds(cells)
                shape = _normal(cells)
                if hr1-hr0 > 3 or hc1-hc0 > 4:
                    break
                molds.append({"shape": shape, "entry": (pins[0][0]-hr0, pins[0][1]-hc0),
                              "top": hr0, "left": hc0})
            else:
                columns = sorted({m["left"] for m in molds})
                rows = sorted({m["top"] for m in molds})
                if len(molds) != len(columns)*len(rows):
                    continue
                if {(m["top"], m["left"]) for m in molds} != {(r, c) for r in rows for c in columns}:
                    continue
                # Bounding boxes, not entry-pin columns, define the lanes.
                molds.sort(key=lambda m: (m["left"], m["top"]))
                programs.append((structural, (r0, r1, c0, c1), molds, len(rows), len(columns)))
    if len(programs) != 1 or len(panes) != 1 or programs[0][0] != panes[0][0]:
        raise ValueError("expected one mold program and one drawing frame")
    return programs[0], panes[0]


def _decode_piece(inp, cells, bg):
    r0, r1, c0, c1 = _bounds(cells)
    component = np.full((r1-r0, c1-c0), bg, dtype=np.int8)
    for r, c in cells:
        component[r-r0, c-c0] = inp[r, c]
    count = int(np.count_nonzero(component == ENTRY))
    scale = isqrt(count)
    if scale not in (2, 3, 4) or count != scale*scale:
        raise ValueError("piece entry must be one complete macrocell")
    if np.count_nonzero(component == EXIT) != count:
        raise ValueError("piece exit must be one complete macrocell")
    h, w = component.shape
    if h % scale or w % scale:
        raise ValueError("piece bounds do not fit its block scale")
    blocks = component.reshape(h//scale, scale, w//scale, scale)
    logical = blocks[:, 0, :, 0].copy()
    if not np.all(blocks == logical[:, None, :, None]):
        raise ValueError("incomplete or mixed-color macrocell")
    shape = tuple(map(tuple, np.argwhere(logical != bg).tolist()))
    colors = set(map(int, logical[logical != bg])) - {ENTRY, EXIT}
    if not 3 <= len(shape) <= 8 or len(colors) != 1:
        raise ValueError("piece must have 3..8 cells and one body color")
    a = tuple(map(int, np.argwhere(logical == ENTRY)[0]))
    b = tuple(map(int, np.argwhere(logical == EXIT)[0]))
    return {"shape": shape, "entry": a, "exit": b, "color": colors.pop(),
            "scale": scale, "bounds": (r0, r1, c0, c1)}


def _render(info, order=None, chain_columns=True, color_map=None):
    """Render logical cells first; collision and boundary behavior is explicit."""
    bg, seed = info["background"], info["start"]
    logical = info["pane"].copy()
    r, c = 0, seed
    steps = info["steps"] if order is None else [info["steps"][i] for i in order]
    for step in steps:
        shape, a, b, color = step
        if a[0] != 0 or b[0] != _size(shape)[0]-1:
            raise ValueError("entry must be at the top and exit at the bottom")
        origin = (r+1-a[0], (c if chain_columns else seed)-a[1])
        positions = [(origin[0]+y, origin[1]+x) for y, x in shape]
        if any(not (0 <= y < logical.shape[0] and 0 <= x < logical.shape[1]) for y, x in positions):
            raise ValueError("stamp leaves the drawing pane")
        if any(logical[p] != bg for p in positions):
            raise ValueError("stamps collide")
        color = color if color_map is None else color_map.get(color, color)
        for p in positions:
            logical[p] = color
        logical[origin[0]+a[0], origin[1]+a[1]] = ENTRY
        logical[origin[0]+b[0], origin[1]+b[1]] = EXIT
        r, c = origin[0]+b[0], origin[1]+b[1]
    scale = info["scale"]
    return np.repeat(np.repeat(logical, scale, axis=0), scale, axis=1)


def _recover(input_grid):
    inp = _grid(input_grid)
    counts = np.bincount(inp.ravel().astype(int), minlength=10)
    bg = int(np.argmax(counts))
    if bg in (ENTRY, EXIT) or counts[bg] <= inp.size//2:
        raise ValueError("background must occupy more than half the input")
    program, pane = _find_panels(inp, bg)
    structural, pbounds, molds, bands, lanes = program
    _, dbounds, drawing, start = pane
    outside = np.ones(inp.shape, dtype=bool)
    for r0, r1, c0, c1 in (pbounds, dbounds):
        if not np.all(outside[r0:r1, c0:c1]):
            raise ValueError("overlapping panels")
        outside[r0:r1, c0:c1] = False
        moat = np.zeros(inp.shape, dtype=bool)
        moat[max(0, r0-1):min(inp.shape[0], r1+1), max(0, c0-1):min(inp.shape[1], c1+1)] = True
        moat[r0:r1, c0:c1] = False
        if np.any(moat & (inp != bg)):
            raise ValueError("panels require a separating moat")
    pieces, noise_sizes = [], []
    for cells in _components((inp != bg) & outside):
        if len(cells) <= 3:
            if any(inp[p] in (structural, ENTRY, EXIT) for p in cells):
                raise ValueError("noise cannot impersonate a role marker")
            noise_sizes.append(len(cells))
        else:
            piece = _decode_piece(inp, cells, bg)
            if piece["color"] == structural:
                raise ValueError("piece body cannot use the frame color")
            pieces.append(piece)
    if not 2 <= len(pieces) <= 3 or len({p["scale"] for p in pieces}) != 1:
        raise ValueError("expected 2..3 keys at a common scale")
    if len({_canonical(p["shape"]) for p in pieces}) != len(pieces):
        raise ValueError("keys must have different full D4 shape classes")
    scale = pieces[0]["scale"]
    if any(d*scale > 30 for d in drawing.shape):
        raise ValueError("enlarged pane exceeds 30")
    steps, used = [], set()
    reflection = rotation = 0
    for mold in molds:
        matches = set()
        matched_index = None
        for index, piece in enumerate(pieces):
            for shape, a, b in _poses(piece["shape"], piece["entry"], piece["exit"]):
                if shape == mold["shape"] and a == mold["entry"]:
                    matches.add((shape, a, b, piece["color"]))
                    matched_index = index
        if len(matches) != 1:
            raise ValueError("mold does not uniquely determine a decorated stamp")
        step = matches.pop()
        steps.append(step)
        used.add(matched_index)
        piece = pieces[matched_index]
        rotations = _poses(piece["shape"], piece["entry"], piece["exit"], False)
        if step[:3] not in rotations:
            reflection += 1
        elif step[:3] != (piece["shape"], piece["entry"], piece["exit"]):
            rotation += 1
    if len(used) != len(pieces):
        raise ValueError("every key must be used by the program")
    info = {"background": bg, "structural": structural, "pane": drawing, "start": start,
            "scale": scale, "steps": steps, "pieces": pieces, "bands": bands, "lanes": lanes,
            "molds": molds, "program_bounds": pbounds, "pane_bounds": dbounds,
            "reflection": reflection, "rotation": rotation, "noise_sizes": tuple(noise_sizes)}
    output = _render(info)
    return output, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all conventions, scale, matching and cursor state from this input."""
    return _recover(input_grid)[0]


def _grow(rng):
    boxes = ((1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 3))
    h, w = boxes[int(rng.integers(len(boxes)))]
    area = int(rng.integers(3, min(8, h*w)+1))
    cells = {(int(rng.integers(h)), int(rng.integers(w)))}
    for _ in range(area-1):
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _NEIGHBORS
                           if 0 <= r+dr < h and 0 <= c+dc < w} - cells)
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return _normal(cells)


def _ports(rng, shape):
    h, _ = _size(shape)
    pairs = [(a, b) for a in shape if a[0] == 0 for b in shape if b[0] == h-1 and a != b]
    for index in rng.permutation(len(pairs)):
        a, b = pairs[int(index)]
        if _positive_poses(shape, a, b):
            return a, b
    return None


def _make_keys(rng, count, forced, short):
    keys, classes = [], set()
    for index in range(count):
        for attempt in range(100):
            if forced:
                shape = (((0, 0), (1, 0), (2, 0), (2, 1)) if index == 0 else
                         ((0, 0), (0, 1), (0, 2), (1, 1)))
                a, b = ((0, 0), (2, 1)) if index == 0 else ((0, 0), (1, 1))
            else:
                shape = tuple((0, c) for c in range(int(rng.integers(3, 5)))) if index == 0 and short else _grow(rng)
                ports = _ports(rng, shape)
                if ports is None:
                    continue
                a, b = ports
            canonical = _canonical(shape)
            if canonical in classes:
                continue
            choices = _positive_poses(shape, a, b)
            if not choices:
                continue
            keys.append({"shape": shape, "entry": a, "exit": b, "choices": choices})
            classes.add(canonical)
            break
        else:
            return None
    return keys


def _free_positions(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), dtype=int)
    summed = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = summed[h:, w:] - summed[:-h, w:] - summed[h:, :-w] + summed[:-h, :-w]
    return np.argwhere(totals == 0)


def _pack(rng, objects, bg, body_colors):
    for attempt in range(36):
        h, w = (map(int, rng.integers(24, 31, size=2)) if attempt < 12 else (30, 30))
        canvas = np.full((h, w), bg, dtype=np.int8)
        blocked = np.zeros((h, w), dtype=bool)
        jitter = rng.random(len(objects))
        order = sorted(range(len(objects)), key=lambda i: (-objects[i].size, jitter[i]))
        for index in order:
            obj = objects[index]
            oh, ow = obj.shape
            positions = _free_positions(blocked, oh, ow)
            if not len(positions):
                break
            r, c = map(int, positions[int(rng.integers(len(positions)))])
            canvas[r:r+oh, c:c+ow] = obj
            blocked[max(0, r-1):min(h, r+oh+1), max(0, c-1):min(w, c+ow+1)] = True
        else:
            # Small clusters are geometrically too small to be a k>=2 key.
            for _ in range(int(rng.integers(0, 5))):
                size = int(rng.integers(1, 4))
                cells = ((0, 0),) if size == 1 else (((0, 0), (0, 1)) if size == 2 else ((0, 0), (0, 1), (1, 0)))
                nh, nw = _size(cells)
                positions = _free_positions(blocked, nh, nw)
                if not len(positions):
                    continue
                r, c = map(int, positions[int(rng.integers(len(positions)))])
                color = int(rng.choice(body_colors))
                for y, x in cells:
                    canvas[r+y, c+x] = color
                blocked[max(0, r-1):min(h, r+nh+1), max(0, c-1):min(w, c+nw+1)] = True
            if np.count_nonzero(canvas == bg) <= canvas.size//2:
                continue
            # Empty-margin cropping is conditional on keeping visible background.
            if rng.random() < .7:
                r0, r1, c0, c1 = _bounds(tuple(map(tuple, np.argwhere(canvas != bg).tolist())))
                r0, c0 = max(0, r0-int(rng.integers(3))), max(0, c0-int(rng.integers(3)))
                r1, c1 = min(h, r1+int(rng.integers(3))), min(w, c1+int(rng.integers(3)))
                cropped = canvas[r0:r1, c0:c1]
                if np.count_nonzero(cropped == bg) > cropped.size//2:
                    canvas = cropped.copy()
            return canvas
    return None


def _orders(info):
    lanes, bands = info["lanes"], info["bands"]
    for major in (0, 1):
        for reverse_columns in (False, True):
            for reverse_rows in (False, True):
                cols = list(range(lanes))[::(-1 if reverse_columns else 1)]
                rows = list(range(bands))[::(-1 if reverse_rows else 1)]
                indices = ([c*bands+r for c in cols for r in rows] if major == 0 else
                           [c*bands+r for r in rows for c in cols])
                yield (major, reverse_columns, reverse_rows), indices


def _alternative_differs(info, output, **kwargs):
    try:
        return not np.array_equal(_render(info, **kwargs), output)
    except ValueError:
        return True  # A boundary violation is also a falsified hypothesis.


def _make_pair(rng, count, scale, roles, forced=False):
    bg, structural = roles[:2]
    available = list(roles[2:])
    for attempt in range(180):
        key_count = 2 if forced or scale >= 3 else int(rng.integers(2, min(3, count)+1))
        keys = _make_keys(rng, key_count, forced, count >= 8 or scale >= 4 or rng.random() < .25)
        if keys is None:
            continue
        if forced:
            colors = available[:2]
            program = [0, 0, 1, 1]
        else:
            palette_size = int(rng.integers(1, key_count+1))
            colors = [available[i % palette_size] for i in range(key_count)]
            program = [int(v) for v in rng.integers(key_count, size=count)]
            for slot, key_id in zip(rng.permutation(count)[:key_count], range(key_count)):
                program[int(slot)] = key_id
        cap = 30//scale
        minimum_heights = [min(_size(p[0])[0] for p in key["choices"]) for key in keys]
        if 1 + sum(minimum_heights[k] for k in program) > cap:
            continue
        steps, cursor, positions = [], (0, 0), [(0, 0)]
        for index, key_id in enumerate(program):
            choices = keys[key_id]["choices"]
            room = cap-1-cursor[0]-sum(minimum_heights[k] for k in program[index+1:])
            choices = [p for p in choices if _size(p[0])[0] <= room]
            if forced and index == 0:
                choices = [(keys[0]["shape"], keys[0]["entry"], keys[0]["exit"])]
            if forced and index == 1:
                choices = [p for p in choices if p[0] != steps[0][0] and p[2][1]-p[1][1] != steps[0][2][1]-steps[0][1][1]] or choices
            if not choices:
                break
            shape, a, b = choices[int(rng.integers(len(choices)))]
            origin = cursor[0]+1, cursor[1]-a[1]
            positions.extend((origin[0]+r, origin[1]+c) for r, c in shape)
            cursor = origin[0]+b[0], origin[1]+b[1]
            steps.append((shape, a, b, colors[key_id]))
        if len(steps) != count:
            continue
        min_c, max_c = min(c for r, c in positions), max(c for r, c in positions)
        span = max_c-min_c+1
        if span > cap:
            continue
        left_margin = int(rng.integers(min(2, cap-span)+1))
        right_margin = int(rng.integers(min(2, cap-span-left_margin)+1))
        width = span+left_margin+right_margin
        path_h = cursor[0]+1
        height = path_h+int(rng.integers(min(3, cap-path_h)+1))
        start = left_margin-min_c
        drawing = np.full((height, width), bg, dtype=np.int8)
        drawing[0, start] = EXIT
        info = {"background": bg, "start": start, "pane": drawing, "scale": scale, "steps": steps}
        output = _render(info)
        layouts = {2: ((2, 1), (1, 2)), 3: ((3, 1), (1, 3)), 4: ((2, 2),),
                   6: ((3, 2), (2, 3)), 8: ((4, 2), (2, 4))}
        bands, lanes = layouts[count][int(rng.integers(len(layouts[count])))]
        gh = max(_size(step[0])[0] for step in steps)
        gw = max(_size(step[0])[1] for step in steps)
        yg, xg = map(int, rng.integers(1, 3, size=2))
        board = np.full((bands*gh+(bands-1)*yg+2, lanes*gw+(lanes-1)*xg+2), structural, dtype=np.int8)
        for index, (shape, a, b, color) in enumerate(steps):
            lane, band = divmod(index, bands)
            top, left = 1+band*(gh+yg), 1+lane*(gw+xg)
            for r, c in shape:
                board[top+r, left+c] = bg
            board[top+a[0], left+a[1]] = ENTRY
        frame = np.full((height+2, width+2), structural, dtype=np.int8)
        frame[1:-1, 1:-1] = drawing
        objects = [board, frame]
        for index, key in enumerate(keys):
            poses = list(_poses(key["shape"], key["entry"], key["exit"]))
            if forced and index == 0:
                target = steps[0][:3]
                poses = [p for p in poses if target not in _poses(*p, reflections=False)]
            shape, a, b = poses[int(rng.integers(len(poses)))]
            logical = np.full(_size(shape), bg, dtype=np.int8)
            for p in shape:
                logical[p] = colors[index]
            logical[a], logical[b] = ENTRY, EXIT
            objects.append(np.repeat(np.repeat(logical, scale, 0), scale, 1))
        inp = _pack(rng, objects, bg, available)
        if inp is None:
            continue
        recovered, observed = _recover(inp)
        if not np.array_equal(output, recovered):
            raise AssertionError("construction and input recovery disagree")
        if forced:
            if not observed["reflection"] or not _alternative_differs(observed, output, chain_columns=False):
                continue
            if any(not _alternative_differs(observed, output, order=order)
                   for name, order in _orders(observed) if name != (0, False, False)):
                continue
        return {"input": inp, "output": output}
    raise RuntimeError("bounded geometric construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    scale_shift = bool(rng.integers(3) == 0)
    query_scale = int(rng.choice([3, 4])) if scale_shift else 2
    puzzle, seen = {"train": [], "test": []}, set()
    previous_roles = None
    for split, total in (("train", nt), ("test", nq)):
        for index in range(total):
            role_palette = rng.permutation([c for c in range(10) if c not in (ENTRY, EXIT)])
            if previous_roles is not None:
                for _ in range(8):
                    if all(role_palette[j] != previous_roles[j] for j in (0, 1)):
                        break
                    role_palette = np.roll(role_palette, 1)
            roles = tuple(map(int, role_palette))
            previous_roles = roles
            if split == "train":
                count = 4 if index < 2 else int(rng.choice([2, 3, 4]))
                scale = 2 if index == 0 or scale_shift else (3 if index == 1 else int(rng.choice([2, 3, 4] if count < 4 else [2, 3])))
            else:
                count = 4 if scale_shift else int(rng.choice([6, 8]))
                scale = query_scale
            for _ in range(12):
                pair = _make_pair(rng, count, scale, roles, forced=(split == "train" and index == 0))
                signature = _key(pair["input"])
                if signature not in seen:
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
            seen.add(signature)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, output = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            expected, info = _recover(inp)
            assert np.array_equal(expected, output), "incorrect fitted, chained output"
            k, bg = info["scale"], info["background"]
            assert output.shape == tuple(k*d for d in info["pane"].shape)
            logical = output[::k, ::k]
            assert np.array_equal(output, np.repeat(np.repeat(logical, k, 0), k, 1))
            # Conservation per executed stamp, accounting for its two ports.
            volume = Counter({EXIT: 1})
            for shape, a, b, color in info["steps"]:
                volume[color] += len(shape)-2
                volume[ENTRY] += 1
                volume[EXIT] += 1
            observed = Counter(map(int, logical[logical != bg]))
            assert observed == volume, "material/port counts were not preserved"
            assert np.array_equal(output[:k], np.repeat(np.repeat(info["pane"][:1], k, 0), k, 1))
            assert sum(info["noise_sizes"]) <= 12
            assert not np.array_equal(inp, output)
            records[split].append((info, output))
    first, first_output = records["train"][0]
    assert first["reflection"] > 0, "reflection requires a visible demonstration"
    assert first["lanes"] == first["bands"] == 2
    classes = [(len(p["shape"]), _canonical(p["shape"]), p["color"]) for p in first["pieces"]]
    assert any(a[0] == b[0] and a[1] != b[1] and a[2] != b[2]
               for i, a in enumerate(classes) for b in classes[i+1:]), "need an equal-area shape-matching witness"
    assert _alternative_differs(first, first_output, chain_columns=False), "need an endpoint witness"
    for name, order in _orders(first):
        if name != (0, False, False):
            assert _alternative_differs(first, first_output, order=order), "need a traversal witness"
    colors = [p["color"] for p in first["pieces"]]
    assert _alternative_differs(first, first_output, color_map={colors[0]: colors[1], colors[1]: colors[0]}), "shape correspondence must affect the answer"
    for role in ("background", "structural"):
        assert len({info[role] for info, out in records["train"]}) >= 2
    max_train_length = max(len(info["steps"]) for info, out in records["train"])
    training_scales = {info["scale"] for info, out in records["train"]}
    assert all(len(info["steps"]) > max_train_length or info["scale"] not in training_scales
               for info, out in records["test"]), "missing declared train/test shift"


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def self_test() -> dict:
    """Bounded 42-seed coverage plus counts, determinism and semantic mutations."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    infos = [_recover(pair["input"])[1] for p in puzzles for s in ("train", "test") for pair in p[s]]
    shapes = {_canonical(piece["shape"]) for info in infos for piece in info["pieces"]}
    assert {i["scale"] for i in infos} == {2, 3, 4}
    assert {len(i["steps"]) for i in infos} >= {2, 3, 4, 6, 8}
    assert {i["lanes"] for i in infos} >= {1, 2, 3, 4}
    assert {len(i["pieces"]) for i in infos} == {2, 3}
    assert len(shapes) >= 25 and {len(s) for s in shapes} >= {3, 4, 5, 6, 7, 8}
    assert any(len({p["color"] for p in i["pieces"]}) < len(i["pieces"]) for i in infos)
    assert any(not i["noise_sizes"] for i in infos)
    assert {n for i in infos for n in i["noise_sizes"]} == {1, 2, 3}
    assert any(i["reflection"] for i in infos) and any(i["rotation"] for i in infos)
    assert len({i["pane"].shape for i in infos}) >= 20
    for counts in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(735, *counts))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for damage in ("output", "exit", "mold"):
        damaged = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
        pair = damaged["train"][0]
        _, info = _recover(pair["input"])
        if damage == "output":
            pair["output"][0, 0] = (int(pair["output"][0, 0])+1) % 10
        elif damage == "exit":
            r0, r1, c0, c1 = info["pieces"][0]["bounds"]
            region = pair["input"][r0:r1, c0:c1]
            region[region == EXIT] = info["pieces"][0]["color"]
        else:
            r0, r1, c0, c1 = info["program_bounds"]
            region = pair["input"][r0:r1, c0:c1]
            r, c = np.argwhere(region == ENTRY)[0]
            region[r, c] = info["background"]
        try:
            validate_generated_puzzle(damaged)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"semantic damage accepted: {damage}")
    # A centered entry on a three-cell bar leaves two possible exits. Reject
    # that actual input rather than breaking the symmetry by spatial ordering.
    diagnostic = np.zeros((22, 22), dtype=np.int8)
    diagnostic[:7, :5] = 3
    diagnostic[1, 1:4] = 0
    diagnostic[1, 2] = ENTRY
    l_shape = ((0, 0), (1, 0), (2, 0), (2, 1))
    for r, c in l_shape:
        diagnostic[3+r, 1+c] = 0
    diagnostic[3, 1] = ENTRY
    diagnostic[:10, 9:17] = 3
    diagnostic[1:9, 10:16] = 0
    diagnostic[1, 12] = EXIT
    bar = np.array([[4, ENTRY, EXIT]], dtype=np.int8)
    diagnostic[12:14, :6] = np.repeat(np.repeat(bar, 2, 0), 2, 1)
    l_key = np.array([[ENTRY, 0], [5, 0], [5, EXIT]], dtype=np.int8)
    diagnostic[12:18, 9:13] = np.repeat(np.repeat(l_key, 2, 0), 2, 1)
    try:
        solve(diagnostic)
    except ValueError as error:
        assert "uniquely determine" in str(error)
    else:
        raise AssertionError("ambiguous port orientation accepted")
    # Moving the entry off the axis resolves the same geometric symmetry.
    diagnostic[1, 2], diagnostic[1, 1] = 0, ENTRY
    bar = np.array([[ENTRY, EXIT, 4]], dtype=np.int8)
    diagnostic[12:14, :6] = np.repeat(np.repeat(bar, 2, 0), 2, 1)
    assert solve(diagnostic).shape == (16, 12)

    # A valid five-command, single-lane member is outside the sampler's count
    # catalogue. Decode it from a fresh hand-constructed input, without labels.
    odd = np.zeros((26, 22), dtype=np.int8)
    odd[:16, :5] = 3
    bar_shape = ((0, 0), (0, 1), (0, 2))
    sideways_l = ((0, 0), (0, 1), (0, 2), (1, 2))
    for index in range(5):
        mask, pin = (bar_shape, (0, 2)) if index % 2 == 0 else (sideways_l, (0, 0))
        for r, c in mask:
            odd[1+3*index+r, 1+c] = 0
        odd[1+3*index+pin[0], 1+pin[1]] = ENTRY
    odd[:12, 7:15] = 3
    odd[1:11, 8:14] = 0
    odd[1, 10] = EXIT
    bar = np.array([[EXIT, 4, ENTRY]], dtype=np.int8)
    odd[18:20, :6] = np.repeat(np.repeat(bar, 2, 0), 2, 1)
    odd[18:24, 9:13] = np.repeat(np.repeat(l_key, 2, 0), 2, 1)
    odd_output, odd_info = _recover(odd)
    assert len(odd_info["steps"]) == 5 and odd_info["lanes"] == 1
    assert odd_output.shape == (20, 12)
    assert np.count_nonzero(odd_output == ENTRY) == 20
    assert np.count_nonzero(odd_output == EXIT) == 24
    report = {"seeds": 42, "pairs": len(infos), "shape_classes": len(shapes),
              "scales": sorted({i["scale"] for i in infos}),
              "program_lengths": sorted({len(i["steps"]) for i in infos}),
              "input_shapes": len({np.asarray(pair["input"]).shape for p in puzzles for s in p for pair in p[s]}),
              "pane_shapes": len({i["pane"].shape for i in infos}),
              "reflection_pairs": sum(i["reflection"] > 0 for i in infos),
              "rotation_pairs": sum(i["rotation"] > 0 for i in infos)}
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
