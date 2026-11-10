"""Analogous chamber weaving. Standard library and NumPy only.

Generation uses room ownership and forward block copying. Solving reads a
reference pair and wall topology from pixels; it never reads generation state.
"""
from __future__ import annotations

from collections import deque
import copy
import json

import numpy as np


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(int, copy=True)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _components(mask):
    """Deterministic four-neighbor components, each as coordinate pairs."""
    h, w = mask.shape
    seen = np.zeros(mask.shape, bool)
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        todo, cells = deque([(r, c)]), []
        while todo:
            y, x = todo.popleft()
            cells.append((y, x))
            for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    todo.append((ny, nx))
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _bbox(cells):
    lo, hi = cells.min(axis=0), cells.max(axis=0) + 1
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _outline(a, box, color):
    r, c, end_r, end_c = box
    a[r, c:end_c] = a[end_r-1, c:end_c] = color
    a[r:end_r, c] = a[r:end_r, end_c-1] = color


def _reference(field, ground):
    """Recognize two isolated square frames, regardless of their order/axis."""
    candidates = []
    for frame in np.unique(field):
        frame = int(frame)
        if frame == ground:
            continue
        outlines = _components(field == frame)
        if len(outlines) != 2:
            continue
        boxes = [_bbox(cells) for cells in outlines]
        sizes = [(y-r, x-c) for r, c, y, x in boxes]
        if sizes[0] != sizes[1] or sizes[0] not in ((4, 4), (5, 5)):
            continue
        expected = np.full(field.shape, ground, int)
        panels = []
        for box in boxes:
            r, c, y, x = box
            _outline(expected, box, frame)
            panels.append(field[r+1:y-1, c+1:x-1].copy())
            expected[r+1:y-1, c+1:x-1] = panels[-1]
        if not np.array_equal(expected, field):
            continue
        sparse_panels = [p for p in panels if np.any(p == ground)]
        full_panels = [p for p in panels if not np.any(p == ground)]
        if len(sparse_panels) != 1 or len(full_panels) != 1:
            continue
        sparse, full = sparse_panels[0], full_panels[0]
        colors = tuple(map(int, np.unique(full)))
        coords = np.argwhere(sparse != ground)
        if (len(colors) != 2 or frame in colors or ground in colors
                or set(map(int, np.unique(sparse))) != {ground, *colors}
                or not 2 <= len(coords) <= 4 or len(coords) >= full.size
                or np.any(np.ptp(coords, axis=0) < 1)
                or not np.all(sparse[sparse != ground] == full[sparse != ground])):
            continue
        candidates.append(dict(frame=frame, full=full, sparse=sparse,
                               cue_coords=coords, colors=colors, n=len(full), boxes=boxes))
    if len(candidates) != 1:
        raise ValueError("not a unique sparse/full reference field")
    return candidates[0]


def _match_cue(board, cells, ref, ground, wall):
    colored = np.array([(int(r), int(c)) for r, c in cells
                        if int(board[r, c]) not in (ground, wall)], dtype=int)
    if colored.shape != ref["cue_coords"].shape:
        raise ValueError("closed marked component must contain exactly one cue")
    shift = colored.min(axis=0) - ref["cue_coords"].min(axis=0)
    expected = ref["cue_coords"] + shift
    if not np.array_equal(colored, expected):
        raise ValueError("cue geometry does not match the sparse reference")
    mapping = {}
    for (r, c), (y, x) in zip(ref["cue_coords"], expected):
        source, target = int(ref["sparse"][r, c]), int(board[y, x])
        if source in mapping and mapping[source] != target:
            raise ValueError("inconsistent color-role transfer")
        mapping[source] = target
    if set(mapping) != set(ref["colors"]) or len(set(mapping.values())) != 2:
        raise ValueError("both distinct target roles must be visible")
    r, c = map(int, shift)
    n = ref["n"]
    region = set(map(tuple, cells.tolist()))
    if any((y, x) not in region for y in range(r, r+n) for x in range(c, c+n)):
        raise ValueError("the complete cue support must fit in its component")
    return mapping, (r, c)


def _decode(input_grid):
    a = _grid(input_grid)
    ground = int(a[0, 0])
    if any(int(a[r, c]) != ground for r, c in ((0, -1), (-1, 0), (-1, -1))):
        raise ValueError("input corners must show neutral padding")
    separators = []
    for axis in (0, 1):
        for i in range(1, a.shape[axis]-1):
            line = a[i, :] if axis == 0 else a[:, i]
            color = int(line[0])
            if color != ground and np.all(line == color) and np.count_nonzero(a == color) == len(line):
                separators.append((axis, i, color))
    if len(separators) != 1:
        raise ValueError("expected one exclusive full separator")
    axis, i, divider = separators[0]
    fields = [a[:i], a[i+1:]] if axis == 0 else [a[:, :i], a[:, i+1:]]
    references = []
    for j, field in enumerate(fields):
        try:
            references.append((j, _reference(field, ground)))
        except ValueError:
            pass
    if len(references) != 1:
        raise ValueError("expected exactly one reference field")
    j, ref = references[0]
    field, frame = fields[1-j], ref["frame"]
    frame_cells = np.argwhere(field == frame)
    if not len(frame_cells):
        raise ValueError("missing map frame")
    box = _bbox(frame_cells)
    r, c, y, x = box
    expected = np.full(field.shape, ground, int)
    _outline(expected, box, frame)
    expected[r+1:y-1, c+1:x-1] = field[r+1:y-1, c+1:x-1]
    if not np.array_equal(expected, field) or np.any(field[r+1:y-1, c+1:x-1] == frame):
        raise ValueError("map field must contain one visibly bounded map")
    board = field[r+1:y-1, c+1:x-1].copy()
    if min(board.shape) < 4 or max(board.shape) * ref["n"] > 30:
        raise ValueError("expanded map does not fit ARC bounds")
    walls = []
    for color in np.unique(board):
        color = int(color)
        if color == ground:
            continue
        if any(len(comp) >= 12 and min(np.ptp(comp, axis=0)) >= 3
               for comp in _components(board == color)):
            walls.append(color)
    if len(walls) != 1:
        raise ValueError("wall color must have unique connected geometry")
    wall = walls[0]
    if len({ground, wall, frame, divider}) != 4 or wall in ref["colors"]:
        raise ValueError("structural roles must be distinct from motif roles")
    regions = []
    h, w = board.shape
    for cells in _components(board != wall):
        closed = not np.any((cells[:, 0] == 0) | (cells[:, 0] == h-1)
                            | (cells[:, 1] == 0) | (cells[:, 1] == w-1))
        marked = bool(np.any(board[cells[:, 0], cells[:, 1]] != ground))
        mapping, origin = None, None
        if marked:
            try:
                mapping, origin = _match_cue(board, cells, ref, ground, wall)
            except ValueError:
                if closed:
                    raise
                # Several independent cues may share the exterior component.
        regions.append(dict(cells=cells, closed=closed, marked=marked,
                            mapping=mapping, origin=origin, active=closed and marked))
    if not any(reg["active"] for reg in regions):
        raise ValueError("map must have a closed marked chamber")
    return dict(board=board, ground=ground, wall=wall, divider=divider,
                ref=ref, regions=regions, layout=(axis, j), map_box=box)


def _tile_for(decoded, region, behavior="normal"):
    ref, ground = decoded["ref"], decoded["ground"]
    first, second = ref["colors"]
    mapping = region["mapping"]
    source = ref["sparse"] if behavior == "no_blueprint" else ref["full"]
    a, b = (first, second) if behavior == "literal_palette" else (mapping[first], mapping[second])
    if behavior == "no_exchange":
        tile = np.where(source == first, a, b)
    else:
        tile = np.where(source == first, b, a)
    if behavior == "no_blueprint":
        tile = np.where(source == ground, ground, tile)
    if behavior == "solid":
        tile[:] = a
    if behavior == "rotate_tile":
        tile = np.rot90(tile, -1)
    return tile.astype(int)


def _render(decoded, behavior="normal"):
    board, n = decoded["board"], decoded["ref"]["n"]
    out = np.repeat(np.repeat(board, n, axis=0), n, axis=1)
    for reg in decoded["regions"]:
        if not reg["active"] and not (behavior == "fill_open" and reg["mapping"] is not None):
            continue
        cells = reg["cells"]
        if behavior == "bounding_box":
            r, c, y, x = _bbox(cells)
            cells = [(rr, cc) for rr in range(r, y) for cc in range(c, x)]
        tile = _tile_for(decoded, reg, behavior)
        for r, c in cells:
            out[r*n:(r+1)*n, c*n:(c+1)*n] = tile
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct the output from this input alone, without training or seeds."""
    return _render(_decode(input_grid))


def _swatches(rng, n, ground, colors):
    """Complete binary tiles; reveal positions, never erase an implied row."""
    for _ in range(64):
        mode = int(rng.integers(4))
        rr, cc = np.indices((n, n))
        if mode == 0:
            bits = rng.random((n, n)) < rng.uniform(.2, .8)
        elif mode == 1:
            bits = (rr == int(rng.integers(n))) | (cc == int(rng.integers(n)))
        elif mode == 2:
            bits = rr < int(rng.integers(1, n))
            if rng.integers(2):
                bits = bits.T
        else:
            bits = np.zeros((n, n), bool)
            r, c = int(rng.integers(n)), int(rng.integers(n))
            for _ in range(int(rng.integers(1, n*n))):
                bits[r, c] = True
                dr, dc = ((0, 1), (0, -1), (1, 0), (-1, 0))[int(rng.integers(4))]
                r, c = int(np.clip(r+dr, 0, n-1)), int(np.clip(c+dc, 0, n-1))
        if bits.all() or not bits.any() or np.array_equal(bits, np.rot90(bits)):
            continue
        if rng.integers(2):
            bits = ~bits
        full = np.where(bits, colors[0], colors[1]).astype(int)
        for _ in range(24):
            count = int(rng.integers(2 if n == 2 else 3, min(4, n*n-1)+1))
            chosen = np.sort(rng.choice(n*n, count, replace=False))
            coords = np.array([divmod(int(p), n) for p in chosen])
            if min(np.ptp(coords, axis=0)) < 1 or len(np.unique(full.flat[chosen])) != 2:
                continue
            sparse = np.full((n, n), ground, int)
            sparse.flat[chosen] = full.flat[chosen]
            return sparse, full
    # Bounded fallback still has both roles, unknown cells, and visible extent.
    full = np.full((n, n), colors[0], int)
    full[0, 0] = colors[1]
    sparse = np.full((n, n), ground, int)
    sparse[0, 0], sparse[-1, -1] = colors[1], colors[0]
    return sparse, full


def _axis_sizes(rng, count, n, maximum):
    sizes = [n] * count
    spare = maximum - (count*n + count + 1)
    for i in rng.permutation(count):
        extra = int(rng.integers(min(3, spare)+1))
        sizes[int(i)] += extra
        spare -= extra
    before = int(rng.integers(min(2, spare)+1))
    spare -= before
    after = int(rng.integers(min(2, spare)+1))
    lines = [before]
    for size in sizes:
        lines.append(lines[-1] + size + 1)
    return sizes, lines, lines[-1] + 1 + after


def _pack(rng, board, sparse, full, ground, frame, divider):
    n, side = len(full), len(full)+2
    gap = int(rng.integers(1, 4))
    rp = [int(rng.integers(1, 3)) for _ in range(4)]
    mp = [int(rng.integers(1, 3)) for _ in range(4)]
    ref_h, ref_w = 2*side+gap+rp[0]+rp[1], side+rp[2]+rp[3]
    map_h, map_w = board.shape[0]+2+mp[0]+mp[1], board.shape[1]+2+mp[2]+mp[3]
    h = max(ref_h, map_h)
    reference_field = np.full((h, ref_w), ground, int)
    map_field = np.full((h, map_w), ground, int)
    top = rp[0] + int(rng.integers(h-ref_h+1))
    panels = [sparse, full] if rng.integers(2) else [full, sparse]
    for k, panel in enumerate(panels):
        r, c = top+k*(side+gap), rp[2]
        _outline(reference_field, (r, c, r+side, c+side), frame)
        reference_field[r+1:r+n+1, c+1:c+n+1] = panel
    r = mp[0] + int(rng.integers(h-map_h+1))
    c = mp[2]
    bh, bw = board.shape
    _outline(map_field, (r, c, r+bh+2, c+bw+2), frame)
    map_field[r+1:r+bh+1, c+1:c+bw+1] = board
    line = np.full((h, 1), divider, int)
    parts = [reference_field, line, map_field] if rng.integers(2) else [map_field, line, reference_field]
    return np.concatenate(parts, axis=1)


def _make_pair(rng, kind):
    n = 2 if kind == "first" else 3 if kind == "second" else int(rng.integers(2, 4))
    if kind in ("first", "second", "test"):
        rows = cols = 2
    else:
        options = [(1, 1), (1, 2), (2, 1), (2, 2)]
        if n == 2:
            options += [(1, 3), (3, 1)]
        rows, cols = options[int(rng.integers(len(options)))]
    palette = list(map(int, rng.permutation(10)))
    ground, frame, divider, wall = palette[:4]
    ink = palette[4:]
    reference_colors = tuple(ink[:2])
    sparse, full = _swatches(rng, n, ground, reference_colors)
    heights, rlines, h = _axis_sizes(rng, rows, n, 30//n)
    widths, clines, w = _axis_sizes(rng, cols, n, 30//n)
    board = np.full((h, w), ground, int)
    rooms = []
    for r in range(rows):
        for c in range(cols):
            box = (rlines[r], clines[c], rlines[r+1]+1, clines[c+1]+1)
            _outline(board, box, wall)
            rooms.append((r, c, box))
    edges = []
    for i, (r, c, _) in enumerate(rooms):
        if c+1 < cols:
            edges.append((i, i+1))
        if r+1 < rows:
            edges.append((i, i+cols))
    joins = []
    if kind == "second":
        joins = [edges[int(rng.integers(len(edges)))]]
    elif kind == "test":
        omitted = int(rng.integers(4))
        joins = [edge for edge in edges if omitted not in edge]
    elif kind == "extra":
        joins = [edge for edge in edges if rng.random() < .28]
    parent = list(range(len(rooms)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    links = []
    for first, second in joins:
        parent[root(second)] = root(first)
        ar, ac, a_box = rooms[first]
        br, bc, _ = rooms[second]
        r0, c0, r1, c1 = a_box
        if ar == br:
            length = int(rng.integers(1, min(3, heights[ar])+1))
            start = int(rng.integers(r0+1, r1-length))
            positions = [(r, c1-1) for r in range(start, start+length)]
        else:
            length = int(rng.integers(1, min(3, widths[ac])+1))
            start = int(rng.integers(c0+1, c1-length))
            positions = [(r1-1, c) for c in range(start, start+length)]
        for point in positions:
            board[point] = ground
        links.append((first, positions))
    groups = {}
    for i in range(len(rooms)):
        groups.setdefault(root(i), []).append(i)
    group_ids = sorted(groups)
    if kind == "first":
        chosen = list(map(int, rng.permutation(group_ids)))
        active, opened, unmarked = chosen[:2], chosen[2], [chosen[3]]
    elif kind == "second":
        merged = next(g for g in group_ids if len(groups[g]) == 2)
        singles = [g for g in group_ids if g != merged]
        rng.shuffle(singles)
        active, opened, unmarked = [merged, singles[0]], singles[1], []
    elif kind == "test":
        active = [next(g for g in group_ids if len(groups[g]) == 3)]
        opened = next(g for g in group_ids if g not in active)
        unmarked = []
    else:
        rng.shuffle(group_ids)
        opened = group_ids[-1] if len(group_ids) > 1 and rng.random() < .7 else None
        available = [g for g in group_ids if g != opened]
        count = int(rng.integers(1, len(available)+1))
        active, unmarked = available[:count], available[count:]
    if opened is not None:
        choices = []
        for room in groups[opened]:
            r, c, (r0, c0, r1, c1) = rooms[room]
            if r == 0:
                choices += [(r0, x) for x in range(c0+1, c1-1)]
            if r == rows-1:
                choices += [(r1-1, x) for x in range(c0+1, c1-1)]
            if c == 0:
                choices += [(y, c0) for y in range(r0+1, r1-1)]
            if c == cols-1:
                choices += [(y, c1-1) for y in range(r0+1, r1-1)]
        if not choices:
            raise AssertionError("open group has no outer wall")
        board[choices[int(rng.integers(len(choices)))]] = ground
    marked_groups = active + ([] if opened is None else [opened])
    cue_rooms, role_maps = {}, {}
    for k, g in enumerate(marked_groups):
        # The second witness deliberately puts the cue beyond the passage.
        room = max(groups[g]) if kind == "second" and len(groups[g]) == 2 else int(rng.choice(groups[g]))
        cue_rooms[g] = room
        rr, cc, (r0, c0, r1, c1) = rooms[room]
        cr = int(rng.integers(r0+1, r1-n))
        cx = int(rng.integers(c0+1, c1-n))
        choices = ink[2:] if kind == "first" and k == 0 else ink
        target = list(map(int, rng.choice(choices, 2, replace=False)))
        role_maps[g] = dict(zip(reference_colors, target))
        for y, x in np.argwhere(sparse != ground):
            board[cr+y, cx+x] = role_maps[g][int(sparse[y, x])]
    # Obstacles are in a cue-free room, strictly away from its walls. They
    # cannot obscure the square support or sever these >=3-wide interiors.
    for g in active:
        empty_rooms = [i for i in groups[g] if i != cue_rooms[g]]
        for room in empty_rooms:
            rr, cc, (r0, c0, r1, c1) = rooms[room]
            if min(heights[rr], widths[cc]) >= 3 and (kind == "second" or rng.random() < .7):
                board[(r0+r1-1)//2, (c0+c1-1)//2] = wall
    # Forward labels from construction ownership, not from solve or decoding.
    out = np.repeat(np.repeat(board, n, axis=0), n, axis=1)
    first_color, second_color = reference_colors
    for g in active:
        mapping = role_maps[g]
        tile = np.where(full == first_color, mapping[second_color], mapping[first_color])
        cells = set()
        for room in groups[g]:
            _, _, (r0, c0, r1, c1) = rooms[room]
            cells.update((r, c) for r in range(r0+1, r1-1) for c in range(c0+1, c1-1))
        for room, positions in links:
            if root(room) == g:
                cells.update(positions)
        for r, c in sorted(cells):
            if board[r, c] != wall:
                out[r*n:(r+1)*n, c*n:(c+1)*n] = tile
    inp = _pack(rng, board, sparse, full, ground, frame, divider)
    if rng.integers(2):
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            kind = "test" if split == "test" else "first" if i == 0 else "second" if i == 1 else "extra"
            for _ in range(32):
                pair = _make_pair(rng, kind)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
    return puzzle


def _inverse_check(decoded, output):
    """Verify labels by output coordinates, without the forward painter."""
    board, n = decoded["board"], decoded["ref"]["n"]
    if output.shape != (board.shape[0]*n, board.shape[1]*n):
        raise AssertionError("wrong expanded-map dimensions")
    tiles = {}
    for reg in decoded["regions"]:
        if reg["active"]:
            tile = _tile_for(decoded, reg)
            for point in map(tuple, reg["cells"].tolist()):
                assert point not in tiles, "overlapping region footprints"
                tiles[point] = tile
    for r in range(output.shape[0]):
        for c in range(output.shape[1]):
            macro = r//n, c//n
            expected = tiles[macro][r % n, c % n] if macro in tiles else board[macro]
            assert output[r, c] == expected, "incorrect block value or untouched cell"


def validate_generated_puzzle(puzzle):
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert all(isinstance(puzzle[s], list) for s in puzzle)
        nt, nq = len(puzzle["train"]), len(puzzle["test"])
        _count(nt, "num_train", 2, 10)
        _count(nq, "num_test", 1, 4)
        assert nt > nq
        seen, decoded_train = set(), []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                inp, out = _grid(pair["input"]), _grid(pair["output"])
                assert _key(inp) not in seen, "duplicate input"
                seen.add(_key(inp))
                d = _decode(inp)
                _inverse_check(d, out)
                if split == "train":
                    decoded_train.append(d)
        first, second = decoded_train[:2]
        assert first["ref"]["n"] == 2 and second["ref"]["n"] == 3
        assert sum(reg["active"] for reg in first["regions"]) >= 2
        assert any(reg["closed"] and not reg["marked"] for reg in first["regions"])
        assert any(not reg["closed"] and reg["mapping"] is not None for reg in first["regions"])
        assert sum(reg["active"] for reg in second["regions"]) >= 2
        assert any(reg["active"] and len(reg["cells"]) > 2*second["ref"]["n"]**2-2
                   for reg in second["regions"]), "missing passage propagation witness"
        assert len(_components(second["board"] == second["wall"])) >= 2, "missing wall island"
        for behavior in ("no_blueprint", "no_exchange", "solid", "literal_palette",
                         "bounding_box", "fill_open", "rotate_tile"):
            assert any(not np.array_equal(_render(d, behavior), np.asarray(p["output"]))
                       for d, p in zip(decoded_train[:2], puzzle["train"][:2])), behavior
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"malformed generated puzzle: {exc}") from exc


def _features(pair):
    d = _decode(pair["input"])
    ref, regs = d["ref"], d["regions"]
    mask = np.zeros(d["board"].shape, np.uint8)
    for reg in regs:
        if reg["active"]:
            mask[reg["cells"][:, 0], reg["cells"][:, 1]] = 1
    return dict(n=ref["n"], map_shape=list(d["board"].shape), layout=list(d["layout"]),
                palette_size=len(np.unique(pair["input"])),
                structural_palette=[d["ground"], ref["frame"], d["divider"], d["wall"]],
                active_regions=sum(reg["active"] for reg in regs),
                active_cells=int(mask.sum()),
                exterior_marked=sum(reg["marked"] and not reg["closed"] for reg in regs),
                unmarked_closed=sum(reg["closed"] and not reg["marked"] for reg in regs),
                wall_components=len(_components(d["board"] == d["wall"])),
                motif=(ref["full"] == ref["colors"][0]).astype(np.uint8).tobytes().hex(),
                stencil=mask.tobytes().hex(), cue_cells=len(ref["cue_coords"]))


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def _counterfactual_tests():
    p = generate_puzzle(0, 2, 1)
    inp = p["train"][0]["input"].copy()
    d = _decode(inp)
    # Change one unrevealed full-reference bit and nothing in the map/cue.
    # Repacking is unnecessary: locate the full panel via input-only geometry.
    axis, field_index = d["layout"]
    div = d["divider"]
    sep = int(np.flatnonzero(np.all(inp == div, axis=1 if axis == 0 else 0))[0])
    field = (inp[:sep] if field_index == 0 else inp[sep+1:]) if axis == 0 else (
        inp[:, :sep] if field_index == 0 else inp[:, sep+1:])
    full_box = next(box for box in d["ref"]["boxes"]
                    if not np.any(field[box[0]+1:box[2]-1, box[1]+1:box[3]-1] == d["ground"]))
    y, x = map(int, np.argwhere(d["ref"]["sparse"] == d["ground"])[0])
    r, c, _, _ = full_box
    colors = d["ref"]["colors"]
    old = int(field[r+1+y, c+1+x])
    field[r+1+y, c+1+x] = colors[0] if old == colors[1] else colors[1]
    changed = _decode(inp)
    assert np.array_equal(changed["board"], d["board"])
    assert np.array_equal(changed["ref"]["sparse"], d["ref"]["sparse"])
    assert not np.array_equal(solve(inp), p["train"][0]["output"]), "reference partner is dispensable"
    # Open one wall adjacent to an active region and the exterior. If they
    # have no immediately shared edge, search a short all-wall path instead.
    d = _decode(p["train"][0]["input"])
    board = d["board"].copy()
    active = next(reg for reg in d["regions"] if reg["active"])
    starts = set(map(tuple, active["cells"].tolist()))
    exterior = set().union(*(set(map(tuple, reg["cells"].tolist()))
                             for reg in d["regions"] if not reg["closed"]))
    todo = deque((point, []) for point in sorted(starts))
    visited = set(starts)
    route = None
    while todo and route is None:
        (r, c), path = todo.popleft()
        for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if not (0 <= y < len(board) and 0 <= x < board.shape[1]):
                route = path
                break
            if (y, x) in exterior:
                route = path
                break
            if (y, x) not in visited and board[y, x] == d["wall"]:
                visited.add((y, x))
                todo.append(((y, x), path+[(y, x)]))
    assert route
    for point in route:
        board[point] = d["ground"]
    inp = _pack(np.random.default_rng(401), board, d["ref"]["sparse"], d["ref"]["full"],
                d["ground"], d["ref"]["frame"], d["divider"])
    opened = _decode(inp)
    assert sum(reg["active"] for reg in opened["regions"]) < sum(reg["active"] for reg in d["regions"])
    assert not np.array_equal(solve(inp), _render(d)), "wall topology is dispensable"
    # A boundary cell and interior cell touching only diagonally are separate.
    mask = np.zeros((5, 5), bool)
    mask[0, 0] = mask[1, 1] = mask[2, 1] = True
    comps = _components(mask)
    assert sorted(map(len, comps)) == [1, 2], "diagonal adjacency leaked into flood fill"


def self_test():
    records, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), "seed is not reproducible"
        signatures.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                records.append(_features(pair))
    assert len(signatures) == 42
    assert {f["n"] for f in records} == {2, 3}
    assert {tuple(f["layout"]) for f in records} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert len({tuple(f["map_shape"]) for f in records}) >= 20
    assert len({(f["n"], f["motif"]) for f in records}) >= 40
    assert len({(tuple(f["map_shape"]), f["stencil"]) for f in records}) >= 100
    assert len({tuple(f["structural_palette"]) for f in records}) >= 40
    assert len({f["active_cells"] for f in records}) >= 20
    assert {f["active_regions"] for f in records} >= {1, 2}
    assert any(f["wall_components"] >= 2 for f in records)
    assert any(f["exterior_marked"] for f in records)
    assert any(f["unmarked_closed"] for f in records)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(803+nt*10+nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
            validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (None, 1), (4, False), (4, 1.5)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    damaged = copy.deepcopy(generate_puzzle(5, 2, 1))
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    _counterfactual_tests()
    return {"seeds": 42, "pairs": len(records), "map_shapes": len({tuple(f["map_shape"]) for f in records}),
            "motifs": len({(f["n"], f["motif"]) for f in records}),
            "stencils": len({(tuple(f["map_shape"]), f["stencil"]) for f in records}),
            "layouts": 4}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
