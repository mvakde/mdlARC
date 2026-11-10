"""Contour-depth quilting. Only NumPy and the Python standard library are used.

The sample is complete and framed. The other field is a damaged wall map.
Straighten evidenced three-cell gaps, erase diagonal scraps, and expand the map.
Walls become solid blocks; enclosed space receives the binary sample, exchanged
at odd enclosure depths and unchanged at even depths. Exterior stays blank.
"""
from collections import deque
import hashlib

import numpy as np


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _sample(field, wall):
    """Decode a single complete framed tile; all unused strip cells are ground."""
    bg = int(field[0, 0])
    if bg == wall:
        raise ValueError("sample needs a blank margin")
    points = np.argwhere(field != bg)
    if not len(points):
        raise ValueError("missing sample")
    lo, hi = points.min(0), points.max(0) + 1
    r, c = map(int, lo)
    b, d = map(int, hi)
    if not (r > 0 and c > 0 and b < field.shape[0] and d < field.shape[1]):
        raise ValueError("sample margin is incomplete")
    box = field[r:b, c:d]
    if box.shape[0] != box.shape[1] or box.shape[0] not in (4, 5):
        raise ValueError("sample must frame a 2x2 or 3x3 tile")
    if not (np.all(box[0] == wall) and np.all(box[-1] == wall)
            and np.all(box[:, 0] == wall) and np.all(box[:, -1] == wall)):
        raise ValueError("sample has no complete wall-colored frame")
    tile = box[1:-1, 1:-1].copy()
    colors = set(map(int, np.unique(tile)))
    if len(colors) != 2 or colors & {bg, wall}:
        raise ValueError("sample needs two distinct content colors")
    return bg, tile


def _decode(input_grid):
    a = _grid(input_grid)
    candidates = []
    for axis in (0, 1):
        # A transpose here changes field coordinates only. The tile is restored
        # before use, so moving the sample strip never rotates its pattern.
        view = a if axis == 1 else a.T
        for cut in range(1, view.shape[1] - 1):
            wall = int(view[0, cut])
            if not np.all(view[:, cut] == wall):
                continue
            for side in (0, 1):
                strip, drawing = ((view[:, :cut], view[:, cut + 1:]) if side == 0
                                  else (view[:, cut + 1:], view[:, :cut]))
                if not (6 <= strip.shape[1] <= 9 and all(8 <= s <= 15 for s in drawing.shape)):
                    continue
                if axis == 0:
                    strip, drawing = strip.T, drawing.T
                try:
                    bg, tile = _sample(strip, wall)
                except ValueError:
                    continue
                if set(map(int, np.unique(drawing))) != {bg, wall}:
                    continue
                if any(s * len(tile) > 30 for s in drawing.shape):
                    continue
                candidates.append((drawing.copy(), tile, bg, wall, axis, side))
    if len(candidates) != 1:
        raise ValueError("expected one unambiguous sample strip and wall map")
    return candidates[0]


def _at(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _straight(mask):
    supported = np.zeros_like(mask)
    for a, b in ((mask, supported), (mask.T, supported.T)):
        triple = a[:, :-2] & a[:, 1:-1] & a[:, 2:]
        for offset in range(3):
            b[:, offset:offset + triple.shape[1]] |= triple
    return supported


def _gaps(mask):
    found = []
    for dr, dc in ((0, 1), (1, 0)):
        a = mask if dc else mask.T
        gaps = a[:, :-4] & a[:, 4:] & ~(a[:, 1:-3] | a[:, 2:-2] | a[:, 3:-1])
        for y, x in np.argwhere(gaps):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            found.append((r, c, dr, dc))
    return found


def _repair(mask):
    """All evidence is read simultaneously from the visible, unmodified map."""
    strong = _straight(mask)
    added, anchors = np.zeros_like(mask), np.zeros_like(mask)
    evidence = []
    for r, c, dr, dc in _gaps(mask):
        if not (_at(mask, r - dr, c - dc) or _at(mask, r + 5*dr, c + 5*dc)):
            continue
        spurs = []
        for endpoint in (0, 4):
            ar, ac = r + endpoint*dr, c + endpoint*dc
            toward = 1 if endpoint == 0 else -1
            for side in (-1, 1):
                sr, sc = toward*dr + side*dc, toward*dc + side*dr
                cells = [(ar + k*sr, ac + k*sc) for k in (1, 2)]
                if (all(_at(mask, y, x) and not strong[y, x] for y, x in cells)
                        and not _at(mask, ar + 3*sr, ac + 3*sc)):
                    spurs.append(cells)
        if len(spurs) > 1:
            raise ValueError("multiple competing diagonal witnesses")
        if spurs:
            for k in (1, 2, 3):
                added[r + k*dr, c + k*dc] = True
            anchors[r, c] = anchors[r + 4*dr, c + 4*dc] = True
            evidence.append((r, c, dr, dc, tuple(spurs[0])))
    weak = mask & ~_straight(mask | added) & ~anchors
    erase = np.zeros_like(mask)
    for dc in (-1, 1):
        left = weak[:-1, :-1] if dc == 1 else weak[:-1, 1:]
        right = weak[1:, 1:] if dc == 1 else weak[1:, :-1]
        linked = left & right
        if dc == 1:
            erase[:-1, :-1] |= linked
            erase[1:, 1:] |= linked
        else:
            erase[:-1, 1:] |= linked
            erase[1:, :-1] |= linked
    return (mask & ~erase) | added, added, erase, evidence


def _depth(walls):
    """Minimum number of wall cells crossed from the virtual exterior (0-1 BFS)."""
    h, w = walls.shape
    distance = np.full((h, w), 32767, np.int16)
    queue = deque()
    boundary = [(r, c) for r in range(h) for c in range(w)
                if r in (0, h - 1) or c in (0, w - 1)]
    for r, c in boundary:
        distance[r, c] = int(walls[r, c])
        (queue.append if walls[r, c] else queue.appendleft)((r, c))
    while queue:
        r, c = queue.popleft()
        for y, x in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= y < h and 0 <= x < w:
                value = int(distance[r, c]) + int(walls[y, x])
                if value < distance[y, x]:
                    distance[y, x] = value
                    (queue.append if walls[y, x] else queue.appendleft)((y, x))
    return distance


def _paint(walls, tile, bg, wall, behavior="normal"):
    depth = _depth(walls)
    n = len(tile)
    out = np.full((walls.shape[0]*n, walls.shape[1]*n), bg, np.int8)
    colors = list(map(int, np.unique(tile)))
    swapped = (colors[0] + colors[1] - tile).astype(np.int8)
    if behavior == "rotate_tile":
        tile, swapped = np.rot90(tile), np.rot90(swapped)
    for r, c in np.ndindex(walls.shape):
        block = np.s_[r*n:(r+1)*n, c*n:(c+1)*n]
        if walls[r, c]:
            out[block] = wall
        elif depth[r, c] > 0:
            if behavior == "solid":
                out[block] = colors[0]
            else:
                exchange = bool(depth[r, c] % 2)
                if behavior == "no_exchange":
                    exchange = False
                elif behavior == "always_exchange":
                    exchange = True
                out[block] = swapped if exchange else tile
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    drawing, tile, bg, wall, _, _ = _decode(input_grid)
    clean, _, _, _ = _repair(drawing == wall)
    return _paint(clean, tile, bg, wall)


def _rect(shape, top, left, bottom, right):
    a = np.zeros(shape, bool)
    a[top, left:right+1] = a[bottom, left:right+1] = True
    a[top:bottom+1, left] = a[top:bottom+1, right] = True
    return a


def _sites(mask):
    sites = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        for dr, dc in ((0, 1), (1, 0)):
            if not all(_at(mask, r+k*dr, c+k*dc) for k in range(5)):
                continue
            if not (_at(mask, r-dr, c-dc) or _at(mask, r+5*dr, c+5*dc)):
                continue
            cells = [(r+k*dr, c+k*dc) for k in (1, 2, 3)]
            if not any(_at(mask, y+dc, x+dr) or _at(mask, y-dc, x-dr) for y, x in cells):
                sites.append((r, c, dr, dc))
    return sites


def _gate(rng, clean, contour):
    sites = [s for s in _sites(clean)
             if all(contour[s[0]+k*s[2], s[1]+k*s[3]] for k in range(5))]
    if not sites:
        raise ValueError("no room for an intentional gate")
    r, c, dr, dc = sites[int(rng.integers(len(sites)))]
    for k in (1, 2, 3):
        clean[r+k*dr, c+k*dc] = False


def _geometry(rng, mode, n):
    if mode in ("nested_open", "nested_closed", "outer_gate", "archipelago"):
        h, w = map(int, rng.integers(13, 16, size=2))
        rh, rw = int(rng.integers(12, h+1)), int(rng.integers(12, w+1))
        if rh == h and rw == w:
            rh -= 1  # Keep visible exterior floor in the two core witnesses.
        t, l = int(rng.integers(h-rh+1)), int(rng.integers(w-rw+1))
        b, r = t+rh-1, l+rw-1
        outer = _rect((h, w), t, l, b, r)
        inner = _rect((h, w), t+3, l+3, b-3, r-3)
        if mode == "archipelago":
            inner = np.zeros((h, w), bool)
            wanted = int(rng.integers(2, 5))
            placed = 0
            for _ in range(48):
                ih, iw = int(rng.integers(3, 6)), int(rng.integers(3, 6))
                y = int(rng.integers(t+2, b-ih))
                x = int(rng.integers(l+2, r-iw))
                if not inner[y-1:y+ih+1, x-1:x+iw+1].any():
                    inner |= _rect((h, w), y, x, y+ih-1, x+iw-1)
                    placed += 1
                    if placed == wanted:
                        break
        clean = outer | inner
        if mode == "nested_open":
            _gate(rng, clean, inner)
        elif mode == "outer_gate":
            _gate(rng, clean, outer)
        target = inner if mode == "outer_gate" else outer
    elif mode == "islands":
        h, w = int(rng.integers(8, 16)), int(rng.integers(13, 16))
        cut = int(rng.integers(6, w-5))
        a = _rect((h, w), int(rng.integers(2)), 0, h-1, cut-1)
        b = _rect((h, w), 0, cut+1, h-1-int(rng.integers(2)), w-1)
        clean, target = a | b, a
        if rng.random() < .5:
            _gate(rng, clean, b)
    else:
        low, high = ((10, 11) if n == 3 else (10, 16))
        h, w = map(int, rng.integers(low, high, size=2))
        t, l = int(rng.integers(2)), int(rng.integers(2))
        b, r = h-1, w-1
        clean = _rect((h, w), t, l, b, r)
        target = clean.copy()
        if mode == "rooms":
            mid = int(rng.integers(l+4, r-3))
            clean[t:b+1, mid] = True
            if rng.random() < .6:
                level = int(rng.integers(t+3, b-2))
                # A T-junction or complete cross produces three or four rooms.
                if rng.random() < .5:
                    clean[level, l:mid+1] = True
                else:
                    clean[level, l:r+1] = True
        elif mode == "notch":
            # Remove a corner, replacing its two sides with an inset elbow.
            dy = int(rng.integers(2, min(4, b-t-3)+1))
            dx = int(rng.integers(2, min(4, r-l-3)+1))
            clean[b, l:l+dx] = False
            clean[b-dy+1:b+1, l] = False
            clean[b-dy, l:l+dx+1] = True
            clean[b-dy:b+1, l+dx] = True
            target = clean.copy()
        elif mode != "single":
            raise ValueError("unknown topology")
    # Optional open branches vary the scaffold without creating another room.
    for _ in range(int(rng.integers(4))):
        for _ in range(12):
            points = np.argwhere(clean)
            y, x = map(int, points[int(rng.integers(len(points)))])
            dy, dx = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
            cells = [(y+k*dy, x+k*dx) for k in range(1, int(rng.integers(3, 6)))]
            if not all(0 <= a < clean.shape[0] and 0 <= b < clean.shape[1]
                       and not clean[a, b] for a, b in cells):
                continue
            if any(_at(clean, a+dr, b+dc) and (a+dr, b+dc) != (y, x)
                   for a, b in cells for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                continue
            trial = clean.copy()
            for a, b in cells:
                trial[a, b] = True
            if (trial[:-1, :-1] & trial[1:, :-1] & trial[:-1, 1:] & trial[1:, 1:]).any():
                continue
            floor = ~trial
            if np.array_equal(_depth(trial)[floor], _depth(clean)[floor]):
                clean = trial
                break
    # Map orientation varies independently of the tile's absolute orientation.
    turns = int(rng.integers(4))
    clean, target = np.rot90(clean, turns).copy(), np.rot90(target, turns).copy()
    if rng.integers(2):
        clean, target = clean[:, ::-1].copy(), target[:, ::-1].copy()
    return clean, target


def _damage(rng, clean, target):
    inp = clean.copy()
    wanted = int(rng.integers(1, 4))
    repairs = 0
    for _ in range(wanted):
        candidates = []
        for r, c, dr, dc in _sites(clean):
            if repairs == 0 and not all(target[r+k*dr, c+k*dc] for k in range(5)):
                continue
            for endpoint in (0, 4):
                ar, ac = r+endpoint*dr, c+endpoint*dc
                toward = 1 if endpoint == 0 else -1
                for side in (-1, 1):
                    spur = [(ar+k*(toward*dr+side*dc), ac+k*(toward*dc+side*dr)) for k in (1, 2)]
                    gap = [(r+k*dr, c+k*dc) for k in (1, 2, 3)]
                    if (all(_at(inp, y, x) for y, x in gap)
                            and all(0 <= y < clean.shape[0] and 0 <= x < clean.shape[1]
                                    and not inp[y, x] and not clean[y, x] for y, x in spur)):
                        candidates.append((gap, spur))
        success = False
        for idx in rng.permutation(len(candidates)):
            gap, spur = candidates[int(idx)]
            trial = inp.copy()
            for y, x in gap:
                trial[y, x] = False
            for y, x in spur:
                trial[y, x] = True
            try:
                restored, _, _, evidence = _repair(trial)
            except ValueError:
                continue
            if np.array_equal(restored, clean) and len(evidence) == repairs+1:
                inp, repairs, success = trial, repairs+1, True
                break
        if not success:
            break
    if not repairs:
        raise ValueError("could not displace a supported wall segment")
    # Optional free diagonal scraps must neither attach a false witness nor
    # alter the intended drawing. No logs or traces are retained per attempt.
    for _ in range(int(rng.integers(3))):
        length = int(rng.integers(2, 4))
        for _ in range(16):
            y, x = int(rng.integers(clean.shape[0])), int(rng.integers(clean.shape[1]))
            dy, dx = int(rng.choice((-1, 1))), int(rng.choice((-1, 1)))
            cells = [(y+k*dy, x+k*dx) for k in range(length)]
            if not all(0 <= a < clean.shape[0] and 0 <= b < clean.shape[1]
                       and not inp[a, b] and not clean[a, b] for a, b in cells):
                continue
            trial = inp.copy()
            for a, b in cells:
                trial[a, b] = True
            try:
                restored = _repair(trial)[0]
            except ValueError:
                continue
            if np.array_equal(restored, clean):
                inp = trial
                break
    return inp


def _tile(rng, n, colors, witness=False):
    for _ in range(32):
        bits = rng.integers(0, 2, size=(n, n), dtype=np.int8)
        if (bits.any() and not bits.all() and not np.array_equal(bits, np.rot90(bits))
                and (not witness or 2*int(bits.sum()) != bits.size)):
            return np.where(bits, colors[0], colors[1]).astype(np.int8)
    bits = np.zeros((n, n), bool)
    bits[0, 0] = True
    return np.where(bits, colors[0], colors[1]).astype(np.int8)


def _pack(rng, drawing, tile, bg, wall, layout):
    n, h, w = len(tile), *drawing.shape
    framed = np.full((n+2, n+2), wall, np.int8)
    framed[1:-1, 1:-1] = tile
    thickness = int(rng.integers(n+4, 10))
    horizontal = layout in (0, 2)
    strip = np.full((h, thickness) if horizontal else (thickness, w), bg, np.int8)
    y = int(rng.integers(1, strip.shape[0]-n-2))
    x = int(rng.integers(1, strip.shape[1]-n-2))
    strip[y:y+n+2, x:x+n+2] = framed
    divider = np.full((h, 1) if horizontal else (1, w), wall, np.int8)
    parts = [strip, divider, drawing] if layout in (0, 1) else [drawing, divider, strip]
    return np.concatenate(parts, axis=1 if horizontal else 0)


def _make_pair(rng, mode, n, palette, layout, witness=False):
    bg, wall, a, b = map(int, palette)
    for _ in range(24):
        clean, target = _geometry(rng, mode, n)
        try:
            raw = _damage(rng, clean, target)
            if not np.array_equal(_repair(raw)[0], clean):
                continue
            depths = _depth(clean)
            common_floor = ~(clean | raw)
            if not np.any(common_floor & (depths != _depth(raw))):
                continue
            if mode == "nested_closed" and int(depths[~clean].max()) < 2:
                continue
            tile = _tile(rng, n, (a, b), witness)
            drawing = np.where(raw, wall, bg).astype(np.int8)
            inp = _pack(rng, drawing, tile, bg, wall, layout)
            out = _paint(clean, tile, bg, wall)
            if not np.array_equal(solve(inp), out):
                continue
            return {"input": inp, "output": out}
        except ValueError:
            continue
    raise RuntimeError(f"bounded construction failed for {mode}")


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(3))
    base_palette = rng.choice(10, 4, replace=False)
    palette_style = int(rng.integers(3))
    # First two witnesses are always available, even for the (2,1) request.
    layouts = list(map(int, rng.permutation(4)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            if split == "train" and i < 2:
                mode, n = ("nested_open" if i == 0 else "nested_closed"), 2
            elif split == "test":
                mode = ("outer_gate", "rooms", "notch")[shift]
                n = 3 if shift == 2 else 2
            else:
                mode = str(rng.choice(("nested_open", "nested_closed", "single", "islands", "archipelago")))
                n = 3 if mode == "single" and rng.random() < .5 else 2
            palette = (rng.choice(10, 4, replace=False) if palette_style == 2 else base_palette.copy())
            if palette_style == 1:
                palette = palette[rng.permutation(4)]
            layout = layouts[i % 4] if split == "train" else int(rng.integers(4))
            for _ in range(16):
                pair = _make_pair(rng, mode, n, palette, layout, split == "train" and i < 2)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded unique-input construction failed")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    drawing, tile, bg, wall, axis, side = _decode(inp)
    raw = drawing == wall
    clean, added, erased, evidence = _repair(raw)
    expected = _paint(clean, tile, bg, wall)
    assert np.array_equal(out, expected), "incorrect quilt or incomplete wall repair"
    assert evidence and int(added.sum()) == 3*len(evidence)
    assert erased.any(), "diagonal evidence must be removed"
    assert not (clean[:-1, :-1] & clean[1:, :-1] & clean[:-1, 1:] & clean[1:, 1:]).any()
    depth = _depth(clean)
    floors = ~clean
    assert np.any(floors & (depth == 1)), "must show binary exchange"
    assert np.any(~(raw | clean) & (depth != _depth(raw))), "repair must affect region topology"
    # Validate each n*n block: uniform wall, blank exterior, or the complete
    # original/exchanged sample. Both sample colors are written in every room.
    n = len(tile)
    assert out.shape == (drawing.shape[0]*n, drawing.shape[1]*n)
    for r, c in np.ndindex(clean.shape):
        block = out[r*n:(r+1)*n, c*n:(c+1)*n]
        if clean[r, c]:
            assert np.all(block == wall)
        elif not depth[r, c]:
            assert np.all(block == bg)
        else:
            assert set(map(int, np.unique(block))) == set(map(int, np.unique(tile)))
    fill_all = clean.copy()
    for r, c, dr, dc in _gaps(clean):
        for k in (1, 2, 3):
            fill_all[r+k*dr, c+k*dc] = True
    rejected = {behavior: not np.array_equal(out, _paint(clean, tile, bg, wall, behavior))
                for behavior in ("no_exchange", "always_exchange", "solid", "rotate_tile")}
    rejected["fill_all_gaps"] = not np.array_equal(out, _paint(fill_all, tile, bg, wall))
    rejected["no_repair"] = not np.array_equal(out, _paint(clean & ~added, tile, bg, wall))
    return {"map_shape": drawing.shape, "output_shape": out.shape, "tile_size": n,
            "depth": int(depth[floors].max()), "repairs": len(evidence),
            "exterior_floor": bool(np.any(floors & (depth == 0))),
            "axes": tuple(sorted({e[2:4] for e in evidence})),
            "layout": (axis, side), "gates": len(_gaps(clean)),
            "palette": (bg, wall, *map(int, np.unique(tile))), "rejected": rejected}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze(pair)
            key = _key(pair["input"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            if split == "train":
                training.append(info)
    assert any(x["depth"] >= 2 for x in training), "missing nested-depth witness"
    assert any(x["exterior_floor"] for x in training), "missing blank-exterior witness"
    for alternative in training[0]["rejected"]:
        assert any(x["rejected"][alternative] for x in training), f"training admits {alternative}"


def self_test():
    """Streaming 42-seed checks; no files, traces, or unbounded retained runs."""
    signatures, shapes, maps, palettes, layouts, sizes, repairs, axes = (set() for _ in range(8))
    depth_values, shifts, structures = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair)
                shapes.add(info["output_shape"])
                maps.add(info["map_shape"])
                palettes.add(info["palette"])
                layouts.add(info["layout"])
                sizes.add(info["tile_size"])
                repairs.add(info["repairs"])
                axes.update(info["axes"])
                depth_values.add(info["depth"])
                drawing, _, bg, wall, _, _ = _decode(pair["input"])
                structures.add((drawing.shape, (drawing == wall).tobytes()))
                for role in ("input", "output"):
                    digest.update(repr(pair[role].shape).encode())
                    digest.update(pair[role].tobytes())
        signatures.add(digest.hexdigest())
        shifts.add((puzzle["test"][0]["output"].shape,
                    _analyze(puzzle["test"][0])["tile_size"]))
    assert len(signatures) == 42 and len(structures) >= 100
    assert len(shapes) >= 15 and len(maps) >= 15 and len(palettes) >= 30
    assert len(layouts) == 4 and sizes == {2, 3}
    assert {1, 2, 3}.issubset(repairs) and axes == {(1, 0), (0, 1)}
    assert {1, 2}.issubset(depth_values) and len(shifts) >= 5
    p, same = generate_puzzle(719, 2, 1), generate_puzzle(719, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], same[s]) for k in x)
    for nt in range(2, 11):
        for nq in range(1, min(nt-1, 4)+1):
            q = generate_puzzle(83, nt, nq)
            assert len(q["train"]) == nt and len(q["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    # Omit an entire repair's label, including its effect on distant rooms.
    pair = bad["test"][0]
    drawing, tile, bg, wall, _, _ = _decode(pair["input"])
    clean, added, _, _ = _repair(drawing == wall)
    pair["output"] = _paint(clean & ~added, tile, bg, wall)
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing repair and resulting topology change accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, all legal counts, topology and ablation checks")
