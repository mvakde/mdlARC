"""Counter-locked inlay: local counted rotations and global cavity exact cover.

Only NumPy and the Python standard library are used. The public solver reads
the input alone; it never consults the generator, a seed, or a stored example.
"""

from collections import deque
import json

import numpy as np


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _check_counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        queue, cells = [first], []
        while queue:
            r, c = queue.pop()
            cells.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    queue.append(q)
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _crop(mask):
    cells = np.argwhere(mask)
    lo, hi = cells.min(0), cells.max(0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _mask(cells):
    cells = cells - cells.min(0)
    result = np.zeros(tuple(cells.max(0) + 1), dtype=bool)
    result[cells[:, 0], cells[:, 1]] = True
    return result


def _holes(wall):
    """Flood through every non-wall cell, ignoring detached objects entirely."""
    h, w = wall.shape
    exterior = np.zeros_like(wall)
    queue = deque()
    for r in range(h):
        for c in (0, w-1):
            if not wall[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h-1):
            if not wall[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= y < h and 0 <= x < w and not wall[y, x] and not exterior[y, x]:
                exterior[y, x] = True
                queue.append((y, x))
    return ~wall & ~exterior


def _dilate(mask):
    p = np.pad(mask, 1)
    h, w = mask.shape
    result = np.zeros_like(mask)
    for r in range(3):
        for c in range(3):
            result |= p[r:r+h, c:c+w]
    return result


def _parse(value):
    inp = _grid(value)
    nums = np.bincount(inp.ravel(), minlength=10)
    if np.count_nonzero(nums == nums.max()) != 1:
        raise ValueError("background must be the unique most frequent input color")
    bg = int(nums.argmax())
    components = [(cells, int(color)) for color in np.unique(inp) if color != bg
                  for cells in _components(inp == color)]
    bands = [(i, axis) for i, (cells, _) in enumerate(components) for axis in (0, 1)
             if len(np.unique(cells[:, axis])) == inp.shape[axis]]
    if len(bands) != 1:
        raise ValueError("expected exactly one monochrome component spanning one axis")
    band_index, span_axis = bands[0]
    band_cells, band = components[band_index]
    if sum(color == band for _, color in components) != 1:
        raise ValueError("structural color may occur only in the band")
    wall = inp == band
    holes = _holes(wall)
    if not np.any(holes) or not np.all(inp[holes] == bg):
        raise ValueError("band must enclose empty cavities")
    pieces, bars = [], {}
    blocked = _dilate(wall | holes)
    for i, (cells, color) in enumerate(components):
        if i == band_index:
            continue
        if np.any(blocked[cells[:, 0], cells[:, 1]]):
            raise ValueError("detached objects must be separated by an empty cell")
        local = _mask(cells)
        if min(local.shape) == 1:
            if not np.all(local) or color in bars or len(cells) > 9:
                raise ValueError("one contiguous bar of length 1..9 per marked color")
            bars[color] = {"cells": cells, "length": len(cells),
                           "orientation": "single" if len(cells) == 1 else
                           ("horizontal" if local.shape[0] == 1 else "vertical")}
        else:
            pieces.append({"cells": cells, "mask": local, "color": color})
    # Pairwise separation includes different colors and diagonally adjacent cells.
    detached = [cells for i, (cells, _) in enumerate(components) if i != band_index]
    occupied = np.zeros_like(wall)
    for cells in detached:
        if np.any(_dilate(occupied)[cells[:, 0], cells[:, 1]]):
            raise ValueError("detached objects touch")
        occupied[cells[:, 0], cells[:, 1]] = True
    colors = {p["color"] for p in pieces}
    if not pieces or not bars or not set(bars) <= colors:
        raise ValueError("counters must match movable pieces")
    if sum(int(p["mask"].sum()) for p in pieces) != int(holes.sum()):
        raise ValueError("piece area must equal the entire enclosed empty area")
    return {"input": inp, "background": bg, "band": band, "wall": wall,
            "holes": holes, "cavities": _components(holes), "pieces": pieces,
            "bars": bars, "span_axis": span_axis}


def _assemblies(info, override=None):
    """Enumerate translations of prescribed orientations into all enclosed cells.

    Exact cover uses each component once and every cavity cell once. Distinct
    colored outputs are counted, so interchangeable same-color pieces are legal.
    No layer/chain grammar, hole assignment, or intended placement is assumed.
    """
    holes, pieces = info["holes"], info["pieces"]
    hole_cells = np.argwhere(holes)
    indices = {tuple(p): i for i, p in enumerate(hole_cells.tolist())}
    h, w = holes.shape
    by_piece = [[] for _ in pieces]
    by_cell = [[] for _ in hole_cells]
    for i, piece in enumerate(pieces):
        color = piece["color"]
        count = info["bars"].get(color, {}).get("length", 0)
        if override is not None:
            count = override.get(color, count)
        rotated = np.rot90(piece["mask"], -count % 4)
        cells = np.argwhere(rotated)
        ph, pw = rotated.shape
        if ph > h or pw > w:
            continue
        valid = np.ones((h-ph+1, w-pw+1), dtype=bool)
        for r, c in cells:
            valid &= holes[r:r+h-ph+1, c:c+w-pw+1]
        for origin in np.argwhere(valid):
            target = cells + origin
            ids = [indices[tuple(p)] for p in target.tolist()]
            bits = sum(1 << j for j in ids)
            candidate = (i, bits, target)
            by_piece[i].append(candidate)
            for j in ids:
                by_cell[j].append(candidate)
    full_pieces = (1 << len(pieces)) - 1
    full_cells = (1 << len(hole_cells)) - 1
    answers = {}

    def search(used, occupied, chosen):
        if len(answers) >= 2:
            return
        if used == full_pieces:
            if occupied != full_cells:
                return
            out = np.full_like(info["input"], info["background"])
            out[info["wall"]] = info["band"]
            for i, _, cells in chosen:
                out[cells[:, 0], cells[:, 1]] = pieces[i]["color"]
            answers.setdefault(out.tobytes(), (out, chosen.copy()))
            return
        options = None
        # Both object and cell constraints participate in the MRV choice.
        for i in range(len(pieces)):
            if not used & (1 << i):
                available = [c for c in by_piece[i] if not occupied & c[1]]
                if not available:
                    return
                if options is None or len(available) < len(options):
                    options = available
        pending = full_cells ^ occupied
        while pending:
            bit = pending & -pending
            j = bit.bit_length() - 1
            pending ^= bit
            available = [c for c in by_cell[j]
                         if not used & (1 << c[0]) and not occupied & c[1]]
            if not available:
                return
            if len(available) < len(options):
                options = available
                if len(options) == 1:
                    break
        for candidate in options:
            i, bits, _ = candidate
            search(used | (1 << i), occupied | bits, chosen + [candidate])

    search(0, 0, [])
    return list(answers.values())


def _analyze(value):
    info = _parse(value)
    solutions = _assemblies(info)
    if len(solutions) != 1:
        raise ValueError("prescribed rotations must give one distinct colored exact cover")
    out, placements = solutions[0]
    info["placements"] = placements
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Count each bar, locally rotate matching components clockwise, and inlay."""
    return _analyze(input_grid)[0].copy()


def _asymmetric(mask):
    return all(_key(mask) != _key(np.rot90(mask, k)) for k in (1, 2, 3))


def _base_unit(rng, count, kind, small=False):
    """A toothed cavity partitioned by profiles or connected region growth."""
    for _ in range(100):
        low = max(4, 2*count)
        h = int(rng.integers(low, 8 if count == 3 else 7))
        w = int(rng.integers(4, 7))
        if small:
            h = w = int(rng.integers(4, 7))
        amplitude = int(rng.integers(1, min(2, h-3)+1))
        top = rng.integers(0, amplitude+1, size=w)
        top[int(rng.integers(w))] = 0
        top[int(rng.integers(w))] = amplitude
        if not top.any():
            continue
        bottom = h - (rng.integers(0, 2, size=w) if rng.random() < .4 else np.zeros(w, dtype=int))
        cavity = np.zeros((h, w), dtype=bool)
        for c in range(w):
            cavity[top[c]:bottom[c], c] = True
        if kind == "growth":
            cells = np.argwhere(cavity)
            starts = cells[rng.choice(len(cells), count, replace=False)]
            owner = np.full((h, w), -1, dtype=int)
            for i, p in enumerate(starts):
                owner[tuple(p)] = i
            remaining = int(cavity.sum()) - count
            while remaining:
                frontier = []
                for r, c in np.argwhere(cavity & (owner < 0)):
                    near = {owner[y, x] for y, x in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                            if 0 <= y < h and 0 <= x < w and owner[y,x] >= 0}
                    frontier.extend((int(r), int(c), int(i)) for i in sorted(near))
                r, c, i = frontier[int(rng.integers(len(frontier)))]
                owner[r, c] = i
                remaining -= 1
            masks = [owner == i for i in range(count)]
        else:
            cuts = [top]
            for i in range(1, count):
                middle = int(round(h * i / count))
                cuts.append(np.clip(middle + rng.integers(-1, 2, size=w),
                                    int(cuts[-1].max())+1, int(bottom.min())-(count-i)))
            cuts.append(bottom)
            masks = []
            for lo, hi in zip(cuts, cuts[1:]):
                m = np.zeros((h, w), dtype=bool)
                for c in range(w):
                    m[lo[c]:hi[c], c] = True
                masks.append(m)
        if (all(m.sum() >= 5 and min(_crop(m).shape) >= 2
                and len(_components(m)) == 1
                and (not small or _asymmetric(_crop(m))) for m in masks)
                and _asymmetric(cavity)):
            return cavity, masks
    raise ValueError("could not make connected non-bar asymmetric pieces")


def _palette(rng):
    colors = [int(x) for x in rng.permutation(10)]
    return colors[0], colors[1], colors[2:]


def _scatter(rng, base, bg, objects):
    """Pack pieces and controls together without a positional control code."""
    h, w = base.shape
    for _ in range(35):
        inp = base.copy()
        forbidden = _dilate((base != bg) | _holes(base != bg))
        order = sorted(range(len(objects)), key=lambda i: -int(objects[i][0].sum()) + rng.random())
        okay = True
        for i in order:
            mask, color, bar = objects[i]
            if bar and rng.random() < .5:
                mask = mask.T
            ph, pw = mask.shape
            if ph > h or pw > w:
                okay = False
                break
            cells = np.argwhere(mask)
            valid = np.ones((h-ph+1, w-pw+1), dtype=bool)
            for r, c in cells:
                valid &= ~forbidden[r:r+h-ph+1, c:c+w-pw+1]
            origins = np.argwhere(valid)
            if not len(origins):
                okay = False
                break
            target = cells + origins[int(rng.integers(len(origins)))]
            inp[target[:, 0], target[:, 1]] = color
            placed = np.zeros_like(forbidden)
            placed[target[:, 0], target[:, 1]] = True
            forbidden |= _dilate(placed)
        if okay:
            return inp
    raise ValueError("scatter exhausted")


def _twin_witness(info):
    """A bar-color swap must change a uniquely soluble answer, geometry fixed."""
    pieces = info["pieces"]
    for i, a in enumerate(pieces):
        for b in pieces[:i]:
            ca, cb = a["color"], b["color"]
            if ca == cb or _key(a["mask"]) != _key(b["mask"]):
                continue
            ka = info["bars"].get(ca, {}).get("length", 0)
            kb = info["bars"].get(cb, {}).get("length", 0)
            if ka % 4 == kb % 4 or not ka or not kb:
                continue
            alternatives = _assemblies(info, {ca: kb, cb: ka})
            original = _assemblies(info)
            if len(alternatives) == 1 and len(original) == 1:
                if not np.array_equal(alternatives[0][0], original[0][0]):
                    return ca, cb
    return None


def _nearest_trap(info):
    centers = [cells.mean(0) for cells in info["cavities"]]
    for i, _, target in info["placements"]:
        source_center = info["pieces"][i]["cells"].mean(0)
        distances = [float(np.abs(source_center - center).sum()) for center in centers]
        minimum = min(distances)
        closest = [j for j, d in enumerate(distances) if abs(d-minimum) < 1e-8]
        actual = next(j for j, cells in enumerate(info["cavities"])
                      if tuple(target[0]) in set(map(tuple, cells.tolist())))
        if len(closest) == 1 and closest[0] != actual:
            return True
    return False


def _make_pair(rng, witness, mode, global_turn):
    bg, band, palette = _palette(rng)
    units, payload, lengths = [], [], []
    if witness in (0, 1):
        cavity, masks = _base_unit(rng, 2, "profile", small=True)
        # Two congruent units. Their first pieces are identical before control;
        # their target phases differ, creating a real color-assignment ambiguity
        # when counters are ignored and arbitrary rotations are allowed.
        phases = (1, 3) if witness == 0 else (0, 2)
        raw = (1, 2, 3, 2) if witness == 0 else (0, 4, 2, 1)
        if witness == 1 and mode == 1:
            raw = (0, 8, 6, 5)
        for u, phase in enumerate(phases):
            units.append(np.rot90(cavity, -phase))
            for j, mask in enumerate(masks):
                payload.append((u, np.rot90(mask, -phase), palette[len(payload)]))
                lengths.append(raw[len(lengths)])
    else:
        unit_count = int(rng.choice([1, 2, 3], p=[.25, .55, .2]))
        total = 0
        for u in range(unit_count):
            if mode == 1:
                # A held-out three-piece connected partition is the shift in
                # this regime; long counters have already been demonstrated.
                count = 3 if witness == 2 and u == 0 else int(rng.choice([1, 2]))
            else:
                count = int(rng.choice([1, 2, 3], p=[.15, .6, .25]))
            count = min(count, 6-total-(unit_count-u-1))
            kind = ("growth" if witness == 2 else "profile") if mode == 1 else (
                "growth" if rng.random() < .55 else "profile")
            cavity, masks = _base_unit(rng, count, kind)
            phase = int(rng.integers(4))
            units.append(np.rot90(cavity, -phase))
            for mask in masks:
                payload.append((u, np.rot90(mask, -phase), palette[len(payload)]))
                lengths.append(int(rng.integers(0, 10 if mode == 1 else 5)))
            total += count
        if witness == 2 and mode == 2:
            lengths[int(rng.integers(len(lengths)))] = int(rng.integers(5, 10))
        if not any(lengths):
            lengths[int(rng.integers(len(lengths)))] = int(rng.integers(1, 5))
        # A counter controls every component of its color. Reuse is allowed
        # only with identical raw counts, so the input remains self-contained.
        for i in range(1, len(payload)):
            same = [j for j in range(i) if lengths[j] == lengths[i]]
            if same and rng.random() < .4:
                u, mask, _ = payload[i]
                payload[i] = (u, mask, payload[int(rng.choice(same))][2])
    maximum_height = max(m.shape[0] for m in units)
    gap = int(rng.integers(1, 4))
    minimum_width = max(15, sum(m.shape[1] for m in units) + gap*(len(units)-1) + 4)
    if minimum_width > 30:
        raise ValueError("units too wide")
    width = int(rng.integers(minimum_width, 31))
    band_height = maximum_height + 6
    source_masks = [_crop(np.rot90(mask, length % 4))
                    for (_, mask, _), length in zip(payload, lengths)]
    max_box = max(max(m.shape) for m in source_masks)
    minimum_height = max(20, band_height+2*max_box+2)
    if minimum_height > 30:
        raise ValueError("units too tall to scatter")
    height = int(rng.integers(minimum_height, 31))
    low = max_box+1
    band_row = int(rng.integers(low, height-band_height-low+1))
    base = np.full((height, width), bg, dtype=np.int8)
    top_detail = rng.integers(0, 3, size=width)
    bottom_detail = rng.integers(0, 3, size=width)
    for c in range(width):
        base[band_row+top_detail[c]:band_row+band_height-bottom_detail[c], c] = band
    total_width = sum(m.shape[1] for m in units)+gap*(len(units)-1)
    col = int(rng.integers(2, width-total_width-1))
    origins = []
    for unit in units:
        row = band_row + int(rng.integers(3, band_height-unit.shape[0]-2))
        cells = np.argwhere(unit) + (row, col)
        base[cells[:, 0], cells[:, 1]] = bg
        origins.append(np.array((row, col)))
        col += unit.shape[1] + gap
    output = base.copy()
    objects = []
    counts = {}
    for (u, mask, color), length, source in zip(payload, lengths, source_masks):
        cells = np.argwhere(mask) + origins[u]
        output[cells[:, 0], cells[:, 1]] = color
        objects.append((source, color, False))
        counts[color] = length
    for color, length in counts.items():
        if length:
            objects.append((np.ones((1, length), dtype=bool), color, True))
    inp = _scatter(rng, base, bg, objects)
    inp, output = np.rot90(inp, -global_turn).copy(), np.rot90(output, -global_turn).copy()
    predicted, info = _analyze(inp)
    if not np.array_equal(predicted, output):
        raise ValueError("construction differs from exact cover")
    if witness == 0:
        orientations = {m["orientation"] for m in info["bars"].values() if m["length"] == 2}
        if orientations != {"horizontal", "vertical"} or not _nearest_trap(info) or not _twin_witness(info):
            raise ValueError("first demonstration needs orientation twins, mixed bars, and a proximity trap")
    return {"input": inp, "output": output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    # 0: orientation transfer; 1: long controls demonstrated, new groupings;
    # 2: shared geometry, control-length extrapolation after a full-cycle example.
    mode = int(rng.integers(3))
    train_parity = int(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = index if split == "train" and index < 2 else (2 if split == "test" else -1)
            if mode == 0:
                parity = train_parity if split == "train" else 1-train_parity
                global_turn = parity + 2*int(rng.integers(2))
            else:
                global_turn = int(rng.integers(4))
            for _ in range(180):
                try:
                    pair = _make_pair(rng, witness, mode, global_turn)
                except ValueError:
                    continue
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded construction could not produce a unique inlay")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _coordinate_rotation(mask, turns):
    """Coordinate check, separate from generation's NumPy array rotations."""
    cells = np.argwhere(mask)
    h, w = mask.shape
    for _ in range(turns % 4):
        cells = np.column_stack((cells[:, 1], h-1-cells[:, 0]))
        h, w = w, h
    return _mask(cells)


def _rotation_only(info):
    """Ablation: rotate at each source bounding-box origin, omit all fitting.

    None denotes a clipped/overlapping invalid result, which also cannot be the
    valid inlay. This diagnostic is never used by solve or by construction.
    """
    result = np.full_like(info["input"], info["background"])
    result[info["wall"]] = info["band"]
    for piece in info["pieces"]:
        k = info["bars"].get(piece["color"], {}).get("length", 0)
        cells = np.argwhere(_coordinate_rotation(piece["mask"], k)) + piece["cells"].min(0)
        if np.any(cells >= np.array(result.shape)):
            return None
        if np.any(result[cells[:, 0], cells[:, 1]] != info["background"]):
            return None
        result[cells[:, 0], cells[:, 1]] = piece["color"]
    return result


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must have exactly train and test keys")
    if not all(isinstance(puzzle[s], list) for s in puzzle):
        raise ValueError("splits must be lists")
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise ValueError("pair must have exactly input and output keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            if _key(inp) in seen:
                raise ValueError("inputs must be distinct")
            seen.add(_key(inp))
            expected, info = _analyze(inp)
            if not np.array_equal(expected, out):
                raise ValueError("output differs from unique inlay")
            if inp.shape != out.shape or np.array_equal(inp, out):
                raise ValueError("inlay must move objects on the same canvas")
            if not np.array_equal(inp == info["band"], out == info["band"]):
                raise ValueError("band was altered")
            if np.any(out[info["holes"]] == info["background"]):
                raise ValueError("unfilled enclosed cell")
            if np.any(out[~(info["holes"] | info["wall"])] != info["background"]):
                raise ValueError("exterior objects must be erased")
            for color in {p["color"] for p in info["pieces"]}:
                lost = info["bars"].get(color, {}).get("length", 0)
                if int((inp == color).sum()) - int((out == color).sum()) != lost:
                    raise ValueError("payload area/color is not conserved")
            for i, _, target in info["placements"]:
                piece = info["pieces"][i]
                k = info["bars"].get(piece["color"], {}).get("length", 0)
                if not np.array_equal(_coordinate_rotation(piece["mask"], k), _mask(target)):
                    raise ValueError("component rotation/translation is incorrect")
            if split == "train":
                training.append(info)
    # Visible evidence is checked rather than trusted from construction flags.
    if not any(_twin_witness(i) and _nearest_trap(i) for i in training):
        raise ValueError("training must demonstrate both control and geometric matching")
    lengths, unmarked, full_cycle, clockwise = set(), False, False, False
    orientation_independent, independently_counted = False, False
    for info in training:
        orientations = {}
        residues = set()
        for piece in info["pieces"]:
            if not _asymmetric(piece["mask"]):
                continue
            k = info["bars"].get(piece["color"], {}).get("length", 0)
            lengths.add(k)
            unmarked |= k == 0
            full_cycle |= k > 0 and k % 4 == 0
            clockwise |= k % 2 == 1
            residues.add(k % 4)
            if k > 1:
                orientations.setdefault(k, set()).add(info["bars"][piece["color"]]["orientation"])
        independently_counted |= len(residues) >= 2
        orientation_independent |= any(v == {"horizontal", "vertical"} for v in orientations.values())
    if not ({1, 2, 3} <= lengths and unmarked and full_cycle and clockwise
            and independently_counted and orientation_independent):
        raise ValueError("training must show short clockwise counts, absence, and a full cycle")


def _features(pair):
    _, info = _analyze(pair["input"])
    pieces, bars = info["pieces"], info["bars"]
    cavity_sets = [set(map(tuple, cells.tolist())) for cells in info["cavities"]]
    groups = [sum(tuple(target[0]) in cells for _, _, target in info["placements"])
              for cells in cavity_sets]
    return {"shape": list(info["input"].shape), "axis": info["span_axis"],
            "background": info["background"], "band": info["band"],
            "cavities": len(info["cavities"]), "pieces": len(pieces),
            "groups": groups,
            "areas": [int(p["mask"].sum()) for p in pieces],
            "boxes": [list(p["mask"].shape) for p in pieces],
            "counts": sorted(m["length"] for m in bars.values()),
            "orientations": sorted({m["orientation"] for m in bars.values()}),
            "unmarked": sum(p["color"] not in bars for p in pieces),
            "reuse": len({p["color"] for p in pieces}) < len(pieces),
            "nonconvex": any(any(np.any(np.diff(np.flatnonzero(row)) > 1) for row in p["mask"])
                             and any(np.any(np.diff(np.flatnonzero(col)) > 1) for col in p["mask"].T)
                             for p in pieces)}


def self_test():
    """Bounded checks over 42 seeds plus API extremes and causal mutations."""
    # A connected C-shaped cavity surrounds a wall peninsula. It is outside
    # the sampler's toothed-rectangle outline grammar. Same-color twin pieces
    # permit two component permutations but only one colored output.
    target = np.array([[1,1,1,1,1], [1,0,0,0,1], [0,0,0,0,1],
                       [1,0,0,0,1], [1,1,1,1,1]], dtype=bool)
    source = np.rot90(target, 1)
    for copies in (1, 2):
        fixture = np.zeros((20, 22), dtype=np.int8)
        fixture[8:15, :] = 8
        expected = fixture.copy()
        for i in range(copies):
            cells = np.argwhere(target) + (9, 3+10*i)
            fixture[cells[:, 0], cells[:, 1]] = 0
            expected[cells[:, 0], cells[:, 1]] = 2
            cells = np.argwhere(source) + (0, 1+8*i)
            fixture[cells[:, 0], cells[:, 1]] = 2
        fixture[18, 12:21] = 2  # Nine cells: one clockwise quarter-turn.
        saved = fixture.copy()
        for turns in range(4):
            assert np.array_equal(solve(np.rot90(fixture, turns)), np.rot90(expected, turns))
        assert np.array_equal(fixture, saved)
    features, signatures, normalized, regimes = [], set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p["input"]) for s in puzzle for p in puzzle[s]))
        local = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                f = _features(pair)
                f["split"] = split
                features.append(f)
                local.append(f)
                mapping = {}
                data = tuple(mapping.setdefault(int(c), len(mapping)) for c in pair["input"].flat)
                normalized.add((pair["input"].shape, data))
        # Counted twins have identical source geometry. Swapping only their
        # controls changes a unique colored output: geometry alone cannot solve.
        pair = puzzle["train"][0]
        _, info = _analyze(pair["input"])
        a, b = _twin_witness(info)
        altered = pair["input"].copy()
        for old, new in ((a, b), (b, a)):
            cells = info["bars"][old]["cells"]
            altered[cells[:, 0], cells[:, 1]] = new
        assert not np.array_equal(solve(altered), pair["output"])
        # Ignoring B, even while keeping A's translation-only exact cover,
        # cannot reproduce this demonstration's answer.
        no_controls = _assemblies(info, {c: 0 for c in info["bars"]})
        assert not any(np.array_equal(o, pair["output"]) for o, _ in no_controls)
        no_assembly = _rotation_only(info)
        assert no_assembly is None or not np.array_equal(no_assembly, pair["output"])
        training = [f for f in local if f["split"] == "train"]
        testing = [f for f in local if f["split"] == "test"]
        shifts = set()
        if not ({f["axis"] for f in training} & {f["axis"] for f in testing}):
            shifts.add("orientation")
        if (max(n for f in training for n in f["groups"]) <= 2
                and all(3 in f["groups"] for f in testing)):
            shifts.add("grouping")
        if (max(k for f in training for k in f["counts"]) <= 4
                and all(max(f["counts"]) >= 5 for f in testing)):
            shifts.add("counting")
        assert shifts, "no observable train/test shift"
        regimes.update(shifts)
    assert len(signatures) == 42 and len(normalized) >= 205
    assert len({tuple(f["shape"]) for f in features}) >= 35
    assert {f["axis"] for f in features} == {0, 1}
    assert {f["background"] for f in features} == set(range(10))
    assert {f["cavities"] for f in features} >= {1, 2, 3}
    assert regimes == {"orientation", "grouping", "counting"}
    assert len({f["pieces"] for f in features}) >= 4
    assert any(f["reuse"] for f in features)
    assert any(f["nonconvex"] for f in features)
    assert max(a for f in features for a in f["areas"]) >= 25
    assert min(a for f in features for a in f["areas"]) <= 7
    assert set(k for f in features for k in f["counts"]) == set(range(1, 10))
    assert any(max(f["counts"]) >= 5 and f["split"] == "train" for f in features)
    assert any(max(f["counts"]) >= 5 and f["split"] == "test" for f in features)
    for a, b in ((2, 1), (10, 4), (10, 1)):
        p = generate_puzzle(1234+a+b, a, b)
        q = generate_puzzle(1234+a+b, a, b)
        assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                   for k in ("input", "output"))
    for a, b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(42, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong label accepted")
    return {"seeds": 42, "pairs": len(features), "distinct_structures": len(normalized),
            "canvas_shapes": len({tuple(f["shape"]) for f in features}),
            "piece_counts": sorted({f["pieces"] for f in features}),
            "cavity_counts": sorted({f["cavities"] for f in features}),
            "train_test_shifts": sorted(regimes),
            "reused_color_pairs": sum(f["reuse"] for f in features),
            "nonconvex_pairs": sum(f["nonconvex"] for f in features)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
