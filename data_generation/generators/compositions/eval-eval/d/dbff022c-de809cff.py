"""Canceling alarms in sealed chambers. Standard library and NumPy only.

The input-only solver never receives generation masks, a seed or demonstrations.
Generation labels use local construction masks; validation infers global rooms.
"""
from collections import deque
import numpy as np


BACKGROUND = 0
CENTER = 9
FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
              if (r, c) != (0, 0))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue, cells = deque([(r, c)]), []
        while queue:
            p = queue.popleft()
            cells.append(p)
            for dr, dc in EIGHT if diagonal else FOUR:
                rr, cc = p[0] + dr, p[1] + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(tuple(cells))
    return result


def _key(inp):
    candidates = []
    h, w = inp.shape
    for cells in _components(inp != 0, diagonal=True):
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        sh, sw = r1 - r0 + 1, c1 - c0 + 1
        if len(cells) != sh * sw:
            continue
        lanes = []
        if sh == 2 and 3 <= sw <= 4:
            if r0 == 0:
                lanes.append(("top", inp[r0, c0:c1+1], inp[r1, c0:c1+1]))
            if r1 == h - 1:
                lanes.append(("bottom", inp[r1, c0:c1+1], inp[r0, c0:c1+1]))
        if sw == 2 and 3 <= sh <= 4:
            if c0 == 0:
                lanes.append(("left", inp[r0:r1+1, c0], inp[r0:r1+1, c1]))
            if c1 == w - 1:
                lanes.append(("right", inp[r0:r1+1, c1], inp[r0:r1+1, c0]))
        for side, outer, inner in lanes:
            sources, alarms = tuple(map(int, outer)), tuple(map(int, inner))
            if (len(set(sources)) == len(sources) and set(sources).isdisjoint(alarms)
                    and not ({0, 9} & (set(sources) | set(alarms)))):
                candidates.append((side, dict(zip(sources, alarms)), frozenset(cells)))
    if len(candidates) != 1:
        raise ValueError("Expected one unambiguous local edge key")
    return candidates[0]


def _enclosed(scene, wall):
    """Flood the complement of one wall color; background connectivity is four."""
    h, w = scene.shape
    return [cells for cells in _components(scene != wall)
            if not any(r in (0, h-1) or c in (0, w-1) for r, c in cells)]


def _counts(mask, centers):
    counts = np.zeros(mask.shape, dtype=np.int16)
    for r, c in centers:
        for dr, dc in EIGHT:
            rr, cc = r + dr, c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                counts[rr, cc] += 1
    return counts


def _infer(inp):
    """All semantic analysis is from the input, before any label is consulted."""
    if np.any(inp == CENTER):
        raise ValueError("Input may not contain the center color")
    side, mapping, key_cells = _key(inp)
    scene = inp.copy()
    for q in key_cells:
        scene[q] = 0
    occupied = np.zeros(inp.shape, dtype=bool)
    rooms = []
    for wall, alarm in mapping.items():
        for cells in _enclosed(scene, wall):
            mask = np.zeros(inp.shape, dtype=bool)
            rr, cc = np.array(cells).T
            mask[rr, cc] = True
            if np.any(occupied & mask):
                raise ValueError("Overlapping chamber ownership")
            values = inp[mask]
            if set(map(int, values)) & set(mapping):
                raise ValueError("Nested or mixed-color chamber walls")
            colors, votes = np.unique(values[values != 0], return_counts=True)
            if not len(colors) or np.count_nonzero(votes == votes.max()) != 1:
                raise ValueError("No unique surviving floor majority")
            floor = int(colors[votes.argmax()])
            if votes.max() < .60 * len(cells) or floor == alarm:
                raise ValueError("Insufficient evidence or identical floor and alarm")
            centers = tuple(q for q in cells if inp[q] == 0)
            counts = _counts(mask, centers)
            occupied |= mask
            rooms.append({"cells": cells, "mask": mask, "wall": wall,
                          "alarm": alarm, "floor": floor, "centers": centers,
                          "counts": counts})
    return {"side": side, "mapping": mapping, "key_cells": key_cells,
            "scene": scene, "occupied": occupied, "rooms": rooms}


def _render_analysis(inp, analysis, union=False):
    out = inp.copy()
    for room in analysis["rooms"]:
        mask, counts = room["mask"], room["counts"]
        out[mask] = room["floor"]
        paint = counts > 0 if union else counts % 2 == 1
        out[paint] = room["alarm"]
        for q in room["centers"]:
            out[q] = CENTER
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Apply the fixed family rule to an input alone, without mutating it."""
    inp = _grid(input_grid)
    return _render_analysis(inp, _infer(inp))


def _dilate(mask, diagonal=False):
    result = mask.copy()
    for dr, dc in EIGHT if diagonal else FOUR:
        dst = (slice(max(0, dr), min(mask.shape[0], mask.shape[0]+dr)),
               slice(max(0, dc), min(mask.shape[1], mask.shape[1]+dc)))
        src = (slice(max(0, -dr), min(mask.shape[0], mask.shape[0]-dr)),
               slice(max(0, -dc), min(mask.shape[1], mask.shape[1]-dc)))
        result[dst] |= mask[src]
    return result


def _profile(rng, h, w):
    mask = np.zeros((h, w), dtype=bool)
    # Overlapping row intervals guarantee four-connectivity, with ample floor.
    for r in range(h):
        left = int(rng.integers(0, max(2, w//3)))
        right = int(rng.integers(max(left+2, 2*w//3), w+1))
        mask[r, left:right] = True
    if np.all(mask):
        mask[0, 0] = False
    return mask


def _shape(rng, kind, small=False, larger=False):
    """Return intact wall mask, original floor mask and whether the wall is open."""
    thickness = 1 if small else int(rng.choice([1, 1, 2]))
    if kind == "diamond":
        radius = int(rng.integers(2, 4 if small else 5))
        rr, cc = np.indices((2*radius+1, 2*radius+1))
        cavity = abs(rr-radius) + abs(cc-radius) < radius
        wall = abs(rr-radius) + abs(cc-radius) == radius
    elif kind == "multi":
        n = int(rng.integers(2, 4))
        h = int(rng.integers(4, 6 if small else 8))
        widths = [int(rng.integers(3, 5 if small else 7)) for _ in range(n)]
        cavity = np.zeros((h+2, sum(widths)+n+1), dtype=bool)
        start = 1
        for width in widths:
            patch = np.ones((h, width), dtype=bool) if rng.random() < .4 else _profile(rng, h, width)
            cavity[1:-1, start:start+width] = patch
            start += width+1
        wall = ~cavity
    else:
        h = int(rng.integers(4, 6 if small else (10 if larger else 8)))
        w = int(rng.integers(5, 7 if small else (11 if larger else 9)))
        if kind in ("rectangle", "open"):
            core = np.ones((h, w), dtype=bool)
        elif kind == "elbow":
            core = np.zeros((h, w), dtype=bool)
            core[:, :int(rng.integers(2, w-1))] = True
            core[-int(rng.integers(2, h)):, :] = True
        else:
            core = _profile(rng, h, w)
        cavity = np.pad(core, thickness)
        expanded = cavity.copy()
        for _ in range(thickness):
            expanded = _dilate(expanded, diagonal=kind in ("rectangle", "open"))
        wall = expanded & ~cavity
    opened = kind == "open"
    if opened:
        # A genuine four-neighbor passage, not a diagonal leak.
        col = int(rng.integers(thickness, cavity.shape[1]-thickness))
        first = int(np.flatnonzero(cavity[:, col])[0])
        wall[:first, col] = False
    turns = int(rng.integers(4))
    wall, cavity = np.rot90(wall, turns), np.rot90(cavity, turns)
    if rng.random() < .5:
        wall, cavity = np.fliplr(wall), np.fliplr(cavity)
    return wall.copy(), cavity.copy(), opened


def _choose_defects(rng, mask, style):
    cells = [tuple(map(int, q)) for q in np.argwhere(mask)]
    centers = []
    if style == "cancel":
        starts = [q for q in cells if q[1]+2 < mask.shape[1] and mask[q[0], q[1]+2]
                  and all(0 <= q[0]+dr < mask.shape[0] and mask[q[0]+dr, q[1]+1]
                          for dr in (-1, 0, 1))]
        if not starts:
            raise ValueError("Cancellation lesson needs a wider cavity")
        q = starts[int(rng.integers(len(starts)))]
        centers = [q, (q[0], q[1]+2)]
    elif style == "adjacent":
        starts = [(q, (q[0]+dr, q[1]+dc)) for q in cells for dr, dc in ((0, 1), (1, 0), (1, 1))
                  if 0 <= q[0]+dr < mask.shape[0] and 0 <= q[1]+dc < mask.shape[1]
                  and mask[q[0]+dr, q[1]+dc]]
        if not starts:
            raise ValueError("Adjacent lesson needs two neighboring floor cells")
        centers = list(starts[int(rng.integers(len(starts)))])
    elif style == "boundary":
        candidates = [q for q in cells if any(not (0 <= q[0]+dr < mask.shape[0]
                                                   and 0 <= q[1]+dc < mask.shape[1])
                                               or not mask[q[0]+dr, q[1]+dc] for dr, dc in EIGHT)]
        centers = [candidates[int(rng.integers(len(candidates)))]]
    else:
        centers = [cells[int(rng.integers(len(cells)))]]
    # Lessons stay readable. Optional random cases range from isolated to clusters.
    target = len(centers) if style in ("cancel", "adjacent", "boundary") else int(rng.integers(1, max(2, min(10, len(cells)//5)+1)))
    rng.shuffle(cells)
    for q in cells:
        if len(centers) >= target:
            break
        if q not in centers:
            centers.append(q)
    # Reserve one ordinary wrong-color error away from all emitted rings.
    far = [q for q in cells if q not in centers and all(max(abs(q[0]-p[0]), abs(q[1]-p[1])) > 1 for p in centers)]
    rng.shuffle(far)
    budget = max(0, int(.35 * len(cells))-len(centers))
    wrong = far[:min(budget, int(rng.integers(1, max(2, min(5, len(cells)//9)+1))))]
    if len(centers)+len(wrong) > .40*len(cells):
        raise ValueError("Too much damage for a majority witness")
    return centers, wrong


def _object(rng, kind, wall_color, alarm, materials, active, style="random", small=False, larger=False):
    wall, cavity, opened = _shape(rng, kind, small, larger)
    local = np.zeros(wall.shape, dtype=np.int8)
    local[wall] = wall_color
    expected = local.copy()
    details = []
    for index, cells in enumerate(_components(cavity)):
        mask = np.zeros(wall.shape, dtype=bool)
        rr, cc = np.array(cells).T
        mask[rr, cc] = True
        floor = int(rng.choice([c for c in materials if c != alarm]))
        local[mask] = floor
        centers, wrong = _choose_defects(rng, mask, style if index == 0 else "random")
        for q in centers:
            local[q] = 0
        for q in wrong:
            local[q] = int(rng.choice([c for c in materials if c != floor]))
        details.append((mask, floor, centers, alarm))
    expected = local.copy()
    if active and not opened:
        # Construction renderer uses sampled masks, not global inference.
        for mask, floor, centers, contrast in details:
            expected[mask] = floor
            parity = np.zeros(mask.shape, dtype=bool)
            for r, c in centers:
                for dr, dc in EIGHT:
                    rr, cc = r+dr, c+dc
                    if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                        parity[rr, cc] = not parity[rr, cc]
            expected[parity] = contrast
            for q in centers:
                expected[q] = CENTER
    return local, expected


def _put_key(grid, side, entries, start):
    cells = []
    for i, (wall, alarm) in enumerate(entries):
        k = start+i
        if side == "top":
            p, q = (0, k), (1, k)
        elif side == "bottom":
            p, q = (grid.shape[0]-1, k), (grid.shape[0]-2, k)
        elif side == "left":
            p, q = (k, 0), (k, 1)
        else:
            p, q = (k, grid.shape[1]-1), (k, grid.shape[1]-2)
        grid[p], grid[q] = wall, alarm
        cells.extend((p, q))
    return cells


def _pack(rng, objects, entries, side, large=False):
    maxh = max(a.shape[0] for a, _ in objects)
    maxw = max(a.shape[1] for a, _ in objects)
    for attempt in range(18):
        low = 18 if len(objects) >= 3 else 13 if len(objects) == 2 else 8
        if large:
            low = max(low, 20)
        h = int(rng.integers(max(low, maxh), 31))
        w = int(rng.integers(max(low, maxw), 31))
        if rng.random() < .2:
            h = w = max(h, w)
        if attempt >= 12:
            h = w = 30
        inp = np.zeros((h, w), dtype=np.int8)
        length = w if side in ("top", "bottom") else h
        start = int(rng.integers(length-len(entries)+1))
        key_cells = _put_key(inp, side, entries, start)
        out = inp.copy()
        blocked = _dilate(inp != 0, True)
        for i in sorted(range(len(objects)), key=lambda i: -objects[i][0].size):
            a, b = objects[i]
            ah, aw = a.shape
            choices = [(r, c) for r in range(h-ah+1) for c in range(w-aw+1)
                       if not blocked[r:r+ah, c:c+aw].any()]
            if not choices:
                break
            border = [(r, c) for r, c in choices if r == 0 or c == 0 or r+ah == h or c+aw == w]
            candidates = border if border and rng.random() < .35 else choices
            r, c = candidates[int(rng.integers(len(candidates)))]
            inp[r:r+ah, c:c+aw] = a
            out[r:r+ah, c:c+aw] = b
            blocked[max(0, r-1):min(h, r+ah+1), max(0, c-1):min(w, c+aw+1)] = True
        else:
            assert all(inp[q] == out[q] for q in key_cells)
            return {"input": inp, "output": out}
    return None


def _make_pair(rng, side, lesson=None, large=False):
    for attempt in range(36):
        palette = list(map(int, rng.permutation(np.arange(1, 9))))
        nsource = int(rng.integers(3, 5))
        nmaterial = int(rng.integers(3, 5)) if nsource == 3 else 3
        sources = palette[:nsource]
        materials = palette[nsource:nsource+nmaterial]
        inert = palette[-1]
        alarms = list(map(int, rng.permutation(materials)[:3]))
        if nsource == 4:
            alarms.append(int(rng.choice(materials)))
        mapping = dict(zip(sources, alarms))
        primary, secondary = sources[0], sources[1]
        specs = []
        if lesson == "cancel":
            specs = [("rectangle", primary, True, "cancel"),
                     ("diamond", secondary, True, "boundary")]
        elif lesson == "controls":
            specs = [("multi", primary, True, "adjacent"),
                     ("open", secondary, True, "random"),
                     ("profile", inert, False, "random")]
        else:
            specs = [(str(rng.choice(["rectangle", "profile", "diamond", "multi", "elbow"])),
                      int(rng.choice(sources)), True, str(rng.choice(["random", "random", "boundary"])))]
        target = int(rng.integers(len(specs), max(len(specs)+1, 7)))
        # Rich packing can use small shapes; it never changes the solving rule.
        if attempt >= 20:
            target = len(specs)
        while len(specs) < target:
            kind = str(rng.choice(["rectangle", "profile", "diamond", "multi", "elbow", "open"]))
            active = rng.random() < .8
            source = int(rng.choice(sources)) if active else inert
            specs.append((kind, source, active, "random"))
        try:
            objects = [_object(rng, kind, wall, mapping.get(wall, int(rng.choice(materials))),
                               materials, active, style, small=attempt >= 12 or target >= 5,
                               larger=large and target <= 3)
                       for kind, wall, active, style in specs]
            entries = list(mapping.items())
            rng.shuffle(entries)
            pair = _pack(rng, objects, entries, side, large)
            if pair is None:
                continue
            a = _infer(pair["input"])
            if not np.array_equal(_render_analysis(pair["input"], a), pair["output"]):
                continue
            if not any(room["centers"] for room in a["rooms"]):
                continue
            f = _features(pair["input"], a)
            if lesson == "cancel" and not (f["cancel"] and f["noise_repair"] and f["clipped"]):
                continue
            if lesson == "controls" and not (f["adjacent"] and f["open"] and f["inert"]):
                continue
            return pair
        except ValueError:
            continue
    raise RuntimeError("Bounded chamber construction failed")


def _counts_args(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts_args(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    sides = list(map(str, rng.permutation(["top", "left", "bottom", "right"])))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            side = sides[i] if split == "train" and i < 2 else str(rng.choice(sides[:2] if split == "train" else sides[2:]))
            lesson = ("cancel" if i == 0 else "controls" if i == 1 else None) if split == "train" else None
            for _ in range(8):
                pair = _make_pair(rng, side, lesson, large=split == "test")
                signature = (pair["input"].shape, pair["input"].tobytes())
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not construct a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp, a):
    adjacent = cancel = clipped = noise_repair = False
    floor_alarm_pairs, silhouettes = set(), set()
    for room in a["rooms"]:
        centers, mask, counts = room["centers"], room["mask"], room["counts"]
        center_mask = np.zeros(inp.shape, dtype=bool)
        for q in centers:
            center_mask[q] = True
        adjacent |= any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) == 1
                        for i, p in enumerate(centers) for q in centers[:i])
        cancel |= bool(np.any((counts > 0) & (counts % 2 == 0) & ~center_mask))
        clipped |= any(any(not (0 <= r+dr < inp.shape[0] and 0 <= c+dc < inp.shape[1])
                               or not mask[r+dr, c+dc] for dr, dc in EIGHT) for r, c in centers)
        noise_repair |= bool(np.any(mask & (inp != 0) & (inp != room["floor"]) & (counts == 0)))
        floor_alarm_pairs.add((room["floor"], room["alarm"]))
        rr, cc = np.nonzero(mask)
        crop = mask[rr.min():rr.max()+1, cc.min():cc.max()+1]
        silhouettes.add((crop.shape, crop.tobytes()))
    # Find actual open source components, rather than trusting sampled flags.
    source_boundaries = set()
    for room in a["rooms"]:
        for r, c in room["cells"]:
            for dr, dc in FOUR:
                q = (r+dr, c+dc)
                if inp[q] == room["wall"]:
                    source_boundaries.add(q)
    open_witness = False
    for source in a["mapping"]:
        wall_mask = a["scene"] == source
        for cells in _components(wall_mask, diagonal=True):
            if source_boundaries.intersection(cells):
                continue
            rr, cc = zip(*cells)
            patch = a["scene"][min(rr):max(rr)+1, min(cc):max(cc)+1]
            if min(patch.shape) >= 3 and np.any(patch == 0) and np.any((patch != 0) & (patch != source)):
                open_witness = True
    inert_witness = False
    candidates = set(map(int, np.unique(a["scene"]))) - set(a["mapping"]) - {0, 9}
    for color in candidates:
        for cells in _enclosed(a["scene"], color):
            values = np.array([inp[q] for q in cells])
            nz = values[values != 0]
            if len(nz) and np.any(values == 0) and not any(a["occupied"][q] for q in cells):
                colors, votes = np.unique(nz, return_counts=True)
                if len(colors) >= 2 and votes.max() >= .60*len(cells):
                    inert_witness = True
    return {"side": a["side"], "key_length": len(a["mapping"]),
            "rooms": len(a["rooms"]), "holes": sum(len(r["centers"]) for r in a["rooms"]),
            "adjacent": adjacent, "cancel": cancel, "clipped": clipped,
            "noise_repair": noise_repair, "open": open_witness, "inert": inert_witness,
            "alarms": {r["alarm"] for r in a["rooms"] if r["centers"]},
            "pairs": floor_alarm_pairs, "silhouettes": silhouettes}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts_args(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            sig = (inp.shape, inp.tobytes())
            assert sig not in seen, "Duplicate input"
            seen.add(sig)
            a = _infer(inp)
            expected = _render_analysis(inp, a)
            assert np.array_equal(out, expected), "Incorrect input-derived output"
            assert np.array_equal(inp[~a["occupied"]], out[~a["occupied"]]), "Protected cells changed"
            centers = set(q for room in a["rooms"] for q in room["centers"])
            assert centers and set(map(tuple, np.argwhere(out == CENTER))) == centers
            assert np.any(inp != out)
            for room in a["rooms"]:
                assert room["floor"] != room["alarm"]
                assert all(inp[q] == 0 for q in room["centers"])
            features[split].append(_features(inp, a))
    train = features["train"]
    for witness in ("adjacent", "cancel", "clipped", "noise_repair", "open", "inert"):
        assert any(f[witness] for f in train), "Missing training witness: " + witness
    assert len(set.union(*(f["alarms"] for f in train))) >= 2
    train_sides = {f["side"] for f in train}
    assert len(train_sides) >= 2
    assert all(f["side"] not in train_sides for f in features["test"]), "Missing orientation shift"
    # These contrasts explicitly establish necessity of the inherited concepts.
    assert any(not np.array_equal(_render_analysis(_grid(p["input"]), _infer(_grid(p["input"])), union=True),
                                  _grid(p["output"])) for p in puzzle["train"])


def _vectorized_render(inp, a):
    """Alternate parity calculation used only for bounded author-owned QA."""
    out = inp.copy()
    for room in a["rooms"]:
        mask = room["mask"]
        holes = mask & (inp == 0)
        count = np.zeros(inp.shape, dtype=np.int16)
        for dr, dc in EIGHT:
            dst = (slice(max(0, dr), min(inp.shape[0], inp.shape[0]+dr)),
                   slice(max(0, dc), min(inp.shape[1], inp.shape[1]+dc)))
            src = (slice(max(0, -dr), min(inp.shape[0], inp.shape[0]-dr)),
                   slice(max(0, -dc), min(inp.shape[1], inp.shape[1]-dc)))
            count[dst] += holes[src]
        out[mask] = room["floor"]
        out[mask & (count % 2 == 1)] = room["alarm"]
        out[holes] = CENTER
    return out


def self_test():
    """Bounded semantic probes and structural diversity measurements on 42 seeds."""
    signatures, shapes, structures, silhouettes = set(), set(), set(), set()
    sides, lengths, room_counts, hole_counts, palette_pairs = set(), set(), set(), set(), set()
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if seed == 0:
            first = puzzle
        signatures.add(tuple((p["input"].shape, p["input"].tobytes(), p["output"].tobytes())
                             for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                a = _infer(inp)
                f = _features(inp, a)
                assert np.array_equal(_vectorized_render(inp, a), out)
                reverse = dict(a, rooms=[dict(r, centers=tuple(reversed(r["centers"])),
                                            counts=_counts(r["mask"], tuple(reversed(r["centers"]))))
                                        for r in reversed(a["rooms"])])
                assert np.array_equal(_render_analysis(inp, reverse), out)
                before = inp.copy()
                assert np.array_equal(solve(inp), out) and np.array_equal(inp, before)
                shapes.add(inp.shape)
                roles = {0: 0}
                normalized = tuple(roles.setdefault(int(c), len(roles)) for c in inp.flat)
                structures.add((inp.shape, normalized))
                silhouettes.update(f["silhouettes"])
                sides.add(f["side"])
                lengths.add(f["key_length"])
                room_counts.add(f["rooms"])
                hole_counts.add(f["holes"])
                palette_pairs.update(f["pairs"])
    assert len(signatures) == 42
    assert len(shapes) >= 35 and len(structures) >= 180 and len(silhouettes) >= 60
    assert sides == {"top", "bottom", "left", "right"} and lengths == {3, 4}
    assert len(room_counts) >= 5 and len(hole_counts) >= 10 and len(palette_pairs) >= 30
    repeat = generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], repeat[s]) for k in ("input", "output"))
    for nt, ne in ((2, 1), (5, 4), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(100+nt+ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Mutated output accepted")
    # A tiny hand-specified scene establishes the precise parity and precedence.
    probe = np.zeros((12, 17), dtype=np.int8)
    probe[:2, 2:5] = [[1, 2, 4], [6, 7, 8]]
    probe[4:11, 3:13] = 1
    probe[5:10, 4:12] = 3
    probe[7, 6] = probe[7, 8] = 0
    probe[5, 10] = 7
    result = solve(probe)
    assert result[7, 6] == result[7, 8] == 9
    assert result[6, 7] == result[7, 7] == result[8, 7] == 3
    assert result[6, 5] == 6 and result[5, 10] == 3
    assert result[4, 6] == 1 and result[3, 6] == 0
    opened = probe.copy()
    opened[4, 7] = 0
    assert np.array_equal(solve(opened), opened)
    inert = probe.copy()
    inert[inert == 1] = 5
    # Restore key source identity; only the chamber is now unlisted.
    inert[0, 2] = 1
    assert np.array_equal(solve(inert), inert)
    tied = probe.copy()
    tied[5:10, 4:12] = np.tile([3, 7], (5, 4))
    try:
        solve(tied)
    except ValueError:
        pass
    else:
        raise AssertionError("Tied floor accepted")
