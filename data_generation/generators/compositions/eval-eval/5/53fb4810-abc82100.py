"""Interlocking stencil conveyor: a cyclic word drives registered mask packing.

Only NumPy and the standard library are used. No parent code is imported.
The decoder reads one input; it never consults generation records or examples.
"""

from __future__ import annotations

import itertools
import numpy as np

BACKGROUND, HUB, NEUTRAL = 0, 5, 8
PAYLOAD = (1, 2, 3, 4, 6, 7, 9)
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer) and
             bool(np.all((0 <= grid) & (grid <= 9))), "grid colors must be integers 0..9")
    return grid.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _components(cells, diagonal=False):
    remaining, groups = set(cells), []
    neighbors = (tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                       if r or c) if diagonal else DIRECTIONS)
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), min(cs), max(rs), max(cs)


def _inside(q, shape):
    return 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]


def _hub_cells(top, left, height, width, kind):
    cells = {(r, c) for r in range(top, top + height)
             for c in range(left, left + width)}
    if kind == "clipped":
        cells -= {(top, left), (top, left + width - 1),
                  (top + height - 1, left), (top + height - 1, left + width - 1)}
    elif kind == "cross":
        mr, mc = top + height // 2, left + width // 2
        cells = {p for p in cells if p[0] == mr or p[1] == mc}
    return cells


def _hub_and_word(grid):
    cells = set(map(tuple, np.argwhere(grid == HUB)))
    _require(bool(cells) and len(_components(cells)) == 1, "one connected hub is required")
    t, l, b, r = _box(cells)
    h, w = b - t + 1, r - l + 1
    _require(h in (3, 5, 7) and w in (3, 5, 7), "hub axes must be odd and 3..7")
    _require(any(cells == _hub_cells(t, l, h, w, kind)
                 for kind in ("rectangle", "clipped", "cross")), "unrecognized hub")
    center = t + h // 2, l + w // 2
    choices = []
    for dr, dc in DIRECTIONS:
        face = h // 2 if dr else w // 2
        q = center[0] + (face + 1) * dr, center[1] + (face + 1) * dc
        if not _inside(q, grid.shape) or int(grid[q]) not in PAYLOAD:
            continue
        word, positions = [], []
        while _inside(q, grid.shape) and int(grid[q]) in PAYLOAD:
            word.append(int(grid[q]))
            positions.append(q)
            q = q[0] + dr, q[1] + dc
        # A trailing blank explicitly delimits one whole program, not a prefix.
        if _inside(q, grid.shape) and int(grid[q]) == BACKGROUND:
            choices.append((center, face, (dr, dc), tuple(word), set(positions)))
    _require(len(choices) == 1, "hub must carry one uniquely delimited centered word")
    center, face, direction, word, positions = choices[0]
    return dict(hub=cells, center=center, face=face, direction=direction,
                word=word, word_cells=positions)


def _read_instructions(grid, rail):
    masks = _components(map(tuple, np.argwhere(grid == NEUTRAL)), diagonal=True)
    _require(bool(masks), "missing stencil instructions")
    excluded = rail["hub"] | rail["word_cells"]
    labels = [tuple(p) for p in np.argwhere(np.isin(grid, PAYLOAD)) if tuple(p) not in excluded]
    records, used_labels = [], set()
    for mask in masks:
        near = {(r + dr, c + dc) for r, c in mask
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        candidates = []
        for s in labels:
            source = int(grid[s])
            for dr, dc in DIRECTIONS:
                target_cell = s[0] + dr, s[1] + dc
                q = s[0] + 2 * dr, s[1] + 2 * dc
                if not _inside(q, grid.shape) or q not in near:
                    continue
                target = int(grid[target_cell])
                if target_cell in excluded or target not in PAYLOAD or target == source:
                    continue
                if int(grid[q]) not in (BACKGROUND, NEUTRAL):
                    continue
                candidates.append(dict(source=source, target=target, q=q,
                                       label_direction=(dr, dc), mask=set(mask),
                                       labels={s, target_cell},
                                       offsets=tuple(sorted((r-q[0], c-q[1]) for r, c in mask))))
        _require(len(candidates) == 1, "each mask needs exactly one label/registration reading")
        record = candidates[0]
        _require(not record["labels"] & used_labels, "instructions cannot share labels")
        used_labels |= record["labels"]
        records.append(record)
    _require(len({r["source"] for r in records}) == len(records), "ambiguous source mapping")
    _require(set(rail["word"]) == {r["source"] for r in records},
             "every word symbol needs one instruction, and every instruction must be used")
    claimed = excluded | used_labels | set().union(*(r["mask"] for r in records))
    _require(claimed == set(map(tuple, np.argwhere(grid != BACKGROUND))), "unclassified input material")
    return records


def _decode(grid):
    rail = _hub_and_word(grid)
    rail["records"] = _read_instructions(grid, rail)
    return rail


def _packing(shape, rail, variant="exact"):
    """Plan stations in the unbounded plane, then clip their painted cells.

    Variants exist only for diagnostic counterexamples; solve uses exact.
    """
    word = rail["word"]
    if variant == "prefix":
        # The tempting extrapolation that treats ABA as an AB prefix.
        p = next(p for p in range(1, len(word) + 1)
                 if all(v == word[i % p] for i, v in enumerate(word)))
        word = word[:p]
    records = {r["source"]: r for r in rail["records"]}
    offsets = {s: set(r["offsets"]) for s, r in records.items()}
    if variant == "centered":
        for s, cells in offsets.items():
            t, l, b, r = _box(cells)
            cr, cc = (t + b) // 2, (l + r) // 2
            offsets[s] = {(y-cr, x-cc) for y, x in cells}
    dr, dc = rail["direction"]
    cr, cc = rail["center"]
    first = offsets[word[0]]
    distance = rail["face"] + 1 - min(y*dr + x*dc for y, x in first)
    hub = set(rail.get("hub", ()))
    occupied, previous, paints, events = hub.copy(), set(), {}, []
    ordinal = 0
    while True:
        if variant == "once" and ordinal == len(word):
            break
        source = word[ordinal % len(word)]
        cells = offsets[source]
        blocking_cells = cells
        if variant == "boxes":
            t, l, b, r = _box(cells)
            blocking_cells = {(y, x) for y in range(t, b+1) for x in range(l, r+1)}
        rejected = []
        while True:
            q = cr + distance*dr, cc + distance*dc
            if not _inside(q, shape):
                return paints, events
            full = {(q[0]+y, q[1]+x) for y, x in cells}
            blocking = {(q[0]+y, q[1]+x) for y, x in blocking_cells}
            obstruction = (previous if variant == "previous" else occupied) | hub
            if not blocking & obstruction:
                break
            rejected.append(q)
            distance += 1
        target = records[source]["target"]
        clipped = {p for p in full if _inside(p, shape)}
        for p in clipped:
            paints[p] = target
        events.append(dict(q=q, source=source, target=target, full=full,
                           clipped=clipped, rejected=tuple(rejected), ordinal=ordinal))
        occupied |= blocking
        previous = blocking
        ordinal += 1
        distance += 1
    return paints, events


def _render(grid, rail, variant="exact"):
    output = grid.copy()
    erase = rail["word_cells"] | set().union(*(r["mask"] | r["labels"] for r in rail["records"]))
    for p in erase:
        output[p] = BACKGROUND
    paints, events = _packing(grid.shape, rail, variant)
    for p, color in paints.items():
        output[p] = color
    return output, events


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire answer from visible geometry and local color labels."""
    grid = _grid(input_grid)
    rail = _decode(grid)
    output, events = _render(grid, rail)
    _require(all(not e["full"] & rail["hub"] for e in events), "stamps may not hit the hub")
    return output


def _choose(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _rotate_cells(cells, turns):
    for _ in range(turns % 4):
        cells = {(-c, r) for r, c in cells}
    return cells


def _label_options(offsets):
    """Directions admitting exactly one reading of this two-color label pair."""
    cells = set(offsets)
    near = {(r+a, c+b) for r, c in cells for a in (-1, 0, 1) for b in (-1, 0, 1)}
    if (0, 0) not in near:
        return []
    result = []
    for dr, dc in DIRECTIONS:
        source, target = (-2*dr, -2*dc), (-dr, -dc)
        reverse = -3*dr, -3*dc
        if source not in cells and target not in cells and reverse not in near:
            result.append((dr, dc))
    return result


def _random_mask(rng, extent=4):
    """Procedural masks, including sparse branches and genuinely hollow shapes."""
    for _ in range(100):
        kind = _choose(rng, ("line", "stairs", "chevron", "outline", "c", "branch", "growth"))
        h, w = int(rng.integers(2, extent+1)), int(rng.integers(2, extent+1))
        if kind == "line":
            cells = {(0, c) for c in range(w)}
        elif kind == "stairs":
            cells = {(i, i) for i in range(max(h, w))}
        elif kind == "chevron":
            cells = {(r, abs(r)) for r in range(-h+1, h)}
        elif kind == "outline":
            cells = {(r, c) for r in range(h) for c in range(w)
                     if r in (0, h-1) or c in (0, w-1)}
        elif kind == "c":
            cells = {(r, w-1) for r in range(h)} | {(r, c) for r in (0, h-1) for c in range(w)}
        elif kind == "branch":
            cells = {(r, 0) for r in range(h)} | {(h//2, c) for c in range(w)}
        else:
            cells = {(0, 0)}
            wanted = int(rng.integers(3, h*w+1))
            for _ in range(wanted*12):
                if len(cells) >= wanted:
                    break
                r, c = _choose(rng, sorted(cells))
                a, b = _choose(rng, ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1)))
                if 0 <= r+a < h and 0 <= c+b < w:
                    cells.add((r+a, c+b))
        cells = _rotate_cells(cells, int(rng.integers(4)))
        if rng.random() < .5:
            cells = {(-r, c) for r, c in cells}
        near = {(r+a, c+b) for r, c in cells for a in (-1, 0, 1) for b in (-1, 0, 1)}
        candidates = []
        for qr, qc in sorted(near):
            offsets = {(r-qr, c-qc) for r, c in cells}
            if _label_options(offsets):
                candidates.append(offsets)
        if candidates:
            # Both occupied and empty registrations remain in the distribution.
            holes = [c for c in candidates if (0, 0) not in c]
            occupied = [c for c in candidates if (0, 0) in c]
            pool = holes if holes and rng.random() < .45 else occupied or holes
            return tuple(sorted(_choose(rng, pool)))
    return ((0, 0), (1, 1), (2, 2))


def _specifications(rng, teaching, extent, count=None, word_length=None):
    count = 2 if teaching is not None else count or int(rng.integers(1, 4))
    palette = [int(x) for x in rng.permutation(PAYLOAD)]
    sources = palette[:count]
    targets = [int(rng.choice([c for c in PAYLOAD if c != s])) for s in sources]
    # Different targets make the two pedagogical word witnesses visible.
    if teaching is not None:
        targets = palette[count:2*count]
    if teaching == 0:
        # Endpoint registration of variable diagonal masks; boxes would prevent interlocking.
        sign = int(rng.choice((-1, 1)))
        lengths = [int(rng.integers(3, 5)), int(rng.integers(2, 5))]
        masks = [tuple((sign*i, i) for i in range(length)) for length in lengths]
        word = (sources[0], sources[1], sources[0])
    elif teaching == 1:
        # The second mask occupies different transverse rows. The next long line
        # must remember the first line, not merely avoid the immediately prior mask.
        length, depth, width = int(rng.integers(3, 6)), int(rng.integers(1, 4)), int(rng.integers(1, 4))
        sign = int(rng.choice((-1, 1)))
        masks = [tuple((0, c) for c in range(length)),
                 tuple((sign*r, c) for r in range(1, depth+1) for c in range(width))]
        word = (sources[0], sources[1])
    else:
        masks = [_random_mask(rng, extent) for _ in sources]
        length = word_length or int(rng.integers(count, max(count+1, extent+2)))
        word = sources + [int(rng.choice(sources)) for _ in range(length-count)]
        rng.shuffle(word)
        if word_length is None and rng.random() < .2 and 2*len(word) <= extent+1:
            word = word*2
        word = tuple(word)
    specs = []
    for source, target, offsets in zip(sources, targets, masks):
        direction = _choose(rng, _label_options(offsets))
        specs.append(dict(source=source, target=target, offsets=tuple(sorted(offsets)),
                          label_direction=direction))
    return specs, tuple(word)


def _make_scene(rng, teaching=None, extent=4, count=None, word_length=None):
    specs, word = _specifications(rng, teaching, extent, count, word_length)
    for attempt in range(36):
        if attempt == 18 and teaching is None:
            specs, word = _specifications(rng, None, extent, len(specs), word_length)
        spacious = attempt >= 18
        height = 30 if spacious else int(rng.integers(12, 31))
        width = 30 if spacious else int(rng.integers(max(18, len(word)+10), 31))
        grid = np.zeros((height, width), dtype=np.int8)
        hh = int(rng.choice((3, 5, 7)))
        hw = int(rng.choice((3, 5)))
        top = int(rng.integers(0, height-hh+1))
        # Leave enough forward runway for several whole cycles in the teaching pairs.
        left = int(rng.integers(0, max(1, width-hw-len(word)-(10 if teaching is not None else 5))))
        hub = _hub_cells(top, left, hh, hw, _choose(rng, ("rectangle", "clipped", "cross")))
        center = top+hh//2, left+hw//2
        word_cells = {(center[0], left+hw+i) for i in range(len(word))}
        for p in hub:
            grid[p] = HUB
        for i, symbol in enumerate(word):
            grid[center[0], left+hw+i] = symbol
        # Reserve the entire hub box, all four adjacent faces, and the delimiter.
        reserved = {(r, c) for r in range(top-1, top+hh+1)
                    for c in range(left-1, left+hw+1)} | word_cells
        reserved.add((center[0], left+hw+len(word)))
        records, neutral_halo = [], set()
        for spec in specs:
            dr, dc = spec["label_direction"]
            relative = set(spec["offsets"]) | {(0, 0), (-dr, -dc), (-2*dr, -2*dc)}
            t, l, b, r = _box(relative)
            locations = [(y, x) for y in range(-t, height-b) for x in range(-l, width-r)]
            rng.shuffle(locations)
            for qr, qc in locations:
                mask = {(qr+y, qc+x) for y, x in spec["offsets"]}
                labels = {(qr-dr, qc-dc), (qr-2*dr, qc-2*dc)}
                cells = mask | labels | {(qr, qc)}
                if cells & reserved or mask & neutral_halo:
                    continue
                candidate = grid.copy()
                for p in mask:
                    candidate[p] = NEUTRAL
                candidate[qr-2*dr, qc-2*dc] = spec["source"]
                candidate[qr-dr, qc-dc] = spec["target"]
                # Check the local reading immediately; accidental label rays are rejected.
                near = {(y+a, x+b) for y, x in mask for a in (-1, 0, 1) for b in (-1, 0, 1)}
                ambiguous = False
                for old in records:
                    old_near = {(y+a, x+b) for y, x in old["mask"]
                                for a in (-1, 0, 1) for b in (-1, 0, 1)}
                    if any(p in old_near for p in labels) or any(p in near for p in old["labels"]):
                        ambiguous = True
                        break
                if ambiguous:
                    continue
                record = dict(spec, q=(qr, qc), mask=mask, labels=labels)
                records.append(record)
                grid = candidate
                reserved |= cells
                neutral_halo |= {(y+a, x+b) for y, x in mask for a in (-1, 0, 1) for b in (-1, 0, 1)}
                break
            else:
                break
        if len(records) != len(specs):
            continue
        rail = dict(hub=hub, center=center, face=hw//2, direction=(0, 1), word=word,
                    word_cells=word_cells, records=records)
        output, events = _render(grid, rail)
        minimum = 2*len(word)+1 if teaching is not None else len(word)+1
        if len(events) < minimum or any(not e["clipped"] for e in events):
            continue
        if teaching is not None:
            # Ensure uncut copies teach the actual shapes and registration.
            if not all(any(e["source"] == s["source"] and e["full"] == e["clipped"]
                           for e in events[:2*len(word)]) for s in specs):
                continue
            alternatives = ("boxes", "prefix", "centered", "once") if teaching == 0 else ("previous", "centered", "once")
            if any(np.array_equal(output, _render(grid, rail, alt)[0]) for alt in alternatives):
                continue
        try:
            decoded = _decode(grid)
            if not np.array_equal(output, _render(grid, decoded)[0]):
                continue
        except ValueError:
            continue
        turns = int(rng.integers(4))
        mirror = bool(rng.integers(2))
        def orient(a):
            a = np.rot90(a, turns)
            return np.ascontiguousarray(np.fliplr(a) if mirror else a)
        return {"input": orient(grid), "output": orient(output)}
    # A deterministic, roomy recovery path retains every pedagogical property.
    return _fallback_scene(rng, teaching, extent, count, word_length)


def _fallback_scene(rng, teaching, extent, count, word_length=None):
    """Safe geometry, used only after bounded placement exhaustion."""
    sources = [int(c) for c in rng.permutation(PAYLOAD)]
    n = 2 if teaching is not None else count or 2
    codes, targets = sources[:n], sources[n:2*n]
    if teaching == 1:
        masks = ([(0, 0), (0, 1), (0, 2), (0, 3)], [(1, 0), (2, 0)])
        word = tuple(codes)
    else:
        masks = [tuple((i, i) for i in range(3+j*2+(max(0, extent-5) if j == 2 else 0)))
                 for j in range(n)]
        word = tuple(codes) + (codes[0],)
        if word_length:
            word = tuple(codes[i % n] for i in range(word_length))
    grid = np.zeros((30, 30), dtype=np.int8)
    hub = _hub_cells(12, 0, 3, 3, "cross")
    for p in hub:
        grid[p] = HUB
    positions = {(13, 3+i) for i in range(len(word))}
    for i, color in enumerate(word):
        grid[13, 3+i] = color
    records = []
    for source, target, cells, q in zip(codes, targets, masks, ((3, 4), (20, 4), (3, 20))):
        mask = {(q[0]+r, q[1]+c) for r, c in cells}
        labels = {(q[0], q[1]-2), (q[0], q[1]-1)}
        for p in mask:
            grid[p] = NEUTRAL
        grid[q[0], q[1]-2], grid[q[0], q[1]-1] = source, target
        records.append(dict(source=source, target=target, mask=mask, labels=labels,
                            q=q, offsets=tuple(cells), label_direction=(0, 1)))
    rail = dict(hub=hub, center=(13, 1), face=1, direction=(0, 1),
                word=word, word_cells=positions, records=records)
    output, _ = _render(grid, rail)
    _require(np.array_equal(solve(grid), output), "fallback interpretation failed")
    return {"input": grid, "output": output}


def _count(value, minimum, maximum, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(name + " out of range")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(3))  # new geometry, third symbol, or longer program
    puzzle, seen, taught_geometry = {"train": [], "test": []}, set(), set()
    for split, n in (("train", num_train), ("test", num_test)):
        for i in range(n):
            teaching = i if split == "train" and i < 2 else None
            extent = 3 if split == "train" else 5 if regime == 0 else 4
            count = 3 if split == "test" and regime == 1 else None
            if split == "train" and regime == 1:
                count = 2
            length = int(rng.integers(5, 7)) if split == "test" and regime == 2 else None
            for _ in range(16):
                pair = _make_scene(rng, teaching, extent, count, length)
                geometries = {_registered_key(r["offsets"]) for r in _decode(pair["input"])["records"]}
                if split == "train" or geometries - taught_geometry:
                    break
            else:
                # A fresh nine-cell diagonal is outside all normal training spans.
                pair = _fallback_scene(rng, None, 7, 3, length)
                geometries = {_registered_key(r["offsets"]) for r in _decode(pair["input"])["records"]}
            if _key(pair["input"]) in seen:
                # Seven payload colors give 5040 finite palette permutations;
                # at most thirteen previous inputs can forbid one.
                for values in itertools.permutations(PAYLOAD):
                    mapping = np.arange(10, dtype=np.int8)
                    mapping[list(PAYLOAD)] = values
                    candidate = {k: mapping[v] for k, v in pair.items()}
                    if _key(candidate["input"]) not in seen:
                        pair = candidate
                        break
                else:
                    raise AssertionError("finite duplicate repair exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train":
                taught_geometry |= geometries
    # Presentation order is not an instruction or a hidden mask dictionary.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    grid = _grid(pair["input"])
    rail = _decode(grid)
    expected, events = _render(grid, rail)
    return rail, expected, events


def _registered_key(offsets):
    """Exact registration preserved; only whole-instruction rotations/reflections quotiented out."""
    return min(tuple(sorted(_rotate_cells({(sign*r, c) for r, c in offsets}, turns)))
               for sign in (-1, 1) for turns in range(4))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "invalid example counts")
    seen, witnesses, taught_geometry = set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            _require(grid.shape == output.shape, "output dimensions changed")
            _require(_key(grid) not in seen, "duplicate input")
            seen.add(_key(grid))
            rail, expected, events = _features(pair)
            geometries = {_registered_key(r["offsets"]) for r in rail["records"]}
            if split == "train":
                taught_geometry |= geometries
            else:
                _require(bool(geometries - taught_geometry), "test lacks fresh registered geometry")
            _require(np.array_equal(expected, output), "incorrect stencil conveyor output")
            _require(len(events) > len(rail["word"]), "insufficient repeated-word evidence")
            _require(np.all(output[grid == HUB] == HUB), "hub must be preserved")
            occupied = set(rail["hub"])
            dr, dc = rail["direction"]
            for event in events:
                _require(not event["full"] & occupied, "full stencil footprints overlap")
                _require(bool(event["clipped"]), "generated station is wholly invisible")
                _require(not event["full"] & rail["hub"], "stencil hits hub")
                for q in event["rejected"]:
                    offsets = next(r["offsets"] for r in rail["records"] if r["source"] == event["source"])
                    _require(bool({(q[0]+r, q[1]+c) for r, c in offsets} & occupied),
                             "station skipped a feasible registration position")
                occupied |= event["full"]
            paints = set().union(*(e["clipped"] for e in events))
            instruction = rail["word_cells"] | set().union(*(r["mask"] | r["labels"] for r in rail["records"]))
            for p in instruction - paints:
                _require(int(output[p]) == BACKGROUND, "unpainted instruction must disappear")
            if split == "train":
                for alt in ("boxes", "prefix", "centered", "previous", "once"):
                    if not np.array_equal(output, _render(grid, rail, alt)[0]):
                        witnesses.add(alt)
                if any((0, 0) not in r["offsets"] for r in rail["records"]):
                    witnesses.add("hole")
                if all(any(e["source"] == r["source"] and e["full"] == e["clipped"]
                           for e in events) for r in rail["records"]):
                    witnesses.add("complete")
    _require({"boxes", "prefix", "centered", "previous", "once", "hole", "complete"} <= witnesses,
             "training does not distinguish the required concepts")


def self_test():
    """Bounded checks: explicit geometry regressions and actual 42-seed diversity."""
    # Hand-authored input/output: labels 1->2 and 2->3 are read once. New 2s
    # produced by the long line do not recursively become additional 3-stencils.
    grid = np.zeros((8, 12), dtype=np.int8)
    hub = _hub_cells(2, 0, 3, 3, "cross")
    for p in hub:
        grid[p] = HUB
    grid[3, 3:5] = (1, 2)
    grid[0, 5:7] = (1, 2)
    grid[0, 7:11] = NEUTRAL
    grid[5, 6:8] = (2, 3)
    grid[6:8, 8] = NEUTRAL
    expected = np.zeros_like(grid)
    for p in hub:
        expected[p] = HUB
    expected[3, 3:12] = 2
    expected[4:6, 4] = 3
    expected[4:6, 8] = 3
    assert np.array_equal(solve(grid), expected)
    for turns in range(4):
        for mirror in (False, True):
            a, b = np.rot90(grid, turns), np.rot90(expected, turns)
            if mirror:
                a, b = np.fliplr(a), np.fliplr(b)
            assert np.array_equal(solve(a), b)
    # A long earlier stamp still blocks the third station, despite a disjoint second stamp.
    rail = dict(center=(3, 1), face=1, direction=(0, 1), word=(1, 2), records=[
        dict(source=1, target=3, offsets=((0, 0), (0, 1), (0, 2), (0, 3))),
        dict(source=2, target=4, offsets=((1, 0), (2, 0)))])
    paint, events = _packing((8, 12), rail)
    assert [e["q"] for e in events[:4]] == [(3, 3), (3, 4), (3, 7), (3, 8)]
    assert events[2]["rejected"] == ((3, 5), (3, 6))
    assert paint[(3, 3)] == 3 and paint[(4, 4)] == 4 and (3, 4) in events[0]["full"]
    # Two three-cell diagonal stencils interlock at unit pitch, despite their boxes.
    rail["records"][0]["offsets"] = ((0, 0), (1, 1), (2, 2))
    rail["records"][1]["offsets"] = ((0, 0), (1, 1), (2, 2))
    _, events = _packing((8, 12), rail)
    assert [e["q"][1] for e in events[:4]] == [3, 4, 5, 6]
    assert any(e["full"] != e["clipped"] for e in events)
    assert _packing((8, 12), rail, "boxes")[1][1]["q"][1] == 6
    # Recovery paths must preserve the witnesses, including empty registration.
    for teaching in (0, 1, None):
        pair = _fallback_scene(np.random.default_rng(173+int(teaching or 0)), teaching, 4, None)
        assert np.array_equal(solve(pair["input"]), pair["output"])
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, shapes, programs, geometries = set(), set(), set(), set()
    directions, hubs, targets, kinds, instruction_counts, station_counts = set(), set(), set(), set(), set(), set()
    changed_within = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        local_geometries = set()
        for s in ("train", "test"):
            for pair in puzzle[s]:
                rail, expected, events = _features(pair)
                assert np.array_equal(solve(pair["input"]), pair["output"])
                shapes.add(pair["input"].shape)
                directions.add(rail["direction"])
                instruction_counts.add(len(rail["records"]))
                station_counts.add(len(events))
                t, l, b, r = _box(rail["hub"])
                hubs.add((b-t+1, r-l+1, len(rail["hub"])))
                role_map = {}
                programs.add(tuple(role_map.setdefault(c, len(role_map)) for c in rail["word"]))
                for rec in rail["records"]:
                    offsets = tuple(sorted(rec["offsets"]))
                    geometries.add(offsets)
                    local_geometries.add(offsets)
                    targets.add((rec["source"], rec["target"]))
                    if (0, 0) in offsets:
                        kinds.add("occupied")
                    else:
                        t, l, b, r = _box(offsets)
                        kinds.add("interior" if t < 0 < b and l < 0 < r else "other_hole")
        changed_within += len(local_geometries) > 2
    assert len(signatures) == 42 and len(shapes) >= 50
    assert len(geometries) >= 70 and len(programs) >= 12
    assert directions == set(DIRECTIONS) and len(hubs) >= 10
    assert instruction_counts == {1, 2, 3} and len(station_counts) >= 12
    assert kinds == {"occupied", "interior", "other_hole"} and len(targets) >= 25
    assert changed_within >= 35
    p = generate_puzzle(923, 10, 4)
    assert tuple(_key(x["input"]) for x in p["train"]+p["test"]) == tuple(
        _key(x["input"]) for x in generate_puzzle(923, 10, 4)["train"]+generate_puzzle(923, 10, 4)["test"])
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(107+a*7+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = generate_puzzle(901, 2, 1)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("incorrect output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and all legal explicit counts)")
