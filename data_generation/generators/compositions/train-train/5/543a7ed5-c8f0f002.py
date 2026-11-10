"""Color broadcast wells: contained pigment defines a scene-wide substitution.

Only original cells are broadcast-recolored. A well additionally becomes solid
pigment and gains an exterior frame in its old wall color. No parent code or
generation state is needed by solve().
"""

from __future__ import annotations

import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= a) & (a <= 9)), "color bounds")
    return a.astype(int, copy=False)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    parts = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        parts.append(cells)
    return parts


def _bbox(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr), min(cc), max(cc)


def _border_mask(shape, box):
    r0, r1, c0, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0 - 1, c0 - 1:c1 + 2] = True
    mask[r1 + 1, c0 - 1:c1 + 2] = True
    mask[r0 - 1:r1 + 2, c0 - 1] = True
    mask[r0 - 1:r1 + 2, c1 + 1] = True
    return mask


def _read_scene(a):
    """Read a distributed palette dictionary, then find its rectangular wells.

    Colors lacking a key-bearing well are protected, even if a component is a
    solid rectangle. The background is explicitly the uniform input boundary.
    """
    bg = int(a[0, 0])
    _require(all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])),
             "input boundary must be background")
    candidates, mapping = [], {}
    for color in sorted(int(c) for c in np.unique(a) if c != bg):
        for cells in _components(a == color):
            r0, r1, c0, c1 = box = _bbox(cells)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            tile = a[r0:r1 + 1, c0:c1 + 1]
            if not all(np.all(edge == color) for edge in
                       (tile[0], tile[-1], tile[:, 0], tile[:, -1])):
                continue
            _require(len(cells) == int(np.count_nonzero(tile == color)),
                     "well wall must be one connected component")
            foreign = set(int(c) for c in np.unique(tile)) - {bg, color}
            _require(len(foreign) <= 1, "ambiguous well pigment")
            pigment = next(iter(foreign)) if foreign else None
            candidates.append({"source": color, "box": box, "sample": pigment})
            if pigment is not None:
                _require(color not in mapping or mapping[color] == pigment,
                         "conflicting samples for one wall color")
                _require(np.any(tile == bg), "key must also show background to fill")
                mapping[color] = pigment
    _require(mapping, "at least one visible pigment key is required")
    _require(not (set(mapping) & set(mapping.values())), "source and pigment roles overlap")
    wells = [item for item in candidates if item["source"] in mapping]
    occupied = np.zeros(a.shape, dtype=bool)
    for item in wells:
        s = item["source"]
        r0, r1, c0, c1 = item["box"]
        _require(1 <= r0 <= r1 < a.shape[0] - 1 and
                 1 <= c0 <= c1 < a.shape[1] - 1, "well has no exterior clearance")
        region = occupied[r0 - 1:r1 + 2, c0 - 1:c1 + 2]
        _require(not np.any(region), "expanded well rectangles overlap")
        region[:] = True
        _require(np.all(a[_border_mask(a.shape, item["box"])] == bg),
                 "exterior frame would overwrite content")
        _require(item["sample"] in (None, mapping[s]), "wrong local sample")
    return bg, mapping, wells


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Apply the fixed family rule using this input alone; never mutate it."""
    a = _grid(input_grid)
    bg, mapping, wells = _read_scene(a)
    out = a.copy()
    # Broadcast on ORIGINAL masks, so replacement is simultaneous.
    for source, pigment in mapping.items():
        out[a == source] = pigment
    for item in wells:
        source = item["source"]
        r0, r1, c0, c1 = item["box"]
        out[r0:r1 + 1, c0:c1 + 1] = mapping[source]
        out[_border_mask(a.shape, item["box"])] = source
    return out


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _tile(rng, h, w, source, pigment, bg, mode, keyed):
    tile = np.full((h, w), source, dtype=int)
    if mode == "solid":
        _require(not keyed, "solid cannot supply a pigment key")
        return tile
    if mode == "split":
        tile[1:-1, 1:-1] = bg
        if w >= 5:
            tile[:, w // 2] = source
        else:
            tile[h // 2, :] = source
    elif mode == "step" and min(h, w) >= 5:
        tile[1:-1, 1:-1] = bg
        tile[1:1 + max(1, (h - 2) // 2), 1:1 + max(1, (w - 2) // 2)] = source
    elif mode == "rect":
        hh = int(rng.integers(1, h - 1))
        ww = int(rng.integers(1, w - 1))
        # A keyed gap must have at least two cells.
        if keyed and hh * ww < 2:
            hh, ww = h - 2, w - 2
        rr = int(rng.integers(1, h - hh))
        cc = int(rng.integers(1, w - ww))
        tile[rr:rr + hh, cc:cc + ww] = bg
    else:
        tile[1:-1, 1:-1] = bg
    if keyed:
        # Confine the sample to one cavity: other cavities must also be filled.
        holes = _components(tile == bg)
        cavity = max(holes, key=len)
        sample_size = min(int(rng.integers(1, 4)), len(cavity) - 1)
        sample_size = max(1, sample_size)
        for i in rng.choice(len(cavity), size=sample_size, replace=False):
            tile[cavity[int(i)]] = pigment
        _require(np.any(tile == bg), "key consumed every empty cell")
    _require(len(_components(tile == source)) == 1, "disconnected well wall")
    return tile


def _pack(rng, tiles, h, w):
    order = sorted(range(len(tiles)), key=lambda i: tiles[i].size, reverse=True)
    for _ in range(5):
        occupied = np.zeros((h, w), dtype=bool)
        positions = [None] * len(tiles)
        for i in order:
            th, tw = tiles[i].shape
            possible = rng.permutation((h - th - 1) * (w - tw - 1))
            for value in possible:
                rr, cc = divmod(int(value), w - tw - 1)
                area = occupied[rr:rr + th + 2, cc:cc + tw + 2]
                if not np.any(area):
                    area[:] = True
                    positions[i] = (rr + 1, cc + 1)
                    break
            else:
                break
        else:
            return positions, occupied
    return None


def _fragment(rng, a, occupied, color, preferred=None):
    """A separated dot, bar or open shape, never a complete 3x3+ well."""
    mode = preferred or str(rng.choice(("dot", "bar", "ell", "stair")))
    if mode == "dot":
        mask = np.ones((1, 1), dtype=bool)
    elif mode == "bar":
        mask = np.ones((1, int(rng.integers(2, 6))), dtype=bool)
        if rng.integers(2):
            mask = mask.T
    elif mode == "ell":
        mask = np.zeros((int(rng.integers(3, 6)), int(rng.integers(3, 6))), dtype=bool)
        mask[:, 0] = True
        mask[-1, :] = True
        mask = np.rot90(mask, int(rng.integers(4)))
    else:
        mask = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool)
        mask = np.rot90(mask, int(rng.integers(4)))
    for candidate in (mask, np.ones((1, 1), dtype=bool)):
        th, tw = candidate.shape
        for v in rng.permutation((a.shape[0] - th - 1) * (a.shape[1] - tw - 1)):
            rr, cc = divmod(int(v), a.shape[1] - tw - 1)
            rr, cc = rr + 1, cc + 1
            if not np.any(occupied[rr:rr + th, cc:cc + tw]):
                a[rr:rr + th, cc:cc + tw][candidate] = color
                occupied[max(0, rr - 1):rr + th + 1, max(0, cc - 1):cc + tw + 1] = True
                return
    raise RuntimeError("no space for a source/pigment fragment")


def _draw_pair(rng, bg, full_mapping, protected, split, index):
    sources = list(full_mapping)
    # Both minimal demonstrations and tests contain every puzzle-level mapping.
    if split == "train" and index >= 2:
        n = int(rng.integers(1, len(sources) + 1))
        sources = sorted(int(x) for x in rng.choice(sources, n, replace=False))
    tiles = []
    for i, source in enumerate(sources):
        h, w = int(rng.integers(4, 8)), int(rng.integers(4, 8))
        mode = str(rng.choice(("ring", "rect", "step", "split")))
        if mode == "split" and max(h, w) < 5:
            w = 5
        if split == "train" and index == 1 and i == 0:
            h, w, mode = 6, 7, "split"
        if split == "test" and i == 0:
            h, w = int(rng.integers(10, 14)), int(rng.integers(3, 5))
            if mode == "split" and w < 5:
                mode = "ring"  # long narrow keys retain a substantial cavity
            if rng.integers(2):
                h, w = w, h
        tiles.append(_tile(rng, h, w, source, full_mapping[source], bg, mode, True))
    extras = (1 if split == "train" and index < 2 else
              int(rng.integers(1 if split == "test" else 0, 3)))
    for j in range(extras):
        source = int(rng.choice(sources))
        mode = ("ring" if split == "train" and index == 0 else
                "solid" if split == "train" and index == 1 else
                str(rng.choice(("solid", "ring", "step", "split", "rect"))))
        h, w = int(rng.integers(3, 8)), int(rng.integers(3, 8))
        if mode == "split" and max(h, w) < 5:
            w = 5
        tiles.append(_tile(rng, h, w, source, full_mapping[source], bg, mode, False))
    # Occasionally repeat a key; consistency rather than unique occurrence is the rule.
    if len(tiles) < 5 and index >= 2 and rng.random() < .25:
        source = int(rng.choice(sources))
        tiles.append(_tile(rng, 4, 5, source, full_mapping[source], bg, "ring", True))
    for attempt in range(30):
        low = 13 if len(tiles) == 1 else 17 if len(tiles) == 2 else 22 if len(tiles) == 3 else 25
        h, w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        h = max(h, max(t.shape[0] + 2 for t in tiles))
        w = max(w, max(t.shape[1] + 2 for t in tiles))
        if attempt >= 20:
            h = w = 30
        packed = _pack(rng, tiles, h, w)
        if packed is not None:
            positions, occupied = packed
            break
    else:
        raise RuntimeError("bounded rectangle placement failed")
    a = np.full((h, w), bg, dtype=int)
    for tile, (rr, cc) in zip(tiles, positions):
        a[rr:rr + tile.shape[0], cc:cc + tile.shape[1]] = tile
    for i, source in enumerate(sources):
        preferred = "ell" if split == "train" and index == 0 and i == 0 else None
        _fragment(rng, a, occupied, source, preferred)
        if rng.random() < .45:
            _fragment(rng, a, occupied, source)
    # Existing pigment outside a well is copied, not expanded or recolored.
    for pigment in sorted({full_mapping[s] for s in sources}):
        _fragment(rng, a, occupied, pigment, "bar")
    for color in protected:
        if rng.random() < .55:
            _fragment(rng, a, occupied, color)
    return {"input": a, "output": solve(a)}


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    palette = [int(x) for x in rng.permutation(10)]
    bg = palette[0]
    nsource = int(rng.integers(1, 4))
    ntarget = int(rng.integers(1, nsource + 1))
    sources = palette[1:1 + nsource]
    pigments = palette[1 + nsource:1 + nsource + ntarget]
    assignments = [pigments[i % ntarget] for i in range(nsource)]
    rng.shuffle(assignments)
    mapping = dict(zip(sources, assignments))
    protected = palette[1 + nsource + ntarget:3 + nsource + ntarget]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(25):
                pair = _draw_pair(rng, bg, mapping, protected, split, index)
                key = _key(pair["input"])
                if key in seen:
                    continue
                if split == "train" and index == 1 and pair["input"].shape == puzzle["train"][0]["input"].shape:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("could not construct a distinct demonstration")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect(a):
    bg, mapping, wells = _read_scene(a)
    well_mask = np.zeros(a.shape, dtype=bool)
    info = {"hollow_receiver": False, "solid_receiver": False,
            "detached_cavity": False, "irregular_cavity": False,
            "edge": False, "interior": False, "ratio": 0.0,
            "wells": len(wells), "keys": 0, "outside_source": 0,
            "protected_outside": 0, "mapping": mapping, "background": bg}
    for item in wells:
        r0, r1, c0, c1 = item["box"]
        well_mask[r0:r1 + 1, c0:c1 + 1] = True
        tile = a[r0:r1 + 1, c0:c1 + 1]
        h, w = tile.shape
        info["ratio"] = max(info["ratio"], h / w, w / h)
        holes = _components(tile == bg)
        keyed = item["sample"] is not None
        info["keys"] += keyed
        info["hollow_receiver"] |= not keyed and bool(holes)
        info["solid_receiver"] |= not keyed and not holes
        # A cavity with no pigment in its original hole (background+pigment) is
        # invisible to a flood-fill started only from the pigment sample.
        voids = _components(tile != item["source"])
        info["detached_cavity"] |= keyed and any(
            all(tile[cell] == bg for cell in part) for part in voids)
        for part in voids:
            y0, y1, x0, x1 = _bbox(part)
            info["irregular_cavity"] |= len(part) != (y1 - y0 + 1) * (x1 - x0 + 1)
        edge = r0 == 1 or c0 == 1 or r1 == a.shape[0] - 2 or c1 == a.shape[1] - 2
        info["edge"] |= edge
        info["interior"] |= not edge
    for source in mapping:
        info["outside_source"] += int(np.count_nonzero((a == source) & ~well_mask))
    for pigment in set(mapping.values()):
        info["protected_outside"] += int(np.count_nonzero((a == pigment) & ~well_mask))
    return info


def validate_generated_puzzle(puzzle) -> None:
    """Check geometry, broadcasts, frame precedence and teaching witnesses."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split types")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "pair counts")
    seen, shared_mapping, backgrounds = set(), {}, set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == out.shape, "shape changed")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            bg, mapping, wells = _read_scene(a)
            backgrounds.add(bg)
            for s, t in mapping.items():
                _require(s not in shared_mapping or shared_mapping[s] == t, "puzzle mapping changed")
                shared_mapping[s] = t
            # Original non-background cells obey only the global substitution.
            for color in np.unique(a):
                color = int(color)
                if color != bg:
                    _require(np.all(out[a == color] == mapping.get(color, color)),
                             "original cell violates global substitution/protection")
            allowed = np.zeros(a.shape, dtype=bool)
            for item in wells:
                s = item["source"]
                r0, r1, c0, c1 = item["box"]
                _require(np.all(out[r0:r1 + 1, c0:c1 + 1] == mapping[s]),
                         "well did not become solid pigment")
                allowed[r0:r1 + 1, c0:c1 + 1] = True
                border = _border_mask(a.shape, item["box"])
                _require(np.all(out[border] == s), "old wall color must frame its own well")
                allowed |= border
            _require(np.all(out[(a == bg) & ~allowed] == bg), "exterior background changed")
            for source in mapping:
                _require(np.any((a == source) & ~allowed), "source has no remote broadcast witness")
            for pigment in set(mapping.values()):
                _require(np.any((a == pigment) & ~allowed), "pigment has no protected exterior witness")
            info = _inspect(a)
            _require(info["outside_source"] > 0 and info["protected_outside"] > 0,
                     "missing broadcast/protected witnesses")
            infos[split].append(info)
    _require(len(backgrounds) == 1, "background changed within puzzle")
    _require(not (set(shared_mapping) & set(shared_mapping.values())), "puzzle roles overlap")
    train = infos["train"]
    for field in ("hollow_receiver", "solid_receiver", "detached_cavity"):
        _require(any(info[field] for info in train), f"training lacks {field}")
    _require(set(train[0]["mapping"]) == set(shared_mapping) == set(train[1]["mapping"]),
             "first two demonstrations must show every mapping")
    _require(len({np.asarray(p["input"]).shape for p in puzzle["train"]}) >= 2,
             "training canvas shapes do not vary")
    train_ratio = max(info["ratio"] for info in train)
    _require(all(info["ratio"] > train_ratio for info in infos["test"]),
             "test must hold out a more elongated well")


def self_test():
    """Bounded 42-seed coverage plus semantic and symmetry checks."""
    shapes, layouts, palettes, well_counts, ratios = set(), set(), set(), set(), set()
    mappings, signatures = set(), set()
    counts = {k: 0 for k in ("hollow_receiver", "solid_receiver", "detached_cavity",
                              "irregular_cavity", "edge", "interior")}
    many_to_one = repeated_keys = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p["input"]) for split in ("train", "test") for p in puzzle[split]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                before = a.copy()
                _require(np.array_equal(solve(a), pair["output"]), "solve disagrees")
                _require(np.array_equal(a, before), "solve mutated input")
                info = _inspect(a)
                mapping = info["mapping"]
                shapes.add(a.shape)
                normalized, roles = [], {}
                for c in a.flat:
                    normalized.append(roles.setdefault(int(c), len(roles)))
                layouts.add((a.shape, tuple(normalized)))
                palettes.add(tuple(sorted(int(c) for c in np.unique(a))))
                mappings.add(tuple(sorted(mapping.items())))
                well_counts.add(info["wells"])
                ratios.add(round(info["ratio"], 3))
                for field in counts:
                    counts[field] += int(info[field])
                many_to_one += len(set(mapping.values())) < len(mapping)
                repeated_keys += info["keys"] > len(mapping)
        # Rule must be invariant under orientation and palette renaming.
        pair = puzzle["test"][0]
        a, out = pair["input"], pair["output"]
        _require(np.array_equal(solve(np.rot90(a)), np.rot90(out)), "rotation failure")
        _require(np.array_equal(solve(np.fliplr(a)), np.fliplr(out)), "reflection failure")
        permutation = np.random.default_rng(seed + 1000).permutation(10)
        _require(np.array_equal(solve(permutation[a]), permutation[out]), "palette failure")
        if seed == 0:
            # Changing a key changes a distant fragment and an unseeded well,
            # but not the source-colored exterior frames.
            a = puzzle["train"][0]["input"]
            bg, mapping, wells = _read_scene(a)
            source = next(s for s in mapping if any(
                w["source"] == s and w["sample"] is None for w in wells))
            new = next(c for c in range(10) if c not in set(np.unique(a)))
            changed = a.copy()
            for well in wells:
                if well["source"] == source and well["sample"] is not None:
                    r0, r1, c0, c1 = well["box"]
                    tile = changed[r0:r1 + 1, c0:c1 + 1]
                    tile[tile == mapping[source]] = new
            result = solve(changed)
            _require(np.all(result[a == source] == new), "key failed to broadcast")
            receiver = next(w for w in wells if w["source"] == source and w["sample"] is None)
            r0, r1, c0, c1 = receiver["box"]
            _require(np.all(result[r0:r1 + 1, c0:c1 + 1] == new), "unseeded receiver ignored key")
            _require(np.all(result[_border_mask(a.shape, receiver["box"])] == source),
                     "new frame was incorrectly substituted")
    _require(len(signatures) == 42 and len(layouts) >= 200, "structural diversity")
    _require(len(shapes) >= 35 and len(palettes) >= 25 and len(mappings) >= 25, "coverage diversity")
    _require({1, 2, 3, 4, 5} <= well_counts, "well-count diversity")
    _require(all(value >= 10 for value in counts.values()), "missing geometric regimes")
    _require(many_to_one >= 15 and repeated_keys >= 5, "missing broadcast regimes")
    for train, test in ((2, 1), (10, 1), (10, 4), (5, 4)):
        puzzle = generate_puzzle(700 + train + test, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test), "explicit counts")
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first, second = generate_puzzle(29), generate_puzzle(29)
    _require(all(np.array_equal(x[k], y[k]) for split in ("train", "test")
                 for x, y in zip(first[split], second[split]) for k in ("input", "output")),
             "non-deterministic seed")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "canvas_shapes": len(shapes), "normalized_layouts": len(layouts),
            "palettes": len(palettes), "visible_maps": len(mappings),
            "well_counts": sorted(well_counts), "aspect_ratios": len(ratios),
            "many_to_one_pairs": many_to_one, "repeated_key_pairs": repeated_keys,
            **counts}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
