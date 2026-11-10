"""Registered echoes: crop addresses transfer from pixels to the tile lattice.

Only standard-library modules and NumPy are used. No source files, saved pairs,
seeds, training labels, caches, or generation metadata are consulted by solve().
All generation loops and diagnostics are bounded; the module writes no files.
"""
from collections import Counter
import json

import numpy as np


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= s <= 30 for s in a.shape), "invalid grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "cells must be integers")
    _need(bool(np.all((0 <= a) & (a <= 9))), "invalid ARC colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        _need(not isinstance(value, (bool, np.bool_)) and
              isinstance(value, (int, np.integer)) and low <= value <= high,
              name + " is outside its legal integer range")
    _need(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _orbit(template):
    rotations = [np.rot90(template, k).copy() for k in range(4)]
    return rotations + [np.fliplr(a).copy() for a in rotations]


def _catalog(template):
    """Index every eligible proper crop under every D4 transformation.

    Each entry stores *all* registrations, including competing source offsets.
    A crop is usable only when its complete image AND its address are unique.
    """
    n = len(template)
    images = _orbit(template)
    index = {}
    for orientation, image in enumerate(images):
        for h in range(2, n + 1):
            for w in range(2, n + 1):
                if h == w == n:
                    continue
                for r in range(n - h + 1):
                    for c in range(n - w + 1):
                        patch = image[r:r+h, c:c+w]
                        if patch[0, 0] and not patch.all():
                            index.setdefault(_key(patch), []).append((orientation, r, c, h, w))
    return images, index


def _runs(indices):
    result = []
    for i in map(int, indices):
        if result and result[-1][1] + 1 == i:
            result[-1] = (result[-1][0], i)
        else:
            result.append((i, i))
    return result


def _lattice(a):
    divider = int(a[0, 0])
    rg = _runs(np.flatnonzero(np.all(a == divider, axis=1)))
    cg = _runs(np.flatnonzero(np.all(a == divider, axis=0)))
    _need(len(rg) >= 4 and len(cg) >= 4, "missing divider lattice")
    _need(rg[0][0] == cg[0][0] == 0 and
          rg[-1][1] == a.shape[0]-1 and cg[-1][1] == a.shape[1]-1,
          "lattice needs complete outer dividers")
    thicknesses = {b-a+1 for a, b in rg+cg}
    hs = {rg[i+1][0]-rg[i][1]-1 for i in range(len(rg)-1)}
    ws = {cg[i+1][0]-cg[i][1]-1 for i in range(len(cg)-1)}
    _need(len(thicknesses) == len(hs) == len(ws) == 1, "irregular lattice")
    t, h, w = thicknesses.pop(), hs.pop(), ws.pop()
    _need(t in (1, 2) and h == w and h in (3, 4), "unsupported tile geometry")
    rows, cols = len(rg)-1, len(cg)-1
    _need(rows >= h and cols >= w, "reference window cannot fit")
    return (h, t, rows, cols), divider


def _tile_slice(geometry, position):
    n, t, _, _ = geometry
    r, c = position
    y, x = t+r*(n+t), t+c*(n+t)
    return np.s_[y:y+n, x:x+n]


def _active(image, origin):
    return frozenset((origin[0]+int(r), origin[1]+int(c))
                     for r, c in np.argwhere(image))


def _field(images, echoes):
    """A bounded map from coarse positions to unions of microscopic ink."""
    field, coverage = {}, Counter()
    for echo in echoes:
        image = images[echo["reg"][0]]
        for position in echo["active"]:
            if position in field:
                field[position] |= image
            else:
                field[position] = image.copy()
            coverage[position] += 1
    return field, coverage


def _render(input_grid, geometry, images, echoes, background, ink):
    result = input_grid.copy()
    field, _ = _field(images, echoes)
    for position, mask in field.items():
        result[_tile_slice(geometry, position)] = np.where(mask, ink, background)
    return result


def _decode(input_grid):
    """Recover geometry, roles, cropped images and addresses from input alone."""
    a = _grid(input_grid)
    geometry, divider = _lattice(a)
    n, t, rows, cols = geometry
    records = {}
    for r in range(rows):
        for c in range(cols):
            tile = a[_tile_slice(geometry, (r, c))]
            _need(divider not in tile, "divider color appears inside a tile")
            records.setdefault(_key(tile), [tile.copy(), []])[1].append((r, c))
    ordinary, ordinary_positions = max(records.values(), key=lambda item: len(item[1]))
    _need(len(ordinary_positions) * 2 > rows*cols, "ordinary tile needs a strict majority")
    seed_records = [(tile, positions) for tile, positions in records.values()
                    if not np.array_equal(tile, ordinary)]
    seed_count = sum(len(positions) for _, positions in seed_records)
    _need(1 <= seed_count <= 3, "need one to three cropped chips")
    ordinary_colors = set(map(int, np.unique(ordinary)))
    _need(len(ordinary_colors) == 2, "ordinary reference must be binary")
    first = seed_records[0][0]
    eraser = int(first[-1, -1])
    common = ordinary_colors & set(map(int, np.unique(first)))
    _need(len(common) == 1, "reference and chips need one shared background")
    background = common.pop()
    ordinary_ink = (ordinary_colors - {background}).pop()
    new_ink = set(map(int, np.unique(first))) - {eraser, background}
    _need(len(new_ink) == 1, "chip needs a unique echo ink")
    ink = new_ink.pop()
    roles = (divider, background, ordinary_ink, ink, eraser)
    _need(len(set(roles)) == 5, "five roles must have distinct colors")
    template = (ordinary == ordinary_ink).astype(np.uint8)
    _need(3 <= int(template.sum()) <= n*n-2, "invalid reference density")
    _need(all((template[0].any(), template[-1].any(),
               template[:, 0].any(), template[:, -1].any())), "reference ink must touch every side")
    images, index = _catalog(template)
    _need(len({_key(image) for image in images}) == 8, "reference must have no D4 symmetry")
    echoes = []
    for tile, positions in seed_records:
        _need(set(map(int, np.unique(tile))) == {background, ink, eraser}, "inconsistent chip colors")
        ys, xs = np.nonzero(tile != eraser)
        _need(len(ys) > 0, "empty chip")
        h, w = int(ys.max())+1, int(xs.max())+1
        support = np.zeros((n, n), dtype=bool)
        support[:h, :w] = True
        _need(np.array_equal(tile != eraser, support), "crop must be packed in the upper left")
        _need(2 <= h <= n and 2 <= w <= n and (h, w) != (n, n), "crop must be proper")
        patch = (tile[:h, :w] == ink).astype(np.uint8)
        _need(bool(patch[0, 0]) and not patch.all(), "anchor must be ink and both roles visible")
        matches = index.get(_key(patch), [])
        _need(len(matches) == 1, "crop image or source offset is ambiguous")
        reg = matches[0]
        orientation, r, c, _, _ = reg
        for position in positions:
            origin = position[0]-r, position[1]-c
            _need(0 <= origin[0] <= rows-n and 0 <= origin[1] <= cols-n,
                  "an echo would cross the lattice boundary")
            active = _active(images[orientation], origin)
            _need(position in active, "chip anchor must be active")
            echoes.append({"reg": reg, "origin": origin, "position": position, "active": active})
    _need(len({(e["reg"][0], e["origin"]) for e in echoes}) == len(echoes),
          "duplicate echoes carry no extra information")
    return {"input": a, "geometry": geometry, "roles": roles, "ordinary": ordinary,
            "template": template, "images": images, "echoes": echoes}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve any valid family input without a seed or demonstrations."""
    info = _decode(input_grid)
    _, background, _, ink, _ = info["roles"]
    return _render(info["input"], info["geometry"], info["images"], info["echoes"], background, ink)


def _geometry(rng, count):
    n = int(rng.choice((3, 4), p=(.55, .45)))
    t = int(rng.choice((1, 2), p=(.75, .25)))
    maximum = (30-t)//(n+t)
    rows, cols = int(rng.integers(n, maximum+1)), int(rng.integers(n, maximum+1))
    # Three distinct echoes need more room for exclusive regions on 3x3 tiles.
    if count == 3 and n == rows == cols == 3:
        rows += 1
    return n, t, rows, cols


def _draw_template(rng, n):
    """Small procedural images: paths, frames, combs and arbitrary polyominoes."""
    for _ in range(48):
        mode = int(rng.integers(4))
        image = np.zeros((n, n), dtype=np.uint8)
        if mode == 0:
            count = int(rng.integers(3, n*n-1))
            image.flat[rng.choice(n*n, count, replace=False)] = 1
        elif mode == 1:
            r, c = int(rng.integers(n)), int(rng.integers(n))
            for _ in range(int(rng.integers(n+2, 3*n+1))):
                image[r, c] = 1
                if rng.integers(2):
                    r = int(rng.integers(n))
                else:
                    c = int(rng.integers(n))
                image[r, c] = 1
        elif mode == 2:
            image[0, :] = image[-1, :] = 1
            image[:, 0] = image[:, -1] = 1
            edits = rng.choice(n*n, int(rng.integers(2, n+2)), replace=False)
            image.flat[edits] ^= 1
        else:
            image[::2, :] = 1
            image[:, int(rng.integers(n))] = 1
            edits = rng.choice(n*n, int(rng.integers(2, n+2)), replace=False)
            image.flat[edits] ^= 1
        if not (3 <= image.sum() <= n*n-2 and all((image[0].any(), image[-1].any(),
                                                 image[:, 0].any(), image[:, -1].any()))):
            continue
        images, index = _catalog(image)
        if len({_key(s) for s in images}) != 8:
            continue
        patches = [v[0] for v in index.values() if len(v) == 1]
        if len(patches) >= 4 and any(r or c for _, r, c, _, _ in patches):
            return image, images, patches
    return None


def _options(images, patches, geometry):
    n, _, rows, cols = geometry
    active_cache, result = {}, []
    for reg in patches:
        k, r, c, _, _ = reg
        for y in range(rows-n+1):
            for x in range(cols-n+1):
                origin = (y, x)
                if (k, origin) not in active_cache:
                    active_cache[k, origin] = _active(images[k], origin)
                active = active_cache[k, origin]
                result.append({"reg": reg, "origin": origin, "position": (y+r, x+c), "active": active})
    return result


def _witnesses(images, echoes):
    """Nontrivial registration, full copying, overlap and exclusive-region evidence."""
    seed_positions = {e["position"] for e in echoes}
    if len(seed_positions) != len(echoes):
        return False
    if len({(e["reg"][0], e["origin"]) for e in echoes}) != len(echoes):
        return False
    if not any(e["reg"][1] or e["reg"][2] for e in echoes):
        return False
    if len(echoes) == 1:
        return echoes[0]["reg"][0] != 0 and bool(echoes[0]["active"]-seed_positions)
    if len({e["reg"][0] for e in echoes}) < 2:
        return False
    for i, echo in enumerate(echoes):
        others = set().union(*(e["active"] for j, e in enumerate(echoes) if i != j))
        if not (echo["active"]-others-seed_positions):
            return False
    # All echoes belong to one interacting group, including at seed tiles.
    reached = {0}
    for _ in range(len(echoes)):
        reached |= {j for i in reached for j in range(len(echoes))
                    if echoes[i]["active"] & echoes[j]["active"]}
    if len(reached) != len(echoes):
        return False
    for i, a in enumerate(echoes):
        for b in echoes[i+1:]:
            sa, sb = images[a["reg"][0]], images[b["reg"][0]]
            if a["reg"][0] == b["reg"][0] or not (sa & sb).any() or (sa | sb).all():
                continue
            # A two-contributor collision distinguishes union from overwrite,
            # XOR, and painting the entire colliding tile solid.
            for position in (a["active"] & b["active"])-seed_positions:
                if sum(position in e["active"] for e in echoes) == 2:
                    return True
    return False


def _choose_echoes(rng, images, patches, geometry, count):
    options = _options(images, patches, geometry)
    if len(options) < count:
        return None
    for _ in range(256):
        echoes = [options[int(i)] for i in rng.choice(len(options), count, replace=False)]
        if _witnesses(images, echoes):
            return echoes
    return None


def _assemble(template, images, echoes, geometry, roles):
    n, t, rows, cols = geometry
    divider, background, ordinary_ink, ink, eraser = roles
    ordinary = np.where(template, ordinary_ink, background).astype(np.int8)
    a = np.full((rows*(n+t)+t, cols*(n+t)+t), divider, dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            a[_tile_slice(geometry, (r, c))] = ordinary
    for echo in echoes:
        k, r, c, h, w = echo["reg"]
        chip = np.full((n, n), eraser, dtype=np.int8)
        chip[:h, :w] = np.where(images[k][r:r+h, c:c+w], ink, background)
        a[_tile_slice(geometry, echo["position"])] = chip
    return {"input": a, "output": _render(a, geometry, images, echoes, background, ink)}


def _fallback(count, roles, used):
    """Sixteen translated inputs per count guarantee a free input among <=13 used.

    These verified small configurations live in a 4x4 part of a 7x7 lattice.
    Fallback affects presentation preferences only; it obeys the full rule.
    """
    template = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1]], dtype=np.uint8)
    images = _orbit(template)
    configurations = {
        1: (((7, 0, 1, 3, 2), (0, 0)),),
        2: (((6, 1, 0, 2, 3), (1, 1)), ((7, 0, 1, 3, 2), (0, 0))),
        3: (((6, 1, 0, 2, 3), (1, 0)), ((0, 1, 0, 2, 3), (0, 1)),
            ((6, 1, 0, 2, 3), (1, 1))),
    }
    for dy in range(4):
        for dx in range(4):
            echoes = []
            for reg, base_origin in configurations[count]:
                origin = base_origin[0]+dy, base_origin[1]+dx
                position = origin[0]+reg[1], origin[1]+reg[2]
                echoes.append({"reg": reg, "origin": origin, "position": position,
                               "active": _active(images[reg[0]], origin)})
            pair = _assemble(template, images, echoes, (3, 1, 7, 7), roles)
            if _key(pair["input"]) not in used:
                return pair
    raise AssertionError("finite fallback bank exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    _need(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)),
          "seed must be an integer")
    rng = np.random.default_rng(seed)
    stable_palette = bool(rng.random() < .4)
    stable_geometry = bool(rng.random() < .25)
    base_roles = tuple(map(int, rng.choice(10, 5, replace=False)))
    base_geometry = _geometry(rng, 3)
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for j in range(count):
            # The first two demonstrations isolate one echo, then show a union.
            # Three-way superposition is the compositional test shift.
            seed_count = 3 if split == "test" else (1 if j == 0 else 2 if j == 1 else int(rng.integers(1, 3)))
            roles = base_roles if stable_palette else tuple(map(int, rng.choice(10, 5, replace=False)))
            pair = None
            for _ in range(48):
                geometry = base_geometry if stable_geometry else _geometry(rng, seed_count)
                drawn = _draw_template(rng, geometry[0])
                if drawn is None:
                    continue
                template, images, patches = drawn
                echoes = _choose_echoes(rng, images, patches, geometry, seed_count)
                if echoes is None:
                    continue
                candidate = _assemble(template, images, echoes, geometry, roles)
                if _key(candidate["input"]) not in used:
                    pair = candidate
                    break
            if pair is None:
                pair = _fallback(seed_count, roles, used)
            puzzle[split].append(pair)
            used.add(_key(pair["input"]))
    return puzzle


def _pair_invariants(pair):
    info = _decode(pair["input"])
    out = _grid(pair["output"])
    a, geometry = info["input"], info["geometry"]
    _need(out.shape == a.shape, "output shape changed")
    divider, background, ordinary_ink, ink, eraser = info["roles"]
    images, echoes = info["images"], info["echoes"]
    field, coverage = _field(images, echoes)
    _need(_witnesses(images, echoes), "generation lacks essential concept witnesses")
    # Check tile contents and unchanged outside support, separately from solve.
    n, t, rows, cols = geometry
    divider_mask = a == divider
    _need(np.array_equal(out[divider_mask], a[divider_mask]), "divider changed")
    _need(not np.any(out == eraser), "an erased area was not reconstructed")
    for r in range(rows):
        for c in range(cols):
            sl = _tile_slice(geometry, (r, c))
            if (r, c) in field:
                target = np.where(field[r, c], ink, background)
                _need(np.array_equal(out[sl], target), "echo tile or union is incorrect")
            else:
                _need(np.array_equal(out[sl], a[sl]), "tile outside all echoes changed")
    _need(np.array_equal(solve(a), out), "input-only solve disagrees")
    seeds = {e["position"] for e in echoes}
    _need(bool(set(field)-seeds), "there must be new whole tiles")
    # Ablation A: keep the recovered orientations but drop all crop addresses.
    # Compare unclipped coarse supports, even when the wrong origin goes out of bounds.
    wrong_supports = [_active(images[e["reg"][0]], e["position"]) for e in echoes]
    _need(any(wrong != e["active"] for wrong, e in zip(wrong_supports, echoes)),
          "crop-address ablation has no effect")
    # Every echo has a non-seed exclusive tile. Thus dropping any whole echo
    # changes the output, and merely restoring the chip cannot solve the pair.
    backgrounds_changed = any(np.any((info["ordinary"] == ordinary_ink) & ~mask)
                              for pos, mask in field.items() if pos not in seeds)
    _need(backgrounds_changed, "ink-only copying is not distinguished")
    info["field"], info["coverage"] = field, coverage
    return info


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, decoded = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            info = _pair_invariants(pair)
            key = _key(info["input"])
            _need(key not in seen, "duplicate input")
            seen.add(key)
            decoded[split].append(info)
    _need(len(decoded["train"][0]["echoes"]) == 1 and
          len(decoded["train"][1]["echoes"]) == 2, "missing demonstration curriculum")
    _need(all(len(f["echoes"]) <= 2 for f in decoded["train"]) and
          all(len(f["echoes"]) == 3 for f in decoded["test"]), "missing count generalization")


def _signature(puzzle):
    return tuple((s, _key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Stream 42 seeds, retaining only bounded sets of small feature signatures."""
    features = {k: set() for k in ("puzzles", "templates", "geometry", "patch_shapes", "offsets",
                                  "orientations", "origins", "palettes", "structures", "overlap_counts")}
    pair_count = 0
    import hashlib
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        # Hashes, rather than full puzzles, bound retained diagnostic memory.
        features["puzzles"].add(hashlib.sha256(repr(_signature(puzzle)).encode()).digest())
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                field, coverage = _field(info["images"], info["echoes"])
                pair_count += 1
                features["templates"].add(min(_key(s) for s in info["images"]))
                features["geometry"].add(info["geometry"])
                features["palettes"].add(info["roles"])
                features["structures"].add((info["geometry"], min(_key(s) for s in info["images"]),
                                            tuple(sorted((e["reg"], e["origin"]) for e in info["echoes"]))))
                features["overlap_counts"].add(sum(v > 1 for v in coverage.values()))
                for echo in info["echoes"]:
                    k, r, c, h, w = echo["reg"]
                    features["patch_shapes"].add((h, w))
                    features["offsets"].add((r, c))
                    features["orientations"].add(k)
                    features["origins"].add(echo["origin"])
        if seed == 0:
            _need(_signature(puzzle) == _signature(generate_puzzle(seed)), "nondeterministic generation")
            _need(_signature(puzzle) != _signature(generate_puzzle(seed+1)), "seed has no effect")
            bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in ("train", "test")}
            bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
            try:
                validate_generated_puzzle(bad)
            except ValueError:
                pass
            else:
                raise AssertionError("changed divider accepted")
    _need(len(features["puzzles"]) == 42, "puzzle diversity failed")
    for name, minimum in (("templates", 35), ("geometry", 15), ("patch_shapes", 6),
                          ("offsets", 6), ("origins", 12), ("palettes", 70), ("structures", 180)):
        _need(len(features[name]) >= minimum, "insufficient variation in " + name)
    _need(features["orientations"] == set(range(8)), "missing orientation")
    _need(len(features["overlap_counts"]) >= 6, "overlap topology is too narrow")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(123, nt, nq)
            _need(len(p["train"]) == nt and len(p["test"]) == nq, "wrong explicit counts")
            validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # The fallback's entire finite bank is decoded, not trusted as latent metadata.
    for count in (1, 2, 3):
        used = set()
        for _ in range(16):
            pair = _fallback(count, (0, 1, 2, 3, 4), used)
            _pair_invariants(pair)
            used.add(_key(pair["input"]))
    # Reject a genuinely ambiguous 2x2 crop rather than choosing the first match.
    pair = _fallback(1, (0, 1, 2, 3, 4), set())
    info = _decode(pair["input"])
    chip_slice = _tile_slice(info["geometry"], info["echoes"][0]["position"])
    ambiguous = pair["input"].copy()
    ambiguous[chip_slice] = np.array([[3, 1, 4], [1, 3, 4], [4, 4, 4]], dtype=np.int8)
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous crop accepted")
    # In the two-echo fixture the ordinary overlap tile (2,2) must merge both
    # orientations. Keeping only one is an observable overwrite error.
    pair = _fallback(2, (0, 1, 2, 3, 4), set())
    info = _decode(pair["input"])
    original = pair["input"].copy()
    solve(pair["input"])
    _need(np.array_equal(pair["input"], original), "solve mutated its input")
    wrong = {k: v.copy() for k, v in pair.items()}
    first_image = info["images"][info["echoes"][0]["reg"][0]]
    wrong["output"][_tile_slice(info["geometry"], (2, 2))] = np.where(first_image, 3, 1)
    try:
        _pair_invariants(wrong)
    except ValueError:
        pass
    else:
        raise AssertionError("overwriting an overlap accepted")
    # An ink-only copy incorrectly retains ordinary ink under the new background.
    wrong = {k: v.copy() for k, v in pair.items()}
    seeds = {e["position"] for e in info["echoes"]}
    field, _ = _field(info["images"], info["echoes"])
    for position, mask in field.items():
        differing = (info["ordinary"] == 2) & ~mask
        if position not in seeds and differing.any():
            sl = _tile_slice(info["geometry"], position)
            wrong["output"][sl][differing] = 2
            break
    try:
        _pair_invariants(wrong)
    except ValueError:
        pass
    else:
        raise AssertionError("ink-only copying accepted")
    return {"seeds": 42, "pairs": pair_count,
            "distinct": {k: len(v) for k, v in features.items()},
            "semantic_verification": "generator-owned invariants and input-only solver"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
