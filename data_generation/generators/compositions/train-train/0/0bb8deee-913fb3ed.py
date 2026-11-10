"""Stitching nurseries: crop ink-keyed tiles, then resolve simultaneous halos.

Only NumPy and the standard library are used.  The solver reads its entire
convention from one input; neither generation nor training examples are needed.
"""
from __future__ import annotations

import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grids must have two dimensions in 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must be in 0..9")
    return a.astype(np.int8, copy=False)


def _ranges(length, lines):
    bounds = [-1] + list(lines) + [length]
    spans = [(a + 1, b) for a, b in zip(bounds, bounds[1:])]
    _require(all(b - a >= 5 for a, b in spans), "empty or undersized panel")
    return spans


def _read_input(value):
    """Recover panels, exact crops, seed/ink associations and the compact map."""
    a = _grid(value)
    candidates = []
    for color in np.unique(a):
        rows = np.flatnonzero(np.all(a == color, axis=1)).tolist()
        cols = np.flatnonzero(np.all(a == color, axis=0)).tolist()
        if not rows and not cols:
            continue
        mask = np.zeros(a.shape, dtype=bool)
        mask[rows, :] = True
        mask[:, cols] = True
        if np.array_equal(a == color, mask):
            candidates.append((int(color), rows, cols))
    _require(len(candidates) == 1, "separator must be a unique union of full lines")
    divider, row_lines, col_lines = candidates[0]
    rr, cc = _ranges(a.shape[0], row_lines), _ranges(a.shape[1], col_lines)
    _require(len(rr) * len(cc) >= 2, "at least two panels are needed")
    # Background dominates the non-divider cells, including in tight layouts.
    colors, counts = np.unique(a[a != divider], return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, "ambiguous background")
    bg = int(colors[np.argmax(counts)])
    tiles, offsets, associations, seeds_by_tile = [], [], {}, []
    for r0, r1 in rr:
        for c0, c1 in cc:
            panel = a[r0:r1, c0:c1]
            foreground = np.argwhere(panel != bg)
            _require(len(foreground) >= 5, "each tile needs corners and seeds")
            lo, hi = foreground.min(axis=0), foreground.max(axis=0) + 1
            tile = panel[lo[0]:hi[0], lo[1]:hi[1]].copy()
            h, w = tile.shape
            _require(h >= 5 and w >= 5, "tile is too small")
            ink = int(tile[0, 0])
            corner_mask = np.zeros(tile.shape, dtype=bool)
            corner_mask[0, 0] = corner_mask[0, -1] = True
            corner_mask[-1, 0] = corner_mask[-1, -1] = True
            _require(ink not in (bg, divider), "invalid ink")
            _require(np.array_equal(tile == ink, corner_mask),
                     "exactly four matching ink corners must mark each crop")
            seed_mask = (tile != bg) & ~corner_mask
            seed_colors = np.unique(tile[seed_mask])
            _require(len(seed_colors) == 1, "one seed type per tile is required")
            seed = int(seed_colors[0])
            _require(seed not in (bg, divider, ink), "seed and structural roles overlap")
            locations = [tuple(map(int, p)) for p in np.argwhere(seed_mask)]
            _require(all(max(abs(r - u), abs(c - v)) >= 2
                         for i, (r, c) in enumerate(locations)
                         for u, v in locations[i + 1:]), "dots must be separated within a tile")
            if seed in associations:
                _require(associations[seed] == ink, "inconsistent seed-to-ink key")
            associations[seed] = ink
            seeds_by_tile.append(locations)
            tiles.append(tile)
            offsets.append((int(lo[0]), int(lo[1]), int(panel.shape[0] - hi[0]),
                            int(panel.shape[1] - hi[1])))
    _require(len({tile.shape for tile in tiles}) == 1, "tile sizes must agree within an input")
    _require(len(set(associations.values())) == len(associations), "ink key must be one-to-one")
    _require(set(associations).isdisjoint(associations.values()), "seed and ink palettes overlap")
    nr, nc = len(rr), len(cc)
    h, w = tiles[0].shape
    _require(nr * h <= 30 and nc * w <= 30, "assembled mosaic is too large")
    base = np.concatenate([np.concatenate(tiles[r * nc:(r + 1) * nc], axis=1)
                           for r in range(nr)], axis=0)
    seed_ink = np.full(base.shape, -1, dtype=np.int8)
    owner = np.empty(base.shape, dtype=np.int8)
    for i, positions in enumerate(seeds_by_tile):
        tr, tc = divmod(i, nc)
        owner[tr * h:(tr + 1) * h, tc * w:(tc + 1) * w] = i
        ink = int(tiles[i][0, 0])
        for r, c in positions:
            seed_ink[tr * h + r, tc * w + c] = ink
    return {"base": base, "background": bg, "divider": divider,
            "layout": (nr, nc), "tile_size": (h, w), "tiles": tiles,
            "seed_ink": seed_ink, "owner": owner, "mapping": associations,
            "offsets": offsets, "input_shape": a.shape}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """For each blank mosaic cell, inspect its neighboring original centers."""
    info = _read_input(input_grid)
    base, seed_ink, bg = info["base"], info["seed_ink"], info["background"]
    out = base.copy()
    h, w = base.shape
    for r, c in np.argwhere(base == bg):
        neighborhood = seed_ink[max(0, r - 1):min(h, r + 2),
                                max(0, c - 1):min(w, c + 2)]
        inks = np.unique(neighborhood[neighborhood >= 0])
        if len(inks) == 1:
            out[r, c] = inks[0]
    return out


def _claims(base, seed_ink, bg, owner=None, confine=False):
    """Forward construction and diagnostics, separate from solve's inverse scan."""
    h, w = base.shape
    bits = np.zeros(base.shape, dtype=np.int16)
    donors = np.zeros(base.shape, dtype=np.int16)
    per_ink = np.zeros((10, h, w), dtype=np.int8)
    cross = np.zeros(base.shape, dtype=bool)
    clipped = protected = 0
    for r, c in np.argwhere(seed_ink >= 0):
        ink = int(seed_ink[r, c])
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                u, v = int(r + dr), int(c + dc)
                if not (0 <= u < h and 0 <= v < w):
                    clipped += 1
                    continue
                if owner is not None and owner[u, v] != owner[r, c]:
                    if confine:
                        continue
                    if base[u, v] == bg:
                        cross[u, v] = True
                if base[u, v] != bg:
                    protected += 1
                    continue
                bits[u, v] |= 1 << ink
                donors[u, v] += 1
                per_ink[ink, u, v] += 1
    unanimous = (bits > 0) & ((bits & (bits - 1)) == 0)
    conflict = (bits > 0) & ~unanimous
    out = base.copy()
    for color in range(10):
        out[bits == (1 << color)] = color
    return out, {"conflicts": int(conflict.sum()),
                 "majority_conflicts": int((conflict & (2 * per_ink.max(axis=0) > donors)).sum()),
                 "agreements": int((unanimous & (donors > 1)).sum()),
                 "painted": int(unanimous.sum()),
                 "seam_painted": int((unanimous & cross).sum()),
                 "clipped_requests": clipped, "protected_requests": protected,
                 "bits": bits}


def _tile(h, w, role, seeds, inks, bg, positions):
    tile = np.full((h, w), bg, dtype=np.int8)
    tile[0, 0] = tile[0, -1] = tile[-1, 0] = tile[-1, -1] = inks[role]
    for r, c in positions:
        tile[r, c] = seeds[role]
    return tile


def _turn_tiles(tiles, nr, nc, turns):
    h, w = tiles[0].shape
    mosaic = np.concatenate([np.concatenate(tiles[r * nc:(r + 1) * nc], axis=1)
                             for r in range(nr)], axis=0)
    mosaic = np.rot90(mosaic, turns)
    if turns % 2:
        nr, nc, h, w = nc, nr, w, h
    return [mosaic[r * h:(r + 1) * h, c * w:(c + 1) * w].copy()
            for r in range(nr) for c in range(nc)], (nr, nc)


def _add_dots(rng, positions, h, w, target):
    if len(positions) >= target:
        return positions
    cells = [(r, c) for r in range(h) for c in range(w)
             if (r, c) not in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))]
    for i in rng.permutation(len(cells)):
        p = cells[int(i)]
        if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) >= 2 for q in positions):
            positions.append(p)
        if len(positions) >= target:
            break
    return positions


def _make_tiles(rng, kind, size, seeds, inks, bg):
    h, w = size
    mr, mc = h // 2, w // 2
    if kind == "clash":
        layout, roles = (1, 2), [0, 1]
        # The third left dot supplies a 2-to-1 conflict.  This makes
        # unanimity distinguishable from majority voting even with two demos.
        positions = [[(mr, 1), (mr, w - 1), (mr - 2, w - 2)],
                     [(mr, 0), (mr, w - 2), (0, mc)]]
    elif kind == "agreement":
        layout = (2, 2)
        roles = [0, 0, 1, 2 if len(seeds) == 3 else 1]
        positions = [[(mr, w - 1), (h - 1, 1)],
                     [(mr, 0), (h - 1, 2)],
                     [(0, 3), (mr, w - 1)],
                     [(0, 3), (mr, 0), (mr, w - 2)]]
    else:
        if kind == "test":
            layout = (2, 3)
            roles = [0, 0, 1, 1, 2 if len(seeds) == 3 else 1, 1]
            positions = [[] for _ in roles]
            # Every internal seam has centers on both sides.  The first row
            # contains a same-ink seam followed by a different-ink seam.
            for r in range(2):
                for c in range(2):
                    positions[r * 3 + c].append((mr, w - 1))
                    positions[r * 3 + c + 1].append((mr, 0))
            for c in range(3):
                positions[c].append((h - 1, mc))
                positions[3 + c].append((0, mc))
            for pts in positions:
                target = len(pts) + int(rng.integers(0, 3))
                _add_dots(rng, pts, h, w, target)
        else:
            choices = ((1, 2), (2, 1), (2, 2), (1, 3), (3, 1))
            layout = choices[int(rng.integers(len(choices)))]
            roles = [int(v) for v in rng.integers(0, len(seeds), size=layout[0] * layout[1])]
            positions = [_add_dots(rng, [], h, w, int(rng.integers(1, 6))) for _ in roles]
    tiles = [_tile(h, w, role, seeds, inks, bg, pts) for role, pts in zip(roles, positions)]
    return _turn_tiles(tiles, *layout, int(rng.integers(0, 4)))


def _spans(rng, count, extent):
    capacity = 30 - count * extent - (count - 1)
    if rng.random() < 0.15:
        extra = capacity
    else:
        extra = int(rng.integers(0, min(capacity, 4 * count) + 1))
    weights = rng.random(count) + 0.1
    allocation = rng.multinomial(extra, weights / weights.sum())
    return [extent + int(x) for x in allocation]


def _place_tiles(rng, tiles, layout, bg, divider):
    nr, nc = layout
    h, w = tiles[0].shape
    rs, cs = _spans(rng, nr, h), _spans(rng, nc, w)
    a = np.full((sum(rs) + nr - 1, sum(cs) + nc - 1), bg, dtype=np.int8)
    row_starts, col_starts = [0], [0]
    for span in rs[:-1]:
        line = row_starts[-1] + span
        a[line, :] = divider
        row_starts.append(line + 1)
    for span in cs[:-1]:
        line = col_starts[-1] + span
        a[:, line] = divider
        col_starts.append(line + 1)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, nc)
        ro = int(rng.integers(0, rs[r] - h + 1))
        co = int(rng.integers(0, cs[c] - w + 1))
        a[row_starts[r] + ro:row_starts[r] + ro + h,
          col_starts[c] + co:col_starts[c] + co + w] = tile
    return a


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(map(int, a.flat))


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    nt, nq = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    _require(seed >= 0, "seed must be nonnegative")
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal example counts")
    rng = np.random.default_rng(seed)
    n_roles = int(rng.integers(2, 4))
    palette = list(map(int, rng.permutation(10)))
    seeds, inks = palette[:n_roles], palette[n_roles:2 * n_roles]
    structural = palette[2 * n_roles:]
    kinds = ["clash", "agreement"] + ["free"] * (nt - 2)
    rng.shuffle(kinds)
    puzzle = {"train": [], "test": []}
    seen = set()
    # Different witness tile areas make variable cropping dimensions explicit.
    witness_sizes = {}
    witness_sizes["clash"] = tuple(map(int, rng.integers(5, 8, size=2)))
    choices = [(h, w) for h in range(5, 8) for w in range(5, 8)
               if h * w != np.prod(witness_sizes["clash"])]
    witness_sizes["agreement"] = choices[int(rng.integers(len(choices)))]
    for index, kind in enumerate(kinds + ["test"] * nq):
        for _ in range(32):
            bg, divider = map(int, rng.choice(structural, size=2, replace=False))
            size = witness_sizes[kind] if kind in witness_sizes else tuple(map(int, rng.integers(5, 8, size=2)))
            tiles, layout = _make_tiles(rng, kind, size, seeds, inks, bg)
            a = _place_tiles(rng, tiles, layout, bg, divider)
            if _key(a) in seen:
                continue
            # Labels start from the known exact tiles, not solve's parser.
            nr, nc = layout
            base = np.concatenate([np.concatenate(tiles[r * nc:(r + 1) * nc], axis=1)
                                   for r in range(nr)], axis=0)
            seed_ink = np.full(base.shape, -1, dtype=np.int8)
            for center, ink in zip(seeds, inks):
                seed_ink[base == center] = ink
            b, _ = _claims(base, seed_ink, bg)
            seen.add(_key(a))
            puzzle["train" if index < nt else "test"].append({"input": a, "output": b})
            break
        else:
            raise RuntimeError("could not construct a distinct example")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check visible semantics, teaching witnesses, and the structural shift."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[k], list) for k in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal counts")
    seen, infos, global_mapping, clean_roles = set(), {"train": [], "test": []}, {}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            info = _read_input(a)
            base, bg, si = info["base"], info["background"], info["seed_ink"]
            expected, stats = _claims(base, si, bg, info["owner"])
            _require(np.array_equal(b, expected) and np.array_equal(b, solve(a)), "incorrect halo consensus output")
            _require(np.array_equal(b[base != bg], base[base != bg]), "original corners or centers changed")
            _require(info["divider"] not in np.unique(b), "separator survived")
            _require(stats["painted"] > 0, "growth must change blank cells")
            _require(a.shape != b.shape, "margins/separators must be removed")
            for s, ink in info["mapping"].items():
                _require(s not in global_mapping or global_mapping[s] == ink, "puzzle key changed")
                global_mapping[s] = ink
            if split == "train":
                h, w = base.shape
                for r, c in np.argwhere(si >= 0):
                    if 0 < r < h - 1 and 0 < c < w - 1:
                        patch = b[r - 1:r + 2, c - 1:c + 2].copy()
                        seed = int(base[r, c])
                        patch[1, 1] = si[r, c]
                        if np.all(patch == si[r, c]):
                            clean_roles.add(seed)
            info["stats"] = stats
            infos[split].append(info)
    _require(2 <= len(global_mapping) <= 3, "two or three seed associations are required")
    _require(len(set(global_mapping.values())) == len(global_mapping), "puzzle ink key is not bijective")
    _require(set(global_mapping).isdisjoint(global_mapping.values()), "puzzle color roles overlap")
    role_palette = set(global_mapping) | set(global_mapping.values())
    _require(all(x["background"] not in role_palette and x["divider"] not in role_palette
                 for split in infos.values() for x in split), "structural and puzzle role palettes overlap")
    _require(clean_roles == set(global_mapping), "every role needs an intact training halo")
    training = infos["train"]
    for feature in ("conflicts", "majority_conflicts", "agreements", "seam_painted", "clipped_requests", "protected_requests"):
        _require(any(item["stats"][feature] > 0 for item in training), f"missing training witness: {feature}")
    _require(len({x["tile_size"] for x in training}) >= 2, "training must show variable crop sizes")
    _require(any(x["layout"] == (2, 2) for x in training), "training needs both seam directions")
    for info in infos["test"]:
        _require(info["layout"] in ((2, 3), (3, 2)), "test should extend the seam network")
        _require(all(s in clean_roles for s in info["mapping"]), "untaught test role")
        _require(info["stats"]["conflicts"] > 0 and info["stats"]["agreements"] > 0,
                 "test must combine agreement and conflict")
    # Ablate A's new adjacency while retaining identical crops and halo sizes.
    _require(any(not np.array_equal(
        _claims(x["base"], x["seed_ink"], x["background"], x["owner"], confine=True)[0],
        _claims(x["base"], x["seed_ink"], x["background"])[0]) for x in training),
        "tile-local growth shortcut is not distinguished")


def _fingerprint(puzzle):
    return tuple((split, tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[split]))
                 for split in ("train", "test"))


def _normalized(a):
    labels = {}
    return tuple(labels.setdefault(int(c), len(labels)) for c in np.asarray(a).flat)


def self_test():
    """Bounded 42-seed audit, all count combinations, and semantic probes."""
    shapes, structures, sizes, layouts, palettes, counts = set(), set(), set(), set(), set(), set()
    observations, fingerprints = set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        fingerprints.add(_fingerprint(p))
        for split in ("train", "test"):
            for pair in p[split]:
                a, b = pair["input"], pair["output"]
                info = _read_input(a)
                shapes.add(a.shape)
                structures.add((a.shape, _normalized(a)))
                sizes.add(info["tile_size"])
                layouts.add(info["layout"])
                palettes.add((info["background"], info["divider"], tuple(sorted(info["mapping"].items()))))
                counts.add(int((info["seed_ink"] >= 0).sum()))
                _, stats = _claims(info["base"], info["seed_ink"], info["background"], info["owner"])
                observations.update(k for k in ("conflicts", "majority_conflicts", "agreements", "seam_painted", "clipped_requests", "protected_requests") if stats[k])
                if all(min(offset) > 0 for offset in info["offsets"]):
                    observations.add("interior_tiles")
                if any(min(offset) == 0 for offset in info["offsets"]):
                    observations.add("edge_tiles")
                if 30 in a.shape:
                    observations.add("extent_30")
                # Equivariance under an arbitrary bijective recoloring.
                permutation = np.roll(np.arange(10), 3)
                _require(np.array_equal(solve(permutation[a]), permutation[b]), "recoloring changed semantics")
        if seed == 0:
            _require(_fingerprint(p) == _fingerprint(generate_puzzle(seed)), "nondeterministic generation")
        # Relocating the same tiles in independently resized panels cannot
        # change any compact-space halo interaction.
        first = p["train"][0]
        info = _read_input(first["input"])
        moved = _place_tiles(np.random.default_rng(9000 + seed), info["tiles"],
                             info["layout"], info["background"], info["divider"])
        _require(np.array_equal(solve(moved), first["output"]), "panel margins affected the answer")
        _require(np.array_equal(solve(np.rot90(moved)), np.rot90(first["output"])),
                 "spatial rotation changed the rule")
    _require(len(fingerprints) == 42, "seeds do not vary the puzzle")
    _require(len(shapes) >= 50 and len(structures) >= 180, "insufficient structural variation")
    _require(len(sizes) == 9 and len(layouts) >= 7 and len(palettes) >= 150,
             "insufficient geometry or palette variation")
    _require(max(counts) >= 20 and min(counts) <= 3, "seed counts do not vary enough")
    _require(observations == {"conflicts", "majority_conflicts", "agreements", "seam_painted", "clipped_requests",
                              "protected_requests", "interior_tiles", "edge_tiles", "extent_30"},
             "a structural case was not exercised")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(4200 + 10 * nt + nq, nt, nq)
            _require((len(p["train"]), len(p["test"])) == (nt, nq), "count not respected")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(81, 2, 1)
    p["test"][0]["output"] = p["test"][0]["output"].copy()
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    # A hand-specified local probe separates unanimous multiplicity from conflict.
    base = np.zeros((5, 7), dtype=np.int8)
    si = np.full(base.shape, -1, dtype=np.int8)
    base[2, 2], base[2, 4] = 1, 2
    si[2, 2], si[2, 4] = 6, 7
    out, _ = _claims(base, si, 0)
    _require(out[2, 3] == 0 and out[1, 1] == 6 and out[1, 5] == 7,
             "different inks did not cancel")
    si[2, 4] = 6
    out, _ = _claims(base, si, 0)
    _require(out[2, 3] == 6 and out[2, 2] == 1 and out[2, 4] == 2,
             "same inks did not merge or centers changed")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "input_shapes": len(shapes),
            "normalized_input_structures": len(structures), "tile_sizes": len(sizes),
            "layouts": sorted(layouts), "palettes": len(palettes), "seed_counts": sorted(counts),
            "observed": sorted(observations)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
