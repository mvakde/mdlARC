"""Beacon march: creatures turn to face a beacon and take one step toward it.

Composition of two parent families.

* Parent A (3e980e27): a complete two-colour prototype (one key cell plus a
  monochrome body) is an anchored stencil.  Every isolated cell of the key colour
  is an incomplete copy that receives the body at the same key-relative offsets,
  either unchanged or mirrored left-right about the key column.
* Parent B (dc433765): a mover aligned with a stationary target on a row or a
  45-degree diagonal takes exactly one king step toward it; its old cell becomes
  background and the target stays put.

Connecting concept (new): *facing*.  Every creature in the scene -- the complete
prototype and each lone key cell -- is aligned with the single beacon cell and
takes one king step toward it.  A creature faces the way it walks.  The
prototype shows what a creature looks like when it approaches the beacon from
its side, so a lone key cell approaching from the same horizontal side is
completed with the body unchanged, and one approaching from the opposite side
is completed with the body mirrored about its key column.  The direction that
parent B computes therefore decides parent A's copy/mirror choice, and parent
A's stencil decides what parent B moves.

Only the standard library and NumPy are used.
"""

from __future__ import annotations

import numpy as np


_N8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
_MOVES = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 1))

# Alternative explanations the training pairs must refute.  Each entry is
# (facing policy, motion policy); ("beacon", "step") is the real rule.
FACING_ALTERNATIVES = (
    "never",            # ignore parent A's chirality: always copy unchanged
    "always",           # always mirror copies
    "direction",        # mirror whenever the full step direction differs
    "half_turn",        # turn around by a 180-degree rotation instead of a mirror
    "per_axis",         # mirror per axis: left-right and up-down sign changes
    "west",             # mirror copies that walk west (absolute convention)
    "east",             # mirror copies that walk east (absolute convention)
    "prototype_ahead",  # mirror copies lying beaconward of the prototype column
    "prototype_behind", # mirror copies lying behind the prototype column
    "grid_center",      # mirror copies across the grid's middle column
)
MOTION_ALTERNATIVES = (
    "none",           # ignore parent B: complete in place, nothing moves
    "proto_static",   # only lone key cells move
    "copies_static",  # only the prototype moves
    "to_target",      # walk all the way up to the beacon
)


def _sign(value):
    return (value > 0) - (value < 0)


def _mirror(offsets):
    return tuple(sorted((r, -c) for r, c in offsets))


def _vflip(offsets):
    return tuple(sorted((-r, c) for r, c in offsets))


def _half_turn(offsets):
    return tuple(sorted((-r, -c) for r, c in offsets))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


# ---------------------------------------------------------------------------
# Input parsing and the rule itself (input only)
# ---------------------------------------------------------------------------

def _components(grid, background):
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    components = []
    for row in range(height):
        for col in range(width):
            if seen[row, col] or grid[row, col] == background:
                continue
            seen[row, col] = True
            stack, cells = [(row, col)], []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in _N8:
                    y, x = r + dr, c + dc
                    if (0 <= y < height and 0 <= x < width and not seen[y, x]
                            and grid[y, x] != background):
                        seen[y, x] = True
                        stack.append((y, x))
            components.append(sorted(cells))
    return components


def _parse(grid):
    """Recover background, beacon, prototypes and lone key cells from an input."""
    g = np.asarray(grid)
    if g.ndim != 2 or g.size == 0:
        raise ValueError("input must be a non-empty 2D grid")
    g = g.astype(int)
    counts = np.bincount(g.ravel(), minlength=10)
    background = int(np.argmax(counts))
    if counts[background] * 2 <= g.size:
        raise ValueError("background is not a strict majority")
    prototypes, singles = {}, []
    for cells in _components(g, background):
        if len(cells) == 1:
            singles.append(cells[0])
            continue
        by_color = {}
        for cell in cells:
            by_color.setdefault(int(g[cell]), []).append(cell)
        if len(by_color) != 2:
            raise ValueError("a creature must use exactly two colours")
        keys = [color for color, members in by_color.items() if len(members) == 1]
        if len(keys) != 1:
            raise ValueError("a creature needs exactly one key cell")
        key_color = keys[0]
        body_color = next(color for color in by_color if color != key_color)
        if key_color in prototypes:
            raise ValueError("two prototypes share a key colour")
        key = by_color[key_color][0]
        offsets = tuple(sorted((r - key[0], c - key[1]) for r, c in by_color[body_color]))
        prototypes[key_color] = {"key": key, "body": body_color, "offsets": offsets}
    beacons = [cell for cell in singles if int(g[cell]) not in prototypes]
    if len(beacons) != 1:
        raise ValueError("expected exactly one beacon cell")
    beacon = beacons[0]
    beacon_color = int(g[beacon])
    bodies = {p["body"] for p in prototypes.values()}
    if beacon_color in bodies or bodies & set(prototypes):
        raise ValueError("beacon, key and body colours must be distinct roles")
    copies = sorted(cell for cell in singles if int(g[cell]) in prototypes)
    return {"grid": g, "shape": g.shape, "background": background,
            "beacon": beacon, "beacon_color": beacon_color,
            "prototypes": prototypes, "copies": copies}


def _motion(key, beacon):
    """King-step direction toward the beacon and the separation along the ray."""
    dr, dc = beacon[0] - key[0], beacon[1] - key[1]
    if dc == 0:
        raise ValueError("a creature directly above/below the beacon has no facing")
    if dr != 0 and abs(dr) != abs(dc):
        raise ValueError("creature key is not on a row or diagonal through the beacon")
    distance = abs(dc)
    if distance < 2:
        raise ValueError("creature key must be at least two cells from the beacon")
    return (_sign(dr), _sign(dc)), distance


def _orient(offsets, step, proto_step, cell, proto_key, width, facing):
    turned = step[1] != proto_step[1]
    if facing == "beacon":
        return _mirror(offsets) if turned else offsets
    if facing == "never":
        return offsets
    if facing == "always":
        return _mirror(offsets)
    if facing == "direction":
        return _mirror(offsets) if step != proto_step else offsets
    if facing == "half_turn":
        return _half_turn(offsets) if turned else offsets
    if facing == "per_axis":
        result = _mirror(offsets) if turned else offsets
        return _vflip(result) if step[0] != proto_step[0] else result
    if facing == "west":
        return _mirror(offsets) if step[1] < 0 else offsets
    if facing == "east":
        return _mirror(offsets) if step[1] > 0 else offsets
    side = _sign(cell[1] - proto_key[1])
    if facing == "prototype_ahead":
        return _mirror(offsets) if side == proto_step[1] else offsets
    if facing == "prototype_behind":
        return _mirror(offsets) if side == -proto_step[1] else offsets
    if facing == "grid_center":
        center = (width - 1) / 2.0
        return (_mirror(offsets) if _sign(cell[1] - center) != _sign(proto_key[1] - center)
                else offsets)
    raise ValueError(f"unknown facing policy {facing}")


def _creatures(info, facing="beacon", motion="step"):
    """List of (key colour, body colour, new key, oriented offsets, role, step, distance)."""
    beacon = info["beacon"]
    width = info["shape"][1]
    result = []

    def shift(step, distance, role):
        if motion == "none":
            return (0, 0)
        if motion == "proto_static" and role == "prototype":
            return (0, 0)
        if motion == "copies_static" and role == "copy":
            return (0, 0)
        if motion == "to_target":
            return (step[0] * (distance - 1), step[1] * (distance - 1))
        if motion != "step" and motion not in MOTION_ALTERNATIVES:
            raise ValueError(f"unknown motion policy {motion}")
        return step

    for key_color in sorted(info["prototypes"]):
        proto = info["prototypes"][key_color]
        step, distance = _motion(proto["key"], beacon)
        move = shift(step, distance, "prototype")
        key = (proto["key"][0] + move[0], proto["key"][1] + move[1])
        result.append((key_color, proto["body"], key, proto["offsets"], "prototype",
                       step, distance, proto["key"]))
    for cell in info["copies"]:
        key_color = int(info["grid"][cell])
        proto = info["prototypes"][key_color]
        step, distance = _motion(cell, beacon)
        proto_step, _ = _motion(proto["key"], beacon)
        offsets = _orient(proto["offsets"], step, proto_step, cell, proto["key"],
                          width, facing)
        move = shift(step, distance, "copy")
        key = (cell[0] + move[0], cell[1] + move[1])
        result.append((key_color, proto["body"], key, offsets, "copy", step, distance, cell))
    return result


def _render(info, facing="beacon", motion="step", strict=False):
    height, width = info["shape"]
    out = np.full(info["shape"], info["background"], dtype=int)
    owner = {}
    beacon = info["beacon"]
    for index, (key_color, body, key, offsets, _, _, _, _) in enumerate(
            _creatures(info, facing, motion)):
        cells = [(key, key_color)] + [((key[0] + r, key[1] + c), body) for r, c in offsets]
        for (r, c), color in cells:
            inside = 0 <= r < height and 0 <= c < width
            if strict:
                if not inside:
                    raise ValueError("a creature would leave the grid")
                if (r, c) == beacon or (r, c) in owner:
                    raise ValueError("creatures would overlap")
            if inside:
                owner[(r, c)] = index
                out[r, c] = color
    out[beacon] = info["beacon_color"]
    if strict:
        for (r, c), index in owner.items():
            for dr, dc in _N8:
                other = owner.get((r + dr, c + dc))
                if other is not None and other != index:
                    raise ValueError("creatures would touch after moving")
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Every creature takes one king step toward the beacon, facing its walk."""
    return _render(_parse(input_grid), strict=True)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= int(value) <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _sample_stencil(rng, body_cells, span, need):
    """Grow an 8-connected body around the key at (0, 0).

    need: "both" (left-right and up-down asymmetric), "h" (left-right
    asymmetric), "sym" (left-right symmetric about the key column) or "any".
    """
    if need == "sym":
        for _ in range(300):
            half = {(0, 0)}
            goal = int(rng.integers(2, max(2, body_cells) + 1))
            while len(half) < goal:
                frontier = sorted({(r + dr, c + dc) for r, c in half for dr, dc in _N8
                                   if c + dc >= 0 and (r + dr, c + dc) not in half
                                   and c + dc < span
                                   and max([p[0] for p in half] + [r + dr])
                                   - min([p[0] for p in half] + [r + dr]) < span})
                if not frontier:
                    break
                half.add(frontier[int(rng.integers(len(frontier)))])
            full = half | {(r, -c) for r, c in half}
            if len(full) >= 3:
                return tuple(sorted(full - {(0, 0)}))
        raise RuntimeError("could not sample a symmetric stencil")
    for _ in range(300):
        cells = {(0, 0)}
        while len(cells) < body_cells + 1:
            frontier = set()
            for r, c in cells:
                for dr, dc in _N8:
                    candidate = (r + dr, c + dc)
                    if candidate in cells:
                        continue
                    rows = [p[0] for p in cells] + [candidate[0]]
                    cols = [p[1] for p in cells] + [candidate[1]]
                    if max(rows) - min(rows) < span and max(cols) - min(cols) < span:
                        frontier.add(candidate)
            if not frontier:
                break
            ordered = sorted(frontier)
            # Prefer orthogonal attachments a little so bodies look solid.
            weights = np.array([1.0 + sum((r + dr, c + dc) in cells
                                          for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)))
                                for r, c in ordered])
            cells.add(ordered[int(rng.choice(len(ordered), p=weights / weights.sum()))])
        if len(cells) < body_cells + 1:
            continue
        offsets = tuple(sorted(cells - {(0, 0)}))
        h_asym = _mirror(offsets) != offsets
        v_asym = _vflip(offsets) != offsets
        if need == "both" and not (h_asym and v_asym):
            continue
        if need == "h" and not h_asym:
            continue
        return offsets
    raise RuntimeError("could not sample a stencil")


def _dims(rng, size_class):
    if size_class == "compact":
        low, high = 7, 12
    elif size_class == "large":
        low, high = 18, 30
    else:
        low, high = 10, 21
    return int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1))


def _place(rng, height, width, species, beacon=None):
    """Place every requested creature; return (beacon, placements) or None.

    species: list of dicts with keys canon (east-facing offsets), proto_h,
    proto_kmin, copies (list of dicts h, between, dr_differs, kmin).
    """
    if beacon is None:
        beacon = (int(rng.integers(height)), int(rng.integers(width)))
    br, bc = beacon
    blocked_in = {(br + dr, bc + dc) for dr, dc in _N8 + ((0, 0),)}
    blocked_out = {beacon}
    placements = []

    def candidates(canon, h, kmin, kmax, rows, is_copy):
        offsets = canon if h == 1 else _mirror(canon)
        found = []
        for dr in rows:
            for k in range(kmin, kmax + 1):
                key = (br - k * dr, bc - k * h)
                cells_in = [key] if is_copy else [key] + [(key[0] + r, key[1] + c)
                                                          for r, c in offsets]
                new_key = (key[0] + dr, key[1] + h)
                cells_out = [new_key] + [(new_key[0] + r, new_key[1] + c) for r, c in offsets]
                if any(not (0 <= r < height and 0 <= c < width)
                       for r, c in cells_in + cells_out):
                    continue
                if any(cell in blocked_in for cell in cells_in):
                    continue
                if any(cell in blocked_out for cell in cells_out):
                    continue
                found.append(((dr, h), k, key, cells_in, cells_out))
        return found

    def commit(choice):
        _, _, _, cells_in, cells_out = choice
        for r, c in cells_in:
            for dr, dc in _N8 + ((0, 0),):
                blocked_in.add((r + dr, c + dc))
        for r, c in cells_out:
            for dr, dc in _N8 + ((0, 0),):
                blocked_out.add((r + dr, c + dc))

    for index, spec in enumerate(species):
        rows = list(rng.permutation(3) - 1)
        found = candidates(spec["canon"], spec["proto_h"], spec.get("proto_kmin", 2), 12,
                           rows, False)
        if not found:
            return None
        proto = found[int(rng.integers(len(found)))]
        commit(proto)
        placements.append({"species": index, "role": "prototype", "step": proto[0],
                           "distance": proto[1], "key": proto[2]})
        for request in spec["copies"]:
            kmax = 12
            if request.get("between"):
                kmax = proto[1] - 1
            rows = [dr for dr in (-1, 0, 1)
                    if not request.get("dr_differs") or dr != proto[0][0]]
            found = candidates(spec["canon"], request["h"], request.get("kmin", 2), kmax,
                               rows, True)
            if not found:
                return None
            choice = found[int(rng.integers(len(found)))]
            commit(choice)
            placements.append({"species": index, "role": "copy", "step": choice[0],
                               "distance": choice[1], "key": choice[2]})
    return beacon, placements


def _build_pair(rng, plan, colors):
    """Realise a pair plan; returns {'input','output'} or None."""
    for _ in range(40):
        height, width = _dims(rng, plan["size"])
        species = []
        key_colors = [int(c) for c in rng.choice(colors["keys"], len(plan["species"]),
                                                 replace=False)]
        for spec, key_color in zip(plan["species"], key_colors):
            if plan["size"] == "compact":
                body_cells, span = int(rng.integers(2, 5)), int(rng.integers(2, 4))
            else:
                body_cells, span = int(rng.integers(2, 7)), int(rng.integers(2, 5))
            body_cells = min(body_cells, span * span - 1)
            canon = _sample_stencil(rng, body_cells, span, spec["need"])
            body = int(colors["bodies"][int(rng.integers(len(colors["bodies"])))])
            species.append(dict(spec, canon=canon, key=key_color, body=body))
        placed = _place(rng, height, width, species)
        if placed is None:
            continue
        beacon, placements = placed
        grid = np.full((height, width), plan["background"], dtype=int)
        grid[beacon] = colors["beacon"]
        for item in placements:
            spec = species[item["species"]]
            key = item["key"]
            grid[key] = spec["key"]
            if item["role"] == "prototype":
                offsets = spec["canon"] if item["step"][1] == 1 else _mirror(spec["canon"])
                for r, c in offsets:
                    grid[key[0] + r, key[1] + c] = spec["body"]
        try:
            output = solve(grid)
        except ValueError:
            continue
        return {"input": grid.astype(np.int64), "output": output.astype(np.int64)}
    return None


def _random_species(rng, max_copies=3):
    copies = [{"h": int(rng.choice((-1, 1)))} for _ in range(int(rng.integers(0, max_copies + 1)))]
    roll = rng.random()
    need = "sym" if roll < 0.1 else ("any" if roll < 0.25 else "h")
    return {"proto_h": int(rng.choice((-1, 1))), "copies": copies, "need": need}


def _training_plans(rng, num_train, colors):
    side = int(rng.choice((-1, 1)))
    teach = {"proto_h": side, "proto_kmin": 3, "need": "both",
             "copies": [{"h": side, "between": True, "dr_differs": True},
                        {"h": -side, "dr_differs": bool(rng.random() < 0.6)}]}
    if rng.random() < 0.4:
        teach["copies"].append({"h": int(rng.choice((-1, 1)))})
    counter = {"proto_h": -side, "need": "h",
               "copies": [{"h": int(rng.choice((-1, 1)))}
                          for _ in range(int(rng.integers(1, 3)))]}
    plans = []
    merged = rng.random() < 0.3
    first = [teach] + ([counter] if merged else [])
    plans.append({"species": first,
                  "size": "large" if merged else ("normal" if rng.random() < 0.8 else "large")})
    if not merged:
        second = [counter]
        if rng.random() < 0.3:
            second.append(_random_species(rng, 1))
        plans.append({"species": second, "size": "normal" if rng.random() < 0.7 else "compact"})
    while len(plans) < num_train:
        roll = rng.random()
        n_species = 1 if roll < 0.55 else (2 if roll < 0.9 else 3)
        species = [_random_species(rng, 3 if n_species == 1 else 2) for _ in range(n_species)]
        size_roll = rng.random()
        size = ("compact" if size_roll < 0.2 and n_species == 1 else
                "large" if size_roll > 0.75 or n_species == 3 else "normal")
        plans.append({"species": species, "size": size})
    order = rng.permutation(len(plans))
    plans = [plans[int(i)] for i in order]
    for index, plan in enumerate(plans):
        plan["background"] = colors["backgrounds"][index % len(colors["backgrounds"])]
    return plans


def _pair_stats(pair):
    info = _parse(pair["input"])
    creatures = _creatures(info)
    return {"shape": info["shape"], "area": info["shape"][0] * info["shape"][1],
            "creatures": len(creatures), "species": len(info["prototypes"]),
            "copies": len(info["copies"]),
            "max_distance": max(item[6] for item in creatures)}


def _test_plan(rng, mode, train_stats, colors, index):
    max_creatures = max(s["creatures"] for s in train_stats)
    background = colors["backgrounds"][int(rng.integers(len(colors["backgrounds"])))]
    if mode == "crowd":
        n_species = int(rng.integers(2, 4))
        target = min(max_creatures + int(rng.integers(1, 3)), 9)
        species = [{"proto_h": int(rng.choice((-1, 1))), "copies": [], "need": "h"}
                   for _ in range(n_species)]
        total = n_species
        while total < max(target, n_species + 1):
            species[int(rng.integers(n_species))]["copies"].append(
                {"h": int(rng.choice((-1, 1)))})
            total += 1
        return {"species": species, "size": "large", "background": background}
    if mode == "compact":
        side = int(rng.choice((-1, 1)))
        copies = [{"h": -side}] + ([{"h": side}] if rng.random() < 0.4 else [])
        return {"species": [{"proto_h": side, "copies": copies, "need": "h"}],
                "size": "compact", "background": background}
    if mode == "far":
        side = int(rng.choice((-1, 1)))
        far_k = max(s["max_distance"] for s in train_stats) + 1
        copies = [{"h": -side, "kmin": min(far_k, 12)}]
        copies += [{"h": int(rng.choice((-1, 1)))} for _ in range(int(rng.integers(0, 3)))]
        species = [{"proto_h": side, "copies": copies, "need": "h"}]
        if rng.random() < 0.4:
            species.append(_random_species(rng, 1))
        return {"species": species, "size": "large", "background": background}
    species = [_random_species(rng, 2) for _ in range(int(rng.integers(1, 3)))]
    species[0]["copies"].append({"h": -species[0]["proto_h"]})
    return {"species": species, "size": "normal", "background": background}


def _test_ok(train_stats, pair, mode):
    stats = _pair_stats(pair)
    if mode == "crowd":
        return stats["creatures"] > max(s["creatures"] for s in train_stats)
    if mode == "compact":
        return stats["area"] < min(s["area"] for s in train_stats)
    if mode == "far":
        return stats["max_distance"] > max(s["max_distance"] for s in train_stats)
    return True


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    palette = [int(c) for c in rng.permutation(10)]
    n_backgrounds = 1 if rng.random() < 0.45 else 2
    colors = {"beacon": palette[0],
              "backgrounds": palette[1:1 + n_backgrounds],
              "keys": palette[1 + n_backgrounds:4 + n_backgrounds],
              "bodies": palette[4 + n_backgrounds:]}

    train = None
    for _ in range(60):
        plans = _training_plans(rng, num_train, colors)
        pairs = [_build_pair(rng, plan, colors) for plan in plans]
        if any(pair is None for pair in pairs):
            continue
        if len({pair["input"].tobytes() + bytes(pair["input"].shape) for pair in pairs}) != len(pairs):
            continue
        if _unrefuted(pairs):
            continue
        train = pairs
        break
    if train is None:
        raise RuntimeError("could not build informative training pairs")

    train_stats = [_pair_stats(pair) for pair in train]
    test = None
    for _ in range(60):
        pairs = []
        for index in range(num_test):
            roll = rng.random()
            mode = ("crowd" if roll < 0.3 else "far" if roll < 0.55 else
                    "compact" if roll < 0.75 else "familiar")
            pair = None
            for _ in range(8):
                candidate = _build_pair(rng, _test_plan(rng, mode, train_stats, colors, index),
                                        colors)
                if candidate is not None and _test_ok(train_stats, candidate, mode):
                    pair = candidate
                    break
            if pair is None:
                candidate = _build_pair(rng, _test_plan(rng, "familiar", train_stats, colors,
                                                        index), colors)
                pair = candidate
            if pair is None:
                break
            pairs.append(pair)
        if len(pairs) != num_test:
            continue
        keys = {pair["input"].tobytes() + bytes(pair["input"].shape) for pair in train + pairs}
        if len(keys) != num_train + num_test:
            continue
        if not _test_informative(pairs):
            continue
        test = pairs
        break
    if test is None:
        raise RuntimeError("could not build test pairs")
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _as_array(grid):
    return np.asarray(grid).astype(int)


def _unrefuted(train_pairs):
    """Alternative rules that reproduce every training output."""
    infos = [(_parse(_as_array(p["input"])), _as_array(p["output"])) for p in train_pairs]
    survivors = []
    for facing in FACING_ALTERNATIVES:
        if all(np.array_equal(_render(info, facing=facing), out) for info, out in infos):
            survivors.append(("facing", facing))
    for motion in MOTION_ALTERNATIVES:
        if all(np.array_equal(_render(info, motion=motion), out) for info, out in infos):
            survivors.append(("motion", motion))
    return survivors


def _test_informative(test_pairs):
    """Ignoring either parent must change at least one test answer."""
    infos = [(_parse(_as_array(p["input"])), _as_array(p["output"])) for p in test_pairs]
    if any(not info["copies"] for info, _ in infos):
        return False
    for facing, motion in (("never", "step"), ("beacon", "none"),
                           ("beacon", "copies_static"), ("always", "step")):
        if all(np.array_equal(_render(info, facing=facing, motion=motion), out)
               for info, out in infos):
            return False
    return True


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    num_train, num_test = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test,
             "illegal pair counts")
    seen = set()
    beacon_colors = set()
    sides = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs exactly input and output")
            for grid in pair.values():
                array = np.asarray(grid)
                _require(array.ndim == 2 and all(1 <= n <= 30 for n in array.shape),
                         "grid dimensions out of range")
                _require(np.issubdtype(array.dtype, np.integer), "grid must be integer")
                _require(bool(np.all((array >= 0) & (array <= 9))), "colour out of range")
            source, target = _as_array(pair["input"]), _as_array(pair["output"])
            _require(source.shape == target.shape, "shape changed")
            key = (source.shape, source.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _parse(source)
            _require(bool(info["prototypes"]), "pair has no prototype")
            _require(np.array_equal(_render(info, strict=True), target),
                     "output does not follow the beacon-march rule")
            _require(int(np.argmax(np.bincount(target.ravel(), minlength=10)))
                     == info["background"], "background is not the output majority")
            beacon_colors.add(info["beacon_color"])
            keys = set(info["prototypes"])
            bodies = {p["body"] for p in info["prototypes"].values()}
            _require(not keys & bodies and info["beacon_color"] not in keys | bodies
                     and info["background"] not in keys | bodies | {info["beacon_color"]},
                     "colour roles overlap")
            for creature in _creatures(info):
                if split == "train" and creature[4] == "prototype":
                    sides.add(creature[5][1])
            if split == "test":
                _require(bool(info["copies"]), "test pair has no lone key cell")
    _require(len(beacon_colors) == 1, "beacon colour must be fixed within a puzzle")
    _require(sides == {-1, 1}, "training must show prototypes approaching from both sides")
    survivors = _unrefuted(puzzle["train"])
    _require(not survivors, f"training does not refute {survivors}")
    _require(_test_informative(puzzle["test"]),
             "test answers do not depend on both parent concepts")


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------

def _signature(puzzle):
    return tuple((split, tuple((_as_array(p["input"]).tobytes(), _as_array(p["input"]).shape,
                                _as_array(p["output"]).tobytes())
                               for p in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_signature(p) for p in puzzles}) == 42, "seeds collided")
    shapes, backgrounds, beacon_colors = set(), set(), set()
    steps = {"train": set(), "test": set()}
    distances = set()
    species_counts, copy_counts, creature_counts = set(), set(), set()
    mirrored = direct = symmetric = shared_body = 0
    test_modes = {"crowd": 0, "far": 0, "compact": 0}
    varied_background = 0
    stencils = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        stats = [_pair_stats(p) for p in puzzle["train"]]
        pair_backgrounds = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(_as_array(pair["input"]))
                shapes.add(info["shape"])
                backgrounds.add(info["background"])
                pair_backgrounds.add(info["background"])
                beacon_colors.add(info["beacon_color"])
                species_counts.add(len(info["prototypes"]))
                copy_counts.add(len(info["copies"]))
                creatures = _creatures(info)
                creature_counts.add(len(creatures))
                bodies = [p["body"] for p in info["prototypes"].values()]
                shared_body += len(bodies) != len(set(bodies))
                for proto in info["prototypes"].values():
                    stencils.add(proto["offsets"])
                    symmetric += _mirror(proto["offsets"]) == proto["offsets"]
                for item in creatures:
                    steps[split].add(item[5])
                    distances.add(item[6])
                    if item[4] == "copy":
                        proto = info["prototypes"][item[0]]
                        if item[3] != proto["offsets"]:
                            mirrored += 1
                        else:
                            direct += 1
                if split == "test":
                    current = _pair_stats(pair)
                    test_modes["crowd"] += current["creatures"] > max(s["creatures"] for s in stats)
                    test_modes["far"] += current["max_distance"] > max(s["max_distance"] for s in stats)
                    test_modes["compact"] += current["area"] < min(s["area"] for s in stats)
        varied_background += len(pair_backgrounds) > 1
    _require(len(shapes) >= 60, "grid shapes lack variety")
    _require(len(backgrounds) >= 8 and len(beacon_colors) >= 8, "palette roles lack variety")
    _require(steps["train"] == set(_MOVES) and steps["test"] == set(_MOVES),
             "not every non-vertical king step occurs in both splits")
    _require(min(distances) == 2 and max(distances) >= 11, "separations lack range")
    _require({1, 2, 3} <= species_counts and 0 in copy_counts and max(copy_counts) >= 4,
             "species or copy counts lack variety")
    _require(mirrored >= 60 and direct >= 60, "both facings must be common")
    _require(symmetric >= 3 and shared_body >= 3, "symmetric or shared-body cases missing")
    _require(all(count >= 5 for count in test_modes.values()), "test shifts lack variety")
    _require(varied_background >= 10, "backgrounds never vary within a puzzle")
    _require(len(stencils) >= 150, "stencil geometry lacks variety")
    _require(_signature(generate_puzzle(4242, 6, 3)) == _signature(generate_puzzle(4242, 6, 3)),
             "generation is not deterministic")
    for num_train, num_test in ((2, 1), (10, 4), (5, 4), (3, 2)):
        puzzle = generate_puzzle(9000 + num_train * 7 + num_test, num_train, num_test)
        _require(len(puzzle["train"]) == num_train and len(puzzle["test"]) == num_test,
                 "requested counts ignored")
    for num_train, num_test in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(7, num_train, num_test)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {num_train}/{num_test}")
    # Hand-built check: one east-walking prototype, two same-side lone keys on
    # different diagonals (unchanged body) and one lone key beyond the beacon
    # (mirrored body).  Every creature takes exactly one king step.
    source = _rows_to_grid((
        "0000000000000",
        "0000002000000",
        "0000000000000",
        "0770000000000",
        "0020000005002",
        "0700000000000",
        "0000000000000",
        "0000002000000",
        "0000000000000",
    ))
    expected = _rows_to_grid((
        "0000000000000",
        "0000007700000",
        "0000000200000",
        "0077007000077",
        "0002000005020",
        "0070007700007",
        "0000000200000",
        "0000007000000",
        "0000000000000",
    ))
    _require(np.array_equal(solve(source), expected), "hand-built beacon march failed")
    info = _parse(source)
    _require(not np.array_equal(_render(info, facing="never"), expected),
             "ignoring the facing rule must change the hand-built answer")
    _require(not np.array_equal(_render(info, facing="half_turn"), expected),
             "a half-turn must differ from a mirror on the hand-built answer")
    _require(not np.array_equal(_render(info, motion="none"), expected),
             "ignoring the step must change the hand-built answer")
    for bad in (_rows_to_grid(("0000000", "0200050", "0700000", "0700005", "0000000")),  # two beacons
                _rows_to_grid(("0000000", "0200000", "0700000", "0700005", "0000000")),  # off-ray
                _rows_to_grid(("00500", "00000", "00000", "00200", "00770"))):  # vertical
        try:
            solve(bad)
        except ValueError:
            continue
        raise AssertionError("solve accepted an input outside the family")


def _rows_to_grid(rows):
    return np.array([[int(ch) for ch in row] for row in rows], dtype=int)
