"""Arrow-launched detouring beams that paint the border they actually leave through.

Composition of 13f06aa5 (directional signal-bearing bars that colour a whole
border, with a black junction where two coloured borders meet) and 69889d6e
(a two-cell diagonal beam that is pushed sideways by the minimum amount needed to
clear an obstacle, joined to its new position by a connector).

Rule, applied to every pair of every puzzle and derivable from the input alone:

* The background is the unique most common colour.  Each two-colour component
  shaped like a broad front bar (width 3, 5 or 7) with a centred narrower rear bar
  is a controller; its centre-of-front cell carries the signal colour.  The facing
  is the direction from the rear bar through the signal.  Every other component is
  a solid obstacle.
* From each signal a two-cell beam advances in the controller's own frame: one
  step forward and one step to the controller's right (clockwise of the facing).
  The first front lies directly ahead of the signal and one cell to its right.
  When the next proposed two-cell front would overlap any non-background cell, it
  slides further right by the minimum amount that clears it, and the current front
  row is extended rightward to connect the old and the shifted front.
* A beam stops when it can no longer advance: either its front reaches the border
  ahead (it leaves through the facing side) or it is clipped to one cell against
  the border on its right (it leaves through the clockwise side).
* The whole border side through which a beam leaves is painted in its signal
  colour.  A corner shared by two painted sides becomes black (0).
* Controllers, obstacles and all other cells are preserved.
"""
from __future__ import annotations

import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left; side index = direction index
SIDE_NAMES = ("top", "right", "bottom", "left")
WIDTHS = (3, 5, 7)
JUNCTION = 0
CONTACTS = tuple((length, offset) for length in (1, 2, 3) for offset in range(1 - length, 2))
REGIMES = ("flip", "crowd", "twins", "deflect", "compact")


# --------------------------------------------------------------------------- geometry

def _contact_kind(length, offset):
    covered = set(range(offset, offset + length))
    return {(True, False): "left", (False, True): "right", (True, True): "both"}[
        (0 in covered, 1 in covered)]


def _frame(shape, signal, direction):
    """Map (forward k, lateral l) in a controller frame to grid cells.

    Lateral increases toward the controller's right (clockwise of its facing).
    Returns the mapping, the forward distance to the border ahead and the lateral
    distance to the border on the right.
    """
    height, width = shape
    row, col = signal
    fr, fc = DIRS[direction]
    lr, lc = DIRS[(direction + 1) % 4]
    forward = (row, width - 1 - col, height - 1 - row, col)[direction]
    lateral = (width - 1 - col, height - 1 - row, col, row)[direction]

    def cell(k, l):
        return (row + k * fr + l * lr, col + k * fc + l * lc)

    return cell, forward, lateral


def _signal_from_frame(shape, direction, forward, lateral):
    height, width = shape
    if direction == 0:
        return (forward, width - 1 - lateral)
    if direction == 1:
        return (height - 1 - lateral, width - 1 - forward)
    if direction == 2:
        return (height - 1 - forward, lateral)
    return (lateral, forward)


def _footprint(signal, direction, width):
    row, col = signal
    fr, fc = DIRS[direction]
    lr, lc = DIRS[(direction + 1) % 4]
    half = width // 2
    cells = [(row + l * lr, col + l * lc) for l in range(-half, half + 1)]
    cells += [(row - fr + l * lr, col - fc + l * lc) for l in range(-half + 1, half)]
    return frozenset(cells)


def _trace(blocked, signal, direction):
    """Follow one beam on the visible input.  Raises ValueError outside the family."""
    cell, forward_limit, lateral_limit = _frame(blocked.shape, signal, direction)
    if forward_limit < 2 or lateral_limit < 1:
        raise ValueError("controller has no room to launch a beam")
    first = [cell(1, 0), cell(1, 1)]
    if any(blocked[p] for p in first):
        raise ValueError("initial beam front is blocked")
    cells = list(first)
    fronts = [(1, 0)]
    collisions = []
    k, lateral = 1, 0
    while True:
        if k == forward_limit:
            if lateral + 1 >= lateral_limit:
                raise ValueError("beam reaches a corner ambiguously")
            exit_side, kind = direction, "forward"
            break
        proposed = lateral + 1
        if proposed > lateral_limit:
            exit_side, kind = (direction + 1) % 4, "lateral"
            break
        landed = proposed
        while landed <= lateral_limit and any(
                blocked[cell(k + 1, j)] for j in (landed, landed + 1) if j <= lateral_limit):
            landed += 1
        if landed > lateral_limit:
            raise ValueError("beam is pinned against the border")
        if landed != proposed:
            hits = tuple(j - proposed for j in (proposed, proposed + 1)
                         if j <= lateral_limit and blocked[cell(k + 1, j)])
            connector = [cell(k, j) for j in range(lateral, landed + 1)]
            if any(blocked[p] for p in connector):
                raise ValueError("connector is obstructed")
            cells.extend(p for p in connector if p not in cells[-2:])
            collisions.append({"step": k + 1, "proposed": proposed, "landed": landed,
                               "shift": landed - proposed,
                               "hit_cells": tuple(cell(k + 1, proposed + h) for h in hits),
                               "contact": {(0,): "left", (1,): "right", (0, 1): "both"}[hits]})
        k, lateral = k + 1, landed
        cells.append(cell(k, lateral))
        if lateral + 1 <= lateral_limit:
            cells.append(cell(k, lateral + 1))
        fronts.append((k, lateral))
    seen, ordered = set(), []
    for p in cells:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return {"cells": tuple(ordered), "exit": exit_side, "kind": kind, "fronts": tuple(fronts),
            "collisions": tuple(collisions), "forward_limit": forward_limit,
            "lateral_limit": lateral_limit}


def _components(mask):
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack, comp = [(r, c)], []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            result.append(tuple(sorted(comp)))
    return result


def _as_controller(grid, comp):
    cells = frozenset(comp)
    found = []
    for width in WIDTHS:
        if len(cells) != 2 * width - 2:
            continue
        for center in cells:
            for direction in range(4):
                if _footprint(center, direction, width) != cells:
                    continue
                signal = int(grid[center])
                carriers = {int(grid[p]) for p in cells if p != center}
                if len(carriers) == 1 and signal not in carriers:
                    found.append({"signal": center, "direction": direction, "width": width,
                                  "color": signal, "carrier": carriers.pop(), "cells": cells})
    if len(found) > 1:
        raise ValueError("component has several controller readings")
    return found[0] if found else None


def _analyze(grid):
    grid = np.asarray(grid)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("grid must be a nonempty 2D array")
    grid = grid.astype(int)
    counts = np.bincount(grid.ravel(), minlength=10)
    modal = np.flatnonzero(counts == counts.max())
    if len(modal) != 1:
        raise ValueError("background must be the unique most common colour")
    background = int(modal[0])
    controllers, obstacles = [], []
    for comp in _components(grid != background):
        parsed = _as_controller(grid, comp)
        if parsed is not None:
            controllers.append(parsed)
        else:
            obstacles.append(comp)
    if not controllers:
        raise ValueError("no controller found")
    controllers.sort(key=lambda c: c["signal"])
    blocked = grid != background
    for controller in controllers:
        controller["trace"] = _trace(blocked, controller["signal"], controller["direction"])
    sides = [c["trace"]["exit"] for c in controllers]
    if len(set(sides)) != len(sides):
        raise ValueError("two beams leave through the same side")
    return {"background": background, "controllers": controllers, "obstacles": obstacles}


def _corners(shape):
    height, width = shape
    return (((0, 0), (0, 3)), ((0, width - 1), (0, 1)),
            ((height - 1, width - 1), (2, 1)), ((height - 1, 0), (2, 3)))


def _render(grid, info):
    grid = np.asarray(grid, dtype=int)
    output = grid.copy()
    background = info["background"]
    painted = {}
    for controller in info["controllers"]:
        for p in controller["trace"]["cells"]:
            if grid[p] == background:
                output[p] = controller["color"]
        painted[controller["trace"]["exit"]] = controller["color"]
    return _paint_sides(output, painted)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the output from the visible input only."""
    grid = np.asarray(input_grid, dtype=int)
    return _render(grid, _analyze(grid))


# --------------------------------------------------------------------------- shortcuts

def _paint_sides(output, painted, junction=True):
    for side, color in painted.items():
        if side == 0:
            output[0, :] = color
        elif side == 1:
            output[:, -1] = color
        elif side == 2:
            output[-1, :] = color
        else:
            output[:, 0] = color
    if junction:
        for pos, (a, b) in _corners(output.shape):
            if a in painted and b in painted:
                output[pos] = JUNCTION
    return output


def _shortcut_outputs(grid):
    """Plausible simpler rules; each must be refuted by the training examples.

    facing_side   -- correct beams, but paint the side the controller faces (parent A only).
    no_detour     -- beams ignore obstacles (parent B's deflection dropped).
    no_border     -- beams only, no border completion or junctions.
    no_junction   -- correct beams and sides, corners never turn black.
    straight_ray  -- parent A unchanged: dotted straight ray and the faced side.
    mirrored      -- beams drift counter-clockwise instead of clockwise.
    """
    grid = np.asarray(grid, dtype=int)
    info = _analyze(grid)
    background = info["background"]
    controllers = info["controllers"]
    results = {}

    def beams(out, traces):
        for c, trace in zip(controllers, traces):
            for p in trace["cells"]:
                if grid[p] == background:
                    out[p] = c["color"]
        return out

    traces = [c["trace"] for c in controllers]
    out = beams(grid.copy(), traces)
    results["facing_side"] = _paint_sides(out.copy(), {c["direction"]: c["color"] for c in controllers})
    results["no_border"] = out.copy()
    results["no_junction"] = _paint_sides(out.copy(), {t["exit"]: c["color"]
                                                       for c, t in zip(controllers, traces)}, False)
    arrows_only = np.zeros(grid.shape, dtype=bool)
    for c in controllers:
        for p in c["cells"]:
            arrows_only[p] = True
    try:
        bare = [_trace(arrows_only, c["signal"], c["direction"]) for c in controllers]
        sides = {t["exit"]: c["color"] for c, t in zip(controllers, bare)}
        results["no_detour"] = (_paint_sides(beams(grid.copy(), bare), sides)
                                if len(sides) == len(bare) else None)
    except ValueError:
        results["no_detour"] = None
    straight = grid.copy()
    for c in controllers:
        cell, forward, _ = _frame(grid.shape, c["signal"], c["direction"])
        for k in range(2, forward + 1, 2):
            straight[cell(k, 0)] = c["color"]
    results["straight_ray"] = _paint_sides(straight, {c["direction"]: c["color"] for c in controllers})
    try:
        results["mirrored"] = np.fliplr(solve(np.fliplr(grid)))
    except ValueError:
        results["mirrored"] = None
    return results


# --------------------------------------------------------------------------- facts

def _chebyshev_far(cells_a, cells_b, gap=2):
    return all(max(abs(a - c), abs(b - d)) >= gap for a, b in cells_a for c, d in cells_b)


def _pair_facts(grid):
    """Analyse one input and check every family invariant; returns descriptive facts."""
    grid = np.asarray(grid, dtype=int)
    info = _analyze(grid)
    background = info["background"]
    height, width = grid.shape
    if not (1 <= height <= 30 and 1 <= width <= 30):
        raise AssertionError("illegal dimensions")
    if np.any(grid == JUNCTION):
        raise AssertionError("junction colour 0 is reserved for outputs")
    border = np.concatenate([grid[0], grid[-1], grid[:, 0], grid[:, -1]])
    if np.any(border != background):
        raise AssertionError("inputs keep every border cell as background")
    controllers = info["controllers"]
    if not 1 <= len(controllers) <= 4:
        raise AssertionError("one to four controllers per pair")
    signals = [c["color"] for c in controllers]
    if len(set(signals)) != len(signals):
        raise AssertionError("signal colours are distinct within a pair")
    carriers = {c["carrier"] for c in controllers}
    if carriers & set(signals):
        raise AssertionError("carrier and signal roles overlap")
    obstacle_colors = {int(grid[p]) for comp in info["obstacles"] for p in comp}
    if len(obstacle_colors) > 1 or obstacle_colors & (carriers | set(signals)):
        raise AssertionError("obstacles share one dedicated colour")
    hit_cells = set()
    for c in controllers:
        for col in c["trace"]["collisions"]:
            hit_cells.update(col["hit_cells"])
    runs = []
    for comp in info["obstacles"]:
        rows = {r for r, _ in comp}
        cols = {q for _, q in comp}
        straight = (len(rows) == 1 and max(cols) - min(cols) + 1 == len(comp)) or \
                   (len(cols) == 1 and max(rows) - min(rows) + 1 == len(comp))
        if not straight or len(comp) > 3:
            raise AssertionError("obstacles are straight runs of one to three cells")
        runs.append({"cells": comp, "hit": bool(hit_cells & set(comp))})
    beams = [set(c["trace"]["cells"]) for c in controllers]
    for i, c in enumerate(controllers):
        if any(grid[p] != background for p in beams[i]):
            raise AssertionError("beams only cover background cells")
        for j, other in enumerate(controllers):
            if i == j:
                continue
            if j > i and not _chebyshev_far(beams[i], beams[j]):
                raise AssertionError("beams stay separated")
            if not _chebyshev_far(beams[i], other["cells"]):
                raise AssertionError("beams keep clear of other controllers")
        for col in c["trace"]["collisions"]:
            if any(p in other["cells"] for other in controllers for p in col["hit_cells"]):
                raise AssertionError("only obstacles deflect beams")
    arrows_only = np.zeros(grid.shape, dtype=bool)
    for c in controllers:
        for p in c["cells"]:
            arrows_only[p] = True
    flips = natural_lateral = 0
    kinds = []
    for c in controllers:
        kinds.append(c["trace"]["kind"])
        if c["trace"]["kind"] == "lateral":
            try:
                bare = _trace(arrows_only, c["signal"], c["direction"])
            except ValueError:
                bare = None
            if bare is not None and bare["kind"] == "forward":
                flips += 1
            if bare is not None and bare["kind"] == "lateral":
                natural_lateral += 1
    sides = {c["trace"]["exit"] for c in controllers}
    junctions = sum(1 for _, (a, b) in _corners(grid.shape) if a in sides and b in sides)
    contacts = [(col["contact"], col["shift"],
                 next(len(r["cells"]) for r in runs if col["hit_cells"][0] in r["cells"]))
                for c in controllers for col in c["trace"]["collisions"]]
    facings = [c["direction"] for c in controllers]
    twins = any(facings.count(d) >= 2 and len({c["trace"]["kind"] for c in controllers
                                               if c["direction"] == d}) == 2 for d in set(facings))
    return {"shape": (height, width), "count": len(controllers), "facings": facings,
            "widths": [c["width"] for c in controllers], "kinds": kinds, "flips": flips,
            "natural_lateral": natural_lateral,
            "sides": sorted(sides), "junctions": junctions, "contacts": contacts,
            "beam_contacts": [len(c["trace"]["collisions"]) for c in controllers],
            "decoys": sum(1 for r in runs if not r["hit"]), "twins": twins,
            "background": background, "carriers": sorted(carriers),
            "palette": sorted({int(v) for v in np.unique(grid)})}


# --------------------------------------------------------------------------- sampling

def _check_counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= int(value) <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _near(mask, cells):
    height, width = mask.shape
    for r, c in cells:
        r0, r1 = max(0, r - 1), min(height, r + 2)
        c0, c1 = max(0, c - 1), min(width, c + 2)
        if mask[r0:r1, c0:c1].any():
            return True
    return False


def _interior(shape, cells):
    height, width = shape
    return all(1 <= r <= height - 2 and 1 <= c <= width - 2 for r, c in cells)


def _gapped_steps(rng, count, low, high):
    """Choose ``count`` increasing steps in [low, high] that differ by at least two."""
    if count == 0:
        return []
    span = high - (count - 1) - low + 1
    if span < count:
        return None
    base = sorted(int(x) for x in rng.choice(np.arange(low, high - (count - 1) + 1),
                                             size=count, replace=False))
    return [b + i for i, b in enumerate(base)]


def _frame_targets(rng, spec, shape):
    """Choose forward/lateral room that makes the requested exit plausible."""
    height, width = shape
    direction, half = spec["facing"], spec["width"] // 2
    contacts = spec["contacts"]
    count = len(contacts)
    shift = sum(length + offset for length, offset in contacts)
    along, across = (height, width) if direction % 2 == 0 else (width, height)
    f_low, f_high = max(4, 2 * count + 1), along - 3
    l_low, l_high = half + 1, across - 2 - half
    if f_high < f_low or l_high < l_low:
        return None
    if spec["kind"] == "forward":
        top = min(f_high, l_high - shift - 1)
        if top < f_low:
            return None
        forward = int(rng.integers(f_low, top + 1))
        lateral = int(rng.integers(max(l_low, forward + shift + 1), l_high + 1))
        return forward, lateral, forward
    if spec["flip"]:
        options = []
        for lateral in range(max(l_low, 2 * count + shift, 4), l_high + 1):
            lo, hi = max(f_low, lateral + 2 - shift), min(f_high, lateral - 1)
            if lo <= hi:
                options.append((lateral, lo, hi))
        if not options:
            return None
        lateral, lo, hi = options[int(rng.integers(len(options)))]
        forward = int(rng.integers(lo, hi + 1))
        return forward, lateral, lateral + 1 - shift
    options = [lat for lat in range(max(l_low, 2 * count + shift, shift + 3), l_high + 1)
               if max(f_low, lat + 2) <= f_high]
    if not options:
        return None
    lateral = options[int(rng.integers(len(options)))]
    forward = int(rng.integers(max(f_low, lateral + 2), f_high + 1))
    return forward, lateral, lateral + 1 - shift


def _try_controller(rng, grid, background, busy, placed, spec, colors):
    shape = grid.shape
    target = _frame_targets(rng, spec, shape)
    if target is None:
        return None
    forward, lateral_limit, exit_step = target
    direction, width = spec["facing"], spec["width"]
    signal = _signal_from_frame(shape, direction, forward, lateral_limit)
    foot = _footprint(signal, direction, width)
    if not _interior(shape, foot) or any(grid[p] != background for p in foot) or _near(busy, foot):
        return None
    contacts = list(spec["contacts"])
    steps = _gapped_steps(rng, len(contacts), 2, exit_step - 1)
    if steps is None:
        return None
    plan = dict(zip(steps, contacts))
    cell, _, _ = _frame(shape, signal, direction)
    runs, k, lateral = [], 1, 0
    while k < forward:
        proposed = lateral + 1
        if proposed > lateral_limit:
            break
        if k + 1 in plan:
            length, offset = plan[k + 1]
            run = [cell(k + 1, j) for j in range(proposed + offset, proposed + offset + length)]
            if not _interior(shape, run) or any(grid[p] != background for p in run) or _near(busy, run):
                return None
            runs.append(run)
            lateral = proposed + offset + length
        else:
            lateral = proposed
        k += 1
    if len(runs) != len(contacts):
        return None
    trial = grid.copy()
    for p in foot:
        trial[p] = colors["carrier"]
    trial[signal] = colors["signal"]
    for run in runs:
        for p in run:
            trial[p] = colors["obstacle"]
    blocked = trial != background
    try:
        trace = _trace(blocked, signal, direction)
    except ValueError:
        return None
    if trace["kind"] != spec["kind"]:
        return None
    if [c["step"] for c in trace["collisions"]] != steps:
        return None
    if [c["shift"] for c in trace["collisions"]] != [plan[s][0] + plan[s][1] for s in steps]:
        return None
    if _near(busy, trace["cells"]):
        return None
    arrows_only = np.zeros(shape, dtype=bool)
    for entry in placed:
        for p in entry["cells"]:
            arrows_only[p] = True
    for p in foot:
        arrows_only[p] = True
    if spec["kind"] == "lateral":
        try:
            bare = _trace(arrows_only, signal, direction)
        except ValueError:
            bare = None
        if spec["flip"] and (bare is None or bare["kind"] != "forward"):
            return None
        if not spec["flip"] and (bare is None or bare["kind"] != "lateral"):
            return None
    for entry in placed:
        try:
            again = _trace(blocked, entry["signal"], entry["direction"])
        except ValueError:
            return None
        if again["cells"] != entry["trace"]["cells"] or again["exit"] != entry["trace"]["exit"]:
            return None
    new_busy = busy.copy()
    for p in list(foot) + [q for run in runs for q in run] + list(trace["cells"]):
        new_busy[p] = True
    entry = {"signal": signal, "direction": direction, "cells": foot, "trace": trace}
    return trial, new_busy, entry


def _try_decoy(rng, grid, background, busy, placed, color):
    shape = grid.shape
    height, width = shape
    style = rng.choice(("ahead", "near", "free"), p=(.4, .35, .25))
    length = int(rng.integers(1, 4))
    vertical = bool(rng.integers(2))
    if style == "ahead" and placed:
        entry = placed[int(rng.integers(len(placed)))]
        cell, forward, _ = _frame(shape, entry["signal"], entry["direction"])
        if forward - 1 < 4:
            return None
        k = int(rng.integers(4, forward))
        if rng.random() < .5:
            start = int(rng.integers(-1, 1))
            run = [cell(k, start + i) for i in range(min(length, 2))]
        else:
            run = [cell(k + i, 0) for i in range(length)]
    elif style == "near" and placed:
        entry = placed[int(rng.integers(len(placed)))]
        cells = entry["trace"]["cells"]
        r, c = cells[int(rng.integers(len(cells)))]
        dr, dc = int(rng.integers(-3, 4)), int(rng.integers(-3, 4))
        if max(abs(dr), abs(dc)) < 2:
            return None
        run = [(r + dr + (i if vertical else 0), c + dc + (0 if vertical else i)) for i in range(length)]
    else:
        r0, c0 = int(rng.integers(1, height - 1)), int(rng.integers(1, width - 1))
        run = [(r0 + (i if vertical else 0), c0 + (0 if vertical else i)) for i in range(length)]
    if not _interior(shape, run) or _near(busy, run):
        return None
    trial = grid.copy()
    for p in run:
        trial[p] = color
    blocked = trial != background
    for entry in placed:
        try:
            again = _trace(blocked, entry["signal"], entry["direction"])
        except ValueError:
            return None
        if again["cells"] != entry["trace"]["cells"]:
            return None
    new_busy = busy.copy()
    for p in run:
        new_busy[p] = True
    return trial, new_busy, style


def _palette(rng, count, carrier_mode):
    colors = [int(c) for c in rng.permutation(np.arange(1, 10))]
    background, obstacle = colors.pop(), colors.pop()
    if carrier_mode == "shared" or count == 4 and carrier_mode == "distinct":
        pool = [colors.pop()] if carrier_mode == "shared" else [colors.pop(), colors.pop()]
        carriers = [pool[i % len(pool)] for i in range(count)]
    else:
        carriers = [colors.pop() for _ in range(count)]
    signals = [colors.pop() for _ in range(count)]
    return {"background": background, "obstacle": obstacle, "carriers": carriers, "signals": signals}


def _size(rng, count, rules):
    low = {1: 9, 2: 11, 3: 14, 4: 16}[count]
    high = {1: 16, 2: 22, 3: 27, 4: 30}[count]
    if rules.get("compact"):
        high = max(low + 2, high - 6)
    if rules.get("roomy"):
        low = min(high - 2, low + 4)
    return int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1))


def _sample_contacts(rng, count, allowed, need_shift=0):
    for _ in range(40):
        picked = [allowed[int(rng.integers(len(allowed)))] for _ in range(count)]
        if sum(a + b for a, b in picked) >= need_shift:
            return picked
    return None


def _sample_specs(rng, rules, vocab):
    """Turn pair-level requirements into controller specs."""
    allowed = vocab["contacts"]
    for _ in range(300):
        count = int(rng.choice(rules["counts"]))
        sides = [int(s) for s in rng.choice(4, size=count, replace=False)]
        kinds = [str(rng.choice(("forward", "lateral"))) for _ in sides]
        if rules.get("twins"):
            if count < 2:
                continue
            d = int(rng.choice(sides))
            if (d + 1) % 4 not in sides:
                continue
            kinds[sides.index(d)] = "forward"
            kinds[sides.index((d + 1) % 4)] = "lateral"
        facings = [s if k == "forward" else (s - 1) % 4 for s, k in zip(sides, kinds)]
        if not set(facings) <= vocab["facings"]:
            continue
        if rules.get("adjacent") and not any((a - b) % 4 in (1, 3) for a in sides for b in sides):
            continue
        if rules.get("both_kinds") and len(set(kinds)) < 2:
            continue
        if rules.get("need_lateral") and "lateral" not in kinds:
            continue
        if rules.get("min_facings", 1) > len(set(facings)):
            continue
        lateral_idx = [i for i, k in enumerate(kinds) if k == "lateral"]
        flip_idx = set()
        if rules.get("flip"):
            if not lateral_idx:
                continue
            flip_idx.add(int(rng.choice(lateral_idx)))
        for i in lateral_idx:
            if i not in flip_idx and rng.random() < rules.get("flip_rate", 0.0):
                flip_idx.add(i)
        specs, total_contacts = [], 0
        max_per = rules.get("max_beam_contacts", 3)
        for i, (side, kind, facing) in enumerate(zip(sides, kinds, facings)):
            flip = i in flip_idx
            number = int(rng.choice((0, 1, 2, 3), p=(.32, .4, .2, .08)))
            number = min(number, max_per)
            if flip:
                number = max(number, 1)
            contacts = _sample_contacts(rng, number, allowed, 3 if flip else 0)
            if contacts is None:
                break
            width = int(rng.choice(sorted(vocab["widths"])))
            specs.append({"side": side, "kind": kind, "facing": facing, "flip": flip,
                          "width": width, "contacts": contacts})
            total_contacts += number
        if len(specs) != count:
            continue
        need = rules.get("min_contacts", 0)
        if total_contacts < need:
            # Add contacts to beams that can take them.
            order = [int(i) for i in rng.permutation(count)]
            for i in order:
                while total_contacts < need and len(specs[i]["contacts"]) < max_per:
                    specs[i]["contacts"].append(allowed[int(rng.integers(len(allowed)))])
                    total_contacts += 1
            if total_contacts < need:
                continue
        if rules.get("distinct_shifts"):
            shifts = {a + b for s in specs for a, b in s["contacts"]}
            if len(shifts) < 2:
                continue
        if rules.get("min_beam_contacts"):
            target = rules["min_beam_contacts"]
            i = int(rng.integers(count))
            while len(specs[i]["contacts"]) < target:
                specs[i]["contacts"].append(allowed[int(rng.integers(len(allowed)))])
        if rules.get("need_contact_lateral"):
            lat = [s for s in specs if s["kind"] == "lateral"]
            if not any(s["contacts"] for s in lat):
                lat[int(rng.integers(len(lat)))]["contacts"].append(
                    allowed[int(rng.integers(len(allowed)))])
        return specs
    return None


def _build_pair(rng, rules, vocab, carrier_mode):
    for _ in range(60):
        specs = _sample_specs(rng, rules, vocab)
        if specs is None:
            continue
        count = len(specs)
        shape = _size(rng, count, rules)
        palette = _palette(rng, count, carrier_mode)
        background = palette["background"]
        grid = np.full(shape, background, dtype=int)
        busy = np.zeros(shape, dtype=bool)
        placed = []
        order = sorted(range(count), key=lambda i: (-len(specs[i]["contacts"]) - 2 * specs[i]["flip"],
                                                   float(rng.random())))
        ok = True
        for i in order:
            colors = {"carrier": palette["carriers"][i], "signal": palette["signals"][i],
                      "obstacle": palette["obstacle"]}
            for _ in range(45):
                result = _try_controller(rng, grid, background, busy, placed, specs[i], colors)
                if result is not None:
                    grid, busy, entry = result
                    placed.append(entry)
                    break
            else:
                ok = False
                break
        if not ok:
            continue
        low, high = rules.get("decoys", (0, 2))
        wanted = int(rng.integers(low, high + 1))
        added = 0
        for _ in range(wanted * 30):
            if added >= wanted:
                break
            result = _try_decoy(rng, grid, background, busy, placed, palette["obstacle"])
            if result is not None:
                grid, busy, _ = result
                added += 1
        if added < low:
            continue
        try:
            facts = _pair_facts(grid)
        except (ValueError, AssertionError):
            continue
        if not _satisfies(facts, rules):
            continue
        return {"input": grid, "output": solve(grid), "facts": facts}
    return None


def _satisfies(facts, rules):
    if facts["count"] not in rules["counts"]:
        return False
    if rules.get("adjacent") and facts["junctions"] < 1:
        return False
    if rules.get("both_kinds") and len(set(facts["kinds"])) < 2:
        return False
    if rules.get("need_lateral") and "lateral" not in facts["kinds"]:
        return False
    if rules.get("flip") and facts["flips"] < 1:
        return False
    if rules.get("twins") and not facts["twins"]:
        return False
    if facts["decoys"] < rules.get("decoys", (0, 2))[0]:
        return False
    if len(facts["contacts"]) < rules.get("min_contacts", 0):
        return False
    if rules.get("distinct_shifts") and len({s for _, s, _ in facts["contacts"]}) < 2:
        return False
    if rules.get("min_beam_contacts") and max(facts["beam_contacts"]) < rules["min_beam_contacts"]:
        return False
    if rules.get("max_beam_contacts") is not None and max(facts["beam_contacts"]) > rules["max_beam_contacts"]:
        return False
    if rules.get("need_contact_lateral") and not any(
            k == "lateral" and n > 0 for k, n in zip(facts["kinds"], facts["beam_contacts"])):
        return False
    return True


def _train_vocab(train_facts):
    facings = {d for f in train_facts for d in f["facings"]}
    widths = {w for f in train_facts for w in f["widths"]}
    kinds = {k for f in train_facts for k, _, _ in f["contacts"]}
    max_len = max((n for f in train_facts for _, _, n in f["contacts"]), default=0)
    contacts = [c for c in CONTACTS if _contact_kind(*c) in kinds and c[0] <= max_len]
    return {"facings": facings, "widths": widths, "contacts": contacts}


def _training_ok(train_facts):
    """Training must expose every sub-rule a test can need."""
    kinds = {k for f in train_facts for k in f["kinds"]}
    shifts = {s for f in train_facts for _, s, _ in f["contacts"]}
    facings = {d for f in train_facts for d in f["facings"]}
    return (kinds == {"forward", "lateral"} and any(f["junctions"] for f in train_facts)
            and len(shifts) >= 2 and len(facings) >= 2 and any(f["decoys"] for f in train_facts)
            and any(s != n for f in train_facts for _, s, n in f["contacts"])
            and any(f["natural_lateral"] for f in train_facts))


def _plan(rng, regime, num_train, num_test):
    counts_all = (1, 2, 3, 4)
    base = {"counts": counts_all, "decoys": (0, 2)}
    teaching = {"counts": (2, 3), "adjacent": True, "both_kinds": True, "min_contacts": 2,
                "distinct_shifts": True, "decoys": (1, 2), "min_facings": 2}
    train_rules = [dict(base, flip_rate=0.15) for _ in range(num_train)]
    # Every test needs both parents: a deflection and a beam leaving through its side.
    test_rules = [dict(base, need_lateral=True, min_contacts=1, flip_rate=0.25) for _ in range(num_test)]
    if regime == "flip":
        for r in test_rules:
            r.update(flip=True, counts=(1, 2, 3))
    elif regime == "crowd":
        teaching["counts"] = (2,)
        for r in train_rules:
            r["counts"] = (1, 2)
        for r in test_rules:
            r.update(counts=(3, 4), adjacent=True, roomy=True)
    elif regime == "twins":
        for r in test_rules:
            r.update(twins=True, counts=(2, 3, 4))
    elif regime == "deflect":
        teaching["max_beam_contacts"] = 1
        teaching["counts"] = (2, 3)
        for r in train_rules:
            r["max_beam_contacts"] = 1
        for r in test_rules:
            r.update(min_beam_contacts=int(rng.choice((2, 3))), counts=(1, 2, 3))
    elif regime == "compact":
        for r in train_rules:
            r["counts"] = (2, 3, 4)
        for r in test_rules:
            r.update(counts=(1, 2), need_lateral=True, need_contact_lateral=True, compact=True)
    teach_at = int(rng.integers(num_train))
    train_rules[teach_at] = dict(train_rules[teach_at], **teaching)
    if regime == "flip" and num_train >= 3:
        others = [i for i in range(num_train) if i != teach_at]
        i = int(rng.choice(others))
        train_rules[i] = dict(train_rules[i], flip=True, counts=(1, 2, 3))
    return train_rules, test_rules


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(REGIMES))
    widths_pool = [(5,), (3, 5), (5, 7), (3, 5, 7), (3, 7), (3,)]
    puzzle_widths = set(widths_pool[int(rng.choice(len(widths_pool), p=(.15, .25, .2, .25, .1, .05)))])
    carrier_mode = str(rng.choice(("shared", "distinct")))
    full_vocab = {"facings": {0, 1, 2, 3}, "widths": puzzle_widths, "contacts": list(CONTACTS)}
    for _ in range(40):
        train_rules, test_rules = _plan(rng, regime, num_train, num_test)
        train, seen = [], set()
        for rules in train_rules:
            pair = None
            for _ in range(6):
                pair = _build_pair(rng, rules, full_vocab, carrier_mode)
                if pair is not None and pair["input"].tobytes() + bytes(pair["input"].shape) not in seen:
                    break
                pair = None
            if pair is None:
                break
            seen.add(pair["input"].tobytes() + bytes(pair["input"].shape))
            train.append(pair)
        if len(train) != num_train or not _training_ok([p["facts"] for p in train]):
            continue
        vocab = _train_vocab([p["facts"] for p in train])
        if regime == "crowd":
            top = max(p["facts"]["count"] for p in train)
            for r in test_rules:
                r["counts"] = tuple(n for n in (3, 4) if n > top) or (4,)
        test = []
        for rules in test_rules:
            pair = None
            for _ in range(6):
                pair = _build_pair(rng, rules, vocab, carrier_mode)
                if pair is not None and pair["input"].tobytes() + bytes(pair["input"].shape) not in seen:
                    break
                pair = None
            if pair is None:
                break
            seen.add(pair["input"].tobytes() + bytes(pair["input"].shape))
            test.append(pair)
        if len(test) != num_test:
            continue
        puzzle = {"train": [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in train],
                  "test": [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in test]}
        validate_generated_puzzle(puzzle)
        return puzzle
    raise RuntimeError("could not build a puzzle for this seed")


# --------------------------------------------------------------------------- validation

def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "train/test keys")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits are lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "pair counts")
    seen = set()
    facts = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            grid = np.asarray(pair["input"])
            out = np.asarray(pair["output"])
            for g in (grid, out):
                _require(g.ndim == 2 and np.issubdtype(g.dtype, np.integer), "integer 2D grids")
                _require(all(1 <= n <= 30 for n in g.shape), "dimensions 1..30")
                _require(bool(np.all((g >= 0) & (g <= 9))), "colours 0..9")
            _require(grid.shape == out.shape, "shape preserved")
            key = (grid.shape, grid.astype(np.int8).tobytes())
            _require(key not in seen, "inputs are distinct")
            seen.add(key)
            info = _pair_facts(grid)
            _require(np.array_equal(solve(grid), out), "output follows the rule")
            # Independent re-statement of the border/corner part of the output.
            sides = set(info["sides"])
            height, width = grid.shape
            for pos, (a, b) in _corners(grid.shape):
                if a in sides and b in sides:
                    _require(out[pos] == JUNCTION, "junction corner")
                else:
                    _require(out[pos] != JUNCTION, "junction only where two painted sides meet")
            for side in range(4):
                line = (out[0, 1:-1], out[1:-1, -1], out[-1, 1:-1], out[1:-1, 0])[side]
                if side in sides:
                    _require(len(set(line.tolist())) <= 1, "painted side is uniform")
                else:
                    _require(not np.any(line == JUNCTION), "unpainted side keeps no junction colour")
            changed = out != grid
            _require(changed.any(), "transformation is not the identity")
            _require(bool(np.all(out[grid != info["background"]] == grid[grid != info["background"]])),
                     "controllers and obstacles are preserved")
            info["shortcut_wrong"] = {name: guess is None or not np.array_equal(guess, out)
                                      for name, guess in _shortcut_outputs(grid).items()}
            facts[split].append(info)
    for name in facts["train"][0]["shortcut_wrong"]:
        _require(any(f["shortcut_wrong"][name] for f in facts["train"]),
                 f"training refutes the {name} shortcut")
    for info in facts["test"]:
        _require(info["contacts"] and "lateral" in info["kinds"], "each test needs a detour and a side exit")
        _require(all(info["shortcut_wrong"][name] for name in
                     ("facing_side", "no_detour", "no_border", "straight_ray")),
                 "ignoring either parent changes every test answer")
    tf = facts["train"]
    _require(_training_ok(tf), "training reveals forward and lateral exits, a junction, "
                               "two shift sizes, two facings and an unhit obstacle")
    vocab = _train_vocab(tf)
    for info in facts["test"]:
        _require(set(info["facings"]) <= vocab["facings"], "test facing was demonstrated")
        _require(set(info["widths"]) <= vocab["widths"], "test controller width was demonstrated")
        _require(all((n, k) in {(c[0], _contact_kind(*c)) for c in vocab["contacts"]}
                     for k, _, n in info["contacts"]), "test contact type was demonstrated")
        if info["junctions"]:
            _require(any(f["junctions"] for f in tf), "junction demonstrated")


def self_test() -> None:
    import time
    start = time.time()
    stats = {"counts": set(), "widths": set(), "facings": set(), "shapes": set(), "kinds": set(),
             "test_kinds": set(), "flip_puzzles": 0, "test_flip_puzzles": 0, "twins": 0,
             "contacts": set(), "shifts": set(), "junctions": set(), "decoys": 0, "carrier_mix": 0,
             "test_bigger": 0, "test_smaller": 0, "multi_beam_contacts": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        tf = [_pair_facts(np.array(p["input"])) for p in puzzle["train"]]
        ef = [_pair_facts(np.array(p["input"])) for p in puzzle["test"]]
        for f in tf + ef:
            stats["counts"].add(f["count"])
            stats["widths"].update(f["widths"])
            stats["facings"].update(f["facings"])
            stats["shapes"].add(f["shape"])
            stats["kinds"].update(f["kinds"])
            stats["contacts"].update(k for k, _, _ in f["contacts"])
            stats["shifts"].update(s for _, s, _ in f["contacts"])
            stats["junctions"].add(f["junctions"])
            stats["decoys"] += f["decoys"]
            stats["carrier_mix"] += len(f["carriers"]) > 1
            stats["multi_beam_contacts"] += max(f["beam_contacts"]) >= 2
        for f in ef:
            stats["test_kinds"].update(f["kinds"])
        stats["flip_puzzles"] += any(f["flips"] for f in tf + ef)
        stats["test_flip_puzzles"] += any(f["flips"] for f in ef)
        stats["twins"] += any(f["twins"] for f in ef)
        tmax, emax = max(f["count"] for f in tf), max(f["count"] for f in ef)
        stats["test_bigger"] += emax > tmax
        stats["test_smaller"] += emax < tmax
    _require(stats["counts"] == {1, 2, 3, 4}, "controller counts 1..4")
    _require(stats["widths"] == set(WIDTHS), "all controller widths")
    _require(stats["facings"] == {0, 1, 2, 3}, "all facings")
    _require(len(stats["shapes"]) >= 40, "grid shapes vary")
    _require(stats["test_kinds"] == {"forward", "lateral"}, "tests use both exit kinds")
    _require(stats["test_flip_puzzles"] >= 5 and stats["twins"] >= 3, "flip and twin tests occur")
    _require(stats["contacts"] == {"left", "right", "both"} and stats["shifts"] >= {1, 2, 3},
             "contact kinds and shift sizes vary")
    _require(len(stats["junctions"]) >= 4, "junction counts vary")
    _require(stats["decoys"] >= 42 and stats["carrier_mix"] >= 10, "decoys and carrier mixes occur")
    _require(stats["test_bigger"] >= 5 and stats["test_smaller"] >= 5, "train/test scale shifts both ways")
    _require(stats["multi_beam_contacts"] >= 20, "accumulated detours occur")
    for train, test in ((2, 1), (10, 4), (5, 4), (3, 2)):
        puzzle = generate_puzzle(7, train, test)
        _require(len(puzzle["train"]) == train and len(puzzle["test"]) == test, "explicit counts")
        _require(puzzle == generate_puzzle(7, train, test), "deterministic")
    for train, test in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(3, train, test)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    _require(time.time() - start < 240, "self-test stays bounded")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
