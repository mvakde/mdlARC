"""Settle keyed markers, then stably deduplicate the resulting full columns.

The intermediate grid preserves the scaffold; column compaction subsequently
applies to the entire grid, including keys and walls. No parent code is imported.
"""
from __future__ import annotations

import hashlib
import numpy as np


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))
CARDINAL = tuple(d for d in DIRECTIONS if 0 in d)
DIAGONAL = tuple(d for d in DIRECTIONS if 0 not in d)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} outside its allowed range")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must use integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid grid color")
    return a


def _signature(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _components(mask):
    h, w = mask.shape
    seen = ~mask.copy()
    result = []
    for rr, cc in np.argwhere(mask):
        start = int(rr), int(cc)
        if seen[start]:
            continue
        seen[start] = True
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in CARDINAL:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(cells)
    return result


def _decode_glyph(sub):
    carrier = int(sub[0, 0])
    if not (np.all(sub[0] == carrier) and np.all(sub[-1] == carrier)
            and np.all(sub[:, 0] == carrier) and np.all(sub[:, -1] == carrier)):
        return []
    other = set(map(int, np.unique(sub))) - {carrier}
    if len(other) != 2:
        return []
    interpretations = []
    for tip in sorted(other):
        trail = next(iter(other - {tip}))
        tips = set(map(tuple, np.argwhere(sub == tip)))
        trails = set(map(tuple, np.argwhere(sub == trail)))
        if len(tips) < 2:
            continue
        for dr, dc in DIRECTIONS:
            used, lengths = set(), []
            for r, c in sorted(tips):
                stroke = []
                r, c = r - dr, c - dc
                while (r, c) in trails:
                    stroke.append((r, c))
                    r, c = r - dr, c - dc
                if len(stroke) < 2 or not used.isdisjoint(stroke):
                    break
                used.update(stroke)
                lengths.append(len(stroke))
            if len(lengths) == len(tips) and used == trails:
                interpretations.append(dict(carrier=carrier, tip=tip, trail=trail,
                                            direction=(dr, dc), lengths=tuple(lengths)))
    return interpretations


def _read_input(a):
    """Recover roles, legends and closed free-space regions from input alone."""
    a = _grid(a)
    bg = int(a[0, 0])
    _require(np.all(a[0] == bg) and np.all(a[-1] == bg)
             and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg),
             "input needs a uniform background frame")
    key_mask = np.zeros(a.shape, bool)
    keys = []
    for cells in _components(a != bg):
        rows, cols = zip(*cells)
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if min(h, w) < 4 or len(cells) != h * w:
            continue
        interpretations = _decode_glyph(a[r0:r1 + 1, c0:c1 + 1])
        _require(len(interpretations) <= 1, "ambiguous direction legend")
        if interpretations:
            key = interpretations[0]
            key["bbox"] = r0, r1, c0, c1
            keys.append(key)
            key_mask[r0:r1 + 1, c0:c1 + 1] = True
    _require(bool(keys), "no direction legends")
    carriers = {k["carrier"] for k in keys}
    _require(len(carriers) == len(keys), "multiple keys for one carrier")
    outside = set(map(int, np.unique(a[~key_mask]))) - {bg}
    marker_colors = outside - carriers
    _require(len(marker_colors) == 1 and carriers <= outside, "unreadable marker role")
    marker = next(iter(marker_colors))
    _require(all(marker not in (k["tip"], k["trail"]) for k in keys), "marker reused in a key")
    _require(len({k["tip"] for k in keys}) == 1, "tips must share one role")
    directions = {k["carrier"]: k["direction"] for k in keys}
    regions = []
    for cells in _components((a == bg) | (a == marker)):
        starts = [p for p in cells if int(a[p]) == marker]
        boundary, edge = set(), False
        for r, c in cells:
            edge |= r == 0 or c == 0 or r == a.shape[0] - 1 or c == a.shape[1] - 1
            for dr, dc in CARDINAL:
                nr, nc = r + dr, c + dc
                if 0 <= nr < a.shape[0] and 0 <= nc < a.shape[1] and int(a[nr, nc]) not in (bg, marker):
                    boundary.add(int(a[nr, nc]))
        if starts:
            _require(not edge and len(boundary) == 1 and boundary <= carriers,
                     "marker must have one enclosing wall color")
        if not edge and len(boundary) == 1 and boundary <= carriers:
            color = next(iter(boundary))
            regions.append(dict(free=set(cells), starts=starts, carrier=color,
                                direction=directions[color]))
    _require(sum(len(r["starts"]) for r in regions) == int(np.count_nonzero(a == marker)),
             "unassociated marker")
    _require(all(any(r["carrier"] == c and r["starts"] for r in regions) for c in carriers),
             "each keyed system needs a marker")
    return dict(background=bg, marker=marker, keys=keys, key_mask=key_mask, regions=regions)


def _ray(free, start, direction, corner=True):
    dr, dc = direction
    r, c = start
    path = [(r, c)]
    while (r + dr, c + dc) in free:
        if corner and dr and dc and ((r + dr, c) not in free or (r, c + dc) not in free):
            break
        r, c = r + dr, c + dc
        path.append((r, c))
    return path


def _settle_read(a, info, corner=True, reverse=False, check=True):
    b = a.copy()
    b[a == info["marker"]] = info["background"]
    paths, witnesses, used = [], [], set()
    for region in info["regions"]:
        dr, dc = region["direction"]
        if reverse:
            dr, dc = -dr, -dc
        for start in region["starts"]:
            path = _ray(region["free"], start, (dr, dc), corner)
            if check:
                _require(used.isdisjoint(path), "interacting trajectories")
            used.update(path)
            b[path[-1]] = info["marker"]
            r, c = path[-1]
            witnesses.append(bool(dr and dc and (r + dr, c + dc) in region["free"]
                                  and ((r + dr, c) not in region["free"]
                                       or (r, c + dc) not in region["free"])))
            paths.append(path)
    return b, paths, witnesses


def _columns(a):
    return [tuple(map(int, a[:, c])) for c in range(a.shape[1])]


def _retained(a):
    seen, indices = set(), []
    for c, column in enumerate(_columns(a)):
        if column not in seen:
            seen.add(column)
            indices.append(c)
    return indices


def _compact(a):
    return a[:, _retained(a)].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    info = _read_input(a)
    settled, _, _ = _settle_read(a, info)
    return _compact(settled)


def _make_key(rng, direction, carrier, trail, tip):
    dr, dc = direction
    count = int(rng.choice([2, 2, 3]))
    length = int(rng.choice([2, 2, 3]))
    points = []
    for i in range(count):
        offset = int(rng.integers(0, 2))
        base_r, base_c = (0, 2 * i) if dr else (2 * i, 0)
        for j in range(length + 1):
            points.append((base_r + (j + offset) * dr, base_c + (j + offset) * dc,
                           tip if j == length else trail))
    r0, r1 = min(p[0] for p in points), max(p[0] for p in points)
    c0, c1 = min(p[1] for p in points), max(p[1] for p in points)
    pads = [int(rng.choice([1, 1, 2])) for _ in range(4)]
    key = np.full((r1 - r0 + 1 + pads[0] + pads[1],
                   c1 - c0 + 1 + pads[2] + pads[3]), carrier, np.int8)
    for r, c, color in points:
        key[r - r0 + pads[0], c - c0 + pads[2]] = color
    return key


def _make_wall(rng, h, w, direction, harder, need_corner):
    # Broad horizontal plateaus produce genuine repeated full-height profiles.
    footprint = np.ones((h, w), bool)
    kind = int(rng.integers(0, 4))
    if kind == 1:
        for corner in rng.choice(4, int(rng.integers(1, 4)), replace=False):
            dh, dw = int(rng.integers(1, 3)), int(rng.integers(1, 4))
            rs = slice(0, dh) if corner < 2 else slice(h - dh, h)
            cs = slice(0, dw) if corner % 2 == 0 else slice(w - dw, w)
            footprint[rs, cs] = False
    elif kind == 2:
        for r in range(h):
            inset = min(2, r, h - 1 - r)
            footprint[r, :2 - inset] = False
            footprint[r, w - (2 - inset):] = False
    elif kind == 3:
        # Inset bays of independently sampled lengths, leaving a broad trunk.
        length = int(rng.integers(2, max(3, w // 3)))
        start = int(rng.integers(1, w - length))
        if rng.integers(2):
            footprint[:2, start:start + length] = False
        else:
            footprint[-2:, start:start + length] = False
    padded = np.pad(footprint, 1)
    free = footprint.copy()
    for dr, dc in CARDINAL:
        free &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    wall = footprint & ~free
    for _ in range(int(rng.integers(0, 4 if harder else 2))):
        if rng.integers(2):
            r = int(rng.integers(2, h - 2))
            cut = w if rng.random() < .35 else int(rng.integers(3, w - 1))
            wall[r, :cut] |= footprint[r, :cut]
        else:
            c = int(rng.integers(3, w - 3))
            cut = h if rng.random() < .2 else int(rng.integers(2, h - 1))
            wall[:cut, c] |= footprint[:cut, c]
    if need_corner:
        # A wall tip has a background diagonal beyond it but a blocked side.
        wall[:h // 2 + 1, w // 2] |= footprint[:h // 2 + 1, w // 2]
        if direction[0] < 0:
            wall, footprint = np.flipud(wall).copy(), np.flipud(footprint).copy()
        if direction[1] < 0:
            wall, footprint = np.fliplr(wall).copy(), np.fliplr(footprint).copy()
    return wall, footprint & ~wall


def _choose_paths(rng, free_mask, direction, desired, need_corner):
    candidates = []
    for cells in _components(free_mask):
        free = set(cells)
        for start in sorted(free):
            path = _ray(free, start, direction)
            dr, dc = direction
            r, c = path[-1]
            witness = bool(dr and dc and (r + dr, c + dc) in free
                           and ((r + dr, c) not in free or (r, c + dc) not in free))
            if len(path) >= 2:
                candidates.append((path, witness))
    witnesses = [i for i, (_, corner) in enumerate(candidates) if corner]
    if not candidates or (need_corner and not witnesses):
        return None
    selected, used = [], set()
    order = ([int(rng.choice(witnesses))] if need_corner else [])
    order += list(map(int, rng.permutation(len(candidates))))
    for i in order:
        path = candidates[i][0]
        if used.isdisjoint(path):
            selected.append(path)
            used.update(path)
        if len(selected) >= desired:
            break
    return selected if len(selected) >= 2 else None


def _scene(rng, directions, marker, tip, harder, need_corner, swap_keys):
    n = len(directions)
    palette = [c for c in range(10) if c not in (marker, tip)]
    bg = int(rng.choice(palette))
    carriers = list(map(int, rng.choice([c for c in palette if c != bg], n, replace=False)))
    trail_pool = [c for c in palette if c != bg and c not in carriers]
    shared_trail = int(rng.choice(trail_pool))
    keys = []
    for i, direction in enumerate(directions):
        cross = [c for c in carriers if c != carriers[i]]
        trail = int(rng.choice(cross)) if cross and rng.random() < .3 else shared_trail
        keys.append(_make_key(rng, direction, carriers[i], trail, tip))
    side = n == 2 or rng.random() < .55
    placements, paths = [], []
    if side:
        width = int(rng.integers(24 if n == 2 else 21, 31))
        row = 1
        if swap_keys is None:
            swap_keys = bool(rng.integers(2))
        key_order = list(reversed(range(n))) if swap_keys else list(range(n))
        for i, direction in enumerate(directions):
            key = keys[key_order[i]]
            kh, kw = key.shape
            eh = int(rng.integers(8, 12 if n == 2 else 18))
            ew_max = width - kw - 3
            if ew_max < 10:
                return None
            ew = int(rng.integers(max(10, ew_max - 3), ew_max + 1))
            band = max(kh, eh)
            left_key = bool(rng.integers(2))
            kc = 1 if left_key else width - kw - 1
            ec = kw + 2 if left_key else 1
            kr, er = row + int(rng.integers(band - kh + 1)), row + int(rng.integers(band - eh + 1))
            placements.append((i, key, kr, kc, er, ec, eh, ew))
            row += band + int(rng.integers(1, 3))
        height = row
    else:
        key = keys[0]
        kh, kw = key.shape
        eh = int(rng.integers(8, min(18, 28 - kh)))
        ew = int(rng.integers(14, 27))
        width = max(ew, kw) + 2 + int(rng.integers(0, 3))
        height = eh + kh + 3 + int(rng.integers(0, 2))
        ec = int(rng.integers(1, width - ew))
        kc = int(rng.integers(1, width - kw))
        if rng.integers(2):
            kr, er = 1, kh + 2
        else:
            er, kr = 1, eh + 2
        placements.append((0, key, kr, kc, er, ec, eh, ew))
    if max(height, width) > 30:
        return None
    a = np.full((height, width), bg, np.int8)
    for i, key, kr, kc, er, ec, eh, ew in placements:
        a[kr:kr + key.shape[0], kc:kc + key.shape[1]] = key
        wall, free = _make_wall(rng, eh, ew, directions[i], harder, need_corner == i)
        desired = int(rng.integers(2, 6 if harder else 5))
        selected = _choose_paths(rng, free, directions[i], desired, need_corner == i)
        if selected is None:
            return None
        a[er:er + eh, ec:ec + ew][wall] = carriers[i]
        for path in selected:
            path = [(r + er, c + ec) for r, c in path]
            paths.append(path)
            a[path[0]] = marker
    # A construction-side label; the parser/solver is checked against this.
    settled = a.copy()
    settled[a == marker] = bg
    for path in paths:
        settled[path[-1]] = marker
    return a, settled


def _evidence(a, settled):
    before, after = _columns(a), _columns(settled)
    retained = _retained(settled)
    out = settled[:, retained]
    new_pairs = [(i, j) for i in range(len(after)) for j in range(i + 1, len(after))
                 if after[i] == after[j] and before[i] != before[j]]
    new_merge = bool(new_pairs)
    before_retained = set(_retained(a))
    # The same newly merged pair must straddle a different vector; its later
    # column must also survive input deduplication and disappear after motion.
    # This cannot be witnessed solely by the unchanged background margins.
    causal_separated = any(j in before_retained and j not in retained
                           and any(after[k] != after[i] for k in range(i + 1, j))
                           for i, j in new_pairs)
    separated = any(after[i] == after[j] and any(after[k] != after[i] for k in range(i + 1, j))
                    for i in range(len(after)) for j in range(i + 2, len(after)))
    crop_fails = not np.array_equal(settled[:, :len(retained)], out)
    run_indices = [i for i, c in enumerate(after) if i == 0 or c != after[i - 1]]
    last = [i for i, c in enumerate(after) if c not in after[i + 1:]]
    return dict(new_merge=new_merge, causal_separated=causal_separated,
                separated=separated, crop_fails=crop_fails,
                run_fails=not np.array_equal(settled[:, run_indices], out),
                last_fails=not np.array_equal(settled[:, last], out),
                omit_motion=not np.array_equal(_compact(a), out),
                omit_dedup=len(retained) < a.shape[1], retained=retained)


def _new_pair(rng, directions, marker, tip, harder, need_corner, seen, swap_keys=None):
    for _ in range(160):
        candidate = _scene(rng, directions, marker, tip, harder, need_corner, swap_keys)
        if candidate is None:
            continue
        a, settled = candidate
        if _signature(a) in seen:
            continue
        evidence = _evidence(a, settled)
        if not all(evidence[k] for k in ("new_merge", "causal_separated", "separated", "crop_fails", "run_fails",
                                         "last_fails", "omit_motion", "omit_dedup")):
            continue
        info = _read_input(a)
        predicted, _, corners = _settle_read(a, info)
        _require(np.array_equal(predicted, settled), "construction and visible ray disagree")
        reversed_grid, _, _ = _settle_read(a, info, reverse=True, check=False)
        if np.array_equal(_compact(reversed_grid), _compact(settled)):
            continue
        if need_corner is not None:
            naive, _, _ = _settle_read(a, info, corner=False, check=False)
            if not any(corners) or np.array_equal(_compact(naive), _compact(settled)):
                continue
        seen.add(_signature(a))
        return {"input": a, "output": _compact(settled)}
    raise RuntimeError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    marker, tip = map(int, rng.choice(10, 2, replace=False))
    horizontal = (0, int(rng.choice([-1, 1])))
    diagonal = DIAGONAL[int(rng.integers(4))]
    # Two-system color association and diagonal corner blocking both occur even
    # with the smallest legal training count. Their presentation order varies.
    plans = [([horizontal, (0, -horizontal[1])], None, True),
             ([diagonal, CARDINAL[int(rng.integers(4))]], 0, False)]
    for _ in range(nt - 2):
        count = int(rng.choice([1, 1, 2]))
        ds = [DIRECTIONS[int(rng.integers(8))] for _ in range(count)]
        if count == 2:
            ds[int(rng.integers(2))] = CARDINAL[int(rng.integers(4))]
        plans.append((ds, None, None))
    rng.shuffle(plans)
    puzzle, seen = {"train": [], "test": []}, set()
    for directions, corner, swap in plans:
        puzzle["train"].append(_new_pair(rng, directions, marker, tip, False, corner, seen, swap))
    for _ in range(nq):
        # A novel combination: all keys in this query encode diagonal motion;
        # training's mandatory two-system scene instead has opposed cardinals.
        count = 2
        directions = [DIAGONAL[int(rng.integers(4))] for _ in range(count)]
        puzzle["test"].append(_new_pair(rng, directions, marker, tip, True, 0, seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, roles, training = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_signature(a) not in seen, "duplicate input")
            seen.add(_signature(a))
            info = _read_input(a)
            settled, paths, corners = _settle_read(a, info)
            roles.add((info["marker"], info["keys"][0]["tip"]))
            _require(np.array_equal(out, _compact(settled)), "incorrect composed output")
            _require(out.shape[0] == a.shape[0] and out.shape[1] < a.shape[1], "no horizontal compaction")
            _require(len(set(_columns(out))) == out.shape[1], "duplicate output columns")
            fixed = (a != info["marker"]) & (a != info["background"])
            _require(np.array_equal(a[fixed], settled[fixed]), "motion changed scaffold")
            _require(np.count_nonzero(a == info["marker"]) == np.count_nonzero(settled == info["marker"]),
                     "motion lost a marker")
            _require(all(len(p) >= 2 for p in paths), "generated marker did not move")
            evidence = _evidence(a, settled)
            _require(all(evidence[k] for k in ("new_merge", "causal_separated", "separated", "crop_fails", "run_fails",
                                              "last_fails", "omit_motion", "omit_dedup")),
                     "missing causal or shortcut witness")
            reversed_grid, _, _ = _settle_read(a, info, reverse=True, check=False)
            _require(not np.array_equal(_compact(reversed_grid), out), "tip polarity is unobservable")
            # Independent direct wall-cell stop check, without region membership.
            for region in info["regions"]:
                dr, dc = region["direction"]
                for start in region["starts"]:
                    r, c = start
                    while True:
                        nr, nc = r + dr, c + dc
                        _require(0 <= nr < a.shape[0] and 0 <= nc < a.shape[1], "ray escaped")
                        blocked = int(a[nr, nc]) == region["carrier"]
                        if dr and dc:
                            blocked |= int(a[nr, c]) == region["carrier"] or int(a[r, nc]) == region["carrier"]
                        if blocked:
                            break
                        _require(int(a[nr, nc]) == info["background"], "ray crosses another object")
                        r, c = nr, nc
                    _require(int(settled[r, c]) == info["marker"], "incorrect wall stop")
            if split == "train":
                naive, _, _ = _settle_read(a, info, corner=False, check=False)
                training.append((info, any(corners) and not np.array_equal(_compact(naive), out)))
            else:
                _require(any(corners), "test needs an actual corner challenge")
                _require(len(info["keys"]) == 2 and all(k["direction"] in DIAGONAL for k in info["keys"]),
                         "test must combine two diagonal systems")
    _require(len(roles) == 1, "puzzle marker/tip conventions changed")
    _require(any(corner for _, corner in training), "no consequential diagonal corner demonstration")
    _require(any(len(info["keys"]) == 2 for info, _ in training), "no color-association demonstration")
    association_orders = set()
    for info, _ in training:
        if len(info["keys"]) != 2:
            continue
        ordered = sorted(info["keys"], key=lambda k: k["bbox"][0])
        upper_key = ordered[0]["carrier"]
        upper_region = min((r for r in info["regions"] if r["starts"]),
                           key=lambda r: min(p[0] for p in r["free"]))
        association_orders.add(upper_key == upper_region["carrier"])
    _require(association_orders == {False, True}, "training must show aligned and exchanged key order")
    train_dirs = {k["direction"] for info, _ in training for k in info["keys"]}
    _require((0, -1) in train_dirs and (0, 1) in train_dirs and any(d in DIAGONAL for d in train_dirs),
             "training lacks direction/polarity evidence")
    _require(all(len(info["keys"]) == 1 or any(k["direction"] in CARDINAL for k in info["keys"])
                 for info, _ in training), "training must leave the two-diagonal combination held out")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; retain only bounded counters and hashes, no traces."""
    fingerprints, shapes, outputs, palettes, dirs, marker_counts, lengths = set(), set(), set(), set(), set(), set(), set()
    structures, key_shapes, backgrounds, retained_widths = set(), set(), set(), set()
    counts, irregular, multichamber, corner_pairs, pairs = set(), 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        if seed in (0, 17, 41):
            _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterminism")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                info = _read_input(a)
                settled, paths, corners = _settle_read(a, info)
                saved_input = a.copy()
                _require(np.array_equal(solve(a), out), "input-only solve mismatch")
                _require(np.array_equal(a, saved_input), "solve mutated its input")
                # Independently rebuild A's stage from vector tuples.
                vectors = list(dict.fromkeys(zip(*settled.tolist())))
                _require(np.array_equal(np.array(vectors).T, out), "vector reconstruction mismatch")
                pairs += 1
                shapes.add(a.shape)
                outputs.add(out.shape)
                palettes.add(tuple(map(int, np.unique(a))))
                backgrounds.add(info["background"])
                retained_widths.add(out.shape[1])
                counts.add(len(info["keys"]))
                dirs.update(k["direction"] for k in info["keys"])
                marker_counts.add(len(paths))
                lengths.update(len(p) - 1 for p in paths)
                corner_pairs += any(corners)
                multichamber += len(info["regions"]) > len(info["keys"])
                for key in info["keys"]:
                    r0, r1, c0, c1 = key["bbox"]
                    key_shapes.add((r1 - r0 + 1, c1 - c0 + 1))
                    mask = (a == key["carrier"]) & ~info["key_mask"]
                    rr, cc = np.where(mask)
                    sub = mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
                    irregular += not (sub[0].all() and sub[-1].all() and sub[:, 0].all() and sub[:, -1].all())
                mapping = {}
                normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in a.flat)
                structures.add(hashlib.sha256(bytes(a.shape) + normalized).hexdigest())
    _require(len(fingerprints) == 42 and len(structures) >= 180, "insufficient structural diversity")
    _require(len(shapes) >= 35 and len(outputs) >= 35, "insufficient dimensions")
    _require(dirs == set(DIRECTIONS) and counts == {1, 2}, "direction/system coverage")
    _require(len(key_shapes) >= 12 and len(marker_counts) >= 5 and {1, 2, 3} <= lengths, "geometry/count coverage")
    _require(len(palettes) >= 20 and len(backgrounds) >= 8 and len(retained_widths) >= 8, "palette/compaction coverage")
    _require(irregular >= 80 and multichamber >= 15 and corner_pairs >= 84, "wall/corner coverage")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(900 + nt * 10 + nq, nt, nq)
            _require((len(p["train"]), len(p["test"])) == (nt, nq), "explicit count mismatch")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    p["test"][0]["output"] = _compact(p["test"][0]["input"])
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("omitting motion passed validation")
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(fingerprints),
                normalized_inputs=len(structures), input_shapes=len(shapes), output_shapes=len(outputs),
                palettes=len(palettes), backgrounds=len(backgrounds), key_shapes=len(key_shapes),
                marker_counts=sorted(marker_counts), move_lengths=sorted(lengths),
                retained_widths=sorted(retained_widths), irregular_systems=irregular,
                multichamber_pairs=multichamber, corner_pairs=corner_pairs)
