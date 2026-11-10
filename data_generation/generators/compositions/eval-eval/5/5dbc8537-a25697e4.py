"""Shared-cavity lockwork: rigid two-color keys collectively exact-cover a plate.

Only NumPy and the standard library are used. Search reads the input, including
the source inventory, and deduplicates complete colored images rather than moves.
"""

from collections import deque
import json
import numpy as np

DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
INVERSE = (0, 3, 2, 1, 4, 5, 6, 7)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer 2D arrays, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIR4:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(frozenset(part))
    return result


def _cells(mask):
    return frozenset(map(tuple, np.argwhere(mask).tolist()))


def _bbox(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr), min(cc), max(cc)


def _shift(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _transform(regions, op):
    def point(r, c):
        if op >= 4:
            c = -c
        for _ in range(op % 4):
            r, c = c, -r
        return r, c
    raw = [frozenset(point(r, c) for r, c in region) for region in regions]
    r0, _, c0, _ = _bbox(set().union(*raw))
    return tuple(_shift(region, -r0, -c0) for region in raw)


def _canonical(cells):
    return min(tuple(sorted(_transform((cells,), op)[0])) for op in range(8))


def _inspect_plate(plate, tray, side):
    pc, tc = set(map(int, np.unique(plate))), set(map(int, np.unique(tray)))
    if len(pc) != 3 or len(tc) != 3 or len(pc & tc) != 1:
        raise ValueError("plate and tray must share only their vacancy/background role")
    vacancy = next(iter(pc & tc))
    wall = int(plate[0, 0])
    border = np.concatenate((plate[0], plate[-1], plate[:, 0], plate[:, -1]))
    if wall == vacancy or not np.all(border == wall):
        raise ValueError("plate needs a complete monochrome rim")
    marker = next(iter(pc - {wall, vacancy}))
    payload = tuple(sorted(tc - {vacancy}))
    fragments = _components(_cells(plate == marker))
    if not fragments or len(fragments) % 2:
        raise ValueError("locks need pairs of marker fragments")
    keys = []
    for comp in _components(_cells(tray != vacancy)):
        regions = tuple(frozenset(p for p in comp if int(tray[p]) == color)
                        for color in payload)
        if any(not region or len(_components(region)) != 1 for region in regions):
            raise ValueError("each source key needs two connected color regions")
        keys.append(regions)
    if len(keys) < len(fragments) // 2:
        raise ValueError("too few keys")
    locks = []
    all_marker = set().union(*fragments)
    for i, first in enumerate(fragments):
        for j in range(i + 1, len(fragments)):
            union = first | fragments[j]
            r0, r1, c0, c1 = _bbox(union)
            rect = frozenset((r, c) for r in range(r0, r1 + 1)
                             for c in range(c0, c1 + 1))
            if rect & all_marker != union or any(int(plate[p]) == wall for p in rect):
                continue
            slot = rect - union
            if slot and len(_components(slot)) == 1:
                locks.append(dict(parts=(i, j), rectangle=rect, slot=slot))
    if not locks:
        raise ValueError("no locks")
    return dict(plate=plate.copy(), tray=tray.copy(), side=side, wall=wall,
                vacancy=vacancy, marker=marker, payload=payload, keys=keys,
                fragments=fragments, locks=locks, holes=_cells(plate == vacancy))


def _panel_analyses(inp):
    for axis in (0, 1):
        for cut in range(3, inp.shape[axis] - 2):
            a, b = (inp[:cut], inp[cut:]) if axis == 0 else (inp[:, :cut], inp[:, cut:])
            for plate, tray, side in ((a, b, "top" if axis == 0 else "left"),
                                      (b, a, "bottom" if axis == 0 else "right")):
                try:
                    yield _inspect_plate(plate, tray, side)
                except ValueError:
                    continue


def _enumerate(analysis, policy=(0, 1), exact=True, expected=None, limit=2, max_nodes=None):
    """Inventory-constrained lock matching AND full-cavity exact covering.

    policy indexes determinant parity (rotation, reflection). exact=False is a
    diagnostic ablation: every lock is fitted, but recesses may be left empty.
    expected is used only to test training hypotheses, never by public solve.
    """
    plate, holes = analysis["plate"], analysis["holes"]
    coords = sorted(holes)
    index = {p: i for i, p in enumerate(coords)}
    full_mask = (1 << len(coords)) - 1
    colors = analysis["payload"]
    candidates = []
    by_fragment = [[] for _ in analysis["fragments"]]
    for li, lock in enumerate(analysis["locks"]):
        sr, _, sc, _ = _bbox(lock["slot"])
        seen = set()
        for ki, regions in enumerate(analysis["keys"]):
            for op in range(8):
                transformed = _transform(regions, op)
                for pi in (0, 1):
                    pr, _, pc, _ = _bbox(transformed[pi])
                    absolute = tuple(_shift(region, sr - pr, sc - pc) for region in transformed)
                    if absolute[pi] != lock["slot"]:
                        continue
                    other = absolute[1 - pi]
                    footprint = absolute[0] | absolute[1]
                    if other & lock["rectangle"] or not footprint <= holes:
                        continue
                    placed_colors = colors[::-1] if policy[op // 4] else colors
                    assignments = tuple(sorted((p, placed_colors[k])
                                                for k, region in enumerate(absolute) for p in region))
                    if expected is not None and any(int(expected[p]) != color for p, color in assignments):
                        continue
                    sig = ki, assignments
                    if sig in seen:
                        continue
                    seen.add(sig)
                    mask = sum(1 << index[p] for p in footprint)
                    ci = len(candidates)
                    candidates.append(dict(key=ki, lock=li, parts=lock["parts"],
                                           footprint=footprint, mask=mask, assignments=assignments,
                                           op=op, plug=pi, regions=absolute))
                    for fi in lock["parts"]:
                        by_fragment[fi].append(ci)
    outcomes, nodes = {}, 0
    all_parts = (1 << len(analysis["fragments"])) - 1

    def search(parts, used_keys, covered, rectangles, path):
        nonlocal nodes
        nodes += 1
        if max_nodes is not None and nodes > max_nodes:
            raise ValueError("exact-cover search budget exceeded; interpretation unresolved")
        if parts == all_parts:
            if exact and covered != full_mask:
                return
            out = plate.copy()
            for ci in path:
                for p, color in candidates[ci]["assignments"]:
                    out[p] = color
            if expected is not None and not np.array_equal(out, expected):
                return
            outcomes.setdefault(_signature(out), (out, tuple(candidates[ci] for ci in path)))
            return
        best = None
        for fi in range(len(by_fragment)):
            if parts & (1 << fi):
                continue
            options = []
            for ci in by_fragment[fi]:
                move = candidates[ci]
                pm = sum(1 << p for p in move["parts"])
                rect = analysis["locks"][move["lock"]]["rectangle"]
                if (pm & parts or used_keys & (1 << move["key"]) or covered & move["mask"]
                        or any(rect & old for old in rectangles)):
                    continue
                options.append(ci)
            if not options:
                return
            if best is None or len(options) < len(best):
                best = options
        for ci in best:
            move = candidates[ci]
            pm = sum(1 << p for p in move["parts"])
            rect = analysis["locks"][move["lock"]]["rectangle"]
            search(parts | pm, used_keys | (1 << move["key"]), covered | move["mask"],
                   rectangles + (rect,), path + (ci,))
            if len(outcomes) >= limit:
                return

    search(0, 0, 0, (), ())
    return list(outcomes.values()), candidates, nodes


def _analyze(inp, max_nodes=None):
    interpretations = list(_panel_analyses(_grid(inp)))
    if len(interpretations) != 1:
        raise ValueError("input must identify one framed plate and source tray")
    analysis = interpretations[0]
    outcomes, candidates, nodes = _enumerate(analysis, max_nodes=max_nodes)
    if len(outcomes) != 1:
        raise ValueError("plate must have exactly one complete colored answer")
    out, moves = outcomes[0]
    analysis.update(output=out, moves=moves, candidates=candidates, nodes=nodes)
    return analysis


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete plate from its visible holes, locks, and key inventory."""
    return _analyze(input_grid)["output"].copy()


def _slot(rng):
    h, w = int(rng.integers(3, 7)), int(rng.integers(2, 7))
    y = int(rng.integers(1, h - 1))
    slot = {(y, 0)}
    straight = rng.random() < 0.35
    for c in range(1, w):
        new = y if straight else int(np.clip(y + rng.integers(-1, 2), 1, h - 2))
        slot.update((r, c) for r in range(min(y, new), max(y, new) + 1))
        y = new
    rectangle = {(r, c) for r in range(h) for c in range(w)}
    # Optional thickening preserves only genuine two-fragment locks.
    for _ in range(int(rng.integers(0, 4))):
        frontier = sorted({(r + dr, c + dc) for r, c in slot for dr, dc in DIR4}
                          & rectangle - slot)
        if frontier:
            trial = slot | {frontier[int(rng.integers(len(frontier)))]}
            target = rectangle - trial
            if len(_components(target)) == 2 and _bbox(target) == (0, h - 1, 0, w - 1):
                slot = trial
    return _transform((rectangle - slot, slot), int(rng.integers(8)))


def _path(start, goals, allowed, rng):
    queue, previous = deque([start]), {start: None}
    directions = [DIR4[int(i)] for i in rng.permutation(4)]
    end = None
    while queue:
        p = queue.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in directions:
            q = p[0] + dr, p[1] + dc
            if q in allowed and q not in previous:
                previous[q] = p
                queue.append(q)
    if end is None:
        return None
    result = []
    while end is not None:
        result.append(end)
        end = previous[end]
    return result[::-1]


def _grow(region, allowed, size, rng, max_span=10):
    region = set(region)
    while len(region) < size:
        frontier = sorted({(r + dr, c + dc) for r, c in region for dr, dc in DIR4}
                          & allowed - region)
        legal = []
        for p in frontier:
            r0, r1, c0, c1 = _bbox(region | {p})
            if max(r1 - r0 + 1, c1 - c0 + 1) <= max_span:
                legal.append(p)
        if not legal:
            break
        region.add(legal[int(rng.integers(len(legal)))])
    return frozenset(region)


def _scatter(keys, height, max_width, vacancy, colors, rng):
    min_width = max(_bbox(set().union(*key))[3] + 1 for key in keys)
    if min_width > max_width:
        return None
    for _ in range(16):
        width = int(rng.integers(min_width, max_width + 1))
        tray = np.full((height, width), vacancy, dtype=np.int8)
        blocked = set()
        success = True
        order = sorted(range(len(keys)), key=lambda k: -sum(map(len, keys[k])) + rng.random())
        for ki in order:
            regions = keys[ki]
            shape = set().union(*regions)
            _, rh, _, cw = _bbox(shape)
            options = [(r, c) for r in range(height - rh) for c in range(width - cw)]
            rng.shuffle(options)
            for r, c in options:
                shifted = _shift(shape, r, c)
                if shifted & blocked:
                    continue
                for region, color in zip(regions, colors):
                    for p in _shift(region, r, c):
                        tray[p] = color
                blocked.update(shifted)
                blocked.update((y + dr, x + dc) for y, x in shifted for dr, dc in DIR4)
                break
            else:
                success = False
                break
        if success:
            return tray
    return None


def _join_panels(plate, tray, out, side):
    if side in ("left", "top"):
        inp = np.concatenate((plate, tray), axis=1)
    else:
        inp = np.concatenate((tray, plate), axis=1)
    if side in ("top", "bottom"):
        inp, out = inp.T, out.T
    return {"input": inp.copy(), "output": out.copy()}


def _build_once(rng, count, side, roles, parities, decoy):
    wall, vacancy, marker, ca, cb = roles
    height, width = int(rng.integers(16, 27)), int(rng.integers(10, 16))
    interior = {(r, c) for r in range(1, height - 1) for c in range(1, width - 1)}
    rectangles, targets, slots = [], [], []
    reserved = set()
    for _ in range(count):
        target, slot = _slot(rng)
        rect = target | slot
        _, rh, _, cw = _bbox(rect)
        positions = [(r, c) for r in range(1, height - rh - 1)
                     for c in range(1, width - cw - 1)]
        rng.shuffle(positions)
        for r, c in positions:
            placed = _shift(rect, r, c)
            if placed & reserved:
                continue
            rectangles.append(placed)
            targets.append(_shift(target, r, c))
            slots.append(_shift(slot, r, c))
            reserved.update(placed)
            reserved.update((y + dr, x + dc) for y, x in placed for dr, dc in DIR4)
            break
        else:
            return None
    free = interior - set().union(*rectangles)
    handles, occupied = [], set()
    for i, slot in enumerate(slots):
        starts = sorted({(r + dr, c + dc) for r, c in slot for dr, dc in DIR4} & free - occupied)
        rng.shuffle(starts)
        region = None
        for start in starts:
            if i == 0:
                region = frozenset({start})
                break
            path = _path(start, occupied, free, rng)
            if path is not None and 2 <= len(path) <= 12:
                region = frozenset(path[:-1])
                break
        if region is None:
            return None
        desired = max(len(region), int(rng.integers(3, 15)))
        region = _grow(region, free - occupied, desired, rng)
        if len(region) < 3:
            return None
        handles.append(region)
        occupied.update(region)
    plate = np.full((height, width), wall, dtype=np.int8)
    out = plate.copy()
    keys, records = [], []
    source_colors = (ca, cb)
    for i, (target, slot, handle, parity) in enumerate(zip(targets, slots, handles, parities)):
        for p in target:
            plate[p] = out[p] = marker
        # Which literal key color is the plug changes between objects.
        regions = (slot, handle) if rng.random() < 0.5 else (handle, slot)
        op = int(rng.integers(4)) + 4 * parity
        source = _transform(regions, INVERSE[op])
        keys.append(source)
        destination_colors = source_colors[::-1] if parity else source_colors
        for region, color in zip(regions, destination_colors):
            for p in region:
                plate[p], out[p] = vacancy, color
        records.append((regions, op))
    if decoy:
        alternatives = []
        for i, (slot, handle) in enumerate(zip(slots, handles)):
            starts = sorted(p for p in handle if any((p[0] + dr, p[1] + dc) in slot for dr, dc in DIR4))
            if decoy == 1:
                short = _grow({starts[0]}, set(handle), max(1, len(handle) // 2), rng)
                alternatives.append((i, short))
                break
            # An equal-area key takes cells from a neighbor's handle. Its plug
            # and total area both match, but the joint packing must reject it.
            for _ in range(28):
                start = starts[int(rng.integers(len(starts)))]
                trial = _grow({start}, occupied, len(handle), rng)
                if len(trial) == len(handle) and trial != handle and trial - handle:
                    alternatives.append((i, trial))
                    break
        if not alternatives:
            return None
        i, alternative = alternatives[int(rng.integers(len(alternatives)))]
        regions, op = records[i]
        reduced = (slots[i], alternative) if regions[0] == slots[i] else (alternative, slots[i])
        keys.append(_transform(reduced, INVERSE[op]))
    tray = _scatter(keys, height, 30 - width, vacancy, source_colors, rng)
    if tray is None:
        return None
    return _join_panels(plate, tray, out, side)


def _build_pair(rng, count, side, roles, parities, decoy):
    for _ in range(240):
        pair = _build_once(rng, count, side, roles, parities, decoy)
        if pair is None:
            continue
        try:
            a = _analyze(pair["input"], max_nodes=12000)
        except ValueError:
            continue
        if not np.array_equal(a["output"], pair["output"]):
            continue
        if decoy and not _features(a)["local_decoys"]:
            continue
        if decoy == 2 and not _features(a)["equal_area_decoys"]:
            continue
        return pair
    raise RuntimeError("bounded construction could not find an unambiguous shared cavity")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("counts require 2..10 train, 1..4 test, train > test")
    rng = np.random.default_rng(seed)
    sides = [str(s) for s in rng.permutation(["left", "right", "top", "bottom"])]
    regime = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(num_train + num_test):
        training = i < num_train
        count = 2 if i < 2 else int(rng.integers(1, 3)) if training else 3
        if regime == 0:
            side = sides[i % 2] if training else sides[2 + (i - num_train) % 2]
        else:
            side = sides[int(rng.integers(4))]
        parities = ([i] * count if i < 2 else [int(x) for x in rng.integers(2, size=count)])
        decoy = 2 if i == 0 else 1 if i == 1 else int(rng.choice([0, 1, 2], p=[0.5, 0.25, 0.25]))
        if count == 1 and decoy == 2:
            decoy = 1  # A stealing handle requires a neighboring assembly.
        roles = tuple(int(c) for c in rng.choice(10, 5, replace=False))
        for _ in range(12):
            pair = _build_pair(rng, count, side, roles, parities, decoy)
            key = _signature(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle["train" if training else "test"].append(pair)
                break
        else:
            raise RuntimeError("could not make distinct inputs")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a):
    used = {move["key"] for move in a["moves"]}
    unused = set(range(len(a["keys"]))) - used
    hole_components = _components(a["holes"])
    max_shared = max(sum(bool(comp & m["footprint"]) for m in a["moves"])
                     for comp in hole_components)
    return dict(side=a["side"], shape=a["plate"].shape,
                roles=(a["wall"], a["vacancy"], a["marker"], *a["payload"]),
                count=len(a["moves"]), supplied=len(a["keys"]),
                surplus=len(unused), local_decoys=sum(any(m["key"] == k for m in a["candidates"]) for k in unused),
                equal_area_decoys=sum(any(c["key"] == k and c["lock"] == m["lock"]
                                          and len(c["footprint"]) == len(m["footprint"])
                                          and c["footprint"] != m["footprint"]
                                          and any(c["footprint"] & other["footprint"]
                                                  for other in a["moves"] if other["lock"] != m["lock"])
                                          for c in a["candidates"] for m in a["moves"]) for k in unused),
                shared=max_shared, components=len(hole_components), nodes=a["nodes"],
                operations=tuple(m["op"] for m in a["moves"]),
                slots=tuple(_canonical(a["locks"][m["lock"]]["slot"]) for m in a["moves"]),
                handles=tuple(_canonical(m["regions"][1 - m["plug"]]) for m in a["moves"]),
                relations=tuple(int(np.sign(len(m["regions"][m["plug"]])
                                           - len(m["regions"][1 - m["plug"]]))) for m in a["moves"]))


def _compatible_policies(pairs):
    compatible = []
    analyses = [(_analyze(p["input"]), _grid(p["output"])) for p in pairs]
    for policy in ((0, 0), (0, 1), (1, 0), (1, 1)):
        if all(_enumerate(a, policy=policy, expected=out, limit=1)[0] for a, out in analyses):
            compatible.append(policy)
    return compatible


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, training_features, testing_features = set(), [], []
    exact_cover_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _signature(inp) not in seen, "duplicate input"
            seen.add(_signature(inp))
            a = _analyze(inp)
            assert np.array_equal(a["output"], out), "wrong complete colored answer"
            assert out.shape == a["plate"].shape and out.shape != inp.shape
            assert np.all(out[a["plate"] != a["vacancy"]] == a["plate"][a["plate"] != a["vacancy"]])
            assert not np.any(out == a["vacancy"]), "unfilled recess"
            assert set().union(*(m["footprint"] for m in a["moves"])) == a["holes"]
            assert sum(len(m["footprint"]) for m in a["moves"]) == len(a["holes"])
            assert len({m["key"] for m in a["moves"]}) == len(a["moves"]), "key reused"
            assert {p for m in a["moves"] for p in m["parts"]} == set(range(len(a["fragments"])))
            f = _features(a)
            if split == "train" and f["local_decoys"]:
                relaxed, _, _ = _enumerate(a, exact=False)
                exact_cover_witness |= len(relaxed) >= 2
            (training_features if split == "train" else testing_features).append(f)
    assert any(f["shared"] >= 2 for f in training_features), "teach joint packing"
    assert any(f["local_decoys"] for f in training_features), "teach locally fitting surplus"
    assert any(f["equal_area_decoys"] for f in training_features), "teach joint geometry beyond area totals"
    assert exact_cover_witness, "training must distinguish lock fitting from full-cavity covering"
    assert {op // 4 for f in training_features for op in f["operations"]} == {0, 1}
    assert _compatible_policies(puzzle["train"]) == [(0, 1)], "training fails to distinguish parity rule"


def self_test():
    """Bounded semantic, ablation, determinism, and structural-diversity checks."""
    signatures, shapes, slots, handles, roles, sides, operations, counts, relations = (set() for _ in range(9))
    shared = local_decoys = equal_area_decoys = ablation = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p["input"]) for p in puzzle["train"] + puzzle["test"]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair["input"])
                f = _features(a)
                shapes.add(f["shape"])
                slots.update(f["slots"])
                handles.update(f["handles"])
                roles.add(f["roles"])
                sides.add(f["side"])
                operations.update(f["operations"])
                counts.add(f["count"])
                relations.update(f["relations"])
                shared += f["shared"] >= 2
                local_decoys += bool(f["local_decoys"])
                equal_area_decoys += bool(f["equal_area_decoys"])
                if split == "train" and f["local_decoys"]:
                    relaxed, _, _ = _enumerate(a, exact=False)
                    ablation += len(relaxed) == 2
    assert len(signatures) == 42 and len(shapes) >= 35
    assert len(slots) >= 30 and len(handles) >= 100 and len(roles) >= 100
    assert sides == {"left", "right", "top", "bottom"}
    assert operations == set(range(8)) and counts == {1, 2, 3} and relations == {-1, 0, 1}
    assert shared >= 100 and local_decoys >= 84 and equal_area_decoys >= 42 and ablation >= 42
    first = generate_puzzle(913, 2, 1)
    second = generate_puzzle(913, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], second[s]) for k in a)
    validate_generated_puzzle({"train": list(reversed(first["train"])), "test": first["test"]})
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(71 + nt * 5 + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in first[s]] for s in first}
    damaged["test"][0]["output"][1, 1] = (int(damaged["test"][0]["output"][1, 1]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged answer accepted")
    # A compatible disconnected-cavity member exercises physical multiplicity
    # and pixel-level ties beyond the connected random construction.
    plate = np.zeros((7, 15), dtype=np.int8)
    output = plate.copy()
    for c in (2, 9):
        for r in (2, 4):
            plate[r, c:c + 3] = output[r, c:c + 3] = 2
        plate[3, c:c + 3] = 1
        output[3, c:c + 3] = 3
        for p in ((3, c + 3), (3, c + 4), (4, c + 4)):
            plate[p], output[p] = 1, 4
    tray = np.ones((7, 6), dtype=np.int8)
    for r in (0, 3):
        tray[r, :3] = 3
        tray[r, 3:5] = 4
        tray[r + 1, 4] = 4
    duplicate_input = np.concatenate((plate, tray), axis=1)
    assert np.array_equal(solve(duplicate_input), output), "identical physical copies should be interchangeable"
    missing = tray.copy()
    missing[3:5] = 1
    conflicting = tray.copy()
    conflicting[3:5] = np.where(tray[3:5] == 3, 4, np.where(tray[3:5] == 4, 3, 1))
    for damaged_tray in (missing, conflicting):
        try:
            solve(np.concatenate((plate, damaged_tray), axis=1))
        except ValueError:
            pass
        else:
            raise AssertionError("missing inventory or competing colored outputs accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), plate_shapes=len(shapes),
                slot_shapes=len(slots), handle_shapes=len(handles), shared_pairs=shared,
                local_decoy_pairs=local_decoys, equal_area_decoy_pairs=equal_area_decoys,
                exact_cover_ablation_witnesses=ablation)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
