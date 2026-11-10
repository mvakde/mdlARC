"""Mirror repair stencils: color-cardinality tickets control local reflection.

Only NumPy and the Python standard library are used. Generation constructs an
answer from a template; solve reconstructs its geometry from input pixels.
"""
from __future__ import annotations

import json
import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer arrays, dimensions 1..30, colors 0..9")
    return a


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    result = []
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue, component = [(r, c)], []
        while queue:
            y, x = queue.pop()
            component.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    queue.append((yy, xx))
        result.append(component)
    return result


def _connected(mask):
    return len(_components(mask)) == 1


def _box(component):
    rows, cols = zip(*component)
    return min(rows), max(rows) + 1, min(cols), max(cols) + 1


def _parse(input_grid):
    """Input-only geometry and role discovery; no generator metadata."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if not (np.all(a[0] == bg) and np.all(a[-1] == bg)
            and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)):
        raise ValueError("the border must be uniform background")
    components = _components(a != bg)
    bodies = [x for x in components if len(x) > 4]
    if len(bodies) != 1:
        raise ValueError("expected one connected figure")
    body = bodies[0]
    box = _box(body)
    markers = []
    for color in np.unique(a):
        if color == bg:
            continue
        pts = np.argwhere(a == color)
        if len(pts) < 5:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if r0 == r1 and c1 - c0 + 1 == len(pts):
            markers.append((int(color), "h", int(r0), int(c0), len(pts)))
        if c0 == c1 and r1 - r0 + 1 == len(pts):
            markers.append((int(color), "v", int(c0), int(r0), len(pts)))
    if len(markers) != 1:
        raise ValueError("expected one uniquely reserved straight spine")
    guide, orientation, fixed, start, length = markers[0]
    r0, r1, c0, c1 = box
    if orientation == "h":
        depth = fixed - r0
        correct_box = (r0 == fixed - depth and r1 == fixed + depth + 1
                       and c0 == start and c1 == start + length)
        negative = sum(r < fixed for r, c in body)
        positive = sum(r > fixed for r, c in body)
    else:
        depth = fixed - c0
        correct_box = (c1 == fixed + depth + 1 and r0 == start and r1 == start + length)
        negative = sum(c < fixed for r, c in body)
        positive = sum(c > fixed for r, c in body)
    if not correct_box or not (5 <= depth <= 7 and 7 <= length <= 25):
        raise ValueError("invalid figure span or depth")
    if negative == positive:
        raise ValueError("authority must have strictly greater support")
    authority = -1 if negative > positive else 1
    tickets = []
    for comp in components:
        if comp is body:
            continue
        y0, y1, x0, x1 = _box(comp)
        if len(comp) != 4 or y1 - y0 != 2 or x1 - x0 != 2:
            raise ValueError("every other component must be a full 2x2 ticket")
        if orientation == "h":
            lane = x0
            if y1 <= fixed - depth - 1:
                side, gap = -1, fixed - depth - y1
            elif y0 >= fixed + depth + 2:
                side, gap = 1, y0 - fixed - depth - 1
            else:
                raise ValueError("ticket too close to figure")
        else:
            lane = y0
            if x1 <= fixed - depth - 1:
                side, gap = -1, fixed - depth - x1
            elif x0 >= fixed + depth + 2:
                side, gap = 1, x0 - fixed - depth - 1
            else:
                raise ValueError("ticket too close to figure")
        if not start <= lane <= start + length - 2:
            raise ValueError("ticket lane outside spine")
        tickets.append({"row": y0, "col": x0, "lane": lane,
                        "n": len(np.unique(a[y0:y1, x0:x1])),
                        "side": side, "gap": gap})
    tickets.sort(key=lambda x: x["lane"])
    if not 1 <= len(tickets) <= 6:
        raise ValueError("expected 1..6 tickets")
    if any(b["lane"] - t["lane"] < 3 for t, b in zip(tickets, tickets[1:])):
        raise ValueError("ticket lanes must be separated")
    return {"background": bg, "guide": guide, "orientation": orientation,
            "fixed": fixed, "start": start, "length": length, "depth": depth,
            "authority": authority, "box": box, "tickets": tickets}


def _locations(info, lane, distance):
    source = info["fixed"] + info["authority"] * distance
    target = info["fixed"] - info["authority"] * distance
    if info["orientation"] == "h":
        return (source, slice(lane, lane + 2)), (target, slice(lane, lane + 2))
    return (slice(lane, lane + 2), source), (slice(lane, lane + 2), target)


def _apply(a, info, mode="stencil"):
    result = a.copy()
    mask = np.zeros(a.shape, bool)
    if mode == "full":
        for d in range(1, info["depth"] + 1):
            for lane in range(info["start"], info["start"] + info["length"] - 1):
                src, dst = _locations(info, lane, d)
                result[dst] = a[src]
                mask[dst] = True
    else:
        for tile in info["tickets"]:
            for d in range(1, (4 if mode == "four" else tile["n"]) + 1):
                src, dst = _locations(info, tile["lane"], d)
                result[dst] = a[src]
                mask[dst] = True
    return result, mask


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reflect through cardinality-controlled stencils; preserve all other cells."""
    a = _grid(input_grid)
    return _apply(a, _parse(a))[0]


def _lanes(rng, length, count):
    slack = length - (3 * count - 1)
    gaps = rng.multinomial(slack, np.full(count + 1, 1 / (count + 1)))
    c, lanes = int(gaps[0]), []
    for i in range(count):
        lanes.append(c)
        c += 3 + int(gaps[i + 1])
    return lanes


def _template(rng, depth, length, lanes, counts, bg, colors):
    mask = np.zeros((depth, length), bool)
    left, right = 0, length
    for d in range(depth):
        if d:
            left = int(np.clip(left + rng.integers(-1, 3), 0, length - 3))
            right = int(np.clip(right + rng.integers(-2, 2), left + 3, length))
        active = [c for c, n in zip(lanes, counts) if d <= n]
        if active:
            left, right = min(left, min(active)), max(right, max(active) + 2)
        # Adjacent bands overlap so the initial silhouette is connected.
        if d:
            previous = np.flatnonzero(mask[d - 1])
            left, right = min(left, int(previous[-1])), max(right, int(previous[0]) + 1)
        mask[d, left:right] = True

    def acceptable():
        return (_connected(mask) and mask[-1].any()
                and all(mask[d, c:c + 2].any() for c, n in zip(lanes, counts)
                        for d in range(n + 1)))

    # Guarantee at least one hole inside a stencil, rather than relying on luck.
    holes = [(d, c + j) for c, n in zip(lanes, counts)
             for d in range(n) for j in range(2) if mask[d, c:c + 2].all()]
    rng.shuffle(holes)
    for pos in holes:
        mask[pos] = False
        if acceptable():
            break
        mask[pos] = True
    else:
        raise RuntimeError("could not place a template hole")
    candidates = [tuple(int(x) for x in p) for p in np.argwhere(mask)]
    rng.shuffle(candidates)
    desired = int(rng.uniform(0.04, 0.24) * len(candidates))
    carved = 0
    for pos in candidates:
        if carved >= desired:
            break
        mask[pos] = False
        if acceptable():
            carved += 1
        else:
            mask[pos] = True
    template = np.full(mask.shape, bg, dtype=int)
    template[mask] = colors[0]
    occupied = np.argwhere(mask)
    take = max(3, int(rng.uniform(0.10, 0.32) * len(occupied)))
    for index in rng.choice(len(occupied), size=take, replace=False):
        template[tuple(occupied[index])] = colors[1]
    return template


def _damage(rng, template, lanes, counts, bg, colors, teaching):
    depth, length = template.shape
    occupied = [tuple(int(x) for x in p) for p in np.argwhere(template != bg)]
    holes = [tuple(int(x) for x in p) for p in np.argwhere(template == bg)]
    permitted = np.zeros(template.shape, bool)
    for c, n in zip(lanes, counts):
        permitted[:n, c:c + 2] = True
    inside_fg = [p for p in occupied if permitted[p]]
    inside_holes = [p for p in holes if permitted[p]]
    for _ in range(160):
        bad = template.copy()
        # Each depth, including the first forbidden depth, has visible evidence.
        for c, n in zip(lanes, counts):
            for d in range(n + 1):
                choices = [j for j in (c, c + 1) if template[d, j] != bg]
                j = int(rng.choice(choices))
                bad[d, j] = (bg if rng.random() < 0.7 else
                             colors[1] if template[d, j] == colors[0] else colors[0])
        for i in rng.choice(len(occupied), size=max(1, int(rng.uniform(0.04, 0.20)
                                                          * len(occupied))), replace=False):
            bad[occupied[int(i)]] = bg
        if holes:
            for i in rng.choice(len(holes), size=min(len(holes), int(rng.integers(0, 4))), replace=False):
                bad[holes[int(i)]] = int(rng.choice(colors))
        if inside_holes and (teaching or rng.random() < 0.65):
            bad[inside_holes[int(rng.integers(len(inside_holes)))]] = colors[0]
        # Teaching pairs have room for separate recolor/fill witnesses. A single
        # depth-one lane with a source hole can have only one foreground cell.
        chosen = rng.choice(len(inside_fg), size=min(2, len(inside_fg)), replace=False)
        missing = inside_fg[int(chosen[0])]
        if len(chosen) == 2:
            wrong = inside_fg[int(chosen[1])]
            bad[wrong] = colors[1] if template[wrong] == colors[0] else colors[0]
        bad[missing] = bg
        if np.count_nonzero(bad != bg) >= len(occupied):
            continue
        if not np.any(bad[-1] != bg):
            continue
        repaired = bad.copy()
        for c, n in zip(lanes, counts):
            repaired[:n, c:c + 2] = template[:n, c:c + 2]
        spine_mask = np.ones((1, length), bool)
        if not (_connected(np.vstack((template[::-1] != bg, spine_mask, bad != bg)))
                and _connected(np.vstack((template[::-1] != bg, spine_mask, repaired != bg)))):
            continue
        return bad, repaired
    raise RuntimeError("could not keep damaged and repaired silhouettes connected")


def _ticket(rng, n, payload, balanced_two=False):
    chosen = [int(x) for x in rng.choice(payload, n, replace=False)]
    if n == 1:
        values = chosen * 4
    elif n == 2:
        balanced = rng.random() < 0.5
        values = chosen * 2 if balanced or balanced_two else [chosen[0]] * 3 + [chosen[1]]
    elif n == 3:
        values = chosen + [int(rng.choice(chosen))]
    else:
        values = chosen
    rng.shuffle(values)
    return np.asarray(values, dtype=int).reshape(2, 2)


def _pair(rng, counts, bg, guide, orientation, authority, ticket_role, teaching):
    length = int(rng.integers(max(7, 3 * len(counts) - 1), 26))
    depth = int(rng.integers(5, 8))
    lanes = _lanes(rng, length, len(counts))
    payload = [c for c in range(10) if c not in (bg, guide)]
    colors = [int(x) for x in rng.choice(payload, 2, replace=False)]
    template = _template(rng, depth, length, lanes, counts, bg, colors)
    bad, repaired = _damage(rng, template, lanes, counts, bg, colors, teaching)
    if ticket_role == "mixed":
        sides = [int(x) for x in rng.choice((-1, 1), len(counts))]
        sides[0], sides[1] = -1, 1
        rng.shuffle(sides)
        gaps = [int(x) for x in rng.integers(1, 4, len(counts))]
        if len(set(gaps)) == 1:
            gaps[0] = gaps[0] % 3 + 1
    else:
        # Canonical authority is above; flipping later transforms all geometry.
        sides = [-1 if ticket_role == "source" else 1] * len(counts)
        gaps = [int(rng.integers(1, 4))] * len(counts)
    tops = [-depth - gap - 2 if side == -1 else depth + gap + 1
            for side, gap in zip(sides, gaps)]
    ymin, ymax = min(-depth, min(tops)), max(depth, max(y + 1 for y in tops))
    base_h = ymax - ymin + 1
    margin_y = int(rng.integers(2, min(6, 30 - base_h) + 1))
    margin_x = int(rng.integers(2, min(6, 30 - length) + 1))
    top_pad = int(rng.integers(1, margin_y))
    left_pad = int(rng.integers(1, margin_x))
    a = np.full((base_h + margin_y, length + margin_x), bg, dtype=int)
    b = a.copy()
    axis = top_pad - ymin
    region = (slice(axis - depth, axis + depth + 1), slice(left_pad, left_pad + length))
    spine = np.full((1, length), guide, dtype=int)
    a[region] = np.vstack((template[::-1], spine, bad))
    b[region] = np.vstack((template[::-1], spine, repaired))
    for y, c, n in zip(tops, lanes, counts):
        tile = _ticket(rng, n, payload, balanced_two=teaching)
        region = (slice(axis + y, axis + y + 2), slice(left_pad + c, left_pad + c + 2))
        a[region], b[region] = tile, tile
    if authority == 1:
        a, b = a[::-1], b[::-1]
    if orientation == "v":
        a, b = a.T, b.T
    return {"input": a.copy(), "output": b.copy()}


def _key(a):
    a = np.asarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    bg, guide = (int(x) for x in rng.choice(10, 2, replace=False))
    repertoire = [int(x) for x in rng.permutation([1, 2, 3, 4])]
    orientations = ["h", "v"]
    rng.shuffle(orientations)
    roles = ["source", "damaged"]
    rng.shuffle(roles)
    plans = []
    for i in range(num_train):
        if i < 2:
            extra = int(rng.integers(0, 3))
            counts = repertoire[2 * i:2 * i + 2] + [int(x) for x in rng.integers(1, 5, extra)]
            orient, role = orientations[i], roles[i]
        else:
            counts = [int(x) for x in rng.integers(1, 5, int(rng.integers(1, 7)))]
            orient, role = str(rng.choice(("h", "v"))), str(rng.choice(roles))
        rng.shuffle(counts)
        authority = int(rng.choice((-1, 1)))
        if i == 1:
            authority = -plans[0][3]
        plans.append(("train", counts, orient, authority, role, i < 2))
    for _ in range(num_test):
        counts = [int(x) for x in rng.integers(1, 5, int(rng.integers(2, 7)))]
        plans.append(("test", counts, str(rng.choice(("h", "v"))),
                      int(rng.choice((-1, 1))), "mixed", False))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, counts, orient, authority, role, teaching in plans:
        for _ in range(60):
            try:
                pair = _pair(rng, counts, bg, guide, orient, authority, role, teaching)
                _parse(pair["input"])
            except (RuntimeError, ValueError):
                continue
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("could not construct a distinct valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    a, b = _grid(pair["input"]), _grid(pair["output"])
    info = _parse(a)
    assert a.shape == b.shape
    expected, mask = _apply(a, info)
    assert np.array_equal(expected, b), "wrong stencil reflection"
    assert np.array_equal(a[~mask], b[~mask]), "changed cells outside permissions"
    bg = info["background"]
    r0, r1, c0, c1 = info["box"]
    assert _connected(b[r0:r1, c0:c1] != bg), "disconnected repaired figure"
    # Recover the actual source half, independent of ticket positions.
    if info["orientation"] == "h":
        half = (a[r0:info["fixed"], c0:c1] if info["authority"] == -1
                else a[info["fixed"] + 1:r1, c0:c1])
    else:
        half = (a[r0:r1, c0:info["fixed"]] if info["authority"] == -1
                else a[r0:r1, info["fixed"] + 1:c1])
    assert _connected(half != bg), "disconnected authoritative silhouette"
    body_colors = set(int(c) for c in np.unique(half)) - {bg}
    assert len(body_colors) == 2 and info["guide"] not in body_colors
    # Keeping B's geometry but replacing A's pattern copying by solid paint fails.
    solid = a.copy()
    solid[mask] = min(body_colors)
    info["solid_ablation_differs"] = not np.array_equal(solid, b)
    assert not np.array_equal(a, b), "no contribution from reflection"
    assert not np.array_equal(_apply(a, info, "full")[0], b), "no contribution from counting"
    for tile in info["tickets"]:
        for d in range(1, tile["n"] + 1):
            src, dst = _locations(info, tile["lane"], d)
            assert np.any(a[dst] != b[dst]), "missing depth witness"
            assert np.array_equal(b[dst], a[src])
        src, dst = _locations(info, tile["lane"], tile["n"] + 1)
        assert np.any(a[src] != a[dst]), "missing forbidden-layer contrast"
        assert np.array_equal(b[dst], a[dst]), "overrepaired past a ticket"
    changed = a != b
    info["fills"] = int(np.count_nonzero(changed & (a == bg) & (b != bg)))
    info["erases"] = int(np.count_nonzero(changed & (a != bg) & (b == bg)))
    info["recolors"] = int(np.count_nonzero(changed & (a != bg) & (b != bg)))
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen, roles, analyses = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _key(_grid(pair["input"]))
            assert key not in seen
            seen.add(key)
            info = _analyze_pair(pair)
            roles.add((info["background"], info["guide"]))
            analyses[split].append(info)
            sides = {t["side"] for t in info["tickets"]}
            gaps = {t["gap"] for t in info["tickets"]}
            if split == "train":
                assert len(sides) == len(gaps) == 1
            else:
                assert sides == {-1, 1} and len(gaps) >= 2
    assert len(roles) == 1
    first = analyses["train"][:2]
    assert {x["orientation"] for x in first} == {"h", "v"}
    assert {x["authority"] for x in first} == {-1, 1}
    assert {x["tickets"][0]["side"] == x["authority"] for x in first} == {True, False}
    assert {t["n"] for x in first for t in x["tickets"]} == {1, 2, 3, 4}
    for pair, info in zip(puzzle["train"][:2], first):
        assert all(info[k] > 0 for k in ("fills", "erases", "recolors"))
        assert info["solid_ablation_differs"]
        for tile in info["tickets"]:
            if tile["n"] == 2:
                a = np.asarray(pair["input"])
                _, frequency = np.unique(a[tile["row"]:tile["row"] + 2,
                                          tile["col"]:tile["col"] + 2], return_counts=True)
                assert sorted(frequency) == [2, 2]
        assert not np.array_equal(_apply(np.asarray(pair["input"]), info, "four")[0], pair["output"])


def _puzzle_key(puzzle):
    return tuple((split, tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[split]))
                 for split in ("train", "test"))


def _normalized(grid):
    roles = {}
    return tuple(roles.setdefault(int(c), len(roles)) for c in np.asarray(grid).flat)


def self_test():
    """Bounded authored checks, including 42 seeds and causal interventions."""
    shapes, structures, palettes, profiles, lengths, depths, counts = (set() for _ in range(7))
    multiplicities, gaps, signatures = set(), set(), set()
    ticket_counts = {"train": set(), "test": set()}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                info = _analyze_pair(pair)
                shapes.add(a.shape)
                structures.add((a.shape, _normalized(a)))
                palettes.add((info["background"], info["guide"]))
                profiles.add((info["orientation"], info["authority"]))
                lengths.add(info["length"])
                depths.add(info["depth"])
                ticket_counts[split].add(len(info["tickets"]))
                for t in info["tickets"]:
                    counts.add(t["n"])
                    gaps.add(t["gap"])
                    if t["n"] == 2:
                        _, freq = np.unique(a[t["row"]:t["row"] + 2, t["col"]:t["col"] + 2], return_counts=True)
                        multiplicities.add(tuple(sorted(int(x) for x in freq)))
                assert np.array_equal(solve(a.tolist()), b)
        # Equivariance verifies conventions are spatial and relational.
        pair = puzzle["test"][0]
        for turns in (1, 2, 3):
            assert np.array_equal(solve(np.rot90(pair["input"], turns)), np.rot90(pair["output"], turns))
        permutation = np.random.default_rng(10000 + seed).permutation(10)
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
        # Increasing a ticket's cardinality by one grants exactly one more layer.
        p = puzzle["train"][0]
        info = _parse(p["input"])
        # Changing a source value without changing support changes exactly its
        # reflected target value too: the pattern, not just the mask, is causal.
        tile = info["tickets"][0]
        src, dst = _locations(info, tile["lane"], 1)
        source_values = p["input"][src]
        j = int(np.flatnonzero(source_values != info["background"])[0])
        source_pos = ((src[0], tile["lane"] + j) if info["orientation"] == "h"
                      else (tile["lane"] + j, src[1]))
        target_pos = ((dst[0], tile["lane"] + j) if info["orientation"] == "h"
                      else (tile["lane"] + j, dst[1]))
        body_color = int(p["input"][source_pos])
        box = p["input"][info["box"][0]:info["box"][1], info["box"][2]:info["box"][3]]
        alternate = next(int(c) for c in np.unique(box)
                         if c not in (info["background"], info["guide"], body_color))
        mutated, predicted = p["input"].copy(), p["output"].copy()
        mutated[source_pos] = alternate
        predicted[source_pos] = predicted[target_pos] = alternate
        assert np.array_equal(solve(mutated), predicted)
        for tile in info["tickets"]:
            if tile["n"] == 4:
                continue
            mutated = p["input"].copy()
            region = mutated[tile["row"]:tile["row"] + 2, tile["col"]:tile["col"] + 2]
            values, freq = np.unique(region, return_counts=True)
            repeated = int(values[np.argmax(freq)])
            new_color = next(c for c in range(10) if c not in set(values) | {info["background"], info["guide"]})
            rr, cc = np.argwhere(region == repeated)[0]
            region[rr, cc] = new_color
            predicted = p["output"].copy()
            predicted[tile["row"]:tile["row"] + 2, tile["col"]:tile["col"] + 2] = region
            src, dst = _locations(info, tile["lane"], tile["n"] + 1)
            predicted[dst] = mutated[src]
            assert np.array_equal(solve(mutated), predicted)
            assert np.any(predicted[dst] != p["output"][dst])
            break
    assert len(signatures) == 42 and len(shapes) >= 30 and len(structures) >= 150
    assert len(palettes) >= 20 and profiles == {("h", -1), ("h", 1), ("v", -1), ("v", 1)}
    assert len(lengths) >= 12 and depths == {5, 6, 7} and counts == {1, 2, 3, 4}
    assert gaps == {1, 2, 3} and multiplicities == {(1, 3), (2, 2)}
    assert ticket_counts["train"] == {1, 2, 3, 4, 5, 6}
    assert ticket_counts["test"] == {2, 3, 4, 5, 6}
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(900 + 10 * a + b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    assert _puzzle_key(generate_puzzle(701, 7, 3)) == _puzzle_key(generate_puzzle(701, 7, 3))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts {a}/{b}")
    return {"seeds": 42, "distinct_puzzles": len(signatures),
            "grid_shapes": len(shapes), "normalized_input_structures": len(structures),
            "role_palettes": len(palettes), "spine_lengths": sorted(lengths),
            "depths": sorted(depths), "counts": sorted(counts),
            "ticket_counts": {s: sorted(v) for s, v in ticket_counts.items()},
            "two_color_multiplicities": sorted(multiplicities),
            "independent_semantic_review": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
