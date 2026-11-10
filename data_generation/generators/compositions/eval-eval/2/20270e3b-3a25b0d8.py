"""Zippered reservoirs: register gates, join walls, propagate room colors.

Only NumPy and the standard library are used. Generation has no filesystem
effects. The solver parses the query and never consults the RNG or examples.
"""
from __future__ import annotations

import hashlib
from collections import Counter
import numpy as np


HOST_SEEDS = (0, 265074768, 162040059, 639583651, 336355054, 186153293,
              811415791, 40786537, 276006847, 644406889, 736316413,
              633851132, 149178341, 507211814, 667447793, 943216783,
              743186468, 658918653, 589690222, 419275799, 501084762,
              440419877, 963105635, 608940098, 618296177, 302367687,
              270180957, 278699963, 157747595, 710767491, 911491384,
              355624416, 687136045, 708651655, 497151467, 882349306,
              397961571, 284409130, 659052547, 898721455, 145728095,
              826906243)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    parts = []
    steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        steps += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        parts.append(part)
    return parts


def _bbox(points):
    return (min(r for r, c in points), min(c for r, c in points),
            max(r for r, c in points) + 1, max(c for r, c in points) + 1)


def _shift(points, dr, dc):
    return {(r + dr, c + dc) for r, c in points}


def _cavities(barrier):
    padded = np.pad(~barrier, 1, constant_values=True)
    h, w = padded.shape
    return [{(int(r - 1), int(c - 1)) for r, c in part}
            for part in _components(padded)
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in part)]


def _runs(values):
    result = []
    for v in sorted(values):
        if not result or v != result[-1][-1] + 1:
            result.append([v])
        else:
            result[-1].append(v)
    return result


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _parse(input_grid):
    """Infer background, wall, plugs, axis and ownership from pixels only."""
    inp = _grid(input_grid)
    bg = int(inp[0, 0])
    assert all(np.all(edge == bg) for edge in (inp[0], inp[-1], inp[:, 0], inp[:, -1])), "blank border required"
    objects = _components(inp != bg, diagonal=True)
    assert len(objects) == 2, "expected two separate reservoir fragments"
    shared = set.intersection(*[{int(inp[p]) for p in part} for part in objects])
    candidates = []
    for marker in sorted(shared):
        for axis in (0, 1):
            scene = inp if axis == 0 else inp.T
            parts = [part if axis == 0 else {(c, r) for r, c in part}
                     for part in objects]
            records = []
            for part in parts:
                gates = {p for p in part if scene[p] == marker}
                if not gates or len({r for r, c in gates}) != 1:
                    break
                gr, gc, _, _ = _bbox(gates)
                t, l, b, r = _bbox(part)
                if gr not in (t, b - 1) or t == b - 1:
                    break
                pattern = {c - gc for rr, c in gates}
                if len(_runs(pattern)) < 2:
                    break
                # No individual piece is turned. Its plugged edge determines
                # which side of the eventual join contains its interior.
                sign = -1 if gr == b - 1 else 1
                records.append((sign, gr, gc, pattern, part))
            if len(records) != 2 or records[0][0] == records[1][0] or records[0][3] != records[1][3]:
                continue
            records.sort(key=lambda x: x[0])
            for wall in sorted(shared - {marker}):
                fragments = []
                for sign, gr, gc, pattern, part in records:
                    t, l, b, r = _bbox(part)
                    crop = np.full((b - t, r - l), bg, dtype=np.int8)
                    for p in part:
                        crop[p[0] - t, p[1] - l] = scene[p]
                    barrier = (crop == wall) | (crop == marker)
                    rooms = _cavities(barrier)
                    if not rooms or len(_components(barrier, diagonal=True)) != 1:
                        break
                    union = set().union(*rooms)
                    payload = {p for p in part if scene[p] not in (wall, marker)}
                    local_payload = {(rr - t, cc - l) for rr, cc in payload}
                    if not local_payload <= union:
                        break
                    colors = []
                    for room in rooms:
                        values = {int(crop[p]) for p in room}
                        if len(values) != 1 or values & {wall, marker}:
                            break
                        colors.append(next(iter(values)))
                    if len(colors) != len(rooms):
                        break
                    # Every gate opens directly into a compartment, rather
                    # than being a decorative line on a solid wall.
                    if any((gr + sign - t, cc - l) not in union
                           for cc in (gc + x for x in pattern)):
                        break
                    local_rooms = [_shift(room, t - gr, l - gc) for room in rooms]
                    fragments.append({
                        "walls": {(rr - gr, cc - gc) for rr, cc in part if scene[rr, cc] == wall},
                        "gates": {(0, x) for x in pattern},
                        "rooms": local_rooms, "colors": colors,
                        "seeds": {(rr - gr, cc - gc): int(scene[rr, cc]) for rr, cc in payload},
                        "origin": (gr, gc), "box": (t, l, b, r)})
                if len(fragments) == 2:
                    colors = set(fragments[0]["seeds"].values()) | set(fragments[1]["seeds"].values())
                    if len(colors) >= 2:
                        candidates.append({"background": bg, "wall": wall, "marker": marker,
                                           "axis": axis, "fragments": fragments,
                                           "pattern": records[0][3]})
    assert len(candidates) == 1, "input must determine one gate, wall and axis assignment"
    return candidates[0]


def _assemble(parsed, seam=1, retain_gates=False):
    """Pixel construction by barriers and enclosed four-connected regions."""
    a, b = parsed["fragments"]
    walls = a["walls"] | _shift(b["walls"], seam, 0)
    gates = a["gates"] | _shift(b["gates"], seam, 0)
    if retain_gates:
        walls |= gates
    t, l, bottom, right = _bbox(walls)
    bg, wc = parsed["background"], parsed["wall"]
    output = np.full((bottom - t, right - l), bg, dtype=np.int8)
    for r, c in walls:
        output[r - t, c - l] = wc
    for fragment, offset in ((a, 0), (b, seam)):
        for (r, c), color in fragment["seeds"].items():
            p = r + offset, c
            if p not in walls:
                output[p[0] - t, p[1] - l] = color
    before_fill = output.copy()
    rooms = _cavities(output == wc)
    inside = set().union(*rooms) if rooms else set()
    seed_positions = set(map(tuple, np.argwhere((output != bg) & (output != wc))))
    assert seed_positions <= inside, "paint may not escape into the exterior"
    chamber_colors = []
    for room in rooms:
        colors = {int(output[p]) for p in room} - {bg}
        assert len(colors) <= 1, "a chamber has conflicting donor colors"
        color = next(iter(colors)) if colors else bg
        chamber_colors.append(color)
        for p in room:
            output[p] = color
    return output, {"before_fill": before_fill, "walls": walls, "gates": gates,
                    "rooms": rooms, "room_colors": chamber_colors, "origin": (t, l)}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return only the registered, opened and recolored reservoir assembly."""
    parsed = _parse(input_grid)
    output, _ = _assemble(parsed)
    return output.T.copy() if parsed["axis"] else output


class _DSU:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, x, y):
        self.parent[self.find(x)] = self.find(y)


def _plan(rng, depth, n, force_merge=False):
    active = np.ones((depth, n), dtype=bool)
    # Rear corners may be absent: the outer outline is allowed to step in.
    for c in (0, n - 1):
        if rng.random() < .45:
            active[0, c] = False
    ids = np.arange(depth * n).reshape(depth, n)
    dsu = _DSU(depth * n)
    edges = [(int(ids[r, c]), int(ids[r + dr, c + dc]))
             for r, c in np.argwhere(active) for dr, dc in ((1, 0), (0, 1))
             if r + dr < depth and c + dc < n and active[r + dr, c + dc]]
    if force_merge:
        c = int(rng.integers(n - 1))
        dsu.join(int(ids[-1, c]), int(ids[-1, c + 1]))
    for i in rng.permutation(len(edges)):
        if rng.random() < .24:
            dsu.join(*edges[int(i)])
    result = np.full((depth, n), -1, dtype=int)
    roots = {}
    for r, c in np.argwhere(active):
        root = dsu.find(int(ids[r, c]))
        result[r, c] = roots.setdefault(root, len(roots))
    return result


def _geometry(rng, plan, widths, plain=False):
    """Render connected rooms, shared boundaries, channels and local notches.

    -1 exterior, -2 permanent wall, nonnegative integers room identities.
    The common plugged edge is the last row; the second piece is flipped only
    at construction time, before either input is presented.
    """
    depth, n = plan.shape
    heights = np.ones(depth, dtype=int) if plain else rng.integers(1, 3, size=depth)
    ys = np.concatenate(([0], np.cumsum(heights + 1)))
    xs = np.concatenate(([0], np.cumsum(np.asarray(widths) + 1)))
    labels = np.full((int(ys[-1]) + 1, int(xs[-1]) + 1), -1, dtype=int)
    for r in range(depth):
        for c in range(n):
            label = int(plan[r, c])
            if label < 0:
                continue
            labels[ys[r] + 1:ys[r + 1], xs[c] + 1:xs[c + 1]] = label
            if r and plan[r - 1, c] == label:
                labels[ys[r], xs[c] + 1:xs[c + 1]] = label
            if c and plan[r, c - 1] == label:
                labels[ys[r] + 1:ys[r + 1], xs[c]] = label
            if r and c and np.all(plan[r - 1:r + 1, c - 1:c + 1] == label):
                labels[ys[r], xs[c]] = label
    # Notches deform the interiors without inventing a texture-stretch rule.
    # Keep the seam-facing row intact so each plug has a visible chamber.
    for label in sorted(set(plan.flat) - {-1}):
        for _ in range(0 if plain else int(rng.integers(3))):
            points = set(map(tuple, np.argwhere(labels == label)))
            t, l, b, r = _bbox(points)
            corners = [(t, l), (t, r - 1), (b - 1, l), (b - 1, r - 1)]
            q = corners[int(rng.integers(4))]
            if len(points) > 2 and q in points and q[0] < labels.shape[0] - 2:
                trial = labels == label
                trial[q] = False
                if len(_components(trial)) == 1:
                    labels[q] = -1
    occupied = labels >= 0
    padded = np.pad(occupied, 1)
    near = np.zeros(labels.shape, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            near |= padded[dr:dr + labels.shape[0], dc:dc + labels.shape[1]]
    labels[(labels < 0) & near] = -2
    # Geometric labels must be exactly the true enclosed regions.
    rooms = _cavities(labels == -2)
    assert len(rooms) == len(set(plan.flat) - {-1})
    assert all(len({int(labels[p]) for p in room}) == 1 and labels[next(iter(room))] >= 0
               for room in rooms)
    return labels, xs


def _paint(labels, gates, colors, palette, final=False):
    bg, wall, marker = palette[:3]
    grid = np.full(labels.shape, bg, dtype=np.int8)
    grid[labels == -2] = wall
    for label, color in colors.items():
        grid[labels == label] = color
    for cols, color in gates:
        grid[-1, cols] = color if final else marker
    return grid


def _stage(rng, top, bottom, bg):
    """Vary source positions independently of their mating roles."""
    objects = [top, bottom]
    if rng.integers(2):
        objects.reverse()
    a, b = objects
    ah, aw = a.shape
    bh, bw = b.shape
    modes = ["stack"]
    if aw + bw + 3 <= 28:
        modes.extend(("side", "diagonal"))
    mode = modes[int(rng.integers(len(modes)))]
    if mode == "side":
        dr = int(rng.integers(-bh + 1, ah))
        dc = aw + int(rng.integers(1, min(5, 29 - aw - bw)))
    elif mode == "diagonal":
        dr = ah + int(rng.integers(1, min(5, 29 - ah - bh)))
        dc = aw + int(rng.integers(1, min(5, 29 - aw - bw)))
    else:
        dr = ah + int(rng.integers(1, min(5, 29 - ah - bh)))
        room = min(4, 28 - max(aw, bw))
        dc = int(rng.integers(-room, room + 1))
    height, width = max(ah, dr + bh) - min(0, dr), max(aw, dc + bw) - min(0, dc)
    assert height <= 28 and width <= 28
    extra_r = int(rng.integers(0, min(4, 28 - height) + 1))
    extra_c = int(rng.integers(0, min(4, 28 - width) + 1))
    pr = 1 + int(rng.integers(extra_r + 1))
    pc = 1 + int(rng.integers(extra_c + 1))
    canvas = np.full((height + 2 + extra_r, width + 2 + extra_c), bg, dtype=np.int8)
    for obj, rr, cc in ((a, pr - min(0, dr), pc - min(0, dc)),
                        (b, pr + max(0, dr), pc + max(0, dc))):
        view = canvas[rr:rr + obj.shape[0], cc:cc + obj.shape[1]]
        view[obj != bg] = obj[obj != bg]
    return canvas


def _make_pair(rng, n, depth, widths, palette, rotation, _fallback_only=False):
    # Attempts retain only one pair of tiny label arrays, never a history.
    for attempt in range(96 if _fallback_only else 0, 97):
        if attempt == 96:
            # Guaranteed fitting member of the SAME rule: one joined seam
            # room faces two separate receiver rooms, with rear rooms sealed.
            pa = np.arange(depth * n).reshape(depth, n)
            pa[-1, 1] = pa[-1, 0]
            old = sorted(set(pa.flat))
            pa = np.array([[old.index(int(v)) for v in row] for row in pa])
            pb = np.arange(depth * n).reshape(depth, n)
        else:
            pa = _plan(rng, depth, n, force_merge=True)
            pb = _plan(rng, depth, n)
        na = int(pa.max()) + 1
        nb = int(pb.max()) + 1
        dsu = _DSU(na + nb)
        for c in range(n):
            dsu.join(int(pa[-1, c]), na + int(pb[-1, c]))
        groups = {}
        for i in range(na + nb):
            groups.setdefault(dsu.find(i), []).append(i)
        gate_roots = sorted({dsu.find(int(pa[-1, c])) for c in range(n)})
        # At least two independent colors, a genuine one-to-many transfer,
        # and a sealed unpainted room are visible in every example.
        if len(gate_roots) < 2 or not any(len(groups[root]) >= 3 for root in gate_roots):
            continue
        isolated = [root for root in groups if root not in gate_roots]
        if not isolated:
            continue
        try:
            la, xs = _geometry(rng, pa, widths, plain=attempt == 96)
            lb, _ = _geometry(rng, pb, widths, plain=attempt == 96)
        except AssertionError:
            continue
        available = list(map(int, palette[3:]))
        richness = int(rng.integers(2, min(5, len(gate_roots)) + 1))
        hues = available[:richness]
        root_colors = {root: hues[i] if i < richness else hues[int(rng.integers(richness))]
                       for i, root in enumerate(gate_roots)}
        initial, final = [{}, {}], [{}, {}]
        for index, root in enumerate(gate_roots):
            nodes = groups[root]
            color = root_colors[root]
            side = index % 2  # both fragments are donors and receivers
            choices = [i for i in nodes if (i >= na) == bool(side)]
            source = choices[int(rng.integers(len(choices)))]
            for i in nodes:
                s, label = (0, i) if i < na else (1, i - na)
                final[s][label] = color
            s, label = (0, source) if source < na else (1, source - na)
            initial[s][label] = color
            # Same-color donors can agree through a gate, but keep all
            # branching components at one donor to expose fan-out clearly.
            if len(nodes) == 2 and rng.random() < .25:
                other = next(i for i in nodes if i != source)
                s, label = (0, other) if other < na else (1, other - na)
                initial[s][label] = color
        # Keep one isolated room blank; other sealed rooms may retain paint.
        for root in isolated[1:]:
            if rng.random() < .25:
                i = groups[root][0]
                s, label = (0, i) if i < na else (1, i - na)
                color = available[int(rng.integers(min(5, len(available))))]
                initial[s][label] = final[s][label] = color
        gates = []
        for c in range(n):
            # Each plug fills its opening; its length and gaps are geometric
            # registration data and do not encode an output color.
            cols = list(range(int(xs[c]) + 1, int(xs[c + 1])))
            gates.append((cols, root_colors[dsu.find(int(pa[-1, c]))]))
        top = _paint(la, gates, initial[0], palette)
        bottom = _paint(lb, gates, initial[1], palette)[::-1].copy()
        expected = np.concatenate((_paint(la, gates, final[0], palette, final=True),
                                   _paint(lb, gates, final[1], palette, final=True)[::-1]), axis=0)
        inp = _stage(rng, top, bottom, int(palette[0]))
        pair = {"input": np.rot90(inp, rotation).copy(),
                "output": np.rot90(expected, rotation).copy()}
        # Labels construct the answer. A separate pixel flood reconstruction
        # must agree; construction indices are never passed into the solver.
        try:
            assert np.array_equal(solve(pair["input"]), pair["output"])
        except AssertionError:
            continue
        return pair
    raise AssertionError("bounded reservoir construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    # These are sampling regimes, not hidden solving conventions.
    shift = int(rng.integers(3))  # gate-count interpolation, depth, plug length
    palette = rng.permutation(10)
    varying_palette = bool(rng.integers(2))
    rotation = int(rng.integers(4))
    mixed_axis = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            test = split == "test"
            n = (4 if test else (3, 5)[index % 2]) if shift == 0 else int(rng.integers(3, 7))
            depth = (3 if test else 2) if shift == 1 else int(rng.integers(2, 4))
            if shift == 2:
                widths = np.full(n, 2 if test else (1, 3)[index % 2], dtype=int)
            else:
                widths = rng.integers(1, 4, size=n)
            colors = rng.permutation(10) if varying_palette else palette
            k = (rotation + index % 2 + 2 * int(rng.integers(2))) % 4 if mixed_axis else rotation
            pair = _make_pair(rng, n, depth, widths, colors, k)
            # A finite 90-way background/wall recoloring is a bounded unique
            # fallback. At most 14 inputs have already been retained.
            if _key(pair["input"]) in seen:
                old = list(map(int, colors))
                for bg in range(10):
                    for wall in range(10):
                        if bg == wall:
                            continue
                        new = [bg, wall] + [c for c in old if c not in (bg, wall)]
                        mapping = np.empty(10, dtype=np.int8)
                        mapping[old] = new
                        replacement = {name: mapping[value] for name, value in pair.items()}
                        if _key(replacement["input"]) not in seen:
                            pair = replacement
                            break
                    else:
                        continue
                    break
            assert _key(pair["input"]) not in seen
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    return puzzle


def measure_pair(pair):
    inp, expected = _grid(pair["input"]), _grid(pair["output"])
    p = _parse(inp)
    answer, info = _assemble(p)
    target = expected.T if p["axis"] else expected
    assert np.array_equal(answer, target), "incorrect assembly, propagation or crop"
    bg, wall = p["background"], p["wall"]
    a, b = p["fragments"]
    t, l = info["origin"]
    before = info["before_fill"]
    # All permanent walls survive exactly; no paint crosses a wall or the
    # exterior, donor interiors are preserved, and each final chamber is flat.
    assert int(np.sum(answer == wall)) == len(a["walls"]) + len(b["walls"])
    assert np.array_equal(answer == wall, before == wall)
    assert np.array_equal(answer[before != bg], before[before != bg])
    assert np.all((before == bg) | (answer == before))
    new_paint = int(np.sum((before == bg) & (answer != bg)))
    assert new_paint > 0
    assert p["marker"] not in answer
    final_index = {q: i for i, room in enumerate(info["rooms"]) for q in room}
    source_rooms = []
    for side, fragment in enumerate((a, b)):
        for room, color in zip(fragment["rooms"], fragment["colors"]):
            converted = {(r + side - t, c - l) for r, c in room}
            dests = {final_index[q] for q in converted}
            assert len(dests) == 1, "a source compartment was split by assembly"
            dest = next(iter(dests))
            if color != bg:
                assert info["room_colors"][dest] == color
            source_rooms.append((side, color, dest, len(room)))
    groups = {i: [r for r in source_rooms if r[2] == i] for i in range(len(info["rooms"]))}
    fanouts = [rows for rows in groups.values()
               if sum(r[1] != bg for r in rows) == 1 and sum(r[1] == bg for r in rows) >= 2]
    assert fanouts, "must expose one donor reaching multiple receiver rooms"
    assert any(c == bg for c in info["room_colors"]), "sealed blank chamber required"
    assert all(any(r[0] == side and r[1] != bg and len(groups[r[2]]) >= 2 for r in source_rooms)
               for side in (0, 1)), "both pieces must supply an active donor"
    for r, c in info["gates"]:
        assert answer[r - t, c - l] not in (bg, wall), "every removed plug belongs to a seeded chamber"
    # Counterfactuals measure real inherited-concept dependence.
    assert not np.array_equal(answer, before), "ignoring compartment transfer changes the answer"
    plugged, _ = _assemble(p, retain_gates=True)
    assert not np.array_equal(answer, plugged), "ignoring transparent gate removal changes the answer"
    wrong_walls = a["walls"] | b["walls"]
    assert wrong_walls != info["walls"], "coincident markers must not imitate adjacency"
    border_walls = np.argwhere(answer == wall)
    assert tuple(border_walls.min(0)) == (0, 0)
    assert tuple(border_walls.max(0) + 1) == answer.shape
    orig_delta = (b["origin"][0] - a["origin"][0], b["origin"][1] - a["origin"][1])
    boxes = [f["box"] for f in (a, b)]
    row_overlap = min(x[2] for x in boxes) > max(x[0] for x in boxes)
    col_overlap = min(x[3] for x in boxes) > max(x[1] for x in boxes)
    layout = "side" if row_overlap else "stack" if col_overlap else "diagonal"
    return {"axis": p["axis"], "roles": (bg, wall, p["marker"]),
            "gates": len(_runs(p["pattern"])), "lengths": tuple(len(r) for r in _runs(p["pattern"])),
            "input_rooms": (len(a["rooms"]), len(b["rooms"])), "final_rooms": len(info["rooms"]),
            "fanouts": len(fanouts), "new_paint": new_paint,
            "blank_chambers": sum(c == bg for c in info["room_colors"]),
            "payload_colors": len(set(info["room_colors"]) - {bg}),
            "nonrectangular": sum(len(room) != (_bbox(room)[2] - _bbox(room)[0]) *
                                  (_bbox(room)[3] - _bbox(room)[1]) for room in info["rooms"]),
            "layout": layout, "initial_marker_delta": orig_delta,
            "structure": (answer.shape, (answer == wall).tobytes())}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_structures = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            assert not np.array_equal(inp, out)
            m = measure_pair(pair)
            if split == "train":
                train_structures.add(m["structure"])
    assert len(train_structures) >= 2, "demonstrations must vary their walls"


def self_test():
    """Bounded in-memory survey of the 42 supplied seeds; writes no files."""
    shapes, structures, roles, fingerprints = set(), set(), set(), set()
    gates, lengths, axes, layouts = set(), set(), set(), set()
    counters = Counter()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                m = measure_pair(pair)
                shapes.add(pair["input"].shape)
                structures.add(m["structure"])
                roles.add(m["roles"])
                gates.add(m["gates"])
                lengths.update(m["lengths"])
                axes.add(m["axis"])
                layouts.add(m["layout"])
                counters["pairs"] += 1
                counters["nonrectangular"] += m["nonrectangular"] > 0
                counters["fanouts"] += m["fanouts"]
                counters["blank_chambers"] += m["blank_chambers"]
                for name in ("input", "output"):
                    digest.update(bytes(pair[name].shape) + pair[name].tobytes())
        fingerprints.add(digest.digest())
    assert len(fingerprints) == 42
    assert len(shapes) >= 45 and len(structures) >= 150 and len(roles) >= 40
    assert gates == {3, 4, 5, 6} and lengths == {1, 2, 3} and axes == {0, 1}
    assert layouts == {"side", "stack", "diagonal"}
    assert counters["nonrectangular"] >= 150 and counters["fanouts"] >= 210
    # Force the final construction across every width/count/depth regime;
    # random notches must not undermine the guaranteed fitting fallback.
    fallback_cases = 0
    rng = np.random.default_rng(454)
    for n in range(3, 7):
        for depth in (2, 3):
            for length in (1, 2, 3):
                pair = _make_pair(rng, n, depth, np.full(n, length), np.arange(10),
                                  fallback_cases % 4, _fallback_only=True)
                measure_pair(pair)
                fallback_cases += 1
    # Every legal explicit count; very small returned grids bound memory use.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(37, a, b)
            validate_generated_puzzle(p)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first, second = generate_puzzle(0, 2, 1), generate_puzzle(0, 2, 1)
    assert all(_key(p[k]) == _key(q[k]) for s in first
               for p, q in zip(first[s], second[s]) for k in ("input", "output"))
    # Equivariance, not memorized labels: recolor roles and rotate whole scenes.
    pair = first["train"][0]
    permutation = np.array([4, 8, 1, 6, 9, 0, 7, 3, 5, 2], dtype=np.int8)
    for k in range(4):
        assert np.array_equal(solve(np.rot90(permutation[pair["input"]], k)),
                              np.rot90(permutation[pair["output"]], k))
    # The validator must actually reject damaged permanent walls or paint.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": counters["pairs"], "input_shapes": len(shapes),
            "wall_structures": len(structures), "role_triples": len(roles),
            "gate_counts": sorted(gates), "gate_lengths": sorted(lengths),
            "axes": sorted(axes), "layouts": sorted(layouts),
            "nonrectangular_pairs": counters["nonrectangular"],
            "forced_fallback_cases": fallback_cases,
            "one_to_many_transfers": counters["fanouts"],
            "preserved_blank_chambers": counters["blank_chambers"]}
