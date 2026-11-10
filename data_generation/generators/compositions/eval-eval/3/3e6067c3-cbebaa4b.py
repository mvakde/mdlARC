"""A color program chooses the order of rigid, geometrically docked stations.

Standard library and NumPy only. No parent imports, file I/O, cached puzzles or
generation-time state in solve(). See task_spec.md for the complete grammar.
"""
from __future__ import annotations

from collections import Counter
from itertools import product
import numpy as np


# Clockwise: up, right, down, left. Port keys always contain TWO sparse pixels.
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return g.astype(np.int8, copy=False)


def _components(cells):
    left, result = set(cells), []
    while left:
        first = min(left)
        left.remove(first)
        stack, found = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _shift(cells, delta):
    return {(r + delta[0], c + delta[1]) for r, c in cells}


def _module(center, size, patch, label, ports, carrier, marker):
    """Construct the full rigid cell map; a port is side -> (spacing, reach)."""
    cy, cx = center
    radius, half_patch = size // 2, patch // 2
    cells = {(cy + r, cx + c): carrier
             for r in range(-radius, radius + 1) for c in range(-radius, radius + 1)}
    for r in range(-half_patch, half_patch + 1):
        for c in range(-half_patch, half_patch + 1):
            cells[cy + r, cx + c] = label
    pins, port_pins = set(), {}
    for side, (spacing, reach) in ports.items():
        dr, dc = DIRS[side]
        tr, tc = -dc, dr
        tips = set()
        for tangent in (-spacing, spacing):
            for normal in range(radius + 1, radius + reach + 1):
                q = cy + dr * normal + tr * tangent, cx + dc * normal + tc * tangent
                assert q not in cells
                cells[q] = marker if normal == radius + reach else carrier
                if normal == radius + reach:
                    tips.add(q)
        pins |= tips
        port_pins[side] = tips
    return dict(center=tuple(center), label=int(label), ports=dict(ports), cells=cells,
                pins=pins, port_pins=port_pins)


def _parse_module(grid, component, common, carrier, marker, size):
    labels = {int(grid[q]) for q in component} - common
    if len(labels) != 1:
        raise ValueError("module needs exactly one label hue")
    label = labels.pop()
    patch_cells = {q for q in component if grid[q] == label}
    top = min(r for r, c in patch_cells)
    left = min(c for r, c in patch_cells)
    patch = max(r for r, c in patch_cells) - top + 1
    if patch not in (1, 3) or patch >= size:
        raise ValueError("invalid patch")
    if patch_cells != set(product(range(top, top + patch), range(left, left + patch))):
        raise ValueError("label patch is not square")
    center = top + patch // 2, left + patch // 2
    radius = size // 2
    grouped = {}
    for y, x in component:
        if grid[y, x] != marker:
            continue
        r, c = y - center[0], x - center[1]
        choices = []
        for side, (dr, dc) in enumerate(DIRS):
            normal, tangent = dr * r + dc * c, -dc * r + dr * c
            if radius < normal <= radius + 3 and abs(tangent) <= radius:
                choices.append((side, normal, tangent))
        if len(choices) != 1:
            raise ValueError("pin is outside the fork grammar")
        side, normal, tangent = choices[0]
        grouped.setdefault(side, []).append((normal, tangent))
    if len(grouped) not in (1, 2):
        raise ValueError("module needs one or two ports")
    ports = {}
    for side, tips in grouped.items():
        if len(tips) != 2:
            raise ValueError("each fork needs two pins")
        (a, u), (b, v) = sorted(tips)
        if a != b or u != -v or not patch // 2 < abs(u) <= radius:
            raise ValueError("fork tips must straddle the central corridor")
        ports[side] = abs(u), a - radius
    module = _module(center, size, patch, label, ports, carrier, marker)
    if module["cells"] != {q: int(grid[q]) for q in component}:
        raise ValueError("piece is not a complete square-and-fork module")
    return module, patch


def _parse(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    if not all(np.all(e == bg) for e in (grid[0], grid[-1], grid[:, 0], grid[:, -1])):
        raise ValueError("background must surround the scene")
    components = _components(map(tuple, np.argwhere(grid != bg).tolist()))
    tokens = sorted((next(iter(c)) for c in components if len(c) == 1), key=lambda q: q[1])
    bodies = [c for c in components if len(c) > 1]
    if not 3 <= len(bodies) == len(tokens) <= 6:
        raise ValueError("one token per module, with 3..6 modules")
    key_y = tokens[0][0]
    if (any(y != key_y for y, x in tokens)
            or any(b[1] - a[1] != 2 for a, b in zip(tokens, tokens[1:]))
            or max(r for comp in bodies for r, c in comp) > key_y - 2):
        raise ValueError("strip must be detached and horizontal")
    # Require an eight-neighbor buffer between input modules. Pins inside each
    # component remain fully attached to their own body.
    index = {q: i for i, comp in enumerate(bodies) for q in comp}
    for (r, c), i in index.items():
        if any(index.get((r + dr, c + dc), i) != i for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
            raise ValueError("input modules touch")
    common = set.intersection(*({int(grid[q]) for q in comp} for comp in bodies))
    if len(common) != 2:
        raise ValueError("carrier and pin hues must be the two shared hues")
    key = [int(grid[q]) for q in tokens]
    parsed = []
    for carrier in sorted(common):
        marker = next(iter(common - {carrier}))
        for size in (3, 5):
            try:
                modules, patches = [], set()
                for comp in bodies:
                    module, patch = _parse_module(grid, comp, common, carrier, marker, size)
                    modules.append(module)
                    patches.add(patch)
                if len(patches) != 1:
                    continue
                labels = [m["label"] for m in modules]
                if (Counter(labels) != Counter(key) or len(set(labels)) < 2
                        or labels.count(key[0]) != 1
                        or sorted(len(m["ports"]) for m in modules) != [1, 1] + [2] * (len(modules) - 2)):
                    continue
                anchor = labels.index(key[0])
                if len(modules[anchor]["ports"]) != 1:
                    continue
                parsed.append(dict(grid=grid, bg=bg, carrier=carrier, marker=marker,
                                   modules=modules, size=size, patch=patches.pop(), key=key,
                                   tokens=tokens, key_y=key_y, anchor=anchor))
            except (ValueError, AssertionError):
                continue
    if len(parsed) != 1:
        raise ValueError("ambiguous or invalid module grammar")
    return parsed[0]


def _band(a, b, size, patch):
    radius, half = size // 2, patch // 2
    if a[0] == b[0] and abs(a[1] - b[1]) > size:
        lo, hi = sorted((a[1], b[1]))
        return set(product(range(a[0] - half, a[0] + half + 1), range(lo + radius + 1, hi - radius)))
    if a[1] == b[1] and abs(a[0] - b[0]) > size:
        lo, hi = sorted((a[0], b[0]))
        return set(product(range(lo + radius + 1, hi - radius), range(a[1] - half, a[1] + half + 1)))
    raise ValueError("station bodies do not have a positive aligned gap")


def _search(info, ordered=True, limit=2):
    """Exhaustive chain placements, stopping only at the requested answer count.

    Each interior module has exactly one remaining exit after docking, so only
    module assignments branch. At most 6! orders exist. No latent generation data.
    """
    modules, marker = info["modules"], info["marker"]
    n, anchor = len(modules), info["anchor"]
    start = modules[anchor]
    positions = {anchor: (0, 0)}
    answers, nodes = [], 0
    width = info["grid"].shape[1]

    def visit(route, entry, occupied, multiplicity, ribbons):
        nonlocal nodes
        nodes += 1
        current = route[-1]
        m = modules[current]
        delta = positions[current]
        center = (m["center"][0] + delta[0], m["center"][1] + delta[1])
        exits = [side for side in m["ports"] if side != entry]
        if len(route) == n:
            if exits or any(v != 2 for v in multiplicity.values()):
                return
            out = np.full(info["grid"].shape, info["bg"], dtype=np.int8)
            for q in info["tokens"]:
                out[q] = info["grid"][q]
            for q, color in occupied.items():
                out[q] = color
            for q, color in ribbons.items():
                out[q] = color
            answers.append(dict(output=out, positions=dict(positions), route=tuple(route)))
            return
        if len(exits) != 1:
            return
        side = exits[0]
        spacing, reach = m["ports"][side]
        opposite = (side + 2) % 4
        tips = _shift(m["port_pins"][side], delta)
        if any(multiplicity[q] != 1 for q in tips):
            return
        dr, dc = DIRS[side]
        for j, other in enumerate(modules):
            if (j in positions or (ordered and other["label"] != info["key"][len(route)])
                    or len(other["ports"]) != (1 if len(route) == n - 1 else 2)
                    or opposite not in other["ports"]):
                continue
            other_spacing, other_reach = other["ports"][opposite]
            if other_spacing != spacing:
                continue
            distance = info["size"] - 1 + reach + other_reach
            target = center[0] + dr * distance, center[1] + dc * distance
            shift = target[0] - other["center"][0], target[1] - other["center"][1]
            moved = {(r + shift[0], c + shift[1]): color for (r, c), color in other["cells"].items()}
            if any(not (1 <= r <= info["key_y"] - 2 and 1 <= c < width - 1) for r, c in moved):
                continue
            overlap = moved.keys() & occupied.keys()
            if overlap != tips or any(moved[q] != marker or occupied[q] != marker for q in overlap):
                continue
            if moved.keys() & ribbons.keys():
                continue
            band = _band(center, target, info["size"], info["patch"])
            if band & (occupied.keys() | moved.keys() | ribbons.keys()):
                continue
            pins = _shift(other["pins"], shift)
            new_counts = multiplicity.copy()
            for q in pins:
                new_counts[q] = new_counts.get(q, 0) + 1
            new_ribbons = ribbons.copy()
            new_ribbons.update({q: m["label"] for q in band})
            positions[j] = shift
            visit(route + [j], opposite, occupied | moved, new_counts, new_ribbons)
            del positions[j]
            if len(answers) >= limit:
                return

    visit([anchor], None, start["cells"].copy(), {q: 1 for q in start["pins"]}, {})
    assert nodes <= 2000  # fewer than sum(P(5,k)) possible partial assignments
    return answers, nodes


def _decode(input_grid):
    info = _parse(input_grid)
    answers, nodes = _search(info)
    if len(answers) != 1:
        raise ValueError("program does not determine a unique complete docking")
    info["answer"], info["nodes"] = answers[0], nodes
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the unique output using only visible input cells and fixed grammar."""
    return _decode(input_grid)["answer"]["output"].copy()


def _path(rng, count):
    """Sample self-avoiding nearest-level paths without a fixed output scaffold."""
    for attempt in range(80):
        rows, cols = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        if rows * cols < count:
            continue
        start = int(rng.integers(rows)), int(rng.integers(cols))
        path = [start]
        while len(path) < count:
            r, c = path[-1]
            choices = [(r + dr, c + dc) for dr, dc in DIRS
                       if 0 <= r + dr < rows and 0 <= c + dc < cols
                       and (r + dr, c + dc) not in path]
            if not choices:
                break
            path.append(choices[int(rng.integers(len(choices)))])
        if len(path) == count:
            return path
    return [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 1)][:count]


def _symmetry(rng, path):
    k = int(rng.integers(4))
    reflected = bool(rng.integers(2))
    result = []
    for r, c in path:
        if reflected:
            c = -c
        for _ in range(k):
            r, c = c, -r
        result.append((r, c))
    top = min(r for r, c in result)
    left = min(c for r, c in result)
    return [(r - top, c - left) for r, c in result]


def _scatter(rng, modules, shape, key_y, bg, token_cells):
    """Bounded packing of complete pieces with the program's first piece fixed."""
    for restart in range(12):
        occupied = dict(modules[0]["cells"])
        forbidden = {(r + dr, c + dc) for r, c in occupied for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        order = list(rng.permutation(np.arange(1, len(modules))))
        order.sort(key=lambda i: -len(modules[i]["cells"]))
        ok = True
        for i in order:
            cells = modules[int(i)]["cells"]
            top, left = min(r for r, c in cells), min(c for r, c in cells)
            bottom, right = max(r for r, c in cells), max(c for r, c in cells)
            height, width = bottom - top + 1, right - left + 1
            if key_y - 2 - height < 0 or shape[1] - 2 - width < 0:
                ok = False
                break
            for trial in range(400):
                y = int(rng.integers(1, key_y - height))
                x = int(rng.integers(1, shape[1] - width))
                shift = y - top, x - left
                if shift == (0, 0):
                    continue
                moved = {(r + shift[0], c + shift[1]): color for (r, c), color in cells.items()}
                if moved.keys() & forbidden:
                    continue
                occupied.update(moved)
                forbidden |= {(r + dr, c + dc) for r, c in moved for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                break
            else:
                ok = False
                break
        if ok:
            grid = np.full(shape, bg, dtype=np.int8)
            for q, color in occupied.items():
                grid[q] = color
            for q, color in token_cells.items():
                grid[q] = color
            return grid
    return None


def _construct(rng, kind, count, reuse, fallback=False):
    compact_backup = fallback and kind == "free"
    if kind == "intro":
        path, size, patch = [(0, i) for i in range(4)], 5, 3
    elif kind == "contrast":
        path, size, patch = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)], 5, 1
    elif compact_backup:
        path = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 1)][:count]
        size, patch = 5, 1
    else:
        path = _path(rng, count)
        size = int(rng.choice((3, 5, 5)))
        patch = 3 if size == 5 and rng.random() < .3 else 1
    path = _symmetry(rng, path)
    rows = max(r for r, c in path) + 1
    cols = max(c for r, c in path) + 1
    gap_max = 2 if kind == "intro" else 3
    ygaps = [int(rng.integers(1, gap_max + 1)) for _ in range(rows - 1)]
    xgaps = [int(rng.integers(1, gap_max + 1)) for _ in range(cols - 1)]
    if kind == "intro":
        (ygaps if ygaps else xgaps)[0] = 2
    if compact_backup:
        ygaps, xgaps = [1] * (rows - 1), [1] * (cols - 1)
    ys = [0] + list(np.cumsum([size + gap for gap in ygaps]).astype(int))
    xs = [0] + list(np.cumsum([size + gap for gap in xgaps]).astype(int))
    centers = [(ys[r], xs[c]) for r, c in path]
    ports = [{} for _ in path]
    spacings = []
    for i, (a, b) in enumerate(zip(centers, centers[1:])):
        vector = np.sign(np.array(b) - np.array(a)).astype(int)
        side = DIRS.index(tuple(vector))
        distance = abs(a[0] - b[0]) + abs(a[1] - b[1])
        total = distance - size + 1
        reach = int(rng.integers(max(1, total - 3), min(3, total - 1) + 1))
        spacing = int(rng.integers(patch // 2 + 1, size // 2 + 1))
        if kind == "contrast" and i in (0, 2):
            spacing = 1 if i == 0 else 2
        if compact_backup:
            spacing = (1, 2, 2, 1, 1)[i]
        ports[i][side] = spacing, reach
        ports[i + 1][(side + 2) % 4] = spacing, total - reach
        spacings.append(spacing)
    palette = [int(c) for c in rng.permutation(10)]
    bg, carrier, marker = palette[:3]
    available = palette[3:]
    if kind == "intro":
        labels = available[:4]
    elif kind == "contrast":
        labels = [available[i] for i in (0, 1, 2, 1, 3)]
    elif compact_backup:
        labels = [available[i] for i in (0, 1, 2, 1, 2, 3)][:count]
    else:
        pool_size = int(rng.integers(2, min(3 if reuse else 5, count - 1) + 1))
        if reuse:
            pool_size = min(pool_size, count - 2)
        pool = available[1:1 + pool_size]
        tail = pool + [int(rng.choice(pool)) for _ in range(count - 1 - len(pool))]
        rng.shuffle(tail)
        labels = [available[0]] + tail
    modules = [_module(center, size, patch, label, pp, carrier, marker)
               for center, label, pp in zip(centers, labels, ports)]
    all_cells = set().union(*(set(m["cells"]) for m in modules))
    top, left = min(r for r, c in all_cells), min(c for r, c in all_cells)
    height = max(r for r, c in all_cells) - top + 1
    width = max(c for r, c in all_cells) - left + 1
    hmin, wmin = max(23, height + 4), max(23, width + 2)
    if hmin > 30 or wmin > 30:
        return None
    h = 30 if fallback else int(rng.integers(hmin, 31))
    w = 30 if fallback else int(rng.integers(wmin, 31))
    offset = int(rng.integers(1, h - height - 2)) - top, int(rng.integers(1, w - width)) - left
    centers = [(r + offset[0], c + offset[1]) for r, c in centers]
    modules = [_module(center, size, patch, label, pp, carrier, marker)
               for center, label, pp in zip(centers, labels, ports)]
    key_y, key_x = h - 2, int(rng.integers(1, w - 2 * len(labels) + 1))
    tokens = {(key_y, key_x + 2 * i): label for i, label in enumerate(labels)}
    # Direct construction of target, independent of the input placement search.
    output = np.full((h, w), bg, dtype=np.int8)
    ownership, payloads = Counter(), set()
    for m in modules:
        for q, color in m["cells"].items():
            if output[q] != bg and (color != marker or output[q] != marker):
                return None
            output[q] = color
            if color == marker:
                ownership[q] += 1
            else:
                payloads.add(q)
    if set(ownership.values()) != {2} or ownership.keys() & payloads:
        return None
    for i, (a, b) in enumerate(zip(centers, centers[1:])):
        band = _band(a, b, size, patch)
        if any(output[q] != bg for q in band):
            return None
        for q in band:
            output[q] = labels[i]
    for q, label in tokens.items():
        output[q] = label
    input_grid = _scatter(rng, modules, (h, w), key_y, bg, tokens)
    if input_grid is None:
        return None
    try:
        info = _decode(input_grid)
    except ValueError:
        return None
    if not np.array_equal(info["answer"]["output"], output):
        return None
    return {"input": input_grid, "output": output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return distinct, deterministic examples with exactly the requested counts."""
    nt, nq = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    reuse = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            kind = "intro" if split == "train" and index == 0 else (
                "contrast" if split == "train" and index == 1 else "free")
            pieces = 4 if kind == "intro" else 5 if kind == "contrast" else (
                int(rng.integers(3, 6)) if split == "train" else int(rng.integers(4, 7)) if reuse else 6)
            for attempt in range(72):
                pair = _construct(rng, kind, pieces, reuse and split == "test", fallback=attempt >= 64)
                if pair is None:
                    continue
                signature = pair["input"].shape, pair["input"].tobytes()
                if signature not in seen:
                    break
            else:
                raise RuntimeError("bounded construction exhausted its attempts")
            seen.add(signature)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    info = _decode(inp)
    answer = info["answer"]
    assert np.array_equal(out, answer["output"]), "wrong assembly or ribbon"
    assert not np.array_equal(inp, out)
    # Recheck whole-piece preservation, all pin ownership, and the entire blank
    # canvas using recovered translations, without trusting the search renderer.
    expected = np.full(inp.shape, info["bg"], dtype=np.int8)
    ownership, occupied, fixed = Counter(), set(), []
    for j, delta in answer["positions"].items():
        m = info["modules"][j]
        if delta == (0, 0):
            fixed.append(j)
        for (r, c), color in m["cells"].items():
            q = r + delta[0], c + delta[1]
            assert 1 <= q[0] <= info["key_y"] - 2 and 1 <= q[1] < inp.shape[1] - 1
            if color == info["marker"]:
                ownership[q] += 1
            else:
                assert q not in occupied
                occupied.add(q)
            assert expected[q] in (info["bg"], info["marker"])
            expected[q] = color
    assert fixed == [info["anchor"]], "only the first-token module stays fixed"
    assert set(ownership.values()) == {2} and not ownership.keys() & occupied
    assert np.count_nonzero(inp == info["marker"]) == 2 * np.count_nonzero(out == info["marker"])
    centers = []
    for j in answer["route"]:
        m, delta = info["modules"][j], answer["positions"][j]
        centers.append((m["center"][0] + delta[0], m["center"][1] + delta[1]))
    band_cells, gaps, sides = set(), [], []
    for i, (a, b) in enumerate(zip(centers, centers[1:])):
        band = _band(a, b, info["size"], info["patch"])
        assert band and not band & band_cells
        assert all(expected[q] == info["bg"] for q in band)
        assert len(band) % info["patch"] == 0
        gaps.append(len(band) // info["patch"])
        sides.append(DIRS.index(tuple(np.sign(np.array(b) - a).astype(int))))
        for q in band:
            expected[q] = info["key"][i]
        band_cells |= band
    for q in info["tokens"]:
        expected[q] = inp[q]
    assert np.array_equal(expected, out), "vacated cells or pieces were not preserved exactly"
    info.update(centers=centers, gaps=gaps, sides=sides, band_cells=band_cells)
    return info


def _wrong_spacing_witness(info):
    """A same-label, same-facing, equal-area candidate rejected by key geometry."""
    route = info["answer"]["route"]
    for step, (a, b) in enumerate(zip(route, route[1:])):
        side = info["sides"][step]
        spacing = info["modules"][a]["ports"][side][0]
        for j, m in enumerate(info["modules"]):
            if (j not in route[:step + 2] and m["label"] == info["modules"][b]["label"]
                    and (side + 2) % 4 in m["ports"]
                    and m["ports"][(side + 2) % 4][0] != spacing):
                return True
    return False


def validate_generated_puzzle(puzzle) -> None:
    """Schema plus semantic, placement, teaching and parent-necessity invariants."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = info["grid"].shape, info["grid"].tobytes()
            assert key not in seen, "duplicate input"
            seen.add(key)
            infos[split].append(info)
    train = infos["train"]
    assert any(a["size"] == b["size"] and a["patch"] != b["patch"] for a in train for b in train)
    assert {s % 2 for a in train for s in a["sides"]} == {0, 1}
    assert any(len(set(a["sides"])) == 1 for a in train)
    assert any(len(set(a["key"])) < len(a["key"]) for a in train)
    assert any(_wrong_spacing_witness(a) for a in train), "no demonstration needs full pin geometry"
    alternatives, _ = _search(train[0], ordered=False)
    assert len(alternatives) == 2
    assert not np.array_equal(alternatives[0]["output"], alternatives[1]["output"]), "key must choose body placement"
    assert max(train[0]["gaps"]) > 1, "reach-dependent gap teaching case missing"


def measure_puzzle(puzzle):
    """Small visible-grid features for bounded diagnostics; no sampled metadata."""
    records = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            a = _check_pair(pair)
            qs = tuple(sorted({q for m in a["modules"] for q, reach in m["ports"].values()}))
            reaches = tuple(sorted({reach for m in a["modules"] for q, reach in m["ports"].values()}))
            normalized = tuple((r - min(y for y, x in a["centers"]), c - min(x for y, x in a["centers"])) for r, c in a["centers"])
            records.append(dict(split=split, shape=tuple(a["grid"].shape), size=a["size"], patch=a["patch"],
                                count=len(a["modules"]), labels=len(set(a["key"])), spacing=qs, reaches=reaches,
                                gaps=tuple(a["gaps"]), turns=sum(x != y for x, y in zip(a["sides"], a["sides"][1:])),
                                roles=(a["bg"], a["carrier"], a["marker"]), key=tuple(a["key"]), layout=normalized,
                                wrong_spacing=_wrong_spacing_witness(a), search_nodes=a["nodes"]))
    return records


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """42 seeds, streamed; no files, logs, retained datasets or extra workers."""
    records, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        records.extend(measure_puzzle(puzzle))
        signatures.add(puzzle["train"][0]["input"].tobytes())
        if seed in (0, 17):
            assert _equal(puzzle, generate_puzzle(seed))
    assert len(signatures) == 42
    assert {m["count"] for m in records} == {3, 4, 5, 6}
    assert {(m["size"], m["patch"]) for m in records} == {(3, 1), (5, 1), (5, 3)}
    assert len({m["shape"] for m in records}) >= 25
    assert len({m["layout"] for m in records}) >= 70
    assert len({m["roles"] for m in records}) >= 60
    assert {r for m in records for r in m["reaches"]} == {1, 2, 3}
    assert {q for m in records for q in m["spacing"]} == {1, 2}
    assert any(m["turns"] >= 3 for m in records if m["split"] == "test")
    assert any(m["count"] < 6 for m in records if m["split"] == "test")
    assert any(len(set(m["gaps"])) >= 3 for m in records)
    assert any(m["labels"] <= 3 and m["count"] >= 5 for m in records if m["split"] == "test")
    for a, b in ((2, 1), (10, 4)):
        assert _equal(generate_puzzle(9941, a, b), generate_puzzle(9941, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    p = generate_puzzle(0, 2, 1)
    for mode in ("band", "pin", "vacated"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = bad["test"][0]
        a = _check_pair(pair)
        if mode == "band":
            q = min(a["band_cells"])
        elif mode == "pin":
            q = tuple(np.argwhere(pair["output"] == a["marker"])[0])
        else:
            q = tuple(np.argwhere((pair["input"] != a["bg"]) & (pair["output"] == a["bg"]))[0])
        pair["output"][q] = a["bg"] if mode != "vacated" else pair["input"][q]
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("semantic corruption accepted")
    # Arbitrary hue bijections cannot change any semantic role.
    remap = np.array([7, 2, 9, 0, 5, 8, 1, 6, 4, 3], dtype=np.int8)
    pair = p["train"][1]
    assert np.array_equal(solve(remap[pair["input"]]), remap[pair["output"]])
    # Changing only the program changes which complete modules land at the docks.
    pair = p["train"][0]
    changed = pair["input"].copy()
    a = _parse(changed)
    u, v = a["tokens"][1:3]
    changed[u], changed[v] = int(changed[v]), int(changed[u])
    assert not np.array_equal(solve(changed), pair["output"])
    # A hand-built grammar member outside the sampler's 1..3 gap range checks
    # longer reaches without enlarging the random generation distribution.
    modules = [
        _module((4, 4), 5, 1, 3, {1: (1, 2)}, 1, 2),
        _module((4, 13), 5, 1, 4, {3: (1, 3), 2: (2, 3)}, 1, 2),
        _module((14, 13), 5, 1, 5, {0: (2, 3)}, 1, 2),
    ]
    tokens = {(28, 2): 3, (28, 4): 4, (28, 6): 5}
    fixture = _scatter(np.random.default_rng(90210), modules, (30, 30), 28, 9, tokens)
    assert fixture is not None
    expected = np.full((30, 30), 9, dtype=np.int8)
    for m in modules:
        for q, color in m["cells"].items():
            expected[q] = color
    for q, color in tokens.items():
        expected[q] = color
    expected[4, 7:11] = 3
    expected[7:12, 13] = 4
    assert np.array_equal(solve(fixture), expected), "reach-3 / gap-4,5 member failed"
    return {"seeds": 42, "pairs": len(records), "shapes": len({m["shape"] for m in records}),
            "layouts": len({m["layout"] for m in records}), "palettes": len({m["roles"] for m in records})}
