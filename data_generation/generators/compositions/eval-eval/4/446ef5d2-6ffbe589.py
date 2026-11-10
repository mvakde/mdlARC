"""Contact clocks: an anchored square-tile jigsaw whose joined rims turn its cores.

Only NumPy and the standard library are used. There is no filesystem access,
example lookup, seed dependency in solve, cache, or saved execution trace.
"""
from __future__ import annotations

import hashlib
import numpy as np


_CORNERS = ("TL", "TR", "BL", "BR")


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return g.astype(np.int8, copy=True)


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        p = min(todo)
        todo.remove(p)
        cells = [p]
        for r, c in cells:
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in todo:
                    todo.remove(q)
                    cells.append(q)
        result.append(cells)
    return result


def _marker(point, corner):
    r, c = point
    dr = -1 if corner[0] == "T" else 1
    dc = -1 if corner[1] == "L" else 1
    return ((r+dr, c), (r, c+dc), (r+dr, c+dc))


def _parse(g):
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    if np.sum(counts == counts.max()) != 1:
        raise ValueError("background must be the unique most frequent input color")
    candidates = []
    for marker_color in range(10):
        coords = np.argwhere(g == marker_color)
        if marker_color == bg or len(coords) != 3:
            continue
        low, high = coords.min(0), coords.max(0)
        if tuple(high-low) != (1, 1):
            continue
        missing = {(r, c) for r in range(int(low[0]), int(high[0])+1)
                   for c in range(int(low[1]), int(high[1])+1)} - set(map(tuple, coords.tolist()))
        point = missing.pop()
        if g[point] in (bg, marker_color):
            continue
        corner = {(1, 1): "TL", (1, 0): "TR", (0, 1): "BL", (0, 0): "BR"}[
            (point[0]-int(low[0]), point[1]-int(low[1]))]
        tiles, anchor, carrier = [], None, None
        good = True
        for cells in _components((g != bg) & (g != marker_color)):
            q = np.array(cells)
            r, c = map(int, q.min(0))
            h, w = map(int, q.max(0)-q.min(0)+1)
            if h != w or h < 5 or len(cells) != h*w:
                good = False
                break
            tile = g[r:r+h, c:c+w].copy()
            corners = tile[[0, 0, -1, -1], [0, -1, 0, -1]]
            if len(set(corners.tolist())) != 1:
                good = False
                break
            if carrier is None:
                carrier = int(corners[0])
            if int(corners[0]) != carrier:
                good = False
                break
            interior_colors = set(map(int, np.unique(tile[1:-1, 1:-1]))) - {carrier}
            rim_colors = set(map(int, np.concatenate((tile[0], tile[-1], tile[1:-1, 0], tile[1:-1, -1])))) - {carrier}
            if not interior_colors or not rim_colors <= interior_colors:
                good = False
                break
            if point in cells:
                expected = (r+(h-1 if corner[0] == "B" else 0),
                            c+(w-1 if corner[1] == "R" else 0))
                if point != expected:
                    good = False
                    break
                anchor = len(tiles)
            tiles.append((tile, (r, c)))
        if good and anchor is not None and 2 <= len(tiles) <= 6:
            candidates.append(dict(bg=bg, marker=marker_color, point=point,
                                   corner=corner, carrier=carrier, tiles=tiles, anchor=anchor))
    if len(candidates) != 1:
        raise ValueError("expected one identifiable external L and square-tile collection")
    return candidates[0]


def _assemblies(info, canvas, match=True, limit=2):
    """Complete row-major exact cover, stopping only after a witnessed ambiguity.

    At the earliest unfilled cell a new solid square must start there. The fixed
    anchor is preplaced, so all four marker corners use the same enumeration.
    A node budget raises an error; it is never interpreted as uniqueness.
    """
    tiles, anchor = info["tiles"], info["anchor"]
    area = sum(t.size for t, _ in tiles)
    largest = max(len(t) for t, _ in tiles)
    result, nodes = [], 0
    for h in range(largest, canvas[0]+1):
        if area % h:
            continue
        w = area//h
        if not largest <= w <= canvas[1]:
            continue
        oy = info["point"][0]-(h-1 if info["corner"][0] == "B" else 0)
        ox = info["point"][1]-(w-1 if info["corner"][1] == "R" else 0)
        if oy < 0 or ox < 0 or oy+h > canvas[0] or ox+w > canvas[1]:
            continue
        board = np.full((h, w), -1, dtype=np.int8)
        labels = np.full((h, w), -1, dtype=np.int8)
        ar, ac = tiles[anchor][1][0]-oy, tiles[anchor][1][1]-ox
        n = len(tiles[anchor][0])
        if ar < 0 or ac < 0 or ar+n > h or ac+n > w:
            continue
        board[ar:ar+n, ac:ac+n] = tiles[anchor][0]
        labels[ar:ar+n, ac:ac+n] = anchor
        positions = [None]*len(tiles)
        positions[anchor] = (ar, ac)

        def search(remaining):
            nonlocal nodes
            nodes += 1
            if nodes > 60000:
                raise ValueError("assembly search budget exhausted, uniqueness not established")
            if not remaining:
                result.append(((oy, ox), board.copy(), labels.copy(), tuple(positions)))
                return len(result) >= limit
            r, c = map(int, np.argwhere(board < 0)[0])
            for i in remaining:
                tile = tiles[i][0]
                n = len(tile)
                if r+n > h or c+n > w or np.any(board[r:r+n, c:c+n] >= 0):
                    continue
                if match:
                    checks = []
                    if r:
                        checks.append((board[r-1, c:c+n], tile[0]))
                    if c:
                        checks.append((board[r:r+n, c-1], tile[:, 0]))
                    if r+n < h:
                        checks.append((board[r+n, c:c+n], tile[-1]))
                    if c+n < w:
                        checks.append((board[r:r+n, c+n], tile[:, -1]))
                    if any(np.any((a >= 0) & (a != b)) for a, b in checks):
                        continue
                board[r:r+n, c:c+n] = tile
                labels[r:r+n, c:c+n] = i
                positions[i] = (r, c)
                if search(tuple(j for j in remaining if j != i)):
                    return True
                board[r:r+n, c:c+n] = -1
                labels[r:r+n, c:c+n] = -1
                positions[i] = None
            return False

        if search(tuple(i for i in range(len(tiles)) if i != anchor)):
            break
    return result


def _contact_counts(info, assembly):
    _, board, labels, _ = assembly
    counts = [{} for _ in info["tiles"]]
    for dr, dc in ((1, 0), (0, 1)):
        for r in range(board.shape[0]-dr):
            for c in range(board.shape[1]-dc):
                a, b = int(labels[r, c]), int(labels[r+dr, c+dc])
                color = int(board[r, c])
                if a != b and color != info["carrier"]:
                    if color != int(board[r+dr, c+dc]):
                        raise ValueError("unequal facing rim cells")
                    for tile in (a, b):
                        counts[tile][color] = counts[tile].get(color, 0)+1
    return counts


def _bar_counts(info, assembly):
    """Counterfactual: count active monochrome runs rather than their cells."""
    _, board, labels, positions = assembly
    result = [{} for _ in info["tiles"]]
    for i, ((tile, _), (r, c)) in enumerate(zip(info["tiles"], positions)):
        n = len(tile)
        for side in range(4):
            previous = None
            for k in range(n):
                y, x, dy, dx = ((r, c+k, -1, 0), (r+k, c+n-1, 0, 1),
                                (r+n-1, c+k, 1, 0), (r+k, c, 0, -1))[side]
                yy, xx = y+dy, x+dx
                color = int(board[y, x])
                active = (0 <= yy < len(board) and 0 <= xx < board.shape[1]
                          and int(labels[yy, xx]) != i and color != info["carrier"])
                if active and color != previous:
                    result[i][color] = result[i].get(color, 0)+1
                previous = color if active else None
    return result


def _render(info, assembly, canvas, coordinate=False, all_rims=False, no_turn=False, bar_turn=False):
    origin, board, _, positions = assembly
    counts = _bar_counts(info, assembly) if bar_turn else _contact_counts(info, assembly)
    result = np.full(canvas, info["bg"], dtype=np.int8)
    oy, ox = origin
    for i, ((tile, _), (r, c)) in enumerate(zip(info["tiles"], positions)):
        changed = tile.copy()
        core = tile[1:-1, 1:-1]
        target = np.full(core.shape, info["carrier"], dtype=np.int8)
        occupied = np.zeros(core.shape, dtype=bool)
        for color in np.unique(core):
            color = int(color)
            if color == info["carrier"]:
                continue
            k = counts[i].get(color, 0)
            if all_rims:
                rim = np.concatenate((tile[0], tile[-1], tile[1:-1, 0], tile[1:-1, -1]))
                k = int(np.sum(rim == color))
            if no_turn:
                k = 0
            if coordinate:
                mask = np.zeros(core.shape, dtype=bool)
                for y, x in np.argwhere(core == color):
                    y, x = int(y), int(x)
                    for _ in range(k % 4):
                        y, x = x, len(core)-1-y
                    mask[y, x] = True
            else:
                mask = np.rot90(core == color, -(k % 4))
            if np.any(occupied & mask):
                raise ValueError("rotated interior layers overlap")
            target[mask] = color
            occupied |= mask
        changed[1:-1, 1:-1] = target
        n = len(tile)
        result[oy+r:oy+r+n, ox+c:ox+c+n] = changed
    return result


def _analyze(value):
    g = _grid(value)
    info = _parse(g)
    candidates = _assemblies(info, g.shape)
    if len(candidates) != 1:
        raise ValueError("tile translations must have a unique exact-rim rectangular assembly")
    return g, info, candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct neighbors and all local clocks entirely from the input."""
    g, info, assembly = _analyze(input_grid)
    return _render(info, assembly, g.shape)


def _orbits(n):
    unseen = {(r, c) for r in range(n) for c in range(n)}
    result = []
    while unseen:
        r, c = min(unseen)
        orbit = []
        for _ in range(4):
            if (r, c) not in orbit:
                orbit.append((r, c))
            r, c = c, n-1-r
        unseen.difference_update(orbit)
        result.append(orbit)
    return result


def _core(rng, n, carrier, colors, asymmetric):
    orbits = _orbits(n)
    four = [o for o in orbits if len(o) == 4]
    rng.shuffle(four)
    owners = [[] for _ in colors]
    for i, orbit in enumerate(four):
        owner = i if i < len(colors) else int(rng.integers(len(colors)))
        owners[owner].append(orbit)
    out = np.full((n, n), carrier, dtype=np.int8)
    for color, territory in zip(colors, owners):
        symmetry = 0 if asymmetric else int(rng.choice(3, p=(.55, .2, .25)))
        density = float(rng.uniform(.25, .85))
        for j, orbit in enumerate(territory):
            start = int(rng.integers(4))
            if symmetry == 2:
                chosen = orbit if j == 0 or rng.random() < density else []
            elif symmetry == 1:
                chosen = [orbit[start], orbit[(start+2) % 4]]
                if j and rng.random() < density*.5:
                    chosen = orbit
            else:
                # The first orbit is a guaranteed asymmetric witness; other
                # orbits add independent area, density, and connectivity changes.
                amount = 1 if j == 0 else int(rng.integers(0, 5))
                chosen = [orbit[(start+k) % 4] for k in range(amount)]
            for r, c in chosen:
                out[r, c] = color
    if n % 2 and rng.random() < .5:
        out[n//2, n//2] = int(rng.choice([carrier]+colors))
    return out


def _partition(rng, mixed):
    if mixed:
        rows, cols = [(2, 3), (3, 2), (3, 3), (2, 4), (4, 2)][int(rng.integers(5))]
        r, c = int(rng.integers(rows-1)), int(rng.integers(cols-1))
        squares = [(r, c, 2)]
        squares += [(y, x, 1) for y in range(rows) for x in range(cols)
                    if not (r <= y < r+2 and c <= x < c+2)]
    else:
        rows, cols = [(1, 4), (4, 1), (2, 2), (2, 3), (3, 2)][int(rng.integers(5))]
        squares = [(y, x, 1) for y in range(rows) for x in range(cols)]
    return rows, cols, squares


def _make_tiles(rng, mixed, witness):
    base = int(rng.integers(5, 7 if mixed else 8))
    rows, cols, squares = _partition(rng, mixed)
    palette = list(map(int, rng.choice(10, 5 if base == 5 else int(rng.integers(5, 7)), replace=False)))
    bg, marker, carrier, *colors = palette
    squares = [(r*base, c*base, n*base) for r, c, n in squares]
    tiles = [np.full((n, n), carrier, dtype=np.int8) for _, _, n in squares]
    labels = np.full((rows*base, cols*base), -1, dtype=np.int8)
    for i, (r, c, n) in enumerate(squares):
        labels[r:r+n, c:c+n] = i
        tiles[i][1:-1, 1:-1] = _core(rng, n-2, carrier, colors, witness >= 0)
    # Each physical contact segment gets one same-colored bar. Its facing cells
    # are equal; endpoints stay carrier to accommodate a neighbor's corners.
    seams = {}
    for r in range(len(labels)):
        for c in range(labels.shape[1]):
            for dr, dc in ((1, 0), (0, 1)):
                y, x = r+dr, c+dc
                if y < len(labels) and x < labels.shape[1] and labels[r, c] != labels[y, x]:
                    key = (int(labels[r, c]), int(labels[y, x]), dr, dc)
                    seams.setdefault(key, []).append(((r, c), (y, x)))
    for contacts in seams.values():
        usable = []
        for a, b in contacts:
            ends = []
            for r, c in (a, b):
                i = int(labels[r, c])
                y, x, n = squares[i]
                ends.append((i, r-y, c-x))
            if all(not (r in (0, len(tiles[i])-1) and c in (0, len(tiles[i])-1)) for i, r, c in ends):
                usable.append(ends)
        length = int(rng.integers(1, len(usable)+1))
        start = int(rng.integers(len(usable)-length+1))
        color = int(rng.choice(colors))
        for ends in usable[start:start+length]:
            for i, r, c in ends:
                tiles[i][r, c] = color
    # Exterior bars are intentional inactive controls, not decoration/noise.
    for i, (r, c, n) in enumerate(squares):
        for side, exterior in enumerate((r == 0, c+n == labels.shape[1], r+n == len(labels), c == 0)):
            if not exterior or rng.random() < .25:
                continue
            length = int(rng.integers(1, n-1))
            start = int(rng.integers(1, n-length))
            color = int(rng.choice(colors))
            if side == 0:
                tiles[i][0, start:start+length] = color
            elif side == 1:
                tiles[i][start:start+length, -1] = color
            elif side == 2:
                tiles[i][-1, start:start+length] = color
            else:
                tiles[i][start:start+length, 0] = color
    board = np.full(labels.shape, carrier, dtype=np.int8)
    for tile, (r, c, n) in zip(tiles, squares):
        board[r:r+n, c:c+n] = tile
    return bg, marker, carrier, colors, squares, tiles, board, labels


def _features(info, assembly):
    counts = _contact_counts(info, assembly)
    raw, residues = set(), set()
    independent = unmarked = cycle = wrap = False
    for (tile, _), clock in zip(info["tiles"], counts):
        informative = []
        for color in np.unique(tile[1:-1, 1:-1]):
            if color == info["carrier"]:
                continue
            mask = tile[1:-1, 1:-1] == color
            if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
                continue
            k = clock.get(int(color), 0)
            raw.add(k)
            residues.add(k % 4)
            informative.append(k % 4)
            unmarked |= k == 0
            cycle |= k > 0 and k % 4 == 0
            wrap |= k > 4 and k % 4 != 0
        independent |= len(set(informative)) >= 2
    return dict(raw=raw, residues=residues, independent=independent,
                unmarked=unmarked, cycle=cycle, wrap=wrap)


def _scatter(rng, tiles, anchor, point, corner, canvas):
    n = len(tiles[anchor])
    anchor_pos = (point[0]-(n-1 if corner[0] == "B" else 0),
                  point[1]-(n-1 if corner[1] == "R" else 0))
    remaining = sorted((i for i in range(len(tiles)) if i != anchor), key=lambda i: -len(tiles[i]))
    for _ in range(18):
        positions = [None]*len(tiles)
        positions[anchor] = anchor_pos
        blocked = np.zeros(canvas, dtype=bool)

        def block(r, c, size):
            blocked[max(0, r-1):min(canvas[0], r+size+1), max(0, c-1):min(canvas[1], c+size+1)] = True

        block(*anchor_pos, n)
        for r, c in _marker(point, corner):
            block(r, c, 1)
        success = True
        for i in remaining:
            n = len(tiles[i])
            prefix = np.pad(blocked.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            sums = prefix[n:, n:]-prefix[:-n, n:]-prefix[n:, :-n]+prefix[:-n, :-n]
            options = np.argwhere(sums == 0)
            if not len(options):
                success = False
                break
            r, c = map(int, options[int(rng.integers(len(options)))])
            positions[i] = (r, c)
            block(r, c, n)
        if success:
            return positions
    return None


def _pair(rng, corner, mixed, witness):
    for _ in range(180):
        bg, marker, carrier, colors, squares, tiles, board, labels = _make_tiles(rng, mixed, witness)
        h, w = board.shape
        local = (h-1 if corner[0] == "B" else 0, w-1 if corner[1] == "R" else 0)
        anchor = int(labels[local])
        prototype = dict(carrier=carrier, tiles=[(t, (r, c)) for t, (r, c, _) in zip(tiles, squares)])
        assembly = ((0, 0), board, labels, tuple((r, c) for r, c, _ in squares))
        features = _features(prototype, assembly)
        if witness == 0 and not ({1, 2, 3} <= features["raw"] and features["independent"] and features["unmarked"]):
            continue
        if witness == 1 and not (features["cycle"] and features["wrap"]):
            continue
        # Qualify clock evidence before doing layout or assembly searches.
        for _ in range(5):
            canvas = (int(rng.integers(max(18, h+2), 31)), int(rng.integers(max(18, w+2), 31)))
            oy = int(rng.integers(1, canvas[0]-h))
            ox = int(rng.integers(1, canvas[1]-w))
            point = (oy+local[0], ox+local[1])
            positions = _scatter(rng, tiles, anchor, point, corner, canvas)
            if positions is None:
                continue
            inp = np.full(canvas, bg, dtype=np.int8)
            for tile, (r, c) in zip(tiles, positions):
                n = len(tile)
                inp[r:r+n, c:c+n] = tile
            for p in _marker(point, corner):
                inp[p] = marker
            # Labels are made from the intended geometry, then independently
            # reconstructed from the scattered input; ambiguity rejects a draw.
            info = dict(bg=bg, marker=marker, carrier=carrier, tiles=list(zip(tiles, positions)),
                        anchor=anchor, point=point, corner=corner)
            expected_assembly = ((oy, ox), board, labels, assembly[3])
            expected = _render(info, expected_assembly, canvas)
            try:
                _, recovered, assembled = _analyze(inp)
                actual = _render(recovered, assembled, canvas)
            except ValueError:
                continue
            if not np.array_equal(actual, expected):
                continue
            if np.array_equal(actual, _render(recovered, assembled, canvas, all_rims=True)):
                continue
            if np.array_equal(actual, _render(recovered, assembled, canvas, no_turn=True)):
                continue
            if witness == 1 and np.array_equal(actual, _render(recovered, assembled, canvas, bar_turn=True)):
                continue
            return {"input": inp, "output": expected}
    raise RuntimeError("bounded contact-clock construction exhausted")


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("counts require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(int(seed))
    train_corners, test_corners = (("TL", "BR"), ("TR", "BL")) if rng.integers(2) else (("TR", "BL"), ("TL", "BR"))
    if rng.integers(2):
        train_corners = train_corners[::-1]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            witness = index if split == "train" and index < 2 else -1
            mixed = witness == 1 if witness >= 0 else bool(rng.integers(2))
            corners = train_corners if split == "train" else test_corners
            corner = corners[index] if split == "train" and index < 2 else str(rng.choice(corners))
            for _ in range(8):
                pair = _pair(rng, corner, mixed, witness)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must have exactly train/test")
    if (not all(isinstance(puzzle[s], list) for s in puzzle)
            or not 2 <= len(puzzle["train"]) <= 10 or not 1 <= len(puzzle["test"]) <= 4
            or len(puzzle["train"]) <= len(puzzle["test"])):
        raise ValueError("illegal pair counts")
    seen, evidence, corners = set(), [], {"train": set(), "test": set()}
    cell_count_witness = False
    training_geometries = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise ValueError("pairs require exactly input/output")
            inp, output = _grid(pair["input"]), _grid(pair["output"])
            if inp.shape != output.shape:
                raise ValueError("output must retain the input canvas")
            key = inp.shape, inp.tobytes()
            if key in seen:
                raise ValueError("duplicate input")
            seen.add(key)
            _, info, assembly = _analyze(inp)
            expected = _render(info, assembly, inp.shape, coordinate=True)
            if not np.array_equal(output, expected):
                raise ValueError("incorrect anchored mosaic or local clock rotations")
            # Explicit conservation includes unchanged rims, each interior layer,
            # marker erasure, and all vacated cells returned to background.
            cin, cout = np.bincount(inp.ravel(), minlength=10), np.bincount(output.ravel(), minlength=10)
            if cout[info["marker"]] or cout[info["bg"]] != cin[info["bg"]]+3:
                raise ValueError("marker erasure/background conservation failed")
            if any(cin[c] != cout[c] for c in range(10) if c not in (info["bg"], info["marker"])):
                raise ValueError("payload color area changed")
            anchor = info["anchor"]
            local = assembly[3][anchor]
            if (local[0]+assembly[0][0], local[1]+assembly[0][1]) != info["tiles"][anchor][1]:
                raise ValueError("anchor moved")
            if np.array_equal(output, _render(info, assembly, inp.shape, all_rims=True)):
                raise ValueError("no witness for contact activation")
            if np.array_equal(output, _render(info, assembly, inp.shape, no_turn=True)):
                raise ValueError("no witness for rotation")
            corners[split].add(info["corner"])
            if split == "train":
                evidence.append(_features(info, assembly))
                training_geometries.add(len({len(t) for t, _ in info["tiles"]}) > 1)
                cell_count_witness |= not np.array_equal(output, _render(info, assembly, inp.shape, bar_turn=True))
    if not ({1, 2, 3} <= set.union(*(e["raw"] for e in evidence))
            and all(any(e[name] for e in evidence) for name in ("independent", "unmarked", "cycle", "wrap"))
            and cell_count_witness):
        raise ValueError("training does not establish the required clock conventions")
    if len(corners["train"]) != 2 or corners["train"] & corners["test"]:
        raise ValueError("missing complementary-corner generalization")
    if training_geometries != {False, True}:
        raise ValueError("training must demonstrate equal and mixed tile sizes")


def self_test():
    """Stream 42 seeds; retain hashes and feature sets, never a sample corpus."""
    fingerprints, shapes, structures, sizes, tile_counts, palettes = set(), set(), set(), set(), set(), set()
    raw_counts, symmetries, corners = set(), set(), set()
    mixed = subedges = geometry_ambiguous = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                digest.update(bytes(inp.shape)+inp.tobytes()+out.tobytes())
                _, info, assembly = _analyze(inp)
                assert np.array_equal(solve(inp.copy()), out)
                assert np.array_equal(_render(info, assembly, inp.shape, coordinate=True), out)
                shapes.add(inp.shape)
                ns = tuple(sorted(len(t) for t, _ in info["tiles"]))
                sizes.update(ns)
                tile_counts.add(len(ns))
                palettes.add((info["bg"], info["marker"], info["carrier"]))
                corners.add(info["corner"])
                mixed += len(set(ns)) > 1
                if len(set(ns)) > 1:
                    subedges += 1
                raw_counts.update(k for clock in _contact_counts(info, assembly) for k in clock.values())
                rename = {}
                normalized = bytes(rename.setdefault(int(v), len(rename)) for v in assembly[1].ravel())
                structures.add(hashlib.sha256(bytes(assembly[1].shape)+normalized).digest())
                for tile, _ in info["tiles"]:
                    for color in np.unique(tile[1:-1, 1:-1]):
                        if color == info["carrier"]:
                            continue
                        mask = tile[1:-1, 1:-1] == color
                        symmetries.add("quarter" if np.array_equal(mask, np.rot90(mask)) else
                                       "half" if np.array_equal(mask, np.rot90(mask, 2)) else "asymmetric")
                if len(_assemblies(info, inp.shape, match=False)) >= 2:
                    geometry_ambiguous += 1
        fingerprints.add(digest.digest())
    assert len(fingerprints) == 42
    assert len(shapes) >= 25 and len(structures) >= 150 and len(palettes) >= 100
    assert {5, 6, 7, 10, 12} <= sizes and {3, 4, 5, 6} <= tile_counts
    assert corners == set(_CORNERS) and symmetries == {"asymmetric", "half", "quarter"}
    assert mixed >= 42 and subedges >= 42 and geometry_ambiguous >= 100
    assert {1, 2, 3, 4, 5} <= raw_counts and max(raw_counts) >= 8
    original = generate_puzzle(41, 2, 1)
    repeated = generate_puzzle(41, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in original for a, b in zip(original[s], repeated[s]) for k in a)
    # Traversal order and literal colors do not choose a translation or clock.
    inp = original["test"][0]["input"]
    _, info, assembly = _analyze(inp)
    reordered = dict(info, tiles=info["tiles"][::-1], anchor=len(info["tiles"])-1-info["anchor"])
    candidates = _assemblies(reordered, inp.shape)
    assert len(candidates) == 1
    assert np.array_equal(_render(reordered, candidates[0], inp.shape), original["test"][0]["output"])
    rename = np.array([3, 7, 2, 9, 1, 8, 4, 0, 6, 5], dtype=np.int8)
    assert np.array_equal(solve(rename[inp]), rename[original["test"][0]["output"]])
    # Plain equal rims cannot distinguish the permutations of three free tiles.
    ambiguous = np.zeros((18, 18), dtype=np.int8)
    for i, (r, c) in enumerate(((2, 2), (2, 10), (10, 2), (10, 10))):
        ambiguous[r:r+5, c:c+5] = 1
        ambiguous[r+1+i//2, c+1+i%2] = 2
    for point in _marker((2, 2), "TL"):
        ambiguous[point] = 9
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous jigsaw accepted")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(700+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Validators must reject an incorrect label and duplicated demonstrations.
    for corruption in ("cell", "duplicate"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
        if corruption == "cell":
            bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
        else:
            bad["train"][1] = {k: v.copy() for k, v in bad["train"][0].items()}
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupted puzzle accepted")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "canvas_shapes": len(shapes),
            "normalized_mosaics": len(structures), "mixed_size_pairs": mixed,
            "geometry_ambiguous_pairs": geometry_ambiguous, "tile_sizes": sorted(sizes),
            "tile_counts": sorted(tile_counts), "clock_counts": sorted(raw_counts)}
