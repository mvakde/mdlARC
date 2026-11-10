"""Mirror-exchanging mosaics: exact inventory assembly with color parity.

Only standard library and NumPy are used. No filesystem access, latent answer
state, or saved execution traces are used by generation or inference.
"""
from collections import Counter
import numpy as np


HOST_SEEDS = (0, 231598213, 330376849, 781693105, 987801600, 687277040,
              499324734, 671925432, 209492690, 721510060, 587373041,
              946041298, 92555555, 272278418, 706153252, 896266664,
              299461480, 195158842, 223771630, 2045215, 770658336,
              297095326, 986596415, 489768926, 610136091, 579219143,
              228869171, 584562473, 168866292, 746875313, 258283256,
              855613663, 169602789, 807033403, 318271225, 401629208,
              63138578, 791422240, 714389384, 226970309, 234639577,
              640988067)
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NODE_LIMIT = 20000


class SearchLimit(ValueError):
    """A timeout is rejection, never evidence of uniqueness."""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "colors must be 0..9")
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
        todo, component = [start], {start}
        while todo:
            r, c = todo.pop()
            for dr, dc in _DIRS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    todo.append(p)
        result.append(frozenset(component))
    return result


def _connected(cells):
    return bool(cells) and len(_components(cells)) == 1


def _transform(piece, op, exchange=True):
    """op=0..3 rotations; op=4..7 a reflection followed by a rotation.

    Coordinates and colors transform together. Active colors internally are
    1/2; 0 is background, 8 is neutral scaffold, and 9 marks the axis endpoints.
    """
    transformed = []
    for r, c, value in piece:
        r, c, value = int(r), int(c), int(value)
        if op >= 4:
            c = -c
            if exchange:
                value = 3 - value
        for _ in range(op % 4):
            r, c = c, -r
        transformed.append((r, c, value))
    top = min(r for r, c, v in transformed)
    left = min(c for r, c, v in transformed)
    return tuple(sorted((r - top, c - left, v) for r, c, v in transformed))


def _variants(piece, exchange=True):
    return tuple(sorted({_transform(piece, op, exchange) for op in range(8)}))


def _geometry_orbit(cells):
    return {tuple((r, c) for r, c, v in _transform([(r, c, 1) for r, c in cells], op, False))
            for op in range(8)}


def _swap_array(a):
    return np.where(a == 1, 2, np.where(a == 2, 1, a)).astype(np.int8)


def _parse(input_grid):
    """Recover visible roles and the marked crop; no construction metadata."""
    inp = _grid(input_grid)
    values, counts = np.unique(inp, return_counts=True)
    bg = int(values[np.argmax(counts)])
    _require(int(counts.max()) * 2 > inp.size, "background must occupy over half the input")
    _require(len(values) == 5, "expected background, frame, axis marker, and two active colors")
    candidates = []
    for frame in map(int, values):
        if frame == bg:
            continue
        rr, cc = np.where(inp == frame)
        top, bottom, left, right = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
        raw = inp[top:bottom, left:right]
        h, w = raw.shape
        if min(h, w) < 5:
            continue
        for horizontal in (False, True):
            b = np.rot90(raw) if horizontal else raw
            bh, bw = b.shape
            if bw % 2 != 1:
                continue
            marker = int(b[0, bw // 2])
            if marker in (bg, frame) or b[-1, bw // 2] != marker:
                continue
            if np.count_nonzero(inp == marker) != 2:
                continue
            border = np.zeros(b.shape, dtype=bool)
            border[[0, -1], :] = True
            border[:, [0, -1]] = True
            expected = np.full(b.shape, frame, dtype=np.int8)
            expected[0, bw // 2] = expected[-1, bw // 2] = marker
            if not np.array_equal(b[border], expected[border]):
                continue
            if not np.all(b[1:-1, bw // 2] == frame):
                continue
            active = sorted(set(map(int, values)) - {bg, frame, marker})
            if len(active) != 2:
                continue
            exterior = inp != bg
            exterior[top:bottom, left:right] = False
            components = _components(map(tuple, np.argwhere(exterior)))
            pieces = []
            for comp in components:
                if {int(inp[p]) for p in comp} != set(active):
                    break
                if any(not _connected({p for p in comp if inp[p] == color}) for color in active):
                    break
                piece = [(r, c, active.index(int(inp[r, c])) + 1) for r, c in comp]
                pieces.append(_transform(piece, 1 if horizontal else 0))
            else:
                if not pieces:
                    continue
                role_map = {bg: 0, active[0]: 1, active[1]: 2, frame: 8, marker: 9}
                board = np.array([[role_map[int(x)] for x in row] for row in b], dtype=np.int8)
                holes = board == 0
                if not holes.any() or sum(map(len, pieces)) != int(holes.sum()):
                    continue
                reflected = board[:, ::-1]
                fixed_both = (board != 0) & (reflected != 0)
                if not np.array_equal(board[fixed_both], _swap_array(reflected)[fixed_both]):
                    continue
                # Neutral material cannot be supplied by an active-color tile.
                if np.any(holes & (reflected >= 8)):
                    continue
                candidates.append({"input": inp, "board": board, "pieces": tuple(pieces),
                                   "roles": (bg, active[0], active[1], frame, marker),
                                   "bbox": (top, left, h, w), "horizontal": horizontal,
                                   "raw_board": raw.copy(), "components": components})
    _require(len(candidates) == 1, "input must identify exactly one framed board and axis")
    return candidates[0]


def _assembly(board, pieces, *, balance=True, exact_inventory=True, exchange=True,
              node_limit=_NODE_LIMIT, target=None, rotation_exchange=False):
    """Enumerate up to two distinct colored completions, not piece permutations.

    Placements are whole covariantly transformed pieces. Mirror requirements
    propagate during exact cover, so packing and color balance are one search.
    Search memory is bounded by placements plus a depth <= number of holes;
    there is no unbounded cache or on-disk search state.
    """
    h, w = board.shape
    holes = [tuple(map(int, p)) for p in np.argwhere(board == 0)]
    indices = {p: i for i, p in enumerate(holes)}
    # Reject collisions before allocating cell tuples. Board-stride masks have
    # exactly the same geometry as the coordinate membership check below;
    # bounding-box translation ranges prevent wrapping between rows.
    hole_bits = sum(1 << (r * w + c) for r, c in holes)
    if exact_inventory and sum(map(len, pieces)) != len(holes):
        return [], {"nodes": 0, "placements": 0, "initial_branching": 0}
    def variants(piece):
        if not rotation_exchange:
            return _variants(piece, exchange)
        result = set()
        for op in range(8):
            moved = _transform(piece, op, exchange)
            if op < 4:
                moved = tuple((r, c, 3 - x) for r, c, x in moved)
            result.add(moved)
        return tuple(sorted(result))

    # Policies that swap even the identity are not group actions. For those
    # diagnostic rivals, preserve each raw input orientation in the inventory.
    inventory = Counter((_transform(piece, 0) if rotation_exchange else min(variants(piece)))
                        for piece in pieces)
    types = sorted(inventory)
    by_cell = [[] for _ in holes]
    placements = 0
    for t, typ in enumerate(types):
        for variant in variants(typ):
            sh = max(r for r, c, x in variant) + 1
            sw = max(c for r, c, x in variant) + 1
            shape_bits = sum(1 << (dr * w + dc) for dr, dc, x in variant)
            for r in range(h - sh + 1):
                for c in range(w - sw + 1):
                    shifted = shape_bits << (r * w + c)
                    if shifted & hole_bits != shifted:
                        continue
                    cells = tuple((r + dr, c + dc, x) for dr, dc, x in variant)
                    if target is not None and any(target[a, b] != x for a, b, x in cells):
                        continue
                    mask = paint = digits = required = required_digits = 0
                    valid = True
                    for a, b, x in cells:
                        i = indices[a, b]
                        mask |= 1 << i
                        paint |= x << (4 * i)
                        digits |= 15 << (4 * i)
                        if balance:
                            mate = (a, w - 1 - b)
                            if mate in indices:
                                j = indices[mate]
                                required |= (3 - x) << (4 * j)
                                required_digits |= 15 << (4 * j)
                            elif board[mate] != 3 - x:
                                valid = False
                                break
                    if not valid or ((paint ^ required) & digits & required_digits):
                        continue
                    move = (t, mask, paint, digits, required, required_digits, cells)
                    placements += 1
                    for a, b, x in cells:
                        by_cell[indices[a, b]].append(move)
    nodes = 0
    answers = {}
    chosen = []

    def search(remaining, counts, paint, required, required_digits):
        nonlocal nodes
        nodes += 1
        if nodes > node_limit:
            raise SearchLimit("assembly proof exceeded its node budget")
        if not remaining:
            if not exact_inventory or not any(counts):
                answers.setdefault(paint, tuple(chosen))
            return
        best = None
        bits = remaining
        while bits:
            bit = bits & -bits
            i = bit.bit_length() - 1
            bits ^= bit
            options = [m for m in by_cell[i]
                       if (not exact_inventory or counts[m[0]]) and m[1] & remaining == m[1]
                       and not ((m[2] ^ required) & m[3] & required_digits)]
            if not options:
                return
            if best is None or len(options) < len(best):
                best = options
            if len(best) == 1:
                break
        for move in best:
            t, mask, p, digits, rq, rqd, cells = move
            rest = list(counts)
            if exact_inventory:
                rest[t] -= 1
            chosen.append(move)
            # Distinct occupied cells imply disjoint mirror requirements.
            search(remaining ^ mask, rest, paint | p, required | rq, required_digits | rqd)
            chosen.pop()
            # A supplied target turns this diagnostic into an existence check:
            # every completion has the same paint, so one witness suffices.
            # Input-only solving still exhausts alternatives to prove uniqueness.
            if len(answers) > 1 or (target is not None and answers):
                return

    search((1 << len(holes)) - 1, [inventory[t] for t in types], 0, 0, 0)
    result = []
    for paint, moves in answers.items():
        out = board.copy()
        for p, i in indices.items():
            out[p] = (paint >> (4 * i)) & 15
        result.append((out, moves))
    return result, {"nodes": nodes, "placements": placements,
                    "initial_branching": max(map(len, by_cell), default=0)}


def _restore(board, analysis):
    bg, a, b, frame, marker = analysis["roles"]
    mapping = {0: bg, 1: a, 2: b, 8: frame, 9: marker}
    out = np.array([[mapping[int(x)] for x in row] for row in board], dtype=np.int8)
    return np.rot90(out, -1).copy() if analysis["horizontal"] else out


def _predict(input_grid):
    analysis = _parse(input_grid)
    answers, stats = _assembly(analysis["board"], analysis["pieces"])
    _require(len(answers) == 1, "inventory and mirror exchange must give one colored answer")
    return _restore(answers[0][0], analysis), analysis, answers[0][1], stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve entirely from this input, independent of seed and generation."""
    return _predict(input_grid)[0]


def _split_region(rng, cells):
    """A procedural connected bipartition, rather than a fixed color stamp."""
    cells = set(cells)
    for _ in range(80):
        start = sorted(cells)[int(rng.integers(len(cells)))]
        first = {start}
        target = int(rng.integers(1, len(cells)))
        while len(first) < target:
            frontier = sorted({(r + dr, c + dc) for r, c in first for dr, dc in _DIRS} & cells - first)
            first.add(frontier[int(rng.integers(len(frontier)))])
        if _connected(cells - first):
            return tuple((r, c, 1 if (r, c) in first else 2) for r, c in sorted(cells))
    raise ValueError("connected color partition sampling exhausted")


def _motif(rng):
    """Two interlocking chiral pieces in an irregular shared cavity."""
    for _ in range(150):
        h, w = map(int, rng.integers(3, 6, size=2))
        rectangle = {(r, c) for r in range(h) for c in range(w)}
        notch = {(0, 0)}
        if rng.random() < 0.55:
            notch.add((0, 1) if rng.random() < 0.5 else (1, 0))
        mask = rectangle - notch
        widths = rng.integers(1, w, size=h)
        s = {(r, c) for r in range(h) for c in range(int(widths[r]))} & mask
        t = mask - s
        if min(len(s), len(t)) < 4 or not _connected(s) or not _connected(t):
            continue
        so, to = _geometry_orbit(s), _geometry_orbit(t)
        if len(so) != 8 or len(to) != 8 or min(so) == min(to):
            continue
        return _split_region(rng, s), _split_region(rng, t)
    raise ValueError("motif sampling exhausted")


def _orient_patch(s, t, phase, op):
    # Transform both pieces about one common origin, including the relative
    # attachment. Transforming them separately here would break the interlock.
    combined = [(r, c, 3 - v if phase else v, 0) for r, c, v in s]
    combined += [(r, c, v, 1) for r, c, v in t]
    moved = []
    for r, c, v, kind in combined:
        if op >= 4:
            c, v = -c, 3 - v
        for _ in range(op % 4):
            r, c = c, -r
        moved.append((r, c, v, kind))
    top = min(r for r, c, v, kind in moved)
    left = min(c for r, c, v, kind in moved)
    return tuple(tuple(sorted((r - top, c - left, v) for r, c, v, k in moved if k == kind))
                 for kind in (0, 1))


def _patch_layout(rng, patches):
    sizes = [(max(r for p in patch for r, c, v in p) + 1,
              max(c for p in patch for r, c, v in p) + 1) for patch in patches]
    n = len(patches)
    for attempt in range(40):
        if attempt < 35:
            w = int(rng.integers(max(x[1] for x in sizes), 9))
            low = max(max(x[0] for x in sizes), (sum(a * b for a, b in sizes) + w - 1) // w)
            if low > 16:
                continue
            h = int(rng.integers(low, min(16, sum(x[0] for x in sizes) + 3) + 1))
            if (h + 2) * (2 * w + 3) > 320:
                continue
            blocked = set()
            origins = [None] * n
            for i in sorted(range(n), key=lambda j: -sizes[j][0] * sizes[j][1]):
                ph, pw = sizes[i]
                choices = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)
                           if not ({(r + a, c + b) for a in range(ph) for b in range(pw)} & blocked)]
                if not choices:
                    break
                r, c = choices[int(rng.integers(len(choices)))]
                origins[i] = (r + 1, c + 1)
                blocked.update((r + a, c + b) for a in range(ph) for b in range(pw))
            else:
                return h + 2, 2 * w + 3, origins
        else:
            # A bounded geometric fallback, still using the newly sampled
            # interlocks, palettes, phases and arbitrary piece orientations.
            w = max(x[1] for x in sizes) + int(rng.integers(2))
            gaps = [int(rng.integers(2)) for _ in range(n - 1)]
            h = sum(x[0] for x in sizes) + sum(gaps)
            origins, row = [], 1
            for i, (ph, pw) in enumerate(sizes):
                origins.append((row, 1 + int(rng.integers(w - pw + 1))))
                row += ph + (gaps[i] if i < n - 1 else 0)
            return h + 2, 2 * w + 3, origins
    raise ValueError("patch packing exhausted")


def _make_board(rng, count):
    s, t = _motif(rng)
    phases = [int(rng.integers(2))]
    phases += [1 - phases[0] if i % 2 else phases[0] for i in range(1, count)]
    patches = [_orient_patch(s, t, phase, int(rng.integers(8))) for phase in phases]
    h, w, origins = _patch_layout(rng, patches)
    board = np.full((h, w), 8, dtype=np.int8)
    board[0, w // 2] = board[-1, w // 2] = 9
    output = board.copy()
    pieces = []
    for i, (patch, (top, left)) in enumerate(zip(patches, origins)):
        anchored_side = int(rng.integers(2))
        keep_other = rng.random() < 0.35
        for kind, region in enumerate(patch):
            base = [(top + r, left + c, v) for r, c, v in region]
            mate = [(r, w - 1 - c, 3 - v) for r, c, v in base]
            for side, cells in enumerate((base, mate)):
                keep = i < count - 1 and side == anchored_side and (kind == 0 or keep_other)
                for r, c, v in cells:
                    output[r, c] = v
                    board[r, c] = v if keep else 0
                if not keep:
                    pieces.append(_transform(cells, 0))
    return board, output, pieces


def _scatter(rng, board, pieces):
    h, w = board.shape
    operations = [int(rng.integers(4)), int(rng.integers(4, 8))]
    operations += list(map(int, rng.integers(8, size=len(pieces) - 2)))
    rng.shuffle(operations)
    loose = [_transform(piece, op) for piece, op in zip(pieces, operations)]
    for attempt in range(20):
        ch = 30 if attempt >= 15 else int(rng.integers(max(18, h + 1), 31))
        cw = 30 if attempt >= 15 else int(rng.integers(max(18, w + 1), 31))
        if ch * cw <= 2 * h * w:
            continue
        br, bc = int(rng.integers(ch - h + 1)), int(rng.integers(cw - w + 1))
        inp = np.zeros((ch, cw), dtype=np.int8)
        inp[br:br + h, bc:bc + w] = board
        blocked = np.zeros_like(inp, dtype=bool)
        blocked[max(0, br - 1):min(ch, br + h + 1), max(0, bc - 1):min(cw, bc + w + 1)] = True
        order = sorted(range(len(loose)), key=lambda i: (-len(loose[i]), float(rng.random())))
        for i in order:
            piece = loose[i]
            ph, pw = max(r for r, c, v in piece) + 1, max(c for r, c, v in piece) + 1
            choices = [(r, c) for r in range(ch - ph + 1) for c in range(cw - pw + 1)
                       if all(not blocked[r + dr, c + dc] for dr, dc, v in piece)]
            if not choices:
                break
            r, c = choices[int(rng.integers(len(choices)))]
            for dr, dc, v in piece:
                a, b = r + dr, c + dc
                inp[a, b] = v
                for da, db in ((0, 0),) + _DIRS:
                    if 0 <= a + da < ch and 0 <= b + db < cw:
                        blocked[a + da, b + db] = True
        else:
            return inp
    raise ValueError("loose pieces did not pack")


def _ablation_counts(analysis, target):
    board, pieces = analysis["board"], analysis["pieces"]
    no_balance, _ = _assembly(board, pieces, balance=False)
    reusable, _ = _assembly(board, pieces, exact_inventory=False)
    # Target filtering is only for demonstration analysis: prove the labeled
    # answer is impossible under the rival policy, even if that policy admits
    # many other answers. Exported solve never receives a target.
    compatible_rivals = []
    for rotation_swap, reflection_swap in ((False, False), (True, False), (True, True)):
        rival, _ = _assembly(board, pieces, exchange=reflection_swap,
                             rotation_exchange=rotation_swap, target=target)
        if rival:
            compatible_rivals.append((rotation_swap, reflection_swap))
    return no_balance, reusable, compatible_rivals


def _generate_pair(rng, count, palette):
    for _ in range(100):
        try:
            board, output, pieces = _make_board(rng, count)
            answers, _ = _assembly(board, pieces)
            if len(answers) != 1 or not np.array_equal(answers[0][0], output):
                continue
            inp = _scatter(rng, board, pieces)
            rotation = int(rng.integers(4))
            inp, output = np.rot90(inp, rotation).copy(), np.rot90(output, rotation).copy()
            lut = np.zeros(10, dtype=np.int8)
            lut[[0, 1, 2, 8, 9]] = palette
            inp, output = lut[inp], lut[output]
            predicted, analysis, moves, stats = _predict(inp)
            if not np.array_equal(predicted, output):
                continue
            target = analysis["board"].copy()
            for move in moves:
                for r, c, v in move[6]:
                    target[r, c] = v
            no_balance, reusable, no_exchange = _ablation_counts(analysis, target)
            if len(no_balance) < 2 or len(reusable) < 2:
                continue
            if no_exchange:
                continue
            return {"input": inp, "output": output}
        except (ValueError, SearchLimit):
            continue
    raise ValueError("bounded unique mosaic construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    _require(_integer(num_train) and 2 <= num_train <= 10, "num_train must be 2..10")
    _require(_integer(num_test) and 1 <= num_test <= 4, "num_test must be 1..4")
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    stable_palette = bool(rng.random() < 0.3)
    palette = rng.permutation(10)[:5]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", int(num_train)), ("test", int(num_test))):
        for i in range(number):
            # The first two demonstrations expose two-module assemblies. Later
            # demonstrations can already recombine three; tests favor three.
            count = (2 if i < 2 else int(rng.choice([2, 2, 3]))) if split == "train" else (3 if i == 0 else int(rng.choice([2, 3])))
            colors = palette if stable_palette else rng.permutation(10)[:5]
            for _ in range(8):
                pair = _generate_pair(rng, count, colors)
                key = _signature(pair["input"])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise ValueError("could not construct distinct inputs")
    rng.shuffle(puzzle["train"])
    return puzzle


def _analyze_pair(pair, ablations=False):
    inp, output = _grid(pair["input"]), _grid(pair["output"])
    predicted, analysis, moves, stats = _predict(inp)
    _require(np.array_equal(predicted, output), "output differs from unique input-only solution")
    raw = analysis["raw_board"]
    bg = analysis["roles"][0]
    _require(raw.shape == output.shape, "output must include exactly the complete frame")
    fixed = raw != bg
    _require(np.array_equal(raw[fixed], output[fixed]), "fixed board cells changed")
    _require(not np.any(output == bg), "board still contains background")
    _require(int(np.count_nonzero(inp != bg)) == output.size, "total exact inventory area changed")
    board = analysis["board"]
    normalized_out = np.rot90(output) if analysis["horizontal"] else output
    a, b = analysis["roles"][1:3]
    converted = np.where(normalized_out == a, 1, np.where(normalized_out == b, 2,
                         np.where(normalized_out == analysis["roles"][3], 8, 9)))
    _require(np.array_equal(converted, _swap_array(converted[:, ::-1])), "mirror colors do not exchange")
    occupied = set()
    cavity_multicolor = 0
    gap_components = _components(map(tuple, np.argwhere(board == 0)))
    for component in gap_components:
        if len({int(converted[p]) for p in component}) == 2:
            cavity_multicolor += 1
    for move in moves:
        cells = {(r, c) for r, c, v in move[6]}
        _require(not (cells & occupied), "overlapping pieces")
        occupied.update(cells)
    _require(len(occupied) == int(np.count_nonzero(board == 0)), "inventory did not cover all holes")
    shared = any(sum(bool({(r, c) for r, c, v in move[6]} & comp) for move in moves) > 1
                 for comp in gap_components)
    _require(shared, "need a cavity shared by multiple inventory pieces")
    _require(stats["initial_branching"] > 1, "all placements were locally forced")
    if ablations:
        no_balance, reusable, no_exchange = _ablation_counts(analysis, converted)
        _require(len(no_balance) == 2, "global mirror rule must rule out a competing answer")
        _require(len(reusable) == 2, "finite inventory must rule out a competing answer")
        _require(not no_exchange, "the labeled answer must exclude every rival parity-to-swap policy")
    return {"board_shape": output.shape, "canvas_shape": inp.shape,
            "piece_count": len(analysis["pieces"]), "horizontal": analysis["horizontal"],
            "background": bg, "roles": analysis["roles"], "gap_count": len(gap_components),
            "shared_cavity": shared, "multicolor_cavities": cavity_multicolor,
            "piece_areas": tuple(sorted(set(map(len, analysis["pieces"])))),
            "geometries": tuple(min(_geometry_orbit({(r, c) for r, c, v in p})) for p in analysis["pieces"]),
            "layout": (board.shape, (board == 0).tobytes()), **stats}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid pair counts")
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            key = _signature(_grid(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _analyze_pair(pair, ablations=True)


def self_test():
    """42 supplied seeds, bounded semantic ablations and diversity checks.

    Keeps one puzzle at a time. Returns compact measurements for optional
    recording; never writes files or retains generated payloads globally.
    """
    shapes, canvases, layouts, geometries, piece_counts, backgrounds, axes = set(), set(), set(), set(), set(), set(), set()
    signatures, roles, gap_counts, areas = set(), set(), set(), set()
    total = max_nodes = max_placements = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _analyze_pair(pair)
                total += 1
                shapes.add(f["board_shape"])
                canvases.add(f["canvas_shape"])
                layouts.add(f["layout"])
                geometries.update(f["geometries"])
                piece_counts.add(f["piece_count"])
                backgrounds.add(f["background"])
                axes.add(f["horizontal"])
                roles.add(f["roles"])
                gap_counts.add(f["gap_count"])
                areas.update(f["piece_areas"])
                max_nodes = max(max_nodes, f["nodes"])
                max_placements = max(max_placements, f["placements"])
    _require(len(signatures) == 42 and len(layouts) >= 100, "insufficient structural diversity")
    _require(len(shapes) >= 30 and len(canvases) >= 40, "insufficient size diversity")
    _require(len(geometries) >= 40 and len(areas) >= 8, "insufficient interlock diversity")
    _require(len(piece_counts) >= 4 and len(gap_counts) >= 3, "insufficient inventory/cavity variation")
    _require(axes == {False, True} and backgrounds == set(range(10)), "missing axis or background roles")
    first = generate_puzzle(123456, 2, 1)
    generate_puzzle(654321, 2, 1)
    second = generate_puzzle(123456, 2, 1)
    _require(all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], second[s])
                 for k in ("input", "output")), "seed does not reproduce the whole puzzle")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Removing an inventory cell violates exact conservation; changing a fixed
    # output cell violates preservation even when the remaining colors look apt.
    pair = first["test"][0]
    analysis = _parse(pair["input"])
    missing = pair["input"].copy()
    missing[min(analysis["components"][0])] = analysis["roles"][0]
    try:
        solve(missing)
    except ValueError:
        pass
    else:
        raise AssertionError("missing inventory cell was accepted")
    damaged = {"input": pair["input"], "output": pair["output"].copy()}
    damaged["output"][0, 0] = analysis["roles"][0]
    try:
        _analyze_pair(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("changed fixed cell was accepted")
    # A genuine two-color ambiguity is rejected, but identical piece swaps do
    # not count as different answers. These small fixtures isolate exact cover.
    fixture = np.array([[0, 8, 0]], dtype=np.int8)
    ambiguous, _ = _assembly(fixture, [((0, 0, 1),), ((0, 0, 2),)], balance=False, exchange=False)
    _require(len(ambiguous) == 2, "color ambiguity not detected")
    identical, _ = _assembly(fixture, [((0, 0, 1),), ((0, 0, 1),)], balance=False, exchange=False)
    _require(len(identical) == 1, "identical-piece exchange counted as ambiguity")
    return {"seeds": 42, "pairs": total, "puzzles": len(signatures), "board_shapes": len(shapes),
            "canvas_shapes": len(canvases), "hole_layouts": len(layouts),
            "piece_geometries": len(geometries), "piece_counts": sorted(piece_counts),
            "gap_counts": sorted(gap_counts), "piece_areas": sorted(areas),
            "backgrounds": sorted(backgrounds), "axis_orientations": len(axes),
            "palettes_with_roles": len(roles), "maximum_solver_nodes": max_nodes,
            "maximum_candidate_placements": max_placements,
            "ablations_checked": ["no_global_exchange_symmetry", "reusable_inventory", "all_three_rival_parity_policies"],
            "independent_solver": False}
