"""Assembly from branching ray exits: oriented inventory and boundary constraints.

Only NumPy and the standard library are used. No parent generator is imported.
The solver discovers all roles and constraints from a single input.
"""
from collections import Counter, defaultdict
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "Grids must be two-dimensional with dimensions 1..30")
    _require(np.issubdtype(a.dtype, np.integer)
             and np.all((a >= 0) & (a <= 9)), "Invalid grid colors or dtype")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _normalize(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _parse(input_grid):
    """Find the wall, the stub side, material roles, holes, and loose pieces.

    Zero is the fixed canvas/missing-cell convention. All other colors are
    geometric roles, not a learned palette. Rotation is undone only internally.
    """
    inp = _grid(input_grid)
    candidates = []
    for wall in map(int, np.unique(inp)):
        if wall == 0:
            continue
        rr, cc = np.where(inp == wall)
        top, left = int(rr.min()), int(cc.min())
        bottom, right = int(rr.max())+1, int(cc.max())+1
        crop = inp[top:bottom, left:right]
        if min(crop.shape) < 5:
            continue
        perimeter = np.zeros(crop.shape, dtype=bool)
        perimeter[0] = perimeter[-1] = True
        perimeter[:, 0] = perimeter[:, -1] = True
        if np.any((crop == wall) & ~perimeter):
            continue
        if any(crop[p] != wall for p in
               ((0, 0), (0, -1), (-1, 0), (-1, -1))):
            continue
        outside = inp != 0
        outside[top:bottom, left:right] = False
        comps = _components(outside)
        if not 2 <= len(comps) <= 8:
            continue
        if any(len({int(inp[p]) for p in comp}) != 1 for comp in comps):
            continue
        for rotation in range(4):
            a = np.rot90(crop, -rotation).copy()
            if not (np.all(a[:, 0] == wall) and np.all(a[:, -1] == wall)):
                continue
            entry_cells = np.flatnonzero(a[0, 1:-1] != wall)
            exit_cells = np.flatnonzero(a[-1, 1:-1] != wall)
            if not len(entry_cells) or not len(exit_cells):
                continue
            guides = set(map(int, a[0, 1:-1][entry_cells]))
            goals = set(map(int, a[-1, 1:-1][exit_cells]))
            if len(guides) != 1 or len(goals) != 1:
                continue
            guide, goal = next(iter(guides)), next(iter(goals))
            if len({0, wall, guide, goal}) != 4:
                continue
            board = a[1:-1, 1:-1].copy()
            stub_mask = np.zeros(board.shape, dtype=bool)
            lengths = []
            for c in entry_cells:
                rows = np.flatnonzero(board[:, c] == guide)
                if not 1 <= len(rows) <= 2 or not np.array_equal(rows, np.arange(len(rows))):
                    break
                stub_mask[:len(rows), c] = True
                lengths.append(int(len(rows)+1))  # Includes the wall port.
            else:
                if not np.array_equal(stub_mask, board == guide):
                    continue
                material_colors = set(map(int, np.unique(board))) - {0, guide}
                if len(material_colors) != 2 or material_colors & {wall, goal}:
                    continue
                counts = {c: int(np.count_nonzero(board == c)) for c in material_colors}
                floor = max(counts, key=counts.get)
                if counts[floor] <= board.size / 2:
                    continue
                solid = next(c for c in material_colors if c != floor)
                pieces = []
                for comp in comps:
                    color = int(inp[comp[0]])
                    if color not in (floor, solid):
                        break
                    # Piece orientation must rotate with the whole scene.
                    local = np.zeros(inp.shape, dtype=bool)
                    for p in comp:
                        local[p] = True
                    shape = _normalize(list(map(tuple, np.argwhere(np.rot90(local, -rotation)))))
                    if (len(shape) > 12 or max(r for r, c in shape) >= 4
                            or max(c for r, c in shape) >= 5):
                        break
                    pieces.append((color, shape))
                else:
                    if sum(len(s) for color, s in pieces) != np.count_nonzero(board == 0):
                        continue
                    if not 1 <= np.count_nonzero(board == 0) <= 64:
                        continue
                    candidates.append(dict(input=inp, crop=crop.copy(), canonical=a,
                        bbox=(top, left, bottom-top, right-left), rotation=rotation,
                        wall=wall, floor=floor, solid=solid, guide=guide, goal=goal,
                        board=board, pieces=pieces, exterior=comps,
                        entries=tuple(map(int, entry_cells)), exits=tuple(map(int, exit_cells)),
                        stub_lengths=tuple(lengths)))
    _require(len(candidates) == 1, "Input must expose exactly one chamber and direction")
    return candidates[0]


class _ConstructionLimit(Exception):
    pass


def _assemblies(board, pieces, *, node_limit=None, answer_limit=None):
    """Enumerate exact translation covers; identical pieces are counted.

    Deduplicate colored boards, rather than counting exchanges or same-color
    decompositions as different answers. Limits are for proposals only; solve
    never calls this function with a computational cutoff.
    """
    holes = [tuple(map(int, p)) for p in np.argwhere(board == 0)]
    if sum(len(s) for color, s in pieces) != len(holes):
        return []
    h, w = board.shape
    index = {p: i for i, p in enumerate(holes)}
    inventory = Counter(pieces)
    types = sorted(inventory)
    moves = [[] for p in holes]
    for t, (color, shape) in enumerate(types):
        sh, sw = max(r for r, c in shape)+1, max(c for r, c in shape)+1
        for r in range(h-sh+1):
            for c in range(w-sw+1):
                cells = tuple((r+dr, c+dc) for dr, dc in shape)
                if all(p in index for p in cells):
                    ids = tuple(index[p] for p in cells)
                    mask = sum(1 << i for i in ids)
                    move = (t, mask, cells)
                    for i in ids:
                        moves[i].append(move)
    results, work, placed = {}, board.copy(), []
    nodes = 0

    def search(remaining, counts):
        nonlocal nodes
        nodes += 1
        if node_limit is not None and nodes > node_limit:
            raise _ConstructionLimit("Exact cover proposal exceeds node budget")
        if not remaining:
            if any(counts):
                return
            results.setdefault(work.tobytes(), (work.copy(), tuple(placed)))
            if answer_limit is not None and len(results) > answer_limit:
                raise _ConstructionLimit("Too many colored proposal assemblies")
            return
        best, bits = None, remaining
        while bits:
            bit = bits & -bits
            i = bit.bit_length()-1
            bits ^= bit
            choices = [m for m in moves[i]
                       if counts[m[0]] and m[1] & remaining == m[1]]
            if not choices:
                return
            if best is None or len(choices) < len(best):
                best = choices
            if len(best) == 1:
                break
        for t, mask, cells in best:
            color, shape = types[t]
            counts[t] -= 1
            placed.append((color, shape, cells))
            for p in cells:
                work[p] = color
            search(remaining ^ mask, counts)
            for p in cells:
                work[p] = 0
            placed.pop()
            counts[t] += 1

    search((1 << len(holes))-1, [inventory[t] for t in types])
    return list(results.values())


def _runs(row, solid):
    c, result = 0, []
    while c < len(row):
        if row[c] != solid:
            c += 1
            continue
        a = c
        while c+1 < len(row) and row[c+1] == solid:
            c += 1
        result.append((a, c))
        c += 1
    return result


def _ray(board, entries, solid, guide, variant=None):
    """Immutable-material layer propagation. Ports lie outside this interior."""
    h, w = board.shape
    out, active = board.copy(), set(entries)
    reached, events = set(), []
    stats = dict(branch=0, clipped=0, simultaneous=0, multi_hit=0,
                 interrupted=0, merge=0, singleton=0)
    for r in range(h):
        runs = _runs(board[r], solid)
        hits, following = {}, set()
        for c in sorted(active):
            if board[r, c] == solid and variant != "straight":
                run = next((a, b) for a, b in runs if a <= c <= b)
                hits.setdefault(run, []).append(c)
            else:
                following.add(c)
                if board[r, c] != solid:
                    out[r, c] = guide
        stats["simultaneous"] += len(hits) > 1
        if variant == "all_obstacles":
            hits = {run: hits.get(run, []) for run in runs}
        elif variant == "one_per_layer":
            hits = dict(list(sorted(hits.items()))[:1])
        elif variant == "ignore_singletons":
            hits = {run: columns for run, columns in hits.items() if run[0] != run[1]}
        for (a, b), incoming in hits.items():
            reached.add((r, a, b))
            events.append((r, a, b))
            stats["branch"] += a > 0 and b < w-1
            stats["clipped"] += a == 0 or b == w-1
            stats["multi_hit"] += len(incoming) > 1
            stats["singleton"] += a == b
            if r:
                stats["interrupted"] += bool(np.any(board[r-1, max(0, a-1):min(w, b+2)] == solid))
                for c in range(max(0, a-1), min(w, b+2)):
                    if board[r-1, c] != solid:
                        out[r-1, c] = guide
            for c in (a-1, b+1):
                if 0 <= c < w:
                    stats["merge"] += c in following
                    following.add(c)
                    out[r, c] = guide
        active = following
    all_runs = {(r, a, b) for r in range(h) for a, b in _runs(board[r], solid)}
    stats.update(events=len(events), hit_runs=tuple(events), inert=len(all_runs-reached),
                 reached_lengths=tuple(b-a+1 for r, a, b in events),
                 solid_links=int(np.count_nonzero((board[:-1] == solid) & (board[1:] == solid))))
    return out, tuple(sorted(active)), stats


def _certified(scene):
    assemblies = _assemblies(scene["board"], scene["pieces"])
    matching, excess_exit_witness = [], False
    for board, placements in assemblies:
        trace, exits, stats = _ray(board, scene["entries"], scene["solid"], scene["guide"])
        if exits == scene["exits"]:
            matching.append((board, placements, trace, stats))
        excess_exit_witness |= set(scene["exits"]) < set(exits)
    _require(len(matching) == 1, "Inventory and exits must certify one colored assembly")
    matching[0][3]["excess_exit_witness"] = excess_exit_witness
    return matching[0], len(assemblies)


def solve(input_grid: np.ndarray) -> np.ndarray:
    scene = _parse(input_grid)
    (board, placements, trace, stats), count = _certified(scene)
    out = scene["canonical"].copy()
    out[1:-1, 1:-1] = trace
    return np.rot90(out, scene["rotation"]).copy()


def _inventory_alternative(board, placements, pieces, entries, exits, solid, floor, guide):
    """A visible B witness: template reuse permits another exit-correct answer."""
    supplied = set(pieces)
    for color, shape, cells in placements:
        other = floor if color == solid else solid
        if (other, shape) not in supplied:
            continue
        changed = board.copy()
        for p in cells:
            changed[p] = other
        trace, actual, _ = _ray(changed, entries, solid, guide)
        if actual == exits:
            return True
    return False


def _gap_metrics(partial, complete):
    comps = _components(partial == 0)
    return dict(gaps=len(comps), shared_gaps=sum(
        len({int(complete[p]) for p in comp}) > 1 for comp in comps))


def _feature_set(stats):
    return {f for f in ("clipped", "simultaneous", "multi_hit", "interrupted", "merge", "singleton")
            if stats[f]}


def _evidenced_features(board, entries, solid, guide, trace, stats):
    flags = _feature_set(stats)
    for name, variant in (("simultaneous", "one_per_layer"),
                          ("singleton", "ignore_singletons")):
        if name in flags and np.array_equal(_ray(board, entries, solid, guide, variant)[0], trace):
            flags.remove(name)
    return flags


def _shape(rng, symmetric):
    for attempt in range(16):
        kind = int(rng.integers(4 if symmetric else 7))
        if kind == 0:
            s = {(0, c) for c in range(int(rng.integers(1, 6)))}
        elif kind == 1:
            s = {(r, 0) for r in range(int(rng.integers(2, 5)))}
        elif kind == 2:
            a, b = int(rng.integers(1, 4)), int(rng.integers(2, 5))
            s = {(r, c) for r in range(a) for c in range(b)}
        elif kind == 3:
            n = int(rng.integers(2, 5))
            s = {(0, c) for c in range(3)} | {(r, 1) for r in range(n)}
        elif kind == 4:
            n = int(rng.integers(2, 5))
            s = {(r, 0) for r in range(n)} | {(n-1, c) for c in range(3)}
        elif kind == 5:
            s = {(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)}
        else:
            s = {(0, 0)}
            for _ in range(int(rng.integers(2, 7))):
                options = sorted({(r+dr, c+dc) for r, c in s
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))} - s)
                s.add(options[int(rng.integers(len(options)))])
        if not symmetric:
            for _ in range(int(rng.integers(4))):
                s = {(c, -r) for r, c in s}
        shape = _normalize(s)
        if max(r for r, c in shape) < 4 and max(c for r, c in shape) < 5:
            return shape
    return ((0, 0), (0, 1))


def _proposal(rng, pairs, entries_count, larger, need_excess=False):
    h = int(rng.integers(8, 15 if larger else 12))
    w = int(rng.integers(9, 17 if larger else 15))
    symmetric = bool(rng.random() < .30)
    board, used, pieces = np.ones((h, w), dtype=np.int8), set(), []
    for i in range(pairs):
        shape = _shape(rng, symmetric)
        sh, sw = max(r for r, c in shape)+1, max(c for r, c in shape)+1
        if sh > h-3 or sw > w//2-1:
            return None
        for attempt in range(80):
            r, c = int(rng.integers(2, h-sh)), int(rng.integers(w-sw+1))
            if symmetric:
                origins = [(r, c), (r, w-sw-c)]
            elif rng.random() < .4:
                if c+2*sw > w:
                    continue
                origins = [(r, c), (r, c+sw)]
            else:
                origins = [(r, c), (int(rng.integers(2, h-sh)), int(rng.integers(w-sw+1)))]
            cells = [{(r+dr, c+dc) for dr, dc in shape} for r, c in origins]
            if cells[0] & cells[1] or (cells[0] | cells[1]) & used:
                continue
            used |= cells[0] | cells[1]
            for p in cells[0] | cells[1]:
                board[p] = 0
            pieces.extend([(1, shape), (2, shape)])
            break
        else:
            return None
    if len(used) > 64:
        return None
    # Fixed bars and connected stems interact with the *assembled* material.
    for i in range(int(rng.integers(1, 5))):
        r, c = int(rng.integers(2, h)), int(rng.integers(w))
        n = int(rng.integers(1, min(7, w-c)+1))
        cells = {(r, x) for x in range(c, c+n)}
        if rng.random() < .35:
            cells |= {(y, c) for y in range(max(2, r-2), r+1)}
        if symmetric:
            cells |= {(r, w-1-c) for r, c in list(cells)}
        if cells & used:
            continue
        for p in cells:
            board[p] = 2
    if not np.any(board == 2) or np.mean(board == 1) < .60:
        return None
    entries = tuple(sorted(map(int, rng.choice(w, entries_count, replace=False))))
    for c in entries:
        board[:int(rng.integers(1, 3)), c] = 3
    if np.mean(board == 1) <= .5:
        return None
    try:
        assemblies = _assemblies(board, pieces, node_limit=100000, answer_limit=4096)
    except _ConstructionLimit:
        return None
    if len(assemblies) < 2:
        return None
    groups = defaultdict(list)
    for complete, placements in assemblies:
        trace, exits, stats = _ray(complete, entries, 2, 3)
        groups[exits].append((complete, placements, trace, stats))
    options = [(exits, values[0]) for exits, values in sorted(groups.items())
               if exits and len(values) == 1 and values[0][3]["branch"]
               and values[0][3]["events"] >= pairs
               and (not need_excess or any(set(exits) < set(other) for other in groups))]
    if not options:
        return None
    exits, (complete, placements, trace, stats) = options[int(rng.integers(len(options)))]
    return dict(board=board, pieces=pieces, entries=entries, exits=exits,
                complete=complete, placements=placements, trace=trace, stats=stats,
                assembly_count=len(assemblies),
                excess_exit_witness=any(set(exits) < set(other) for other in groups))


def _fallback_checksum(rng):
    """Shared-cavity witness against accepting extra unmarked exits.

    A transparent near domino preserves the main ray; moving the solid domino
    there makes a detour that later recovers that ray *and* leaves extra exits.
    The supplied horizontal triple separately determines a clipped later gate.
    """
    h, w = 13+int(rng.integers(3)), 13+int(rng.integers(3))
    board = np.ones((h, w), dtype=np.int8)
    board[:int(rng.integers(1, 3)), 2] = 3
    board[2, 2:4] = 2
    board[7, 3] = 2
    board[5, 4:8] = 0
    board[10, :3] = board[10, 9:12] = 0
    complete = board.copy()
    complete[5, 4:6], complete[5, 6:8] = 1, 2
    complete[10, :3], complete[10, 9:12] = 2, 1
    domino, triple = ((0, 0), (0, 1)), ((0, 0), (0, 1), (0, 2))
    pieces = [(1, domino), (2, domino), (1, triple), (2, triple)]
    trace, exits, stats = _ray(complete, (2,), 2, 3)
    assemblies = _assemblies(board, pieces)
    matching = [(b, p) for b, p in assemblies if _ray(b, (2,), 2, 3)[1] == exits]
    excess = any(set(exits) < set(_ray(b, (2,), 2, 3)[1]) for b, p in assemblies)
    _require(len(matching) == 1 and np.array_equal(matching[0][0], complete)
             and excess, "Constructive checksum witness must be unique")
    return dict(board=board, pieces=pieces, entries=(2,), exits=exits,
        complete=complete, placements=matching[0][1], trace=trace,
        stats=stats, assembly_count=len(assemblies), excess_exit_witness=excess)


def _fallback(rng, pairs, shared=False):
    """A single serial cascade, with a proof-readable inventory dependency.

    Each successful gate leaves one permanent flank and advances the frontier
    by two columns. The exit fingerprint requires every near gate to be solid;
    the finite inventory consequently makes every remote gate transparent.
    """
    if shared and pairs == 2:
        return _fallback_checksum(rng)
    h, w = 3*pairs+4+int(rng.integers(3)), 2*pairs+9+int(rng.integers(3))
    board = np.ones((h, w), dtype=np.int8)
    entry = 2
    board[:int(rng.integers(1, 3)), entry] = 3
    board[1, w-2:w] = 2  # Explicit inert fixed obstacle and material swatch.
    shape = ((0, 0), (0, 1))
    pieces, complete = [], board.copy()
    for i in range(pairs):
        r, c = 3*i+3, entry+2*i
        remote = c+2 if shared and i == 0 else c+4
        board[r, c:c+2] = board[r, remote:remote+2] = 0
        complete[r, c:c+2] = 2
        complete[r, remote:remote+2] = 1
        pieces.extend([(1, shape), (2, shape)])
    trace, exits, stats = _ray(complete, (entry,), 2, 3)
    assemblies = _assemblies(board, pieces)
    matching = [(b, p) for b, p in assemblies if _ray(b, (entry,), 2, 3)[1] == exits]
    _require(len(matching) == 1 and np.array_equal(matching[0][0], complete),
             "Constructive cascade must be certified")
    return dict(board=board, pieces=pieces, entries=(entry,), exits=exits,
        complete=complete, placements=matching[0][1], trace=trace,
        stats=stats, assembly_count=len(assemblies),
        excess_exit_witness=any(set(exits) < set(_ray(b, (entry,), 2, 3)[1]) for b, p in assemblies))


def _scatter(rng, chamber, pieces):
    bh, bw = chamber.shape
    order = sorted(range(len(pieces)), key=lambda i: (-len(pieces[i][1]), i))
    for attempt in range(7):
        if attempt == 6:
            # Guaranteed shelves for the generated bounding boxes. Full board
            # occupies the upper-left; loose pieces fit in the lower strip.
            canvas = np.zeros((30, 30), dtype=np.int8)
            canvas[:bh, :bw] = chamber
            r, c, shelf_height = bh+1, 0, 0
            for i in order:
                color, shape = pieces[i]
                sh, sw = max(r for r, c in shape)+1, max(c for r, c in shape)+1
                if c+sw > 30:
                    r, c, shelf_height = r+shelf_height+1, 0, 0
                _require(r+sh <= 30, "Generated inventory exceeds shelf capacity")
                for dr, dc in shape:
                    canvas[r+dr, c+dc] = color
                c += sw+1
                shelf_height = max(shelf_height, sh)
            return canvas
        ch = int(rng.integers(max(20, bh+4), 31))
        cw = int(rng.integers(max(20, bw+4), 31))
        br, bc = int(rng.integers(ch-bh+1)), int(rng.integers(cw-bw+1))
        canvas, blocked = np.zeros((ch, cw), dtype=np.int8), np.zeros((ch, cw), dtype=bool)
        canvas[br:br+bh, bc:bc+bw] = chamber
        blocked[max(0, br-1):min(ch, br+bh+1), max(0, bc-1):min(cw, bc+bw+1)] = True
        for i in order:
            color, shape = pieces[i]
            sh, sw = max(r for r, c in shape)+1, max(c for r, c in shape)+1
            choices = [(r, c) for r in range(ch-sh+1) for c in range(cw-sw+1)
                       if all(not blocked[r+dr, c+dc] for dr, dc in shape)]
            if not choices:
                break
            r, c = choices[int(rng.integers(len(choices)))]
            for dr, dc in shape:
                rr, cc = r+dr, c+dc
                canvas[rr, cc] = color
                for y, x in ((rr, cc), (rr-1, cc), (rr+1, cc), (rr, cc-1), (rr, cc+1)):
                    if 0 <= y < ch and 0 <= x < cw:
                        blocked[y, x] = True
        else:
            return canvas
    raise AssertionError("Unreachable scatter exhaustion")


def _render_candidate(rng, candidate, palette, rotation):
    # Canonical colors: 0 canvas/hole, 1 floor, 2 solid, 3 guide, 4 wall, 5 exit.
    board = candidate["board"]
    h, w = board.shape
    chamber = np.full((h+2, w+2), 4, dtype=np.int8)
    chamber[1:-1, 1:-1] = board
    chamber[0, [c+1 for c in candidate["entries"]]] = 3
    chamber[-1, [c+1 for c in candidate["exits"]]] = 5
    output = chamber.copy()
    output[1:-1, 1:-1] = candidate["trace"]
    canvas = _scatter(rng, chamber, candidate["pieces"])
    mapping = np.array((0, *palette), dtype=np.int8)
    return dict(input=np.rot90(mapping[canvas], rotation).copy(),
                output=np.rot90(mapping[output], rotation).copy())


def _acceptable(candidate, witness, allowed):
    if candidate is None:
        return False
    b, stats = candidate["complete"], candidate["stats"]
    if allowed is not None and not _feature_set(stats) <= allowed:
        return False
    if witness == "inventory":
        if not stats["inert"] or not _inventory_alternative(b, candidate["placements"],
                candidate["pieces"], candidate["entries"], candidate["exits"], 2, 1, 3):
            return False
        if np.array_equal(_ray(b, candidate["entries"], 2, 3, "all_obstacles")[0], candidate["trace"]):
            return False
    if witness == "shared":
        if (stats["events"] < 2 or not _gap_metrics(candidate["board"], b)["shared_gaps"]
                or not candidate["excess_exit_witness"]):
            return False
    return True


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name in ((seed, "seed"), (num_train, "num_train"), (num_test, "num_test")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name+" must be an integer")
    if seed < 0 or not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("Invalid seed or example counts")
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.choice(np.arange(1, 10), 5, replace=False)))
    exchange = bool(rng.random() < .65)
    rotation = int(rng.integers(4))
    mixed_directions = bool(rng.random() < .5)
    puzzle, seen, covered, maximum_entries = {"train": [], "test": []}, set(), set(), 1
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for i in range(count):
            witness = ("inventory" if i == 0 else "shared") if split == "train" and i < 2 else None
            pairs = (1 if i == 0 else 2) if split == "train" else int(rng.choice([3, 4], p=[.75, .25]))
            if split == "train" and i >= 2:
                pairs = int(rng.integers(1, 3))
            entries_count = 1 if witness == "inventory" else int(rng.integers(1, 4))
            if witness == "shared":
                entries_count = int(rng.integers(2, 4))
            if split == "test":
                entries_count = int(rng.integers(1, maximum_entries+1))
            allowed = covered if split == "test" else None
            candidate = None
            for attempt in range(260):
                proposed = _proposal(rng, pairs, entries_count, split == "test", need_excess=witness == "shared")
                if _acceptable(proposed, witness, allowed):
                    candidate = proposed
                    break
            if candidate is None:
                candidate = _fallback(rng, pairs, shared=witness == "shared")
                _require(_acceptable(candidate, witness, allowed), "Fallback lost a demonstration invariant")
            local_palette = tuple(map(int, rng.choice(np.arange(1, 10), 5, replace=False))) if exchange else palette
            local_rotation = (rotation+i+int(split == "test")) % 4 if mixed_directions else rotation
            # Scattering is irrelevant to solving and can safely be resampled.
            for attempt in range(30):
                pair = _render_candidate(rng, candidate, local_palette, local_rotation)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise AssertionError("Unable to scatter a distinct input")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train":
                maximum_entries = max(maximum_entries, len(candidate["entries"]))
                covered |= _evidenced_features(candidate["complete"], candidate["entries"],
                    2, 3, candidate["trace"], candidate["stats"])
    return puzzle


def _analyze_pair(pair):
    scene = _parse(pair["input"])
    (board, placements, trace, stats), assembly_count = _certified(scene)
    output = _grid(pair["output"])
    expected = scene["canonical"].copy()
    expected[1:-1, 1:-1] = trace
    _require(np.array_equal(output, np.rot90(expected, scene["rotation"])), "Incorrect certified assembly or trace")
    partial = scene["board"]
    _require(np.all(board[partial != 0] == partial[partial != 0]), "Fixed chamber cells changed during assembly")
    _require(not np.any(board == 0), "Unfilled chamber cell")
    _require(np.all(trace[board == scene["solid"]] == scene["solid"]), "Solid cells overwritten by rays")
    _require(np.all(trace[partial == scene["guide"]] == scene["guide"]), "Entry stub overwritten")
    _require(assembly_count >= 2, "Rays must resolve a real material ambiguity")
    _require(stats["branch"] > 0, "Missing two-flank collision")
    _require(Counter((color, shape) for color, shape, cells in placements) == Counter(scene["pieces"]),
             "Piece multiplicities changed")
    for color in (scene["floor"], scene["solid"]):
        _require(np.count_nonzero(board[partial == 0] == color) ==
                 sum(len(s) for c, s in scene["pieces"] if c == color), "Material mass is not conserved")
    _require(not np.array_equal(_ray(board, scene["entries"], scene["solid"], scene["guide"], "straight")[0], trace),
             "Straight rays accidentally explain the answer")
    inventory_witness = _inventory_alternative(board, placements, scene["pieces"],
        scene["entries"], scene["exits"], scene["solid"], scene["floor"], scene["guide"])
    stats.update(_gap_metrics(partial, board))
    stats["mixed_hit_runs"] = sum(any(partial[r, c] == 0 for c in range(a, b+1))
        and any(partial[r, c] == scene["solid"] for c in range(a, b+1))
        for r, a, b in stats["hit_runs"])
    stats.update(assembly_count=assembly_count, piece_count=len(scene["pieces"]),
                 inventory_witness=inventory_witness, direction=scene["rotation"],
                 board_shape=board.shape, canvas_shape=scene["input"].shape,
                 entries=len(scene["entries"]), exits=len(scene["exits"]),
                 shape_signatures=tuple(s for c, s in scene["pieces"]),
                 palette=(scene["floor"], scene["solid"], scene["guide"], scene["wall"], scene["goal"]),
                 symmetric_scaffold=np.array_equal(np.where(partial == scene["guide"], scene["floor"], partial),
                     np.where(partial == scene["guide"], scene["floor"], partial)[:, ::-1]),
                 evidence=_evidenced_features(board, scene["entries"], scene["solid"], scene["guide"], trace, stats),
                 all_obstacles_wrong=not np.array_equal(_ray(board, scene["entries"], scene["solid"], scene["guide"], "all_obstacles")[0], trace),
                 layout_key=(partial.shape, np.where(partial == 0, 0, np.where(partial == scene["solid"], 2,
                     np.where(partial == scene["guide"], 3, 1))).tobytes()))
    return stats


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "Illegal pair counts")
    seen, covered, training = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair schema")
            a, o = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(a) not in seen, "Duplicate input")
            seen.add(_key(a))
            _require(a.size > o.size, "Output must extract the chamber")
            stats = _analyze_pair(pair)
            if split == "train":
                covered |= stats["evidence"]
                training.append(stats)
            else:
                _require(_feature_set(stats) <= covered, "Undemonstrated collision case in test")
                _require(stats["piece_count"] > max(s["piece_count"] for s in training), "Missing inventory-size shift")
                _require(stats["entries"] <= max(s["entries"] for s in training), "Undemonstrated entry count")
    _require(any(s["inventory_witness"] and s["all_obstacles_wrong"] and s["inert"] for s in training),
             "Training must distinguish exact inventory and causal reachability")
    _require(any(s["shared_gaps"] and s["events"] >= 2 for s in training),
             "Training must show a shared multicolor cavity and a collision cascade")
    _require(any(s["excess_exit_witness"] for s in training),
             "Training must distinguish exact exits from merely reaching marked exits")


def self_test():
    """Bounded 42-seed variation and semantic counterexample checks."""
    signatures, layouts, shapes, canvases, piece_shapes = set(), set(), set(), set(), set()
    directions, entry_counts, exit_counts, pieces, gaps, assembly_counts = set(), set(), set(), set(), set(), set()
    role_colors = [set() for _ in range(5)]
    totals = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(_grid(p[k])) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output")))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "Input-only solve regression")
                s = _analyze_pair(pair)
                layouts.add(s["layout_key"])
                shapes.add(s["board_shape"])
                canvases.add(s["canvas_shape"])
                piece_shapes.update(s["shape_signatures"])
                directions.add(s["direction"])
                entry_counts.add(s["entries"])
                exit_counts.add(s["exits"])
                pieces.add(s["piece_count"])
                gaps.add(s["gaps"])
                assembly_counts.add(s["assembly_count"])
                for i, color in enumerate(s["palette"]):
                    role_colors[i].add(color)
                for flag in ("clipped", "simultaneous", "multi_hit", "interrupted", "merge", "singleton",
                             "shared_gaps", "solid_links", "symmetric_scaffold", "inventory_witness",
                             "excess_exit_witness", "mixed_hit_runs"):
                    totals[flag] += bool(s[flag])
    _require(len(signatures) == 42 and len(layouts) >= 120, "Structural seed diversity collapsed")
    _require(len(shapes) >= 35 and len(canvases) >= 40 and len(piece_shapes) >= 25, "Geometry vocabulary collapsed")
    _require(directions == set(range(4)) and entry_counts == {1, 2, 3}, "Direction or entry variation collapsed")
    _require(pieces == {2, 4, 6, 8} and len(gaps) >= 5 and len(assembly_counts) >= 5 and len(exit_counts) >= 5,
             "Inventory, topology, ambiguity or exit variation collapsed")
    _require(all(colors == set(range(1, 10)) for colors in role_colors), "Material/marker roles lack palette variation")
    _require(all(totals[f] > 0 for f in totals), "Missing observed mechanism")
    # Identical-piece swaps are one colored assembly; different materials are two.
    slot = np.zeros((1, 2), dtype=np.int8)
    dot = ((0, 0),)
    _require(len(_assemblies(slot, [(1, dot), (1, dot)])) == 1, "Identical pieces falsely ambiguous")
    _require(len(_assemblies(slot, [(1, dot), (2, dot)])) == 2, "Material swap ambiguity missed")
    t = ((0, 0), (0, 1), (0, 2), (1, 1))
    reflected = ((0, 1), (1, 0), (1, 1), (1, 2))
    slot = np.full((2, 3), 1, dtype=np.int8)
    for p in t:
        slot[p] = 0
    _require(not _assemblies(slot, [(2, reflected)]), "Piece was illegally reflected")
    # Concrete collision: entry 3 hits [2,4], upstream span [1,5], exits {1,5}.
    b = np.ones((5, 7), dtype=np.int8)
    b[2, 2:5] = 2
    trace, exits, s = _ray(b, (3,), 2, 3)
    _require(exits == (1, 5) and np.all(trace[1, 1:6] == 3)
             and np.all(trace[2, 2:5] == 2), "Branch physics regression")
    # Two disjoint hits, a shared flank, and a subsequent same-run multi-hit.
    b = np.ones((8, 11), dtype=np.int8)
    b[2, 2:5] = b[2, 6:9] = 2
    b[5, 1:6] = 2
    trace, exits, s = _ray(b, (3, 7), 2, 3)
    _require(s["simultaneous"] and s["merge"] and s["multi_hit"], "Coupled branching regression")
    _require(not np.array_equal(trace, _ray(b, (3, 7), 2, 3, "one_per_layer")[0]), "Simultaneous witness redundant")
    # Force the constructive recovery route for all inventory sizes.
    rng = np.random.default_rng(9181)
    for n in range(1, 5):
        c = _fallback(rng, n, shared=n == 2)
        pair = _render_candidate(rng, c, (7, 2, 6, 8, 3), n % 4)
        _require(np.array_equal(solve(pair["input"]), pair["output"]), "Fallback solve regression")
        _require(_acceptable(c, "shared" if n == 2 else "inventory", None), "Fallback witness regression")
    # Check determinism even after intervening calls and a previous answer edit.
    a = generate_puzzle(171, 2, 1)
    fingerprint = tuple(_key(p[k]) for s in a for p in a[s] for k in p)
    a["train"][0]["output"][:] = 0
    generate_puzzle(172, 2, 1)
    b = generate_puzzle(171, 2, 1)
    _require(fingerprint == tuple(_key(p[k]) for s in b for p in b[s] for k in p), "Seed reproducibility failed")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # A corrupted fixed barrier and an inventory deletion must be rejected.
    p = generate_puzzle(19, 2, 1)
    for corruption in ("fixed", "missing"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = bad["train"][0]
        scene = _parse(pair["input"])
        if corruption == "missing":
            pair["input"][scene["exterior"][0][0]] = 0
        else:
            canonical = np.rot90(pair["output"], -scene["rotation"]).copy()
            r, c = map(int, np.argwhere(scene["board"] == scene["solid"])[0])
            canonical[r+1, c+1] = scene["floor"]
            pair["output"] = np.rot90(canonical, scene["rotation"]).copy()
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Validator accepted corruption")
