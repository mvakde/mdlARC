"""Registered zipper: geometric keys propagate a boundary registration.

Only Python's standard library and NumPy are used. Construction records are
discarded; solve and validation recover pieces and the complete chain from input.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    array = np.asarray(value)
    _require(array.ndim == 2 and all(1 <= n <= 30 for n in array.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(array.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((0 <= array) & (array <= 9))), "illegal grid colors")
    return array.astype(np.int8, copy=True)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(grid, background):
    remaining = {tuple(map(int, p)) for p in np.argwhere(grid != background)}
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        found, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _normalized(rows):
    return tuple(r - rows[0] for r in rows) if rows else ()


def _chain_orders(pieces, start, end, count_only=False):
    """Return at most two all-piece directed chains; repeats are not tie-broken."""
    def signature(rows):
        return len(rows) if count_only else _normalized(rows)

    answers = []

    def visit(order, remaining):
        if len(answers) == 2:
            return
        if not remaining:
            if order[-1] == end:
                answers.append(tuple(order))
            return
        outgoing = pieces[order[-1]]["exit"]
        if not outgoing:
            return
        for j in remaining:
            if signature(pieces[j]["entry"]) == signature(outgoing):
                visit(order + [j], tuple(k for k in remaining if k != j))

    visit([start], tuple(j for j in range(len(pieces)) if j != start))
    return answers


def _decode(input_grid):
    original = _grid(input_grid)
    h, w = original.shape
    corners = {int(original[r, c]) for r, c in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1))}
    _require(len(corners) == 1, "corners must identify one background")
    bg = corners.pop()
    boundary = [(r, c) for r, c in np.argwhere(original != bg)
                if r in (0, h-1) or c in (0, w-1)]
    _require(len(boundary) == 1, "exactly one noncorner boundary anchor is required")
    r, c = map(int, boundary[0])
    _require(r not in (0, h-1) or c not in (0, w-1), "corner anchor")
    turns = 0 if c == 0 else 1 if r == 0 else 2 if c == w-1 else 3
    g = np.rot90(original, turns).copy()
    labels = np.rot90(np.arange(original.size).reshape(original.shape), turns)
    guide = int(np.flatnonzero(g[:, 0] != bg)[0])
    marker = int(g[guide, 0])
    comps = _components(g, bg)
    _require({(guide, 0)} in comps, "anchor must be isolated")
    comps = [p for p in comps if p != {(guide, 0)}]
    _require(2 <= len(comps) <= 6, "need 2..6 pieces")
    pieces = []
    for points in comps:
        payload = [p for p in points if int(g[p]) != marker]
        _require(payload, "piece without a carrier")
        top, left = min(r for r, _ in payload), min(c for _, c in payload)
        bottom, right = max(r for r, _ in payload), max(c for _, c in payload)
        ph, pw = bottom-top+1, right-left+1
        _require(ph >= 3 and pw >= 2, "carrier core too small")
        teeth, internal = {"entry": [], "exit": []}, []
        for rr, cc in points:
            if int(g[rr, cc]) == marker:
                if top <= rr <= bottom and left <= cc <= right:
                    internal.append((rr-top, cc-left))
                elif top <= rr <= bottom and cc == left-1:
                    teeth["entry"].append(rr-top)
                elif top <= rr <= bottom and cc == right+1:
                    teeth["exit"].append(rr-top)
                else:
                    raise AssertionError("tooth is not on a longitudinal core side")
            else:
                _require(top <= rr <= bottom and left <= cc <= right, "payload outside core")
        perimeter = {(rr, cc) for rr in range(top, bottom+1) for cc in range(left, right+1)
                     if rr in (top, bottom) or cc in (left, right)}
        _require(perimeter <= points, "carrier perimeter must be complete")
        _require(len(internal) <= 1, "multiple internal registrations")
        if internal:
            ir, ic = internal[0]
            _require(pw >= 3 and not (ir in (0, ph-1) and ic in (0, pw-1)),
                     "registration cannot replace a carrier corner")
        orig_positions = [divmod(int(labels[p]), w) for p in points]
        pieces.append(dict(top=top, left=left, h=ph, w=pw,
                           entry=tuple(sorted(teeth["entry"])),
                           exit=tuple(sorted(teeth["exit"])),
                           registration=internal[0] if internal else None,
                           cells=tuple((rr-top, cc-left, int(g[rr, cc])) for rr, cc in sorted(points)),
                           source_keys=(min(int(labels[p]) for p in points),
                                        min(cc for _, cc in orig_positions),
                                        min(rr for rr, _ in orig_positions))))
    starts = [j for j, p in enumerate(pieces) if not p["entry"] and p["exit"]]
    ends = [j for j, p in enumerate(pieces) if p["entry"] and not p["exit"]]
    _require(len(starts) == len(ends) == 1, "need exactly two directed terminals")
    start, end = starts[0], ends[0]
    _require(pieces[start]["registration"] is not None, "start needs an internal registration")
    _require(all(p["registration"] is None for j, p in enumerate(pieces) if j != start),
             "only the starting terminal is internally registered")
    _require(all(p["entry"] and p["exit"] for j, p in enumerate(pieces) if j not in (start, end)),
             "intermediate pieces need both faces")
    orders = _chain_orders(pieces, start, end)
    _require(len(orders) == 1, "the full geometric chain must be unique")
    _require(pieces[start]["left"] >= 2, "start is too close to the anchor")
    return dict(grid=g, background=bg, marker=marker, guide=guide, turns=turns,
                pieces=pieces, start=start, end=end, order=orders[0])


def _placements(parsed, order=None, no_registration=False, flat=False):
    order = parsed["order"] if order is None else order
    p = parsed["pieces"][order[0]]
    y = p["top"] if no_registration else parsed["guide"]-p["registration"][0]
    x = p["left"]
    result = [(order[0], y, x)]
    for prev_index, j in zip(order, order[1:]):
        prev, current = parsed["pieces"][prev_index], parsed["pieces"][j]
        y = (parsed["guide"]-current["entry"][0] if flat else
             y + prev["exit"][0] - current["entry"][0])
        x += prev["w"]+2
        result.append((j, y, x))
    return result


def _paint(shape, bg, marker, guide, pieces, placements, strict=True):
    out = np.full(shape, bg, dtype=np.int8)
    out[guide, 0] = marker
    occupied = {(guide, 0)}
    clipped = []
    for j, y, x in placements:
        lost, kept = 0, 0
        for r, c, color in pieces[j]["cells"]:
            rr, cc = y+r, x+c
            if strict:
                _require(0 <= cc < shape[1], "longitudinal clipping is outside the family")
            if not (0 <= rr < shape[0] and 0 <= cc < shape[1]):
                lost += 1
                continue
            _require((rr, cc) not in occupied, "destinations overlap")
            occupied.add((rr, cc))
            out[rr, cc] = color
            kept += 1
        clipped.append((lost, kept))
    return out, clipped


def _render(parsed, **kwargs):
    result, clips = _paint(parsed["grid"].shape, parsed["background"], parsed["marker"],
                           parsed["guide"], parsed["pieces"], _placements(parsed, **kwargs))
    return np.rot90(result, -parsed["turns"]).copy(), clips


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all conventions, keys, order, and translations from input alone."""
    return _render(_decode(input_grid))[0]


def _spatial_order(parsed, kind):
    root = parsed["start"]
    end = parsed["end"]
    others = [j for j in range(len(parsed["pieces"])) if j not in (root, end)]
    others.sort(key=lambda j: (parsed["pieces"][j]["source_keys"][kind],
                               parsed["pieces"][j]["source_keys"]))
    return (root, *others, end)


def _features(input_grid, output_grid):
    p = _decode(input_grid)
    expected, clips = _render(p)
    _require(np.array_equal(expected, output_grid), "output violates complete rigid zipper assembly")
    order, pieces = p["order"], p["pieces"]
    joins = [pieces[j]["exit"] for j in order[:-1]]
    targets = _placements(p)
    # The all-cell prediction checks material conservation including decoration;
    # inspect cardinality as well so absent/extra in-bounds cells cannot pass.
    copied = sum(kept for lost, kept in clips)
    _require(np.count_nonzero(expected != p["background"]) == copied+1, "material count")
    raw_mismatch = any(tuple(pieces[a]["top"]+r for r in pieces[a]["exit"]) !=
                       tuple(pieces[b]["top"]+r for r in pieces[b]["entry"])
                       for a, b in zip(order, order[1:]))
    no_a = not np.array_equal(_render(p, no_registration=True)[0], expected)
    no_b = not np.array_equal(_render(p, order=_spatial_order(p, 0))[0], expected)
    flat_differs = not np.array_equal(_render(p, flat=True)[0], expected)
    root = pieces[p["start"]]
    rr, cc = root["registration"]
    eccentric = 2*rr != root["h"]-1 or 2*cc != root["w"]-1
    levels = {y+pieces[j]["exit"][0] for j, y, x in targets[:-1]}
    masks = []
    holes = 0
    for piece in pieces:
        core = {(r, c) for r, c, color in piece["cells"] if 0 <= c < piece["w"]}
        holes += len(core) < piece["h"]*piece["w"]
        masks.append((piece["h"], piece["w"], tuple(sorted(core))))
    return dict(parsed=p, n=len(pieces), axis=p["turns"] % 2, side=p["turns"],
                background=p["background"], marker=p["marker"],
                tooth_counts={len(s) for s in joins},
                signatures={_normalized(s) for s in joins},
                repeated=len({_normalized(s) for s in joins}) < len(joins),
                count_ambiguous=len(_chain_orders(pieces, p["start"], p["end"], True)) > 1,
                raw_mismatch=raw_mismatch, no_a=no_a, no_b=no_b, flat_differs=flat_differs,
                eccentric=eccentric, levels=len(levels),
                spatial_conflict=all(_spatial_order(p, k) != order for k in range(3)),
                clipped=sum(bool(lost) for lost, kept in clips),
                complete=sum(not lost for lost, kept in clips),
                retained=all(kept for lost, kept in clips),
                holes=holes, masks=masks, core_shapes={(q["h"], q["w"]) for q in pieces})


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal example counts")
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas size must stay fixed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            f = _features(inp, out)
            _require(f["no_a"] and f["flat_differs"] and f["retained"],
                     "each example must register, propagate, and retain every piece")
            records[split].append(f)
    train, test = records["train"], records["test"]
    _require({f["axis"] for f in train} == {0, 1}, "demonstrate both axes")
    _require(any(f["no_b"] and f["spatial_conflict"] and f["raw_mismatch"] and
                 f["eccentric"] and f["complete"] == f["n"] for f in train),
             "missing complete geometric-ordering witness")
    _require(any(f["repeated"] and f["count_ambiguous"] and f["raw_mismatch"] and f["no_b"]
                 for f in train), "missing gap-versus-cardinality and repeated-key witness")
    demonstrated = set().union(*(f["tooth_counts"] for f in train))
    _require(set().union(*(f["tooth_counts"] for f in test)) <= demonstrated,
             "test tooth counts not demonstrated")
    _require(not any(f["clipped"] for f in test) or any(f["clipped"] for f in train),
             "test clipping not demonstrated")
    _require(max(f["n"] for f in train) <= 4 and min(f["n"] for f in test) >= 5,
             "longer-chain test shift is required")


def _choose(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _pool(triples):
    result = [(0,)] + [(0, gap) for gap in range(1, 5)]
    if triples:
        result += [(0, a, b) for a, b in combinations(range(1, 5), 2)]
    return result


def _make_links(rng, n, triples, witness):
    pool = _pool(triples)
    if witness == "counts":
        a = _choose(rng, [s for s in pool if len(s) == 2])
        b = _choose(rng, [s for s in pool if len(s) == 2 and s != a])
        return [a, a, b]  # one forced self-loop, not two swappable ones
    if witness == "order":
        links = [(0,), _choose(rng, [s for s in pool if len(s) == 2])]
        if triples:
            links.append(_choose(rng, [s for s in pool if len(s) == 3]))
        elif n == 4:
            links.append(_choose(rng, [s for s in pool if s not in links]))
        rng.shuffle(links)
        return links
    for _ in range(30):
        links = [_choose(rng, pool) for _ in range(n-1)]
        pieces = [dict(entry=() if i == 0 else links[i-1], exit=() if i == n-1 else links[i])
                  for i in range(n)]
        if len(_chain_orders(pieces, 0, n-1)) == 1:
            return links
    indices = rng.choice(len(pool), n-1, replace=False)
    return [pool[int(i)] for i in indices]


def _make_piece(rng, h, w, incoming, outgoing, palette, root=False):
    bg, marker = palette[:2]
    body = int(_choose(rng, palette[2:]))
    core = {(r, c) for r in range(h) for c in range(w)
            if r in (0, h-1) or c in (0, w-1)}
    style = int(rng.integers(4))
    if style == 0:
        core |= {(r, c) for r in range(h) for c in range(w)}
    elif style == 2:
        desired = int(rng.integers(len(core), h*w+1))
        while len(core) < desired:
            frontier = sorted({(r+dr, c+dc) for r, c in core for dr, dc in _DIRS
                               if 0 < r+dr < h-1 and 0 < c+dc < w-1} - core)
            core.add(_choose(rng, frontier))
    elif style == 3:
        core |= {(r, c) for r in range(1, h-1) for c in range(1, w-1) if rng.random() < .65}
        # Every interior cell belongs to the perimeter component; discard islands.
        reached, stack = set(core & {(0, 0)}), [(0, 0)]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                q = r+dr, c+dc
                if q in core and q not in reached:
                    reached.add(q)
                    stack.append(q)
        core = reached
    colors = {p: body for p in sorted(core)}
    if rng.random() < .60:
        alternatives = [int(v) for v in palette[2:] if v != body]
        count = int(rng.integers(1, min(4, len(core))+1))
        choices = rng.choice(len(core), count, replace=False)
        ordered = sorted(core)
        for j in choices:
            colors[ordered[int(j)]] = int(_choose(rng, alternatives))
    registration = None
    if root:
        candidates = [p for p in sorted(core) if not (p[0] in (0, h-1) and p[1] in (0, w-1))
                      and (2*p[0] != h-1 or 2*p[1] != w-1)]
        registration = _choose(rng, candidates)
        colors[registration] = int(marker)
    entry_min = int(rng.integers(h-incoming[-1])) if incoming else 0
    exit_min = int(rng.integers(h-outgoing[-1])) if outgoing else 0
    entry = tuple(entry_min+r for r in incoming)
    exit_rows = tuple(exit_min+r for r in outgoing)
    colors.update({(r, -1): int(marker) for r in entry})
    colors.update({(r, w): int(marker) for r in exit_rows})
    return dict(h=h, w=w, entry=entry, exit=exit_rows, registration=registration,
                cells=tuple((r, c, color) for (r, c), color in sorted(colors.items())))


def _scatter(rng, shape, pieces, root_x, bg, marker, guide):
    grid = np.full(shape, bg, dtype=np.int8)
    grid[guide, 0] = marker
    occupied = {(guide, 0)}
    order = [0] + sorted(range(1, len(pieces)), key=lambda j: -len(pieces[j]["cells"]))
    placements = []
    for j in order:
        p = pieces[j]
        blocked = occupied | {(r+dr, c+dc) for r, c in occupied for dr, dc in _DIRS}
        low_c, high_c = min(c for r, c, v in p["cells"]), max(c for r, c, v in p["cells"])
        xs = [root_x] if j == 0 else list(range(1-low_c, shape[1]-1-high_c))
        candidates = []
        for y in range(1, shape[0]-p["h"]):
            if j == 0 and y+p["registration"][0] == guide:
                continue
            for x in xs:
                if all((y+r, x+c) not in blocked for r, c, v in p["cells"]):
                    candidates.append((y, x))
        if not candidates:
            return None
        y, x = _choose(rng, candidates)
        for r, c, color in p["cells"]:
            grid[y+r, x+c] = color
            occupied.add((y+r, x+c))
        placements.append((j, y, x))
    return grid


def _make_pair(rng, plan, params, seen):
    for attempt in range(160):
        witness = plan["witness"]
        n = (4 if witness in ("counts", "order") else
             int(rng.integers(5, 7)) if plan["test"] else int(rng.integers(2, 5)))
        links = _make_links(rng, n, params["triples"], witness)
        palette = params["palette"] if params["stable_palette"] else tuple(map(int, rng.permutation(10)))
        bg, marker = palette[:2]
        widths = [int(rng.integers(3, 5))] + list(map(int, rng.integers(2, 5, size=n-1)))
        length = sum(widths)+2*(n-1)
        if length > 28:
            continue
        root_x = int(rng.integers(2, min(5, 30-length)+1))
        pieces = []
        for j in range(n):
            entry, outgoing = () if j == 0 else links[j-1], () if j == n-1 else links[j]
            minimum = max(3, (entry[-1]+1 if entry else 0), (outgoing[-1]+1 if outgoing else 0))
            height = int(rng.integers(minimum, 9))
            pieces.append(_make_piece(rng, height, widths[j], entry, outgoing, palette, j == 0))
        # Full hypothetical targets relative to the root's registration, before clipping.
        y, x = -pieces[0]["registration"][0], root_x
        relative = [(0, y, x)]
        for j in range(1, n):
            y += pieces[j-1]["exit"][0]-pieces[j]["entry"][0]
            x += pieces[j-1]["w"]+2
            relative.append((j, y, x))
        lo = min(y for j, y, x in relative)
        hi = max(y+pieces[j]["h"]-1 for j, y, x in relative)
        if hi-lo+1 > 27:
            continue
        min_height = max(hi-lo+3, 18 if n >= 4 else 11)
        canvas_h = int(rng.integers(min_height, 31))
        canvas_w = int(rng.integers(root_x+length, 31))
        safe_low, safe_high = max(1, -lo), min(canvas_h-2, canvas_h-1-hi)
        if safe_low > safe_high:
            continue
        if plan["clip"]:
            possible = [g for g in range(1, canvas_h-1)
                        if 1 <= max(-g-lo, g+hi-canvas_h+1) <= 3
                        and all(g+y < canvas_h and g+y+pieces[j]["h"] > 0 for j, y, x in relative)]
            if not possible:
                continue
            guide = int(_choose(rng, possible))
        else:
            guide = int(rng.integers(safe_low, safe_high+1))
        inp = _scatter(rng, (canvas_h, canvas_w), pieces, root_x, bg, marker, guide)
        if inp is None:
            continue
        targets = [(j, guide+y, x) for j, y, x in relative]
        out, clips = _paint(inp.shape, bg, marker, guide, pieces, targets)
        inp, out = (np.rot90(g, -plan["turns"]).copy() for g in (inp, out))
        if _key(inp) in seen:
            continue
        f = _features(inp, out)
        if not f["no_a"] or not f["flat_differs"] or not f["retained"]:
            continue
        if witness == "order" and not (f["no_b"] and f["spatial_conflict"] and f["raw_mismatch"] and
                                       f["eccentric"] and f["complete"] == n):
            continue
        if witness == "counts" and not (f["repeated"] and f["count_ambiguous"] and
                                         f["raw_mismatch"] and f["no_b"]):
            continue
        if plan["clip"] and not f["clipped"]:
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded zipper construction exhausted; no arbitrary fallback rule")


def _check_counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    # Deterministic uint64 normalization also supports negative integer seeds.
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    params = dict(palette=tuple(map(int, rng.permutation(10))),
                  stable_palette=bool(rng.random() < .25),
                  triples=bool(rng.random() < .65), clipping=bool(rng.random() < .55))
    first_side = int(rng.integers(4))
    train_plans = [dict(test=False, witness="order", turns=first_side, clip=False),
                   dict(test=False, witness="counts", turns=(first_side+int(rng.choice([1, 3]))) % 4,
                        clip=params["clipping"])]
    for _ in range(int(num_train)-2):
        train_plans.append(dict(test=False, witness=None, turns=int(rng.integers(4)),
                                clip=bool(params["clipping"] and rng.random() < .25)))
    rng.shuffle(train_plans)
    test_plans = [dict(test=True, witness=None, turns=int(rng.integers(4)),
                      clip=bool(params["clipping"] and rng.random() < .5)) for _ in range(int(num_test))]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, plans in (("train", train_plans), ("test", test_plans)):
        for plan in plans:
            pair = _make_pair(rng, plan, params, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(tuple((_key(pair["input"]), _key(pair["output"])) for pair in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded semantic checks and measured structural diversity across 42 seeds."""
    fingerprints, shapes, cores, masks, backgrounds, markers, sides, signatures = (set() for _ in range(8))
    counts, records = set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair["input"], pair["output"])
                records.append(f)
                shapes.add(pair["input"].shape)
                cores.update(f["core_shapes"])
                masks.update(f["masks"])
                backgrounds.add(f["background"])
                markers.add(f["marker"])
                sides.add(f["side"])
                signatures.update(f["signatures"])
                counts.add(f["n"])
    _require(len(fingerprints) == 42, "whole-puzzle diversity")
    _require(len(shapes) >= 50 and len(cores) >= 15 and len(masks) >= 100, "geometry diversity")
    _require(counts == set(range(2, 7)) and sides == set(range(4)), "piece-count/edge coverage")
    _require(backgrounds == markers == set(range(10)), "palette role coverage")
    _require(len(signatures) >= 10, "gap-pattern diversity")
    _require(sum(f["clipped"] > 0 for f in records) >= 15, "clipping coverage")
    _require(sum(f["holes"] > 0 for f in records) >= 50, "hollow-carrier coverage")
    _require(sum(f["levels"] >= 3 for f in records) >= 20, "propagated transverse levels")
    _require(sum(f["repeated"] for f in records) >= 42, "forced-key repetition")
    # All explicit legal counts, deterministic repeats, and type/range failures.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(446551821, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "explicit counts")
    p = generate_puzzle(0)
    _require(_puzzle_key(p) == _puzzle_key(generate_puzzle(0)), "seed determinism")
    _require(_puzzle_key(generate_puzzle(-1)) == _puzzle_key(generate_puzzle(-1)), "negative seed determinism")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(9, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Conventions are geometric and pair-local. Test exact palette and D4 equivariance.
    rng = np.random.default_rng(123456)
    for pair in p["train"]+p["test"]:
        for turns in range(4):
            palette = rng.permutation(10)
            for reflect in (False, True):
                a, b = (np.rot90(palette[g], turns) for g in (pair["input"], pair["output"]))
                if reflect:
                    a, b = np.fliplr(a), np.fliplr(b)
                _require(np.array_equal(solve(a), b), "palette/orientation equivariance")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    # Family-specific failures: the fixed anchor, copied carrier material, and
    # erased source cells are separate invariants, not just schema properties.
    witness = next(pair for pair in p["train"]
                   if (lambda f: f["complete"] == f["n"] and f["spatial_conflict"])
                   (_features(pair["input"], pair["output"])))
    parsed = _decode(witness["input"])
    canonical_out = np.rot90(witness["output"], parsed["turns"]).copy()
    occupied_payload = np.argwhere((canonical_out != parsed["background"]) &
                                  (canonical_out != parsed["marker"]))
    ghosts = np.argwhere((parsed["grid"] != parsed["background"]) &
                        (canonical_out == parsed["background"]))
    _require(len(occupied_payload) > 0 and len(ghosts) > 0, "material/erasure witnesses")
    for mutation in ("erase_anchor", "delete_payload", "leave_source"):
        bad_out = canonical_out.copy()
        if mutation == "erase_anchor":
            bad_out[parsed["guide"], 0] = parsed["background"]
        elif mutation == "delete_payload":
            bad_out[tuple(occupied_payload[0])] = parsed["background"]
        else:
            point = tuple(ghosts[0])
            bad_out[point] = parsed["grid"][point]
        try:
            _features(witness["input"], np.rot90(bad_out, -parsed["turns"]))
        except AssertionError:
            pass
        else:
            raise AssertionError("accepted family-specific corruption: " + mutation)
    # Explicit signature ambiguities and normalization, independent of palettes.
    a, b = (0, 1), (0, 2)
    good = [dict(entry=(), exit=a), dict(entry=a, exit=a), dict(entry=a, exit=b), dict(entry=b, exit=())]
    _require(len(_chain_orders(good, 0, 3)) == 1 and len(_chain_orders(good, 0, 3, True)) == 2,
             "full-pattern matching must distinguish count-only ambiguity")
    ambiguous = [dict(entry=(), exit=a), dict(entry=a, exit=a), dict(entry=a, exit=a), dict(entry=a, exit=())]
    _require(len(_chain_orders(ambiguous, 0, 3)) == 2, "two self-loops must be rejected")
    # Test rejection by the actual input parser, not only by its graph helper.
    pieces = [_make_piece(rng, 6, 3, () if j == 0 else a,
                          () if j == 3 else a, tuple(range(10)), j == 0)
              for j in range(4)]
    ambiguous_input = _scatter(rng, (24, 29), pieces, 3, 0, 1, 12)
    _require(ambiguous_input is not None, "ambiguity witness placement")
    try:
        solve(ambiguous_input)
    except AssertionError as error:
        _require("unique" in str(error), "ambiguity witness must fail for nonunique order")
    else:
        raise AssertionError("ambiguous complete input accepted")
    _require(_normalized((3, 5, 7)) == _normalized((0, 2, 4)) != _normalized((0, 1, 4)),
             "normalize complete gaps, not cardinality or span")
    return dict(seeds=42, distinct_puzzles=len(fingerprints), grid_shapes=len(shapes),
                core_shapes=len(cores), carrier_masks=len(masks), connector_patterns=len(signatures),
                piece_counts=sorted(counts), anchor_sides=sorted(sides),
                clipped_pairs=sum(f["clipped"] > 0 for f in records),
                hollow_pairs=sum(f["holes"] > 0 for f in records),
                three_level_pairs=sum(f["levels"] >= 3 for f in records),
                semantic_scope="generator-owned checks, not independent approval")


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
