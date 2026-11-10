"""Equal port panels -> one horizontal strip -> width-parity 2x2 readout.

This implements proposal.json literally. Ordering does NOT cause width parity:
W=n*w+n-1. Moreover the crop only needs the directed terminal, not the interior
permutation. These unavoidable limitations are documented in composition.md.
No files, parent generators, seeds or training labels are read by solve().
"""
from collections import Counter, deque
from itertools import combinations
import hashlib
import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "color outside 0..9")
    return a


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _choose(rng, items):
    return items[int(rng.integers(len(items)))]


def _neighbors(p):
    r, c = p
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _connected(mask):
    cells = {tuple(map(int, p)) for p in np.argwhere(mask)}
    if not cells:
        return False
    seen = {min(cells)}
    stack = list(seen)
    while stack:
        for q in _neighbors(stack.pop()):
            if q in cells and q not in seen:
                seen.add(q)
                stack.append(q)
    return seen == cells


def _orders(signatures):
    """Search the whole directed chain, stopping after two solutions."""
    starts = [i for i, (a, b) in enumerate(signatures) if not a and b]
    ends = [i for i, (a, b) in enumerate(signatures) if a and not b]
    if len(starts) != 1 or len(ends) != 1 or any(not a and not b for a, b in signatures):
        return []
    answers = []

    def visit(order, remaining):
        if len(answers) == 2:
            return
        if not remaining:
            if order[-1] == ends[0]:
                answers.append(tuple(order))
            return
        port = signatures[order[-1]][1]
        if port:
            for i in remaining:
                if signatures[i][0] == port:
                    visit(order + [i], tuple(j for j in remaining if j != i))

    visit([starts[0]], tuple(i for i in range(len(signatures)) if i != starts[0]))
    return answers


def _partition(grid):
    """Detect full divider lines and a unique regular partition from cells."""
    candidates = []
    H, W = grid.shape
    for color in np.unique(grid):
        rows = list(map(int, np.flatnonzero(np.all(grid == color, axis=1))))
        cols = list(map(int, np.flatnonzero(np.all(grid == color, axis=0))))
        if not rows and not cols:
            continue
        rr, cc = [-1] + rows + [H], [-1] + cols + [W]
        hs = [b - a - 1 for a, b in zip(rr, rr[1:])]
        ws = [b - a - 1 for a, b in zip(cc, cc[1:])]
        if min(hs + ws) < 3 or len(set(hs)) != 1 or len(set(ws)) != 1:
            continue
        panels = [grid[r + 1:rr[i + 1], c + 1:cc[j + 1]].copy()
                  for i, r in enumerate(rr[:-1]) for j, c in enumerate(cc[:-1])]
        if not 2 <= len(panels) <= 6 or any(np.any(p == color) for p in panels):
            continue
        corners = {int(p[r, c]) for p in panels for r, c in
                   ((0, 0), (0, -1), (-1, 0), (-1, -1))}
        if len(corners) != 1:
            continue
        bg = next(iter(corners))
        if any(not _connected(p != bg) for p in panels):
            continue
        candidates.append(dict(panels=panels, background=bg, divider=int(color),
                               layout=(len(hs), len(ws)), panel_shape=panels[0].shape))
    _require(len(candidates) == 1, "one unambiguous equal-panel partition required")
    return candidates[0]


def _decode(input_grid):
    parsed = _partition(_grid(input_grid))
    panels, bg = parsed["panels"], parsed["background"]
    signatures = [(tuple(map(int, np.flatnonzero(p[:, 0] != bg))),
                   tuple(map(int, np.flatnonzero(p[:, -1] != bg)))) for p in panels]
    orders = _orders(signatures)
    _require(len(orders) == 1, "ports must specify one complete horizontal chain")
    h, w = parsed["panel_shape"]
    _require(len(panels) * (w + 1) - 1 <= 30, "assembled strip exceeds 30 columns")
    parsed.update(order=orders[0], signatures=signatures,
                  width=len(panels) * (w + 1) - 1)
    return parsed


def _layout(panels, order, layout, divider):
    nr, nc = layout
    h, w = panels[0].shape
    grid = np.full((nr * (h + 1) - 1, nc * (w + 1) - 1), divider, dtype=np.int8)
    for slot, index in enumerate(order):
        r, c = divmod(slot, nc)
        grid[r * (h + 1):r * (h + 1) + h,
             c * (w + 1):c * (w + 1) + w] = panels[index]
    return grid


def _canvas(parsed):
    return _layout(parsed["panels"], parsed["order"], (1, len(parsed["panels"])),
                   parsed["divider"])


def _crop(canvas):
    return (canvas[:2, :2] if canvas.shape[1] % 2 else canvas[:2, -2:]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Assemble all intact panels; crop by the resulting width, including dividers."""
    return _crop(_canvas(_decode(input_grid)))


def _signature_kind(s):
    return "single" if len(s) == 1 else "thick" if s[-1] - s[0] + 1 == len(s) else "separated"


def _links(rng, n, h, allowed_kinds=None):
    pool = [s for size in range(1, min(3, h - 2) + 1)
            for s in combinations(range(1, h - 1), size)
            if allowed_kinds is None or _signature_kind(s) in allowed_kinds]
    _require(len(pool) >= max(2 if n >= 3 else 1, n // 2), "insufficient signature pool")
    for _ in range(32):
        links = [_choose(rng, pool) for _ in range(n - 1)]
        if n >= 3 and len(set(links)) < 2:
            continue
        if len(_orders(list(zip([()] + links, links + [()])))) == 1:
            return links
    # Distinct consecutive signature blocks with at most one self-loop each.
    k = max(2 if n >= 3 else 1, n // 2)
    ids = list(map(int, rng.choice(len(pool), size=k, replace=False)))
    doubled = set(map(int, rng.choice(k, size=n - 1 - k, replace=False)))
    return [pool[i] for j, i in enumerate(ids) for _ in range(2 if j in doubled else 1)]


def _join(start, body, allowed, rng):
    """Connect one interior point to the motif using a randomized shortest path."""
    queue, parent = deque([start]), {start: None}
    endpoint = None
    while queue:
        p = queue.popleft()
        if p in body:
            endpoint = p
            break
        adjacent = list(_neighbors(p))
        rng.shuffle(adjacent)
        for q in adjacent:
            if q in allowed and q not in parent:
                parent[q] = p
                queue.append(q)
    _require(endpoint is not None, "disconnected interior")
    while endpoint is not None:
        body.add(endpoint)
        endpoint = parent[endpoint]


def _panel(rng, h, w, incoming, outgoing, bg, colors, terminal_payload):
    # Top/bottom interior cells may carry motif detail. Only left/right contacts
    # are ports in this fixed horizontal family; all four corners remain bg.
    allowed = {(r, c) for r in range(h) for c in range(1, w - 1)}
    roots = {(r, 1) for r in incoming} | {(r, w - 2) for r in outgoing}
    style = int(rng.integers(4))
    if style == 3 and h >= 5 and w >= 5:
        r0, r1 = 1, h - 2
        c0, c1 = 1, w - 2
        hole = {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}
        allowed -= hole
        body = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                if r in (r0, r1) or c in (c0, c1)}
    else:
        body = {_choose(rng, sorted(roots))}
    for point in sorted(roots):
        _join(point, body, allowed, rng)
    if terminal_payload:
        col = 1 if not incoming else w - 2
        _join((int(rng.integers(2)), col), body, allowed, rng)
    if style == 0:
        for _ in range(int(rng.integers(3))):
            _join(_choose(rng, sorted(allowed)), body, allowed, rng)
    else:
        target = max(len(body), int(rng.uniform(.25, .90) * len(allowed)))
        while len(body) < target:
            frontier = sorted({q for p in body for q in _neighbors(p) if q in allowed - body})
            if not frontier:
                break
            body.add(_choose(rng, frontier))
    occupied = sorted(body | {(r, 0) for r in incoming} | {(r, w - 1) for r in outgoing})
    panel = np.full((h, w), bg, dtype=np.int8)
    # No color is an order label, including within a multicolored body.
    offset = int(rng.integers(len(colors)))
    for i, point in enumerate(occupied):
        panel[point] = colors[(i + offset) % len(colors)]
    return panel


def _configuration(rng, parity=None, longer=False):
    counts = [3, 5] if parity == 0 else list(range(2, 7))
    if longer:
        counts = [n for n in counts if n >= 4] or counts
    n = int(_choose(rng, counts))
    widths = [w for w in range(3, min(9, 31 // n - 1) + 1)
              if parity is None or (n * (w + 1) - 1) % 2 == parity]
    _require(bool(widths), "configuration must have a compatible width")
    w = int(_choose(rng, widths))
    h = int(rng.integers(4, 11))
    return n, h, w


def _make_pair(rng, config, seen, witness=False, force_vertical=False, shared=False,
               allowed_kinds=None, allow_multicolor=True):
    n, h, w = config
    layouts = [(nr, n // nr) for nr in range(1, n + 1) if n % nr == 0
               and nr * (h + 1) - 1 <= 30 and (n // nr) * (w + 1) - 1 <= 30]
    if force_vertical:
        layouts = [(n, 1)]
    for _ in range(64):
        palette = list(map(int, rng.permutation(10)))
        bg, divider, motif = palette[0], palette[1], palette[2:]
        modes = ("shared", "partial", "distinct") + (("multi",) if allow_multicolor else ())
        mode = "shared" if shared else _choose(rng, modes)
        if mode == "shared":
            colors = [[motif[0]]] * n
        elif mode == "distinct":
            colors = [[c] for c in motif[:n]]
        elif mode == "partial":
            k = int(rng.integers(1, min(3, n) + 1))
            colors = [[motif[int(rng.integers(k))]] for _ in range(n)]
        else:
            colors = [list(map(int, rng.choice(motif, size=int(rng.integers(2, 5)), replace=False)))
                      for _ in range(n)]
        links = _links(rng, n, h, allowed_kinds)
        panels = [_panel(rng, h, w, () if i == 0 else links[i - 1],
                         () if i == n - 1 else links[i], bg, colors[i],
                         (i in (0, n - 1)) and (witness or rng.random() < .75))
                  for i in range(n)]
        canvas = _layout(panels, list(range(n)), (1, n), divider)
        output = _crop(canvas)
        layout = _choose(rng, layouts)
        # Select a permutation that actually defeats reading-order assembly.
        # This is checked from returned cells, rather than assumed from labels.
        for _ in range(24):
            permutation = list(map(int, rng.permutation(n)))
            input_grid = _layout(panels, permutation, layout, divider)
            reading = _layout(panels, permutation, (1, n), divider)
            if witness and (np.array_equal(_crop(reading), output)
                            or np.array_equal(_crop(input_grid), output)):
                continue
            if witness and np.array_equal(canvas[:2, :2], canvas[:2, -2:]):
                continue
            if _key(input_grid) not in seen:
                seen.add(_key(input_grid))
                return {"input": input_grid, "output": output}
    raise RuntimeError("bounded input construction failed")


def _count(value, lo, hi, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, 2, 10, "num_train")
    ntest = _count(num_test, 1, 4, "num_test")
    _require(ntrain > ntest, "num_train must exceed num_test")
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
             and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    even_width = int(_choose(rng, (4, 6)))
    n = int(_choose(rng, (2, 4)))
    h = int(rng.integers(4, min(9, 31 // n)))
    # The first anchor's input width is even but its assembled width is odd.
    plans = [(n, h, even_width), (3, int(rng.integers(4, 10)), even_width)]
    if ntrain >= 3:
        plans.append((3, int(rng.integers(4, 10)), int(_choose(rng, (3, 5, 7, 9)))))
    if ntrain >= 4:
        # A second even assembled width, at a distinct width, excludes one-size rules.
        other = int(_choose(rng, [w for w in (4, 6, 8) if w != even_width]))
        plans.append((3, int(rng.integers(4, 10)), other))
    while len(plans) < ntrain:
        plans.append(_configuration(rng))
    puzzle, seen = {"train": [], "test": []}, set()
    for index, config in enumerate(plans):
        puzzle["train"].append(_make_pair(rng, config, seen, witness=True,
                                         force_vertical=index == 0, shared=index == 0))
    rng.shuffle(puzzle["train"])
    demonstrated = [_decode(p["input"]) for p in puzzle["train"]]
    kinds = {_signature_kind(s) for p in demonstrated for ab in p["signatures"] for s in ab if s}
    multicolor = any(len(np.unique(panel[panel != p["background"]])) > 1
                     for p in demonstrated for panel in p["panels"])
    parities = [0, 1] if ntest >= 2 else [int(rng.integers(2))]
    while len(parities) < ntest:
        parities.append(int(rng.integers(2)))
    rng.shuffle(parities)
    for parity in parities:
        # Longer chains are a preference, not a new semantic rule.
        config = _configuration(rng, parity, longer=bool(rng.random() < .8))
        # At height six every demonstrated signature category has at least
        # three available signatures, enough even for a six-panel chain.
        config = (config[0], max(6, config[1]), config[2])
        puzzle["test"].append(_make_pair(rng, config, seen, shared=bool(rng.random() < .6),
                                        allowed_kinds=kinds, allow_multicolor=multicolor))
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    grid, out = _grid(pair["input"]), _grid(pair["output"])
    parsed = _decode(grid)
    canvas = _canvas(parsed)
    _require(out.shape == (2, 2) and np.array_equal(out, _crop(canvas)), "incorrect parity crop")
    panels, order = parsed["panels"], parsed["order"]
    h, w = parsed["panel_shape"]
    # Verify all intermediate content even though the public result is a crop.
    copied = [canvas[:, i * (w + 1):i * (w + 1) + w] for i in range(len(panels))]
    _require(Counter(_key(p) for p in copied) == Counter(_key(p) for p in panels),
             "intact-panel conservation")
    reading = _layout(panels, list(range(len(panels))), (1, len(panels)), parsed["divider"])
    left, right = canvas[:2, :2], canvas[:2, -2:]
    parsed.update(parity=canvas.shape[1] % 2,
                  corner_contrast=not np.array_equal(left, right),
                  reading_wrong=not np.array_equal(_crop(reading), out),
                  raw_wrong=not np.array_equal(_crop(grid), out),
                  input_parity_wrong=(grid.shape[1] % 2 != canvas.shape[1] % 2),
                  repeated=len({b for a, b in parsed["signatures"] if b}) < len(panels) - 1)
    return parsed


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _count(nt, 2, 10, "num_train")
    _count(ne, 1, 4, "num_test")
    _require(nt > ne, "invalid split sizes")
    records, seen = {}, set()
    for split in ("train", "test"):
        records[split] = []
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            item = _analyze(pair)
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            records[split].append(item)
    train = records["train"]
    for parity in (0, 1):
        _require(any(p["parity"] == parity and p["corner_contrast"] and p["reading_wrong"]
                     and p["raw_wrong"] for p in train), "missing contrastive parity/assembly witness")
    _require(any(p["input_parity_wrong"] and p["corner_contrast"] for p in train),
             "must distinguish raw and assembled width parity")
    _require(any(len(p["panels"]) >= 3 and len({b for a, b in p["signatures"] if b}) >= 2
                 for p in train), "must teach matching different complete signatures")
    _require(any(len({int(v) for panel in p["panels"] for v in panel.flat
                      if v != p["background"]}) == 1 for p in train),
             "must teach color-independent matching")
    if nt >= 4:
        _require(all(len({p["width"] for p in train if p["parity"] == k}) >= 2 for k in (0, 1)),
                 "need multiple widths of each parity")
    demonstrated = {_signature_kind(s) for p in train for ab in p["signatures"] for s in ab if s}
    tested = {_signature_kind(s) for p in records["test"] for ab in p["signatures"] for s in ab if s}
    _require(tested <= demonstrated, "test port categories need training evidence")
    def multicolor(p):
        return any(len(np.unique(panel[panel != p["background"]])) > 1 for panel in p["panels"])
    _require(not any(map(multicolor, records["test"])) or any(map(multicolor, train)),
             "test multicolor motifs need training evidence")
    return True


def _digest(puzzle):
    data = {s: [{k: np.asarray(v).tolist() for k, v in pair.items()} for pair in puzzle[s]]
            for s in ("train", "test")}
    return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """Streaming 42-seed checks; no files or retained execution traces."""
    digests, shapes, masks, palettes, layouts, widths, counts = (set() for _ in range(7))
    test_parities, roles, signature_sizes = set(), set(), set()
    repeated = multi = input_mismatch = pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digests.add(_digest(puzzle))
        if seed in (0, 17, 41):
            _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterminism")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                p = _analyze(pair)
                a = np.asarray(pair["input"])
                before = a.copy()
                _require(np.array_equal(solve(a), pair["output"]), "input-only solve mismatch")
                _require(np.array_equal(a, before), "solve mutated input")
                shapes.add(p["panel_shape"])
                layouts.add(p["layout"])
                widths.add(p["width"])
                counts.add(len(p["panels"]))
                palettes.add(tuple(map(int, np.unique(a))))
                roles.add((p["background"], p["divider"]))
                for panel in p["panels"]:
                    masks.add(_key((panel != p["background"]).astype(np.int8)))
                    multi += len(set(map(int, panel[panel != p["background"]]))) > 1
                signature_sizes.update(len(s) for ab in p["signatures"] for s in ab if s)
                repeated += p["repeated"]
                input_mismatch += p["input_parity_wrong"]
                pairs += 1
                if split == "test":
                    test_parities.add(p["parity"])
    _require(len(digests) == 42 and len(masks) >= 200, "insufficient structural diversity")
    _require(len(shapes) >= 25 and len(layouts) >= 7 and len(widths) >= 12, "geometry diversity")
    _require(counts == set(range(2, 7)) and test_parities == {0, 1}, "count/parity coverage")
    _require(len(palettes) >= 25 and len(roles) >= 30 and multi >= 20, "palette coverage")
    _require(signature_sizes == {1, 2, 3} and repeated >= 5 and input_mismatch >= 42,
             "port and width witnesses")
    # Ambiguous chains, missing links, and signature cardinality approximations.
    a, b, c = (1,), (2,), (1, 2)
    _require(len(_orders([((), a), (a, a), (a, ())])) == 1, "forced self-loop")
    _require(len(_orders([((), a), (a, a), (a, a), (a, ())])) == 2, "ambiguity detection")
    _require(not _orders([((), a), (b, ())]), "must match positions, not port count")
    _require(len(_orders([((), c), (c, a), (a, ())])) == 1, "whole signature match")
    # Color permutations and arbitrary input panel order must preserve the rule.
    sample = generate_puzzle(971, 2, 1)
    for pair in sample["train"] + sample["test"]:
        parsed = _decode(pair["input"])
        n = len(parsed["panels"])
        shuffled = _layout(parsed["panels"], list(range(n))[::-1], parsed["layout"], parsed["divider"])
        _require(np.array_equal(solve(shuffled), pair["output"]), "layout-order dependence")
        remap = np.array([7, 4, 8, 0, 6, 9, 1, 3, 5, 2], dtype=np.int8)
        _require(np.array_equal(solve(remap[pair["input"]]), remap[pair["output"]]), "color dependence")
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in sample[s]]
               for s in ("train", "test")}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("output mutation accepted")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            generate_puzzle(1000 + nt * 10 + ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(1, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(digests),
            "panel_shapes": len(shapes), "motif_masks": len(masks), "layouts": len(layouts),
            "strip_widths": sorted(widths), "panel_counts": sorted(counts),
            "palettes": len(palettes), "role_assignments": len(roles),
            "repeated_signature_pairs": repeated, "multicolored_panels": multi,
            "input_width_parity_mismatches": input_mismatch}
