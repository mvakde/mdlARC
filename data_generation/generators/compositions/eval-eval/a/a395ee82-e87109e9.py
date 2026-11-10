"""Stencil-contact routing: the copied mask is also a routing sensor.

Only NumPy and the standard library are used. No files, saved examples, parent
generators, or generation-time state are needed by solve().
"""

from collections import Counter
import hashlib
import json

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    todo = [next(iter(cells))]
    reached = set(todo)
    while todo:
        r, c = todo.pop()
        for dr, dc in DIRS:
            q = r + dr, c + dc
            if q in cells and q not in reached:
                reached.add(q)
                todo.append(q)
    return reached == cells


def _asymmetric(mask):
    return all(not np.array_equal(mask, b) for b in
               (mask[::-1], mask[:, ::-1], mask[::-1, ::-1], mask.T,
                mask.T[::-1], mask.T[:, ::-1], mask.T[::-1, ::-1]))


def _parse(input_grid):
    """Decode the visible key, grid, source stencil, and local peg offsets."""
    a = _grid(input_grid)
    frame = int(a[0, 0])
    _require(bool(np.all(a[0] == frame)), "missing key top border")
    full = np.flatnonzero(np.all(a == frame, axis=1))
    _require(len(full) >= 3, "missing board borders")
    crop = int(full[1])
    _require(crop >= 2, "key must have an interior")
    key_rows = a[1:crop]
    _require(bool(np.all(key_rows == key_rows[0])), "inconsistent key rows")
    row = key_rows[0]
    walls = np.flatnonzero(row == frame)
    _require(len(walls) >= 3 and walls[0] == 0 and walls[-1] == a.shape[1]-1,
             "invalid key walls")
    background_candidates = []
    for start, end in zip(walls[:-1], walls[1:]):
        _require(end-start >= 4, "key panels too narrow")
        background_candidates.extend(map(int, row[start+2:end-1]))
    _require(bool(background_candidates) and len(set(background_candidates)) == 1,
             "ambiguous panel background")
    bg = background_candidates[0]
    _require(bg != frame, "background equals frame")
    turns = {}
    panel_widths = []
    for start, end in zip(walls[:-1], walls[1:]):
        left, right = int(row[start+1]), int(row[end-1])
        _require((left == bg) != (right == bg), "ambiguous left/right bar")
        color, side = (right, 1) if left == bg else (left, -1)
        _require(color not in turns and color not in (bg, frame), "conflicting key")
        turns[color] = side
        panel_widths.append(int(end-start+1))
    _require(set(turns.values()) == {-1, 1}, "both turn instructions required")
    board = a[crop:]
    rows = np.flatnonzero(np.all(board == frame, axis=1))
    cols = np.flatnonzero(np.all(board == frame, axis=0))
    _require(len(rows) >= 3 and len(cols) >= 3 and rows[0] == cols[0] == 0
             and rows[-1] == board.shape[0]-1 and cols[-1] == board.shape[1]-1,
             "invalid board borders")
    _require(len(set(map(int, np.diff(rows)))) == 1
             and len(set(map(int, np.diff(cols)))) == 1, "irregular board pitch")
    h, w = int(rows[1]-1), int(cols[1]-1)
    _require(h >= 2 and w >= 2, "stencil cells too small")
    nr, nc = len(rows)-1, len(cols)-1
    inks = set(map(int, np.unique(board))) - {bg, frame} - set(turns)
    _require(len(inks) == 1, "source ink not unique")
    ink = inks.pop()
    source = None
    mask = None
    pegs = {}
    for r in range(nr):
        for c in range(nc):
            tile = board[1+r*(h+1):(r+1)*(h+1), 1+c*(w+1):(c+1)*(w+1)]
            _require(not np.any(tile == frame), "frame inside a tile")
            if np.any(tile == ink):
                _require(source is None and set(map(int, np.unique(tile))) <= {bg, ink},
                         "source must occupy one peg-free tile")
                source = (r, c)
                mask = tile == ink
                _require(_connected(mask) and 3 <= int(mask.sum()) < h*w,
                         "stencil must be connected with a hole")
                _require(all((mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any())),
                         "stencil bounding box must equal tile interior")
            else:
                points = np.argwhere(tile != bg)
                _require(len(points) <= 1, "more than one peg in a tile")
                if len(points):
                    y, x = map(int, points[0])
                    color = int(tile[y, x])
                    _require(color in turns, "unkeyed peg")
                    pegs[(r, c)] = (y, x, color)
    _require(source is not None, "missing source")
    return dict(board=board, crop=crop, bg=bg, frame=frame, ink=ink,
                h=h, w=w, nr=nr, nc=nc, source=source, mask=mask,
                pegs=pegs, turns=turns, panel_widths=tuple(panel_widths))


def _walk(info, contact="stencil", turn_rule="relative", stop_after=None,
          reject_cycles=True):
    """Logical visits only: printed pixels never become new control objects.

    Alternative modes are bounded counterfactuals for validator/self-test use.
    No execution traces are serialized.
    """
    sr, sc = info["source"]
    visited = {(sr, sc)}
    events, skipped, chains = [], [], []
    for initial in range(4):
        r, c, d = sr, sc, initial
        seen = set()
        chain = []
        while True:
            state = (r, c, d)
            if state in seen:
                _require(not reject_cycles, "cyclic printhead path")
                break
            seen.add(state)
            dr, dc = DIRS[d]
            r, c = r+dr, c+dc
            if not (0 <= r < info["nr"] and 0 <= c < info["nc"]):
                break
            visited.add((r, c))
            peg = info["pegs"].get((r, c))
            if peg is None:
                continue
            y, x, color = peg
            live = contact == "all" or (contact == "stencil" and info["mask"][y, x])
            if not live:
                skipped.append((initial, r, c, d, color))
                continue
            if stop_after is not None and len(chain) >= stop_after:
                continue
            old = d
            side = info["turns"][color]
            if turn_rule == "relative":
                d = (d+side) % 4
            elif turn_rule == "reverse":
                d = (d-side) % 4
            elif turn_rule == "absolute":
                d = 3 if side == -1 else 1
            else:
                raise ValueError("unknown turn rule")
            chain.append((r, c))
            events.append((initial, r, c, old, d, color))
        chains.append(tuple(chain))
    return visited, tuple(events), tuple(skipped), tuple(chains)


def _print(info, visited, mask=None):
    mask = info["mask"] if mask is None else mask
    out = info["board"].copy()
    for r, c in visited:
        top, left = 1+r*(info["h"]+1), 1+c*(info["w"]+1)
        tile = out[top:top+info["h"], left:left+info["w"]]
        tile[mask & (tile == info["bg"])] = info["ink"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only the input: decode contact-sensitive routes and print the stencil."""
    info = _parse(input_grid)
    return _print(info, _walk(info)[0])


def _features(pair):
    info = _parse(pair["input"])
    visited, events, skipped, chains = _walk(info)
    expected = _print(info, visited)
    _require(np.array_equal(expected, pair["output"]), "wrong routed stencil output")
    _require(expected.shape == info["board"].shape, "wrong key crop")
    # Stronger than ink decoration: changing the sensor changes occupied tiles.
    all_live = _walk(info, contact="all", reject_cycles=False)[0]
    straight = _walk(info, contact="none")[0]
    first_only = _walk(info, stop_after=1, reject_cycles=False)[0]
    reverse = _walk(info, turn_rule="reverse", reject_cycles=False)[0]
    absolute = _walk(info, turn_rule="absolute", reject_cycles=False)[0]
    headings = {side: {e[3] for e in events if info["turns"][e[5]] == side}
                for side in (-1, 1)}
    return dict(info=info, visited=visited, events=events, skipped=skipped,
                chains=chains, headings=headings, contact_matters=visited != all_live,
                turns_matter=visited != straight, recursion_matters=visited != first_only,
                reverse_matters=visited != reverse, absolute_matters=visited != absolute)


def validate_generated_puzzle(puzzle):
    """Check outputs, contact causality, route legality and teaching evidence."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid counts")
    seen = set()
    groups = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            key = _key(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            f = _features(dict(input=a, output=b))
            _require(f["contact_matters"] and f["turns_matter"], "either parent can be ignored")
            _require(bool(f["events"]) and bool(f["skipped"]), "contact and hole passage required")
            q = f["info"]
            # Frames, every control peg, and the source are immutable.
            keep = q["board"] != q["bg"]
            _require(np.array_equal(b[keep], q["board"][keep]), "input objects overwritten")
            gain = int(np.count_nonzero(b == q["ink"])) - int(q["mask"].sum())
            live_pegs = sum(p in f["visited"] and q["mask"][peg[0], peg[1]]
                            for p, peg in q["pegs"].items())
            _require(gain == (len(f["visited"])-1)*int(q["mask"].sum())-live_pegs,
                     "incorrect stencil support or duplicate printing")
            groups[split].append(f)
    train = groups["train"]
    _require(all(len(set().union(*(f["headings"][s] for f in train))) >= 2 for s in (-1, 1)),
             "training must show both relative turns from multiple headings")
    _require(any(f["recursion_matters"] for f in train), "recursion not taught")
    _require(any(f["reverse_matters"] for f in train) and any(f["absolute_matters"] for f in train),
             "simpler turn conventions not distinguished")
    _require(any(_asymmetric(f["info"]["mask"]) for f in train), "exact orientation not taught")
    _require(len({f["info"]["h"] for f in train}) >= 2
             and len({f["info"]["w"] for f in train}) >= 2, "both tile pitches must vary")
    _require(len({f["info"]["source"] for f in train}) >= 2, "anchor position must vary")
    _require(any(len(f["visited"]) < f["info"]["nr"]*f["info"]["nc"] for f in train),
             "printing the whole board not distinguished")
    # Generated-puzzle policy, separate from the input-only semantic solver:
    # every test extends one demonstrated dimension or both board extents.
    th = max(f["info"]["h"] for f in train)
    tw = max(f["info"]["w"] for f in train)
    tr = max(f["info"]["nr"] for f in train)
    tc = max(f["info"]["nc"] for f in train)
    _require(all(f["info"]["h"] > th for f in groups["test"])
             or all(f["info"]["w"] > tw for f in groups["test"])
             or all(f["info"]["nr"] > tr and f["info"]["nc"] > tc for f in groups["test"]),
             "missing geometric train/test shift")
    return None


def _mask(rng, h, w, asymmetric=False):
    # Tight spanning path followed by frontier growth: broad connected masks,
    # always with at least one aperture for inactive control pegs.
    for _ in range(40):
        a = np.zeros((h, w), dtype=bool)
        steps = [0]*(h-1) + [1]*(w-1)
        rng.shuffle(steps)
        r = c = 0
        a[r, c] = True
        for axis in steps:
            r += axis == 0
            c += axis == 1
            a[r, c] = True
        target = int(rng.integers(int(a.sum()), h*w))
        while int(a.sum()) < target:
            frontier = sorted({(r+dr, c+dc) for r, c in np.argwhere(a) for dr, dc in DIRS
                               if 0 <= r+dr < h and 0 <= c+dc < w and not a[r+dr, c+dc]})
            a[_pick(rng, frontier)] = True
        if rng.integers(2):
            a = a[:, ::-1].copy()
        if not asymmetric or _asymmetric(a):
            return a
    # A rectangular tile with one missing corner has no nonidentity symmetry.
    _require(h != w, "asymmetric teaching fallback requires a rectangle")
    a = np.ones((h, w), dtype=bool)
    a[0, 0] = False
    return a


def _make_pair(rng, dims, extent, palette, informative=False):
    h, w = dims
    hh = int(rng.integers(3, 6))
    max_rows = (30-hh)//(h+1)
    max_cols = 29//(w+1)
    lo, hi = extent
    nr = int(rng.integers(lo, min(hi, max_rows)+1))
    nc = int(rng.integers(lo, min(hi, max_cols)+1))
    height, width = nr*(h+1)+1, nc*(w+1)+1
    nkeys = int(rng.integers(2, min(4, (width-1)//4)+1))
    bg, frame, ink = map(int, palette[:3])
    colors = list(map(int, palette[3:3+nkeys]))
    sides = [-1, 1] + [int(rng.choice((-1, 1))) for _ in range(nkeys-2)]
    rng.shuffle(sides)
    turns = dict(zip(colors, sides))
    # Panels share walls; widths need not match the board's tile columns.
    pitches = 4 + rng.multinomial(width-1-4*nkeys, np.ones(nkeys)/nkeys)
    header = np.full((hh, width), bg, dtype=np.int8)
    header[[0, -1]] = frame
    x = 0
    for color, pitch in zip(rng.permutation(colors), pitches):
        end = x+int(pitch)
        header[:, [x, end]] = frame
        header[1:-1, x+1 if turns[int(color)] == -1 else end-1] = color
        x = end
    board = np.full((height, width), bg, dtype=np.int8)
    board[::h+1, :] = frame
    board[:, ::w+1] = frame
    mask = _mask(rng, h, w, informative)
    sr, sc = int(rng.integers(nr)), int(rng.integers(nc))
    y, x = 1+sr*(h+1), 1+sc*(w+1)
    tile = board[y:y+h, x:x+w]
    tile[mask] = ink
    live, holes = np.argwhere(mask), np.argwhere(~mask)
    density = float(rng.uniform(.30, .72))
    for r in range(nr):
        for c in range(nc):
            if (r, c) == (sr, sc) or rng.random() > density:
                continue
            positions = live if rng.random() < .60 else holes
            ly, lx = map(int, _pick(rng, positions))
            board[1+r*(h+1)+ly, 1+c*(w+1)+lx] = _pick(rng, colors)
    inp = np.vstack((header[:-1], board))
    info = _parse(inp)
    visited = _walk(info)[0]
    return dict(input=inp, output=_print(info, visited))


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name+" must be an integer")
        if not low <= value <= high:
            raise ValueError(name+" out of range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _teaching_ready(features):
    return (all(len(set().union(*(f["headings"][s] for f in features))) >= 2 for s in (-1, 1))
            and any(f["recursion_matters"] for f in features)
            and any(f["reverse_matters"] for f in features)
            and any(f["absolute_matters"] for f in features)
            and any(len(f["visited"]) < f["info"]["nr"]*f["info"]["nc"] for f in features))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(3))  # width, height, or logical route extent
    stable_palette = bool(rng.random() < .2)
    base_palette = rng.permutation(10)
    if shift == 0:
        hs = rng.choice([2, 3, 4], 2, replace=False)
        ws = rng.permutation([2, 3])
    elif shift == 1:
        hs = rng.permutation([2, 3])
        ws = rng.choice([2, 3, 4], 2, replace=False)
    else:
        hs = rng.permutation([2, 3])
        ws = rng.permutation([2, 3])
    # At least one rectangular asymmetric stencil; avoid two square lessons.
    if all(h == w for h, w in zip(hs, ws)):
        ws = ws[::-1].copy()
    puzzle = dict(train=[], test=[])
    seen, lessons, sources = set(), [], set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            if split == "train":
                dims = (int(hs[i % 2]), int(ws[i % 2])) if i < 2 else (
                    int(rng.choice(hs)), int(rng.choice(ws)))
                extent = (4, 5) if shift == 2 else (4, 7)
            else:
                dims = ((int(rng.choice(hs)), 4) if shift == 0 else
                        (4, int(rng.choice(ws))) if shift == 1 else
                        (int(rng.choice(hs)), int(rng.choice(ws))))
                extent = (6, 9) if shift == 2 else (4, 7)
            informative = split == "train" and i < 2 and dims[0] != dims[1]
            for attempt in range(800):
                palette = base_palette if stable_palette else rng.permutation(10)
                try:
                    pair = _make_pair(rng, dims, extent, palette, informative)
                    f = _features(pair)
                except AssertionError:  # A candidate may contain a routing cycle.
                    continue
                if not (f["contact_matters"] and f["turns_matter"] and f["events"] and f["skipped"]):
                    continue
                if _key(pair["input"]) in seen:
                    continue
                if split == "train" and i == 1:
                    if f["info"]["source"] in sources or not _teaching_ready(lessons+[f]):
                        continue
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                if split == "train":
                    lessons.append(f)
                    sources.add(f["info"]["source"])
                break
            else:
                raise RuntimeError("bounded scene sampler exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def diversity_metrics(puzzles):
    stats = Counter()
    masks, layouts, shapes, anchors, palettes, peg_counts, max_turns = (set() for _ in range(7))
    dimensions = {"train": set(), "test": set()}
    headings = {-1: set(), 1: set()}
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                q = f["info"]
                stats["pairs"] += 1
                stats["contact_sensitive"] += bool(f["contact_matters"])
                stats["turn_sensitive"] += bool(f["turns_matter"])
                stats["recursive"] += bool(f["recursion_matters"])
                stats["unvisited_pegs"] += any(p not in f["visited"] for p in q["pegs"])
                stats["edge_source"] += q["source"][0] in (0, q["nr"]-1) or q["source"][1] in (0, q["nc"]-1)
                stats["unequal_panels"] += len(set(q["panel_widths"])) > 1
                stats["asymmetric"] += _asymmetric(q["mask"])
                masks.add((q["mask"].shape, q["mask"].tobytes()))
                layouts.add((q["nr"], q["nc"], q["source"], tuple(sorted(q["pegs"].items()))))
                shapes.add(pair["input"].shape)
                anchors.add(q["source"])
                palettes.add((q["bg"], q["frame"], q["ink"], tuple(sorted(q["turns"].items()))))
                peg_counts.add(len(q["pegs"]))
                max_turns.add(max(map(len, f["chains"])))
                dimensions[split].add((q["h"], q["w"]))
                for side in (-1, 1):
                    headings[side].update(f["headings"][side])
    return dict(stats, puzzles=len(puzzles), masks=len(masks), layouts=len(layouts),
                input_shapes=len(shapes), anchors=len(anchors), palettes=len(palettes),
                peg_counts=sorted(peg_counts), max_turns=sorted(max_turns),
                stencil_dimensions={s: sorted(d) for s, d in dimensions.items()},
                incoming_headings={str(s): sorted(d) for s, d in headings.items()})


def self_test():
    """Bounded 42-seed measurements, count checks and semantic negative tests."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    def fingerprint(puzzle):
        return hashlib.sha256(b"".join(repr(_key(pair[n])).encode() for s in ("train", "test")
                                       for pair in puzzle[s] for n in ("input", "output"))).hexdigest()
    assert fingerprint(puzzles[0]) == fingerprint(generate_puzzle(0))
    assert len({fingerprint(p) for p in puzzles}) == 42
    for nt, nq in ((2, 1), (10, 1), (10, 4), (6, 3)):
        p = generate_puzzle(1234+nt+nq, nt, nq)
        assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    metrics = diversity_metrics(puzzles)
    assert metrics["pairs"] == metrics["contact_sensitive"] == metrics["turn_sensitive"] == 210
    assert metrics["masks"] >= 40 and metrics["layouts"] >= 180
    assert metrics["input_shapes"] >= 25 and metrics["anchors"] >= 18
    assert len(metrics["peg_counts"]) >= 10 and max(metrics["max_turns"]) >= 4
    assert metrics["recursive"] >= 42 and metrics["unvisited_pegs"] >= 42
    assert metrics["edge_source"] >= 20 and metrics["unequal_panels"] >= 30
    assert all(len(v) == 4 for v in metrics["incoming_headings"].values())
    # Hand-calculated contact lesson. East hits L at (2,3), ignores a peg in
    # the stencil's top-right aperture at (1,3), then hits R at (0,3).
    lesson = dict(nr=5, nc=6, source=(2, 1),
                  mask=np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
                  pegs={(2, 3): (0, 0, 6), (1, 3): (0, 2, 7), (0, 3): (1, 1, 7)},
                  turns={6: -1, 7: 1})
    lesson_visits = {(2, 1), (1, 1), (0, 1), (3, 1), (4, 1), (2, 0),
                     (2, 2), (2, 3), (1, 3), (0, 3), (0, 4), (0, 5)}
    visits, events, skipped, chains = _walk(lesson)
    assert visits == lesson_visits
    assert [(e[1], e[2], e[3], e[4]) for e in events] == [(2, 3, 1, 0), (0, 3, 0, 1)]
    assert len(skipped) == 1 and max(map(len, chains)) == 2
    cycle = dict(lesson, nr=5, nc=5, source=(2, 2), turns={7: 1},
                 pegs={p: (0, 0, 7) for p in ((2, 4), (4, 4), (4, 0), (2, 0))})
    try:
        _walk(cycle)
    except AssertionError:
        pass
    else:
        raise AssertionError("cyclic route accepted")
    # A local sensor counterfactual keeps positions and turn keys fixed: making
    # all holes opaque changes which tiles are reached, not just their decoration.
    f = _features(puzzles[0]["train"][0])
    q = f["info"]
    filled = dict(q, mask=np.ones_like(q["mask"]))
    assert f["visited"] != _walk(filled, reject_cycles=False)[0]
    # Same stencil, no turning: the geometric answer changes too.
    assert f["visited"] != _walk(q, contact="none")[0]
    # Semantic colors are local. A complete permutation preserves the answer.
    palette = np.array([7, 0, 9, 3, 1, 8, 4, 6, 2, 5], dtype=np.int8)
    pair = puzzles[0]["test"][0]
    assert np.array_equal(solve(palette[pair["input"]]), palette[pair["output"]])
    # Invalid data must not be accepted merely because solve and generation agree.
    bad = {s: [dict(pair) for pair in puzzles[0][s]] for s in ("train", "test")}
    bad["train"][0]["output"] = bad["train"][0]["output"].copy()
    bad["train"][0]["output"][0, 0] = (int(bad["train"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return metrics


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
