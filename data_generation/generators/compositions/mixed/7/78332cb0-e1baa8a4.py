"""Recover independently stretched panels, then assemble their latent port chain.

Only NumPy and the standard library are used. No parent module is imported.
The public solver reads only its grid; construction metadata never reaches it.
No files, execution traces, logs, or random global state are used by this module.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import hashlib
import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("a grid must be an integer matrix of size 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _runs(a, axis):
    vectors = a if axis == 0 else a.T
    starts = [0] + [i for i in range(1, len(vectors))
                    if not np.array_equal(vectors[i], vectors[i - 1])]
    ends = starts[1:] + [len(vectors)]
    return starts, tuple(b - a for a, b in zip(starts, ends))


def _collapse(a):
    rows, _ = _runs(a, 0)
    cols, _ = _runs(a, 1)
    return a[np.ix_(rows, cols)].copy()


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    stack = [cells.pop()]
    while stack:
        r, c = stack.pop()
        for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if q in cells:
                cells.remove(q)
                stack.append(q)
    return not cells


def _signatures(panel, background, axis):
    p = panel if axis == "H" else panel.T
    return (tuple(map(int, np.flatnonzero(p[:, 0] != background))),
            tuple(map(int, np.flatnonzero(p[:, -1] != background))))


def _orders(signatures):
    """Return at most two complete directed chains; repeated ports are legal."""
    starts = [i for i, (a, b) in enumerate(signatures) if not a and b]
    ends = [i for i, (a, b) in enumerate(signatures) if a and not b]
    if len(starts) != 1 or len(ends) != 1 or any(not a and not b for a, b in signatures):
        return []
    answers = []

    def visit(order, remaining):
        if len(answers) >= 2:
            return
        if not remaining:
            if order[-1] == ends[0]:
                answers.append(tuple(order))
            return
        outgoing = signatures[order[-1]][1]
        if not outgoing:
            return
        for i in remaining:
            if signatures[i][0] == outgoing:
                visit(order + [i], tuple(j for j in remaining if j != i))

    visit([starts[0]], tuple(i for i in range(len(signatures)) if i != starts[0]))
    return answers


def _partitions(a):
    """Discover exclusive divider lines, allowing unequal physical tile sizes."""
    h, w = a.shape
    for color in np.unique(a):
        rr = [-1] + list(map(int, np.flatnonzero(np.all(a == color, axis=1)))) + [h]
        cc = [-1] + list(map(int, np.flatnonzero(np.all(a == color, axis=0)))) + [w]
        count = (len(rr) - 1) * (len(cc) - 1)
        if not 2 <= count <= 6:
            continue
        if min([b - x - 1 for x, b in zip(rr, rr[1:])]
               + [b - x - 1 for x, b in zip(cc, cc[1:])]) < 3:
            continue
        raw = [a[rr[r] + 1:rr[r + 1], cc[c] + 1:cc[c + 1]].copy()
               for r in range(len(rr) - 1) for c in range(len(cc) - 1)]
        if any(np.any(p == color) for p in raw):
            continue
        corners = {int(p[r, c]) for p in raw for r, c in
                   ((0, 0), (0, -1), (-1, 0), (-1, -1))}
        if len(corners) != 1:
            continue
        compact = [_collapse(p) for p in raw]
        if len({p.shape for p in compact}) != 1 or min(compact[0].shape) < 3:
            continue
        yield dict(raw=raw, panels=compact, background=corners.pop(),
                   divider=int(color), layout=(len(rr) - 1, len(cc) - 1))


def _decode(a):
    candidates = []
    for parsed in _partitions(a):
        panels, bg = parsed["panels"], parsed["background"]
        if any(not _connected(p != bg) for p in panels):
            continue
        for axis in ("H", "V"):
            oriented = [p if axis == "H" else p.T for p in panels]
            if any(np.any(p[0] != bg) or np.any(p[-1] != bg) for p in oriented):
                continue
            signatures = [_signatures(p, bg, axis) for p in panels]
            orders = _orders(signatures)
            if len(orders) == 1:
                candidate = dict(parsed, axis=axis, signatures=signatures, order=orders[0])
                candidates.append(candidate)
    if len(candidates) != 1:
        raise ValueError("input must specify one partition, port axis, and complete chain")
    result = candidates[0]
    p = result["panels"][0]
    along = p.shape[1] if result["axis"] == "H" else p.shape[0]
    if len(result["panels"]) * (along + 1) - 1 > 30:
        raise ValueError("assembled chain exceeds the grid bound")
    return result


def _strip(panels, order, axis, divider):
    dimension = 1 if axis == "H" else 0
    h, w = panels[0].shape
    separator = np.full((h, 1) if dimension == 1 else (1, w), divider, dtype=np.int8)
    pieces = []
    for j, i in enumerate(order):
        if j:
            pieces.append(separator)
        pieces.append(panels[i])
    return np.concatenate(pieces, axis=dimension)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Collapse maximal equal full-vector runs per panel and match latent ports."""
    d = _decode(_grid(input_grid))
    return _strip(d["panels"], d["order"], d["axis"], d["divider"])


def _tile(panels, layout, divider):
    nr, nc = layout
    heights = [panels[r * nc].shape[0] for r in range(nr)]
    widths = [panels[c].shape[1] for c in range(nc)]
    out = np.full((sum(heights) + nr - 1, sum(widths) + nc - 1), divider, dtype=np.int8)
    top = 0
    for r, height in enumerate(heights):
        left = 0
        for c, width in enumerate(widths):
            p = panels[r * nc + c]
            _require(p.shape == (height, width), "physical tile row/column totals disagree")
            out[top:top + height, left:left + width] = p
            left += width + 1
        top += height + 1
    return out


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _pool(height, kind="all"):
    values = [(r,) for r in range(1, height - 1)]
    if kind == "all":
        values += list(combinations(range(1, height - 1), 2))
    return values


def _links(rng, n, height, witness):
    pool = _pool(height, "single" if witness == "single" else "all")
    if witness == "extended":
        thick = _pick(rng, [s for s in pool if len(s) == 2 and s[1] == s[0] + 1])
        spaced = _pick(rng, [s for s in pool if len(s) == 2 and s[1] > s[0] + 1])
        # One thick self-loop is forced into its unique place. Its two fully
        # occupied adjacent rows have different literal vectors, teaching that
        # compression compares values even though ports compare occupancy.
        return [thick, thick, spaced] if rng.random() < .5 else [spaced, thick, thick]
    # Distinct vertices, optionally one repeated join (a forced self-loop).
    repeat = n >= 4 and witness != "single" and rng.random() < .30
    count = n - 2 if repeat else n - 1
    chosen = rng.choice(len(pool), count, replace=False)
    links = [pool[int(i)] for i in chosen]
    if repeat:
        at = int(rng.integers(len(links)))
        links.insert(at, links[at])
    return links


def _panel(rng, height, width, incoming, outgoing, bg, colors):
    """Connected branching bodies, with fixed contacts and observable vectors."""
    for _ in range(96):
        mask = np.zeros((height, width), dtype=bool)
        hub = int(rng.integers(1, width - 1))
        mask[1:-1, hub] = True
        for row in incoming:
            mask[row, :hub + 1] = True
        for row in outgoing:
            mask[row, hub:] = True
        # Interior growth preserves connectivity and never invents edge ports.
        target = int(rng.integers(height - 2, (height - 2) * (width - 2) + 1))
        for _ in range(max(0, target - int(mask[1:-1, 1:-1].sum()))):
            frontier = []
            for r in range(1, height - 1):
                for c in range(1, width - 1):
                    if not mask[r, c] and (mask[r - 1, c] or mask[r + 1, c]
                                           or mask[r, c - 1] or mask[r, c + 1]):
                        frontier.append((r, c))
            if not frontier:
                break
            mask[_pick(rng, frontier)] = True
        p = np.full((height, width), bg, dtype=np.int8)
        p[mask] = rng.choice(colors, int(mask.sum()))
        if _collapse(p).shape == p.shape and _connected(mask):
            return p
    return None


def _schedule(rng, length, total, force_nonuniform=False):
    extra = total - length
    mode = int(rng.integers(5))
    if mode == 0 and total % length == 0 and not force_nonuniform:
        return np.full(length, total // length, dtype=int)
    result = np.ones(length, dtype=int)
    if mode == 1:
        result[int(rng.integers(length))] += extra
    else:
        result += rng.multinomial(extra, rng.dirichlet(np.ones(length)))
        if mode == 2:
            result.sort()
        elif mode == 3:
            result = np.sort(result)[::-1].copy()
    if force_nonuniform and len(set(map(int, result))) == 1:
        # extra > 0; equality here implies every entry is at least two.
        result[0] -= 1
        result[-1] += 1
    return result


def _configurations(n, axis, witness):
    result = []
    for transverse in range(5, 9):
        for along in range(5 if witness == "single" else 4, min(10, 31 // n - 1) + 1):
            h, w = (transverse, along) if axis == "H" else (along, transverse)
            for nr in range(1, n + 1):
                if n % nr:
                    continue
                nc = n // nr
                # At least one duplicate is available on both axes in every tile.
                if nr * (h + 2) - 1 <= 30 and nc * (w + 2) - 1 <= 30:
                    result.append((h, w, nr, nc))
    return result


def _make_pair(rng, n, axis, witness, color_mode, test):
    for _ in range(120):
        h, w, nr, nc = _pick(rng, _configurations(n, axis, witness))
        transverse, along = (h, w) if axis == "H" else (w, h)
        palette = list(map(int, rng.permutation(10)))
        bg, divider = palette[:2]
        motif_colors = palette[2:]
        links = _links(rng, n, transverse, witness)
        if color_mode == "shared":
            colors = [[motif_colors[0]]] * n
        elif color_mode == "distinct":
            colors = [[c] for c in motif_colors[:n]]
        elif color_mode == "partial":
            colors = [[motif_colors[int(rng.integers(2))]] for _ in range(n)]
        else:
            colors = [motif_colors[:int(rng.integers(2, 5))] for _ in range(n)]
        compact = [_panel(rng, transverse, along, () if i == 0 else links[i - 1],
                          () if i == n - 1 else links[i], bg, colors[i]) for i in range(n)]
        if any(p is None for p in compact):
            continue
        if witness == "extended" and all(
                np.array_equal(compact[i][list(s), -1], compact[i + 1][list(s), 0])
                for i, s in enumerate(links)):
            continue
        if axis == "V":
            compact = [p.T.copy() for p in compact]
        order = list(map(int, rng.permutation(n)))
        if order == list(range(n)):
            order[0], order[-1] = order[-1], order[0]
        # Common totals permit a rectangular tiling; individual schedules are independent.
        extra_cap = 10 if test else 6
        row_totals = [h + int(rng.integers(1, min(extra_cap, 31 // nr - 1 - h) + 1))
                      for _ in range(nr)]
        col_totals = [w + int(rng.integers(1, min(extra_cap, 31 // nc - 1 - w) + 1))
                      for _ in range(nc)]
        raw = []
        for slot, index in enumerate(order):
            r, c = divmod(slot, nc)
            rows = _schedule(rng, h, row_totals[r], bool(witness))
            cols = _schedule(rng, w, col_totals[c], bool(witness))
            raw.append(np.repeat(np.repeat(compact[index], rows, axis=0), cols, axis=1))
        # B must matter to A's comparison, rather than just shrinking the answer.
        raw_signatures = [_signatures(p, bg, axis) for p in raw]
        if _orders(raw_signatures):
            continue
        inp = _tile(raw, (nr, nc), divider)
        out = _strip(compact, range(n), axis, divider)
        # Ground truth comes from the constructed latent panels, independently
        # of solve. Decode is used only to check the observable construction.
        parsed = _decode(inp)
        if not np.array_equal(_strip(parsed["panels"], parsed["order"], axis, divider), out):
            raise AssertionError("latent construction and input decode disagree")
        return {"input": inp, "output": out}
    raise RuntimeError("bounded pair construction exhausted")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train_count, test_count = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    first_axis = _pick(rng, ("H", "V"))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", train_count), ("test", test_count)):
        for i in range(count):
            test = split == "test"
            if not test and i == 0:
                n, axis, witness, color_mode = 3, first_axis, "single", "shared"
            elif not test and i == 1:
                n = 4
                axis = "V" if first_axis == "H" else "H"
                witness, color_mode = "extended", "multi"
            else:
                n = int(rng.integers(5, 7) if test else rng.integers(2, 5))
                axis, witness = _pick(rng, ("H", "V")), ""
                color_mode = _pick(rng, ("shared", "partial", "distinct", "multi"))
            for _ in range(20):
                pair = _make_pair(rng, n, axis, witness, color_mode, test)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    inp, out = (_grid(pair[role]) for role in ("input", "output"))
    d = _decode(inp)
    panels, bg = d["panels"], d["background"]
    expected = _strip(panels, d["order"], d["axis"], d["divider"])
    _require(np.array_equal(out, expected), "output must be the intact compact port chain")
    # Independently split output dividers, then conserve exact compact matrices.
    output_parts = list(_partitions(out))
    _require(len(output_parts) == 1, "output divider partition must be unique")
    actual = output_parts[0]
    target_layout = (1, len(panels)) if d["axis"] == "H" else (len(panels), 1)
    _require(actual["layout"] == target_layout, "output must be a single strip")
    _require(Counter(map(_key, panels)) == Counter(map(_key, actual["raw"])),
             "compact panel values and separated repeats must be preserved")
    schedules = []
    for large, small in zip(d["raw"], panels):
        _, rows = _runs(large, 0)
        _, cols = _runs(large, 1)
        restored = np.repeat(np.repeat(small, rows, axis=0), cols, axis=1)
        _require(np.array_equal(large, restored), "not a lossless row/column enlargement")
        schedules.append((rows, cols))
    _require(not _orders([_signatures(p, bg, d["axis"]) for p in d["raw"]]),
             "raw ports should fail: compression must causally enable matching")
    _require(d["order"] != tuple(range(len(panels))), "reading order must not solve the puzzle")
    _require(np.array_equal(np.unique(inp), np.unique(out)), "colors must be preserved")
    oriented = [p if d["axis"] == "H" else p.T for p in panels]
    color_mismatch = any(not np.array_equal(oriented[a][list(d["signatures"][a][1]), -1],
                                            oriented[b][list(d["signatures"][a][1]), 0])
                         for a, b in zip(d["order"], d["order"][1:]))
    return dict(d, schedules=schedules, color_mismatch=color_mismatch)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "exact train/test keys required")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "exact input/output keys required")
            a = _grid(pair["input"])
            key = _key(a)
            _require(key not in seen, "inputs must be distinct")
            seen.add(key)
            records[split].append(_analyze_pair(pair))
    train, tests = records["train"], records["test"]
    _require({d["axis"] for d in train} == {"H", "V"}, "training must teach both axes")
    _require(any(all(len(set(rows)) > 1 and len(set(cols)) > 1
                     and max(rows) > 1 and max(cols) > 1 for rows, cols in d["schedules"])
                 for d in train), "teach independent nonuniform bands on both axes")
    signatures = {s for d in train for a, b in d["signatures"] for s in (a, b) if s}
    _require(any(len(s) == 2 and s[1] == s[0] + 1 for s in signatures), "teach thick ports")
    _require(any(len(s) == 2 and s[1] > s[0] + 1 for s in signatures), "teach separated ports")
    _require(any(_collapse((p != d["background"]).astype(np.int8)).shape != p.shape
                 for d in train for p in d["panels"]),
             "teach literal-vector equality rather than occupancy compression")
    _require(any(len({d["signatures"][i][1] for i in d["order"][:-1]})
                 < len(d["panels"]) - 1 for d in train), "teach a forced repeated join")
    _require(any(d["color_mismatch"] for d in train), "teach port matches across different literal colors")
    _require(any(len({tuple(map(int, np.unique(p[p != d["background"]])))
                      for p in d["panels"]}) == 1
                 and len(np.unique(d["panels"][0])) == 2 for d in train), "teach color-independent ordering")
    _require(all(len(d["panels"]) > max(len(t["panels"]) for t in train) for d in tests),
             "tests must extend the training chain lengths")


def _puzzle_digest(puzzle):
    serial = {s: [{role: np.asarray(p[role]).tolist() for role in ("input", "output")}
                  for p in puzzle[s]] for s in ("train", "test")}
    return hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """Bounded, streaming 42-seed checks. Returns small descriptive measurements."""
    shapes, layouts, latent_shapes, schedules, palettes, geometry, puzzles = (set() for _ in range(7))
    counts, axes, backgrounds, dividers = set(), set(), set(), set()
    repeated_ports = local_joins = repeated_vectors = 0
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _analyze_pair(pair)
                pairs += 1
                shapes.add(pair["input"].shape)
                layouts.add(d["layout"])
                latent_shapes.add(d["panels"][0].shape)
                counts.add(len(d["panels"]))
                axes.add(d["axis"])
                backgrounds.add(d["background"])
                dividers.add(d["divider"])
                palettes.add(tuple(map(int, np.unique(pair["input"]))))
                schedules.update(d["schedules"])
                joins = [d["signatures"][i][1] for i in d["order"][:-1]]
                repeated_ports += len(set(joins)) < len(joins)
                for large, small in zip(d["raw"], d["panels"]):
                    geometry.add(hashlib.sha256((small != d["background"]).tobytes()
                                                + bytes(small.shape)).digest())
                    # Compare whole vectors by a second, boolean-mask formulation.
                    rk = np.r_[True, np.any(large[1:] != large[:-1], axis=1)]
                    ck = np.r_[True, np.any(large[:, 1:] != large[:, :-1], axis=0)]
                    _require(np.array_equal(large[rk][:, ck], small), "full-vector compression check failed")
                    repeated_vectors += any(np.array_equal(small[i], small[j])
                                            for i in range(len(small)) for j in range(i + 2, len(small)))
                    local_joins += bool(np.any(small[:-1] == small[1:])
                                        and np.any(small[:, :-1] == small[:, 1:]))
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solver mismatch")
        if seed in (0, 17, 41):
            _require(_puzzle_digest(generate_puzzle(seed)) == _puzzle_digest(puzzle), "seed not deterministic")
    for value, minimum, name in ((len(puzzles), 42, "puzzles"), (len(shapes), 40, "physical shapes"),
                                  (len(latent_shapes), 15, "latent shapes"), (len(layouts), 5, "layouts"),
                                  (len(schedules), 80, "independent schedules"), (len(palettes), 30, "palettes"),
                                  (len(geometry), 150, "motif geometries"), (repeated_ports, 1, "forced self-loops"),
                                  (repeated_vectors, 1, "separated repeated vectors")):
        _require(value >= minimum, f"insufficient {name}: {value}")
    _require(counts == set(range(2, 7)) and axes == {"H", "V"}, "missing counts or axes")
    _require(backgrounds == set(range(10)) and dividers == set(range(10)), "missing color roles")
    # Exhaust all legal counts, including the smallest two-demonstration case.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(81000 + 10 * a + b, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "count not honored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    # Literal repeated vectors separated by another vector survive on both axes.
    witness = np.array([[0, 0, 0], [1, 2, 1], [0, 0, 0]], dtype=np.int8)
    expanded = np.repeat(np.repeat(witness, [2, 3, 1], axis=0), [3, 1, 2], axis=1)
    _require(np.array_equal(_collapse(expanded), witness), "separated repeats were deleted")
    _require(len(_orders([((), (1,)), ((1,), (1,)), ((1,), ())])) == 1, "forced loop lost")
    _require(len(_orders([((), (1,)), ((1,), (1,)), ((1,), (1,)), ((1,), ())])) == 2,
             "ambiguous loops must be detected")
    # A wrong reconstruction that erases detail must not pass the validator.
    bad = generate_puzzle(37, 2, 1)
    bad["train"][0]["output"] = bad["train"][0]["output"].copy()
    bad["train"][0]["output"][1, 1] = (int(bad["train"][0]["output"][1, 1]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError("changed compact content accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(puzzles),
            "input_shapes": len(shapes), "compact_shapes": len(latent_shapes),
            "layouts": len(layouts), "schedules": len(schedules), "palettes": len(palettes),
            "motif_geometries": len(geometry), "repeated_port_pairs": repeated_ports,
            "separated_repeated_row_panels": repeated_vectors, "local_join_panels": local_joins}
