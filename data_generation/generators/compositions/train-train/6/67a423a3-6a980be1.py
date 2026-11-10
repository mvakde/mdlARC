"""Shadow rendezvous: equal-label projected footprints acquire hollow halos.

Only standard-library modules and NumPy are used.  Solving parses the input;
there is no dependency on seeds, generated examples, or puzzle-level state.
"""
import numpy as np


HOST_SEEDS = (0, 206128589, 719140879, 518501233, 276332854, 571071951,
              523811751, 215158566, 864540246, 773955202, 241054317,
              290654342, 822711401, 615484319, 383424340, 162530622,
              368399578, 358001004, 546363815, 691594933, 733301305,
              21335314, 72103442, 1146814, 770496202, 8382789, 175668312,
              114119071, 903214801, 598899832, 745811558, 756572961,
              675892738, 992477693, 544446138, 871341254, 281245284,
              828595575, 598166927, 537519865, 254190505, 108719267)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grids must be two dimensional with sides 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid colors must be integers 0..9")
    return a


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _count(name, value, low, high):
    _require(not isinstance(value, (bool, np.bool_)) and
             isinstance(value, (int, np.integer)) and low <= value <= high,
             f"{name} must be an integer in {low}..{high}")
    return int(value)


def _parse_runs(a, start, end, vertical):
    """Parse the payload outside the hub; intervals use exclusive end points."""
    bg, g = a["background"], a["grid"]
    h0, h1, v0, v1 = a["guides"]
    k = v1 - v0 - 1 if vertical else h1 - h0 - 1
    sequence = g[start:end, v0+1:v1] if vertical else g[h0+1:h1, start:end].T
    result = []
    i = 0
    while i < len(sequence):
        cells = sequence[i]
        _require(np.all(cells == cells[0]), "band slice is not monochrome")
        color = int(cells[0])
        if color == bg:
            i += 1
            continue
        _require(color not in (a["frame"], a["guide"]), "invalid payload color")
        j = i + 1
        while j < len(sequence) and np.all(sequence[j] == color):
            j += 1
        _require(j - i == k, "payload must be a full-thickness square")
        _require(j == len(sequence) or np.all(sequence[j] == bg),
                 "payload squares require a separating background slice")
        result.append((start + i, start + j, color))
        i = j
    return result


def _analyze(input_grid):
    g = _grid(input_grid)
    H, W = g.shape
    _require(H >= 5 and W >= 5, "canvas too small for crossing bands")
    f = int(g[0, 0])
    _require(np.all(g[0] == f) and np.all(g[-1] == f) and
             np.all(g[:, 0] == f) and np.all(g[:, -1] == f),
             "input requires a uniform outer frame")
    candidates = []
    for color in sorted(set(map(int, np.unique(g))) - {f}):
        rows = [r for r in range(1, H-1) if np.all(g[r, 1:-1] == color)]
        cols = [c for c in range(1, W-1) if np.all(g[1:-1, c] == color)]
        if len(rows) == len(cols) == 2 and rows[1]-rows[0] >= 2 and cols[1]-cols[0] >= 2:
            candidates.append((color, *rows, *cols))
    _require(len(candidates) == 1, "guides must have one unique interpretation")
    guide, h0, h1, v0, v1 = candidates[0]
    bg = int(g[h0+1, v0+1])
    _require(bg not in (f, guide) and np.all(g[h0+1:h1, v0+1:v1] == bg),
             "hub must be uniformly background")
    a = {"grid": g, "frame": f, "guide": guide, "background": bg,
         "guides": (h0, h1, v0, v1), "h": h1-h0-1, "v": v1-v0-1}
    horizontal = _parse_runs(a, 1, v0, False) + _parse_runs(a, v1+1, W-1, False)
    vertical = _parse_runs(a, 1, h0, True) + _parse_runs(a, h1+1, H-1, True)
    _require(horizontal and vertical, "each orientation needs a source square")
    for runs, near, far in ((horizontal, v0, v1), (vertical, h0, h1)):
        _require(all(e <= near-1 or s >= far+2 for s, e, _ in runs),
                 "sources need a background slice next to the perpendicular guides")
    # Rebuild the entire observed input: no quadrant clutter or extra markers.
    template = np.full(g.shape, bg, dtype=np.int8)
    template[0] = template[-1] = f
    template[:, 0] = template[:, -1] = f
    template[[h0, h1], 1:-1] = guide
    template[1:-1, [v0, v1]] = guide
    for s, e, c in horizontal:
        template[h0+1:h1, s:e] = c
    for s, e, c in vertical:
        template[s:e, v0+1:v1] = c
    _require(np.array_equal(g, template), "cells outside the cross must be background")
    a["horizontal"], a["vertical"] = horizontal, vertical
    return a


def _base(a):
    out = a["grid"].copy()
    out[0] = out[-1] = a["background"]
    out[:, 0] = out[:, -1] = a["background"]
    h0, h1, v0, v1 = a["guides"]
    out[[h0, h1], :] = a["guide"]
    out[:, [v0, v1]] = a["guide"]
    return out


def _halo_mask(a, ignore_labels=False, points=False):
    mask = np.zeros(a["grid"].shape, dtype=bool)
    for c0, c1, hc in a["horizontal"]:
        for r0, r1, vc in a["vertical"]:
            if hc != vc and not ignore_labels:
                continue
            if points:
                rr, cc = (r0+r1-1)//2, (c0+c1-1)//2
                r0, r1, c0, c1 = rr, rr+1, cc, cc+1
            # Four strips, with corners included exactly in top/bottom.
            mask[r0-1, c0-1:c1+1] = True
            mask[r1, c0-1:c1+1] = True
            mask[r0:r1, c0-1] = True
            mask[r0:r1, c1] = True
    return mask


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every output cell solely from the visible input."""
    a = _analyze(input_grid)
    out = _base(a)
    out[_halo_mask(a)] = a["frame"]
    return out


def _sharing(a, name):
    """Whether same-label neighboring squares have exactly one blank slice."""
    runs = a[name]
    return any(c == d and s-e == 1
               for (_, e, c), (s, _, d) in zip(runs, runs[1:]))


def _minimum(n, size, gap):
    return n*size + max(0, n-1)*gap + (1 if n else 0)


def _axis_options(n, size, center, gap, all_arms):
    sides = range(1, n) if all_arms else range(n+1)
    return [left for left in sides
            if center+4+_minimum(left, size, gap)+_minimum(n-left, size, gap) <= 30]


def _axis(rng, n, size, center, touch, all_arms):
    """Construct an axis with its central guide band and source intervals."""
    gap = 1 if touch else 3
    sides = _axis_options(n, size, center, gap, all_arms)
    _require(sides, "axis has no fitting placement")
    left = int(rng.choice(sides))
    counts = [left, n-left]
    # Pick two adjacent sources on one arm. The common repeated label goes here.
    eligible = [(side, i) for side, count in enumerate(counts) for i in range(count-1)]
    side, i = eligible[int(rng.integers(len(eligible)))]
    repeated = (i, i+1) if side == 0 else (left+i, left+i+1)
    pieces = []
    for arm, count in enumerate(counts):
        pieces.append({"far": 0, "near": 1 if count else 0,
                       "gaps": [gap]*(max(0, count-1))})
    slots = []
    for arm in range(2):
        slots.extend([(arm, "far", None), (arm, "near", None)])
        for j in range(len(pieces[arm]["gaps"])):
            if not (touch and (arm, j) == (side, i)):
                slots.append((arm, "gaps", j))
    base_length = center+4+sum(_minimum(count, size, gap) for count in counts)
    slack = int(rng.integers(0, 31-base_length))
    extras = rng.multinomial(slack, [1/len(slots)]*len(slots))
    for (arm, field, j), extra in zip(slots, extras):
        if j is None:
            pieces[arm][field] += int(extra)
        else:
            pieces[arm][field][j] += int(extra)
    lengths = [count*size + sum(piece["gaps"]) + piece["far"] + piece["near"]
               for count, piece in zip(counts, pieces)]
    g0, g1 = lengths[0]+1, lengths[0]+center+2
    dimension = sum(lengths)+center+4
    intervals = []
    for arm, (count, piece) in enumerate(zip(counts, pieces)):
        p = 1+piece["far"] if arm == 0 else g1+1+piece["near"]
        for j in range(count):
            intervals.append((p, p+size))
            p += size
            if j < count-1:
                p += piece["gaps"][j]
    return dimension, g0, g1, intervals, repeated


def _labels(rng, n, palette, repeated):
    values = [None]*n
    values[repeated[0]] = values[repeated[1]] = palette[0]
    free = [i for i in range(n) if values[i] is None]
    rng.shuffle(free)
    for index, c in zip(free, palette[1:]):
        values[index] = c
    for index in free[len(palette)-1:]:
        values[index] = int(rng.choice(palette))
    return values


def _configs(label_count, h_values, v_values, touch_h, touch_v, all_arms,
             max_count=5):
    choices = []
    for h in h_values:
        for v in v_values:
            for nh in range(label_count+1, max_count+1):
                if not _axis_options(nh, h, v, 1 if touch_h else 3, all_arms):
                    continue
                for nv in range(label_count+1, max_count+1):
                    if _axis_options(nv, v, h, 1 if touch_v else 3, all_arms):
                        choices.append((h, v, nh, nv))
    return choices


def _render(rng, config, colors, touch_h, touch_v, all_arms):
    h, v, nh, nv = config
    bg, f, guide, *payload = colors
    W, v0, v1, hs, hp = _axis(rng, nh, h, v, touch_h, all_arms)
    H, h0, h1, vs, vp = _axis(rng, nv, v, h, touch_v, all_arms)
    hc, vc = _labels(rng, nh, payload, hp), _labels(rng, nv, payload, vp)
    inp = np.full((H, W), bg, dtype=np.int8)
    inp[0] = inp[-1] = f
    inp[:, 0] = inp[:, -1] = f
    inp[[h0, h1], 1:-1] = guide
    inp[1:-1, [v0, v1]] = guide
    for (s, e), c in zip(hs, hc):
        inp[h0+1:h1, s:e] = c
    for (s, e), c in zip(vs, vc):
        inp[s:e, v0+1:v1] = c
    # Construction uses source metadata; solve independently recovers intervals.
    out = np.full((H, W), bg, dtype=np.int8)
    out[[h0, h1], :] = guide
    out[:, [v0, v1]] = guide
    for (s, e), c in zip(hs, hc):
        out[h0+1:h1, s:e] = c
    for (s, e), c in zip(vs, vc):
        out[s:e, v0+1:v1] = c
    for (c0, c1), x in zip(hs, hc):
        for (r0, r1), y in zip(vs, vc):
            if x == y:
                for r in range(r0-1, r1+1):
                    for c in range(c0-1, c1+1):
                        if r < r0 or r >= r1 or c < c0 or c >= c1:
                            out[r, c] = f
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count("num_train", num_train, 2, 10)
    nv = _count("num_test", num_test, 1, 4)
    _require(nt > nv, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    label_count = int(rng.integers(2, 4))
    base_colors = list(map(int, rng.permutation(10)[:label_count+3]))
    stable = bool(rng.integers(2))
    teaching_axis = int(rng.integers(2))
    train_palettes = []
    seen = set()
    puzzle = {"train": [], "test": []}
    for split, number in (("train", nt), ("test", nv)):
        for i in range(number):
            if split == "train" and i == 0:
                th, tv, arms = False, False, True
                choices = _configs(label_count, [1], [1], th, tv, arms)
            elif split == "train" and i == 1:
                th, tv, arms = teaching_axis == 0, teaching_axis == 1, True
                choices = [c for c in _configs(label_count, range(1, 4), range(1, 4),
                                                th, tv, arms) if c[0] != c[1] and min(c[:2]) >= 2]
            elif split == "train":
                mode = int(rng.integers(3))
                th, tv, arms = mode == 1, mode == 2, True
                choices = _configs(label_count, range(1, 5), range(1, 5), th, tv, arms)
            else:
                th, tv, arms = True, True, False
                choices = _configs(label_count, range(1, 5), range(1, 5), th, tv, arms,
                                   max_count=6)
            if split == "test" and rng.random() < 0.45:
                colors = train_palettes[int(rng.integers(len(train_palettes)))].copy()
            elif stable:
                colors = base_colors.copy()
                colors[1] = int(rng.choice([c for c in range(10)
                                           if c not in colors[:1]+colors[2:]]))
            else:
                colors = list(map(int, rng.permutation(10)[:label_count+3]))
            if split == "train" and i == 1 and colors[1] == train_palettes[0][1]:
                colors[1] = int(rng.choice([c for c in range(10)
                                           if c not in colors and c != train_palettes[0][1]]))
            for _ in range(80):
                config = choices[int(rng.integers(len(choices)))]
                pair = _render(rng, config, colors, th, tv, arms)
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("unique-pair sampling exhausted")
            if split == "train":
                train_palettes.append(colors)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _cellwise_expected(a):
    """Geometric predicate, rather than the renderer's rectangle drawing loop."""
    out = _base(a)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            for c0, c1, hc in a["horizontal"]:
                for r0, r1, vc in a["vertical"]:
                    if hc == vc:
                        row_distance = max(r0-r, r-(r1-1), 0)
                        col_distance = max(c0-c, c-(c1-1), 0)
                        if max(row_distance, col_distance) == 1:
                            out[r, c] = a["frame"]
                            break
    return out


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             "pair requires exactly input and output")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "input/output sizes must agree")
    a = _analyze(inp)
    expected = _base(a)
    halo = _halo_mask(a)
    expected[halo] = a["frame"]
    _require(np.array_equal(out, expected), "incorrect halo union or changed source")
    h0, h1, v0, v1 = a["guides"]
    _require(not np.any(halo[h0:h1+1, :]) and not np.any(halo[:, v0:v1+1]),
             "halos must be outside both source bands")
    for c0, c1, _ in a["horizontal"]:
        for r0, r1, _ in a["vertical"]:
            _require(np.all(out[r0:r1, c0:c1] == a["background"]),
                     "all matched or unmatched footprint interiors must stay background")
    # Concrete alternatives which omit one inherited mechanism.
    no_a = _base(a)
    strips = _base(a)
    for c0, c1, hc in a["horizontal"]:
        strips[:h0, c0:c1] = strips[h1+1:, c0:c1] = a["frame"]
        for r0, r1, vc in a["vertical"]:
            if hc == vc:
                no_a[r0:r1, c0:c1] = a["frame"]
    for r0, r1, _ in a["vertical"]:
        strips[r0:r1, :v0] = strips[r0:r1, v1+1:] = a["frame"]
    no_b = _base(a)
    for r in (h0, h1):
        for c in (v0, v1):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr, dc) != (0, 0):
                        no_b[r+dr, c+dc] = a["frame"]
    _require(not np.array_equal(out, no_a) and not np.array_equal(out, strips),
             "omitting A must change the answer")
    _require(not np.array_equal(out, no_b), "omitting B must change the answer")
    filled = expected.copy()
    for c0, c1, hc in a["horizontal"]:
        for r0, r1, vc in a["vertical"]:
            if hc == vc:
                filled[r0:r1, c0:c1] = a["frame"]
    _require(not np.array_equal(out, filled), "preserved interior must matter")
    retained_border = expected.copy()
    retained_border[0] = retained_border[-1] = a["frame"]
    retained_border[:, 0] = retained_border[:, -1] = a["frame"]
    _require(not np.array_equal(out, retained_border), "border removal must matter")
    if a["h"] != 1 or a["v"] != 1:
        point_answer = _base(a)
        point_answer[_halo_mask(a, points=True)] = a["frame"]
        _require(not np.array_equal(out, point_answer), "footprint sizes must matter")
    all_pairs = _base(a)
    all_pairs[_halo_mask(a, ignore_labels=True)] = a["frame"]
    a["labels_matter"] = not np.array_equal(all_pairs, out)
    return a


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle requires exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "train and test must be lists")
    nt, nv = len(puzzle["train"]), len(puzzle["test"])
    _count("num_train", nt, 2, 10)
    _count("num_test", nv, 1, 4)
    _require(nt > nv, "num_train must exceed num_test")
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            a = _check_pair(pair)
            key = _key(a["grid"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            hlabels = set(c for _, _, c in a["horizontal"])
            vlabels = set(c for _, _, c in a["vertical"])
            _require(hlabels == vlabels and 2 <= len(hlabels) <= 3,
                     "sampled examples require two or three labels on each orientation")
            _require(a["labels_matter"], "sample must distinguish color matching")
            a["sharing"] = (_sharing(a, "horizontal"), _sharing(a, "vertical"))
            records[split].append(a)
    train = records["train"]
    for a in train:
        h0, h1, v0, v1 = a["guides"]
        _require(any(e < v0 for _, e, _ in a["horizontal"]) and
                 any(s > v1 for s, _, _ in a["horizontal"]) and
                 any(e < h0 for _, e, _ in a["vertical"]) and
                 any(s > h1 for s, _, _ in a["vertical"]),
                 "training must demonstrate all four source arms")
    for a in records["train"] + records["test"]:
        hc = [c for _, _, c in a["horizontal"]]
        vc = [c for _, _, c in a["vertical"]]
        _require(any(hc.count(c) >= 2 and vc.count(c) >= 2 for c in set(hc)),
                 "sample must demonstrate repeated labels with multiple partners")
    _require(any(a["h"] == a["v"] == 1 and all(
        s-e >= 3 for name in ("horizontal", "vertical")
        for (_, e, _), (s, _, _) in zip(a[name], a[name][1:])) for a in train),
             "missing isolated unit-footprint lesson")
    _require(any(a["h"] != a["v"] and min(a["h"], a["v"]) >= 2 and sum(a["sharing"]) == 1 for a in train),
             "missing rectangular one-axis-sharing lesson")
    _require(len({a["frame"] for a in train}) >= 2, "demonstrate frame color transfer")
    _require(all(sum(a["sharing"]) <= 1 for a in train), "training shift violated")
    _require(all(all(a["sharing"]) for a in records["test"]),
             "tests must combine sharing on both axes")
    # The same literal shared label must cause an actual 2-by-2 mesh in tests.
    for a in records["test"]:
        shared_labels = []
        for name in ("horizontal", "vertical"):
            shared_labels.append({c for (_, e, c), (s, _, d) in
                                  zip(a[name], a[name][1:]) if c == d and s-e == 1})
        _require(bool(shared_labels[0] & shared_labels[1]), "missing connected mesh")
    return None


def _puzzle_key(p):
    return tuple(tuple((_key(np.asarray(pair["input"])), _key(np.asarray(pair["output"])))
                       for pair in p[s]) for s in ("train", "test"))


def self_test():
    """Bounded checks; returns descriptive measurements, not semantic approval."""
    shapes, normalized, layouts, palettes, signatures = set(), set(), set(), set(), set()
    thicknesses, counts, label_counts, arms, boundary = set(), set(), set(), set(), set()
    pairs = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        _require(_puzzle_key(p) == _puzzle_key(generate_puzzle(seed)), "nondeterministic seed")
        signatures.add(_puzzle_key(p))
        for split in ("train", "test"):
            for pair in p[split]:
                a = _analyze(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve disagrees")
                _require(np.array_equal(_cellwise_expected(a), pair["output"]),
                         "distance-based cell check disagrees")
                h0, h1, v0, v1 = a["guides"]
                shape = pair["input"].shape
                shapes.add(shape)
                thicknesses.add((a["h"], a["v"]))
                counts.add((len(a["horizontal"]), len(a["vertical"])))
                label_counts.add(len({c for _, _, c in a["horizontal"]}))
                arms.add((sum(e < v0 for _, e, _ in a["horizontal"]),
                          sum(s > v1 for s, _, _ in a["horizontal"]),
                          sum(e < h0 for _, e, _ in a["vertical"]),
                          sum(s > h1 for s, _, _ in a["vertical"])))
                layouts.add((shape, a["guides"], tuple(a["horizontal"]), tuple(a["vertical"])))
                palettes.add((a["background"], a["frame"], a["guide"]))
                mapping = {}
                norm = tuple(mapping.setdefault(int(c), len(mapping)) for c in pair["input"].flat)
                normalized.add((shape, norm))
                mask = _halo_mask(a)
                for name, cells in (("top", mask[0]), ("bottom", mask[-1]),
                                    ("left", mask[:, 0]), ("right", mask[:, -1])):
                    if np.any(cells):
                        boundary.add(name)
                pairs += 1
        # Structural and color symmetries must not depend on generation.
        src, tgt = p["test"][0]["input"], p["test"][0]["output"]
        for k in range(4):
            _require(np.array_equal(solve(np.rot90(src, k)), np.rot90(tgt, k)),
                     "rotation equivariance failed")
        _require(np.array_equal(solve(src.T), tgt.T), "transpose equivariance failed")
        permutation = np.random.default_rng(seed+1).permutation(10)
        _require(np.array_equal(solve(permutation[src]), permutation[tgt]),
                 "color equivariance failed")
    # Every supported explicit count, including minimum demonstrations.
    for nt in range(2, 11):
        for nv in range(1, min(4, nt-1)+1):
            p = generate_puzzle(314159, nt, nv)
            _require(len(p["train"]) == nt and len(p["test"]) == nv, "counts not honored")
    for nt, nv in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nv)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    _solver_edge_cases()
    _require(len(signatures) == 42 and len(shapes) >= 50, "insufficient puzzle/shape variation")
    _require(len(normalized) >= 195 and len(layouts) >= 195, "too few structural layouts")
    _require(len(palettes) >= 35 and label_counts == {2, 3}, "palette variation too narrow")
    _require(len(thicknesses) >= 12 and len(counts) >= 8 and len(arms) >= 30,
             "insufficient scale/count/arm variation")
    _require(boundary == {"top", "bottom", "left", "right"}, "missing boundary contacts")
    return {"seeds": len(HOST_SEEDS), "pairs": pairs, "puzzles": len(signatures),
            "shapes": len(shapes), "normalized_inputs": len(normalized),
            "layouts": len(layouts), "palette_role_triples": len(palettes),
            "thickness_pairs": sorted(thicknesses), "source_count_pairs": sorted(counts),
            "arm_distributions": len(arms), "boundary_contacts": sorted(boundary),
            "verification": "authored geometric checks, not independent semantic approval"}


def _solver_edge_cases():
    # Not generated by the sampler: one-sided singleton sources, unmatched
    # labels, and thickness >4. Build inputs directly from the specification.
    for h, v, match in ((1, 1, True), (5, 2, True), (2, 5, False)):
        H, W = 23, 25
        h0, v0 = 2, 2
        h1, v1 = h0+h+1, v0+v+1
        r0, c0 = h1+3, v1+3
        inp = np.zeros((H, W), dtype=np.int8)
        inp[0] = inp[-1] = 8
        inp[:, 0] = inp[:, -1] = 8
        inp[[h0, h1], 1:-1] = 3
        inp[1:-1, [v0, v1]] = 3
        inp[h0+1:h1, c0:c0+h] = 2
        inp[r0:r0+v, v0+1:v1] = 2 if match else 6
        expected = inp.copy()
        expected[0] = expected[-1] = 0
        expected[:, 0] = expected[:, -1] = 0
        expected[[h0, h1], :] = 3
        expected[:, [v0, v1]] = 3
        if match:
            expected[r0-1:r0+v+1, c0-1:c0+h+1] = 8
            expected[r0:r0+v, c0:c0+h] = 0
        _require(np.array_equal(solve(inp), expected), "broader-family edge case failed")


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
