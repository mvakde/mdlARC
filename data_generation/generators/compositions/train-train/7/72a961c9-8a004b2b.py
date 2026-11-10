"""Hidden canopy: color-selective stalks reveal an aligned virtual blueprint.

Only standard-library modules and NumPy are used. All solving conventions are
local to the input; no generation parameters or training pairs enter solve().
"""
from __future__ import annotations

from collections import defaultdict
import json
import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and a.dtype != np.dtype(bool), "grid must be integer")
    _require(np.all((a >= 0) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _components(a, background, marker):
    """Monochromatic components above the carrier, excluding corner markers."""
    seen = set()
    result = []
    for r, c in np.argwhere((a != background) & (a != marker)):
        p = int(r), int(c)
        if p in seen:
            continue
        color = int(a[p])
        seen.add(p)
        stack, points = [p], []
        while stack:
            x, y = stack.pop()
            points.append((x, y))
            for xx, yy in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                q = xx, yy
                if (0 <= xx < a.shape[0] and 0 <= yy < a.shape[1]
                        and q not in seen and int(a[q]) == color):
                    seen.add(q)
                    stack.append(q)
        rows, cols = zip(*points)
        top, left = min(rows), min(cols)
        h, w = max(rows)-top+1, max(cols)-left+1
        _require(h == w and len(points) == h*h, "references must be solid separate squares")
        result.append((top, left, color, h))
    return result


def _alignments(bp, refs, k, shape, line, background):
    """Enumerate translations; k is measured from visible reference squares."""
    height, width = shape
    bh, bw = bp.shape
    rr, cc, color = refs[0]
    answers = set()
    for r, c in np.argwhere(bp == color):
        top, left = rr-int(r)*k, cc-int(c)*k
        # The virtual map is strictly inside the window and above a blank gap.
        if top < 1 or left < 1 or top+bh*k > line-1 or left+bw*k > width-1:
            continue
        ok = True
        for x, y, q in refs:
            dr, dc = x-top, y-left
            if dr % k or dc % k or not (0 <= dr//k < bh and 0 <= dc//k < bw):
                ok = False
                break
            if int(bp[dr//k, dc//k]) != q or q == background:
                ok = False
                break
        if ok:
            answers.add((top, left))
    return sorted(answers)


def _parse_candidate(a, bg, marker, bounds):
    t, b, l, r = bounds
    crop = a[t:b+1, l:r+1].copy()
    h, w = crop.shape
    _require(h >= 8 and w >= 7, "window too small")
    lines = [i for i in range(1, h-1) if np.all(crop[i] != bg)]
    _require(len(lines) == 1, "need one full carrier row")
    line = lines[0]
    vals, counts = np.unique(crop[line], return_counts=True)
    carrier = int(vals[np.argmax(counts)])
    _require(int(counts.max()) > w/2 and carrier not in (bg, marker), "carrier must be the majority")
    roots = [(int(c), int(crop[line, c])) for c in np.flatnonzero(crop[line] != carrier)]
    _require(2 <= len(roots) <= 8 and all(0 < c < w-1 and q not in (bg, marker) for c, q in roots), "invalid roots")
    _require(np.all(crop[line+1:][crop[line+1:] != marker] == bg), "objects below carrier")
    outside = np.ones(a.shape, dtype=bool)
    outside[t:b+1, l:r+1] = False
    points = np.argwhere(outside & (a != bg))
    _require(len(points) > 0, "missing external blueprint")
    bt, bl = points.min(axis=0)
    bb, br = points.max(axis=0)
    _require(outside[bt:bb+1, bl:br+1].all(), "blueprint overlaps window")
    bp = a[bt:bb+1, bl:br+1].copy()
    _require(2 <= bp.shape[0] <= 5 and 3 <= bp.shape[1] <= 6, "invalid blueprint bounds")
    payload = set(int(q) for q in np.unique(bp)) - {bg}
    _require(2 <= len(payload) <= 4 and not payload.intersection((marker, carrier)), "invalid palette roles")
    _require(set(q for _, q in roots).issubset(payload), "roots must use blueprint colors")
    _require(np.count_nonzero(a == carrier) == np.count_nonzero(crop[line] == carrier), "carrier color outside line")
    components = _components(crop[:line], bg, marker)
    _require(2 <= len(components) <= 5, "need two to five reference squares")
    scales = {x[3] for x in components}
    _require(len(scales) == 1 and 2 <= next(iter(scales)) <= 4, "inconsistent reference scale")
    k = next(iter(scales))
    refs = [(x, y, q) for x, y, q, _ in components]
    _require(all(q in payload for _, _, q in refs), "unknown reference color")
    alignment = _alignments(bp, refs, k, crop.shape, line, bg)
    _require(len(alignment) == 1, "reference alignment is ambiguous")
    top, left = alignment[0]
    ref_cells = {(int((x-top)//k), int((y-left)//k)) for x, y, _ in refs}
    return dict(crop=crop, bp=bp, background=bg, marker=marker, carrier=carrier,
                line=line, roots=roots, refs=refs, ref_cells=ref_cells, scale=k,
                top=top, left=left, bounds=bounds,
                blueprint_bounds=(int(bt), int(bb), int(bl), int(br)))


def _analyze(input_grid):
    a = _grid(input_grid)
    vals, counts = np.unique(a, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, "background must be the unique modal color")
    bg = int(vals[np.argmax(counts)])
    answers = []
    for q in vals:
        marker = int(q)
        if marker == bg:
            continue
        p = np.argwhere(a == marker)
        if len(p) != 4:
            continue
        rs, cs = sorted(set(map(int, p[:, 0]))), sorted(set(map(int, p[:, 1])))
        if len(rs) != 2 or len(cs) != 2:
            continue
        if set(map(tuple, p)) != {(x, y) for x in rs for y in cs}:
            continue
        try:
            answers.append(_parse_candidate(a, bg, marker, (rs[0], rs[1], cs[0], cs[1])))
        except AssertionError:
            continue
    _require(len(answers) == 1, "need exactly one interpretable four-corner window")
    return answers[0]


def _targets(info, mode="normal"):
    bp, bg, k = info["bp"], info["background"], info["scale"]
    chosen = []
    for column, color in info["roots"]:
        c = (column-info["left"])//k
        rows = []
        if 0 <= c < bp.shape[1]:
            for r in range(bp.shape[0]):
                if int(bp[r, c]) == bg:
                    continue
                if mode != "ignore_color" and int(bp[r, c]) != color:
                    continue
                if mode == "visible_only" and (r, c) not in info["ref_cells"]:
                    continue
                rows.append(r)
        chosen.append(None if not rows else (min(rows) if mode == "farthest" else max(rows), c))
    return chosen


def _render(info, mode="normal"):
    out = info["crop"].copy()
    k, bg = info["scale"], info["background"]
    targets = _targets(info, mode)
    # Stems go behind every existing reference and every subsequently revealed cap.
    for (column, _), cell in zip(info["roots"], targets):
        if cell is None:
            continue
        bottom = info["top"]+(cell[0]+1)*k
        segment = out[bottom:info["line"], column]
        segment[segment == bg] = info["carrier"]
    for r, c in set(x for x in targets if x is not None):
        y, x = info["top"]+r*k, info["left"]+c*k
        out[y:y+k, x:x+k] = info["bp"][r, c]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer crop, roles, square scale and unique virtual alignment from input."""
    return _render(_analyze(input_grid))


def _palette(rng):
    colors = list(map(int, rng.permutation(10)))
    if rng.random() < .35:
        i = colors.index(0)
        colors[0], colors[i] = colors[i], colors[0]
    return tuple(colors[:7])  # background, carrier, corner, four potential payloads


def _blueprint(rng, h, w, bg, colors):
    mode = int(rng.integers(5))
    if mode == 0:
        mask = np.ones((h, w), dtype=bool)
    elif mode == 1 and h >= 3:
        mask = np.ones((h, w), dtype=bool)
        mask[1:-1, 1:-1] = False
    else:
        mask = rng.random((h, w)) < float(rng.uniform(.42, .85))
    mask[0, int(rng.integers(w))] = True
    mask[-1, int(rng.integers(w))] = True
    mask[int(rng.integers(h)), 0] = True
    mask[int(rng.integers(h)), -1] = True
    cells = list(map(tuple, np.argwhere(mask)))
    _require(len(cells) >= 6, "too few canopy cells")
    ncolors = int(rng.integers(2, min(4, len(cells)//2)+1))
    active = colors[:ncolors]
    labels = list(active)*2
    labels += [int(rng.choice(active)) for _ in range(len(cells)-len(labels))]
    rng.shuffle(labels)
    bp = np.full((h, w), bg, dtype=np.int8)
    for p, q in zip(cells, labels):
        bp[p] = q
    return bp


def _options(bp, bg):
    options = []
    for c in range(bp.shape[1]):
        for q in sorted(set(map(int, bp[:, c])) - {bg}):
            rows = list(map(int, np.flatnonzero(bp[:, c] == q)))
            r = max(rows)
            filtered = any(int(bp[y, c]) not in (bg, q) for y in range(r+1, bp.shape[0]))
            options.append((c, q, r, filtered, len(rows) > 1))
    return options


def _layout(rng, crop, bp, bg):
    """Place the external blueprint on any feasible side, with independent offsets."""
    h, w = crop.shape
    bh, bw = bp.shape
    sides = list(map(int, rng.permutation(4)))
    for side in sides:
        gap = int(rng.integers(1, 4))
        if side < 2:
            bt = -bh-gap if side == 0 else h+gap
            bl = int(rng.integers(-bw+1, w))
        else:
            bl = -bw-gap if side == 2 else w+gap
            bt = int(rng.integers(-bh+1, h))
        minr, minc = min(0, bt), min(0, bl)
        maxr, maxc = max(h, bt+bh), max(w, bl+bw)
        if maxr-minr > 30 or maxc-minc > 30:
            continue
        pad = [int(rng.integers(0, 3)) for _ in range(4)]
        # Reduce padding, never geometry, to satisfy the canvas bound.
        if maxr-minr+pad[0]+pad[1] > 30:
            pad[0] = pad[1] = 0
        if maxc-minc+pad[2]+pad[3] > 30:
            pad[2] = pad[3] = 0
        out = np.full((maxr-minr+pad[0]+pad[1], maxc-minc+pad[2]+pad[3]), bg, dtype=np.int8)
        wr, wc = pad[0]-minr, pad[2]-minc
        out[wr:wr+h, wc:wc+w] = crop
        out[wr+bt:wr+bt+bh, wc+bl:wc+bl+bw] = bp
        return out
    raise AssertionError("no blueprint side fits")


def _make_pair(rng, scale, nroots, kind, palette):
    bg, carrier, marker = palette[:3]
    for _ in range(900):
        try:
            h = int(rng.integers(3 if kind == "depth" else 2, 6))
            w = int(rng.integers(3, 7))
            bp = _blueprint(rng, h, w, bg, palette[3:])
            options = _options(bp, bg)
            if kind == "color":
                eligible = [o for o in options if o[3]]
                _require(eligible, "need a color-selective pass")
                first = eligible[int(rng.integers(len(eligible)))]
                # Two colors in the same virtual column show that each root searches
                # for its own color, rather than merely stopping at the first tile.
                lower = [o for o in options if o[0] == first[0] and o[2] > first[2]]
                planned = [first, lower[int(rng.integers(len(lower)))]]
            elif kind == "depth":
                eligible = [(a, b) for a in options for b in options
                            if a[4] and a[1] == b[1] and a[2] != b[2] and a[0] != b[0]]
                _require(eligible, "need repeated color at different target depths")
                planned = list(eligible[int(rng.integers(len(eligible)))]) + [None]
            else:
                indices = rng.choice(len(options), 2, replace=False)
                planned = [options[int(i)] for i in indices]
            while len(planned) < nroots:
                planned.append(None if rng.random() < .16 else options[int(rng.integers(len(options)))])
            top = int(rng.integers(1, 4))
            left = int(rng.integers(2 if kind == "depth" else 1, 5))
            width = max(left+w*scale+int(rng.integers(1, 5)), 2*nroots+1)
            gap = int(rng.integers(1, 4))
            line = top+h*scale+gap
            height = line+int(rng.integers(1, 4))+1
            _require(height <= 28 and width <= 28, "window does not fit")
            roots, occupied_cols = [], set()
            payload = sorted(set(map(int, bp.flat)) - {bg})
            for option in planned:
                if option is None:
                    dormant = [(x, q) for x in range(1, width-1) if x not in occupied_cols for q in payload
                               if not (0 <= (x-left)//scale < w) or q not in bp[:, (x-left)//scale]]
                    _require(dormant, "no unused dormant location")
                    x, q = dormant[int(rng.integers(len(dormant)))]
                else:
                    c, q = option[:2]
                    choices = [x for x in range(left+c*scale, left+(c+1)*scale) if x not in occupied_cols]
                    _require(choices, "too many roots in one tile column")
                    x = int(rng.choice(choices))
                occupied_cols.add(x)
                roots.append((x, q))
            crop = np.full((height, width), bg, dtype=np.int8)
            crop[0, 0] = crop[0, -1] = crop[-1, 0] = crop[-1, -1] = marker
            crop[line] = carrier
            for x, q in roots:
                crop[line, x] = q
            known = dict(crop=crop, bp=bp, background=bg, marker=marker, carrier=carrier,
                         line=line, roots=roots, scale=scale, top=top, left=left, ref_cells=set())
            selected = {x for x in _targets(known) if x is not None}
            cells = list(map(tuple, np.argwhere(bp != bg)))
            unselected = set(cells)-selected
            _require(unselected and selected, "need both reached and unreached tiles")
            # Hide the diagnostic root's target, plus an unselected tile. Spatial
            # references must therefore support selective inference, not copying.
            required_hidden = (planned[0][2], planned[0][0])
            untouched = list(sorted(unselected))[int(rng.integers(len(unselected)))]
            available = [p for p in cells if p not in (required_hidden, untouched)]
            _require(len(available) >= 2, "not enough reference choices")
            accepted = False
            for _ in range(55):
                nrefs = int(rng.integers(2, min(5, len(available))+1))
                refs_cells = [available[int(i)] for i in rng.choice(len(available), nrefs, replace=False)]
                if any(bp[a] == bp[b] and abs(a[0]-b[0])+abs(a[1]-b[1]) == 1
                       for i, a in enumerate(refs_cells) for b in refs_cells[i+1:]):
                    continue
                refs = [(top+r*scale, left+c*scale, int(bp[r, c])) for r, c in refs_cells]
                if _alignments(bp, refs, scale, crop.shape, line, bg) != [(top, left)]:
                    continue
                accepted = True
                break
            _require(accepted, "cannot make unique separated references")
            for r, c in refs_cells:
                y, x = top+r*scale, left+c*scale
                crop[y:y+scale, x:x+scale] = bp[r, c]
            known["ref_cells"] = set(refs_cells)
            expected = _render(known)
            _require(not np.array_equal(expected, _render(known, "visible_only")), "visible-only shortcut survives")
            if kind == "color":
                _require(not np.array_equal(expected, _render(known, "ignore_color")), "color diagnostic ineffective")
            if kind == "depth":
                _require(not np.array_equal(expected, _render(known, "farthest")), "nearest diagnostic ineffective")
            inp = _layout(rng, crop, bp, bg)
            info = _analyze(inp)
            _require(np.array_equal(expected, _render(info)), "input parsing changed construction")
            return {"input": inp, "output": expected}
        except AssertionError:
            continue
    raise RuntimeError(f"could not construct {kind} example at scale {scale}")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, "num_train")
    ne = _count(num_test, 1, 4, "num_test")
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    base_palette = _palette(rng)
    vary_palette = bool(rng.integers(2))
    scales = list(map(int, rng.permutation([2, 3])))
    plans = [("color", scales[0], int(rng.integers(2, 5))),
             ("depth", scales[1], int(rng.integers(3, 5)))]
    plans += [("general", int(rng.choice([2, 3])), int(rng.integers(2, 5))) for _ in range(nt-2)]
    rng.shuffle(plans)
    maximum = max(n for _, _, n in plans)
    plans += [("general", int(rng.choice([2, 3, 4], p=[.2, .35, .45])), int(rng.integers(maximum+1, 9)))
              for _ in range(ne)]
    puzzle = {"train": [], "test": []}
    used = set()
    for i, (kind, scale, nroots) in enumerate(plans):
        palette = _palette(rng) if vary_palette and i else base_palette
        for _ in range(12):
            pair = _make_pair(rng, scale, nroots, kind, palette)
            key = _key(pair["input"])
            if key not in used:
                used.add(key)
                puzzle["train" if i < nt else "test"].append(pair)
                break
        else:
            raise RuntimeError("could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    targets = _targets(info)
    reached = {x for x in targets if x is not None}
    cells = set(map(tuple, np.argwhere(info["bp"] != info["background"])))
    by_color = defaultdict(set)
    for (_, q), target in zip(info["roots"], targets):
        if target is not None:
            by_color[q].add(target[0])
    return dict(new_caps=reached-info["ref_cells"], unrevealed=cells-reached-info["ref_cells"],
                dormant=sum(x is None for x in targets),
                shared=sum(x is not None for x in targets) > len(reached),
                different_depths=any(len(rows) >= 2 for rows in by_color.values()))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid counts")
    seen, train_infos, test_infos = set(), [], []
    for split, infos in (("train", train_infos), ("test", test_infos)):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _analyze(inp)
            expected = _render(info)
            _require(np.array_equal(expected, out), "output violates selective canopy growth")
            f = _features(info)
            _require(f["new_caps"] and f["unrevealed"], "need a new cap and an unrevealed blueprint tile")
            _require(not np.array_equal(out, info["crop"]), "omitting A has no effect")
            _require(not np.array_equal(out, _render(info, "visible_only")), "omitting B has no effect")
            # The carrier is unchanged and at least one non-cap stem cell is added.
            _require(np.array_equal(out[info["line"]], info["crop"][info["line"]]), "root/carrier changed")
            _require(np.any((info["crop"] == info["background"]) & (out == info["carrier"])), "no stalk construction")
            infos.append(info)
    _require({2, 3}.issubset({i["scale"] for i in train_infos}), "train must demonstrate two scales")
    _require(any(not np.array_equal(_render(i), _render(i, "ignore_color")) for i in train_infos), "no color-selection evidence")
    _require(any(not np.array_equal(_render(i), _render(i, "farthest")) for i in train_infos), "no nearest-target evidence")
    _require(any(_features(i)["different_depths"] for i in train_infos), "fixed color heights not contradicted")
    _require(any(_features(i)["dormant"] for i in train_infos), "no dormant-root evidence")
    _require(min(len(i["roots"]) for i in test_infos) > max(len(i["roots"]) for i in train_infos), "missing root-count shift")


def self_test():
    """Bounded 42-seed structural/semantic audit, plus counts and determinism."""
    shapes, masks, palettes, sides, scales, ref_counts, root_counts = (set() for _ in range(7))
    properties = defaultdict(int)
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve mismatch")
                f = _features(info)
                bp, bg = info["bp"], info["background"]
                mask = bp != bg
                masks.add((bp.shape, mask.tobytes()))
                shapes.add((pair["input"].shape, pair["output"].shape))
                palettes.add((bg, info["carrier"], info["marker"]))
                scales.add(info["scale"])
                ref_counts.add(len(info["refs"]))
                root_counts.add(len(info["roots"]))
                t, b, l, r = info["bounds"]
                bt, bb, bl, br = info["blueprint_bounds"]
                sides.add("above" if bb < t else "below" if bt > b else "left" if br < l else "right")
                properties["full"] += bool(mask.all())
                properties["gapped"] += not bool(mask.all())
                properties["dormant"] += f["dormant"] > 0
                properties["shared"] += f["shared"]
                properties["depth"] += f["different_depths"]
                properties["background_zero"] += bg == 0
                properties["edge_window"] += t == 0 or l == 0 or b == pair["input"].shape[0]-1 or r == pair["input"].shape[1]-1
                properties["inset_window"] += t > 0 and l > 0 and b < pair["input"].shape[0]-1 and r < pair["input"].shape[1]-1
    _require(len(signatures) == 42 and len(masks) >= 60 and len(shapes) >= 100, "insufficient structural variation")
    _require(len(palettes) >= 30 and sides == {"above", "below", "left", "right"}, "layout/palette variation missing")
    _require(scales == {2, 3, 4} and ref_counts >= {2, 3, 4} and root_counts >= set(range(2, 9)), "counts/scales missing")
    _require(all(properties[name] > 0 for name in ("full", "gapped", "dormant", "shared", "depth", "background_zero", "edge_window", "inset_window")), "family feature missing")
    a, b = generate_puzzle(12345, 2, 1), generate_puzzle(12345, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x), "nondeterministic")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(100+nt*10+ne, nt, ne)
            _require(len(p["train"]) == nt and len(p["test"]) == ne, "count request ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (2.0, 1), (4, True), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "puzzles": len(signatures), "mask_shapes": len(masks),
            "grid_shape_pairs": len(shapes), "palettes": len(palettes),
            "sides": sorted(sides), "properties": dict(properties)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
