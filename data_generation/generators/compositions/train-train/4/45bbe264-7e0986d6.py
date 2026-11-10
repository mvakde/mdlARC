"""Mended light guides: rectangular recovery controls finite colored crosses.

Only NumPy and the standard library are used. No parent module or saved puzzle
is imported. The input parser, repair, and ray tracing work without generation.
"""
from __future__ import annotations

import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _roles(a):
    """Singleton colors are sources; the other three have strict frequencies."""
    values, counts = np.unique(a, return_counts=True)
    ordinary = sorted(((int(n), int(c)) for c, n in zip(values, counts) if n > 1),
                      reverse=True)
    if len(ordinary) != 3 or not ordinary[0][0] > ordinary[1][0] > ordinary[2][0]:
        raise ValueError("background, material, and ink need distinct decreasing frequencies")
    background, material, ink = (p[1] for p in ordinary)
    source_colors = {int(c) for c, n in zip(values, counts) if n == 1}
    if not 1 <= len(source_colors) <= 7:
        raise ValueError("need one to seven singleton source colors")
    sources = [(int(r), int(c), int(a[r, c])) for r, c in np.argwhere(
        np.isin(a, list(source_colors)))]
    if (len({p[0] for p in sources}) != len(sources)
            or len({p[1] for p in sources}) != len(sources)):
        raise ValueError("sources must have distinct rows and columns")
    return background, material, ink, sources


def _runs(mask):
    result = []
    for r, row in enumerate(mask):
        transitions = np.diff(np.r_[False, row, False].astype(np.int8))
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        result.extend((r, int(l), int(u)) for l, u in zip(starts, ends) if u-l >= 2)
    return result


def _repair(a, background, material, ink):
    """Union of rectangular patches witnessed by an intact row AND column.

    Source dots count as intact material. A visible background cell vetoes a
    patch; ink is unknown. Witnesses are maximal runs of length at least two.
    Bounds come from the horizontal run's columns and vertical run's rows.
    This uses long-range shape evidence, including for erased corners.
    """
    evidence = (a != background) & (a != ink)
    horizontal = _runs(evidence)
    vertical = _runs(evidence.T)
    prefix = np.pad((a == background).astype(int), ((1, 0), (1, 0)))
    prefix = prefix.cumsum(0).cumsum(1)
    mask = evidence.copy()
    for r, left, right in horizontal:
        for c, top, bottom in vertical:
            if left <= c < right and top <= r < bottom:
                veto = (prefix[bottom, right] - prefix[top, right]
                        - prefix[bottom, left] + prefix[top, left])
                if veto == 0:
                    mask[top:bottom, left:right] = True
    return mask


def _ray_maps(mask, sources):
    horizontal = np.full(mask.shape, -1, dtype=np.int8)
    vertical = np.full(mask.shape, -1, dtype=np.int8)
    h, w = mask.shape
    for i, (r, c, _) in enumerate(sources):
        if not mask[r, c]:
            raise ValueError("source outside recovered material")
        left = right = c
        top = bottom = r
        while left > 0 and mask[r, left-1]:
            left -= 1
        while right+1 < w and mask[r, right+1]:
            right += 1
        while top > 0 and mask[top-1, c]:
            top -= 1
        while bottom+1 < h and mask[bottom+1, c]:
            bottom += 1
        horizontal[r, left:right+1] = i
        vertical[top:bottom+1, c] = i
    return horizontal, vertical


def _render(mask, sources, background, material, ink):
    horizontal, vertical = _ray_maps(mask, sources)
    out = np.where(mask, material, background).astype(np.int8)
    for i, (_, _, color) in enumerate(sources):
        out[(horizontal == i) | (vertical == i)] = color
    out[(horizontal >= 0) & (vertical >= 0) & (horizontal != vertical)] = ink
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    background, material, ink, sources = _roles(a)
    mask = _repair(a, background, material, ink)
    return _render(mask, sources, background, material, ink)


def _clipped_crosses(mask, sources, bg, mat, ink):
    """Deliberately wrong ablation: crosses can jump over background gaps."""
    out = np.where(mask, mat, bg).astype(np.int8)
    for r, c, color in sources:
        out[r, mask[r]] = color
        out[mask[:, c], c] = color
    for i, (r, _, _) in enumerate(sources):
        for j, (_, c, _) in enumerate(sources):
            if i != j and mask[r, c]:
                out[r, c] = ink
    for r, c, color in sources:
        out[r, c] = color
    return out


def _ray_frontier(mask, sources):
    """First outside cell after each ray, when it ends before the canvas edge."""
    result = set()
    h, w = mask.shape
    for r, c, _ in sources:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            y, x = r+dr, c+dc
            while 0 <= y < h and 0 <= x < w and mask[y, x]:
                y, x = y+dr, x+dc
            if 0 <= y < h and 0 <= x < w:
                result.add((y, x))
    return sorted(result)


def _rectangle_supported(mask):
    blocks = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
    covered = np.zeros_like(mask)
    covered[:-1, :-1] |= blocks
    covered[1:, :-1] |= blocks
    covered[:-1, 1:] |= blocks
    covered[1:, 1:] |= blocks
    return bool(np.all(covered[mask]))


def _span(mask):
    result = np.zeros_like(mask)
    result[1:-1] |= mask[:-2] & mask[2:]
    result[:, 1:-1] |= mask[:, :-2] & mask[:, 2:]
    return result


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _counts(rng, num_train, num_test):
    # Explicit None is an optional convenience; ordinary defaults are 4/1.
    nt = None if num_train is None else _count(num_train, "num_train", 2, 10)
    ne = None if num_test is None else _count(num_test, "num_test", 1, 4)
    if nt is None and ne is None:
        ne = int(rng.integers(1, 5))
    if nt is None:
        nt = int(rng.integers(max(2, ne+1), 11))
    if ne is None:
        ne = int(rng.integers(1, min(4, nt-1)+1))
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    return nt, ne


def _base(rng, mode, count):
    """Return a local rectangle union and, when useful, two semantic anchors."""
    if mode == "rectangle":
        h = int(rng.integers(max(5, count+1), 13))
        w = int(rng.integers(max(5, count+1), 15))
        if rng.random() < .25:
            w = int(rng.integers(15, 24))
        mask = np.ones((h, w), bool)
        anchors = [(int(rng.integers(0, max(1, h//3))),
                    int(rng.integers(0, max(1, w//3)))),
                   (int(rng.integers(2*h//3, h)),
                    int(rng.integers(2*w//3, w)))]
        return mask, anchors if count >= 2 else []
    if mode in ("bend", "frame"):
        h, w = int(rng.integers(10, 19)), int(rng.integers(10, 20))
        top, bottom, left = (int(x) for x in rng.integers(2, 5, size=3))
        if top+bottom+left < count+1:
            top = bottom = left = 3
        top_end = w-int(rng.integers(0, 3)) if mode == "bend" else w
        bottom_end = w-int(rng.integers(0, 3)) if mode == "bend" else w
        mask = np.zeros((h, w), bool)
        mask[:top, :top_end] = True
        mask[h-bottom:, :bottom_end] = True
        mask[:, :left] = True
        if mode == "frame":
            mask[:, -int(rng.integers(2, 4)):] = True
        anchors = [(int(rng.integers(h-bottom, h)), int(rng.integers(left))),
                   (int(rng.integers(top)),
                    int(rng.integers(left+1, min(top_end, bottom_end)-1)))]
        return mask, anchors
    # A freely grown union: branches can form L/T/stepped/plus shapes.
    mask = np.zeros((23, 23), bool)
    r, c = int(rng.integers(5, 10)), int(rng.integers(5, 10))
    rh, cw = int(rng.integers(4, 10)), int(rng.integers(4, 10))
    rectangles = [(r, r+rh, c, c+cw)]
    for _ in range(int(rng.integers(1, 6))):
        t, b, l, u = rectangles[int(rng.integers(len(rectangles)))]
        rh, cw = int(rng.integers(2, 8)), int(rng.integers(2, 9))
        side = int(rng.integers(4))
        if side < 2:
            r = int(rng.integers(max(0, t-rh+1), min(23-rh, b-1)+1))
            c = u-int(rng.integers(1, min(cw, 3))) if side == 0 else l-cw+1
        else:
            c = int(rng.integers(max(0, l-cw+1), min(23-cw, u-1)+1))
            r = b-int(rng.integers(1, min(rh, 3))) if side == 2 else t-rh+1
        if 0 <= r <= 23-rh and 0 <= c <= 23-cw:
            rectangles.append((r, r+rh, c, c+cw))
    for t, b, l, u in rectangles:
        mask[t:b, l:u] = True
    coords = np.argwhere(mask)
    t, l = coords.min(0)
    b, u = coords.max(0)+1
    return mask[t:b, l:u], []


def _layout(rng, mode, count):
    local, anchors = _base(rng, mode, count)
    dots = np.zeros(local.shape, np.int8)
    for i, (r, c) in enumerate(anchors):
        dots[r, c] = i+1
    rotations, flip = int(rng.integers(4)), bool(rng.integers(2))
    local, dots = np.rot90(local, rotations), np.rot90(dots, rotations)
    if flip:
        local, dots = np.fliplr(local), np.fliplr(dots)
    lh, lw = local.shape
    h = int(rng.integers(min(30, lh+4), min(30, max(lh+8, 16))+1))
    w = int(rng.integers(min(30, lw+4), min(30, max(lw+8, 17))+1))
    # Unequal padding, including flush edges, has no semantic meaning.
    r, c = int(rng.integers(h-lh+1)), int(rng.integers(w-lw+1))
    mask = np.zeros((h, w), bool)
    mask[r:r+lh, c:c+lw] = local
    anchors = [(int(y+r), int(x+c)) for i in range(1, len(anchors)+1)
               for y, x in np.argwhere(dots == i)]
    for _ in range(int(rng.choice([0, 0, 1, 1, 2, 3]))):
        for _ in range(25):
            rh, cw = int(rng.integers(2, 6)), int(rng.integers(2, 7))
            y, x = int(rng.integers(h-rh+1)), int(rng.integers(w-cw+1))
            if not np.any(mask[max(0, y-1):min(h, y+rh+1),
                               max(0, x-1):min(w, x+cw+1)]):
                mask[y:y+rh, x:x+cw] = True
                break
    return mask, anchors


def _positions(rng, mask, anchors, count):
    """Random bipartite matching supplies distinct rows and columns."""
    used_r, used_c = {r for r, _ in anchors}, {c for _, c in anchors}
    neighbors = {}
    for r in rng.permutation(mask.shape[0]):
        r = int(r)
        if r not in used_r:
            cols = [int(c) for c in rng.permutation(np.flatnonzero(mask[r])) if int(c) not in used_c]
            neighbors[r] = cols
    matched = {}

    def augment(r, seen):
        for c in neighbors[r]:
            if c in seen:
                continue
            seen.add(c)
            if c not in matched or augment(matched[c], seen):
                matched[c] = r
                return True
        return False

    for r in neighbors:
        augment(r, set())
    rest = [(r, c) for c, r in matched.items()]
    needed = count-len(anchors)
    if len(rest) < needed:
        return None
    rng.shuffle(rest)
    return anchors+rest[:needed]


def _noise(rng, mask, sources, bg, mat, ink):
    clean = np.where(mask, mat, bg).astype(np.int8)
    occupied = np.zeros_like(mask)
    for r, c, color in sources:
        clean[r, c] = color
        occupied[r, c] = True
    hm, vm = _ray_maps(mask, sources)
    paths = (hm >= 0) | (vm >= 0)
    inside = [tuple(map(int, x)) for x in np.argwhere(mask & ~occupied)]
    outside = [tuple(map(int, x)) for x in np.argwhere(~mask)]
    # Non-spanned boundary losses include corners, which a local hole-filler misses.
    corners = [tuple(map(int, x)) for x in np.argwhere(mask & ~occupied & ~_span(mask))]
    critical = [p for p in inside if paths[p]]
    adjacent = _ray_frontier(mask, sources)
    coords = np.argwhere(mask)
    lo, hi = coords.min(0), coords.max(0)
    internal_gap = [p for p in adjacent if lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]]
    if internal_gap:
        adjacent = internal_gap
    if not corners or not critical or not adjacent:
        return None
    selected_inside = {corners[int(rng.integers(len(corners)))],
                       critical[int(rng.integers(len(critical)))]}
    selected_outside = {adjacent[int(rng.integers(len(adjacent)))]}
    ni = int(rng.integers(3, min(13, max(4, int(mask.sum())//12))+1))
    no = int(rng.integers(2, 9))
    for pool, chosen, target in ((inside, selected_inside, ni), (outside, selected_outside, no)):
        for index in rng.permutation(len(pool)):
            chosen.add(pool[int(index)])
            if len(chosen) >= target:
                break
    a = clean.copy()
    for p in selected_inside | selected_outside:
        a[p] = ink
    try:
        if _roles(a)[:3] != (bg, mat, ink):
            return None
    except ValueError:
        return None
    if not np.array_equal(_repair(a, bg, mat, ink), mask):
        return None
    out = _render(mask, sources, bg, mat, ink)
    if np.any(out == ink) and not np.any((out == ink) & (a != ink)):
        return None
    # Repair must change ray reach at some already correct input cell.
    unhealed = _render((a != bg) & (a != ink), sources, bg, mat, ink)
    if not np.any((out != unhealed) & (a != ink)):
        return None
    # Likewise, promoting all noise to material must visibly leak a beam.
    promoted = _render(a != bg, sources, bg, mat, ink)
    beam_colors = [color for _, _, color in sources]+[ink]
    if not np.any(~mask & np.isin(promoted, beam_colors)):
        return None
    return {"input": a, "output": out}


def _make_pair(rng, mode, count, palette):
    for _ in range(100):
        mask, anchors = _layout(rng, mode, count)
        if not .10 <= mask.mean() <= .45:
            continue
        positions = _positions(rng, mask, anchors, count)
        if positions is None:
            continue
        bg, mat, ink = palette
        colors = rng.permutation([c for c in range(10) if c not in palette])[:count]
        sources = [(r, c, int(color)) for (r, c), color in zip(positions, colors)]
        intended = _render(mask, sources, bg, mat, ink)
        if mode == "bend" and np.array_equal(intended, _clipped_crosses(mask, sources, bg, mat, ink)):
            continue
        for _ in range(30):
            pair = _noise(rng, mask, sources, bg, mat, ink)
            if pair is not None:
                return pair
    raise RuntimeError(f"bounded construction failed: {mode}, {count} sources")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    nt, ne = _counts(rng, num_train, num_test)
    palette = tuple(int(x) for x in rng.permutation(10)[:3])
    stable_palette = bool(rng.integers(2))
    training = [("rectangle", 2), ("bend", 3)]
    for _ in range(nt-2):
        mode = str(rng.choice(["rectangle", "bend", "grown", "grown"]))
        training.append((mode, int(rng.integers(1, 5))))
    max_train = max(n for _, n in training)
    tests = [(str(rng.choice(["rectangle", "bend", "grown", "frame"])),
              int(rng.integers(max_train+1, 8))) for _ in range(ne)]
    rng.shuffle(training)
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, plans in (("train", training), ("test", tests)):
        for mode, n in plans:
            for _ in range(20):
                colors = palette if stable_palette else tuple(int(x) for x in rng.permutation(10)[:3])
                pair = _make_pair(rng, mode, n, colors)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
    a, out = _grid(pair["input"]), _grid(pair["output"])
    _require(a.shape == out.shape, "dimensions changed")
    bg, mat, ink, sources = _roles(a)
    mask = _repair(a, bg, mat, ink)
    evidence = (a != bg) & (a != ink)
    noisy = a == ink
    _require(np.all(mask[evidence]) and not np.any(mask[a == bg]), "repair contradicted intact evidence")
    _require(_rectangle_supported(mask), "material is not supported by solid rectangles of thickness at least two")
    _require(np.count_nonzero(noisy & mask) >= 2, "missing inside noise")
    _require(np.count_nonzero(noisy & ~mask) >= 2, "missing outside noise")
    _require(np.any(noisy & mask & ~_span(evidence)), "no geometry-dependent boundary loss")
    hm, vm = _ray_maps(mask, sources)
    crossings = (hm >= 0) & (vm >= 0) & (hm != vm)
    _require(np.array_equal(out == ink, crossings), "ink is not exactly the reached foreign crossings")
    _require(not np.any(crossings) or np.any(crossings & ~noisy), "all junction ink was already present")
    _require(np.all(out[~mask] == bg), "beam leaked outside material")
    _require(np.all(out[mask & (hm < 0) & (vm < 0)] == mat), "unlit material changed")
    _require(all(out[r, c] == color for r, c, color in sources), "self intersections changed")
    _require(np.array_equal(solve(a), out), "input-only solve mismatch")
    # Cellwise straight-line visibility check does not use ray-map interval bounds.
    for r, c in np.argwhere(mask):
        hits = []
        for rr, cc, color in sources:
            if r == rr and np.all(mask[r, min(c, cc):max(c, cc)+1]):
                hits.append(color)
            elif c == cc and np.all(mask[min(r, rr):max(r, rr)+1, c]):
                hits.append(color)
        expected = ink if len(hits) >= 2 else hits[0] if hits else mat
        _require(out[r, c] == expected, "straight visibility check failed")
    unhealed = _render(evidence, sources, bg, mat, ink)
    _require(np.any((unhealed != out) & ~noisy), "B omission does not change downstream rays")
    promoted = _render(a != bg, sources, bg, mat, ink)
    _require(np.any(~mask & np.isin(promoted, [p[2] for p in sources]+[ink])),
             "all-noise-is-material shortcut survived")
    no_a = np.where(mask, mat, bg).astype(np.int8)
    for r, c, color in sources:
        no_a[r, c] = color
    _require(not np.array_equal(no_a, out), "A omission does not change answer")
    blocked = sum(mask[r, c] and not (hm[r, c] == i and vm[r, c] == j)
                  for i, (r, _, _) in enumerate(sources)
                  for j, (_, c, _) in enumerate(sources) if i != j)
    return {"count": len(sources), "mask": mask, "sources": sources,
            "palette": (bg, mat, ink), "crossings": int(crossings.sum()),
            "blocked_crossings": int(blocked),
            "occluded": not np.array_equal(_clipped_crosses(mask, sources, bg, mat, ink), out),
            "noise": int(noisy.sum()), "hard_losses": int(np.sum(noisy & mask & ~_span(evidence)))}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid pair counts")
    seen, records = set(), {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            record = _inspect_pair(pair)
            key = _key(np.asarray(pair["input"]))
            _require(key not in seen, "duplicate inputs")
            seen.add(key)
            records[section].append(record)
    train = records["train"]
    _require(any(r["count"] == 2 and r["crossings"] == 2 for r in train),
             "missing transparent two-source crossing demonstration")
    _require(any(r["blocked_crossings"] >= 1 and r["crossings"] >= 1 for r in train),
             "missing reachable AND occluded crossings demonstration")
    _require(len({r["count"] for r in train}) >= 2, "training counts do not vary")
    _require(all(r["count"] > max(q["count"] for q in train) for r in records["test"]),
             "missing source-count train/test shift")


def _puzzle_key(p):
    return tuple((section, tuple((_key(np.asarray(q["input"])), _key(np.asarray(q["output"])))
                                 for q in p[section])) for section in ("train", "test"))


def self_test():
    """Bounded 42-seed audit, legal counts, ablations, and spatial symmetries."""
    signatures, shapes, layouts, palettes = set(), set(), set(), set()
    counts, noises, crossings, areas = set(), set(), set(), set()
    occluded = boundary = nonrectangular = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_puzzle_key(p))
        _require(_puzzle_key(p) == _puzzle_key(generate_puzzle(seed)), "nondeterministic seed")
        for section in ("train", "test"):
            for pair in p[section]:
                a, out = pair["input"], pair["output"]
                r = _inspect_pair(pair)
                shapes.add(a.shape)
                layouts.add(_key(r["mask"]))
                palettes.add(r["palette"])
                counts.add(r["count"])
                noises.add(r["noise"])
                crossings.add(r["crossings"])
                areas.add(int(r["mask"].sum()))
                occluded += r["occluded"]
                mask = r["mask"]
                boundary += bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
                coords = np.argwhere(mask)
                lo, hi = coords.min(0), coords.max(0)+1
                nonrectangular += int(mask.sum()) < int(np.prod(hi-lo))
        # Rule equivariance for a rotation, reflection, and a full palette permutation.
        a, out = p["test"][0]["input"], p["test"][0]["output"]
        _require(np.array_equal(solve(np.rot90(a)), np.rot90(out)), "rotation failure")
        _require(np.array_equal(solve(np.fliplr(a)), np.fliplr(out)), "reflection failure")
        permutation = np.random.default_rng(seed+9000).permutation(10)
        _require(np.array_equal(solve(permutation[a]), permutation[out]), "palette equivariance failure")
    _require(len(signatures) == 42, "duplicate seeds")
    _require(len(shapes) >= 70 and len(layouts) >= 170, "insufficient structural variation")
    _require(len(palettes) >= 35 and len(areas) >= 70, "insufficient palette/area variation")
    _require(counts == set(range(1, 8)), "source count coverage")
    _require(len(noises) >= 10 and len(crossings) >= 8, "noise/interaction counts too narrow")
    _require(occluded >= 42 and boundary >= 12 and nonrectangular >= 84, "missing structural cases")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(12345+nt*10+ne, nt, ne)
            _require(len(p["train"]) == nt and len(p["test"]) == ne, "explicit counts failed")
    for nt, ne in ((None, 1), (None, 4), (2, None), (10, None), (None, None)):
        generate_puzzle(54321, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, "1")):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes), "layouts": len(layouts),
            "palettes": len(palettes), "source_counts": sorted(counts),
            "noise_counts": sorted(noises), "crossing_counts": sorted(crossings),
            "material_areas": len(areas), "occluded_pairs": occluded,
            "border_contact_pairs": boundary, "nonrectangular_scenes": nonrectangular}


if __name__ == "__main__":
    report = self_test()
    print(json.dumps(report, separators=(",", ":")))
