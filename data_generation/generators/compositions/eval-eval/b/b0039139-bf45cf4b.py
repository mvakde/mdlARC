"""Self-counting wavefront stamps. Standard library and NumPy only.

The visible shared color links a graph anchor to counter islands in its stamp.
The island count limits alternating geodesic fronts of exact block substitution.
"""
from collections import deque
from numbers import Integral

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _groups(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, pending = [start], [start]
        while pending:
            r, c = pending.pop()
            for dr, dc in steps:
                point = (r + dr, c + dc)
                if point in remaining:
                    remaining.remove(point)
                    group.append(point)
                    pending.append(point)
        groups.append(group)
    return groups


def _bounds(cells):
    points = np.asarray(cells)
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return tuple(map(int, (lo[0], hi[0], lo[1], hi[1])))


def _crop(a, box):
    r0, r1, c0, c1 = box
    return a[r0:r1, c0:c1].copy()


def _separated(a, b):
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _distances(network, root):
    """Shortest edge paths, including the anchor itself as traversable."""
    distances = np.full(network.shape, -1, dtype=int)
    root = tuple(map(int, root))
    distances[root] = 0
    pending = deque([root])
    h, w = network.shape
    while pending:
        r, c = pending.popleft()
        for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if (0 <= x < h and 0 <= y < w and network[x, y]
                    and distances[x, y] < 0):
                distances[x, y] = distances[r, c] + 1
                pending.append((x, y))
    return distances


def _selected(distances, count):
    return (distances >= 0) & (distances % 2 == 0) & (distances < 2*count)


def _transforms(a):
    return [np.rot90(a[:, ::-1] if k >= 4 else a, k % 4) for k in range(8)]


def _asymmetric(a):
    return all(not np.array_equal(a, b) for b in _transforms(a)[1:])


def _parse(input_grid):
    g = _grid(input_grid)
    colors, counts = np.unique(g, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(int(counts.max()) * 2 > g.size, "background is not a strict majority")
    encodings = _groups(g != bg, diagonal=True)
    _require(len(encodings) == 2, "expected two separated foreground encodings")
    boxes = [_bounds(cells) for cells in encodings]
    _require(_separated(*boxes), "encoding boxes need a blank separator")
    crops = [_crop(g, b) for b in boxes]
    candidates = []
    for i in (0, 1):
        coarse, stamp = crops[i], crops[1-i]
        coarse_colors = set(map(int, np.unique(coarse))) - {bg}
        fine_colors = set(map(int, np.unique(stamp))) - {bg}
        shared = coarse_colors & fine_colors
        if len(coarse_colors) != 2 or len(fine_colors) < 3 or len(shared) != 1:
            continue
        key = next(iter(shared))
        if np.count_nonzero(coarse == key) != 1:
            continue
        track = next(iter(coarse_colors - {key}))
        network = coarse != bg
        if int(network.sum()) < 3 or len(_groups(network)) != 1:
            continue
        key_mask = stamp == key
        islands = _groups(key_mask)
        if len(islands) != len(_groups(key_mask, diagonal=True)):
            continue
        if any(not np.any(side != bg) for side in
               (stamp[0], stamp[-1], stamp[:, 0], stamp[:, -1])):
            continue
        if (network.shape[0]*stamp.shape[0] > 30
                or network.shape[1]*stamp.shape[1] > 30):
            continue
        root = tuple(map(int, np.argwhere(coarse == key)[0]))
        distances = _distances(network, root)
        candidates.append({"background": bg, "track": track, "key": key,
                           "network": network, "root": root, "stamp": stamp,
                           "count": len(islands), "island_sizes": tuple(map(len, islands)),
                           "distances": distances,
                           "network_box": boxes[i], "stamp_box": boxes[1-i]})
    _require(len(candidates) == 1, "input does not have one unique role interpretation")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input alone; no training, random state, or stored labels."""
    d = _parse(input_grid)
    selected = _selected(d["distances"], d["count"])
    stamp = d["stamp"]
    h, w = stamp.shape
    rr, cc = np.indices((selected.shape[0]*h, selected.shape[1]*w))
    return np.where(selected[rr//h, cc//w], stamp[rr % h, cc % w],
                    d["background"]).astype(np.int8)


def _render(network, root, count, stamp, bg):
    """Generation writes complete blocks; solve uses fine-cell indexing."""
    selected = _selected(_distances(network, root), count)
    h, w = stamp.shape
    output = np.full((network.shape[0]*h, network.shape[1]*w), bg, dtype=np.int8)
    for r, c in np.argwhere(selected):
        output[r*h:(r+1)*h, c*w:(c+1)*w] = stamp
    return output


def _key(a):
    return a.shape, a.tobytes()


def _partition(length, number, rng, balanced=False):
    available = length-number+1
    if balanced:
        sizes = [available//number + (k < available % number) for k in range(number)]
        gaps = [1]*(number-1)
    else:
        sizes, gaps = [1]*number, [1]*(number-1)
        for _ in range(length-(2*number-1)):
            k = int(rng.integers(2*number-1))
            if k < number:
                sizes[k] += 1
            else:
                gaps[k-number] += 1
    spans, offset = [], 0
    for k, size in enumerate(sizes):
        spans.append((offset, offset+size))
        offset += size + (gaps[k] if k < len(gaps) else 0)
    return spans


def _partition_options(shape, count, area):
    h, w = shape
    result = []
    for nr in range(1, (h+1)//2+1):
        for nc in range(1, (w+1)//2+1):
            if not count <= nr*nc <= count+2:
                continue
            rh = [(h-nr+1)//nr + (i < (h-nr+1) % nr) for i in range(nr)]
            cw = [(w-nc+1)//nc + (i < (w-nc+1) % nc) for i in range(nc)]
            capacities = sorted([a*b for a in rh for b in cw], reverse=True)
            if sum(capacities[:count]) >= area:
                result.append((nr, nc))
    return result


def _grow(shape, area, rng):
    mask = np.zeros(shape, dtype=bool)
    frontier = {(int(rng.integers(shape[0])), int(rng.integers(shape[1])))}
    for _ in range(area):
        choices = sorted(frontier)
        r, c = choices[int(rng.integers(len(choices)))]
        frontier.remove((r, c))
        mask[r, c] = True
        for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= x < shape[0] and 0 <= y < shape[1] and not mask[x, y]:
                frontier.add((x, y))
    return mask


def _make_keys(shape, count, rng, area=None):
    minimum = count if area is None else area
    options = _partition_options(shape, count, minimum)
    _require(bool(options), "stamp cannot hold counter islands")
    for attempt in range(20):
        nr, nc = options[int(rng.integers(len(options)))]
        rows = _partition(shape[0], nr, rng, balanced=attempt == 19)
        cols = _partition(shape[1], nc, rng, balanced=attempt == 19)
        boxes = [(a, b, c, d) for a, b in rows for c, d in cols]
        rng.shuffle(boxes)
        if area is not None:
            boxes.sort(key=lambda b: (b[1]-b[0])*(b[3]-b[2]), reverse=True)
        boxes = boxes[:count]
        capacities = [(b-a)*(d-c) for a, b, c, d in boxes]
        if sum(capacities) >= minimum:
            break
    target = (area if area is not None else int(rng.integers(
        count, max(count, min(sum(capacities), shape[0]*shape[1]*3//5))+1)))
    sizes = [1]*count
    for _ in range(target-count):
        choices = [k for k in range(count) if sizes[k] < capacities[k]]
        k = choices[int(rng.integers(len(choices)))]
        sizes[k] += 1
    mask = np.zeros(shape, dtype=bool)
    for (a, b, c, d), size in zip(boxes, sizes):
        bh, bw = b-a, d-c
        # Geometry, not just color, varies: exact-area growth for witnesses;
        # optional bars, blocks and rings for ordinary islands.
        mode = int(rng.integers(5)) if area is None else 0
        if mode == 1:
            oh, ow = 1, int(rng.integers(1, bw+1))
            obj = np.ones((oh, ow), dtype=bool)
        elif mode == 2:
            oh, ow = int(rng.integers(1, bh+1)), 1
            obj = np.ones((oh, ow), dtype=bool)
        elif mode in (3, 4):
            oh, ow = int(rng.integers(1, bh+1)), int(rng.integers(1, bw+1))
            obj = np.ones((oh, ow), dtype=bool)
            if mode == 4 and min(oh, ow) >= 3:
                obj[1:-1, 1:-1] = False
        else:
            oh, ow = bh, bw
            obj = _grow((bh, bw), size, rng)
        a += int(rng.integers(bh-oh+1))
        c += int(rng.integers(bw-ow+1))
        mask[a:a+oh, c:c+ow] = obj
    _require(len(_groups(mask)) == count == len(_groups(mask, True)), "counter islands merged")
    return mask


def _make_stamp(shape, count, palette, rng, witness=False):
    bg, _, key, *payload = palette
    h, w = shape
    rr, cc = np.indices(shape)
    for _ in range(60):
        keys = _make_keys(shape, count, rng, area=6 if witness else None)
        family = int(rng.integers(6)) if not witness else 5
        if family == 0:
            labels = rr if rng.random() < .5 else cc
        elif family == 1:
            labels = np.minimum.reduce([rr, cc, h-1-rr, w-1-cc])
        elif family == 2:
            labels = rr+cc*int(rng.integers(1, 4))
        elif family == 3:
            tile = rng.integers(len(payload), size=(2, 2))
            labels = tile[rr % 2, cc % 2]
        elif family == 4:
            labels = abs(2*rr-h+1)+abs(2*cc-w+1)
        else:
            labels = rng.integers(len(payload), size=shape)
        stamp = np.asarray(payload, dtype=np.int8)[labels % len(payload)].copy()
        stamp[keys] = key
        # Insert holes only into payload positions; counted components survive.
        possible = [(r, c) for r in range(h) for c in range(w) if not keys[r, c]]
        rng.shuffle(possible)
        if witness:
            possible = [(r, c) for r, c in possible if 0 < r < h-1 and 0 < c < w-1]
        target = 1 if witness else int(rng.integers(0, max(2, h*w//5)))
        inserted = 0
        for point in possible:
            if inserted >= target:
                break
            old = stamp[point]
            stamp[point] = bg
            fg = stamp != bg
            if (len(_groups(fg, True)) != 1
                    or any(not np.any(s) for s in (fg[0], fg[-1], fg[:, 0], fg[:, -1]))
                    or len(set(map(int, np.unique(stamp))) - {bg, key}) < 2):
                stamp[point] = old
            else:
                inserted += 1
        if len(set(map(int, np.unique(stamp))) - {bg, key}) < 2:
            continue
        if witness and (inserted == 0 or not _asymmetric(stamp)):
            continue
        return stamp
    raise RuntimeError("could not build an informative multicolor stamp")


def _orient(network, root, stamp_shape, rng):
    marked = network.astype(np.int8)
    marked[root] = 2
    choices = [g for g in _transforms(marked)
               if g.shape[0]*stamp_shape[0] <= 30 and g.shape[1]*stamp_shape[1] <= 30]
    _require(bool(choices), "network orientation exceeds output budget")
    g = choices[int(rng.integers(len(choices)))].copy()
    return g != 0, tuple(map(int, np.argwhere(g == 2)[0]))


def _witness_network(count, stamp_shape, rng):
    h = 3 if count == 2 else 5
    max_w = min(6, 30//stamp_shape[1])
    w = int(rng.integers(4, max_w+1))
    net = np.zeros((h, w), dtype=bool)
    net[0] = net[-1] = True
    net[:, -1] = True
    root = (0, 0)
    if count == 3:
        net[1:3, 1] = True
        root = (0, 1)
    return _orient(net, root, stamp_shape, rng)


def _network(shape, family, rng, endpoint=False):
    h, w = shape
    net = np.zeros(shape, dtype=bool)
    if family == "hairpin":
        net[0] = net[-1] = True
        net[:, -1] = True
        root = (0, 0)
    elif family == "serpent":
        net[::2] = True
        for r in range(1, h, 2):
            net[r, w-1 if (r//2) % 2 == 0 else 0] = True
        root = (0, 0)
    elif family == "ring":
        net[[0, -1]] = True
        net[:, [0, -1]] = True
        root = (0, int(rng.integers(w)))
    elif family == "fork":
        r, c = int(rng.integers(h)), int(rng.integers(w))
        net[r] = net[:, c] = True
        for _ in range(int(rng.integers(0, max(1, h*w//4)))):
            fringe = [(a, b) for a in range(h) for b in range(w) if not net[a, b]
                      and any(0 <= x < h and 0 <= y < w and net[x, y]
                              for x, y in ((a-1, b), (a+1, b), (a, b-1), (a, b+1)))]
            if fringe:
                net[fringe[int(rng.integers(len(fringe)))]] = True
        root = (r, c)
    elif family == "room":
        net[:] = True
        if min(h, w) >= 3:
            a, b = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
            net[a, b] = False
            if rng.random() < .5:
                net[0:a+1, b] = False
        root = (h-1, 0)
    else:
        raise ValueError("unknown graph shape")
    if not endpoint:
        positions = np.argwhere(net)
        root = tuple(map(int, positions[int(rng.integers(len(positions)))]))
    return net, root


def _layout(network, root, stamp, palette, rng):
    bg, track, key = palette[:3]
    coarse = np.where(network, track, bg).astype(np.int8)
    coarse[root] = key
    mh, mw = coarse.shape
    sh, sw = stamp.shape
    horizontal, reverse = bool(rng.integers(2)), bool(rng.integers(2))
    gap = int(rng.integers(1, 8))
    base_h = max(mh, sh) if horizontal else mh+sh+gap
    base_w = mw+sw+gap if horizontal else max(mw, sw)
    h = base_h + int(rng.integers(min(6, 30-base_h)+1))
    w = base_w + int(rng.integers(min(6, 30-base_w)+1))
    foreground = int(network.sum()) + int(np.count_nonzero(stamp != bg))
    while h*w <= 2*foreground:
        if h < 30 and (h <= w or w == 30):
            h += 1
        else:
            w += 1
    if horizontal:
        start = int(rng.integers(w-base_w+1))
        mc, sc = (start+sw+gap, start) if reverse else (start, start+mw+gap)
        mr, sr = int(rng.integers(h-mh+1)), int(rng.integers(h-sh+1))
    else:
        start = int(rng.integers(h-base_h+1))
        mr, sr = (start+sh+gap, start) if reverse else (start, start+mh+gap)
        mc, sc = int(rng.integers(w-mw+1)), int(rng.integers(w-sw+1))
    g = np.full((h, w), bg, dtype=np.int8)
    g[mr:mr+mh, mc:mc+mw] = coarse
    g[sr:sr+sh, sc:sc+sw] = stamp
    return g


def _make_pair(rng, palette, count, mode, witness=False):
    if witness:
        options = [(h, w) for h in range(4, 7) for w in range(4, 7)
                   if _partition_options((h, w), count, 6)]
        sh, sw = options[int(rng.integers(len(options)))]
        stamp = _make_stamp((sh, sw), count, palette, rng, True)
        network, root = _witness_network(count, stamp.shape, rng)
    else:
        fine_options = [(h, w) for h in range(2, 9 if mode == "stamp" else 7)
                        for w in range(2, 9 if mode == "stamp" else 7)
                        if _partition_options((h, w), count, count)
                        and (mode != "topology" or min(h, w) <= 4)
                        and (mode != "stamp" or max(h, w) >= 7)]
        for _ in range(160):
            sh, sw = fine_options[int(rng.integers(len(fine_options)))]
            max_h, max_w = min(10, 30//sh), min(10, 30//sw)
            limit = 6 if mode == "train" else 10
            shape = (int(rng.integers(2, min(limit, max_h)+1)),
                     int(rng.integers(2, min(limit, max_w)+1)))
            if mode == "topology" and max(shape) <= 6:
                continue
            families = (["serpent"] if mode == "quota" else ["ring", "serpent"]
                        if mode == "topology" else ["hairpin", "serpent", "fork", "room"]
                        if mode == "train" else ["hairpin", "serpent", "fork", "room", "ring"])
            family = str(rng.choice(families))
            network, root = _network(shape, family, rng, endpoint=mode == "quota")
            dist = _distances(network, root)
            if int(network.sum()) < 3 or np.any(dist[network] < 0):
                continue
            if mode == "quota" and dist.max() < 2*count:
                continue
            stamp = _make_stamp((sh, sw), count, palette, rng)
            network, root = _orient(network, root, stamp.shape, rng)
            break
        else:
            raise RuntimeError("could not fit network and stamp to the output budget")
    inp = _layout(network, root, stamp, palette, rng)
    out = _render(network, root, count, stamp, palette[0])
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, lower, upper, name in ((num_train, 2, 10, "num_train"),
                                      (num_test, 1, 4, "num_test")):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                or not lower <= int(value) <= upper):
            raise ValueError(f"illegal {name}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    stable = bool(rng.random() < .25)
    base_palette = list(map(int, rng.permutation(10)))
    shift = str(rng.choice(["quota", "topology", "stamp"]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, amount in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(amount):
            witness = split == "train" and index < 2
            count = index+2 if witness else int(rng.integers(1, 4)) if split == "train" else (
                int(rng.integers(4, 7)) if shift == "quota" else int(rng.integers(2, 4)))
            mode = "train" if split == "train" else shift
            for _ in range(80):
                roles = base_palette if stable else list(map(int, rng.permutation(10)))
                palette = roles[:int(rng.integers(5, 8))]
                pair = _make_pair(rng, palette, count, mode, witness)
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _witness_features(d):
    net, distances, n = d["network"], d["distances"], d["count"]
    selected = _selected(distances, n)
    rr, cc = np.indices(net.shape)
    manhattan = abs(rr-d["root"][0]) + abs(cc-d["root"][1])
    return {
        "quota": bool(np.any(net & (distances % 2 == 0) & (distances >= 2*n))),
        "geodesic": bool(np.any(selected != (net & (manhattan % 2 == 0) & (manhattan < 2*n)))),
        "branch": any(np.count_nonzero(distances == k) > 1 for k in range(2, 2*n, 2)),
        "gap": bool(np.any(distances == 1) and np.any(distances == 2) and n > 1),
        "asymmetric_mask": _asymmetric(selected),
        "asymmetric_stamp": _asymmetric(d["stamp"]),
        "hole": bool(np.any(d["stamp"] == d["background"])),
        "count_area": sum(d["island_sizes"]),
    }


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal pair counts")
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _parse(inp)
            _require(np.all(d["distances"][d["network"]] >= 0), "unreachable network cells")
            _require(np.array_equal(solve(inp), out), "incorrect wavefront substitution")
            _require(not np.array_equal(inp, out), "identity transformation")
            selected = _selected(d["distances"], d["count"])
            h, w = d["stamp"].shape
            _require(out.shape == (selected.shape[0]*h, selected.shape[1]*w), "wrong output extent")
            for r in range(selected.shape[0]):
                for c in range(selected.shape[1]):
                    block = out[r*h:(r+1)*h, c*w:(c+1)*w]
                    _require(np.array_equal(block, d["stamp"]) if selected[r, c]
                             else bool(np.all(block == d["background"])), "incorrect individual block")
            _require(d["track"] not in out, "track color leaked into output")
            if split == "train":
                training.append((d, _witness_features(d)))
            else:
                testing.append(d)
    counts = {d["count"] for d, _ in training}
    _require({2, 3} <= counts, "missing two/three-front demonstrations")
    _require(any(a["count"] != b["count"] and x["count_area"] == y["count_area"]
                 for a, x in training for b, y in training), "count confounded with pixel area")
    for feature in ("quota", "geodesic", "branch", "gap", "asymmetric_mask", "asymmetric_stamp", "hole"):
        _require(any(f[feature] for _, f in training), "missing demonstration: " + feature)
    _require(any(max(d["island_sizes"]) > 1 for d, _ in training), "no object-versus-pixel evidence")
    _require(any(np.any(_selected(d["distances"], d["count"]) !=
                           _selected(d["distances"], f["count_area"]))
                 for d, f in training), "pixel-count alternative agrees with all training")
    _require(any(np.any(_selected(d["distances"], d["count"]) !=
                           _selected(d["distances"], 1)) for d, _ in training), "one-front alternative survives")
    _require(any(np.count_nonzero(_selected(d["distances"], d["count"])) > d["count"]
                 for d, _ in training), "front count could be mistaken for total copies")
    max_count = max(d["count"] for d, _ in training)
    max_network = max(max(d["network"].shape) for d, _ in training)
    max_stamp = max(max(d["stamp"].shape) for d, _ in training)
    _require(all(d["count"] > max_count or max(d["network"].shape) > max_network
                 or max(d["stamp"].shape) > max_stamp for d in testing),
             "test lacks a visible count or dimension shift")
    return None


def self_test():
    """Bounded semantic, training-evidence and structural-variation audit."""
    # Literal arithmetic regressions outside the sampler's network templates:
    # a one-row network, an internal anchor, and a rectangular two-island stamp.
    # The distance-two front branches in both directions and clips at the ends.
    regression_rng = np.random.default_rng(90210)
    palette = [0, 1, 2, 3, 4]
    stamp = np.array([[2, 3, 2], [2, 4, 2]], dtype=np.int8)
    net = np.ones((1, 9), dtype=bool)
    inp = _layout(net, (0, 4), stamp, palette, regression_rng)
    expected = np.zeros((2, 27), dtype=np.int8)
    for col in (2, 4, 6):
        expected[:, col*3:(col+1)*3] = stamp
    _require(np.array_equal(solve(inp), expected), "one-row/internal-anchor regression failed")
    inp = _layout(net.T, (4, 0), stamp.T, palette, regression_rng)
    _require(np.array_equal(solve(inp), expected.T), "vertical arithmetic regression failed")
    # More counter islands than available fronts is legal clipping, not an
    # instruction to extend the graph. Nine islands are outside sampled quotas.
    stamp = _make_stamp((5, 5), 9, palette, regression_rng)
    net = np.ones((6, 6), dtype=bool)
    inp = _layout(net, (0, 0), stamp, palette, regression_rng)
    _require(_parse(inp)["count"] == 9, "solver imposes the sampler's quota cap")
    expected = np.zeros((30, 30), dtype=np.int8)
    for r in range(6):
        for c in range(6):
            if (r+c) % 2 == 0:
                expected[r*5:(r+1)*5, c*5:(c+1)*5] = stamp
    _require(np.array_equal(solve(inp), expected), "finite-network clipping failed")
    puzzles, records = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.append(p)
        for split in ("train", "test"):
            for pair in p[split]:
                d = _parse(pair["input"])
                records.append((d, pair))
    signatures = {tuple(_key(pair["input"]) for s in ("train", "test") for pair in p[s])
                  for p in puzzles}
    _require(len(signatures) == 42, "different seeds produced the same puzzle")
    for seed in (0, 17, 41):
        other = generate_puzzle(seed)
        _require(all(np.array_equal(a[k], b[k]) for s in ("train", "test")
                     for a, b in zip(puzzles[seed][s], other[s]) for k in ("input", "output")),
                 "seed is not deterministic")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(2026, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    network_patterns = {_key(d["network"]) for d, _ in records}
    stamp_shapes = {d["stamp"].shape for d, _ in records}
    normalized_stamps = set()
    roots, layouts, palettes, shapes, areas = set(), set(), set(), set(), set()
    for d, pair in records:
        mapping = {d["background"]: 0, d["key"]: 1}
        normalized_stamps.add((d["stamp"].shape, tuple(
            mapping.setdefault(int(c), len(mapping)) for c in d["stamp"].flat)))
        roots.add(d["root"])
        a, b = d["network_box"], d["stamp_box"]
        layouts.add("left" if a[3] < b[2] else "right" if b[3] < a[2]
                    else "above" if a[1] < b[0] else "below")
        palettes.add((d["background"], d["track"], d["key"]))
        shapes.add(pair["output"].shape)
        areas.update(d["island_sizes"])
    _require(len(network_patterns) >= 45, "network geometry lacks variation")
    _require(len(normalized_stamps) >= 100, "stamp content lacks structural variation")
    _require(len(stamp_shapes) >= 20 and len(shapes) >= 50, "dimension variation collapsed")
    _require(len(roots) >= 15 and len(palettes) >= 30, "root or role variation collapsed")
    _require(layouts == {"left", "right", "above", "below"}, "placement variation collapsed")
    _require({1, 2, 3, 4, 5, 6} <= {d["count"] for d, _ in records}, "quota coverage collapsed")
    _require(len(areas) >= 8 and max(areas) >= 10, "counter objects are too uniform")
    _require(any(d["stamp"].shape[0] != d["stamp"].shape[1] for d, _ in records), "only square stamps")
    _require(any(np.any(d["stamp"][0] == d["background"]) for d, _ in records), "no edge gaps")
    # Corrupting one label must be caught by input-only prediction and block checks.
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in puzzles[0]}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted label")
    return {"seeds": 42, "pairs": len(records), "network_patterns": len(network_patterns),
            "normalized_stamps": len(normalized_stamps), "stamp_shapes": len(stamp_shapes),
            "output_shapes": len(shapes), "root_positions": len(roots),
            "counter_component_sizes": sorted(areas)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
