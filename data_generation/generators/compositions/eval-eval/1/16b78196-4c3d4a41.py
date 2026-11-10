"""Assemble a mosaic pressure card, project its selected material, and push pins.

Only Python's standard library and NumPy are used. The solver reads pixels;
construction parameters never enter the prediction path.
"""
from itertools import combinations
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "color range")
    return a


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _orient(a, code):
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4).copy()


def _unorient(a, code):
    b = np.rot90(a, -(code % 4))
    return (np.fliplr(b) if code >= 4 else b).copy()


def _components(a, mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        color = int(a[first])
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining and a[p] == color:
                    remaining.remove(p)
                    stack.append(p)
        result.append((np.array(sorted(cells), dtype=int), color))
    return result


def _description(cells, width):
    """Column intervals in the outward direction, which is upward canonically."""
    x = cells[:, 1] - cells[:, 1].min()
    u = -cells[:, 0]
    _require(int(x.max()) + 1 == width, "piece does not span the card")
    lo, hi = [], []
    for c in range(width):
        values = np.sort(u[x == c])
        _require(len(values) > 0, "missing piece column")
        _require(np.array_equal(values, np.arange(values[0], values[-1]+1)),
                 "piece is not column-convex")
        lo.append(int(values[0]))
        hi.append(int(values[-1]) + 1)
    lo, hi = np.array(lo), np.array(hi)
    _require(bool(np.all(np.maximum(lo[:-1], lo[1:]) < np.minimum(hi[:-1], hi[1:]))),
             "neighboring piece columns do not connect")
    origin = int(lo.min())
    local = np.column_stack((u-origin, x))
    return lo-origin, hi-origin, local


def _assemble(f):
    """Enumerate ALL layer orders, deduplicating actual colored card results."""
    width, depth = f["width"], f["depth"]
    descriptions = [_description(cells, width) for cells, _ in f["pieces"]]
    root_card = np.full((depth, width), f["background"], dtype=np.int8)
    for c, h in enumerate(f["root"]):
        root_card[depth-int(h):, c] = f["structure"]
    solutions = {}
    full = (1 << len(descriptions))-1

    def visit(profile, used, placements):
        if len(solutions) > 1:
            return
        if used == full:
            if np.all(profile == depth):
                card = root_card.copy()
                for i, cells in placements:
                    card[cells[:, 0]-f["base"]-1, cells[:, 1]-f["left"]-1] = f["pieces"][i][1]
                solutions.setdefault(card.tobytes(), (card, placements))
            return
        offset = int(profile.min())
        normalized = profile-offset
        for i, (lo, hi, local) in enumerate(descriptions):
            if used & (1 << i) or not np.array_equal(lo, normalized):
                continue
            next_profile = hi+offset
            if np.any(next_profile > depth):
                continue
            target = np.column_stack((f["floor"]-1-local[:, 0]-offset,
                                      f["left"]+1+local[:, 1]))
            visit(next_profile, used | (1 << i), placements+[(i, target)])

    visit(np.array(f["root"], dtype=int), 0, [])
    _require(len(solutions) == 1, "card must have one distinct colored assembly")
    return next(iter(solutions.values()))


def _read_candidate(a, bg, s):
    cells = np.argwhere(a == s)
    _require(len(cells) > 0, "missing structure")
    t, l = map(int, cells.min(axis=0))
    f, r = map(int, cells.max(axis=0))
    _require(r-l >= 5 and f-t >= 8, "frame too small")
    _require(bool(np.all(a[t, l:r+1] == s) and np.all(a[f, l:r+1] == s)), "roof/floor")
    _require(bool(np.all(a[t:f+1, r] == s)), "opposite side wall")
    marker_rows = np.flatnonzero(a[t:f+1, l] != s) + t
    _require(len(marker_rows) == 1, "one wall selector required")
    b = int(marker_rows[0])
    active = int(a[b, l])
    _require(active not in (bg, s) and t+3 <= b <= f-4, "selector/separator position")
    _require(bool(np.all(a[b, l+1:r+1] == s)), "separator")
    _require(l > 0 and r < a.shape[1]-1, "space for reading marks")
    left_marks = np.flatnonzero(a[t:f+1, l-1] != bg)+t
    right_marks = np.flatnonzero(a[t:f+1, r+1] != bg)+t
    _require(len(left_marks) == len(right_marks) == 1 and left_marks[0] == right_marks[0],
             "paired reading marks")
    read_row = int(left_marks[0])
    _require(b+2 <= read_row <= f-2 and a[read_row, l-1] == a[read_row, r+1] == active,
             "reading window boundary/color")
    width, depth = r-l-1, f-b-1
    expected = np.full_like(a, bg)
    expected[t, l:r+1] = expected[f, l:r+1] = s
    expected[t:f+1, l] = expected[t:f+1, r] = s
    expected[b, l:r+1] = s
    expected[b, l] = active
    expected[read_row, l-1] = expected[read_row, r+1] = active
    root = []
    for c in range(l+1, r):
        column = a[b+1:f, c]
        occupied = np.flatnonzero(column == s)
        _require(len(occupied) > 0 and len(occupied) < depth, "root column")
        h = len(occupied)
        _require(np.array_equal(occupied, np.arange(depth-h, depth)), "root must grow from floor")
        _require(bool(np.all(column[:depth-h] == bg)), "input socket must be empty above root")
        root.append(h)
        expected[f-h:f, c] = s
    _require(len(set(root)) > 1, "jagged root required")
    lanes = (np.flatnonzero(a[t+1, l+1:r] != bg) + l+1).tolist()
    _require(2 <= len(lanes) <= 5 and min(np.diff(lanes)) >= 2, "separated pin lanes")
    sequences = []
    for c in lanes:
        segment = a[t+1:b, c]
        indices = np.flatnonzero(segment != bg)
        _require(len(indices) > 0 and np.array_equal(indices, np.arange(len(indices))), "contiguous pin")
        seq = tuple(map(int, segment[:len(indices)]))
        prefix = next((i for i, value in enumerate(seq) if value != s), len(seq))
        _require(prefix < len(seq) and s not in seq[prefix:], "structural prefix then payload")
        sequences.append(seq)
        expected[t+1:t+1+len(seq), c] = seq
    _require(np.array_equal(a[t:f+1, l:r+1], expected[t:f+1, l:r+1]), "extra or malformed housing cells")
    outside = a != bg
    outside[t:f+1, l:r+1] = False
    outside[read_row, l-1] = outside[read_row, r+1] = False
    pieces = _components(a, outside)
    _require(2 <= len(pieces) <= 5, "two to five detached pieces required")
    _require(all(color != s for _, color in pieces), "structure outside housing")
    _require(any(color == active for _, color in pieces), "selector material absent")
    for source, _ in pieces:
        _description(source, width)
        # The frame's rectangular reserved region has background clearance.
        _require(not any(t-1 <= y <= f+1 and l-1 <= x <= r+1 for y, x in source),
                 "piece touches housing")
        _require(not any(abs(y-read_row) <= 1 and (abs(x-(l-1)) <= 1 or abs(x-(r+1)) <= 1)
                         for y, x in source), "piece touches reading mark")
    owner = {}
    for i, (source, _) in enumerate(pieces):
        for y, x in source:
            owner[(int(y), int(x))] = i
    for (y, x), i in owner.items():
        _require(all(owner.get((y+dy, x+dx), i) == i
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1)), "pieces touch")
    result = dict(background=bg, structure=s, active=active, top=t, left=l, right=r,
                  base=b, floor=f, width=width, depth=depth, root=tuple(root),
                  read_row=read_row, lanes=tuple(lanes), sequences=tuple(sequences), pieces=pieces)
    card, placements = _assemble(result)
    heights = tuple(int(np.count_nonzero(card[:read_row-b, c-l-1] == active)) for c in lanes)
    capacities = tuple(b-t-h-1 for h in heights)
    _require(all(1 <= cap <= len(seq) <= b-t-1 for cap, seq in zip(capacities, sequences)),
             "pin must contact/overlap piston and retain payload")
    result.update(card=card, placements=placements, heights=heights,
                  shifts=tuple(len(seq)-cap for seq, cap in zip(sequences, capacities)))
    return result


def _read(input_grid):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, "unique modal background required")
    bg = int(values[np.argmax(counts)])
    readings = []
    for code in range(8):
        canonical = _unorient(a, code)
        for value in values:
            if value == bg:
                continue
            try:
                f = _read_candidate(canonical, bg, int(value))
            except ValueError:
                continue
            f["orientation"] = code
            f["canonical"] = canonical
            readings.append(f)
    _require(len(readings) == 1, "input must have one geometric interpretation")
    return readings[0]


def _actuate(f, heights=None, *, stop_after_card=False):
    a = f["canonical"]
    result = a.copy()
    bg, s, t, b = f["background"], f["structure"], f["top"], f["base"]
    for source, _ in f["pieces"]:
        result[source[:, 0], source[:, 1]] = bg
    result[b+1:f["floor"], f["left"]+1:f["right"]] = f["card"]
    if stop_after_card:
        return _orient(result, f["orientation"])
    heights = f["heights"] if heights is None else heights
    for col in f["lanes"]:
        result[t+1:b, col] = bg
    for col, h, sequence in zip(f["lanes"], heights, f["sequences"]):
        result[b-h:b, col] = s
        first = b-h-len(sequence)
        for j, value in enumerate(sequence):
            row = first+j
            if row > t:
                result[row, col] = value
    return _orient(result, f["orientation"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the unique card and every actuation solely from this input."""
    return _actuate(_read(input_grid))


def _profile(rng, width, amplitude):
    mode = int(rng.integers(4))
    if mode == 0:
        p = rng.integers(0, amplitude+1, width)
    elif mode == 1:
        p = np.floor(np.linspace(0, amplitude, width)).astype(int)
        if rng.random() < .5:
            p = p[::-1]
    elif mode == 2:
        p = np.clip(np.cumsum(rng.integers(-1, 2, width)), -amplitude, amplitude)
    else:
        p = np.zeros(width, dtype=int)
        for _ in range(int(rng.integers(1, width))):
            start = int(rng.integers(width))
            end = int(rng.integers(start+1, width+1))
            p[start:end] = int(rng.integers(amplitude+1))
    p = p-p.min()
    if p.max() > amplitude:
        p = np.rint(p * amplitude / p.max()).astype(int)
    if p.max() == 0:
        p[int(rng.integers(width))] = 1
    return p


def _next_profile(rng, previous, richness):
    shape = _profile(rng, len(previous), int(rng.integers(1, richness+2)))
    offset = max(int(np.max(previous+1-shape)),
                 int(np.max(np.maximum(previous[:-1], previous[1:])+1-
                            np.minimum(shape[:-1], shape[1:]))))
    return shape+offset+int(rng.integers(0, 2))


def _profiles(rng, width, count, richness, flat):
    cuts = [1+_profile(rng, width, int(rng.integers(1, 3)))]
    if flat:
        cuts.append(_next_profile(rng, cuts[-1], richness))
        thickness = int(np.max(np.abs(np.diff(cuts[-1]))))+int(rng.integers(1, 3))
        cuts.append(cuts[-1]+thickness)
    else:
        for _ in range(count-1):
            cuts.append(_next_profile(rng, cuts[-1], richness))
    cuts.append(np.full(width, int(cuts[-1].max())+int(rng.integers(1, 4))))
    return cuts


def _dilate(mask):
    padded = np.pad(mask, 1)
    h, w = mask.shape
    return np.logical_or.reduce([padded[y:y+h, x:x+w] for y in range(3) for x in range(3)])


def _scatter(rng, frame, target_pieces, bg, rectangle):
    normalized = [(cells-cells.min(axis=0), color) for cells, color in target_pieces]
    h, w = frame.shape
    t, left, floor, right = rectangle
    reserved = np.zeros_like(frame, dtype=bool)
    reserved[t:floor+1, left:right+1] = True
    reserved |= frame != bg
    for _ in range(30):
        result = frame.copy()
        forbidden = _dilate(reserved)
        order = sorted(range(len(normalized)), key=lambda i: -len(normalized[i][0])+rng.random()*5)
        success = True
        for i in order:
            cells, color = normalized[i]
            ph, pw = cells.max(axis=0)+1
            valid = np.ones((h-ph+1, w-pw+1), dtype=bool)
            for y, x in cells:
                valid &= ~forbidden[y:y+h-ph+1, x:x+w-pw+1]
            choices = np.argwhere(valid)
            if not len(choices):
                success = False
                break
            placed = cells+choices[int(rng.integers(len(choices)))]
            result[placed[:, 0], placed[:, 1]] = color
            mask = np.zeros_like(reserved)
            mask[placed[:, 0], placed[:, 1]] = True
            forbidden |= _dilate(mask)
        if success:
            return result
    # Guaranteed left bay: canvas construction reserves these dimensions.
    result, row = frame.copy(), 1
    for cells, color in normalized:
        placed = cells+np.array((row, 1))
        _require(np.all(result[placed[:, 0], placed[:, 1]] == bg), "fallback scatter space")
        result[placed[:, 0], placed[:, 1]] = color
        row += int(cells[:, 0].max())+2
    return result


def _sample_pair(rng, roles, code, kind, n, richness):
    bg, s, active = roles
    minimum_width = max(4, 2*n-1)
    width = int(rng.integers(minimum_width, min(10, minimum_width+3)+1))
    count = 2 if kind == "single" else (3 if kind == "flat" else
             int(rng.integers(3, 5)) if kind == "prefix" else
             int(rng.integers(4, 6)) if kind == "rich" else int(rng.integers(2, 6)))
    cuts = _profiles(rng, width, count, richness, kind == "flat")
    other = [c for c in range(10) if c not in roles]
    colors = [int(rng.choice(other)) for _ in range(count)]
    if kind == "flat":
        active_indices = [1]
    elif kind in ("prefix", "rich"):
        active_indices = [0, count-1]
    else:
        active_indices = [int(rng.integers(count))]
        if count >= 3 and rng.random() < .4:
            active_indices = [0, count-1]
    for i in active_indices:
        colors[i] = active
    full_heights = sum((cuts[i+1]-cuts[i] for i in active_indices), np.zeros(width, dtype=int))
    depth = int(cuts[-1][0])
    if kind == "flat":
        read_cut = int(rng.integers(1, min(int(cuts[1].min()), depth-2)+1))
    elif kind in ("single", "prefix", "rich"):
        i = active_indices[0]
        cut_low = int(cuts[i].min())+1
        cut_high = min(int(cuts[i+1].max())-1, depth-2)
        _require(cut_low <= cut_high, "active/window intersection")
        read_cut = int(rng.integers(cut_low, cut_high+1))
    else:
        read_cut = int(rng.integers(1, depth-1))
    heights_all = sum((np.maximum(0, cuts[i+1]-np.maximum(cuts[i], read_cut))
                       for i in active_indices), np.zeros(width, dtype=int))
    lane_options = [cols for cols in combinations(range(width), n)
                    if min(np.diff(cols)) >= 2 and
                    (kind not in ("single", "prefix") or len({int(heights_all[c]) for c in cols}) > 1) and
                    (kind != "single" or any(full_heights[c] > heights_all[c] for c in cols)) and
                    (kind != "prefix" or any(cuts[1][c] > max(cuts[0][c], read_cut) for c in cols))]
    _require(bool(lane_options), "lane/profile feasibility")
    local_lanes = lane_options[int(rng.integers(len(lane_options)))]
    heights = [int(heights_all[c]) for c in local_lanes]
    _require(max(heights) > 0, "some active pressure required")
    span = max(int(full_heights[c]) for c in local_lanes)+int(rng.integers(5 if kind == "prefix" else 3, 7))
    top = int(rng.integers(1, 3))
    base = top+span
    floor = base+depth+1
    read_row = floor-1-read_cut
    local_pieces = []
    for i in range(count):
        cells = np.array([(floor-1-u, c) for c in range(width)
                          for u in range(int(cuts[i][c]), int(cuts[i+1][c]))], dtype=int)
        local_pieces.append((cells, colors[i]))
    scatter_height = sum(int(np.ptp(cells[:, 0]))+1 for cells, _ in local_pieces)+count+1
    min_height = max(floor+2, scatter_height)
    _require(min_height <= 30, "vertical space")
    canvas_h = int(rng.integers(min_height, min(30, min_height+3)+1))
    min_width = 2*width+7
    _require(min_width <= 30, "horizontal space")
    canvas_w = int(rng.integers(min_width, 31))
    left = int(rng.integers(width+3, canvas_w-width-3))
    right = left+width+1
    lanes = [left+1+c for c in local_lanes]
    pins = []
    a = int(rng.choice([i for i, h in enumerate(heights) if h > 0]))
    b = int(rng.choice([i for i in range(n) if i != a]))
    prefix_mode = int(rng.integers(3))
    primary = int(rng.choice([c for c in range(10) if c not in (bg, s)]))
    secondary = int(rng.choice([c for c in range(10) if c not in (bg, s, primary)]))
    for i, height in enumerate(heights):
        capacity = span-height-1
        shift = 0 if height == 0 or rng.random() < .35 else int(rng.integers(1, height+1))
        if kind in ("single", "prefix"):
            if i == a:
                shift = int(rng.integers(1, height+1))
            if i == b:
                shift = 0
        length = capacity+shift
        prefix = 0 if prefix_mode == 0 else int(rng.integers(1 if prefix_mode == 1 else 0, length))
        if kind == "single" and i == a:
            prefix = 0
        if kind == "prefix" and i == a:
            prefix = shift+1
        payload_color = primary if kind in ("single", "prefix") or rng.random() < .55 else int(
            rng.choice([c for c in range(10) if c not in (bg, s)]))
        payload = [payload_color]*(length-prefix)
        if (kind == "prefix" and i == a) or (len(payload) >= 2 and rng.random() < .3):
            payload[-1] = secondary if secondary != payload_color else primary
        pins.append(tuple([s]*prefix+payload))
    frame = np.full((canvas_h, canvas_w), bg, dtype=np.int8)
    frame[top, left:right+1] = frame[floor, left:right+1] = s
    frame[top:floor+1, left] = frame[top:floor+1, right] = s
    frame[base, left:right+1] = s
    frame[base, left] = active
    frame[read_row, left-1] = frame[read_row, right+1] = active
    for c, root_height in enumerate(cuts[0]):
        frame[floor-int(root_height):floor, left+1+c] = s
    for col, seq in zip(lanes, pins):
        frame[top+1:top+1+len(seq), col] = seq
    pieces = [(cells+np.array((0, left+1)), color) for cells, color in local_pieces]
    inp = _scatter(rng, frame, pieces, bg, (top, left, floor, right))
    # Direct construction of the label; not a call to solve or its actuator.
    output = frame.copy()
    for cells, color in pieces:
        output[cells[:, 0], cells[:, 1]] = color
    for col in lanes:
        output[top+1:base, col] = bg
    for col, height, seq in zip(lanes, heights, pins):
        output[base-height:base, col] = s
        start = base-height-len(seq)
        for j, color in enumerate(seq):
            if start+j > top:
                output[start+j, col] = color
    pair = {"input": _orient(inp, code), "output": _orient(output, code)}
    f = _read(pair["input"])
    _require(np.array_equal(_actuate(f), pair["output"]), "construction/solver disagreement")
    return pair


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+" must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name+" outside legal range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    schedule = int(rng.integers(4))  # orientation, lanes, flat pressure, recombination
    initial_code = int(rng.integers(8))
    stable_roles = bool(rng.random() < .3)
    shared_roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            roles = shared_roles if stable_roles else tuple(map(int, rng.choice(10, 3, replace=False)))
            code = initial_code
            if schedule == 0 and split == "test":
                code = ((initial_code % 4)+int(rng.choice((1, 3)))) % 4 + 4*int(rng.integers(2))
            elif schedule == 3:
                code = (initial_code+index+4*(split == "test")) % 8
            if split == "train":
                kind = ("single", "prefix")[index] if index < 2 else "ordinary"
                lane_choices = (2, 3) if schedule == 1 else (2, 3, 4)
            else:
                kind = "flat" if schedule == 2 else "rich" if schedule == 3 else "ordinary"
                lane_choices = (4, 5) if schedule == 1 else (2, 3, 4, 5)
            for _ in range(180):
                n = int(rng.choice(lane_choices))
                try:
                    pair = _sample_pair(rng, roles, code, kind, n, 2 if kind == "rich" else 1)
                    f = _read(pair["input"])
                    if schedule == 2 and split == "train" and len(set(f["heights"])) == 1:
                        continue
                except ValueError:
                    continue
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded geometry search exhausted")
            seen.add(key)
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(f):
    s, active = f["structure"], f["active"]
    prefixes = [next(i for i, c in enumerate(seq) if c != s) for seq in f["sequences"]]
    separated = []
    for col in f["lanes"]:
        bits = f["card"][:f["read_row"]-f["base"], col-f["left"]-1] == active
        separated.append(int(bits[0])+int(np.count_nonzero(bits[1:] & ~bits[:-1])))
    return dict(shape=f["canonical"].shape, orientation=f["orientation"],
                roles=(f["background"], s, active), width=f["width"], depth=f["depth"],
                root_amplitude=max(f["root"])-min(f["root"]), pieces=len(f["pieces"]),
                lanes=len(f["lanes"]), heights=f["heights"], shifts=f["shifts"],
                lane_gaps=tuple(np.diff(f["lanes"]).tolist()),
                window_depth=f["read_row"]-f["base"],
                excluded_active=int(np.count_nonzero(f["card"][f["read_row"]-f["base"]:] == active)),
                separated_active=max(separated) >= 2,
                repeated_material=len({c for _, c in f["pieces"]}) < len(f["pieces"]),
                mixed_contact=min(f["shifts"]) == 0 < max(f["shifts"]),
                retained_moving_prefix=any(0 < d < p for d, p in zip(f["shifts"], prefixes)),
                payload_clip=any(d > p for d, p in zip(f["shifts"], prefixes)),
                multicolor_survivor=any(len({c for c in seq[d:] if c != s}) >= 2
                                        for seq, d in zip(f["sequences"], f["shifts"])),
                flat=len(set(f["heights"])) == 1)


def _validate_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "shape preservation")
    f = _read(inp)
    expected = _actuate(f)
    _require(np.array_equal(out, expected), "incorrect output")
    canonical_out = _unorient(out, f["orientation"])
    # Fixed structure/selector are unchanged, even when pistons add new structure.
    fixed = np.zeros_like(f["canonical"], dtype=bool)
    t, b, floor, l, r = f["top"], f["base"], f["floor"], f["left"], f["right"]
    fixed[t, l:r+1] = fixed[floor, l:r+1] = True
    fixed[t:floor+1, l] = fixed[t:floor+1, r] = True
    fixed[b, l:r+1] = True
    fixed[f["read_row"], l-1] = fixed[f["read_row"], r+1] = True
    for c, h in enumerate(f["root"]):
        fixed[floor-h:floor, l+1+c] = True
    _require(np.array_equal(canonical_out[fixed], f["canonical"][fixed]), "fixed cells changed")
    used, target_union = set(), set()
    for i, target in f["placements"]:
        source, color = f["pieces"][i]
        translated = source+target.min(axis=0)-source.min(axis=0)
        _require(np.array_equal(translated, target[np.lexsort((target[:, 1], target[:, 0]))]),
                 "piece mask not translated exactly")
        _require(np.all(canonical_out[target[:, 0], target[:, 1]] == color), "material changed")
        _require(np.all(canonical_out[source[:, 0], source[:, 1]] == f["background"]), "source not erased")
        cells = set(map(tuple, target.tolist()))
        _require(i not in used and not cells & target_union, "assembly overlap/reuse")
        used.add(i)
        target_union |= cells
    _require(used == set(range(len(f["pieces"]))), "piece omitted")
    _require(not np.any(f["card"] == f["background"]), "card has gaps")
    for col, h, shift, seq in zip(f["lanes"], f["heights"], f["shifts"], f["sequences"]):
        _require(np.all(canonical_out[b-h:b, col] == f["structure"]), "piston length")
        _require(tuple(map(int, canonical_out[t+1:b-h, col])) == seq[shift:], "pin suffix/order/clipping")
    _require(not np.array_equal(inp, out), "identity pair")
    return f


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal puzzle counts")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            f = _validate_pair(pair)
            key = _key(np.asarray(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            if split == "train":
                training.append(f)
    features = [_features(f) for f in training]
    for flag in ("mixed_contact", "retained_moving_prefix", "payload_clip",
                 "multicolor_survivor", "separated_active"):
        _require(any(item[flag] for item in features), "training lacks "+flag)
    _require(any(len(set(f["heights"])) > 1 for f in training), "training lacks nonflat pressure")
    no_a = []
    no_b = []
    for f in training:
        # This is the stronger shortcut: sum every loose component's active
        # pixels per local column, retaining the entire unassembled inventory.
        # It would be sufficient without the visibly bounded reading window.
        inventory = tuple(int(np.count_nonzero(f["card"][:, col-f["left"]-1] == f["active"]))
                          for col in f["lanes"])
        no_a.append(not np.array_equal(_actuate(f, inventory), _actuate(f)))
        no_b.append(not np.array_equal(_actuate(f, stop_after_card=True), _actuate(f)))
    _require(any(no_a) and any(no_b), "parent-concept ablation failed")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                       for p in puzzle[split]) for split in ("train", "test"))


def _regression_tests():
    """Small hand-specified pixel cases for the window, contact and uniqueness."""
    inp = np.zeros((15, 20), dtype=np.int8)
    top, base, floor, left, right, read_row = 1, 7, 13, 10, 15, 10
    inp[top, left:right+1] = inp[floor, left:right+1] = 8
    inp[top:floor+1, left] = inp[top:floor+1, right] = 8
    inp[base, left:right+1] = 8
    inp[base, left] = 2
    inp[read_row, left-1] = inp[read_row, right+1] = 2
    cuts = (np.array((1, 2, 1, 2)), np.array((3, 3, 4, 4)), np.full(4, 5))
    for c, h in enumerate(cuts[0]):
        inp[floor-h:floor, left+1+c] = 8
    destinations = []
    for i, (lo, hi) in enumerate(zip(cuts, cuts[1:])):
        cells = np.array([(floor-1-u, left+1+c) for c in range(4)
                          for u in range(int(lo[c]), int(hi[c]))])
        destinations.append((cells, 2+i))
        source = cells-cells.min(axis=0)+np.array((1+5*i, 1+2*i))
        inp[source[:, 0], source[:, 1]] = 2+i
    inp[2:6, 11] = (8, 4, 4, 4)
    inp[2:6, 13] = (8, 5, 6, 6)
    card_output = inp.copy()
    card_output[:, :8] = 0  # The two known scattered originals.
    for cells, color in destinations:
        card_output[cells[:, 0], cells[:, 1]] = color
    expected = card_output.copy()
    expected[2:7, 11] = (8, 4, 4, 4, 8)
    expected[2:7, 13] = (5, 6, 6, 8, 8)
    f = _read(inp)
    _require(f["heights"] == (1, 2) and f["shifts"] == (0, 1), "inclusive reading window")
    _require(np.array_equal(solve(inp), expected), "hand-specified window/pin result")
    # Move only the visible reading marks: same unique assembly, new pressure.
    moved = inp.copy()
    moved[10, [9, 16]] = 0
    moved[11, [9, 16]] = 2
    expected_moved = card_output.copy()
    expected_moved[10, [9, 16]] = 0
    expected_moved[11, [9, 16]] = 2
    expected_moved[2:7, 11] = (4, 4, 4, 8, 8)
    expected_moved[2:7, 13] = (6, 6, 8, 8, 8)
    _require(np.array_equal(solve(moved), expected_moved), "window position must control pressure")
    # A zero-count lane contacts the separator itself and stays stationary.
    zero = inp.copy()
    zero[10, [9, 16]] = 0
    zero[9, [9, 16]] = 2
    zero[2:7, 11] = (8, 4, 4, 4, 4)
    expected_zero = card_output.copy()
    expected_zero[10, [9, 16]] = 0
    expected_zero[9, [9, 16]] = 2
    expected_zero[2:7, 11] = (8, 4, 4, 4, 4)
    expected_zero[2:7, 13] = (8, 5, 6, 6, 8)
    _require(_read(zero)["heights"] == (0, 1), "zero pressure reading")
    _require(np.array_equal(solve(zero), expected_zero), "zero pressure/contact result")
    # Identical masks can exchange order. Different colors yield two cards;
    # identical colors yield the same card and are correctly deduplicated.
    root = np.array((1, 2, 1, 2))
    pieces = []
    for index, (lo, hi, color) in enumerate(((root, root+2, 2), (root, root+2, 3),
                                             (root+4, np.full(4, 8), 4))):
        cells = np.array([(20-u, 10*index+c) for c in range(4)
                          for u in range(int(lo[c]), int(hi[c]))])
        pieces.append((cells, color))
    fixture = dict(background=0, structure=8, width=4, depth=8, root=tuple(root),
                   base=1, floor=10, left=1, pieces=pieces)
    try:
        _assemble(fixture)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous colored assembly accepted")
    fixture["pieces"][1] = (pieces[1][0], 2)
    _require(_assemble(fixture)[0].shape == (8, 4), "equivalent object permutations rejected")


def self_test():
    """Bounded semantic regressions and measurements over exactly 42 base seeds."""
    _regression_tests()
    records, signatures, structures, shapes, regimes = [], set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), "nondeterministic seed")
        signatures.add(_puzzle_key(puzzle))
        local = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _read(pair["input"])
                feature = _features(f)
                records.append(feature)
                local.append(feature)
                shapes.add(pair["input"].shape)
                roles, normalized = {}, []
                for value in pair["input"].flat:
                    normalized.append(roles.setdefault(int(value), len(roles)))
                structures.add((pair["input"].shape, tuple(normalized)))
                before = pair["input"].copy()
                _require(np.array_equal(solve(before), pair["output"]), "solver disagreement")
                _require(np.array_equal(before, pair["input"]), "solve mutated input")
        train = local[:4]
        test = local[4:]
        regimes.add((len({f["orientation"] for f in train}) > 1,
                     any(f["orientation"] not in {g["orientation"] for g in train} for f in test),
                     max(f["lanes"] for f in train) < min(f["lanes"] for f in test),
                     all(not f["flat"] for f in train) and all(f["flat"] for f in test)))
    _require(len(signatures) == 42, "seed diversity")
    _require(len(structures) >= 200 and len(shapes) >= 25, "geometry diversity")
    for name, wanted in (("pieces", {2, 3, 4, 5}), ("lanes", {2, 3, 4, 5}),
                         ("orientation", set(range(8)))):
        _require({f[name] for f in records} >= wanted, "missing variation: "+name)
    _require(len({f["width"] for f in records}) >= 5 and len({f["depth"] for f in records}) >= 7,
             "card scale diversity")
    _require(len({f["heights"] for f in records}) >= 70, "pressure profile diversity")
    _require(any(len(set(f["lane_gaps"])) > 1 for f in records), "nonuniform lanes absent")
    _require(any(f[0] for f in regimes) and any(f[1] for f in regimes) and
             any(f[2] for f in regimes) and any(f[3] for f in regimes), "transfer regimes missing")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1700+a*10+b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Equivariance on actual pixels, with all literal roles recolored.
    pair = generate_puzzle(0)["train"][0]
    for code in range(8):
        mapping = np.array([(3*i+code) % 10 for i in range(10)], dtype=np.int8)
        transformed = _orient(mapping[pair["input"]], code)
        _require(np.array_equal(solve(transformed), _orient(mapping[pair["output"]], code)),
                 "orientation/recoloring equivariance")
    # Alter a known pin output: labels must be checked, not merely well-formed.
    puzzle = generate_puzzle(0)
    broken = {split: [{role: np.array(value, copy=True) for role, value in p.items()}
                       for p in puzzle[split]] for split in puzzle}
    broken["test"][0]["output"] = broken["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    return {"seeds": 42, "pairs": len(records), "distinct_puzzles": len(signatures),
            "input_structures": len(structures), "grid_shapes": len(shapes),
            "piece_counts": sorted({f["pieces"] for f in records}),
            "lane_counts": sorted({f["lanes"] for f in records}),
            "widths": sorted({f["width"] for f in records}),
            "depths": sorted({f["depth"] for f in records}),
            "pressure_profiles": len({f["heights"] for f in records}),
            "window_depths": sorted({f["window_depth"] for f in records}),
            "excluded_active_pairs": sum(f["excluded_active"] > 0 for f in records),
            "zero_pressure_lanes": sum(h == 0 for f in records for h in f["heights"]),
            "orientations": sorted({f["orientation"] for f in records}),
            "mixed_contact_pairs": sum(f["mixed_contact"] for f in records),
            "separated_active_pairs": sum(f["separated_active"] for f in records),
            "payload_clipping_pairs": sum(f["payload_clip"] for f in records),
            "retained_moving_prefix_pairs": sum(f["retained_moving_prefix"] for f in records),
            "multicolor_survivor_pairs": sum(f["multicolor_survivor"] for f in records),
            "independent_solver": False, "blind_prediction": False}


if __name__ == "__main__":
    print(self_test())
