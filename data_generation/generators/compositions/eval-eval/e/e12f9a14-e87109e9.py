"""Keyed looms: port-launched threads collectively choose and populate a ribbon.

Only the standard library and NumPy are used. The public solver reads the
visible key, launchers, and collector; it never reads generation parameters.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


PORTS = {
    "NW": (-1, -1), "N0": (-1, 0), "N1": (-1, 1), "NE": (-1, 2),
    "W0": (0, -1), "E0": (0, 2), "W1": (1, -1), "E1": (1, 2),
    "SW": (2, -1), "S0": (2, 0), "S1": (2, 1), "SE": (2, 2),
}
COMPASS = ((-1, 0), (-1, 1), (0, 1), (1, 1),
           (1, 0), (1, -1), (0, -1), (-1, -1))
PAYLOADS = (1, 6, 7, 9)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(np.all((a >= 0) & (a <= 9)), "colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _inside(point, shape):
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


def _direction(offset):
    r, c = offset
    return (-1 if r == -1 else 1 if r == 2 else 0,
            -1 if c == -1 else 1 if c == 2 else 0)


def _rotate(v, side):
    r, c = v
    return (-c, r) if side == "L" else (c, -r)


def _dominant(v):
    r, c = v
    _need(abs(r) != abs(c), "collector vote has a tie or cancels")
    return (int(np.sign(r)), 0) if abs(r) > abs(c) else (0, int(np.sign(c)))


def _read_header(a):
    frame = int(a[0, 0])
    _need(np.all(a[0] == frame), "missing top key frame")
    bottoms = np.flatnonzero(np.all(a[1:] == frame, axis=1)) + 1
    _need(len(bottoms) > 0, "missing bottom key frame")
    bottom = int(bottoms[0])
    _need(2 <= bottom <= 4 and bottom < a.shape[0] - 1, "invalid key height")
    middle = a[1:bottom]
    _need(np.all(middle == middle[0]), "key interior rows must agree")
    row = middle[0]
    _need(row[0] == row[-1] == frame, "missing key side walls")
    runs, c = [], 0
    while c < len(row):
        if row[c] == frame:
            c += 1
            continue
        start = c
        while c < len(row) and row[c] != frame:
            c += 1
        runs.append((start, c))
    _need(1 <= len(runs) <= 4 and all(b-a >= 3 for a, b in runs), "ambiguous panels")
    background = int(row[runs[0][0]+1])
    _need(background != frame, "key interior equals frame")
    table = {}
    for start, end in runs:
        _need(np.all(row[start+1:end-1] == background), "inconsistent panel interiors")
        left, right = int(row[start]), int(row[end-1])
        _need((left == background) != (right == background), "panel needs one side bar")
        color, side = (right, "R") if left == background else (left, "L")
        _need(color not in (frame, background) and color not in table, "conflicting key colors")
        table[color] = side
    return bottom+1, frame, background, table, tuple(b-a+2 for a, b in runs)


def _decode(value):
    a = _grid(value)
    crop, frame, bg, table, widths = _read_header(a)
    field = a[crop:]
    keyed = np.argwhere(np.isin(field, list(table)))
    _need(len(keyed) > 0, "no collector")
    low, high = keyed.min(axis=0), keyed.max(axis=0)+1
    r, c = map(int, low)
    k = int(high[0]-r)
    color = int(field[r, c])
    _need(k in (3, 5, 7) and high[1]-c == k and len(keyed) == k*k
          and np.all(field[r:r+k, c:c+k] == color), "one solid odd square collector required")
    launchers, covered, footprints = [], set(map(tuple, keyed)), set()
    for rr in range(field.shape[0]-1):
        for cc in range(field.shape[1]-1):
            payload = int(field[rr, cc])
            if payload in (frame, bg) or payload in table:
                continue
            if not np.all(field[rr:rr+2, cc:cc+2] == payload):
                continue
            ring = [(rr+dr, cc+dc, name) for name, (dr, dc) in PORTS.items()
                    if _inside((rr+dr, cc+dc), field.shape)]
            values = [int(field[x, y]) for x, y, _ in ring]
            if any(v not in (frame, bg) for v in values) or values.count(frame) < 3 or bg not in values:
                continue
            footprint = {(x, y) for x in range(rr-1, rr+3) for y in range(cc-1, cc+3)
                         if _inside((x, y), field.shape)}
            _need(not footprint & footprints, "launcher footprints overlap")
            footprints |= footprint
            ports = []
            for x, y, name in ring:
                if field[x, y] == bg:
                    ports.append({"p": (x, y), "v": _direction(PORTS[name]),
                                  "color": payload, "name": name, "launcher": len(launchers)})
                else:
                    covered.add((x, y))
            covered |= {(x, y) for x in (rr, rr+1) for y in (cc, cc+1)}
            launchers.append({"core": (rr, cc), "color": payload, "ports": ports,
                              "clipped": len(footprint) < 16})
    _need(2 <= len(launchers) <= 5, "need two to five identifiable launchers")
    actual = set(map(tuple, np.argwhere(field != bg)))
    _need(covered == actual, "unaccounted field objects")
    return dict(field=field, crop=crop, frame=frame, bg=bg, table=table,
                panel_widths=widths, collector=(r, c, k, color), launchers=launchers)


def _threads(info):
    """Straight segments end at the collector or boundary, never at old paint."""
    field, bg = info["field"], info["bg"]
    collector_color = info["collector"][3]
    occupied, diagonals, result = set(), set(), []
    for launcher in info["launchers"]:
        for port in launcher["ports"]:
            point, v = port["p"], port["v"]
            path, hit = [], False
            while _inside(point, field.shape):
                value = int(field[point])
                if value == collector_color:
                    hit = True
                    break
                _need(value == bg, "a thread hits a launcher")
                _need(point not in occupied, "threads overlap outside the collector")
                occupied.add(point)
                if path and v[0] and v[1]:
                    mid = (path[-1][0]+point[0], path[-1][1]+point[1])
                    slope = v[0]*v[1]
                    _need((mid, -slope) not in diagonals, "threads cross between cells")
                    diagonals.add((mid, slope))
                path.append(point)
                point = (point[0]+v[0], point[1]+v[1])
            result.append(dict(port, path=path, hit=hit, contact=point if hit else None))
    arriving = [p for p in result if p["hit"]]
    _need(2 <= len(arriving) <= 4, "collector must receive two to four threads")
    vote = tuple(sum(p["v"][i] for p in arriving) for i in (0, 1))
    _dominant(vote)
    return result, arriving, vote, occupied


def _render(info, threads, arriving, vote, *, ignore_key=False,
            use_first=False, reverse=False, fixed_width=None, all_turn=None,
            order_mode="projection"):
    field = info["field"]
    answer = field.copy()
    for thread in threads:
        for p in thread["path"]:
            answer[p] = thread["color"]
    r, c, k, color = info["collector"]
    heading = _dominant(vote)
    if use_first:
        v = min(arriving, key=lambda p: p["p"])["v"]
        # A specific plausible shortcut: take the first donor and prefer the
        # row component when its diagonal has no unique dominant axis.
        heading = (v[0], 0) if v[0] else (0, v[1])
    side = all_turn or info["table"][color]
    outgoing = heading if ignore_key else _rotate(heading, side)
    lateral = _rotate(outgoing, "R")
    projection = lambda p: p["p"][0]*lateral[0]+p["p"][1]*lateral[1]
    _need(len({projection(p) for p in arriving}) == len(arriving), "ribbon ordering has a spatial tie")
    if order_mode == "reading":
        order_key = lambda p: p["p"]
    elif order_mode == "contact":
        order_key = lambda p: (p["contact"][0]*lateral[0]+p["contact"][1]*lateral[1], p["p"])
    elif order_mode == "color":
        order_key = lambda p: p["color"]
    else:
        order_key = projection
    ordered = sorted(arriving, key=order_key, reverse=reverse)
    width = len(ordered) if fixed_width is None else fixed_width
    _need(1 <= width <= k, "ribbon does not fit the collector face")
    center = (r+k//2, c+k//2)
    starts, ribbon = [], set()
    inbound = {p for t in threads for p in t["path"]}
    for i in range(width):
        offset = i-width//2
        p = (center[0]+outgoing[0]*(k//2+1)+lateral[0]*offset,
             center[1]+outgoing[1]*(k//2+1)+lateral[1]*offset)
        starts.append(p)
        payload = ordered[i % len(ordered)]["color"]
        while _inside(p, field.shape):
            _need(field[p] == info["bg"], "outgoing ribbon hits a foreground object")
            _need(p not in inbound, "outgoing ribbon crosses an incoming or missed thread")
            _need(p not in ribbon, "ribbon lanes overlap")
            ribbon.add(p)
            answer[p] = payload
            p = (p[0]+outgoing[0], p[1]+outgoing[1])
    return answer, dict(vote=vote, heading=heading, outgoing=outgoing, side=side,
                        width=width, ordered=ordered, starts=starts, ribbon=ribbon)


def _analyze(value):
    info = _decode(value)
    threads, arriving, vote, occupied = _threads(info)
    answer, meta = _render(info, threads, arriving, vote)
    info.update(meta, threads=threads, arriving=arriving, inbound=occupied)
    return answer, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the cropped answer entirely from the visible input."""
    return _analyze(input_grid)[0]


def _header(rng, width, height, keys, sides):
    sizes = np.full(len(keys), 5, dtype=int)
    sizes += rng.multinomial(width-5*len(keys), np.full(len(keys), 1/len(keys)))
    band = np.zeros((height, width), dtype=np.int8)
    band[[0, -1], :] = 8
    col = 0
    for key, size in zip(rng.permutation(keys), sizes):
        band[:, col] = band[:, col+size-1] = 8
        band[1:-1, col+1 if sides[int(key)] == "L" else col+size-2] = key
        col += int(size)
    return band


def _put_launcher(field, core, color, names):
    r, c = core
    _need(0 <= r < field.shape[0]-1 and 0 <= c < field.shape[1]-1, "core out of bounds")
    footprint = [(x, y) for x in range(r-1, r+3) for y in range(c-1, c+3)
                 if _inside((x, y), field.shape)]
    _need(all(field[p] == 0 for p in footprint), "placement overlaps foreground")
    visible = [name for name, (dr, dc) in PORTS.items() if _inside((r+dr, c+dc), field.shape)]
    _need(set(names) <= set(visible) and len(visible)-len(names) >= 3, "invalid port set")
    candidate = field.copy()
    candidate[r:r+2, c:c+2] = color
    for name in visible:
        dr, dc = PORTS[name]
        candidate[r+dr, c+dc] = 0 if name in names else 8
    return candidate


def _contact(rng, box, v):
    r, c, k = box
    faces = []
    if v[0]:
        faces.append((r+k-1 if v[0] < 0 else r, int(rng.integers(c, c+k))))
    if v[1]:
        faces.append((int(rng.integers(r, r+k)), c if v[1] > 0 else c+k-1))
    return faces[int(rng.integers(len(faces)))]


def _directions(rng, n, teaching=False):
    if n == 2:
        # Either a symmetric diagonal pair, or an agreeing cardinal/diagonal
        # pair. Both have a unique resultant face.
        base = ((-1, -1), (-1, 1)) if rng.random() < .65 else ((-1, 0), (-1, 1))
    elif n == 3 and (teaching or rng.random() < .25):
        # A cardinal donor is outvoted by two diagonals. This explicitly
        # distinguishes collective voting from following the cardinal donor.
        base = ((-1, 0), (-1, 1), (1, 1))
    else:
        for _ in range(40):
            base = tuple(COMPASS[int(i)] for i in rng.choice(8, n, replace=rng.random() < .3))
            vote = tuple(sum(v[i] for v in base) for i in (0, 1))
            if abs(vote[0]) != abs(vote[1]) and len(set(base)) >= 2:
                break
        else:
            base = ((-1, 0), (-1, 1), (1, 1)) if n == 3 else ((-1, 0), (-1, 1), (-1, -1), (0, 1))
    rotation = int(rng.integers(4))
    for _ in range(rotation):
        base = tuple(_rotate(v, "L") for v in base)
    if rng.random() < .5:
        base = tuple((r, -c) for r, c in base)
    result = list(base)
    rng.shuffle(result)
    return result


def _scene(rng, n, side, k, extra_target, active_color=None, teaching=False):
    """Sample contact locations and donor distances, then certify visible paths."""
    last_error = None
    for attempt in range(250):
        hh = int(rng.integers(3, 6))
        key_count = int(rng.integers(1, 5))
        h = int(rng.integers(max(16, k+9), 31-hh))
        w = int(rng.integers(max(16, 5*key_count, k+9), 31))
        r = int(rng.integers(2, h-k-1))
        c = int(rng.integers(2, w-k-1))
        field = np.zeros((h, w), dtype=np.int8)
        field[r:r+k, c:c+k] = 2
        directions = _directions(rng, n, teaching)
        failed = False
        for color, v in zip(PAYLOADS, directions):
            options = [name for name, offset in PORTS.items() if _direction(offset) == v]
            for trial in range(100):
                target = _contact(rng, (r, c, k), v)
                distance = int(rng.integers(3, 13))
                p = (target[0]-v[0]*distance, target[1]-v[1]*distance)
                name = str(rng.choice(options))
                dr, dc = PORTS[name]
                try:
                    candidate = _put_launcher(field, (p[0]-dr, p[1]-dc), color, (name,))
                except ValueError:
                    continue
                field = candidate
                break
            else:
                failed = True
                break
        if failed:
            continue
        keys = list(range(2, 2+key_count))
        sides = {key: str(rng.choice(("L", "R"))) for key in keys}
        sides[2] = side
        band = _header(rng, w, hh, keys, sides)
        try:
            out, info = _analyze(np.vstack((band, field)))
            _need(len(info["arriving"]) == n and len(info["ribbon"]) >= 3*n, "too little visible ribbon")
            _need(len({p["color"] for p in info["arriving"]}) == n, "arrival colors repeat")
            if teaching:
                alternative = _alternative(info, use_first=True)
                _need(alternative is None or not np.array_equal(alternative, out), "first-donor shortcut survives")
        except ValueError as error:
            last_error = error
            continue
        # A missed emission may come from a separate launcher or an extra port.
        # Count both toward the requested number, so the split shift is visible.
        added = 0
        if extra_target and rng.random() < .3:
            for trial in range(40):
                rr = int(rng.integers(h-1)); cc = int(rng.integers(w-1))
                name = str(rng.choice(tuple(PORTS)))
                payload = PAYLOADS[n] if n < 4 else int(rng.choice(PAYLOADS))
                try:
                    candidate = _put_launcher(field, (rr, cc), payload, (name,))
                    new_out, new_info = _analyze(np.vstack((band, candidate)))
                    _need(len(new_info["arriving"]) == n, "extra launcher contributes unexpectedly")
                except ValueError:
                    continue
                field, out, info = candidate, new_out, new_info
                added = 1
                break
        # Optional missed ports alter emission structure, but not the vote.
        for trial in range(75):
            if added >= extra_target:
                break
            launcher = info["launchers"][int(rng.integers(len(info["launchers"])))]
            rr, cc = launcher["core"]
            options = [name for name, (dr, dc) in PORTS.items()
                       if _inside((rr+dr, cc+dc), field.shape) and field[rr+dr, cc+dc] == 8]
            if not options:
                continue
            name = str(rng.choice(options))
            dr, dc = PORTS[name]
            candidate = field.copy()
            candidate[rr+dr, cc+dc] = 0
            try:
                new_out, new_info = _analyze(np.vstack((band, candidate)))
                _need(len(new_info["arriving"]) == n, "extra port changes arrival count")
            except ValueError:
                continue
            field, out, info = candidate, new_out, new_info
            added += 1
        if added < extra_target:
            continue
        palette = rng.permutation(10)
        if active_color is not None:
            index = int(np.flatnonzero(palette == active_color)[0])
            palette[index], palette[2] = palette[2], palette[index]
        pair = {"input": palette[np.vstack((band, field))].astype(np.int8),
                "output": palette[out].astype(np.int8)}
        # Reparse actual returned grids, rather than trusting latent metadata.
        _, returned = _analyze(pair["input"])
        if teaching:
            fits = False
            for alternative in ({"use_first": True}, {"order_mode": "reading"},
                                {"order_mode": "contact"}, {"order_mode": "color"}):
                other = _alternative(returned, **alternative)
                fits |= other is not None and np.array_equal(other, pair["output"])
            if fits:
                continue
        return pair
    raise RuntimeError(f"bounded loom construction exhausted: {last_error}")


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return one coherent family instance, with visible conventions per pair."""
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(("more_arrivals", "larger_collector", "more_ports")))
    active_color = int(rng.integers(10))
    first_side = str(rng.choice(("L", "R")))
    lessons = [(2, first_side, 3, 1), (3, "R" if first_side == "L" else "L", 5, 1)]
    for index in range(2, num_train):
        n = int(rng.integers(2, 4 if shift == "more_arrivals" else 5))
        k = int(rng.choice((3, 5) if n <= 3 else (5,)))
        lessons.append((n, str(rng.choice(("L", "R"))), k,
                        int(rng.integers(0, 2 if shift == "more_ports" else 3))))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train":
                n, side, k, extras = lessons[index]
            else:
                n = 4 if shift == "more_arrivals" else int(rng.integers(2, 5))
                side = str(rng.choice(("L", "R")))
                k = 7 if shift == "larger_collector" else int(rng.choice((5, 7)))
                extras = 2 if shift == "more_ports" else int(rng.integers(0, 4))
            for attempt in range(12):
                pair = _scene(rng, n, side, k, extras,
                              active_color if split == "train" and index < 2 else None,
                              teaching=split == "train" and index == 1)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("could not create a unique input")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative(info, **kwargs):
    try:
        return _render(info, info["threads"], info["arriving"], info["vote"], **kwargs)[0]
    except ValueError:
        # An alternative that runs into a launcher or a thread is inconsistent
        # with the specified unchanged-foreground and separated-path evidence.
        return None


def validate_generated_puzzle(puzzle):
    """Validate schema, geometry, collective routing, conservation, and lessons."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys"
    assert all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists"
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid counts"
    seen, infos = set(), {"train": [], "test": []}
    alternatives = ({"ignore_key": True}, {"use_first": True}, {"reverse": True},
                    {"fixed_width": 2}, {"all_turn": "L"}, {"all_turn": "R"},
                    {"order_mode": "reading"}, {"order_mode": "contact"},
                    {"order_mode": "color"})
    rejected = [False]*len(alternatives)
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair"
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            expected, info = _analyze(a)
            assert np.array_equal(expected, b), "output differs from input-derived rule"
            assert b.shape == info["field"].shape, "incorrect crop"
            field, bg = info["field"], info["bg"]
            assert np.array_equal(b[field != bg], field[field != bg]), "foreground changed"
            changed = set(map(tuple, np.argwhere(b != field)))
            assert changed == info["inbound"] | info["ribbon"], "incorrect path support"
            assert info["width"] == len(info["arriving"]), "arrival/ribbon conservation failed"
            assert len({p["color"] for p in info["arriving"]}) == info["width"], "unreadable repeated arrival colors"
            assert len(info["ribbon"]) >= 3*info["width"], "ribbon is too short to read"
            infos[split].append(info)
            if split == "train":
                for i, alt in enumerate(alternatives):
                    other = _alternative(info, **alt)
                    rejected[i] |= other is None or not np.array_equal(other, b)
    train = infos["train"]
    assert {"L", "R"} <= {q["side"] for q in train}, "both key meanings must be demonstrated"
    assert {2, 3} <= {q["width"] for q in train}, "training must teach two ribbon widths"
    assert {1, 2} <= {sum(map(abs, p["v"])) for q in train for p in q["arriving"]}, "teach side and corner ports"
    assert any(any(not p["hit"] for p in q["threads"]) for q in train), "teach that misses do not contribute"
    assert all(rejected), "training leaves a named shortcut undistinguished"
    tables = {}
    for q in train:
        tables.setdefault(q["collector"][3], set()).add(q["side"])
    assert any(len(v) == 2 for v in tables.values()), "teach that the visible key overrides fixed color meanings"
    # The specified test shift is visible in arrival counts, scale, or port counts.
    train_width = max(q["width"] for q in train)
    train_k = max(q["collector"][2] for q in train)
    train_extra = max(len(q["threads"])-q["width"] for q in train)
    assert all(q["width"] > train_width or q["collector"][2] > train_k
               or len(q["threads"])-q["width"] > train_extra for q in infos["test"]), "missing test shift"


def _fingerprint(puzzle):
    encoded = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
               for s in ("train", "test")}
    return hashlib.sha256(json.dumps(encoded, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _fixture():
    """A hand-defined NE+NW encounter, independent of randomized placement."""
    field = np.zeros((19, 23), dtype=np.int8)
    field[4:9, 8:13] = 2
    # Contacts (8,8)/(8,12), reached at distances 4/5 from the two ports.
    field = _put_launcher(field, (13, 2), 1, ("NE",))
    field = _put_launcher(field, (14, 18), 6, ("NW",))
    band = np.zeros((3, 23), dtype=np.int8)
    band[[0, 2], :] = 8
    band[:, [0, 22]] = 8
    band[1, 1] = 2  # left key
    return np.vstack((band, field))


def _majority_fixture():
    """N+NE+SE sums to (-1,2); one extra eastward port misses the loom."""
    field = np.zeros((25, 26), dtype=np.int8)
    field[8:13, 12:17] = 2
    field = _put_launcher(field, (20, 16), 1, ("N0", "E0"))
    field = _put_launcher(field, (19, 3), 6, ("NE",))
    field = _put_launcher(field, (2, 6), 9, ("SE",))
    band = np.zeros((3, 26), dtype=np.int8)
    band[[0, 2], :] = 8
    band[:, [0, 12, 13, 25]] = 8
    band[1, 1] = 2
    band[1, 24] = 3  # unused right-turn instruction
    return np.vstack((band, field))


def self_test():
    """Bounded 42-seed variation audit plus hand-derived and negative checks."""
    fixture = _fixture()
    answer, q = _analyze(fixture)
    assert q["vote"] == (-2, 0) and q["outgoing"] == (0, -1)
    assert q["width"] == 2 and q["starts"] == [(7, 7), (6, 7)]
    assert [p["color"] for p in q["ordered"]] == [6, 1]
    assert np.all(answer[7, :8] == 6) and np.all(answer[6, :8] == 1)
    right = fixture.copy(); right[1, 1] = 0; right[1, -2] = 2
    right_answer, qr = _analyze(right)
    assert qr["outgoing"] == (0, 1) and qr["starts"] == [(5, 13), (6, 13)]
    assert np.all(right_answer[5, 13:] == 1) and np.all(right_answer[6, 13:] == 6)
    majority, qm = _analyze(_majority_fixture())
    assert qm["vote"] == (-1, 2) and qm["heading"] == (0, 1) and qm["outgoing"] == (-1, 0)
    assert qm["width"] == 3 and len(qm["threads"]) == 4
    assert qm["starts"] == [(7, 13), (7, 14), (7, 15)]
    assert [p["color"] for p in qm["ordered"]] == [6, 9, 1]
    assert np.all(majority[:8, 13] == 6) and np.all(majority[:8, 14] == 9) and np.all(majority[:8, 15] == 1)
    # A canceled and a diagonally tied vote are intentionally outside the family.
    for vote in ((0, 0), (2, 2), (-1, 1)):
        try:
            _dominant(vote)
        except ValueError:
            pass
        else:
            raise AssertionError("tied vote accepted")
    signatures, shapes, sizes, widths, headings, port_sets, roles, votes = set(), set(), set(), set(), set(), set(), set(), set()
    counts, clipping, missed, mixed = set(), False, False, False
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _fingerprint(puzzle)
        assert signature not in signatures, "puzzle diversity collapsed"
        signatures.add(signature)
        if seed in (0, 17, 41):
            assert signature == _fingerprint(generate_puzzle(seed)), "nondeterministic puzzle"
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _, info = _analyze(pair["input"])
                shapes.add(info["field"].shape); sizes.add(info["collector"][2])
                widths.add(info["width"]); headings.add(info["outgoing"])
                counts.add(len(info["launchers"])); votes.add(info["vote"])
                roles.add((info["bg"], info["frame"], info["collector"][3]))
                for launcher in info["launchers"]:
                    port_sets.add(tuple(p["name"] for p in launcher["ports"]))
                    clipping |= launcher["clipped"]
                missed |= any(not p["hit"] for p in info["threads"])
                mixed |= len({sum(map(abs, p["v"])) for p in info["arriving"]}) == 2
    assert len(shapes) >= 50 and len(roles) >= 100 and len(port_sets) >= 35
    assert sizes == {3, 5, 7} and widths == {2, 3, 4}
    assert len(headings) == 4 and len(votes) >= 12 and {2, 3, 4} <= counts
    assert clipping and missed and mixed
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(101+10*nt+ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (5, 5), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    corrupt = generate_puzzle(0, 2, 1)
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
