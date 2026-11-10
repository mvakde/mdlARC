"""Raster-fed color bins: normalize cards, split masks, and settle in arrival order.

Only NumPy and the standard library are used. Parsing and solving require only
the input pixels; the sampler does not supply conventions or layout metadata.
"""
from __future__ import annotations

import json

import numpy as np


FAST, SLOW, ORIGIN = 1, 2, 3
FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grids must contain integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(a, background, monochrome=True, diagonal=False):
    unseen = set(map(tuple, np.argwhere(a != background).tolist()))
    result = []
    neighbors = EIGHT if diagonal else FOUR
    while unseen:
        first = min(unseen)
        color = int(a[first])
        unseen.remove(first)
        todo, cells = [first], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in neighbors:
                p = (r + dr, c + dc)
                if p in unseen and (not monochrome or int(a[p]) == color):
                    unseen.remove(p)
                    todo.append(p)
        result.append((color, tuple(sorted(cells))))
    # min(cells) is the first cell encountered in normalized row-major reading.
    return sorted(result, key=lambda item: item[1][0])


def _normalize(a):
    candidates = []
    for turn in range(4):
        b = np.rot90(a, turn)
        if (min(b.shape) >= 5 and b[0, 0] == ORIGIN
                and np.all(b[0, 1:] == FAST) and np.all(b[1:, 0] == SLOW)):
            bg = int(b[-1, -1])
            if (bg not in (FAST, SLOW, ORIGIN) and np.all(b[-1, 1:] == bg)
                    and np.all(b[1:, -1] == bg)):
                candidates.append((b.copy(), bg, turn))
    _require(len(candidates) == 1, "one properly rotated L-key is required")
    return candidates[0]


def _parse(value):
    a, bg, turn = _normalize(_grid(value))
    interior = a[1:, 1:]
    boxes = []
    for _, cells in _components(interior, bg, monochrome=False):
        r0 = min(r for r, c in cells) + 1
        r1 = max(r for r, c in cells) + 2
        c0 = min(c for r, c in cells) + 1
        c1 = max(c for r, c in cells) + 2
        _require(len(cells) == (r1-r0)*(c1-c0), "foreground objects must be full rectangles")
        _require(r0 >= 2 and c0 >= 2 and r1 < a.shape[0] and c1 < a.shape[1],
                 "rectangles need a background margin from key and outer edge")
        boxes.append((r0, r1, c0, c1))
    _require(len(boxes) >= 2, "cards and bins are required")
    span = max(c1-c0 for r0, r1, c0, c1 in boxes)
    width = span - 2
    bin_boxes = sorted(b for b in boxes if b[3]-b[2] == span)
    card_boxes = [b for b in boxes if b[3]-b[2] != span]
    _require(1 <= len(bin_boxes) <= 4 and card_boxes and width >= 2, "invalid card/bin counts")
    bins, routes = [], set()
    for r0, r1, c0, c1 in bin_boxes:
        b = a[r0:r1, c0:c1]
        rail, route = int(b[0, 0]), int(b[0, 1])
        _require(np.all(b[:, 0] == rail) and np.all(b[:, -1] == rail)
                 and np.all(b[:, 1:-1] == route), "a bin needs two uniform rails and a uniform interior")
        _require(route not in (bg, FAST, SLOW, ORIGIN) and rail not in (bg, route),
                 "invalid visible bin color roles")
        _require(route not in routes, "each routing color must identify one bin")
        routes.add(route)
        bins.append({"route": route, "rail": rail, "depth": r1-r0, "box": (r0, r1, c0, c1)})
    _require(len({(b[2], b[3]) for b in bin_boxes}) == 1, "bins must share their lateral span")
    _require(all(y[0] >= x[1]+1 for x, y in zip(bin_boxes, bin_boxes[1:])), "bins need separators")
    _require(max(b[1] for b in card_boxes) < bin_boxes[0][0], "all cards must precede all bins")
    lanes = {}
    for box in card_boxes:
        r0, r1, c0, c1 = box
        _require(2 <= r1-r0 <= 4 and 2 <= c1-c0 <= width, "invalid card dimensions")
        _require((width-(c1-c0)) % 2 == 0, "card widths must center exactly")
        _require(set(map(int, a[r0:r1, c0:c1].flat)) <= routes, "every card color needs a bin")
        lanes.setdefault((r0, r1), []).append(box)
    lane_order = sorted(lanes)
    _require(all(y[0] >= x[1]+1 for x, y in zip(lane_order, lane_order[1:])),
             "lane bands must be separated and unstaggered")
    cards, counts = [], []
    for lane in lane_order:
        row = sorted(lanes[lane], key=lambda b: b[2])
        _require(all(y[2] >= x[3]+1 for x, y in zip(row, row[1:])), "cards need lateral separation")
        counts.append(len(row))
        for r0, r1, c0, c1 in row:
            payload = a[r0:r1, c0:c1].copy()
            parts = _components(payload, bg)
            _require(parts == _components(payload, bg, diagonal=True),
                     "diagonal-only same-color grouping is excluded")
            cards.append({"payload": payload, "parts": parts, "box": (r0, r1, c0, c1)})
    _require(max(c["payload"].shape[1] for c in cards) == width,
             "bin interior width must equal the widest card")
    _require(set().union(*(set(map(int, c["payload"].flat)) for c in cards)) == routes,
             "every bin must receive a component")
    return {"canonical": a, "background": bg, "turn": turn, "width": width,
            "cards": cards, "bins": bins, "lane_counts": tuple(counts)}


def _arrivals(cards, width, *, center=True, group=False, reflect=False,
              reverse_parts=False, pixel_scan=False):
    arrivals, positions = [], []
    for card_index, card in enumerate(cards):
        payload = card["payload"]
        parts = card["parts"]
        if reflect:
            parts = _components(payload[::-1], -1)
        if group:
            grouped = {}
            for color, cells in parts:
                grouped.setdefault(color, []).extend(cells)
            parts = sorted(((color, tuple(sorted(cells))) for color, cells in grouped.items()),
                           key=lambda p: p[1][0])
        if reverse_parts:
            parts = parts[::-1]
        pad = (width-payload.shape[1]) // 2 if center else 0
        for color, cells in parts:
            bottom = max(r for r, c in cells)
            # y is height above the bin floor, not an input row coordinate.
            mask = tuple(sorted((bottom-r, pad+c) for r, c in cells))
            arrivals.append((color, mask, card_index))
            positions.append((card["box"][0]+cells[0][0], card["box"][2]+cells[0][1]))
    return [arrivals[i] for i in sorted(range(len(arrivals)), key=lambda i: positions[i])] if pixel_scan else arrivals


def _settle(arrivals, width, *, bounding_boxes=False, independent=False):
    """Rigid vertical drop from above; an earlier mask is a solid obstacle."""
    piles, skylines, events = {}, {}, []
    for color, mask, card_index in arrivals:
        pile = piles.setdefault(color, set())
        top = skylines.setdefault(color, [-1]*width)
        previous_height = max(top)+1
        if independent:
            lift = 0
        elif bounding_boxes:
            lift = previous_height
        else:
            lows = {}
            for y, x in mask:
                lows[x] = min(y, lows.get(x, y))
            lift = max([0]+[top[x]+1-y for x, y in lows.items()])
        target = {(y+lift, x) for y, x in mask}
        if not independent:
            _require(not target & pile, "settling collision")
        pile.update(target)
        for y, x in target:
            top[x] = max(top[x], y)
        events.append((color, lift, previous_height, card_index))
    return piles, events


def _render(piles, bins, width):
    blocks = []
    for b in bins:
        block = np.full((b["depth"], width+2), b["route"], dtype=np.int8)
        block[:, [0, -1]] = b["rail"]
        for y, x in piles.get(b["route"], ()):
            _require(0 <= y < b["depth"] and 0 <= x < width, "pile exceeds its bin")
            block[b["depth"]-1-y, x+1] = b["rail"]
        blocks.append(block)
    return np.concatenate(blocks, axis=0)


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _parse(input_grid)
    piles, _ = _settle(_arrivals(d["cards"], d["width"]), d["width"])
    return _render(piles, d["bins"], d["width"])


def _fingerprint(piles):
    return tuple(sorted((color, tuple(sorted(cells))) for color, cells in piles.items()))


def _features(d):
    cards, width = d["cards"], d["width"]
    arrivals = _arrivals(cards, width)
    piles, events = _settle(arrivals, width)
    actual = _fingerprint(piles)
    def differs(order=cards, **kwargs):
        return actual != _fingerprint(_settle(_arrivals(order, width, **kwargs), width)[0])
    fast_major = sorted(cards, key=lambda c: (c["box"][2], c["box"][0]))
    return {
        "reverse_order": differs(cards[::-1]),
        "fast_major": differs(fast_major),
        "group_by_card_color": differs(group=True),
        "reverse_parts": differs(reverse_parts=True),
        "pixel_scan": differs(pixel_scan=True),
        "left_align": differs(center=False),
        "reflect_rows": differs(reflect=True),
        "bounding_box_stack": actual != _fingerprint(_settle(arrivals, width, bounding_boxes=True)[0]),
        "independent_floor": actual != _fingerprint(_settle(arrivals, width, independent=True)[0]),
        "collision_lifts": sum(lift > 0 for color, lift, old, card in events),
        "interleaving": any(0 < lift < old for color, lift, old, card in events),
        "contact": any(len(set(map(int, c["payload"].flat))) > 1 for c in cards),
        "parts": len(arrivals), "cards": len(cards), "bins": len(d["bins"]),
        "lanes": d["lane_counts"], "widths": tuple(c["payload"].shape[1] for c in cards),
        "heights": tuple(c["payload"].shape[0] for c in cards),
        "depths": tuple(b["depth"] for b in d["bins"]), "turn": d["turn"],
        "background": d["background"], "input_shape": d["canonical"].shape,
        "first_shorter": cards[0]["payload"].shape[1] < width,
        "rail_reuse": any(b["rail"] in {z["route"] for z in d["bins"]} for b in d["bins"]),
        "shared_rails": len({b["rail"] for b in d["bins"]}) < len(d["bins"]),
        "mask_signatures": tuple(tuple((y, x-min(c for r, c in mask)) for y, x in mask)
                                 for color, mask, index in arrivals),
    }


def _card(rng, height, width, routes, kind=None):
    if kind is None:
        kind = int(rng.choice(6, p=[.22, .18, .12, .20, .22, .06]))
    for _ in range(30):
        if len(routes) == 1 or kind == 0:
            a = np.full((height, width), rng.choice(routes), dtype=np.int8)
        elif kind == 1:
            a = np.repeat(rng.choice(routes, size=(1, width)), height, axis=0).astype(np.int8)
        elif kind == 2:
            a = np.repeat(rng.choice(routes, size=(height, 1)), width, axis=1).astype(np.int8)
        elif kind == 3 and height >= 3 and width >= 3:
            outer, inner = rng.choice(routes, size=2, replace=False)
            a = np.full((height, width), outer, dtype=np.int8)
            a[1:-1, 1:-1] = inner
            if width >= 6 and len(routes) >= 3 and rng.random() < .5:
                a[1:-1, width//2] = outer
                a[1:-1, width//2+1:-1] = rng.choice([r for r in routes if r not in (outer, inner)])
        elif kind == 4:
            # Connected regions grown from colored seeds produce varied masks.
            a = np.full((height, width), -1, dtype=np.int8)
            local = list(map(int, rng.choice(routes, size=int(rng.integers(2, len(routes)+1)), replace=False)))
            starts = rng.choice(height*width, size=min(len(local), height*width), replace=False)
            for color, start in zip(local, starts):
                a[int(start)//width, int(start)%width] = color
            while np.any(a == -1):
                frontier = [(r, c) for r in range(height) for c in range(width) if a[r, c] == -1
                            and any(0 <= r+dr < height and 0 <= c+dc < width and a[r+dr, c+dc] != -1
                                    for dr, dc in FOUR)]
                r, c = frontier[int(rng.integers(len(frontier)))]
                choices = [int(a[r+dr, c+dc]) for dr, dc in FOUR
                           if 0 <= r+dr < height and 0 <= c+dc < width and a[r+dr, c+dc] != -1]
                a[r, c] = rng.choice(choices)
        else:
            a = rng.choice(routes, size=(height, width)).astype(np.int8)
        if _components(a, -1) == _components(a, -1, diagonal=True):
            return a
    # A stripe payload always satisfies the declared connectivity restriction.
    return np.repeat(rng.choice(routes, size=(1, width)), height, axis=0).astype(np.int8)


def _spaced(rng, occupied, items, extent):
    spare = extent - occupied - items - 1
    _require(spare >= 0, "packing budget exhausted")
    return (np.ones(items+1, dtype=int) + rng.multinomial(spare, rng.dirichlet(np.ones(items+1)))).tolist()


def _candidate(rng, lesson, turn):
    k = int(rng.choice([2, 2, 3] if lesson else [1, 2, 2, 3, 3, 4]))
    n = int(rng.integers(3, 6) if lesson else rng.integers(2, 9))
    lane_count = int(rng.integers(2, min(3, n-1)+1) if lesson else rng.integers(1, min(4, n)+1))
    width = int(rng.choice([5, 6, 7, 8] if lesson else [2, 3, 4, 5, 6, 7, 8, 10, 12, 14]))
    base = 3 if width % 2 else 2
    lengths = rng.choice(np.arange(base, width+1, 2), size=n).astype(int).tolist()
    longest = int(rng.integers(n))
    lengths[longest] = width
    counts = [1]*lane_count
    for _ in range(n-lane_count):
        counts[int(rng.integers(lane_count))] += 1
    heights = [int(rng.choice([2, 2, 3, 4])) for _ in range(lane_count)]
    witness = int(rng.integers(n)) if lesson else None
    if lesson:
        lengths[witness] = width
        lane = next(j for j in range(lane_count) if witness < sum(counts[:j+1]))
        heights[lane] = 4 if lesson == "independent" else int(rng.choice([3, 4]))
    minimum_width, start = 0, 0
    for count in counts:
        minimum_width = max(minimum_width, sum(lengths[start:start+count])+count+1)
        start += count
    canvas_width = max(minimum_width, width+4)
    if canvas_width > 29:
        return None
    canvas_width += int(rng.integers(min(4, 29-canvas_width)+1))
    background = int(rng.choice([0, 4, 5, 6, 7, 8, 9]))
    route_pool = [c for c in range(10) if c not in (background, FAST, SLOW, ORIGIN)]
    routes = list(map(int, rng.choice(route_pool, size=k, replace=False)))
    rails = []
    for route in routes:
        candidates = [c for c in range(10) if c not in (background, route)]
        rails.append(int(rng.choice(candidates)))
    cards, r0, index = [], 1, 0
    for count, height in zip(counts, heights):
        row_lengths = lengths[index:index+count]
        gaps = _spaced(rng, sum(row_lengths), count, canvas_width)
        c0 = gaps[0]
        for j, length in enumerate(row_lengths):
            payload = _card(rng, height, length, routes)
            if index == witness and lesson == "split":
                a, b = rng.choice(routes, size=2, replace=False)
                payload[:] = a
                payload[1:-1, 1:-1] = b
                if length >= 7 and len(routes) >= 3:
                    payload[1:-1, length//2] = a
                    payload[1:-1, length//2+1:-1] = next(c for c in routes if c not in (a, b))
            elif index == witness and lesson == "independent":
                a, b = rng.choice(routes, size=2, replace=False)
                payload[:] = b
                left_tip = int(rng.integers(length-2))
                payload[0, left_tip] = a
                payload[-1, left_tip:left_tip+3] = a
                payload[-2, left_tip+2:] = a
                if rng.random() < .5:
                    payload = payload[:, ::-1].copy()
            cards.append({"payload": payload, "parts": _components(payload, -1),
                          "box": (r0+1, r0+height+1, c0+1, c0+length+1)})
            c0 += length+gaps[j+1]
            index += 1
        r0 += height + int(rng.integers(1, 3))
    if set().union(*(set(map(int, c["payload"].flat)) for c in cards)) != set(routes):
        return None
    arrivals = _arrivals(cards, width)
    piles, events = _settle(arrivals, width)
    depths = [max(y for y, x in piles[color])+1 for color in routes]
    # interior rows: card field, at least one gap per bin, and a far margin.
    near = r0
    gaps = [int(rng.integers(1, 3)) for _ in routes]
    minimum_height = near + sum(depths) + sum(gaps) + 1
    if minimum_height > 29:
        return None
    spare = 29-minimum_height
    extra = int(rng.integers(min(spare, 6)+1))
    allocation = rng.multinomial(extra, np.ones(k+1)/(k+1))
    depths = [d+int(e) for d, e in zip(depths, allocation[:k])]
    canvas_height = minimum_height + extra
    grid = np.full((canvas_height+1, canvas_width+1), background, dtype=np.int8)
    grid[0, 0], grid[0, 1:], grid[1:, 0] = ORIGIN, FAST, SLOW
    for card in cards:
        a, b, c, d = card["box"]
        grid[a:b, c:d] = card["payload"]
    left = int(rng.integers(2, canvas_width-width-1))
    cursor = near+1
    bins = []
    for i in rng.permutation(k):
        i = int(i)
        cursor += gaps[i]
        b = {"route": routes[i], "rail": rails[i], "depth": depths[i],
             "box": (cursor, cursor+depths[i], left, left+width+2)}
        bins.append(b)
        grid[cursor:cursor+depths[i], left:left+width+2] = routes[i]
        grid[cursor:cursor+depths[i], [left, left+width+1]] = rails[i]
        cursor += depths[i]
    # minimum_height includes the far margin; allocation's last slot adds space.
    if cursor >= grid.shape[0]:
        return None
    pair = {"input": np.rot90(grid, turn).copy(), "output": _render(piles, bins, width)}
    try:
        d = _parse(pair["input"])
        features = _features(d)
    except ValueError:
        return None
    if lesson == "split" and not all(features[name] for name in
            ("reverse_order", "fast_major", "left_align", "reflect_rows", "bounding_box_stack", "independent_floor")):
        return None
    if lesson == "independent" and not all(features[name] for name in ("group_by_card_color", "reverse_parts", "pixel_scan")):
        return None
    return pair


def _reserve_pair(lesson, turn, serial):
    """Finite reserve with 210 distinct palettes per layout and orientation.

    Four cards use the same settling rule. The two teaching layouts
    jointly satisfy all evidence checks, including the minimum (2,1) request.
    This is constructive insurance, not an output-affecting exception.
    """
    serial %= 210
    bg = [0, 4, 5, 6, 7, 8, 9][serial//30]
    pool = [c for c in range(10) if c not in (bg, FAST, SLOW, ORIGIN)]
    a = pool[(serial % 30)//5]
    b = [c for c in pool if c != a][serial % 5]
    if lesson == "independent":
        first = np.full((4, 5), b, dtype=np.int8)
        first[0, 0], first[3, :3], first[2, 2:] = a, a, a
        last = np.full((2, 5), b, dtype=np.int8)
        last[-1, 0] = a
        payloads = [first, np.full((4, 3), a, dtype=np.int8),
                    np.full((2, 3), b, dtype=np.int8), last]
        boxes = [(2, 6, 2, 7), (2, 6, 8, 11), (7, 9, 2, 5), (7, 9, 6, 11)]
    else:
        first = np.full((3, 3), a, dtype=np.int8)
        first[0, 0] = b
        ring = np.full((3, 5), a, dtype=np.int8)
        ring[1, 1:-1] = b
        last = np.full((3, 5), a, dtype=np.int8)
        last[-1, 0] = b
        payloads = [first, ring, np.full((3, 3), a, dtype=np.int8), last]
        boxes = [(2, 5, 2, 5), (2, 5, 6, 11), (6, 9, 2, 5), (6, 9, 6, 11)]
    cards = [{"payload": p, "parts": _components(p, -1), "box": box}
             for p, box in zip(payloads, boxes)]
    piles, _ = _settle(_arrivals(cards, 5), 5)
    cursor, bins = 11, []
    for route, rail in ((a, FAST), (b, SLOW)):
        depth = max(y for y, x in piles[route])+1
        bins.append({"route": route, "rail": rail, "depth": depth,
                     "box": (cursor, cursor+depth, 4, 11)})
        cursor += depth+1
    grid = np.full((cursor+1, 15), bg, dtype=np.int8)
    grid[0, 0], grid[0, 1:], grid[1:, 0] = ORIGIN, FAST, SLOW
    for card in cards:
        r0, r1, c0, c1 = card["box"]
        grid[r0:r1, c0:c1] = card["payload"]
    for z in bins:
        r0, r1, c0, c1 = z["box"]
        grid[r0:r1, c0:c1] = z["route"]
        grid[r0:r1, [c0, c1-1]] = z["rail"]
    return {"input": np.rot90(grid, turn).copy(), "output": _render(piles, bins, 5)}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    first = int(rng.integers(4))
    second = (first + int(rng.choice([1, 3]))) % 4
    training_turns = [first, second]
    test_turns = [r for r in range(4) if r not in training_turns]
    result, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            lesson = ("split" if index == 0 else "independent" if index == 1 else None) if split == "train" else None
            turn = training_turns[index % 2] if split == "train" else test_turns[index % 2]
            for attempt in range(120):
                pair = _candidate(rng, lesson, turn)
                if pair is not None and _key(pair["input"]) not in seen:
                    result[split].append(pair)
                    seen.add(_key(pair["input"]))
                    break
            else:
                start = int(rng.integers(210))
                # There are at most 13 previous inputs, so 15 different reserve
                # candidates guarantee a fresh input without unbounded retries.
                for offset in range(15):
                    pair = _reserve_pair(lesson, turn, start+offset)
                    if _key(pair["input"]) not in seen:
                        result[split].append(pair)
                        seen.add(_key(pair["input"]))
                        break
                else:
                    raise AssertionError("finite reserve uniqueness proof failed")
    rng.shuffle(result["train"])
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal pair counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _parse(inp)
            arrivals = _arrivals(d["cards"], d["width"])
            piles, _ = _settle(arrivals, d["width"])
            for color in piles:
                _require(len(piles[color]) == sum(len(mask) for route, mask, index in arrivals if route == color),
                         "settling must preserve every occupied mask cell without overlap")
            expected = solve(inp)
            _require(np.array_equal(out, expected), "incorrect normalized bin contents")
            _require(out.shape == (sum(z["depth"] for z in d["bins"]), d["width"]+2),
                     "output must concatenate all receiving bins without gaps")
            features[split].append(_features(d))
    for name in ("contact", "reverse_order", "fast_major", "pixel_scan", "group_by_card_color", "reverse_parts", "left_align",
                 "reflect_rows", "bounding_box_stack", "independent_floor"):
        _require(any(f[name] for f in features["train"]), "missing training distinction: "+name)
    train_turns = {f["turn"] for f in features["train"]}
    _require(len(train_turns) == 2, "training must teach two normalization orientations")
    _require(all(f["turn"] not in train_turns for f in features["test"]), "test orientation must be held out")


def self_test():
    """Bounded, generator-owned checks; reports observed structural diversity."""
    records, puzzle_keys = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzle_keys.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _parse(pair["input"])
                f = _features(d)
                f["split"] = split
                records.append(f)
                original = pair["input"].copy()
                assert np.array_equal(solve(original), pair["output"])
                assert np.array_equal(original, pair["input"]), "solve mutated its input"
                for turn in range(1, 4):
                    assert np.array_equal(solve(np.rot90(original, turn)), pair["output"])
    assert len(puzzle_keys) == 42
    assert {f["turn"] for f in records} == set(range(4))
    assert len({f["input_shape"] for f in records}) >= 30
    assert len({f["lanes"] for f in records}) >= 12
    assert len({f["widths"] for f in records}) >= 40
    assert len({f["depths"] for f in records}) >= 35
    assert len({sig for f in records for sig in f["mask_signatures"]}) >= 50
    assert {f["bins"] for f in records} >= {1, 2, 3}
    assert len({f["cards"] for f in records}) >= 5
    assert {h for f in records for h in f["heights"]} == {2, 3, 4}
    assert any(f["rail_reuse"] for f in records) and any(f["shared_rails"] for f in records)
    assert any(f["interleaving"] for f in records)
    assert any(f["first_shorter"] for f in records)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(10000+10*a+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(812, 2, 1), generate_puzzle(812, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt label accepted")
    # An L interlocks with an earlier singleton; rectangles would over-raise it.
    arrivals = [(4, ((0, 0),), 0), (4, ((0, 1), (1, 0), (1, 1)), 1)]
    pile, events = _settle(arrivals, 2)
    assert pile[4] == {(0, 0), (0, 1), (1, 0), (1, 1)} and events[-1][1] == 0
    # A rigid drop stops on the top obstacle even if a lower cavity is empty.
    pile, _ = _settle([(4, ((0, 0), (1, 0), (2, 0)), 0), (4, ((0, 0),), 1)], 2)
    assert (3, 0) in pile[4]
    # Test the reserve directly and force the public generation path to use it.
    for turn in range(4):
        p = {"train": [_reserve_pair("split", turn, 0),
                       _reserve_pair("independent", (turn+1) % 4, 1)],
             "test": [_reserve_pair(None, (turn+2) % 4, 2)]}
        validate_generated_puzzle(p)
    candidate = globals()["_candidate"]
    try:
        globals()["_candidate"] = lambda rng, lesson, turn: None
        p = generate_puzzle(919, 10, 4)
        validate_generated_puzzle(p)
    finally:
        globals()["_candidate"] = candidate
    summary = {"seeds": 42, "pairs": len(records), "input_shapes": len({f["input_shape"] for f in records}),
               "lane_populations": len({f["lanes"] for f in records}),
               "width_sequences": len({f["widths"] for f in records}),
               "depth_sequences": len({f["depths"] for f in records}),
               "mask_signatures": len({s for f in records for s in f["mask_signatures"]}),
               "card_counts": sorted({f["cards"] for f in records}),
               "bin_counts": sorted({f["bins"] for f in records}),
               "orientations": sorted({f["turn"] for f in records}),
               "interleaving_pairs": sum(f["interleaving"] for f in records)}
    return summary


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
