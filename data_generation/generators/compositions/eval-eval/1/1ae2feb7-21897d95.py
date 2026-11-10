"""Periodic color waves on the graph of original connected territories.

Only the capped T geometry, its handle, and the restored input determine an
answer. Construction uses planned IDs; decoding rebuilds IDs from input pixels.
"""
from collections import deque
import json
from math import lcm

import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _need(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _need(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _add(point, direction, k=1):
    return point[0] + k * direction[0], point[1] + k * direction[1]


def _inside(point, shape):
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


def _components(grid):
    ids = np.full(grid.shape, -1, dtype=np.int16)
    colors, areas = [], []
    for point in np.ndindex(grid.shape):
        if ids[point] >= 0:
            continue
        label, color = len(colors), int(grid[point])
        colors.append(color)
        ids[point] = label
        stack, size = [point], 0
        while stack:
            p = stack.pop()
            size += 1
            for d in DIRECTIONS:
                q = _add(p, d)
                if _inside(q, grid.shape) and ids[q] < 0 and grid[q] == color:
                    ids[q] = label
                    stack.append(q)
        areas.append(size)
    return ids, np.array(colors, dtype=np.int8), areas


def _adjacency(ids):
    graph = [set() for _ in range(int(ids.max()) + 1)]
    for left, right in ((ids[:-1], ids[1:]), (ids[:, :-1], ids[:, 1:])):
        mask = left != right
        for a, b in zip(left[mask], right[mask]):
            graph[int(a)].add(int(b))
            graph[int(b)].add(int(a))
    return graph


def _distances(graph, root, host):
    dist = [-1] * len(graph)
    dist[root] = 0
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for other in sorted(graph[node]):
            if other != host and dist[other] < 0:
                dist[other] = dist[node] + 1
                queue.append(other)
    return dist


def _winner(runs, distance, kind="true"):
    if distance < 0:
        return None
    if kind == "flood":
        return runs[0][1]
    hits = [i for i, (period, _) in enumerate(runs)
            if (distance + (kind == "phase")) % (period + (kind == "period")) == 0]
    if not hits:
        return None
    if kind == "short":
        index = min(hits, key=lambda i: (runs[i][0], i))
    elif kind == "long":
        index = min(hits, key=lambda i: (-runs[i][0], i))
    elif kind == "far":
        index = hits[-1]
    else:
        index = hits[0]
    return runs[index][1]


def _paint(ids, colors, distances, runs, kind="true"):
    final = colors.copy()
    for label, distance in enumerate(distances):
        payload = _winner(runs, distance, kind)
        if payload is not None:
            final[label] = payload
    return final[ids], final


def _decode(value):
    """Enumerate unique visible emitter interpretations; never consult examples."""
    grid = _grid(value)
    answers = []
    for ink in map(int, np.unique(grid)):
        ink_cells = set(map(tuple, np.argwhere(grid == ink)))
        if len(ink_cells) != 5:
            continue
        for center in sorted(ink_cells):
            for direction in DIRECTIONS:
                perp = (-direction[1], direction[0])
                tip = _add(center, direction)
                arms = {_add(center, perp), _add(center, perp, -1)}
                stencil = {center, tip, *arms}
                if not stencil <= ink_cells:
                    continue
                cap = next(iter(ink_cells - stencil))
                delta = (cap[0] - center[0], cap[1] - center[1])
                length = -(delta[0] * direction[0] + delta[1] * direction[1])
                if length < 2 or cap != _add(center, direction, -length):
                    continue
                rootcell = _add(center, direction, 2)
                handle = [_add(center, direction, -k) for k in range(1, length)]
                support = {_add(tip, perp), _add(tip, perp, -1)}
                for p in [cap, *handle]:
                    support.update((_add(p, perp), _add(p, perp, -1)))
                if not all(_inside(p, grid.shape) for p in support | {rootcell}):
                    continue
                fills = {int(grid[p]) for p in support}
                if len(fills) != 1:
                    continue
                host_color = fills.pop()
                if host_color == ink or grid[rootcell] in (ink, host_color):
                    continue
                runs, cursor = [], 0
                while cursor < len(handle):
                    color = int(grid[handle[cursor]])
                    if color == host_color:
                        cursor += 1
                        continue
                    if color == ink:
                        break
                    end = cursor + 1
                    while end < len(handle) and grid[handle[end]] == color:
                        end += 1
                    runs.append((end - cursor, color))
                    cursor = end
                if not runs or cursor != len(handle):
                    continue
                annotation = ink_cells | set(handle)
                clean = grid.copy()
                for p in annotation:
                    clean[p] = host_color
                if ink in clean:
                    continue
                ids, colors, areas = _components(clean)
                host, root = int(ids[center]), int(ids[rootcell])
                if host == root or any(int(ids[p]) != host for p in annotation | support):
                    continue
                graph = _adjacency(ids)
                distances = _distances(graph, root, host)
                answers.append(dict(clean=clean, ids=ids, colors=colors, areas=areas,
                                    host=host, root=root, graph=graph, distances=distances,
                                    runs=runs, annotation=annotation, ink=ink,
                                    center=center, direction=direction, cap=cap,
                                    handle=handle, host_color=host_color, rootcell=rootcell))
    _need(len(answers) == 1, "input must have one unambiguous capped T emitter")
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only prediction for any uniquely marked member of this family."""
    info = _decode(input_grid)
    output, _ = _paint(info["ids"], info["colors"], info["distances"], info["runs"])
    return output.astype(np.int8, copy=True)


def _rival(info, kind):
    if kind == "ray":
        # Transfer full original components, but only those intersected by the
        # straight arrow ray, counting successive encounters instead of BFS.
        distances = [-1] * len(info["colors"])
        p, previous, ordinal = info["rootcell"], None, -1
        while _inside(p, info["ids"].shape):
            label = int(info["ids"][p])
            if label == info["host"]:
                break
            if label != previous:
                ordinal += 1
                if distances[label] < 0:
                    distances[label] = ordinal
            previous = label
            p = _add(p, info["direction"])
        return _paint(info["ids"], info["colors"], distances, info["runs"])[0]
    if kind == "pixel":
        # A spatial periodic rule with whole-component output: use first pixel
        # encounter distance along the ray instead of territory-hop distance.
        distances = [-1] * len(info["colors"])
        p, offset = info["rootcell"], 0
        while _inside(p, info["ids"].shape):
            label = int(info["ids"][p])
            if label == info["host"]:
                break
            if distances[label] < 0:
                distances[label] = offset
            p, offset = _add(p, info["direction"]), offset + 1
        return _paint(info["ids"], info["colors"], distances, info["runs"])[0]
    return _paint(info["ids"], info["colors"], info["distances"], info["runs"], kind)[0]


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _features(info):
    _, final = _paint(info["ids"], info["colors"], info["distances"], info["runs"])
    colors, distances, graph = info["colors"], info["distances"], info["graph"]
    reached = [i for i, d in enumerate(distances) if d >= 0]
    locality = any(colors[a] == colors[b] and final[a] != final[b]
                   for a in range(len(colors)) for b in range(a))
    nonrectangular = 0
    for label, area in enumerate(info["areas"]):
        rows, cols = np.nonzero(info["ids"] == label)
        nonrectangular += area != (int(rows.max()-rows.min())+1) * (int(cols.max()-cols.min())+1)
    return dict(shape=info["ids"].shape, direction=DIRECTIONS.index(info["direction"]),
                territories=len(colors), runs=len(info["runs"]),
                periods=tuple(p for p, _ in info["runs"]), radius=max(distances),
                branch=any(sum(n in reached for n in graph[i]) >= 3 for i in reached),
                cycle=sum(sum(n in reached for n in graph[i]) for i in reached)//2 >= len(reached),
                unreachable=sum(d < 0 for i, d in enumerate(distances) if i != info["host"]),
                locality=locality, nonrectangular=nonrectangular,
                repeated=len(set(map(int, colors))) < len(colors), ink=info["ink"],
                payloads=tuple(c for _, c in info["runs"]))


def _evidence(pairs, infos):
    short, long, edges, recurrence = False, False, set(), set()
    for pair, info in zip(pairs, infos):
        output = np.asarray(pair["output"])
        runs = info["runs"]
        observed = {}
        for label, d in enumerate(info["distances"]):
            if d >= 0:
                values = np.unique(output[info["ids"] == label])
                _need(len(values) == 1, "whole original territory must have one output fill")
                observed[d] = int(values[0])
        for i, (p, c) in enumerate(runs):
            visible = [d for d, value in observed.items()
                       if d % p == 0 and not any(d % r[0] == 0 for r in runs[:i]) and value == c]
            if any(b-a == p for a in visible for b in visible if b > a):
                recurrence.add(p)
            # Occlusion can hide alternating repetitions. Two separated visible
            # pulses still establish recurrence at multiples of the encoded L.
            if len(visible) >= 2 and all(d % p == 0 for d in visible):
                recurrence.add(p)
            for q, other in runs[i+1:]:
                survives = any(d % q == 0 and _winner(runs, d) == other and value == other
                               for d, value in observed.items())
                later_collision = any(d > 0 and d % p == d % q == 0 and value == c
                                      and _winner(runs, d) == c for d, value in observed.items())
                if c != other and survives and later_collision:
                    short |= p < q
                    long |= p > q
                    edges.add((c, other))
    return dict(short=short, long=long, color_cycle=any((b,a) in edges for a,b in edges),
                recurrence=recurrence)


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal example counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape, "output geometry must be preserved")
            _need(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _decode(inp)
            expected, final = _paint(info["ids"], info["colors"], info["distances"], info["runs"])
            _need(np.array_equal(expected, out), "wrong wave phase, period, distance, locality or preservation")
            _need(not np.array_equal(inp, out), "identity pair")
            _need(all(out[p] == info["host_color"] for p in info["annotation"]), "emitter not erased")
            _need(final[info["host"]] == info["host_color"], "host must be excluded")
            # BFS certificates, checked on the reconstructed graph.
            for node, distance in enumerate(info["distances"]):
                if distance > 0:
                    _need(any(info["distances"][n] == distance-1 for n in info["graph"][node]),
                          "distance lacks a predecessor")
                if node != info["host"]:
                    for neighbor in info["graph"][node] - {info["host"]}:
                        other = info["distances"][neighbor]
                        _need((distance < 0) == (other < 0), "inconsistent reachability")
                        if distance >= 0:
                            _need(abs(distance-other) <= 1, "not a shortest-distance labeling")
            infos[split].append(info)
    evidence = _evidence(puzzle["train"], infos["train"])
    _need(evidence["short"] and evidence["long"] and evidence["color_cycle"], "priority demonstrations missing")
    _need(len(evidence["recurrence"]) >= 2, "two visible period recurrences required")
    for rival in ("flood", "short", "long", "far", "phase", "period", "ray", "pixel"):
        _need(any(not np.array_equal(_rival(i, rival), p["output"])
                  for p, i in zip(puzzle["train"], infos["train"])), "training fits rival: " + rival)
    train_features = [_features(i) for i in infos["train"]]
    _need(any(f["locality"] for f in train_features), "component locality is not visible")
    _need(any(f["branch"] for f in train_features), "branching not demonstrated")
    _need(any(f["nonrectangular"] for f in train_features), "stepped geometry not demonstrated")
    _need(all(max(p for p,_ in i["runs"]) <= max(evidence["recurrence"]) for i in infos["test"]),
          "test period exceeds demonstrated numerical range")
    if any(_features(i)["unreachable"] for i in infos["test"]):
        _need(any(f["unreachable"] for f in train_features), "unreachable pockets not taught")
    # A controlled geometric shift must be visible directly in returned grids.
    directions = {f["direction"] for f in train_features}
    witness_radii = [max(i["distances"]) for i in infos["train"] if len(i["runs"]) == 2
                     and lcm(*(p for p,_ in i["runs"])) <= max(i["distances"])]
    test_features = [_features(i) for i in infos["test"]]
    _need(any(f["direction"] not in directions
              or f["radius"] < min(witness_radii)
              or f["territories"] > max(t["territories"] for t in train_features)
              for f in test_features), "no visible train/test geometry shift")


def _choose_palette(rng, graph, ink, host_color, payloads, avoid_payloads):
    available = [c for c in range(10) if c not in {ink, host_color}
                 and (not avoid_payloads or c not in payloads)]
    colors = np.full(len(graph), -1, dtype=np.int8)
    colors[0] = host_color
    for label in range(1, len(graph)):
        valid = [c for c in available if c not in {int(colors[n]) for n in graph[label]}]
        if not valid:
            return None
        # Repeated fills are intentional: component identity cannot be a color.
        colors[label] = int(rng.choice(valid))
    return colors


def _layout(rng, h, w, source_width, bands, center_row, split_mode, stepped, pocket):
    target_width = w - source_width
    widths = np.ones(bands, dtype=int)
    if stepped:
        widths[0] += 1
    if int(widths.sum()) > target_width:
        return None
    for _ in range(target_width - int(widths.sum())):
        widths[int(rng.integers(bands))] += 1
    # Shared row shifts retain layer order and connectedness, while making
    # boundaries, source silhouette, and region areas genuinely variable.
    shift = np.zeros(h, dtype=int)
    if stepped:
        value = 0
        for row in range(h):
            value = int(np.clip(value + rng.integers(-1, 2), -1, 1))
            shift[row] = value
        shift[center_row-1:center_row+2] = 0
        # Ensure the mandatory stepped witness is visibly nonrectangular.
        shift[0] = 1
        if center_row > 2:
            shift[1] = 1
        for row in range(1, h):
            shift[row] = int(np.clip(shift[row], shift[row-1]-1, shift[row-1]+1))
    labels = np.zeros((h, w), dtype=np.int16)
    next_id = 1
    offset = source_width
    for index, width in enumerate(widths):
        if split_mode == "chain":
            pieces = 1
        elif split_mode == "dense":
            pieces = 7
        else:
            pieces = int(rng.integers(1, 4))
            if index == bands // 2:
                pieces = 3
        cuts = [0, *sorted(map(int, rng.choice(range(1, h), pieces-1, replace=False))), h]
        for top, bottom in zip(cuts, cuts[1:]):
            for row in range(top, bottom):
                # A one-cell strip must remain four-connected: shifting its
                # successive rows would make only diagonal contact. Only move
                # boundaries whose two adjoining strips have width >= 2.
                left_moves = index == 0 or (widths[index-1] >= 2 and width >= 2)
                right_moves = index < bands-1 and width >= 2 and widths[index+1] >= 2
                left = offset + (int(shift[row]) if left_moves else 0)
                right = w if index == bands-1 else offset + int(width) + (int(shift[row]) if right_moves else 0)
                labels[row, left:right] = next_id
            next_id += 1
        offset += int(width)
    if pocket:
        # A little disconnected territory surrounded entirely by source host.
        rows = [r for r in range(1, h-1) if abs(r-center_row) >= 3]
        if not rows:
            return None
        row = int(rng.choice(rows))
        col = int(rng.integers(1, max(2, source_width-3)))
        labels[row:row+1, col:col+2] = next_id
        next_id += 1
    # Optional enclosed target island adds a graph leaf rather than a new rule.
    if split_mode != "chain" and rng.random() < .45:
        choices = []
        for row in range(1, h-1):
            for col in range(source_width+1, w-1):
                patch = labels[row-1:row+2, col-1:col+2]
                if np.all(patch == labels[row, col]):
                    choices.append((row, col))
        if choices:
            point = choices[int(rng.integers(len(choices)))]
            labels[point] = next_id
    return labels


def _build_pair(rng, runs, base_payloads, rotation, geometry, pocket, mandatory=False,
                dense_floor=0, stable_palette=None):
    # Runs are stored nearest-first; touching equal colors must remain one run.
    if mandatory:
        gaps = [0] * len(runs)
        gaps[int(rng.integers(len(runs)))] = int(rng.integers(2))
        margin = int(rng.integers(2))
    else:
        gaps = [int(rng.integers(2)) for _ in runs]
        for i in range(1, len(runs)):
            gaps[i] = max(gaps[i], int(runs[i][1] == runs[i-1][1]))
        margin = int(rng.integers(3))
    span = sum(p for p,_ in runs) + sum(gaps)
    source_width = span + 3 + margin
    if geometry in ("witness_chain", "witness_branch"):
        bands = 3 * max(p for p,_ in runs) + 1
    elif geometry == "clip":
        bands = int(rng.integers(2, 5))
    elif geometry == "dense":
        bands = max(4, (dense_floor + 7) // 7)
    else:
        bands = int(rng.integers(4, 9))
    stepped = geometry != "witness_chain" and (mandatory or rng.random() < .8)
    min_target = bands + int(stepped)
    # At least one unequal width in the straight witness refutes pixel periods.
    if geometry == "witness_chain":
        min_target += 2
    if source_width + min_target > 30:
        return None
    h = int(rng.integers(14, 25) if geometry == "dense" else rng.integers(8, 19))
    w = int(rng.integers(source_width + min_target, 31))
    center_row = int(rng.integers(3, h-2))
    mode = "chain" if geometry == "witness_chain" else ("dense" if geometry == "dense" else "branch")
    labels = _layout(rng, h, w, source_width, bands, center_row, mode, stepped, pocket)
    if labels is None:
        return None
    payloads = {c for _,c in runs} | set(base_payloads)
    if stable_palette is None:
        available = [c for c in range(10) if c not in payloads]
        ink, host_color = map(int, rng.choice(available, 2, replace=False))
    else:
        ink, host_color = stable_palette
        if ink in payloads or host_color in payloads:
            return None
    graph = _adjacency(labels)
    colors = _choose_palette(rng, graph, ink, host_color, payloads, mandatory)
    if colors is None:
        return None
    center = (center_row, source_width-2)
    rootcell = (center_row, source_width)
    root = int(labels[rootcell])
    distances = _distances(graph, root, 0)
    clean = colors[labels]
    inp = clean.copy()
    # Install handle back from T. All intervening gaps show original host fill.
    cursor = center[1]-1
    for (period, color), gap in zip(runs, gaps):
        cursor -= gap
        inp[center_row, cursor-period+1:cursor+1] = color
        cursor -= period
    cap = (center_row, cursor)
    for p in (center, (center_row-1, center[1]), (center_row+1, center[1]),
              (center_row, center[1]+1), cap):
        inp[p] = ink
    out, _ = _paint(labels, colors, distances, runs)
    pair = {"input": np.rot90(inp, rotation).copy(), "output": np.rot90(out, rotation).copy()}
    try:
        info = _decode(pair["input"])
    except AssertionError:
        return None
    if not np.array_equal(info["clean"], np.rot90(clean, rotation)) or info["runs"] != runs:
        return None
    if len(info["colors"]) != len(colors):
        return None  # Every planned ID must really be one original component.
    if not np.array_equal(solve(pair["input"]), pair["output"]):
        return None
    feature = _features(info)
    if mandatory:
        if feature["radius"] < 3 * max(p for p,_ in runs):
            return None
        if geometry == "witness_branch" and (not feature["branch"] or not feature["locality"]
                                             or not feature["nonrectangular"]
                                             or np.array_equal(_rival(info, "ray"), pair["output"])):
            return None
        if geometry == "witness_chain" and np.array_equal(_rival(info, "pixel"), pair["output"]):
            return None
    if geometry == "dense" and feature["territories"] <= dense_floor:
        return None
    return pair, info


def _random_runs(rng, maximum, payloads):
    count = int(rng.choice([1, 2, 3], p=[.35, .4, .25]))
    for _ in range(20):
        runs = [(int(rng.integers(1, maximum+1)), int(rng.choice(payloads))) for _ in range(count)]
        if sum(p for p,_ in runs) <= 11:
            return runs
    return [(maximum, int(rng.choice(payloads)))]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for value, low, high, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        if not _integer(value) or not low <= value <= high:
            raise ValueError(name + " is outside its legal integer range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    # Sampling choices only. Solve has no access to any of these values.
    small, large = 2, int(rng.choice([3, 5], p=[.8, .2]))
    payloads = list(map(int, rng.choice(10, 2, replace=False)))
    rotation = int(rng.integers(4))
    shift = str(rng.choice(["reverse", "dense", "clip"], p=[.4, .25, .35]))
    pocket = bool(rng.random() < .6)
    stable_palette = None
    if rng.random() < .35:
        stable_palette = tuple(map(int, rng.choice([c for c in range(10) if c not in payloads], 2, replace=False)))
    slots = list(map(int, rng.permutation(num_train)))
    requests = {slots[0]: ([(small, payloads[0]), (large, payloads[1])], "witness_chain"),
                slots[1]: ([(large, payloads[1]), (small, payloads[0])], "witness_branch")}
    puzzle, seen, train_infos = {"train": [], "test": []}, set(), []
    for index in range(num_train):
        mandatory = index in requests
        for _ in range(160):
            if mandatory:
                runs, geometry = requests[index]
            else:
                runs, geometry = _random_runs(rng, large, payloads), "ordinary"
            result = _build_pair(rng, runs, payloads, rotation, geometry,
                                 pocket and geometry != "witness_chain", mandatory,
                                 stable_palette=stable_palette)
            if result is None or _key(result[0]["input"]) in seen:
                continue
            pair, info = result
            puzzle["train"].append(pair)
            train_infos.append(info)
            seen.add(_key(pair["input"]))
            break
        else:
            raise RuntimeError("bounded training construction exhausted")
    dense_floor = max(len(i["colors"]) for i in train_infos)
    # Avoid optional pockets unless the actual training set demonstrates them.
    taught_pocket = any(_features(i)["unreachable"] for i in train_infos)
    witness_radius = min(max(i["distances"]) for i in train_infos
                         if len(i["runs"]) == 2 and lcm(*(p for p,_ in i["runs"])) <= max(i["distances"]))
    for index in range(num_test):
        for _ in range(160):
            runs = _random_runs(rng, large, payloads)
            geometry = shift if shift in ("dense", "clip") else "ordinary"
            test_rotation = (rotation+2) % 4 if shift == "reverse" else rotation
            result = _build_pair(rng, runs, payloads, test_rotation, geometry,
                                 taught_pocket and rng.random() < .7,
                                 dense_floor=dense_floor, stable_palette=stable_palette)
            if result is None or _key(result[0]["input"]) in seen:
                continue
            pair, info = result
            if shift == "clip" and max(info["distances"]) >= witness_radius:
                continue
            puzzle["test"].append(pair)
            seen.add(_key(pair["input"]))
            break
        else:
            raise RuntimeError("bounded test construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _fixed_cases():
    """Hand-calculated fixtures: answers are literal fills, never BFS results."""
    cases = []
    host = np.ones((9, 8), dtype=np.int8)
    host[1, 2:4] = 4  # Unreachable, equal to two reachable territory fills.
    chain = np.concatenate([host, *[np.full((9, 2), c, dtype=np.int8)
                                   for c in (4, 2, 6, 4, 7, 6, 5)]], axis=1)
    answer = np.concatenate([host, *[np.full((9, 2), c, dtype=np.int8)
                                    for c in (2, 2, 2, 3, 2, 6, 2)]], axis=1)
    branch = np.concatenate([host, np.full((9, 2), 4, dtype=np.int8),
                             np.vstack([np.full((4, 2), 5), np.full((5, 2), 6)]),
                             np.vstack([np.full((4, 2), 7), np.full((5, 2), 8)]),
                             np.full((9, 2), 4, dtype=np.int8)], axis=1).astype(np.int8)
    branch_answer = np.concatenate([host, np.full((9, 2), 2, dtype=np.int8),
                                    np.vstack([np.full((4, 2), 5), np.full((5, 2), 6)]),
                                    np.full((9, 2), 2, dtype=np.int8),
                                    np.full((9, 2), 3, dtype=np.int8)], axis=1).astype(np.int8)
    for clean, expected in ((chain, answer), (branch, branch_answer)):
        inp = clean.copy()
        inp[4, 1:4] = 3
        inp[4, 4:6] = 2
        for point in ((4,0), (4,6), (3,6), (5,6), (4,7)):
            inp[point] = 9
        cases.append((inp, expected))
    return cases


def self_test():
    for inp, expected in _fixed_cases():
        for turn in range(4):
            _need(np.array_equal(solve(np.rot90(inp, turn)), np.rot90(expected, turn)),
                  "hand-calculated graph/period regression failed")
        permutation = np.array([8, 4, 9, 0, 6, 2, 5, 3, 1, 7], dtype=np.int8)
        _need(np.array_equal(solve(permutation[inp]), permutation[expected]), "literal palette assumption")
    features, keys, shapes, normalized, directions = [], set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        keys.add(_puzzle_key(puzzle))
        if seed < 3:
            _need(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), "nondeterminism")
        local = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                feature = _features(info)
                features.append(feature)
                local.append(feature)
                directions.add(feature["direction"])
                shapes.add(pair["input"].shape)
                mapping = {}
                normalized.add((pair["input"].shape,
                                tuple(mapping.setdefault(int(c), len(mapping)) for c in pair["input"].flat)))
                _need(np.array_equal(solve(pair["input"].copy()), pair["output"]), "input-only mismatch")
        _need(len({f["territories"] for f in local}) >= 2, "region counts do not vary within puzzle")
        _need(len({f["shape"] for f in local}) >= 2, "geometry does not vary within puzzle")
    _need(len(keys) == 42, "collapsed seeds")
    _need(len(shapes) >= 40 and len(normalized) >= 190, "collapsed structural variation")
    _need(directions == set(range(4)), "direction coverage missing")
    _need({f["runs"] for f in features} == {1,2,3}, "run-count coverage missing")
    _need({p for f in features for p in f["periods"]} >= {1,2,3,4,5}, "period range collapsed")
    for key in ("branch", "cycle", "locality", "nonrectangular", "unreachable"):
        _need(sum(bool(f[key]) for f in features) >= 10, "missing family variation: " + key)
    _need({f["ink"] for f in features} == set(range(10)), "ink roles collapsed")
    _need(any(len(set(f["payloads"])) < len(f["payloads"]) for f in features), "no repeated payload runs")
    # All legal explicit counts and representative invalid types/boundaries.
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            p = generate_puzzle(51000 + 10*a+b, a, b)
            _need(len(p["train"]) == a and len(p["test"]) == b, "ignored requested counts")
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(4.0,1),(4,1.0),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzle = generate_puzzle(0)
    damaged = {s: [{k:v.copy() for k,v in p.items()} for p in puzzle[s]] for s in puzzle}
    damaged["test"][0]["output"][0,0] = (int(damaged["test"][0]["output"][0,0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds_checked":42, "distinct_puzzles":len(keys), "shapes":len(shapes),
            "normalized_inputs":len(normalized), "pair_features":len(features),
            "runs":sorted({f["runs"] for f in features}),
            "territory_count_range":[min(f["territories"] for f in features), max(f["territories"] for f in features)],
            "feature_counts":{key:sum(bool(f[key]) for f in features)
                              for key in ("branch","cycle","locality","nonrectangular","unreachable")}}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
