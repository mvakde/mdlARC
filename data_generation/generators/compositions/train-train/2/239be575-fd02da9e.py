"""Consensus circuits: corner-selected conductive sockets and neutral wires.

Only standard library and NumPy are used.  The solver reads no generation state.
"""
from __future__ import annotations

import heapq
import json
from pathlib import Path

import numpy as np


D8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
D4 = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _corner(k, h, w):
    return (0 if k < 2 else h - 1, 0 if k % 2 == 0 else w - 1)


def _socket(k, h, w, mode="true"):
    if mode == "corner":
        return {_corner(k, h, w)}
    if k < 2 or mode == "squares":
        depth = ((1, 1), (1, 2), (2, 1), (2, 2))
    else:
        depth = ((3, 2), (2, 2), (1, 3))
    return {(d if k < 2 else h - 1 - d,
             x if k % 2 == 0 else w - 1 - x) for d, x in depth}


def _neighbors(p, shape, diagonal=True):
    for dr, dc in D8 if diagonal else D4:
        q = p[0] + dr, p[1] + dc
        if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]:
            yield q


def _halo(cells, shape, diagonal=True):
    return {q for p in cells for q in _neighbors(p, shape, diagonal)} - set(cells)


def _components(cells, shape, diagonal=True):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        comp, stack = {start}, [start]
        while stack:
            for q in _neighbors(stack.pop(), shape, diagonal):
                if q in remaining:
                    remaining.remove(q)
                    comp.add(q)
                    stack.append(q)
        result.append(comp)
    return result


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=False)


def _read(value):
    a = _grid(value)
    h, w = a.shape
    if min(h, w) < 9:
        raise ValueError("consensus-circuit canvases have both dimensions >=9")
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if int(max(counts)) * 2 <= a.size:
        raise ValueError("background must occupy a strict majority")
    corners = {_corner(k, h, w) for k in range(4)}
    markers = {k: int(a[_corner(k, h, w)]) for k in range(4)
               if int(a[_corner(k, h, w)]) != bg}
    if not 2 <= len(markers) <= 4:
        raise ValueError("expected two to four corner markers")
    interior = {(int(r), int(c)) for r, c in np.argwhere(a != bg)} - corners
    wire_colors = {int(a[p]) for p in interior}
    if len(wire_colors) != 1:
        raise ValueError("all non-corner foreground must be one wire color")
    wire = next(iter(wire_colors))
    if wire in markers.values():
        raise ValueError("wire and payload colors must differ")
    sockets = {k: _socket(k, h, w) for k in markers}
    if any(a[p] != bg for cells in sockets.values() for p in cells):
        raise ValueError("socket footprints must be empty in the input")
    return a, bg, wire, markers, interior, sockets


def _networks(wires, sockets, shape, links=True, contacts=True, relay=True):
    """Bipartite graph of wire components and whole socket objects.

    Keeping each socket as one node is deliberate: it can join several otherwise
    disconnected wire fragments, including the diagonal tip of a lower socket.
    """
    comps = _components(wires, shape, links)
    halos = {k: _halo(s, shape, contacts) for k, s in sockets.items()}
    touches = [{k for k in sockets if comp & halos[k]} for comp in comps]
    if not relay:
        return [(comp, ports) for comp, ports in zip(comps, touches)]
    ids = sorted(sockets)
    port_node = {k: len(comps) + i for i, k in enumerate(ids)}
    parent = list(range(len(comps) + len(ids)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, ports in enumerate(touches):
        for k in sorted(ports):
            parent[root(i)] = root(port_node[k])
    nets = {}
    for i, comp in enumerate(comps):
        cells, ports = nets.setdefault(root(i), (set(), set()))
        cells.update(comp)
    for k, node in port_node.items():
        cells, ports = nets.setdefault(root(node), (set(), set()))
        ports.add(k)
    return list(nets.values())


def _render(value, *, links=True, contacts=True, relay=True,
            socket_mode="true", policy="consensus"):
    a, bg, wire, markers, wires, true_sockets = _read(value)
    sockets = {k: _socket(k, *a.shape, socket_mode) for k in markers}
    nets = _networks(wires, sockets, a.shape, links, contacts, relay)
    out = np.full(a.shape, bg, dtype=np.int8)
    for cells, ports in nets:
        payloads = {markers[k] for k in ports}
        active = len(ports) >= 2 and len(payloads) == 1
        if policy == "any_pair":
            active = any(sum(markers[k] == color for k in ports) >= 2
                         for color in payloads)
        elif policy == "any_bridge":
            active = len(ports) >= 2
        elif policy == "one_socket":
            active = len(ports) >= 1 and len(payloads) == 1
        if active:
            # Alternative policies only serve as documented ablation checks.
            color = min(payloads)
            for p in cells:
                out[p] = color
    for k, cells in true_sockets.items():
        for p in cells:
            out[p] = markers[k]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Unfold sockets, join wire fragments through them, and retain consensus nets."""
    return _render(input_grid)


def _key(a):
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for n, lo, hi, name in ((num_train, 2, 10, "num_train"),
                            (num_test, 1, 4, "num_test")):
        if (isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
                or not lo <= n <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _path(rng, free, starts, goals, shape):
    starts, goals = sorted(set(starts) & free), set(goals) & free
    if not starts or not goals:
        return None
    weights = {p: float(rng.uniform(.65, 3.4)) for p in sorted(free)}
    dist, prev, heap = {}, {}, []
    for p in starts:
        dist[p], prev[p] = weights[p], None
        heapq.heappush(heap, (dist[p], p))
    goal = None
    while heap:
        cost, p = heapq.heappop(heap)
        if cost != dist[p]:
            continue
        if p in goals:
            goal = p
            break
        for q in _neighbors(p, shape):
            if q not in free:
                continue
            nc = cost + weights[q]
            if nc < dist.get(q, float("inf")):
                dist[q], prev[q] = nc, p
                heapq.heappush(heap, (nc, q))
    if goal is None:
        return None
    route = []
    while goal is not None:
        route.append(goal)
        goal = prev[goal]
    return set(route)


def _plan(rng, index, training, train_count, payloads):
    a, b, c = payloads[:3]
    if training and index == 0:
        low = int(rng.choice([2, 3]))
        high = int(rng.choice([0, 1]))
        group = [high, low]
        groups = [group] + [[k] for k in range(4) if k not in group]
        return groups, {k: a for k in range(4)}, True
    if training and index == 1:
        groups = ([[0, 2], [1, 3]] if rng.integers(2) else [[0, 1], [2, 3]])
        if rng.integers(2):
            groups.reverse()
        markers = {k: a for k in groups[0]}
        markers.update({groups[1][0]: b, groups[1][1]: c})
        return groups, markers, False
    if training and index in (2, 3):
        corners = sorted(int(k) for k in rng.choice(4, 3, replace=False))
        markers = {k: a for k in corners}
        if index == 2:
            markers[int(rng.choice(corners))] = b
        return [corners], markers, False
    n = int(rng.integers(2, 5))
    corners = sorted(int(k) for k in rng.choice(4, n, replace=False))
    mode = int(rng.integers(4))
    if not training and index == 0 and train_count >= 3 and rng.random() < .65:
        n = 3 if train_count == 3 else int(rng.integers(2, 5))
        corners = sorted(int(k) for k in rng.choice(4, n, replace=False))
        mode = 0
    if mode == 0 and (training or train_count >= 3):
        groups = [corners]
    elif mode <= 2:
        order = [int(k) for k in rng.permutation(corners)]
        groups = [order[:2]] + [[k] for k in order[2:]]
        if len(order) == 4 and mode == 2:
            # Only noncrossing pairings: crossing paths would merge electrically.
            groups = ([[0, 2], [1, 3]] if rng.integers(2) else [[0, 1], [2, 3]])
    else:
        groups = [[k] for k in corners]
    markers = {}
    for group in groups:
        color = int(rng.choice(payloads))
        markers.update({k: color for k in group})
        conflict = len(group) >= 2 and rng.random() < .45
        if not training and train_count == 3 and len(group) > 2:
            conflict = True
        if conflict:
            markers[int(rng.choice(group))] = int(rng.choice([x for x in payloads if x != color]))
    return groups, markers, False


def _build(rng, shape, groups, markers, palette, witness):
    h, w = shape
    bg, wire = palette[:2]
    sockets = {k: _socket(k, h, w) for k in markers}
    all_masks = set().union(*(_socket(k, h, w) for k in range(4)))
    halos = {k: _halo(s, shape) for k, s in sockets.items()}
    raw_halos = set().union(*({ _corner(k, h, w)} |
                            _halo({_corner(k, h, w)}, shape) for k in range(4)))
    free = {(r, c) for r in range(h) for c in range(w)} - all_masks - raw_halos
    owner = {}
    for gi, group in enumerate(groups):
        forbidden = set().union(*(halos[k] for k in markers if k not in group)) if len(group) < len(markers) else set()
        previous = {p for p, g in owner.items() if g != gi}
        forbidden |= previous | _halo(previous, shape)
        allowed = free - forbidden
        endpoints = {k: halos[k] & allowed for k in group}
        if witness and gi == 0:
            high, low = group
            hp = (3, 3 if high == 0 else w - 4)
            lp = (h - 1, 4 if low == 2 else w - 5)
            endpoints = {high: {hp}, low: {lp}}
            allowed -= (halos[high] | halos[low]) - {hp, lp}
        if any(not endpoints[k] for k in group):
            return None
        if len(group) == 1:
            choices = sorted(endpoints[group[0]])
            route = {choices[int(rng.integers(len(choices)))]}
        else:
            route = set()
            order = list(group)
            if not witness:
                rng.shuffle(order)
            for k1, k2 in zip(order, order[1:]):
                path = _path(rng, allowed, endpoints[k1], endpoints[k2], shape)
                if path is None:
                    return None
                route |= path
        for p in route:
            owner[p] = gi
    # A socket must visibly act as a junction, not merely an endpoint.
    if witness:
        group = groups[0]
        candidates = sorted((halos[group[0]] | halos[group[1]]) & free)
        rng.shuffle(candidates)
        leaf = next((p for p in candidates if p not in owner
                     and not any(q in owner for q in _neighbors(p, shape))
                     and not any(p in halos[k] for k in markers if k not in group)), None)
        if leaf is None:
            return None
        owner[leaf] = 0
    # Grow branches, loops and disconnected islands, while keeping different
    # planned port groups separated by at least one king-move gap.
    group_of = {k: gi for gi, group in enumerate(groups) for k in group}
    choices = sorted(free - set(owner))
    rng.shuffle(choices)
    target = max(len(owner) + 3, int(h * w * rng.uniform(.075, .19)))
    for p in choices:
        if len(owner) >= target:
            break
        neighbors = {owner[q] for q in _neighbors(p, shape) if q in owner}
        touched = {group_of[k] for k in markers if p in halos[k]}
        ids = neighbors | touched
        if len(ids) > 1:
            continue
        if witness and touched:
            continue
        if neighbors and rng.random() < .13:
            continue
        owner[p] = next(iter(ids)) if ids else -1
    # Always add a genuinely detached distractor, not another contact branch.
    if not any(g == -1 for g in owner.values()):
        for p in choices:
            if (p not in owner and not any(q in owner for q in _neighbors(p, shape))
                    and not any(p in halo for halo in halos.values())):
                owner[p] = -1
                break
    a = np.full(shape, bg, dtype=np.int8)
    for k, color in markers.items():
        a[_corner(k, h, w)] = color
    for p in owner:
        a[p] = wire
    return a


def _evidence(a, out):
    def differs(**kwargs):
        return not np.array_equal(out, _render(a, **kwargs))
    return {
        "diagonal_links": differs(links=False),
        "diagonal_contacts": differs(contacts=False),
        "socket_relay": differs(relay=False),
        "corner_ablation": differs(socket_mode="corner"),
        "lower_geometry": differs(socket_mode="squares"),
        "conflict": differs(policy="any_bridge"),
        "single_socket": differs(policy="one_socket"),
        "matching_pair_is_insufficient": differs(policy="any_pair"),
    }


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = tuple(int(x) for x in rng.permutation(10))
    varied_palette = num_train >= 3 and rng.random() < .55
    used, result = set(), {"train": [], "test": []}
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            training = split == "train"
            for attempt in range(180):
                pal = (tuple(int(x) for x in rng.permutation(10))
                       if varied_palette and (index or not training) else palette)
                groups, markers, witness = _plan(rng, index, training, num_train, pal[2:6])
                if training:
                    lo = 12 if index < 2 else 9
                    hi = 20 if index < 2 else 23
                    shape = (int(rng.integers(lo, hi)), int(rng.integers(lo, hi)))
                elif index == 0:
                    shape = (int(rng.integers(24, 31)), int(rng.integers(11, 27)))
                    if rng.integers(2):
                        shape = shape[::-1]
                else:
                    shape = (int(rng.integers(9, 31)), int(rng.integers(9, 31)))
                a = _build(rng, shape, groups, markers, pal, witness)
                if a is None or _key(a) in used:
                    continue
                out = solve(a)
                _, bg, wire, ms, wires, sockets = _read(a)
                nets = _networks(wires, sockets, a.shape)
                actual = {frozenset(ports) for cells, ports in nets if ports}
                if actual != {frozenset(g) for g in groups}:
                    continue
                # Both rejected and retained wire evidence must survive texture.
                retained = any(out[p] != bg for p in wires)
                rejected = any(out[p] == bg for p in wires)
                if not rejected:
                    continue
                if training and index < 2 and not retained:
                    continue
                if training and index == 0:
                    ev = _evidence(a, out)
                    if not all(ev[k] for k in ("diagonal_links", "diagonal_contacts",
                                               "socket_relay", "lower_geometry", "single_socket")):
                        continue
                if training and index == 1 and not _evidence(a, out)["conflict"]:
                    continue
                if training and index == 2 and not _evidence(a, out)["matching_pair_is_insufficient"]:
                    continue
                used.add(_key(a))
                result[split].append({"input": a, "output": out})
                break
            else:
                raise RuntimeError(f"could not construct {split} example {index}")
    # Demonstration position is not an input convention.
    result["train"] = [result["train"][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training_evidence = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape and _key(a) not in seen
            seen.add(_key(a))
            _, bg, wire, markers, wires, sockets = _read(a)
            assert np.array_equal(solve(a), out)
            assert all(out[_corner(k, *a.shape)] == bg for k in range(4))
            assert all(out[p] == markers[k] for k, cells in sockets.items() for p in cells)
            assert not np.any(out == wire)
            allowed = wires | set().union(*sockets.values())
            assert all(tuple(p) in allowed for p in np.argwhere(out != bg))
            nets = _networks(wires, sockets, a.shape)
            for cells, ports in nets:
                active = len(ports) >= 2 and len({markers[k] for k in ports}) == 1
                assert all(out[p] == (markers[next(iter(ports))] if active else bg) for p in cells)
            assert any(out[p] == bg for p in wires)
            if split == "train":
                training_evidence.append(_evidence(a, out))
    for name in ("diagonal_links", "diagonal_contacts", "socket_relay", "corner_ablation",
                 "lower_geometry", "conflict", "single_socket"):
        assert any(e[name] for e in training_evidence), f"missing demonstration: {name}"
    if len(puzzle["train"]) >= 3:
        assert any(e["matching_pair_is_insufficient"] for e in training_evidence)
    else:
        for pair in puzzle["test"]:
            a, _, _, _, wires, sockets = _read(pair["input"])
            assert all(len(ports) <= 2 for _, ports in _networks(wires, sockets, a.shape))
    assert max(max(np.asarray(p["input"]).shape) for p in puzzle["test"]) > max(
        max(np.asarray(p["input"]).shape) for p in puzzle["train"])


def self_test():
    """Bounded semantic, metamorphic, count and diversity checks on 42 seeds."""
    signatures, shapes, palettes, subsets, profiles = set(), set(), set(), set(), set()
    wire_counts, raw_component_counts, relay_count = set(), set(), 0
    retained_counts, erased_counts = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = tuple((_key(p["input"]), _key(p["output"]))
                          for split in ("train", "test") for p in puzzle[split])
        signatures.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                _, bg, wire, markers, wires, sockets = _read(a)
                shapes.add(a.shape)
                palettes.add((bg, wire, tuple(sorted(set(markers.values())))))
                subsets.add(tuple(sorted(markers)))
                wire_counts.add(len(wires))
                raw_component_counts.add(len(_components(wires, a.shape)))
                nets = _networks(wires, sockets, a.shape)
                profiles.add(tuple(sorted((len(cells), len(ports), len({markers[k] for k in ports}))
                                          for cells, ports in nets)))
                retained_counts.add(sum(out[p] != bg for p in wires))
                erased_counts.add(sum(out[p] == bg for p in wires))
                relay_count += not np.array_equal(out, _render(a, relay=False))
                # Color permutations and reflection must commute with the rule.
                perm = np.random.default_rng(seed + 1000).permutation(10)
                assert np.array_equal(solve(perm[a]), perm[out])
                assert np.array_equal(solve(np.fliplr(a)), np.fliplr(out))
        assert tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test")
                     for p in generate_puzzle(seed)[s]) == signature
    assert len(signatures) == 42 and len(shapes) >= 50 and len(palettes) >= 40
    assert len(subsets) >= 8 and len(profiles) >= 120, (len(subsets), len(profiles))
    assert len(wire_counts) >= 40 and len(raw_component_counts) >= 8
    assert len(retained_counts) >= 30 and len(erased_counts) >= 20 and relay_count >= 42
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            puzzle = generate_puzzle(4200 + 10 * tr + te, tr, te)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (tr, te)
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {tr}/{te}")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "role_palettes": len(palettes), "corner_subsets": len(subsets),
            "network_profiles": len(profiles), "wire_counts": len(wire_counts),
            "raw_component_counts": len(raw_component_counts), "relay_examples": relay_count}


if __name__ == "__main__":
    report = self_test()
    Path("scratch").mkdir(exist_ok=True)
    Path("scratch/self_test.json").write_text(json.dumps(report, separators=(",", ":")) + "\n")
    print(json.dumps(report, separators=(",", ":")))
