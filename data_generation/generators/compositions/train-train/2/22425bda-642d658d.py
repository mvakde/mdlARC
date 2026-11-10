"""Layer-aware duplicate embroidery: an input-only ARC family.

Threads establish depth. A hub marks an ornament on exactly one thread.
Only a higher thread can explain a missing cardinal arm. Restore and retain
the two complete ornaments with the repeated arm label, discarding the rest.
"""

from collections import Counter
import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_ORIENTATIONS = ("H", "V", "D", "A")


def _cells(line, height, width):
    kind, value = line
    if kind == "H":
        return [(value, c) for c in range(width)]
    if kind == "V":
        return [(r, value) for r in range(height)]
    if kind == "D":
        return [(r, r - value) for r in range(height) if 0 <= r - value < width]
    if kind == "A":
        return [(r, value - r) for r in range(height) if 0 <= value - r < width]
    raise ValueError("unknown line direction")


def _foot(center):
    r, c = center
    return {(r, c)} | {(r + dr, c + dc) for dr, dc in _DIRS}


def _members(lines, h, w):
    result = {}
    for i, line in enumerate(lines):
        for p in _cells(line, h, w):
            result.setdefault(p, []).append(i)
    return result


def _topological(n, edges):
    remaining, result = set(range(n)), []
    while remaining:
        ready = [i for i in remaining if not any(v == i and u in remaining
                                                 for u, v in edges)]
        if len(ready) != 1:
            raise ValueError("depth is cyclic or not uniquely determined")
        remaining.remove(ready[0])
        result.append(ready[0])
    return result


def _as_grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=False)


def _infer_line(grid, color):
    points = np.argwhere(grid == color)
    if len(points) < 2:
        raise ValueError("thread lacks visible support")
    rr, cc = points[:, 0], points[:, 1]
    candidates = []
    for kind, values in (("H", rr), ("V", cc), ("D", rr - cc), ("A", rr + cc)):
        if np.all(values == values[0]):
            candidates.append((kind, int(values[0])))
    if len(candidates) != 1:
        raise ValueError("thread pixels do not specify one straight line")
    return candidates[0]


def _inspect(input_grid):
    """Recover all semantic roles and local decisions from the finished input."""
    a = _as_grid(input_grid)
    h, w = a.shape
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if np.count_nonzero(counts == counts[bg]) != 1:
        raise ValueError("background must be uniquely modal")
    boundary = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    colors = sorted(set(map(int, boundary)) - {bg})
    if not 3 <= len(colors) <= 4:
        raise ValueError("expected three or four boundary-to-boundary threads")
    lines = [_infer_line(a, color) for color in colors]
    if len(set(lines)) != len(lines):
        raise ValueError("two thread identities share a spine")
    members = _members(lines, h, w)
    color_index = {color: i for i, color in enumerate(colors)}
    edges = set()
    for p, indices in members.items():
        if len(indices) > 1:
            visible = color_index.get(int(a[p]))
            if visible not in indices:
                raise ValueError("a thread crossing is covered by a marker")
            edges.update((i, visible) for i in indices if i != visible)
    order = _topological(len(lines), edges)
    rank = {i: k for k, i in enumerate(order)}
    extras = set(map(int, np.unique(a))) - {bg} - set(colors)
    # Hubs are the sole extra color all of whose pixels lie on a single spine.
    # Each arm label has visible off-spine pixels, including in axis-only scenes.
    hubs = [color for color in extras
            if all(len(members.get(tuple(map(int, p)), ())) == 1
                   for p in np.argwhere(a == color))]
    if len(hubs) != 1:
        raise ValueError("the hub role is ambiguous")
    hub = hubs[0]
    labels = extras - {hub}
    if not 2 <= len(labels) <= 4:
        raise ValueError("expected two through four arm labels")
    motifs = []
    for pp in np.argwhere(a == hub):
        p = tuple(map(int, pp))
        r, c = p
        if not (1 < r < h - 2 and 1 < c < w - 2):
            raise ValueError("ornaments must leave the boundary clear")
        own = members[p][0]
        arms = [(r + dr, c + dc) for dr, dc in _DIRS]
        visible = Counter(int(a[q]) for q in arms if int(a[q]) in labels)
        candidates = [label for label, count in visible.items() if count >= 3]
        if len(candidates) != 1:
            raise ValueError("an ornament needs three agreeing visible arms")
        label = candidates[0]
        blocked, lower = [], []
        complete = True
        for q in arms:
            value = int(a[q])
            if value == label:
                continue
            index = color_index.get(value)
            if index is not None and index != own and index in members.get(q, ()):
                if rank[index] > rank[own]:
                    blocked.append(q)
                    continue
                lower.append(q)
            complete = False
        motifs.append({"center": p, "layer": own, "label": label,
                       "complete": complete, "blocked": tuple(blocked),
                       "lower": tuple(lower), "arms": tuple(arms)})
    valid_counts = Counter(m["label"] for m in motifs if m["complete"])
    repeated = [c for c, n in valid_counts.items() if n == 2]
    if (len(repeated) != 1 or set(valid_counts) != labels
            or any(n not in (1, 2) for n in valid_counts.values())):
        raise ValueError("complete ornaments lack a unique duplicated label")
    return {"grid": a, "background": bg, "hub": hub, "colors": colors,
            "lines": lines, "members": members, "order": order, "rank": rank,
            "labels": labels, "motifs": motifs, "counts": valid_counts,
            "answer": repeated[0], "pixel_counts": counts, "edges": edges}


def _render_selection(info, motifs):
    out = np.full(info["grid"].shape, info["background"], dtype=np.int8)
    for m in motifs:
        for p in m["arms"]:
            out[p] = m["label"]
        out[m["center"]] = info["hub"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _inspect(input_grid)
    return _render_selection(info, [m for m in info["motifs"]
                                    if m["complete"] and m["label"] == info["answer"]])


def _dimensions(rng):
    regime = int(rng.integers(4))
    if regime == 0:
        return int(rng.integers(16, 23)), int(rng.integers(17, 25))
    if regime == 1:
        return int(rng.integers(22, 31)), int(rng.integers(22, 31))
    if regime == 2:
        return int(rng.integers(15, 21)), int(rng.integers(25, 31))
    return int(rng.integers(25, 31)), int(rng.integers(15, 21))


def _pool(h, w, kind):
    values = (range(2, h - 2) if kind == "H" else range(2, w - 2)
              if kind == "V" else range(2 - w, h - 1)
              if kind == "D" else range(1, h + w - 2))
    return [(kind, v) for v in values if len(_cells((kind, v), h, w)) >= 10]


def _layout(rng, n, required):
    h, w = _dimensions(rng)
    if required is not None:
        kinds = list(required)
        lines = [pool[int(rng.integers(len(pool)))]
                 for pool in [_pool(h, w, kind) for kind in kinds]]
    elif rng.random() < 0.32:
        # Adjacent parallel threads can hide one another's off-spine arms.
        kind = str(rng.choice(("H", "V")))
        bound = h if kind == "H" else w
        v = int(rng.integers(3, bound - 4))
        other = "V" if kind == "H" else "H"
        pool = _pool(h, w, other)
        lines = [(kind, v), (kind, v + 1), pool[int(rng.integers(len(pool)))]]
        if n == 4:
            pool = _pool(h, w, str(rng.choice(("D", "A"))))
            lines.append(pool[int(rng.integers(len(pool)))])
    else:
        if n == 4 and rng.random() < 0.5:
            kinds = list(_ORIENTATIONS)
        else:
            kinds = [str(rng.choice(("H", "V"))), str(rng.choice(("D", "A")))]
            kinds += [str(rng.choice(_ORIENTATIONS)) for _ in range(n - 2)]
        lines = []
        for kind in kinds:
            choices = [line for line in _pool(h, w, kind) if line not in lines]
            if not choices:
                return None
            lines.append(choices[int(rng.integers(len(choices)))])
    if len(set(lines)) != n:
        return None
    members = _members(lines, h, w)
    if any(len(indices) > 1 and (p[0] in (0, h - 1) or p[1] in (0, w - 1))
           for p, indices in members.items()):
        return None
    order = list(map(int, rng.permutation(n)))
    rank = {i: k for k, i in enumerate(order)}
    edges = {(i, max(indices, key=rank.get)) for indices in members.values()
             if len(indices) > 1 for i in indices if i != max(indices, key=rank.get)}
    try:
        if _topological(n, edges) != order:
            return None
    except ValueError:
        return None
    candidates = {"masked": [], "lower": [], "plain": []}
    for own, line in enumerate(lines):
        for p in _cells(line, h, w):
            r, c = p
            if not (1 < r < h - 2 and 1 < c < w - 2) or len(members[p]) != 1:
                continue
            if any(len(members.get(q, ())) > 1 for q in _foot(p)):
                continue
            intruders = [(q, i) for q in _foot(p) - {p}
                         for i in members.get(q, ()) if i != own]
            if len(intruders) > 1:
                continue
            kind = ("plain" if not intruders else "masked"
                    if rank[intruders[0][1]] > rank[own] else "lower")
            candidates[kind].append((p, own, intruders[0][0] if intruders else None))
    if len(candidates["masked"]) < 1 or len(candidates["lower"]) < 2:
        return None
    return h, w, lines, order, members, candidates


def _choose_ornaments(rng, layout, label_count, both_hidden, trap_count):
    candidates = layout[-1]
    for _ in range(90):
        chosen, occupied = [], set()

        def take(options, different_layer=None):
            choices = [x for x in options if not (_foot(x[0]) & occupied)
                       and (different_layer is None or x[1] != different_layer)]
            if not choices:
                return None
            x = choices[int(rng.integers(len(choices)))]
            chosen.append(x)
            occupied.update(_foot(x[0]))
            return x

        if any(take(candidates["lower"]) is None for _ in range(trap_count)):
            continue
        first = take(candidates["masked"])
        if first is None:
            continue
        options = (candidates["masked"] if both_hidden else
                   candidates["plain"] + candidates["lower"])
        if take(options, None if both_hidden else first[1]) is None:
            continue
        all_options = sum(candidates.values(), [])
        if any(take(all_options) is None for _ in range(label_count - 1)):
            continue
        return chosen
    return None


def _make_pair(rng, n, label_count, required, both_hidden, noise_level, trap_labels):
    for _ in range(1600):
        layout = _layout(rng, n, required)
        if layout is None:
            continue
        trap_count = len(trap_labels)
        chosen = _choose_ornaments(rng, layout, label_count, both_hidden, trap_count)
        if chosen is None:
            continue
        h, w, lines, order, members, _ = layout
        palette = list(map(int, rng.permutation(10)))
        bg, hub = palette[:2]
        thread_colors = palette[2:2 + n]
        arm_labels = palette[2 + n:2 + n + label_count]
        answer, decoy = arm_labels[:2]
        # Trap counts and labels vary. In particular, training includes a tied
        # candidate-count case, so counting all three-arm hubs cannot solve it.
        labels = [arm_labels[i] for i in trap_labels] + [answer, answer] + arm_labels[1:]
        ornaments = []
        for index, ((center, layer, collision), label) in enumerate(zip(chosen, labels)):
            ornaments.append({"center": center, "layer": layer, "label": label,
                              "missing": collision if index < trap_count else None})
        a = np.full((h, w), bg, dtype=np.int8)
        for layer in order:
            for p in _cells(lines[layer], h, w):
                a[p] = thread_colors[layer]
            for m in ornaments:
                if m["layer"] != layer:
                    continue
                for p in _foot(m["center"]) - {m["center"]}:
                    if p != m["missing"]:
                        a[p] = m["label"]
                a[m["center"]] = hub
        protected = set().union(*(_foot(m["center"]) for m in ornaments))
        free = [p for p in np.ndindex(a.shape)
                if 0 < p[0] < h - 1 and 0 < p[1] < w - 1
                and p not in protected and p not in members]
        rng.shuffle(free)
        # Unattached arm-colored pixels never create hubs or overwrite threads.
        # They can include diagonal neighbors: diagonals do not define a plus.
        if noise_level:
            for label in arm_labels:
                for _ in range(int(rng.integers(0, 9))):
                    if free:
                        a[free.pop()] = label
            if noise_level == 2:
                needed = max(0, int(np.count_nonzero(a == answer))
                             - int(np.count_nonzero(a == decoy))) + int(rng.integers(3, 10))
                for _ in range(needed):
                    if free:
                        a[free.pop()] = decoy
        output = np.full((h, w), bg, dtype=np.int8)
        for m in ornaments[trap_count:trap_count + 2]:
            for p in _foot(m["center"]) - {m["center"]}:
                output[p] = answer
            output[m["center"]] = hub
        pair = {"input": a, "output": output}
        try:
            info = _pair_features(pair)
            if noise_level == 2 and info["pixel_counts"][decoy] <= info["pixel_counts"][answer]:
                continue
            return pair
        except (ValueError, AssertionError):
            continue
    raise RuntimeError("bounded search could not construct a readable thread scene")


def _check_count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _check_count(num_train, 2, 10, "num_train")
    num_test = _check_count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(3))
    slope = str(rng.choice(("D", "A")))
    puzzle, seen = {"train": [], "test": []}, set()
    for index in range(num_train + num_test):
        training = index < num_train
        n = 3 if shift == 2 and training else 4 if shift == 2 else int(rng.integers(3, 5))
        required = (("H", "V", slope) if training else _ORIENTATIONS) if shift == 2 else None
        label_count = (2 + index if training and index < 2 else
                       4 if not training and shift == 0 else
                       int(rng.integers(2, 4)) if training and shift == 0 else
                       int(rng.integers(2, 5)))
        label_count = min(label_count, 8 - n)
        both_hidden = not training and shift == 1
        noise_level = (0 if training and index == 0 else 2 if training and index == 1
                       else int(rng.integers(2)))
        if training and index == 0:
            trap_labels = [0, 1]
        elif training and index == 1:
            trap_labels = [1]
        else:
            max_traps = 2 if required is not None else 3
            trap_labels = list(map(int, rng.integers(label_count, size=int(rng.integers(1, max_traps + 1)))))
        for _ in range(8):
            pair = _make_pair(rng, n, label_count, required, both_hidden, noise_level, trap_labels)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle["train" if training else "test"].append(pair)
                break
        else:
            raise RuntimeError("could not produce distinct inputs")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _depth_blind(info):
    """A deliberate wrong rule: every thread may explain an interrupted arm."""
    accepted = []
    for m in info["motifs"]:
        if all(int(info["grid"][q]) == m["label"]
               or int(info["grid"][q]) in info["colors"] for q in m["arms"]):
            accepted.append(m)
    counts = Counter(m["label"] for m in accepted)
    return _render_selection(info, [m for m in accepted if counts[m["label"]] == 2])


def _pair_features(pair):
    a, out = _as_grid(pair["input"]), _as_grid(pair["output"])
    info = _inspect(a)
    assert np.array_equal(solve(a), out)
    assert out.shape == a.shape
    assert 3 <= sum(info["counts"].values()) <= 5
    assert 1 <= len(info["motifs"]) - sum(info["counts"].values()) <= 3
    assert all(len(m["lower"]) == 1 for m in info["motifs"] if not m["complete"])
    assert any(m["complete"] and m["blocked"] and m["label"] == info["answer"]
               for m in info["motifs"])
    # Every identity has two unmodified boundary endpoints, with no marker there.
    h, w = a.shape
    for color, line in zip(info["colors"], info["lines"]):
        boundary_cells = [p for p in _cells(line, h, w)
                          if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
        assert len(boundary_cells) == 2
        assert all(int(a[p]) == color for p in boundary_cells)
    occupied = set()
    for m in info["motifs"]:
        footprint = _foot(m["center"])
        assert not (footprint & occupied)
        occupied.update(footprint)
        assert not any(len(info["members"].get(q, ())) > 1 for q in footprint)
        intruding_arms = [q for q in m["arms"]
                          if any(i != m["layer"] for i in info["members"].get(q, ()))]
        assert len(intruding_arms) <= 1
        for q in m["arms"]:
            if int(a[q]) == m["label"]:
                assert all(info["rank"][i] <= info["rank"][m["layer"]]
                           for i in info["members"].get(q, ()))
    # Check the skeleton wherever an ornament or an unattached colored dot has
    # not replaced it. Crossing observations must repaint to exactly this input.
    for p, indices in info["members"].items():
        top = max(indices, key=info["rank"].get)
        if int(a[p]) in info["colors"] or int(a[p]) == info["background"]:
            assert int(a[p]) == info["colors"][top]
        else:
            assert p in occupied
    all_complete = _render_selection(info, [m for m in info["motifs"] if m["complete"]])
    assert not np.array_equal(all_complete, out)
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen, train_info = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            info = _pair_features(pair)
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            if split == "train":
                train_info.append(info)
    assert {2, 3} <= {len(info["labels"]) for info in train_info}
    assert any(not np.array_equal(_depth_blind(info), solve(info["grid"])) for info in train_info)
    # Equal hub-candidate counts for the answer and a singleton rule out the
    # accidental shortcut "the answer is the most frequent three-arm hub".
    assert any(any(Counter(m["label"] for m in info["motifs"])[c]
                       == Counter(m["label"] for m in info["motifs"])[info["answer"]]
                       for c in info["labels"] if c != info["answer"]) for info in train_info)
    assert any(max(info["pixel_counts"][c] for c in info["labels"] if c != info["answer"])
               > info["pixel_counts"][info["answer"]] for info in train_info)
    assert all(any(m["complete"] and not m["blocked"] and m["label"] == info["answer"]
                   for m in info["motifs"]) for info in train_info)


def _signature(puzzle):
    return tuple(_key(p[name]) for split in ("train", "test")
                 for p in puzzle[split] for name in ("input", "output"))


def self_test():
    """Bounded 42-seed checks, ablations, equivariances, counts, and corruption."""
    shapes, structures, palettes, line_counts, label_counts = set(), set(), set(), set(), set()
    directions, parallel, masked_pair, transitive, signatures = set(), 0, 0, 0, set()
    trap_counts, candidate_ranks, depth_failures = set(), set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _pair_features(pair)
                shapes.add(pair["input"].shape)
                roles = {}
                structures.add((pair["input"].shape, tuple(roles.setdefault(int(c), len(roles))
                                                         for c in pair["input"].ravel())))
                palettes.add((info["background"], info["hub"], tuple(info["colors"])))
                line_counts.add(len(info["lines"]))
                label_counts.add(len(info["labels"]))
                kinds = [line[0] for line in info["lines"]]
                directions.update(kinds)
                parallel += len(set(kinds)) < len(kinds)
                masked_pair += sum(m["complete"] and bool(m["blocked"])
                                   and m["label"] == info["answer"] for m in info["motifs"]) == 2
                transitive += len(info["edges"]) < len(info["lines"]) * (len(info["lines"]) - 1) // 2
                trap_counts.add(sum(not m["complete"] for m in info["motifs"]))
                candidates = Counter(m["label"] for m in info["motifs"])
                answer_count = candidates[info["answer"]]
                others = [n for c, n in candidates.items() if c != info["answer"]]
                candidate_ranks.add("tie" if answer_count in others else
                                    "high" if answer_count > max(others) else "low")
                depth_failures += not np.array_equal(_depth_blind(info), pair["output"])
        if seed < 4:
            pair = puzzle["test"][0]
            perm = np.random.default_rng(10000 + seed).permutation(10)
            assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
            for transform in (np.transpose, np.fliplr, np.flipud):
                assert np.array_equal(solve(transform(pair["input"])), transform(pair["output"]))
    assert len(signatures) == 42 and len(structures) >= 190
    assert len(shapes) >= 65 and len(palettes) >= 100
    assert line_counts == {3, 4} and label_counts == {2, 3, 4}
    assert directions == set(_ORIENTATIONS) and parallel >= 20
    assert masked_pair >= 5 and transitive >= 10
    assert trap_counts == {1, 2, 3} and {"tie", "high", "low"} <= candidate_ranks
    assert depth_failures >= 84
    assert _signature(generate_puzzle(0)) == _signature(generate_puzzle(0))
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            validate_generated_puzzle(generate_puzzle(700 + train * 7 + test, train, test))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = generate_puzzle(999)
    broken["train"][0]["output"] = broken["train"][0]["input"].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted target")


if __name__ == "__main__":
    self_test()
    print("42-seed self-test passed")
