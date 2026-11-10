"""First-witness mosaics: register damaged copies and arbitrate D4 orbits.

Only NumPy and the standard library are used. No parent code is imported.
"""

from collections import Counter

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return a.astype(np.int8, copy=False)


def _orbit(n, r, c, axes_only=False):
    pairs = ((r, c),) if axes_only else ((r, c), (c, r))
    return frozenset((rr, cc) for a, b in pairs
                     for rr in (a, n - 1 - a)
                     for cc in (b, n - 1 - b))


def _orbits(n, axes_only=False):
    seen, result = set(), []
    for r in range(n):
        for c in range(n):
            if (r, c) not in seen:
                o = _orbit(n, r, c, axes_only)
                result.append(o)
                seen.update(o)
    return result


def _bbox(points):
    return (min(r for r, c in points), min(c for r, c in points),
            max(r for r, c in points), max(c for r, c in points))


def _analyze(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] * 2 <= a.size:
        raise ValueError("background must be a strict majority")
    colors = [int(c) for c in np.unique(a) if c != bg]
    if not 2 <= len(colors) <= 4:
        raise ValueError("expected two to four distinct copy colors")
    copies, sizes = [], set()
    for color in colors:
        points = {(int(r), int(c)) for r, c in np.argwhere(a == color)}
        top, left, bottom, right = _bbox(points)
        n = bottom - top + 1
        if n != right - left + 1 or not 5 <= n <= 9:
            raise ValueError("each color must expose a square footprint of side 5..9")
        sizes.add(n)
        copies.append({"color": color, "top": top, "left": left,
                       "points": {(r - top, c - left) for r, c in points}})
    if len(sizes) != 1:
        raise ValueError("all copies must have the same square size")
    n = sizes.pop()
    for i, x in enumerate(copies):
        for y in copies[i + 1:]:
            if not (x["top"] + n < y["top"] or y["top"] + n < x["top"]
                    or x["left"] + n < y["left"] or y["left"] + n < x["left"]):
                raise ValueError("square footprints need a blank row or column between them")
    return {"background": bg, "side": n, "copies": copies}


def _decisions(info, axes_only=False, priority="witness", only_first=False):
    """Input-derived orbit candidates, including counterfactual priority rules."""
    copies = info["copies"]
    if only_first:
        copies = [min(copies, key=lambda x: (x["top"], x["left"]))]
    result = []
    for orbit in _orbits(info["side"], axes_only):
        candidates = []
        for copy in copies:
            evidence = orbit & copy["points"]
            if not evidence:
                continue
            witness = min((r + copy["top"], c + copy["left"]) for r, c in evidence)
            if priority == "witness":
                key = witness
            elif priority == "box":
                key = (copy["top"], copy["left"])
            elif priority == "count":
                key = (-len(evidence), *witness)
            elif priority == "last":
                key = (-witness[0], -witness[1])
            else:
                raise ValueError("unknown priority")
            candidates.append({"color": copy["color"], "witness": witness,
                               "count": len(evidence), "key": key})
        if candidates:
            candidates.sort(key=lambda x: x["key"])
            result.append({"orbit": orbit, "winner": candidates[0]["color"],
                           "candidates": candidates})
    return result


def _render(info, **kwargs):
    out = np.full((info["side"], info["side"]), info["background"], dtype=np.int8)
    for decision in _decisions(info, **kwargs):
        for r, c in decision["orbit"]:
            out[r, c] = decision["winner"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Register each color by its visible square, then fuse first-witness orbits."""
    return _render(_analyze(input_grid))


def _sample_template(rng, n):
    orbits = _orbits(n)
    boundary = [o for o in orbits if any(r == 0 for r, c in o)]
    interior = [o for o in orbits if not any(r == 0 for r, c in o)]
    # The delayed orbit has eight members, so diagonal exchange is observable.
    delayed_choices = [o for o in boundary if len(o) == 8]
    delayed = delayed_choices[int(rng.integers(len(delayed_choices)))]
    guards = [o for o in boundary if o != delayed]
    guard = guards[int(rng.integers(len(guards)))]
    order = rng.permutation(len(interior))
    selected = [delayed, guard] + [interior[int(i)] for i in order[:2]]
    remaining = [o for o in orbits if o not in selected]
    desired = int(rng.integers(4, min(8, len(orbits)) + 1))
    if remaining:
        selected.extend(remaining[int(i)] for i in rng.permutation(len(remaining))[:desired - 4])
    borrow = selected[2 + int(rng.integers(2))]
    return set().union(*selected), selected, delayed, guard, borrow


def _layout(rng, n, count, delta, lead_right):
    """Irregular bands or staggered clusters, without a separator or fixed panel."""
    band_possible = count * n + count - 1 <= 30
    band = band_possible and (count == 2 or rng.random() < 0.52)
    if band:
        gaps = [1] * (count - 1)
        spare = 30 - count * n - sum(gaps)
        for j in rng.permutation(len(gaps)):
            extra = int(rng.integers(0, min(4, spare) + 1))
            gaps[int(j)] += extra
            spare -= extra
        xs, cursor = [], 0
        for j in range(count):
            xs.append(cursor)
            cursor += n + (gaps[j] if j < len(gaps) else 0)
        lead_slot, rival_slot = (1, 0) if lead_right else (0, 1)
        positions = [(0, xs[lead_slot]), (delta, xs[rival_slot])]
        for j in range(count):
            if j not in (lead_slot, rival_slot):
                positions.append((int(rng.integers(max(1, delta), n)), xs[j]))
    else:
        # Top pair compete directly; the lower copies can supply absent orbits.
        gap_x = int(rng.integers(1, min(6, 30 - 2 * n) + 1))
        xs = [0, n + gap_x]
        lead_slot, rival_slot = (1, 0) if lead_right else (0, 1)
        positions = [(0, xs[lead_slot]), (delta, xs[rival_slot])]
        gap_y = int(rng.integers(1, min(5, 30 - (2 * n + delta)) + 1))
        base = n + delta + gap_y
        bottom_gap = int(rng.integers(1, min(6, 30 - 2 * n) + 1))
        bottom_left = int(rng.integers(0, 30 - 2 * n - bottom_gap + 1))
        bottom_xs = [bottom_left, bottom_left + n + bottom_gap]
        for j in rng.permutation(2)[:count - 2]:
            x = bottom_xs[int(j)]
            y = base + int(rng.integers(0, min(3, 30 - base - n) + 1))
            positions.append((y, x))
    used_h = max(y for y, x in positions) + n
    used_w = max(x for y, x in positions) + n
    h = int(rng.integers(used_h, min(30, used_h + 6) + 1))
    w = int(rng.integers(used_w, min(30, used_w + 6) + 1))
    my = int(rng.integers(0, h - used_h + 1))
    mx = int(rng.integers(0, w - used_w + 1))
    return h, w, [(y + my, x + mx) for y, x in positions]


def _damage(rng, template, selected, delayed, guard, borrow, positions, hard):
    copies = []
    for _ in positions:
        points = sorted(template)
        fraction = float(rng.uniform(0.08, 0.30 if not hard else 0.42))
        deleted = int(round(len(points) * fraction))
        remove = {points[int(i)] for i in rng.permutation(len(points))[:deleted]}
        copies.append(template - remove)

    # The globally first square is NOT the earliest witness for the delayed orbit.
    # Its surviving low witnesses must lose to a single high witness of its rival.
    local_top = min(r for r, c in delayed)
    threshold = positions[1][0] - positions[0][0] + local_top
    late = {p for p in delayed if p[0] > threshold}
    assert len(late) >= 2
    copies[0].difference_update(delayed | borrow)
    copies[0].update(late)
    high = sorted(p for p in delayed if p[0] == local_top)
    witness = high[int(rng.integers(len(high)))]
    copies[1].difference_update(delayed)
    copies[1].add(witness)
    for copy in copies[2:]:
        copy.difference_update(delayed)
    # An orbit completely absent in the leading copy is supplied by another.
    donor = int(rng.integers(1, len(copies)))
    points = sorted(borrow)
    copies[donor].add(points[int(rng.integers(len(points)))])

    # Each square exposes its own bounds; no generation-only registration data.
    # A full guard in the first two also teaches the reading-order tie at delta=0.
    copies[0].update(guard)
    copies[1].update(guard)
    n = max(r for r, c in template) + 1
    for i, copy in enumerate(copies):
        allowed = template - delayed - (borrow if i == 0 else set())
        for axis, value in ((0, 0), (0, n - 1), (1, 0), (1, n - 1)):
            if not any(p[axis] == value for p in copy):
                options = sorted(p for p in allowed if p[axis] == value)
                copy.add(options[int(rng.integers(len(options)))])
    # Preserve collective evidence of every intended orbit, including singletons.
    for orbit in selected:
        if not any(copy & orbit for copy in copies):
            options = list(range(1, len(copies))) if orbit == borrow else list(range(len(copies)))
            i = options[int(rng.integers(len(options)))]
            points = sorted(orbit)
            copies[i].add(points[int(rng.integers(len(points)))])
    return copies


def _make_pair(rng, n, count, delta, lead_right, hard):
    bg = int(rng.integers(10))
    colors = [int(c) for c in rng.choice([c for c in range(10) if c != bg], count, replace=False)]
    template, selected, delayed, guard, borrow = _sample_template(rng, n)
    h, w, positions = _layout(rng, n, count, delta, lead_right)
    copies = _damage(rng, template, selected, delayed, guard, borrow, positions, hard)
    inp = np.full((h, w), bg, dtype=np.int8)
    for color, (top, left), copy in zip(colors, positions, copies):
        for r, c in copy:
            inp[top + r, left + c] = color
    if np.count_nonzero(inp == bg) * 2 <= inp.size:
        return None
    # Label from the intended sampled template and the actual surviving evidence.
    # solve independently rediscovers the extents and occupied orbit classes.
    out = np.full((n, n), bg, dtype=np.int8)
    for orbit in selected:
        candidates = []
        for color, (top, left), copy in zip(colors, positions, copies):
            if copy & orbit:
                first = min((top + r, left + c) for r, c in copy & orbit)
                candidates.append((first, color))
        winner = min(candidates)[1]
        for r, c in orbit:
            out[r, c] = winner
    pair = {"input": inp, "output": out}
    # These are construction predicates, never extra rules applied by solve.
    info = _analyze(inp)
    if np.array_equal(out, _render(info, axes_only=True)):
        return None
    return pair


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    base = int(rng.choice([5, 7]))
    train_counts = [2, int(rng.choice([3, 4]))]
    train_counts += [int(rng.integers(2, 5)) for _ in range(num_train - 2)]
    specs = []
    for i, count in enumerate(train_counts):
        n = base if i == 0 else base + 1 if i == 1 else int(rng.choice([base, base + 1]))
        delta = 0 if i == 0 else 1 if i == 1 else int(rng.integers(0, 3))
        right = False if i == 0 else True if i == 1 else bool(delta and rng.integers(2))
        specs.append(("train", n, count, delta, right, False))
    for i in range(num_test):
        n = base + 2 if i == 0 else int(rng.choice([base, base + 1, base + 2]))
        count = int(rng.choice(train_counts))
        delta = int(rng.integers(0, 3))
        specs.append(("test", n, count, delta, bool(delta and rng.integers(2)), True))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, n, count, delta, right, hard in specs:
        for _ in range(80):
            pair = _make_pair(rng, n, count, delta, right, hard)
            if pair is None:
                continue
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("could not sample a distinct informative pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _raw_fusion(info):
    """Ablation: registered pixel overlay without symmetry completion."""
    n = info["side"]
    out = np.full((n, n), info["background"], dtype=np.int8)
    for r in range(n):
        for c in range(n):
            candidates = [((x["top"] + r, x["left"] + c), x["color"])
                          for x in info["copies"] if (r, c) in x["points"]]
            if candidates:
                out[r, c] = min(candidates)[1]
    return out


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _analyze(inp)
    assert out.shape == (info["side"], info["side"])
    decisions = _decisions(info)
    assert 4 <= len(decisions) <= 8
    assert len({d["winner"] for d in decisions}) >= 2
    occupied = set().union(*(d["orbit"] for d in decisions))
    assert sum(any(r == 0 for r, c in d["orbit"]) for d in decisions) >= 2
    for d in decisions:
        assert all(out[r, c] == d["winner"] for r, c in d["orbit"])
    assert all(out[r, c] == info["background"] for r in range(info["side"])
               for c in range(info["side"]) if (r, c) not in occupied)
    assert np.array_equal(solve(inp), out)
    assert any(not (d["orbit"] & x["points"])
               for d in decisions for x in info["copies"])
    # Counterexamples to neglecting rank, cross-copy borrowing, full D4,
    # actual witness locations, and priority independent of evidence count.
    for alternative in (_render(info, priority="box"), _render(info, priority="count"),
                        _render(info, priority="last"), _render(info, only_first=True),
                        _render(info, axes_only=True), _raw_fusion(info)):
        assert not np.array_equal(alternative, out)
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            inp = _grid(pair["input"])
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            infos[split].append(info)
    # The two guaranteed demonstrations teach both center parities, a tie,
    # and a higher square on the right. Tests recombine demonstrated counts.
    first, second = infos["train"][:2]
    assert first["side"] % 2 == 1 and second["side"] == first["side"] + 1
    assert any(len(d["candidates"]) >= 2 and
               d["candidates"][0]["witness"][0] == d["candidates"][1]["witness"][0]
               for d in _decisions(first))
    lead = min(second["copies"], key=lambda x: (x["top"], x["left"]))
    assert any(x["left"] < lead["left"] and x["top"] > lead["top"] for x in second["copies"])
    assert infos["test"][0]["side"] == first["side"] + 2
    seen_counts = {len(x["copies"]) for x in infos["train"]}
    assert all(len(x["copies"]) in seen_counts for x in infos["test"])


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def self_test():
    """Bounded 42-seed sweep, all explicit count combinations, and ablations."""
    shapes, backgrounds, sides, counts, patterns, layouts = (set() for _ in range(6))
    owners, orbit_counts = Counter(), Counter()
    for seed in range(42):
        p = generate_puzzle(seed)
        for split in ("train", "test"):
            for pair in p[split]:
                info = _analyze(pair["input"])
                decisions = _decisions(info)
                shapes.add(pair["input"].shape)
                backgrounds.add(info["background"])
                sides.add(info["side"])
                counts.add(len(info["copies"]))
                patterns.add((info["side"], tuple(sorted(min(d["orbit"]) for d in decisions))))
                layouts.add((pair["input"].shape, tuple(sorted((x["top"], x["left"])
                                                             for x in info["copies"]))))
                owners[len({d["winner"] for d in decisions})] += 1
                orbit_counts[len(decisions)] += 1
    assert len(shapes) >= 45 and len(backgrounds) == 10
    assert sides == {5, 6, 7, 8, 9} and counts == {2, 3, 4}
    assert len(patterns) >= 60 and len(layouts) >= 150
    assert len(owners) >= 2 and len(orbit_counts) >= 4
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(12345 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _equal(generate_puzzle(987, 5, 2), generate_puzzle(987, 5, 2))
    assert not _equal(generate_puzzle(987), generate_puzzle(988))
    # Palette renaming is equivariant; margin translation preserves every rank.
    pair = generate_puzzle(2026, 2, 1)["train"][0]
    permutation = np.random.default_rng(17).permutation(10)
    assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    info = _analyze(pair["input"])
    top = min(x["top"] for x in info["copies"])
    left = min(x["left"] for x in info["copies"])
    cropped = pair["input"][top:, left:]
    # Cropping can remove the majority background, so pad visibly when it fits.
    if max(cropped.shape) <= 28:
        padded = np.pad(cropped, 1, constant_values=info["background"])
        assert np.array_equal(solve(padded), pair["output"])
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes),
            "backgrounds": len(backgrounds), "sides": sorted(sides),
            "copy_counts": sorted(counts), "templates": len(patterns),
            "layouts": len(layouts), "output_owner_counts": dict(sorted(owners.items())),
            "occupied_orbit_counts": dict(sorted(orbit_counts.items()))}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
