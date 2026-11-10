"""Glyph-locked paint gates: a scale-matched cargo opens a prefix of gates.

Only standard-library modules and NumPy are used.  All interpretation is local
to the supplied grid; neither solve nor its parser knows a puzzle's seed.
"""

from itertools import combinations
import json

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _bounds(mask):
    rr, cc = np.where(mask)
    if not len(rr):
        raise ValueError("Empty object")
    return int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1


def _small_shapes(h, w, count):
    """Small, tightly cropped binary motifs, including disconnected motifs."""
    result = []
    for cells in combinations(range(h * w), count):
        m = np.zeros((h, w), dtype=bool)
        m.flat[list(cells)] = True
        if m[0].any() and m[-1].any() and m[:, 0].any() and m[:, -1].any():
            result.append(m)
    return result


def _reduce_cargo(mask):
    candidates = []
    for scale in (2, 3):
        h, w = mask.shape
        if h % scale or w % scale or not (2 <= h // scale <= 3 and 2 <= w // scale <= 3):
            continue
        small = mask[::scale, ::scale]
        if small.all() or not np.array_equal(mask, np.repeat(np.repeat(small, scale, 0), scale, 1)):
            continue
        if h % 2 == 0:
            candidates.append((scale, small.copy()))
    if len(candidates) != 1:
        raise ValueError("Cargo must have a unique uniform scale of two or three")
    return candidates[0]


def _parse_canonical(a, bg, launcher, cargo):
    """Try a scene whose launcher is on the left, with labels on either side."""
    lm, pm = a == launcher, a == cargo
    lr0, lr1, lc0, depth = _bounds(lm)
    if lc0 != 0 or depth > 6:
        raise ValueError("Launcher is not on the launch boundary")
    pr0, pr1, pu0, pu1 = _bounds(pm)
    width, length = pr1 - pr0, pu1 - pu0
    if pu0 != depth or width % 2 or not (4 <= width <= 6):
        raise ValueError("Cargo does not touch a centered launcher")
    if (pr0 + pr1) % 2:
        raise ValueError("Cargo center is not between two transverse cells")
    center, shift = (pr0 + pr1) // 2, width // 2
    pattern = pm[pr0:pr1, pu0:pu1]
    scale, glyph = _reduce_cargo(pattern)
    profile = []
    for u in range(depth):
        ys = np.flatnonzero(lm[:, u])
        if (len(ys) < width or len(ys) % 2 or len(ys) != ys[-1] - ys[0] + 1
                or int(ys[0] + ys[-1] + 1) != 2 * center):
            raise ValueError("Launcher cross sections must be centered and solid")
        profile.append(len(ys))
    if any(x < y for x, y in zip(profile, profile[1:])):
        raise ValueError("Launcher must not widen toward the cargo")
    # Tight cropping guarantees at least one cargo cell contacts the tip.
    gates = []
    for color in np.unique(a):
        color = int(color)
        if color in (bg, launcher, cargo):
            continue
        mask = a == color
        xs = np.flatnonzero(mask[center - 1])
        if not len(xs) or len(xs) != xs[-1] - xs[0] + 1:
            raise ValueError("A gate must cross the travel axis")
        u0, u1 = int(xs[0]), int(xs[-1]) + 1
        if u0 < pu1 or not 1 <= u1 - u0 <= 3:
            raise ValueError("Gate is not ahead of cargo or is too thick")
        r0, r1 = center - 1, center
        while r0 and mask[r0 - 1, u0:u1].all():
            r0 -= 1
        while r1 < len(a) and mask[r1, u0:u1].all():
            r1 += 1
        if r0 + r1 != 2 * center or (r1 - r0) <= width:
            raise ValueError("Gate must be centered, with two substantial halves")
        if r0 - shift < 0 or r1 + shift > len(a):
            raise ValueError("Opened gate would leave the grid")
        slab = np.zeros_like(mask)
        slab[r0:r1, u0:u1] = True
        if not mask[slab].all():
            raise ValueError("Gate slab must be solid")
        label = mask & ~slab
        gr0, gr1, gu0, gu1 = _bounds(label)
        small = label[gr0:gr1, gu0:gu1]
        if (not (2 <= small.shape[0] <= 3 and 2 <= small.shape[1] <= 3)
                or small.all() or not (gr1 < r0 - shift or gr0 > r1 + shift)
                or gu1 <= u0 or gu0 >= u1):
            raise ValueError("Each gate needs an outboard, overlapping-axis glyph")
        gates.append(dict(color=color, u0=u0, u1=u1, r0=r0, r1=r1,
                          label=label, glyph=small.copy(), side="top" if gr1 < center else "bottom",
                          matches=np.array_equal(small, glyph)))
    gates.sort(key=lambda g: g["u0"])
    if not 1 <= len(gates) <= 4:
        raise ValueError("Expected one to four gates")
    if any(g["u1"] >= h["u0"] for g, h in zip(gates, gates[1:])):
        raise ValueError("Gate slabs must have a gap")
    opened = 0
    for g in gates:
        if not g["matches"]:
            break
        opened += 1
    stop = gates[opened]["u0"] if opened < len(gates) else a.shape[1]
    destination = stop - length
    if destination < depth or destination <= pu0:
        raise ValueError("Cargo must travel forward without hitting the launcher")
    return dict(canonical=a, background=bg, launcher=launcher, cargo=cargo,
                depth=depth, profile=tuple(profile), center=center, width=width,
                length=length, scale=scale, glyph=glyph, pattern=pattern.copy(),
                pr0=pr0, pr1=pr1, pu0=pu0, pu1=pu1, gates=gates,
                opened=opened, destination=destination)


def _interpret(input_grid):
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    # A strict majority makes background independent of colors or orientation.
    i = int(np.argmax(counts))
    if counts[i] * 2 <= grid.size or not 4 <= len(values) <= 7:
        raise ValueError("Expected a majority background and three to six role colors")
    bg = int(values[i])
    foreground = [int(c) for c in values if c != bg]
    found = []
    for rotation in range(4):
        a = np.rot90(grid, rotation).copy()
        for launcher in foreground:
            if not np.any(a[:, 0] == launcher):
                continue
            for cargo in foreground:
                if cargo == launcher:
                    continue
                try:
                    info = _parse_canonical(a, bg, launcher, cargo)
                except (ValueError, IndexError):
                    continue
                info["rotation"] = rotation
                found.append(info)
    if len(found) != 1:
        raise ValueError(f"Expected one scene interpretation, found {len(found)}")
    return found[0]


def _render(info, open_all=False, stationary=False):
    a, bg = info["canonical"], info["background"]
    out = a.copy()
    out[a == info["cargo"]] = bg
    n = len(info["gates"]) if open_all else info["opened"]
    shift, center = info["width"] // 2, info["center"]
    for g in info["gates"][:n]:
        out[g["r0"]:g["r1"], g["u0"]:g["u1"]] = bg
        out[g["label"]] = bg
        if not stationary:
            out[g["r0"] - shift:center - shift, g["u0"]:g["u1"]] = g["color"]
            out[center + shift:g["r1"] + shift, g["u0"]:g["u1"]] = g["color"]
        else:
            out[g["r0"]:g["r1"], g["u0"]:g["u1"]] = g["color"]
    stop = a.shape[1] if open_all or n == len(info["gates"]) else info["gates"][n]["u0"]
    dest = info["pu0"] if stationary else stop - info["length"]
    color = info["gates"][n - 1]["color"] if n else info["cargo"]
    view = out[info["pr0"]:info["pr1"], dest:dest + info["length"]]
    view[info["pattern"]] = color
    return np.rot90(out, -info["rotation"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Open exactly the matching gate prefix and move/repaint the cargo."""
    return _render(_interpret(input_grid))


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _make_pair(rng, motifs, cargo_class, matches, rotation, seen):
    glyph = motifs[cargo_class]
    gh, gw = glyph.shape
    for _ in range(300):
        scale = int(rng.choice((2, 3))) if gh == 2 else 2
        pattern = np.repeat(np.repeat(glyph, scale, 0), scale, 1)
        width, length = pattern.shape
        shift = width // 2
        n = len(matches)
        halfspans = [int(rng.integers(shift + 1, shift + 4)) for _ in matches]
        maxhalf = max(halfspans)
        thicknesses = [int(rng.integers(1, 4)) for _ in matches]
        sides = [int(rng.integers(2)) for _ in matches]
        top = gh + 1 + int(rng.integers(3)) if 0 in sides else int(rng.integers(4))
        bottom = gh + 1 + int(rng.integers(3)) if 1 in sides else int(rng.integers(4))
        height = top + 2 * maxhalf + width + bottom
        if height > 30:
            continue
        center = top + maxhalf + shift
        depth = int(rng.integers(1, 5))
        tip = int(rng.integers(0, 2))
        levels = sorted([int(rng.integers(tip, min(maxhalf, 4) + 1)) for _ in range(depth)], reverse=True)
        profile = [width + 2 * x for x in levels]
        # The largest launcher cross section fits even with zero side clearance.
        if profile[0] // 2 > min(center, height - center):
            continue
        positions = [depth + length + int(rng.integers(1, 4))]
        for j in range(n - 1):
            positions.append(positions[-1] + max(gw, thicknesses[j]) + int(rng.integers(1, 3)))
        end = positions[-1] + max(gw, thicknesses[-1])
        axis_length = end + int(rng.integers(0, 5))
        # Sometimes the terminal slab is flush with the destination boundary.
        if rng.random() < 0.3:
            axis_length = positions[-1] + thicknesses[-1]
        if axis_length > 30:
            continue
        colors = [int(x) for x in rng.choice(10, n + 3, replace=False)]
        bg, launcher, cargo = colors[:3]
        inp = np.full((height, axis_length), bg, dtype=np.int8)
        out = inp.copy()
        for u, w in enumerate(profile):
            inp[center - w // 2:center + w // 2, u] = launcher
            out[center - w // 2:center + w // 2, u] = launcher
        pr0, pu0 = center - shift, depth
        inp[pr0:pr0 + width, pu0:pu0 + length][pattern] = cargo
        opened = next((j for j, m in enumerate(matches) if not m), n)
        destination = (positions[opened] if opened < n else axis_length) - length
        if destination <= depth:
            continue
        valid = True
        for j, (u0, t, half, side) in enumerate(zip(positions, thicknesses, halfspans, sides)):
            color = colors[3 + j]
            g = glyph if matches[j] else motifs[1 - cargo_class]
            gu0 = max(0, min(u0 + (t - gw) // 2, axis_length - gw))
            if side == 0:
                gr0 = int(rng.integers(0, top - gh))
            else:
                gr0 = center + maxhalf + shift + 1 + int(rng.integers(bottom - gh))
            # Per-gate labels use the same dimensions/population but different
            # exact masks for a mismatch; their ink is the gate's paint.
            view = inp[gr0:gr0 + gh, gu0:gu0 + gw]
            if np.any(view != bg):
                valid = False
                break
            view[g] = color
            inp[center - half:center + half, u0:u0 + t] = color
            if j < opened:
                out[center - half - shift:center - shift, u0:u0 + t] = color
                out[center + shift:center + half + shift, u0:u0 + t] = color
            else:
                out[center - half:center + half, u0:u0 + t] = color
                out[gr0:gr0 + gh, gu0:gu0 + gw][g] = color
        if not valid or np.count_nonzero(inp == bg) * 2 <= inp.size:
            continue
        final_color = colors[2 + opened] if opened else cargo
        target_view = out[pr0:pr0 + width, destination:destination + length]
        if target_view.shape != pattern.shape or np.any(target_view[pattern] != bg):
            continue
        target_view[pattern] = final_color
        inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
        if _key(inp) in seen:
            continue
        # Construction and interpretation are deliberately separate.
        info = _interpret(inp)
        if info["opened"] != opened or not np.array_equal(solve(inp), out):
            raise AssertionError("Constructed scene disagrees with its visible interpretation")
        seen.add(_key(inp))
        return {"input": inp, "output": out}
    raise RuntimeError("Unable to place a unique glyph-gate scene")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    h, w = ((2, 2), (2, 3), (3, 2), (3, 3))[int(rng.integers(4))]
    count = int(rng.integers(max(h, w, 3), h * w))
    vocabulary = _small_shapes(h, w, count)
    chosen = rng.choice(len(vocabulary), 2, replace=False)
    motifs = [vocabulary[int(i)].copy() for i in chosen]
    base_rotation = int(rng.integers(4))
    # The first two demonstrations are mandatory witnesses.  Their order is
    # shuffled after construction, so position is not a class cue.
    schedules = [(0, [True, False, True], base_rotation),
                 (1, [True, True], (base_rotation + 1) % 4)]
    for _ in range(num_train - 2):
        if rng.random() < 0.5:
            flags = [True] * int(rng.integers(1, 3))
        else:
            flags = [True, False] + [bool(rng.integers(2)) for _ in range(int(rng.integers(3)))]
        schedules.append((int(rng.integers(2)), flags, int(rng.integers(4))))
    seen = set()
    train = [_make_pair(rng, motifs, cls, flags, rot, seen) for cls, flags, rot in schedules]
    rng.shuffle(train)
    test = []
    for i in range(num_test):
        # Hold out stopping after >=2 openings, or a run of >=3 openings.
        if rng.random() < 0.55:
            prefix = int(rng.integers(2, 4))
            flags = [True] * prefix + [False]
        else:
            flags = [True] * int(rng.integers(3, 5))
        rot = (base_rotation + 2 + i) % 4
        test.append(_make_pair(rng, motifs, int(rng.integers(2)), flags, rot, seen))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _positions(mask):
    return set(map(tuple, np.argwhere(mask).tolist()))


def _check_pair(pair):
    inp, actual = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == actual.shape
    info = _interpret(inp)
    a = info["canonical"]
    out = np.rot90(actual, info["rotation"])
    bg, center, shift = info["background"], info["center"], info["width"] // 2
    # A set-based conservation check, independent of the rectangle painter.
    expected = {(int(r), int(c)): int(a[r, c]) for r, c in np.argwhere(a == info["launcher"])}
    for j, g in enumerate(info["gates"]):
        slab_cells = {(r, u) for r in range(g["r0"], g["r1"]) for u in range(g["u0"], g["u1"])}
        label_cells = _positions(g["label"])
        if j < info["opened"]:
            cells = {(r - shift if r < center else r + shift, u) for r, u in slab_cells}
            assert len(cells) == len(slab_cells)
            assert all(not info["pr0"] <= r < info["pr1"] for r, _ in cells)
        else:
            cells = slab_cells | label_cells
            assert all(out[r, u] == a[r, u] for r, u in cells)
        assert not (set(expected) & cells)
        expected.update({pos: g["color"] for pos in cells})
        assert _positions(out == g["color"]) - {
            (info["pr0"] + int(r), info["destination"] + int(c))
            for r, c in np.argwhere(info["pattern"])
        } == cells
    final_color = info["gates"][info["opened"] - 1]["color"] if info["opened"] else info["cargo"]
    payload_cells = {(info["pr0"] + int(r), info["destination"] + int(c)) for r, c in np.argwhere(info["pattern"])}
    assert not (set(expected) & payload_cells)
    expected.update({pos: final_color for pos in payload_cells})
    assert _positions(out != bg) == set(expected)
    assert all(int(out[pos]) == color for pos, color in expected.items())
    assert np.array_equal(solve(inp), actual)
    assert not np.array_equal(inp, actual)
    info["blocked"] = info["opened"] < len(info["gates"])
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            sig = _key(_grid(pair["input"]))
            assert sig not in seen
            seen.add(sig)
            infos[split].append(_check_pair(pair))
    vocab = {_key(i["glyph"]) for i in infos["train"]}
    assert len(vocab) == 2
    for i in infos["train"] + infos["test"]:
        assert _key(i["glyph"]) in vocab
        assert i["opened"] >= 1
        for g in i["gates"]:
            assert _key(g["glyph"]) in vocab
            assert g["glyph"].shape == i["glyph"].shape
            assert g["glyph"].sum() == i["glyph"].sum()
        # Removing the B mechanism (leaving cargo and gates stationary) fails.
        assert not np.array_equal(_render(i, stationary=True), _render(i))
    train = infos["train"]
    assert any(i["opened"] == 1 and len(i["gates"]) >= 3
               and not i["gates"][1]["matches"] and i["gates"][2]["matches"] for i in train)
    assert any(i["opened"] == 2 and not i["blocked"] for i in train)
    assert len({i["rotation"] % 2 for i in train}) == 2
    # Removing A's shape discrimination would open all barriers.  The mandatory
    # blocked witness gives a different position, paint, and barrier state.
    assert any(not np.array_equal(_render(i, open_all=True), _render(i)) for i in train)
    assert all(i["opened"] <= 2 and (i["opened"] < 2 or not i["blocked"]) for i in train)
    assert all((i["opened"] >= 2 and i["blocked"]) or i["opened"] >= 3 for i in infos["test"])


def _fingerprint(puzzle):
    return tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded tests: 42 seeds, all legal counts, ablations and metamorphisms."""
    puzzles = [generate_puzzle(s) for s in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    shapes, motifs, profiles, palettes, scales, directions, gate_counts = (set() for _ in range(7))
    thicknesses, halfspans, open_counts, label_sides = (set() for _ in range(4))
    blocked = all_pass = terminal_edge = transverse_edge = disconnected = 0
    for puzzle in puzzles:
        for pair in puzzle["train"] + puzzle["test"]:
            info = _interpret(pair["input"])
            shapes.add(pair["input"].shape)
            motifs.add(_key(info["glyph"]))
            profiles.add(info["profile"])
            palettes.add(tuple(int(x) for x in np.unique(pair["input"])))
            scales.add(info["scale"])
            directions.add(info["rotation"])
            gate_counts.add(len(info["gates"]))
            open_counts.add(info["opened"])
            blocked += info["opened"] < len(info["gates"])
            all_pass += info["opened"] == len(info["gates"])
            for g in info["gates"]:
                thicknesses.add(g["u1"] - g["u0"])
                halfspans.add((g["r1"] - g["r0"]) // 2)
                label_sides.add(g["side"])
                terminal_edge += g["u1"] == info["canonical"].shape[1]
                transverse_edge += g["r0"] - info["width"] // 2 == 0 or g["r1"] + info["width"] // 2 == len(info["canonical"])
            # Entire-scene rotation and color renaming must commute with solve.
            assert np.array_equal(solve(np.rot90(pair["input"])), np.rot90(pair["output"]))
            permutation = np.array([7, 3, 9, 0, 6, 1, 8, 4, 2, 5], dtype=np.int8)
            assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    assert len(shapes) >= 40 and len(motifs) >= 15 and len(profiles) >= 25
    assert len(palettes) >= 30 and scales == {2, 3} and directions == {0, 1, 2, 3}
    assert gate_counts == {1, 2, 3, 4} and open_counts == {1, 2, 3, 4}
    assert thicknesses == {1, 2, 3} and len(halfspans) >= 4 and label_sides == {"top", "bottom"}
    assert min(blocked, all_pass, terminal_edge, transverse_edge) >= 5
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            validate_generated_puzzle(generate_puzzle(2026, train, test))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, 1.5), (None, 1), (4, None), (4.0, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    return {"seeds": 42, "pairs": sum(len(p[s]) for p in puzzles for s in ("train", "test")),
            "grid_shapes": len(shapes), "motifs": len(motifs), "launcher_profiles": len(profiles),
            "palettes": len(palettes), "scales": sorted(scales), "gate_counts": sorted(gate_counts),
            "opened_counts": sorted(open_counts), "blocked_pairs": blocked, "all_pass_pairs": all_pass,
            "terminal_gate_edge_contacts": terminal_edge, "potential_split_edge_contacts": transverse_edge}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
