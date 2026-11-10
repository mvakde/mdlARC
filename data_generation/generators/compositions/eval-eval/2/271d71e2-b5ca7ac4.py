"""Fuel-limited queues. All conventions are readable from the input.

No files, parent generators, random state, or training examples are read by solve.
Generation uses destination-order boundary constraints; validation also ticks
vehicles one cell at a time. No execution traces or generated corpora are saved.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


SELF_TEST_SEEDS = (0, 494035496, 640793275, 651929074, 633562110, 933478146,
                  669079431, 225744663, 687356112, 155539139, 424010618,
                  122306281, 881355888, 228299878, 573053600, 819334234,
                  662934388, 860975053, 626023340, 127686475, 174921458,
                  904864116, 9110640, 523797931, 974715701, 14978359,
                  444015948, 246116391, 968446514, 251645278, 985410134,
                  704544121, 834632572, 315541904, 762038910, 695187518,
                  184856121, 308588882, 87087280, 781497585, 445000197, 465653766)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid dimensions")
    _need(np.issubdtype(g.dtype, np.integer) and np.all((0 <= g) & (g <= 9)),
          "grids must contain integer ARC colors")
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, g.tobytes()


def _order(h, w, side):
    # Travel-up row-major ordering, rotated with the vehicle, as in parent A.
    if side == 0:  # left: leading column first, bottom to top within each column
        return [(r, c) for c in range(1, w-1) for r in range(h-2, 0, -1)]
    return [(r, c) for c in range(w-2, 0, -1) for r in range(1, h-1)]


def _band(a, b):
    return max(a["r"], b["r"]) < min(a["r"]+a["h"], b["r"]+b["h"])


def _overlap(a, b):
    return _band(a, b) and max(a["c"], b["c"]) < min(a["c"]+a["w"], b["c"]+b["w"])


def _disjoint(objects):
    return not any(_overlap(a, b) for i, a in enumerate(objects) for b in objects[i+1:])


def _decode(input_grid):
    original = _grid(input_grid)
    _need(min(original.shape) >= 5, "arena is too small")
    horizontal = (np.all(original[:, 0] == original[0, 0])
                  and np.all(original[:, -1] == original[0, -1])
                  and original[0, 0] != original[0, -1])
    vertical = (np.all(original[0] == original[0, 0])
                and np.all(original[-1] == original[-1, 0])
                and original[0, 0] != original[-1, 0])
    _need(horizontal != vertical, "exactly one opposite pair of rails is required")
    axis = 1 if horizontal else 0
    g = original if horizontal else np.rot90(original, 1)
    H, W = g.shape
    rails = (int(g[0, 0]), int(g[0, -1]))
    bg = int(g[0, 1])
    _need(bg not in rails and np.all(g[0, 1:-1] == bg)
          and np.all(g[-1, 1:-1] == bg), "missing empty guard bands")
    covered = g == bg
    covered[:, 0] = covered[:, -1] = True
    objects, stamps, other_states = [], set(), set()
    # The corner stamps delimit each frame, even if two rectangles touch.
    for r in range(1, H-1):
        for c in range(1, W-1):
            if covered[r, c]:
                continue
            _need(r+2 < H-1 and c+2 < W-1, "incomplete vehicle")
            spent, frame = int(g[r, c]), int(g[r, c+1])
            _need(frame in rails and spent not in (*rails, bg), "invalid corner/frame roles")
            right, bottom = c+2, r+2
            while right < W-1 and g[r, right] == frame:
                right += 1
            while bottom < H-1 and g[bottom, c] == frame:
                bottom += 1
            _need(right < W-1 and bottom < H-1, "open frame")
            patch = g[r:bottom+1, c:right+1].copy()
            h, w = patch.shape
            border = np.full((h, w), frame, dtype=np.int8)
            border[0, 0] = border[0, -1] = border[-1, 0] = border[-1, -1] = spent
            _need(np.array_equal(patch[0], border[0]) and np.array_equal(patch[-1], border[-1])
                  and np.array_equal(patch[:, 0], border[:, 0])
                  and np.array_equal(patch[:, -1], border[:, -1]), "broken keyed frame")
            _need(not np.any(covered[r:bottom+1, c:right+1]), "overlapping or hollow vehicles")
            interior = patch[1:-1, 1:-1]
            colors = set(map(int, np.unique(interior)))
            _need(not colors.intersection((*rails, bg)), "counter uses a structural color")
            other_states.update(colors - {spent})
            stamps.add(spent)
            side = rails.index(frame)
            order = _order(h, w, side)
            values = [int(patch[p]) for p in order]
            a = values.count(spent)
            _need(all(v == spent for v in values[:a]) and all(v != spent for v in values[a:]),
                  "spent cells must be a direction-relative prefix")
            objects.append(dict(r=r, c=c, h=h, w=w, side=side, frame=frame,
                                a=a, q=len(order)-a, patch=patch))
            covered[r:bottom+1, c:right+1] = True
    _need(objects and np.all(covered) and len(stamps) == 1 and len(other_states) <= 1,
          "scene is not completely explained by one counter alphabet")
    spent = next(iter(stamps))
    _need(spent not in other_states, "inconsistent counter alphabet")
    return dict(g=g, axis=axis, bg=bg, rails=rails, spent=spent,
                fuel=next(iter(other_states), None), objects=objects)


def _settle(objects, width, *, fuel=True, blockers=True):
    result = [dict(o) for o in objects]
    for side in (0, 1):
        indices = [i for i, o in enumerate(objects) if o["side"] == side]
        indices.sort(key=lambda i: (objects[i]["c"] if side == 0
                                   else -objects[i]["c"]-objects[i]["w"], objects[i]["r"]))
        settled = []
        for i in indices:
            src, dst = objects[i], result[i]
            ahead = [p for p in settled if _band(src, p)] if blockers else []
            if side == 0:
                boundary = max((p["c"]+p["w"] for p in ahead), default=1)
                free = src["c"]-boundary
            else:
                boundary = min((p["c"] for p in ahead), default=width-1)
                free = boundary-src["c"]-src["w"]
            _need(free >= 0, "initial same-class order is invalid")
            m = min(src["q"], free) if fuel else free
            dst.update(c=src["c"]+(-m if side == 0 else m), m=m, free=free)
            settled.append(dst)
    return result


def _draw(scene, settled):
    g = np.full(scene["g"].shape, scene["bg"], dtype=np.int8)
    g[:, 0], g[:, -1] = scene["rails"]
    for o in settled:
        patch = o["patch"].copy()
        for p in _order(o["h"], o["w"], o["side"])[o["a"]:o["a"]+o.get("m", 0)]:
            patch[p] = scene["spent"]
        g[o["r"]:o["r"]+o["h"], o["c"]:o["c"]+o["w"]] = patch
    return g if scene["axis"] == 1 else np.rot90(g, -1).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read rails, corner keys and unary meters; settle both stable queues."""
    scene = _decode(input_grid)
    settled = _settle(scene["objects"], scene["g"].shape[1])
    _need(_disjoint(settled), "opposite classes have incompatible final footprints")
    return _draw(scene, settled)


def _tick_prediction(scene):
    """Validation by local one-cell motion, without the boundary recurrence."""
    objects = [dict(o, patch=o["patch"].copy(), used=0) for o in scene["objects"]]
    H, W = scene["g"].shape
    for _ in range(W*len(objects)+1):
        moved = False
        for i, o in enumerate(objects):
            if o["used"] == o["q"]:
                continue
            candidate = dict(o, c=o["c"]+(-1 if o["side"] == 0 else 1))
            if not 1 <= candidate["c"] <= W-1-o["w"]:
                continue
            if any(j != i and p["side"] == o["side"] and _overlap(candidate, p)
                   for j, p in enumerate(objects)):
                continue
            p = _order(o["h"], o["w"], o["side"])[o["a"]+o["used"]]
            o["patch"][p] = scene["spent"]
            o["c"] = candidate["c"]
            o["used"] += 1
            moved = True
        if not moved:
            break
    else:
        raise AssertionError("unit-step simulation did not converge")
    _need(_disjoint(objects), "unit-step final classes overlap")
    # Counters were already updated one cell at a time, so draw adds no ticks.
    return _draw(scene, objects)


def _depth(objects):
    depths, degrees = [1]*len(objects), [0]*len(objects)
    for side in (0, 1):
        indices = [i for i, o in enumerate(objects) if o["side"] == side]
        indices.sort(key=lambda i: objects[i]["c"] if side == 0
                     else -objects[i]["c"]-objects[i]["w"])
        for k, i in enumerate(indices):
            parents = [j for j in indices[:k] if _band(objects[i], objects[j])]
            depths[i] = 1+max((depths[j] for j in parents), default=0)
            degrees[i] = len(parents)
    return max(depths), max(degrees)


def _edge_first(objects, width):
    """A competing final-position rule that can jump same-class blockers."""
    result = [dict(o) for o in objects]
    for side in (0, 1):
        indices = [i for i, o in enumerate(objects) if o["side"] == side]
        indices.sort(key=lambda i: objects[i]["c"] if side == 0
                     else -objects[i]["c"]-objects[i]["w"])
        settled = []
        for i in indices:
            o = objects[i]
            positions = (range(max(1, o["c"]-o["q"]), o["c"]+1) if side == 0 else
                         range(min(width-1-o["w"], o["c"]+o["q"]), o["c"]-1, -1))
            for c in positions:
                candidate = dict(o, c=c)
                if all(not _overlap(candidate, p) for p in settled):
                    result[i] = candidate
                    settled.append(candidate)
                    break
            else:
                raise AssertionError("source position must remain feasible")
    return result


def _features(scene):
    objects, width = scene["objects"], scene["g"].shape[1]
    settled = _settle(objects, width)
    unlimited = _settle(objects, width, fuel=False)
    independent = _settle(objects, width, blockers=False)
    two_pass = [o["c"]+(-1 if o["side"] == 0 else 1)*min(o["q"], abs(o["c"]-p["c"]))
                for o, p in zip(objects, unlimited)]
    columns = [o["c"] for o in settled]
    depth, degree = _depth(objects)
    merge_reversal = False
    for i, o in enumerate(objects):
        ahead = [j for j, p in enumerate(objects) if p["side"] == o["side"] and _band(o, p)
                 and (p["c"]+p["w"] <= o["c"] if o["side"] == 0
                      else p["c"] >= o["c"]+o["w"])]
        if len(ahead) < 2 or settled[i]["m"] >= o["q"]:
            continue
        source = [objects[j]["c"]+objects[j]["w"] if o["side"] == 0
                  else -objects[j]["c"] for j in ahead]
        final = [settled[j]["c"]+settled[j]["w"] if o["side"] == 0
                 else -settled[j]["c"] for j in ahead]
        if (source.count(max(source)) == final.count(max(final)) == 1
                and source.index(max(source)) != final.index(max(final))):
            merge_reversal = True
    passed = any(a["side"] == 0 and b["side"] == 1 and _band(a, b)
                 and a["c"] > b["c"] and settled[i]["c"]+a["w"] <= settled[j]["c"]
                 for i, a in enumerate(objects) for j, b in enumerate(objects))
    return dict(count=len(objects), depth=depth, degree=degree, merge_reversal=merge_reversal,
                no_A=columns != [o["c"] for o in unlimited],
                no_B=columns != [o["c"] for o in independent],
                nonsequential=columns != two_pass,
                overtaking=columns != [o["c"] for o in _edge_first(objects, width)],
                blocked=sum(o["m"] < o["q"] and o["m"] <
                            (p["c"]-1 if p["side"] == 0 else width-1-p["c"]-p["w"])
                            for p, o in zip(objects, settled)),
                edge=sum(o["c"] == (1 if o["side"] == 0 else width-1-o["w"])
                         for o in settled),
                fuel_short=sum(o["m"] == o["q"] and o["m"] < o["free"] for o in settled),
                fresh=sum(o["a"] == 0 for o in objects),
                exhausted=sum(o["q"] == 0 for o in objects),
                stationary=sum(o["m"] == 0 for o in settled),
                shapes=sorted({(o["h"], o["w"]) for o in objects}),
                moves=[o["m"] for o in settled],
                sides=sorted({o["side"] for o in objects}),
                scan_sides=sorted({p["side"] for p, o in zip(objects, settled)
                                   if p["h"] > 3 and p["w"] > 3 and 0 < o["m"] < p["q"]
                                   and (p["a"] % (p["h"]-2)
                                        or (p["a"]+o["m"]) % (p["h"]-2))}),
                crossing=any(a["side"] == 0 and b["side"] == 1 and _band(a, b)
                             and a["c"] > b["c"] for a in objects for b in objects),
                passed=passed)


def _patch(o, roles):
    h, w = o["h"], o["w"]
    spent, fuel = roles["spent"], roles["fuel"]
    p = np.full((h, w), roles["rails"][o["side"]], dtype=np.int8)
    p[0, 0] = p[0, -1] = p[-1, 0] = p[-1, -1] = spent
    order = _order(h, w, o["side"])
    a = len(order)-o["q"]
    for i, cell in enumerate(order):
        p[cell] = spent if i < a else fuel
    return dict(o, frame=roles["rails"][o["side"]], a=a, patch=p)


def _charge(rng, capacity):
    mode = int(rng.integers(5))
    return capacity if mode == 0 else 0 if mode == 1 else int(rng.integers(capacity+1))


def _chain(rng, long=False):
    n = int(rng.integers(4, 6)) if long else 3
    for _ in range(100):
        h = int(rng.integers(5, 7)) if long else int(rng.integers(7, 10))
        widths = list(map(int, rng.integers(4, 6, n))) if long else list(map(int, rng.integers(5, 7, n)))
        step = int(rng.integers((h+1)//2, h))
        head = int(rng.integers(3, 6))
        gaps = [1]*(n-1) if long else list(map(int, rng.integers(1, 3, n-1)))
        used = head+sum(widths)+sum(gaps)
        min_m = h+(n-1)*step
        if used > 28 or min_m > 28:
            continue
        lead = int(rng.integers(1, min(3, head-1)))
        capacities = [(h-2)*(w-2) for w in widths]
        moves = [lead+sum(gaps[:i]) for i in range(n)]
        needed = moves[-1] if long else lead+sum(gaps)+widths[1]+widths[2]
        if capacities[-1] < needed or any(capacities[i] < moves[i]+1 for i in range(1, n-1)):
            continue
        N, M = int(rng.integers(used, 29)), int(rng.integers(min_m, 29))
        r0 = 1+int(rng.integers(M-min_m+1))
        reverse = bool(rng.integers(2))
        objects, c = [], 1+head
        for i, w in enumerate(widths):
            q = (lead if i == 0 else int(rng.integers(
                needed if i == n-1 else moves[i]+1, capacities[i]+1)))
            r = r0+(n-1-i if reverse else i)*step
            objects.append(dict(r=r, c=c, h=h, w=w, side=0, q=q))
            c += w+(gaps[i] if i < n-1 else 0)
        return M, N, objects
    raise RuntimeError("bounded chain construction failed")


def _fork(rng):
    for _ in range(256):
        ha, hb, wa, wb, wc = map(int, rng.integers(4, 8, 5))
        gap = int(rng.integers(3))
        oa, ob = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        hc = gap+oa+ob
        if hc < 4:
            continue
        head_a = int(rng.integers(1, min(5, (ha-2)*(wa-2)-1)+1))
        head_b = int(rng.integers(3, 8))
        qa = int(rng.integers(head_a+1, (ha-2)*(wa-2)+1))
        qb = int(rng.integers(1, min(head_b-1, (hb-2)*(wb-2))+1))
        ca, cb = 1+head_a, 1+head_b
        # The initially closest front must not remain the farthest front.
        # This makes the merge distinguish final-boundary reasoning from
        # merely following the nearest source rectangle.
        if (ca+wa-cb-wb)*((1+wa)-(cb-qb+wb)) >= 0:
            continue
        cc = max(ca+wa, cb+wb)+int(rng.integers(1, 4))
        free = cc-max(1+wa, cb-qb+wb)
        cap = (hc-2)*(wc-2)
        used, min_m = cc+wc-1, ha+gap+hb
        if used > 28 or cap <= free:
            continue
        M, N = int(rng.integers(min_m, 29)), int(rng.integers(used, 29))
        r = 1+int(rng.integers(M-min_m+1))
        objects = [dict(r=r, c=ca, h=ha, w=wa, side=0, q=qa),
                   dict(r=r+ha+gap, c=cb, h=hb, w=wb, side=0, q=qb),
                   dict(r=r+ha-oa, c=cc, h=hc, w=wc, side=0,
                        q=int(rng.integers(free+1, cap+1)))]
        return M, N, objects
    raise RuntimeError("bounded fork construction failed")


def _add_random(rng, objects, M, N, target, training, preserve=None):
    objects = list(objects)
    bias = float(rng.choice((0, .2, .5, .5, .8, 1)))
    for _ in range(max(0, target-len(objects))):
        for _ in range(70):
            h, w = int(rng.integers(3, min(9, M+1))), int(rng.integers(3, min(9, N+1)))
            r, c = int(rng.integers(1, M-h+2)), int(rng.integers(1, N-w+2))
            if rng.random() < .25:
                c = int(rng.choice((1, N+1-w)))
            if objects and rng.random() < .45:
                other = objects[int(rng.integers(len(objects)))]
                r = int(rng.choice((other["r"], other["r"]+other["h"]-1,
                                    other["r"]-h+1, other["r"]+other["h"])))
            if not 1 <= r <= M+1-h:
                continue
            candidate = dict(r=r, c=c, h=h, w=w, side=int(rng.random() >= bias),
                             q=_charge(rng, (h-2)*(w-2)))
            proposed = objects+[candidate]
            if not _disjoint(proposed) or training and _depth(proposed)[0] > 3:
                continue
            settled = _settle(proposed, N+2)
            if not _disjoint(settled):
                continue
            if preserve:
                # Preserve the teaching geometry by keeping additions out of
                # its perpendicular bands. Other-class additions may cross.
                if any(_band(candidate, o) and candidate["side"] == o["side"] for o in preserve):
                    continue
            objects = proposed
            break
        else:
            break
    return objects


def _make_pair(rng, roles, purpose, side, background):
    if purpose in ("witness", "long"):
        M, N, objects = _chain(rng, long=purpose == "long")
    elif purpose == "fork":
        M, N, objects = _fork(rng)
    else:
        M, N = int(rng.integers(4, 29)), int(rng.integers(6, 29))
        objects = []
    if side:
        objects = [dict(o, c=N+2-o["c"]-o["w"], side=1) for o in objects]
    target = (len(objects)+int(rng.integers(4)) if objects else int(rng.integers(1, 13)))
    objects = _add_random(rng, objects, M, N, target, training=purpose != "long",
                          preserve=objects if objects else None)
    if not objects:
        return None
    objects = [_patch(o, roles) for o in objects]
    scene = dict(g=np.empty((M+2, N+2), dtype=np.int8), axis=roles["axis"],
                 bg=background, rails=roles["rails"], spent=roles["spent"],
                 fuel=roles["fuel"], objects=objects)
    settled = _settle(objects, N+2)
    if not _disjoint(settled):
        return None
    return dict(input=_draw(scene, objects), output=_draw(scene, settled))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name in ((seed, "seed"), (num_train, "num_train"), (num_test, "num_test")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(int(seed))
    colors = list(map(int, rng.permutation(10)))
    roles = dict(rails=tuple(colors[:2]), spent=colors[2], fuel=colors[3],
                 axis=int(rng.integers(2)))
    backgrounds = colors[4:]
    shared_bg = bool(rng.integers(2))
    lesson_indices = list(map(int, rng.choice(num_train, 2, replace=False)))
    lesson_side = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    taught_passing = False
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            purpose, side = "free", int(rng.integers(2))
            if split == "train" and i in lesson_indices:
                k = lesson_indices.index(i)
                purpose, side = ("witness" if k == 0 else "fork"), lesson_side ^ k
            if split == "test" and i == 0:
                purpose, side = "long", int(rng.integers(2))
            for _ in range(64):
                bg = backgrounds[0] if shared_bg else int(rng.choice(backgrounds))
                pair = _make_pair(rng, roles, purpose, side, bg)
                if pair is None or _key(pair["input"]) in seen:
                    continue
                scene = _decode(pair["input"])
                f = _features(scene)
                if purpose == "witness" and not all(f[k] for k in ("no_A", "no_B", "nonsequential", "overtaking")):
                    continue
                if purpose == "fork" and not (f["edge"] and f["blocked"]
                                               and f["degree"] >= 2 and f["merge_reversal"]):
                    continue
                if purpose in ("witness", "fork") and side not in f["scan_sides"]:
                    continue
                if split == "test" and f["passed"] and not taught_passing:
                    continue
                if split == "train":
                    taught_passing |= f["passed"]
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded distinct-pair construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check visible structure, local dynamics, conservation, and lessons."""
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _need(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid example counts")
    seen, conventions, fuel_colors = set(), set(), set()
    features = {"train": [], "test": []}
    moved_sides, informative_sides = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape and _key(inp) not in seen, "shape change or duplicate input")
            seen.add(_key(inp))
            scene = _decode(inp)
            conventions.add((scene["axis"], scene["rails"], scene["spent"]))
            if scene["fuel"] is not None:
                fuel_colors.add(scene["fuel"])
            settled = _settle(scene["objects"], scene["g"].shape[1])
            _need(_disjoint(settled), "overlapping final classes")
            _need(np.array_equal(out, _draw(scene, settled)), "incorrect output")
            _need(np.array_equal(out, _tick_prediction(scene)), "local motion disagrees with output")
            # Each body's footprint and perpendicular coordinate are preserved;
            # exactly one counter cell changes per unit of actual displacement.
            for src, dst in zip(scene["objects"], settled):
                _need(abs(src["c"]-dst["c"]) == dst["m"] <= src["q"], "fuel conservation failed")
                if split == "train" and dst["m"]:
                    moved_sides.add(src["side"])
                    if (src["h"] > 3 and src["w"] > 3 and dst["m"] < src["q"]
                            and (src["a"] % (src["h"]-2)
                                 or (src["a"]+dst["m"]) % (src["h"]-2))):
                        informative_sides.add(src["side"])
            features[split].append(_features(scene))
    _need(len(conventions) == 1 and len(fuel_colors) == 1, "puzzle conventions changed")
    _need(moved_sides == informative_sides == {0, 1}, "training must teach both directions and meter order")
    _need(any(all(f[k] for k in ("no_A", "no_B", "nonsequential", "overtaking"))
              for f in features["train"]), "missing causal queue/ordering witness")
    _need(any(f["degree"] >= 2 and f["edge"] and f["blocked"] and f["merge_reversal"]
              for f in features["train"]),
          "missing two-front merge demonstration")
    _need(max(f["depth"] for f in features["train"]) <= 3,
          "training exceeds the declared depth curriculum")
    _need(max(f["depth"] for f in features["test"]) >= 4, "test lacks the depth shift")
    _need(not any(f["passed"] for f in features["test"])
          or any(f["passed"] for f in features["train"]), "test introduces undemonstrated passing")


def self_test():
    """Bounded in-memory checks over the 42 supplied host seeds; returns metrics."""
    signatures, shapes, body_shapes, palettes, axes, counts, depths = set(), set(), set(), set(), set(), set(), set()
    totals = Counter()
    first = None
    for seed in SELF_TEST_SEEDS:
        p = generate_puzzle(seed)
        palette_scene = next(_decode(pair["input"]) for pair in p["train"]
                             if _decode(pair["input"])["fuel"] is not None)
        palettes.add((palette_scene["rails"], palette_scene["spent"], palette_scene["fuel"]))
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                for key in ("input", "output"):
                    digest.update(bytes(pair[key].shape))
                    digest.update(pair[key].tobytes())
                scene = _decode(pair["input"])
                f = _features(scene)
                shapes.add(pair["input"].shape)
                body_shapes.update(f["shapes"])
                axes.add(scene["axis"])
                counts.add(f["count"])
                depths.add(f["depth"])
                for key in ("no_A", "no_B", "nonsequential", "overtaking", "blocked", "edge",
                            "fuel_short", "fresh", "exhausted", "stationary", "crossing", "passed",
                            "merge_reversal"):
                    totals[key] += int(f[key])
                totals["pairs"] += 1
        signatures.add(digest.digest())
        if first is None:
            first = p
    assert len(signatures) == 42 and axes == {0, 1}
    assert len(shapes) >= 50 and len(body_shapes) >= 20 and len(palettes) >= 30
    assert {1, 3, 4, 5} <= counts and {1, 2, 3, 4} <= depths
    assert all(totals[k] for k in totals)
    repeated = generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], repeated[s])
               for k in ("input", "output"))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(17, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Hand-calculated staggered chain: exhausted leader stops short
    # of its rail. Both following vehicles retain fuel after contacting it.
    roles = dict(rails=(1, 3), spent=8, fuel=2, axis=1)
    objs = [_patch(dict(r=r, c=c, h=7, w=5, side=0, q=q), roles)
            for r, c, q in ((1, 5, 1), (7, 11, 7), (13, 17, 15))]
    scene = dict(roles, g=np.empty((21, 30), dtype=np.int8), bg=0, objects=objs)
    settled = _settle(objs, 30)
    assert [o["c"] for o in settled] == [4, 9, 14]
    assert [o["m"] for o in settled] == [1, 2, 3]
    assert [o["c"] for o in _edge_first(objs, 30)] == [4, 9, 2]
    inp, expected = _draw(scene, objs), _draw(scene, settled)
    for turns in range(4):
        rotated_inp, rotated_out = np.rot90(inp, turns), np.rot90(expected, turns)
        assert np.array_equal(solve(rotated_inp), rotated_out)
    permutation = np.array([9, 7, 5, 0, 2, 8, 1, 6, 3, 4], dtype=np.int8)
    assert np.array_equal(solve(permutation[inp]), permutation[expected])
    broken = {s: [{k: g.copy() for k, g in pair.items()} for pair in first[s]] for s in first}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("validator accepted a wrong output")
    return dict(seeds=42, distinct_puzzles=len(signatures), canvas_shapes=len(shapes),
                body_shapes=len(body_shapes), role_palettes=len(palettes),
                axes=sorted(axes), object_counts=sorted(counts), depths=sorted(depths),
                totals=dict(totals), independent_solver=False, blind_prediction=False)
