"""Hinged stamp lineup: composition of 6aa20dc0 (scaled dihedral stamp completion)
and e41c6fd3 (anchor-fixed vertical alignment of copies).

Rule (input-only):
  * The unique three-colour object is a complete n x n prototype stamp (n = 3 or 4).
    Two different colours sit at opposite corners (the endpoints); the third colour
    is the connector.  The endpoint in the prototype's TOP row is the hinge colour;
    the prototype's top row is the hinge line.  The prototype never moves.
  * Every other object is an endpoint-only copy: a solid k x k hinge-colour square
    and a k x k square of the other endpoint colour, diagonally (n-1)k apart.  Their
    relative position fixes the copy's scale k and its orientation.
  * Each copy keeps its columns and slides vertically until its hinge corner lies on
    the hinge line: copies whose hinge square is on top hang below the line (top row
    = line), copies whose hinge square is at the bottom stand on it (bottom row =
    line).  Its connector blocks are restored from the prototype, reflected to match
    the endpoint orientation and enlarged by k.  Old endpoint squares are erased.
"""
from __future__ import annotations

import numpy as np

# Role codes inside a canonical prototype (hinge at top-left).
BG, HINGE, OTHER, CONN = 0, 1, 2, 3
CORNERS = ("TL", "TR", "BL", "BR")


# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------

def orient(roles: np.ndarray, corner: str) -> np.ndarray:
    """Place the canonical (hinge top-left) role grid so the hinge sits at `corner`."""
    if corner == "TL":
        return roles.copy()
    if corner == "TR":
        return roles[:, ::-1].copy()
    if corner == "BL":
        return roles[::-1, :].copy()
    if corner == "BR":
        return roles[::-1, ::-1].copy()
    raise ValueError(corner)


def _components(mask: np.ndarray, eight: bool):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if eight:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                stack, cells = [(r, c)], []
                seen[r, c] = True
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in steps:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                comps.append(cells)
    return comps


def _connected8(roles: np.ndarray) -> bool:
    comps = _components(roles != BG, eight=True)
    return len(comps) == 1


# ---------------------------------------------------------------------------
# Input-only analysis and solver
# ---------------------------------------------------------------------------

def analyze(grid) -> dict:
    """Parse an input grid into prototype, hinge line and endpoint-only copies."""
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")
    h, w = g.shape
    values, counts = np.unique(g, return_counts=True)
    bg = int(values[np.argmax(counts)])

    comps = _components(g != bg, eight=True)
    protos = [cells for cells in comps if len({int(g[y, x]) for y, x in cells}) == 3]
    if len(protos) != 1:
        raise ValueError("expected exactly one three-colour prototype")
    cells = protos[0]
    rows = [y for y, _ in cells]
    cols = [x for _, x in cells]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    n = r1 - r0 + 1
    if n != c1 - c0 + 1 or n < 3:
        raise ValueError("prototype must be a square of side >= 3")
    P = g[r0:r1 + 1, c0:c1 + 1]
    tl, tr, bl, br = int(P[0, 0]), int(P[0, -1]), int(P[-1, 0]), int(P[-1, -1])
    diag_main = tl != bg and br != bg and tl != br
    diag_anti = tr != bg and bl != bg and tr != bl
    if diag_main == diag_anti:
        raise ValueError("prototype endpoint diagonal is ambiguous")
    if diag_main:
        hinge, other, canon = tl, br, P
        proto_corner = "TL"
    else:
        hinge, other, canon = tr, bl, P[:, ::-1]
        proto_corner = "TR"
    colors = {int(v) for v in np.unique(P)} - {bg}
    conn_set = colors - {hinge, other}
    if len(conn_set) != 1:
        raise ValueError("prototype must contain exactly one connector colour")
    conn = conn_set.pop()
    roles = np.zeros((n, n), dtype=int)
    roles[canon == hinge] = HINGE
    roles[canon == other] = OTHER
    roles[canon == conn] = CONN
    if (roles == HINGE).sum() != 1 or (roles == OTHER).sum() != 1:
        raise ValueError("endpoint colours must occur once in the prototype")
    if not np.array_equal(roles, roles.T):
        raise ValueError("prototype connector mask must be symmetric about its endpoint diagonal")
    line = r0

    outside = np.ones_like(g, dtype=bool)
    outside[r0:r1 + 1, c0:c1 + 1] = False
    if np.any((g != bg) & outside & ~np.isin(g, [hinge, other])):
        raise ValueError("stray colour outside prototype")
    squares = {hinge: [], other: []}
    for color in (hinge, other):
        for sq in _components((g == color) & outside, eight=False):
            ys = [y for y, _ in sq]
            xs = [x for _, x in sq]
            y0, x0 = min(ys), min(xs)
            k = max(ys) - y0 + 1
            if max(xs) - x0 + 1 != k or len(sq) != k * k:
                raise ValueError("endpoint marks must be solid squares")
            squares[color].append((y0, x0, k))

    used, copies = set(), []
    for (hy, hx, k) in sorted(squares[hinge]):
        d = (n - 1) * k
        cands = [i for i, (oy, ox, ok) in enumerate(squares[other])
                 if ok == k and abs(oy - hy) == d and abs(ox - hx) == d]
        if len(cands) != 1:
            raise ValueError("hinge square has no unique partner")
        i = cands[0]
        if i in used:
            raise ValueError("endpoint square paired twice")
        used.add(i)
        oy, ox, _ = squares[other][i]
        corner = ("T" if oy > hy else "B") + ("L" if ox > hx else "R")
        top_in = min(hy, oy)
        left = min(hx, ox)
        size = n * k
        top_out = line if corner[0] == "T" else line - size + 1
        copies.append({"k": k, "corner": corner, "left": left, "top_in": top_in,
                       "top_out": top_out, "size": size, "hinge_sq": (hy, hx), "other_sq": (oy, ox)})
    if len(used) != len(squares[other]):
        raise ValueError("unpaired endpoint square")
    if not copies:
        raise ValueError("no incomplete copies")
    return {"bg": bg, "hinge": hinge, "other": other, "conn": conn, "roles": roles, "n": n,
            "line": line, "proto_box": (r0, c0), "proto_corner": proto_corner, "copies": copies,
            "shape": (h, w)}


def render(info: dict, base: np.ndarray) -> np.ndarray:
    """Erase copy endpoint squares and draw completed stamps on the hinge line."""
    out = np.array(base, dtype=int).copy()
    h, w = out.shape
    bg = info["bg"]
    palette = {HINGE: info["hinge"], OTHER: info["other"], CONN: info["conn"]}
    for cp in info["copies"]:
        k = cp["k"]
        for (y, x) in (cp["hinge_sq"], cp["other_sq"]):
            out[y:y + k, x:x + k] = bg
    for cp in info["copies"]:
        k, size = cp["k"], cp["size"]
        stamp = np.kron(orient(info["roles"], cp["corner"]), np.ones((k, k), dtype=int))
        top, left = cp["top_out"], cp["left"]
        if top < 0 or top + size > h or left < 0 or left + size > w:
            raise ValueError("completed stamp leaves the grid")
        region = out[top:top + size, left:left + size]
        if np.any(region != bg):
            raise ValueError("completed stamp collides with another object")
        for role, color in palette.items():
            region[stamp == role] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    g = np.asarray(input_grid, dtype=int)
    return render(analyze(g), g)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _random_mask(rng, n: int) -> np.ndarray:
    """Canonical transpose-symmetric role grid with hinge TL and other endpoint BR."""
    orbits = []
    for i in range(n):
        for j in range(i, n):
            if (i, j) in ((0, 0), (n - 1, n - 1)):
                continue
            orbits.append((i, j))
    for _ in range(400):
        p = rng.uniform(0.3, 0.75)
        roles = np.zeros((n, n), dtype=int)
        roles[0, 0], roles[n - 1, n - 1] = HINGE, OTHER
        for (i, j) in orbits:
            if rng.random() < p:
                roles[i, j] = roles[j, i] = CONN
        nconn = int((roles == CONN).sum())
        if nconn < 2 or nconn > n * n - 3:
            continue
        if _connected8(roles):
            return roles
    roles = np.zeros((n, n), dtype=int)
    roles[0, 0], roles[n - 1, n - 1] = HINGE, OTHER
    for i in range(1, n - 1):
        roles[i, i] = CONN
    roles[0, 1] = roles[1, 0] = CONN
    return roles


def _palette(rng, policy, base):
    if policy == "fixed" and base is not None:
        return base
    if policy == "bg_fixed" and base is not None:
        bg = base[0]
        rest = [c for c in range(10) if c != bg]
        picks = rng.choice(rest, size=3, replace=False)
        return (bg, int(picks[0]), int(picks[1]), int(picks[2]))
    bg = 0 if rng.random() < 0.4 else int(rng.integers(0, 10))
    rest = [c for c in range(10) if c != bg]
    picks = rng.choice(rest, size=3, replace=False)
    return (bg, int(picks[0]), int(picks[1]), int(picks[2]))


def _sample_copies(rng, n, count, scales, force=None):
    """List of (k, corner).  `force` = list of (k_min, vertical) requirements."""
    copies = []
    for req in (force or []):
        kmin, vert = req
        ks = [k for k in scales if k >= kmin] or [max(scales)]
        k = int(rng.choice(ks))
        corner = vert + str(rng.choice(["L", "R"]))
        copies.append((k, corner))
    while len(copies) < count:
        copies.append((int(rng.choice(scales)), str(rng.choice(CORNERS))))
    order = rng.permutation(len(copies))
    return [copies[i] for i in order]


def _layout_pair(rng, spec):
    """Build one (input, output) pair from a spec, or return None if it does not fit."""
    n, roles, proto_corner, copies, pal = (spec["n"], spec["roles"], spec["proto_corner"],
                                           spec["copies"], spec["palette"])
    bg, hc, oc, cc = pal
    sizes = [n * k for k, _ in copies]
    # Horizontal layout: prototype and copies in random left-to-right order.
    items = ["P"] + list(range(len(copies)))
    order = [items[i] for i in rng.permutation(len(items))]
    widths = [n if it == "P" else sizes[it] for it in order]
    gaps = []
    for a, b in zip(order, order[1:]):
        if a == "P" or b == "P":
            gaps.append(int(rng.integers(1, 4)))
        else:
            gaps.append(0 if rng.random() < 0.15 else int(rng.integers(1, 4)))
    lm, rm = int(rng.integers(0, 4)), int(rng.integers(0, 4))
    W = lm + sum(widths) + sum(gaps) + rm
    while W > 30 and any(g > (0 if i else 0) for i, g in enumerate(gaps)):
        # shrink the largest gap (keep prototype gaps >= 1)
        best = None
        for i, gp in enumerate(gaps):
            floor = 1 if "P" in (order[i], order[i + 1]) else 0
            if gp > floor and (best is None or gp > gaps[best]):
                best = i
        if best is None:
            break
        gaps[best] -= 1
        W -= 1
    if W > 30:
        trim = min(lm, W - 30)
        lm -= trim
        W -= trim
        trim = min(rm, W - 30)
        rm -= trim
        W -= trim
    if W > 30:
        return None
    lefts, x = {}, lm
    for it, wd, gp in zip(order, widths, gaps + [0]):
        lefts[it] = x
        x += wd + gp

    stand = [sizes[i] for i, (_, c) in enumerate(copies) if c[0] == "B"]
    hang = [sizes[i] for i, (_, c) in enumerate(copies) if c[0] == "T"]
    above = max(stand) - 1 if stand else 0
    below = max([n] + hang)
    tm, bm = int(rng.integers(0, 4)), int(rng.integers(0, 4))
    slack = int(rng.integers(0, 6))
    H = tm + above + below + bm + slack
    H = max(H, max(sizes) + 2)
    if H > 30:
        H = 30
    if above + below > H:
        return None
    line = int(rng.integers(above, H - below + 1))

    inp = np.full((H, W), bg, dtype=int)
    out = np.full((H, W), bg, dtype=int)
    pal_map = {HINGE: hc, OTHER: oc, CONN: cc}
    proto_roles = orient(roles, proto_corner)
    px = lefts["P"]
    for role, color in pal_map.items():
        inp[line:line + n, px:px + n][proto_roles == role] = color
        out[line:line + n, px:px + n][proto_roles == role] = color

    tops_in, moved = [], 0
    for i, (k, corner) in enumerate(copies):
        size = sizes[i]
        target = line if corner[0] == "T" else line - size + 1
        choices = [t for t in range(0, H - size + 1)]
        if rng.random() < 0.15:
            t = target
        else:
            others = [t for t in choices if t != target]
            if not others:
                t = target
            else:
                t = int(rng.choice(others))
        tops_in.append(t)
        moved += t != target
    if moved == 0:
        return None

    owner = np.full((H, W), -1, dtype=int)
    for i, (k, corner) in enumerate(copies):
        size, left = sizes[i], lefts[i]
        stamp = orient(roles, corner)
        big = np.kron(stamp, np.ones((k, k), dtype=int))
        t_in = tops_in[i]
        for role in (HINGE, OTHER):
            ys, xs = np.nonzero(big == role)
            inp[t_in + ys, left + xs] = pal_map[role]
            owner[t_in + ys, left + xs] = i
        target = line if corner[0] == "T" else line - size + 1
        region = out[target:target + size, left:left + size]
        for role, color in pal_map.items():
            region[big == role] = color
    # Different copies' endpoint marks must not touch (8-neighbourhood).
    for y in range(H):
        for x in range(W):
            if owner[y, x] < 0:
                continue
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and owner[ny, nx] >= 0 and owner[ny, nx] != owner[y, x]:
                        return None
    try:
        if not np.array_equal(solve(inp), out):
            return None
    except ValueError:
        return None
    return {"input": inp, "output": out}


SHIFT_MODES = ("familiar", "new_scale", "more_copies", "proto_flip", "new_size", "all_standing")


def _puzzle_plan(rng):
    plan = {}
    modes = [str(rng.choice(SHIFT_MODES))]
    if rng.random() < 0.3:
        extra = str(rng.choice(SHIFT_MODES))
        if extra not in modes and "familiar" not in (extra, modes[0]):
            modes.append(extra)
    plan["modes"] = modes
    plan["size_policy"] = str(rng.choice(["3", "4", "mixed"], p=[0.45, 0.25, 0.3]))
    if "new_size" in modes:
        plan["size_policy"] = str(rng.choice(["3", "4"]))
    plan["mask_policy"] = str(rng.choice(["same", "per_pair", "two"], p=[0.25, 0.5, 0.25]))
    plan["palette_policy"] = str(rng.choice(["fixed", "varied", "bg_fixed"], p=[0.25, 0.5, 0.25]))
    plan["proto_policy"] = str(rng.choice(["TL", "TR", "mixed"]))
    if "proto_flip" in modes and plan["proto_policy"] == "mixed":
        plan["proto_policy"] = str(rng.choice(["TL", "TR"]))
    return plan


def _max_scale(n):
    return 4 if n == 3 else 3


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not isinstance(num_train, (int, np.integer)) or not isinstance(num_test, (int, np.integer)):
        raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("illegal train/test counts")
    rng = np.random.default_rng(seed)
    for _attempt in range(50):
        puzzle = _try_puzzle(rng, int(num_train), int(num_test))
        if puzzle is not None:
            return puzzle
    raise RuntimeError("could not generate puzzle")


def _try_puzzle(rng, num_train, num_test):
    plan = _puzzle_plan(rng)
    modes = plan["modes"]
    base_n = int(rng.choice([3, 4])) if plan["size_policy"] == "mixed" else int(plan["size_policy"])
    masks = {}

    def mask_for(n, idx):
        key = (n, 0 if plan["mask_policy"] == "same" else
               (idx % 2 if plan["mask_policy"] == "two" else idx))
        if key not in masks:
            masks[key] = _random_mask(rng, n)
        return masks[key]

    base_pal = None
    specs = []
    total = num_train + num_test
    train_max_copies = 3 if "more_copies" in modes else 4
    for idx in range(total):
        is_test = idx >= num_train
        # prototype size
        if plan["size_policy"] == "mixed":
            n = int(rng.choice([3, 4]))
        else:
            n = base_n
        if "new_size" in modes and is_test:
            n = 7 - base_n
        kmax = _max_scale(n)
        train_kmax = kmax - 1 if "new_scale" in modes else kmax
        if is_test and "new_scale" in modes:
            scales = list(range(1, kmax + 1))
        else:
            scales = list(range(1, train_kmax + 1))
        # prototype orientation
        if plan["proto_policy"] == "mixed":
            pc = str(rng.choice(["TL", "TR"]))
        else:
            pc = plan["proto_policy"]
        if is_test and "proto_flip" in modes:
            pc = "TR" if plan["proto_policy"] == "TL" else "TL"
        # copy count and forced features
        force = []
        if not is_test:
            count = int(rng.integers(1, train_max_copies + 1))
            if idx == 0:
                force = [(2, "B")]
                count = max(count, 2)
            elif idx == 1:
                force = [(2, "T"), (1, "B")]
                count = max(count, 2)
        else:
            lo = 2
            hi = 4
            if "more_copies" in modes:
                lo = 4
            count = int(rng.integers(lo, hi + 1))
            if "all_standing" in modes:
                force = [(1, "B")] * count
            else:
                force = [(1, "B"), (1, "T")]
            if "new_scale" in modes:
                force = [(kmax, force[0][1])] + force[1:]
        count = max(count, len(force))
        copies = _sample_copies(rng, n, count, scales, force)
        if "new_scale" in modes and not is_test:
            copies = [(min(k, train_kmax), c) for k, c in copies]
        pal = _palette(rng, plan["palette_policy"], base_pal)
        if base_pal is None:
            base_pal = pal
        specs.append({"n": n, "roles": mask_for(n, idx), "proto_corner": pc,
                      "copies": copies, "palette": pal, "test": is_test})

    pairs = []
    seen = set()
    for spec in specs:
        pair = None
        for _ in range(60):
            cand = _layout_pair(rng, spec)
            if cand is None:
                # relax: drop a random unforced copy if the scene does not fit
                if len(spec["copies"]) > 2 and rng.random() < 0.3:
                    cp = list(spec["copies"])
                    cp.pop(int(rng.integers(len(cp))))
                    spec = dict(spec, copies=cp)
                continue
            key = cand["input"].tobytes() + bytes(cand["input"].shape)
            if key in seen:
                continue
            seen.add(key)
            pair = cand
            break
        if pair is None:
            return None
        pairs.append(pair)
    puzzle = {"train": [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in pairs[:num_train]],
              "test": [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in pairs[num_train:]]}
    try:
        validate_generated_puzzle(puzzle)
    except (AssertionError, ValueError):
        return None
    return puzzle


# ---------------------------------------------------------------------------
# Validation and self test
# ---------------------------------------------------------------------------

ALTERNATIVES = ("no_move", "no_completion", "top_align", "hinge_square_top", "other_corner_line",
                "center_align")


def _alternative_differs(pair, name) -> bool:
    """True if the named simpler rule would produce a different output for this pair."""
    inp = np.asarray(pair["input"], dtype=int)
    out = np.asarray(pair["output"], dtype=int)
    info = analyze(inp)
    n, line = info["n"], info["line"]
    alt = dict(info)
    if name == "no_completion":
        res = inp.copy()
        for cp in info["copies"]:
            k = cp["k"]
            for (y, x) in (cp["hinge_sq"], cp["other_sq"]):
                res[y:y + k, x:x + k] = info["bg"]
        for cp in info["copies"]:
            k, dy = cp["k"], cp["top_out"] - cp["top_in"]
            (hy, hx), (oy, ox) = cp["hinge_sq"], cp["other_sq"]
            for (y, x), col in (((hy, hx), info["hinge"]), ((oy, ox), info["other"])):
                if 0 <= y + dy and y + dy + k <= res.shape[0]:
                    res[y + dy:y + dy + k, x:x + k] = col
        return not np.array_equal(res, out)
    new = []
    for cp in info["copies"]:
        c = dict(cp)
        size, k = cp["size"], cp["k"]
        if name == "no_move":
            c["top_out"] = cp["top_in"]
        elif name == "top_align":
            c["top_out"] = line
        elif name == "hinge_square_top":
            c["top_out"] = line if cp["corner"][0] == "T" else line - size + k
        elif name == "other_corner_line":
            c["top_out"] = (line + n - size) if cp["corner"][0] == "T" else line + n - 1
        elif name == "center_align":
            c["top_out"] = line + (n - size) // 2
        new.append(c)
    alt["copies"] = new
    try:
        res = render(alt, inp)
    except ValueError:
        return True
    return not np.array_equal(res, out)


def validate_generated_puzzle(puzzle) -> dict:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert len(puzzle["train"]) >= 2 and len(puzzle["test"]) >= 1
    seen = set()
    stats = {"train_standing_big": False, "train_hanging_big": False, "train_sides": set(),
             "train_scales": set(), "train_stand": False, "train_hang": False}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert set(pair) == {"input", "output"}
            inp = np.asarray(pair["input"], dtype=int)
            out = np.asarray(pair["output"], dtype=int)
            assert inp.ndim == 2 and inp.shape == out.shape
            assert 1 <= inp.shape[0] <= 30 and 1 <= inp.shape[1] <= 30
            assert inp.min() >= 0 and inp.max() <= 9
            key = inp.tobytes() + bytes(inp.shape)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = analyze(inp)
            assert np.array_equal(render(info, inp), out), "solver mismatch"
            # prototype untouched
            r0, c0 = info["proto_box"]
            n = info["n"]
            assert np.array_equal(inp[r0:r0 + n, c0:c0 + n], out[r0:r0 + n, c0:c0 + n])
            # every copy's hinge corner lies on the hinge line in the output
            moved = 0
            for cp in info["copies"]:
                size = cp["size"]
                hinge_row = cp["top_out"] if cp["corner"][0] == "T" else cp["top_out"] + size - 1
                assert hinge_row == info["line"]
                cols = slice(cp["left"], cp["left"] + size)
                assert np.any(out[hinge_row, cols] == info["hinge"])
                moved += cp["top_in"] != cp["top_out"]
                if split == "train":
                    stats["train_scales"].add(cp["k"])
                    stats["train_sides"].add(cp["corner"][1])
                    if cp["corner"][0] == "B":
                        stats["train_stand"] = True
                        stats["train_standing_big"] |= cp["k"] >= 2
                    else:
                        stats["train_hang"] = True
                        stats["train_hanging_big"] |= cp["k"] >= 2
            assert moved >= 1, "every pair must move at least one copy"
            # column ranges disjoint (lineup)
            spans = sorted([(c0, c0 + n)] + [(cp["left"], cp["left"] + cp["size"]) for cp in info["copies"]])
            for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
                assert a1 <= b0, "column ranges overlap"
            if split == "test":
                assert any(cp["corner"][0] == "B" for cp in info["copies"]), "test needs a standing copy"
    for name in ALTERNATIVES:
        assert any(_alternative_differs(pair, name) for pair in puzzle["train"]), \
            f"training does not refute shortcut {name}"
    assert stats["train_standing_big"], "train must show a standing copy with scale >= 2"
    assert stats["train_hanging_big"], "train must show a hanging copy with scale >= 2"
    assert stats["train_sides"] == {"L", "R"}, "train must show both horizontal endpoint orders"
    assert len(stats["train_scales"]) >= 2, "train must show at least two scales"
    return stats


def self_test() -> bool:
    ns, scales, counts, modes_seen, corners, shapes = set(), set(), set(), set(), set(), set()
    for seed in range(42):
        rng = np.random.default_rng(seed)
        modes_seen.update(_puzzle_plan(rng)["modes"])
        pz = generate_puzzle(seed)
        validate_generated_puzzle(pz)
        for split in ("train", "test"):
            for pair in pz[split]:
                info = analyze(pair["input"])
                ns.add(info["n"])
                counts.add(len(info["copies"]))
                shapes.add(np.asarray(pair["input"]).shape)
                for cp in info["copies"]:
                    scales.add(cp["k"])
                    corners.add(cp["corner"])
                assert np.array_equal(solve(np.asarray(pair["input"])), np.asarray(pair["output"]))
    assert ns == {3, 4}, ns
    assert {1, 2, 3, 4} <= scales, scales
    assert {1, 2, 3, 4} <= counts, counts
    assert set(CORNERS) <= corners
    assert len(modes_seen) == len(SHIFT_MODES), modes_seen
    assert len(shapes) >= 30
    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted {bad}")
    return True


if __name__ == "__main__":
    print(self_test())
