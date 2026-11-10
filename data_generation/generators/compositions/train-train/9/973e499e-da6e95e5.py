"""Glyph-addressed frame masks.

The input is an R x C lattice of equally sized corner-bracket panels (parent B
style).  Each panel has a frame color F[i, j] and a one-color glyph G[i, j]
inside its two-cell inset.  The output is an R*R x C*C grid of R x C tiles
(parent A style).  Tile (i, j) is the whole-lattice support mask of the frame
color named by panel (i, j)'s glyph:

    out[i*R + p, j*C + q] = F[p, q] if F[p, q] == G[i, j] else background

A self-matching panel (G == F, parent B's selection test) therefore lights its
own position inside its own tile; when every panel self-matches the rule
reduces exactly to parent A applied to the frame-color lattice.  Glyph shapes
are decorative; only glyph color is read.  The blank is the input background.
"""

from __future__ import annotations

import json

import numpy as np


# --------------------------------------------------------------------------
# counts and basic helpers
# --------------------------------------------------------------------------

def _counts(num_train, num_test):
    if num_train is None:
        num_train = 4
    if num_test is None:
        num_test = 1
    for name, value in (("num_train", num_train), ("num_test", num_test)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    num_train, num_test = int(num_train), int(num_test)
    if not 2 <= num_train <= 10:
        raise ValueError("num_train must be in 2..10")
    if not 1 <= num_test <= 4:
        raise ValueError("num_test must be in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _frame_mask(height, width, arm):
    mask = np.zeros((height, width), dtype=bool)
    for d in range(arm):
        for r, c, dr, dc in ((0, 0, 1, 1), (0, width - 1, 1, -1),
                             (height - 1, 0, -1, 1), (height - 1, width - 1, -1, -1)):
            mask[r + dr * d, c] = True
            mask[r, c + dc * d] = True
    return mask


def _key(grid):
    return json.dumps(np.asarray(grid).astype(int).tolist(), separators=(",", ":"))


# --------------------------------------------------------------------------
# the rule (shared by solve, validation and alternative-rule checks)
# --------------------------------------------------------------------------

def _construct(frames, glyphs, background):
    frames = np.asarray(frames)
    glyphs = np.asarray(glyphs)
    rows, cols = frames.shape
    out = np.full((rows * rows, cols * cols), background, dtype=int)
    for i in range(rows):
        for j in range(cols):
            match = frames == glyphs[i, j]
            tile = np.where(match, frames, background)
            out[i * rows:(i + 1) * rows, j * cols:(j + 1) * cols] = tile
    return out


def _alternatives(frames, glyphs, background):
    """Outputs of plausible shortcut rules (used to prove the demos discriminate)."""
    frames = np.asarray(frames)
    glyphs = np.asarray(glyphs)
    rows, cols = frames.shape
    alt = {
        # parent A on the frame lattice only (ignores glyphs)
        "frames_only": _construct(frames, frames, background),
        # parent A on the glyph lattice only (ignores frames)
        "glyphs_only": _construct(glyphs, glyphs, background),
    }
    # reversed addressing: frame color selects the glyph-color mask
    inv = np.full((rows * rows, cols * cols), background, dtype=int)
    for i in range(rows):
        for j in range(cols):
            inv[i * rows:(i + 1) * rows, j * cols:(j + 1) * cols] = np.where(
                glyphs == frames[i, j], glyphs, background)
    alt["reversed"] = inv
    # parent B only: just the self-matching panels contribute their own frame mask
    selfish = np.full((rows * rows, cols * cols), background, dtype=int)
    for i in range(rows):
        for j in range(cols):
            if frames[i, j] == glyphs[i, j]:
                selfish[i * rows:(i + 1) * rows, j * cols:(j + 1) * cols] = np.where(
                    frames == frames[i, j], frames, background)
    alt["self_only"] = selfish
    return alt


# --------------------------------------------------------------------------
# parsing (input only)
# --------------------------------------------------------------------------

def _corner_candidates(grid, background):
    """Find L-shaped one-color 4-connected components and classify them."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    corners = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == background:
                continue
            color = grid[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and grid[ny, nx] == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            n = len(cells)
            if n < 3 or n % 2 == 0:
                continue
            arm = (n + 1) // 2
            ys = [y for y, _ in cells]
            xs = [x for _, x in cells]
            y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
            if y1 - y0 + 1 != arm or x1 - x0 + 1 != arm:
                continue
            cellset = set(cells)
            for kind, (ey, ex) in (("tl", (y0, x0)), ("tr", (y0, x1)),
                                   ("bl", (y1, x0)), ("br", (y1, x1))):
                expected = {(ey, x) for x in range(x0, x1 + 1)} | {(y, ex) for y in range(y0, y1 + 1)}
                if expected == cellset:
                    corners.append((kind, ey, ex, arm, int(color)))
                    break
    return corners


def _parse(grid):
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2:
        raise ValueError("grid must be 2D")
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    corners = _corner_candidates(grid, background)
    lookup = {(kind, y, x): (arm, color) for kind, y, x, arm, color in corners}
    panels = []
    for kind, top, left, arm, color in corners:
        if kind != "tl":
            continue
        for (k2, y2, x2), (arm2, color2) in lookup.items():
            if k2 != "br" or arm2 != arm or color2 != color:
                continue
            height, width = y2 - top + 1, x2 - left + 1
            if height < 5 or width < 5 or 2 * arm >= min(height, width):
                continue
            if lookup.get(("tr", top, x2)) != (arm, color) or lookup.get(("bl", y2, left)) != (arm, color):
                continue
            box = grid[top:y2 + 1, left:x2 + 1]
            frame = _frame_mask(height, width, arm)
            inset = np.zeros_like(frame)
            inset[2:height - 2, 2:width - 2] = True
            if np.any(box[frame] != color):
                continue
            if np.any(box[~frame & ~inset] != background):
                continue
            payload = box[2:height - 2, 2:width - 2]
            glyph_colors = set(np.unique(payload).tolist()) - {background}
            if len(glyph_colors) != 1:
                continue
            panels.append((top, left, height, width, arm, color, glyph_colors.pop()))
    if not panels:
        raise ValueError("no bracket panels found")
    sizes = {(p[2], p[3]) for p in panels}
    if len(sizes) != 1:
        raise ValueError("panels must share one size")
    tops = sorted({p[0] for p in panels})
    lefts = sorted({p[1] for p in panels})
    rows, cols = len(tops), len(lefts)
    if rows * cols != len(panels):
        raise ValueError("panels do not form a full lattice")
    frames = np.zeros((rows, cols), dtype=int)
    glyphs = np.zeros((rows, cols), dtype=int)
    filled = np.zeros((rows, cols), dtype=bool)
    for top, left, _, _, _, color, glyph in panels:
        i, j = tops.index(top), lefts.index(left)
        if filled[i, j]:
            raise ValueError("overlapping panels")
        filled[i, j] = True
        frames[i, j] = color
        glyphs[i, j] = glyph
    if not filled.all():
        raise ValueError("incomplete lattice")
    return {"background": background, "frames": frames, "glyphs": glyphs,
            "panels": panels, "tops": tops, "lefts": lefts}


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    return _construct(info["frames"], info["glyphs"], info["background"])


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

def _glyph_mask(rng, height, width, style):
    rows, cols = np.indices((height, width))
    if style == "dot" or height * width == 1:
        mask = np.zeros((height, width), dtype=bool)
        if rng.random() < 0.6:
            mask[height // 2, width // 2] = True
        else:
            mask.flat[int(rng.integers(0, height * width))] = True
        return mask
    if style == "solid":
        return np.ones((height, width), dtype=bool)
    mode = int(rng.integers(0, 7))
    if mode == 0:
        mask = rng.random((height, width)) < rng.uniform(.3, .8)
    elif mode == 1:
        mask = (rows + cols + int(rng.integers(0, 2))) % 2 == 0
    elif mode == 2:
        mask = (rows == 0) | (rows == height - 1) | (cols == 0) | (cols == width - 1)
    elif mode == 3:
        mask = (rows == int(rng.integers(0, height))) | (cols == int(rng.integers(0, width)))
    elif mode == 4:
        mask = np.zeros((height, width), dtype=bool)
        mask[int(rng.integers(0, height)):, int(rng.integers(0, width)):] = True
    elif mode == 5:
        mask = np.ones((height, width), dtype=bool)
        holes = int(rng.integers(1, max(2, height * width // 2)))
        mask.flat[rng.choice(height * width, size=holes, replace=False)] = False
    else:
        mask = np.ones((height, width), dtype=bool)
    if not mask.any():
        mask.flat[int(rng.integers(0, height * width))] = True
    if rng.integers(0, 2):
        mask = np.flipud(mask)
    if rng.integers(0, 2):
        mask = np.fliplr(mask)
    return mask


def _category_lattice(rng, rows, cols, k):
    """Categorical R x C lattice using exactly k categories (indices 0..k-1)."""
    for _ in range(60):
        mode = str(rng.choice(["scatter", "scatter", "bands", "modular", "blocks", "mirror"]))
        if mode == "scatter":
            lat = rng.integers(0, k, size=(rows, cols))
        elif mode == "bands":
            if rng.random() < 0.5:
                order = rng.integers(0, k, size=rows)
                lat = np.repeat(order[:, None], cols, axis=1)
            else:
                order = rng.integers(0, k, size=cols)
                lat = np.repeat(order[None, :], rows, axis=0)
            # sprinkle a few exceptions so bands are not the whole story
            if rng.random() < 0.5:
                lat = lat.copy()
                lat[int(rng.integers(0, rows)), int(rng.integers(0, cols))] = int(rng.integers(0, k))
        elif mode == "modular":
            a, b = int(rng.integers(1, 3)), int(rng.integers(0, 3))
            ii, jj = np.indices((rows, cols))
            lat = (a * ii + b * jj + int(rng.integers(0, k))) % k
        elif mode == "blocks":
            sr, sc = int(rng.integers(1, rows + 1)), int(rng.integers(1, cols + 1))
            lat = np.zeros((rows, cols), dtype=int)
            lat[:sr, sc:] = 1 % k
            lat[sr:, :sc] = 2 % k
            lat[sr:, sc:] = int(rng.integers(0, k))
            if k > 3:
                for _ in range(k):
                    lat[int(rng.integers(0, rows)), int(rng.integers(0, cols))] = int(rng.integers(0, k))
        else:
            half = rng.integers(0, k, size=(rows, (cols + 1) // 2))
            lat = np.concatenate([half, half[:, : cols // 2][:, ::-1]], axis=1)
            if rng.random() < 0.5:
                lat = lat.copy()
                lat[int(rng.integers(0, rows)), int(rng.integers(0, cols))] = int(rng.integers(0, k))
        lat = np.asarray(lat, dtype=int)
        if len(np.unique(lat)) == k:
            return lat
    lat = rng.integers(0, k, size=(rows, cols))
    flat = lat.reshape(-1)
    flat[rng.choice(rows * cols, size=k, replace=False)] = np.arange(k)
    return flat.reshape(rows, cols)


def _spaces(rng, minimums, extra):
    minimums = np.asarray(minimums, dtype=int)
    if extra <= 0:
        return minimums
    return minimums + rng.multinomial(extra, np.full(len(minimums), 1 / len(minimums)))


def _payload_options(rows, cols):
    out = []
    for ph in range(1, 5):
        for pw in range(1, 5):
            if rows * (ph + 4) + (rows - 1) <= 30 and cols * (pw + 4) + (cols - 1) <= 30:
                out.append((ph, pw))
    return out


def _render(rng, frames, glyphs, background, payload, arm, style):
    rows, cols = frames.shape
    ph, pw = payload
    ph_out, pw_out = ph + 4, pw + 4
    min_h = rows * ph_out + (rows - 1)
    min_w = cols * pw_out + (cols - 1)
    spacious = rng.random() < 0.5
    extra_h = int(rng.integers(0, 30 - min_h + 1)) if spacious else int(rng.integers(0, min(3, 30 - min_h) + 1))
    extra_w = int(rng.integers(0, 30 - min_w + 1)) if spacious else int(rng.integers(0, min(3, 30 - min_w) + 1))
    v = _spaces(rng, [0] + [1] * (rows - 1) + [0], extra_h)
    hsp = _spaces(rng, [0] + [1] * (cols - 1) + [0], extra_w)
    height, width = min_h + extra_h, min_w + extra_w
    grid = np.full((height, width), background, dtype=int)
    frame = _frame_mask(ph_out, pw_out, arm)
    tops, lefts = [], []
    cursor = int(v[0])
    for i in range(rows):
        tops.append(cursor)
        cursor += ph_out + int(v[i + 1])
    cursor = int(hsp[0])
    for j in range(cols):
        lefts.append(cursor)
        cursor += pw_out + int(hsp[j + 1])
    for i in range(rows):
        for j in range(cols):
            t, l = tops[i], lefts[j]
            box = grid[t:t + ph_out, l:l + pw_out]
            box[frame] = frames[i, j]
            mask = _glyph_mask(rng, ph, pw, style)
            inset = box[2:2 + ph, 2:2 + pw]
            inset[mask] = glyphs[i, j]
    return grid


def _make_pair(rng, spec):
    rows, cols, k = spec["rows"], spec["cols"], spec["k"]
    background = spec["background"]
    palette = [c for c in range(10) if c != background]
    rng.shuffle(palette)
    frame_colors = palette[:k]
    orphan_pool = palette[k:]
    cats = _category_lattice(rng, rows, cols, k)
    frames = np.asarray(frame_colors, dtype=int)[cats]
    p_self = spec["p_self"]
    p_orphan = 0.18 if spec["orphan"] else 0.0
    for _ in range(200):
        glyphs = frames.copy()
        for i in range(rows):
            for j in range(cols):
                u = rng.random()
                if u < p_orphan:
                    glyphs[i, j] = int(orphan_pool[int(rng.integers(0, min(2, len(orphan_pool))))])
                elif u < p_orphan + p_self:
                    glyphs[i, j] = frames[i, j]
                else:
                    others = [c for c in frame_colors if c != frames[i, j]]
                    glyphs[i, j] = int(others[int(rng.integers(0, len(others)))])
        if spec["orphan"] and not np.any(~np.isin(glyphs, frame_colors)):
            i, j = int(rng.integers(0, rows)), int(rng.integers(0, cols))
            glyphs[i, j] = int(orphan_pool[0])
        if spec["need_self"] and not np.any(glyphs == frames):
            continue
        cross = (glyphs != frames) & np.isin(glyphs, frame_colors)
        if not cross.any():
            continue
        out = _construct(frames, glyphs, background)
        alts = _alternatives(frames, glyphs, background)
        if any(np.array_equal(out, a) for name, a in alts.items()
               if name in ("frames_only", "glyphs_only", "reversed")):
            continue
        break
    else:
        return None
    payload, arm, style = spec["payload"], spec["arm"], spec["style"]
    grid = _render(rng, frames, glyphs, background, payload, arm, style)
    return {"input": grid, "output": _construct(frames, glyphs, background)}


def _pair_spec(rng, plan, split, index):
    small = [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4)]
    squares_all = [(2, 2), (3, 3), (4, 4), (5, 5)]
    rects_all = [(2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3), (3, 5), (5, 3), (4, 5), (5, 4), (2, 5)]
    if split == "train":
        if index == 0:
            rows, cols = (3, 3) if rng.random() < 0.6 else ((3, 4) if plan["rect"] and rng.random() < 0.5 else (4, 4))
        elif index == 1 and plan["rect"]:
            rows, cols = [(2, 3), (3, 2), (3, 4), (4, 3), (2, 4)][int(rng.integers(0, 5))]
        elif plan["regime"] == "grow":
            pool = small if plan["rect"] else [s for s in small if s[0] == s[1]] + [(3, 3)]
            rows, cols = pool[int(rng.integers(0, len(pool)))]
        else:
            pool = squares_all + (rects_all if plan["rect"] else [])
            rows, cols = pool[int(rng.integers(0, len(pool)))]
    else:
        if plan["regime"] == "grow":
            pool = [(4, 4), (5, 5), (5, 5)] + ([(4, 5), (5, 4), (3, 5), (5, 3)] if plan["test_rect"] else [])
            if plan["test_rect"] and index == 0:
                pool = [(4, 5), (5, 4), (3, 5), (5, 3)]
        else:
            pool = squares_all[1:] + (rects_all if plan["test_rect"] else [])
            if plan["test_rect"] and index == 0:
                pool = rects_all
        rows, cols = pool[int(rng.integers(0, len(pool)))]
    cells = rows * cols
    kmax = min(6, cells - 1, 8)
    if split == "train" and index == 0:
        k = int(rng.integers(2, min(3, kmax) + 1))
    elif split == "train" and index == 1:
        k = int(rng.integers(3, max(3, min(5, kmax)) + 1)) if kmax >= 3 else 2
    else:
        k = int(rng.integers(2, max(2, min(5, kmax)) + 1))
        if split == "test" and plan["regime"] == "grow" and kmax >= 4 and rng.random() < 0.5:
            k = int(rng.integers(4, min(6, kmax) + 1))
    options = _payload_options(rows, cols)
    payload = options[int(rng.integers(0, len(options)))]
    arm = 3 if min(payload) >= 3 and rng.random() < 0.5 else 2
    style = str(rng.choice(["dot", "shape", "shape", "solid"])) if max(payload) > 1 else "dot"
    if plan["bg_fixed"] is not None:
        background = plan["bg_fixed"]
    else:
        background = int(rng.integers(0, 10)) if rng.random() < 0.7 else 0
    if split == "test":
        orphan = plan["test_orphan"] if index == 0 else (plan["orphan"] and rng.random() < 0.5)
    else:
        orphan = plan["orphan"] and (index == plan["orphan_train_index"] or rng.random() < 0.35)
    return {"rows": rows, "cols": cols, "k": k, "payload": payload, "arm": arm, "style": style,
            "background": background, "orphan": orphan,
            "p_self": float(rng.uniform(0.15, 0.5)),
            "need_self": split == "train" and index == 0}


def _plan(rng, num_train):
    orphan = rng.random() < 0.5
    rect = rng.random() < 0.45
    bg_fixed = None
    if rng.random() < 0.45:
        bg_fixed = 0 if rng.random() < 0.4 else int(rng.integers(1, 10))
    return {
        "orphan": orphan,
        "test_orphan": orphan and rng.random() < 0.6,
        "orphan_train_index": int(rng.integers(0, num_train)),
        "rect": rect,
        "test_rect": rect and rng.random() < 0.6,
        "regime": "grow" if rng.random() < 0.55 else "mixed",
        "bg_fixed": bg_fixed,
    }


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(50):
        plan = _plan(rng, num_train)
        pairs, keys = {"train": [], "test": []}, set()
        ok = True
        for split, count in (("train", num_train), ("test", num_test)):
            for index in range(count):
                pair = None
                for _ in range(40):
                    spec = _pair_spec(rng, plan, split, index)
                    pair = _make_pair(rng, spec)
                    if pair is None:
                        continue
                    key = _key(pair["input"])
                    if key in keys:
                        pair = None
                        continue
                    try:
                        if not np.array_equal(solve(pair["input"]), pair["output"]):
                            pair = None
                            continue
                    except ValueError:
                        pair = None
                        continue
                    keys.add(key)
                    break
                if pair is None:
                    ok = False
                    break
                pairs[split].append({"input": pair["input"].tolist(), "output": pair["output"].tolist()})
            if not ok:
                break
        if not ok:
            continue
        try:
            validate_generated_puzzle(pairs)
        except AssertionError:
            continue
        return pairs
    raise RuntimeError(f"could not generate a valid puzzle for seed {seed}")


# --------------------------------------------------------------------------
# validation and self test
# --------------------------------------------------------------------------

def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _pair_facts(pair):
    grid = np.asarray(pair["input"], dtype=int)
    out = np.asarray(pair["output"], dtype=int)
    info = _parse(grid)
    frames, glyphs, bg = info["frames"], info["glyphs"], info["background"]
    rows, cols = frames.shape
    _require(2 <= rows <= 5 and 2 <= cols <= 5, "lattice must be 2..5 per side")
    _require(out.shape == (rows * rows, cols * cols), "output must be R^2 x C^2")
    _require(np.array_equal(out, _construct(frames, glyphs, bg)), "output disagrees with rule")
    frame_colors = set(np.unique(frames).tolist())
    _require(len(frame_colors) >= 2, "need at least two frame categories")
    _require(bg not in frame_colors and bg not in set(np.unique(glyphs).tolist()), "background reused")
    sizes = {(p[2], p[3]) for p in info["panels"]}
    arms = {p[4] for p in info["panels"]}
    _require(len(sizes) == 1 and len(arms) == 1, "panels must share size and arm")
    alts = _alternatives(frames, glyphs, bg)
    orphan = bool(np.any(~np.isin(glyphs, list(frame_colors))))
    cross = bool(np.any((glyphs != frames) & np.isin(glyphs, list(frame_colors))))
    return {
        "rows": rows, "cols": cols, "k": len(frame_colors), "bg": bg,
        "self": int(np.sum(glyphs == frames)), "cross": cross, "orphan": orphan,
        "rect": rows != cols, "payload": tuple(d - 4 for d in sizes.pop()), "arm": arms.pop(),
        "differs": {n: not np.array_equal(out, a) for n, a in alts.items()},
        "shape": grid.shape,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "bad counts")
    keys = set()
    facts = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(set(pair) == {"input", "output"}, "bad pair keys")
            for role in ("input", "output"):
                arr = np.asarray(pair[role])
                _require(arr.ndim == 2 and 1 <= arr.shape[0] <= 30 and 1 <= arr.shape[1] <= 30, "bad dims")
                _require(arr.min() >= 0 and arr.max() <= 9, "bad colors")
            key = _key(pair["input"])
            _require(key not in keys, "duplicate input")
            keys.add(key)
            f = _pair_facts(pair)
            _require(np.array_equal(solve(np.asarray(pair["input"])), np.asarray(pair["output"])), "solve mismatch")
            facts[split].append(f)
    tr = facts["train"]
    # both concepts must be visible in training
    _require(any(f["self"] > 0 for f in tr), "training needs a self-matching panel")
    _require(all(f["cross"] for f in tr), "every training pair needs a cross-addressed panel")
    _require(any(f["k"] >= 3 for f in tr), "training needs a pair with >=3 frame categories")
    for name in ("frames_only", "glyphs_only", "reversed"):
        _require(all(f["differs"][name] for f in tr), f"training pair matches shortcut {name}")
        _require(all(f["differs"][name] for f in facts["test"]), f"test pair matches shortcut {name}")
    _require(any(f["differs"]["self_only"] for f in tr), "training does not refute self-only rule")
    for f in facts["test"]:
        if f["orphan"]:
            _require(any(g["orphan"] for g in tr), "test orphan glyph without training orphan")
        if f["rect"]:
            _require(any(g["rect"] for g in tr), "rectangular test without rectangular training")


def self_test() -> None:
    seeds = list(range(42))
    lattices, payloads, ks, bgs, shapes = set(), set(), set(), set(), set()
    orphan_puzzles = rect_puzzles = grow = 0
    signatures = set()
    for seed in seeds:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        again = generate_puzzle(seed)
        _require(json.dumps(again) == json.dumps(puzzle), "nondeterministic")
        signatures.add(json.dumps(puzzle))
        any_orphan = any_rect = False
        tr_cells = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(np.asarray(pair["input"]))
                fr = info["frames"]
                lattices.add(fr.shape)
                h, w = info["panels"][0][2], info["panels"][0][3]
                payloads.add((h - 4, w - 4))
                ks.add(len(np.unique(fr)))
                bgs.add(info["background"])
                shapes.add(np.asarray(pair["input"]).shape)
                if not np.all(np.isin(info["glyphs"], fr)):
                    any_orphan = True
                if fr.shape[0] != fr.shape[1]:
                    any_rect = True
                if split == "train":
                    tr_cells.append(fr.size)
                elif fr.size > max(tr_cells):
                    grow += 1
        orphan_puzzles += any_orphan
        rect_puzzles += any_rect
    _require(len(signatures) == len(seeds), "duplicate puzzles across seeds")
    _require(len(lattices) >= 8, f"too few lattice shapes {lattices}")
    _require(len(payloads) >= 6, f"too few payload sizes {payloads}")
    _require(ks >= {2, 3, 4}, f"category counts too narrow {ks}")
    _require(len(bgs) >= 6, f"too few backgrounds {bgs}")
    _require(orphan_puzzles >= 8 and rect_puzzles >= 8, "orphan/rect regimes too rare")
    _require(grow >= 8, "too few test pairs larger than all training lattices")
    _require(len(shapes) >= 20, "too few canvas shapes")
    for a, b in ((1, 1), (11, 1), (4, 0), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test ok")
