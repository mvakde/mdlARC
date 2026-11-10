"""Translated repair stencils. Standard library and NumPy only.

No files, seeds, examples, or generation metadata are consulted by solve().
Generation retains one puzzle at a time; self_test writes no files or traces.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import numpy as np


OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0,674779909,417860691,305191827,907377902,632071658,
739037859,287666261,84280190,92937774,898314777,994978835,381949683,
226793452,98825789,288001270,659888050,27535298,69286069,342535350,
999079559,156761145,950680087,912341957,407985759,600236860,781950740,
506030047,446797723,756199962,146708184,636606871,458341918,403318640,
667878322,866763448,733307504,499680470,376180043,503243510,147365660,239598688)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=False)


def _sign(x):
    return int(x > 0) - int(x < 0)


def _segment(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    _require((dr or dc) and (not dr or not dc or abs(dr) == abs(dc)), "unaligned vertices")
    return [(a[0] + k * _sign(dr), a[1] + k * _sign(dc))
            for k in range(max(abs(dr), abs(dc)) + 1)]


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def _intersect(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return _cross(a,b,c)*_cross(a,b,d) <= 0 and _cross(c,d,a)*_cross(c,d,b) <= 0


def _contour(points):
    """Unique noncrossing full cycle, including continuous half-cell crossings."""
    n = len(points)
    _require(4 <= n <= 12 and n == len(set(points)), "need 4..12 distinct corners")
    vertices, edges = set(points), []
    for i, a in enumerate(points):
        for j in range(i+1, n):
            b = points[j]
            dr, dc = abs(a[0]-b[0]), abs(a[1]-b[1])
            if (not dr or not dc or dr == dc) and not (set(_segment(a,b)[1:-1]) & vertices):
                edges.append((i,j))
    adjacent = [[] for _ in points]
    for k, (i,j) in enumerate(edges):
        adjacent[i].append((j,k))
        adjacent[j].append((i,k))
    conflicts = [set() for _ in edges]
    for k, (a,b) in enumerate(edges):
        for q, (c,d) in enumerate(edges[:k]):
            if not ({a,b} & {c,d}) and _intersect(points[a],points[b],points[c],points[d]):
                conflicts[k].add(q)
                conflicts[q].add(k)
    answers = {}

    def visit(path, used, chosen):
        if len(answers) > 1:
            return
        if len(path) == n:
            for neighbor, edge in adjacent[path[-1]]:
                if neighbor == 0 and not (conflicts[edge] & chosen):
                    answers[tuple(sorted(chosen | {edge}))] = path
            return
        for neighbor, edge in adjacent[path[-1]]:
            if neighbor not in used and not (conflicts[edge] & chosen):
                visit(path+[neighbor], used | {neighbor}, chosen | {edge})

    visit([0], {0}, set())
    _require(len(answers) == 1, "stencil has no unique simple full contour")
    selected, order = next(iter(answers.items()))
    pixels = set().union(*(set(_segment(points[edges[k][0]], points[edges[k][1]])) for k in selected))
    rejected = [edges[k] for k in range(len(edges)) if k not in selected]
    return {"pixels": pixels, "edges": [edges[k] for k in selected],
            "rejected": rejected, "order": order}


def _orth_neighbors(mask):
    neighbors = np.zeros_like(mask)
    neighbors[1:] |= mask[:-1]
    neighbors[:-1] |= mask[1:]
    neighbors[:,1:] |= mask[:,:-1]
    neighbors[:,:-1] |= mask[:,1:]
    return neighbors


def _at(mask, p):
    return 0 <= p[0] < mask.shape[0] and 0 <= p[1] < mask.shape[1] and bool(mask[p])


def _add(p, d, k=1):
    return p[0]+k*d[0], p[1]+k*d[1]


def _evidence(mask):
    """All bracketed gaps; cued gaps also need a straight continuation."""
    weak = mask & ~_orth_neighbors(mask)
    gaps = []
    for dr, dc in ((0,1),(1,0)):
        m = mask if dc else mask.T
        starts = m[:,:-4] & m[:,4:] & ~(m[:,1:-3] | m[:,2:-2] | m[:,3:-1])
        for y, x in np.argwhere(starts):
            p = (int(y),int(x)) if dc else (int(x),int(y))
            d = dr, dc
            cells = tuple(_add(p,d,k) for k in (1,2,3))
            matches = []
            if _at(mask,_add(p,d,-1)) or _at(mask,_add(p,d,5)):
                for endpoint in (0,4):
                    anchor = _add(p,d,endpoint)
                    toward = 1 if endpoint == 0 else -1
                    for side in (-1,1):
                        deviation = toward*dr+side*dc, toward*dc-side*dr
                        spur = tuple(_add(anchor,deviation,k) for k in (1,2))
                        if all(_at(weak,q) for q in spur) and not _at(mask,_add(anchor,deviation,3)):
                            matches.append((endpoint,side,spur))
            gaps.append({"cells":cells,"axis":"h" if dc else "v","matches":matches})
    return gaps


def _scraps(mask):
    """Delete only diagonal-linked cells without any orthogonal line neighbor."""
    weak = mask & ~_orth_neighbors(mask)
    erased = np.zeros_like(mask)
    for dc in (-1,1):
        a, b = (weak[:-1,:-1],weak[1:,1:]) if dc == 1 else (weak[:-1,1:],weak[1:,:-1])
        pairs = a & b
        if dc == 1:
            erased[:-1,:-1] |= pairs
            erased[1:,1:] |= pairs
        else:
            erased[:-1,1:] |= pairs
            erased[1:,:-1] |= pairs
    return erased


def _interpret(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    _require(len(colors) == 4 and max(counts) > a.size//2, "need four visible roles and dominant background")
    bg = int(colors[np.argmax(counts)])
    remaining = [int(c) for c in colors if c != bg]
    lines = [c for c in remaining if np.any((a == c) & _orth_neighbors(a == c))]
    _require(len(lines) == 1, "line role must be uniquely orthogonally connected")
    line = lines[0]
    markers = [c for c in remaining if np.count_nonzero(a == c) == 1]
    _require(len(markers) == 1, "need one singleton registration marker")
    marker = markers[0]
    guide = next(c for c in remaining if c not in (line,marker))
    points = [tuple(map(int,p)) for p in np.argwhere(a == guide)]
    p = tuple(map(int,np.argwhere(a == marker)[0]))
    anchors = [q for q in points if max(abs(p[0]-q[0]),abs(p[1]-q[1])) == 1]
    _require(len(anchors) == 1, "ambiguous registration marker")
    offset = p[0]-anchors[0][0], p[1]-anchors[0][1]
    contour = _contour(points)
    stencil = {_add(q,offset) for q in contour["pixels"]}
    _require(all(0 <= r < a.shape[0] and 0 <= c < a.shape[1] for r,c in stencil), "clipped stencil")
    mask = a == line
    gaps = _evidence(mask)
    out = a.copy()
    erased = _scraps(mask)
    out[erased] = bg
    for gap in gaps:
        gap["on"] = all(q in stencil for q in gap["cells"])
        gap["free"] = all(a[q] == bg for q in gap["cells"])
        gap["selected"] = bool(gap["matches"]) and gap["on"] and gap["free"]
        if gap["selected"]:
            for q in gap["cells"]:
                out[q] = marker
    return out, {"background":bg,"line":line,"guide":guide,"marker":marker,
                 "points":points,"offset":offset,"contour":contour,"stencil":stencil,
                 "gaps":gaps,"erased":erased}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the registered stencil, discard scraps, and transfer eligible patches."""
    return _interpret(input_grid)[0]


def _transform(points, code):
    result = []
    for r,c in points:
        if code & 4:
            c = -c
        for _ in range(code % 4):
            r,c = c,-r
        result.append((r,c))
    r0,c0 = min(r for r,c in result),min(c for r,c in result)
    return [(r-r0,c-c0) for r,c in result]


def _shape(rng, kind, extent):
    h,w = int(rng.integers(extent-3,extent+1)), int(rng.integers(extent-3,extent+1))
    if kind == "rectangle":
        if rng.random() < .35:
            h = w
        return [(0,0),(0,w),(h,w),(h,0)]
    if kind == "notch":
        left = int(rng.integers(4,max(5,w-9)))
        right = int(rng.integers(left+5,w-2))
        depth = int(rng.integers(3,max(4,h-5)))
        return [(0,0),(0,left),(depth,left),(depth,right),(0,right),(0,w),(h,w),(h,0)]
    if kind == "double_notch":
        left = int(rng.integers(3,6))
        right = int(rng.integers(left+5,w-2))
        a = int(rng.integers(3,6))
        b = int(rng.integers(a+5,w-2))
        depth = int(rng.integers(3,max(4,h//2-1)))
        other = int(rng.integers(3,max(4,h//2-1)))
        return [(0,0),(0,left),(depth,left),(depth,right),(0,right),(0,w),
                (h,w),(h,b),(h-other,b),(h-other,a),(h,a),(h,0)]
    if kind == "elbow":
        r,c = int(rng.integers(4,h-3)),int(rng.integers(4,w-3))
        return [(0,0),(0,w),(r,w),(r,c),(h,c),(h,0)]
    if kind == "clip":
        k = int(rng.integers(2,5))
        return [(0,k),(0,w-k),(k,w),(h-k,w),(h,w-k),(h,k),(h-k,0),(k,0)]
    # Two independently sized steps on opposing sides.
    r = int(rng.integers(4,h-3))
    k = int(rng.integers(2,4))
    return [(0,0),(0,w-k),(r,w-k),(r,w),(h,w),(h,k),(h-k,k),(h-k,0)]


def _make_registration(rng, kind, extent, code, offset, boundary):
    for _ in range(60):
        points = _transform(_shape(rng,kind,extent),code)
        if any(max(abs(a[0]-b[0]),abs(a[1]-b[1])) < 3
               for i,a in enumerate(points) for b in points[:i]):
            continue
        try:
            contour = _contour(points)
        except ValueError:
            continue
        rejected_orth = [(i,j) for i,j in contour["rejected"]
                         if (points[i][0] == points[j][0] or points[i][1] == points[j][1])
                         and len(_segment(points[i],points[j])) >= 6]
        if kind == "notch" and not rejected_orth:
            continue
        footprint = contour["pixels"] | {_add(p,offset) for p in contour["pixels"]}
        r0,c0 = min(r for r,c in footprint),min(c for r,c in footprint)
        hh,ww = max(r for r,c in footprint)-r0+1,max(c for r,c in footprint)-c0+1
        before = [int(rng.integers(1,4)),int(rng.integers(1,4))]
        after = [int(rng.integers(1,4)),int(rng.integers(1,4))]
        if boundary:
            axis = int(rng.integers(2))
            (before if rng.integers(2) else after)[axis] = 0
        h,w = min(30,hh+before[0]+after[0]),min(30,ww+before[1]+after[1])
        shift = before[0]-r0,before[1]-c0
        points = [_add(p,shift) for p in points]
        contour = _contour(points)
        stencil = {_add(p,offset) for p in contour["pixels"]}
        marker_candidates = [_add(p,offset) for p in points]
        rng.shuffle(marker_candidates)
        return h,w,points,contour,stencil,marker_candidates,rejected_orth
    raise ValueError("registration proposal exhausted")


def _halo(cells):
    return {(r+dr,c+dc) for r,c in cells for dr in (-1,0,1) for dc in (-1,0,1)}


def _candidate_starts(rng, points, contour, stencil, offset, status, force_chord):
    starts = []
    if status == "on" or force_chord:
        for i,j in (contour["rejected"] if force_chord else contour["edges"]):
            a,b = _add(points[i],offset),_add(points[j],offset)
            if a[0] != b[0] and a[1] != b[1]:
                continue
            segment = _segment(a,b)
            d = _sign(b[0]-a[0]),_sign(b[1]-a[1])
            for k in range(1,len(segment)-3):
                cells = [_add(segment[k],d,t) for t in range(3)]
                if all(q in stencil for q in cells) == (status == "on"):
                    starts.append((segment[k],d))
    else:
        r0,r1 = min(r for r,c in stencil),max(r for r,c in stencil)
        c0,c1 = min(c for r,c in stencil),max(c for r,c in stencil)
        for _ in range(180):
            p = int(rng.integers(r0,r1+1)),int(rng.integers(c0,c1+1))
            d = (0,1) if rng.integers(2) else (1,0)
            cells = [_add(p,d,t) for t in range(3)]
            if all(r0 <= r <= r1 and c0 <= c <= c1 for r,c in cells) and not all(q in stencil for q in cells):
                starts.append((p,d))
    rng.shuffle(starts)
    return starts


def _place_track(rng, shape, marks, occupied, starts, cued, stencil, must_shift=False,
                 source_pixels=frozenset()):
    h,w = shape
    blocked = _halo(occupied)
    for g,d in starts[:160]:
        gap = tuple(_add(g,d,k) for k in range(3))
        if must_shift and all(p in source_pixels for p in gap):
            continue
        for _ in range(8):
            left,right = int(rng.integers(2,5)),int(rng.integers(2,5))
            kept = {_add(g,d,-k) for k in range(1,left+1)} | {_add(g,d,k) for k in range(3,3+right)}
            # L/T/cross/branched supports vary independently of the missing span.
            attachments = [_add(g,d,-left),_add(g,d,2+right)]
            rng.shuffle(attachments)
            for anchor in attachments[:int(rng.integers(1,3))]:
                for side in rng.choice([-1,1],size=int(rng.integers(1,3)),replace=False):
                    perp = int(side)*d[1],-int(side)*d[0]
                    kept.update(_add(anchor,perp,k) for k in range(1,int(rng.integers(1,5))+1))
            spur = set()
            if cued:
                endpoint = int(rng.choice([0,4]))
                anchor = _add(g,d,-1 if endpoint == 0 else 3)
                toward = 1 if endpoint == 0 else -1
                side = int(rng.choice([-1,1]))
                dev = toward*d[0]+side*d[1],toward*d[1]-side*d[0]
                spur = {_add(anchor,dev,k) for k in (1,2)}
            footprint = kept | spur | set(gap)
            if (any(not (0 <= r < h and 0 <= c < w) for r,c in footprint)
                    or footprint & (blocked | marks) or kept & set(gap) or spur & (kept | set(gap))):
                continue
            mask = np.zeros(shape,bool)
            for p in kept | spur:
                mask[p] = True
            if any(_orth_neighbors(mask)[p] for p in spur):
                continue
            evidence = _evidence(mask)
            intended = [x for x in evidence if set(x["cells"]) == set(gap)]
            if len(intended) != 1 or bool(intended[0]["matches"]) != cued:
                continue
            if any(x["matches"] for x in evidence if x is not intended[0]):
                continue
            if int(_scraps(mask).sum()) != len(spur):
                continue
            return {"kept":kept,"spur":spur,"gap":gap,"cued":cued,
                    "on":all(p in stencil for p in gap),"axis":"h" if d[1] else "v"}
    return None


def _build_pair(rng, kind, extent, code, offset, boundary, patches, palette, chord_demo, repair_axis=None):
    for _ in range(50):
        try:
            h,w,points,contour,stencil,markers,_ = _make_registration(rng,kind,extent,code,offset,boundary)
        except ValueError:
            continue
        marker = markers[int(rng.integers(len(markers)))]
        marks = set(points) | {marker}
        tracks,occupied = [],set()
        # Place the most constrained contrast first. Positions themselves vary.
        requests = [("off",True,chord_demo)] + [("on",True,False)]*patches + [("on",False,False)]
        for status,cued,force_chord in requests:
            starts = _candidate_starts(rng,points,contour,stencil,offset,status,force_chord)
            if status == "on" and cued and repair_axis:
                starts = [(p,d) for p,d in starts if ("h" if d[1] else "v") == repair_axis]
            track = _place_track(rng,(h,w),marks,occupied,starts,cued,stencil,
                                 must_shift=(status == "on" and cued),source_pixels=contour["pixels"])
            if track is None:
                break
            tracks.append(track)
            occupied.update(track["kept"] | track["spur"] | set(track["gap"]))
        if len(tracks) != len(requests):
            continue
        # Extra tracks and detached scraps are optional, never hidden exceptions.
        for _ in range(int(rng.integers(3))):
            status,cued = str(rng.choice(["on","off"])),bool(rng.integers(2))
            if cued and status == "on":
                continue  # Patch-count shifts stay measurable.
            starts = _candidate_starts(rng,points,contour,stencil,offset,status,False)
            track = _place_track(rng,(h,w),marks,occupied,starts,cued,stencil)
            if track is not None:
                tracks.append(track)
                occupied.update(track["kept"] | track["spur"] | set(track["gap"]))
        loose = set()
        for _ in range(int(rng.integers(4))):
            for _ in range(35):
                p = int(rng.integers(h)),int(rng.integers(w))
                d = int(rng.choice([-1,1])),int(rng.choice([-1,1]))
                cells = {_add(p,d,k) for k in range(int(rng.integers(2,4)))}
                if (all(0 <= r < h and 0 <= c < w for r,c in cells)
                        and not (cells & (marks | _halo(occupied | loose)))):
                    loose.update(cells)
                    break
        bg,guide,line,patch = palette
        inp = np.full((h,w),bg,np.int8)
        for p in points:
            inp[p] = guide
        inp[marker] = patch
        out = inp.copy()
        for track in tracks:
            for p in track["kept"]:
                inp[p] = out[p] = line
            for p in track["spur"]:
                inp[p] = line
            if track["cued"] and track["on"]:
                for p in track["gap"]:
                    out[p] = patch
        for p in loose:
            inp[p] = line
        try:
            expected,info = _interpret(inp)
            if not np.array_equal(expected,out):
                continue
            if len([g for g in info["gaps"] if g["selected"]]) != patches:
                continue
            # Optionally join supported arms into a larger orthogonal network.
            # Accept an added route only when it preserves the complete label,
            # all three contrasts, thinness, and unique local damage evidence.
            for _ in range(int(rng.integers(3))):
                first,second = rng.choice(len(tracks),2,replace=False)
                a = sorted(tracks[int(first)]["kept"])[int(rng.integers(len(tracks[int(first)]["kept"])))]
                b = sorted(tracks[int(second)]["kept"])[int(rng.integers(len(tracks[int(second)]["kept"])))]
                bend = (a[0],b[1]) if rng.integers(2) else (b[0],a[1])
                route = {a,b,bend}
                for p,q in ((a,bend),(bend,b)):
                    if p != q:
                        route.update(_segment(p,q))
                forbidden = marks | loose | set().union(*(x["spur"] | set(x["gap"]) for x in tracks))
                if route & forbidden or len(route) > 27:
                    continue
                trial_in,trial_out = inp.copy(),out.copy()
                for p in route:
                    trial_in[p] = trial_out[p] = line
                mask = trial_out == line
                if np.any(mask[:-1,:-1] & mask[1:,:-1] & mask[:-1,1:] & mask[1:,1:]):
                    continue
                try:
                    _analyze_pair({"input":trial_in,"output":trial_out})
                except ValueError:
                    continue
                inp,out = trial_in,trial_out
            return {"input":inp,"output":out}
        except ValueError:
            continue
    raise ValueError("bounded track placement exhausted")


def _count(name, value, low, high):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _fingerprint(a):
    return a.shape,a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n,t = _count("num_train",num_train,2,10),_count("num_test",num_test,1,4)
    _require(n > t,"num_train must exceed num_test")
    _require(not isinstance(seed,(bool,np.bool_)) and isinstance(seed,(int,np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(["count","scale","boundary"]))
    base_code = int(rng.integers(8))
    fixed_palette = tuple(map(int,rng.choice(10,4,replace=False)))
    varied_palette = bool(rng.integers(2))
    witness = int(rng.integers(n))
    second = (witness+1) % n
    witness_axis = str(rng.choice(["h","v"]))
    puzzle,seen = {"train":[],"test":[]},set()
    for split,count in (("train",n),("test",t)):
        for index in range(count):
            chord_demo = split == "train" and index == witness
            kind = "notch" if chord_demo else str(rng.choice(["rectangle","notch","double_notch","elbow","clip","step"]))
            code = (base_code+(1 if index == second else 0)) % 8 if split == "train" and index in (witness,second) else int(rng.integers(8))
            # The second witness uses an orthogonally rotated registration.
            if split == "train" and index == second:
                kind = "rectangle"
            repair_axis = (witness_axis if index == witness else ("v" if witness_axis == "h" else "h")) if split == "train" and index in (witness,second) else None
            extent = int(rng.integers(17,23))
            if regime == "scale":
                extent = int(rng.integers(16,19) if split == "train" else rng.integers(23,26))
            boundary = regime == "boundary" and split == "test"
            patches = 2 if regime == "count" and split == "test" else 1
            if patches == 2:
                extent = max(21,extent)
            allowed_offsets = [d for d in OFFSETS if not repair_axis
                               or (d[0] != 0 if repair_axis == "h" else d[1] != 0)]
            offset = allowed_offsets[int(rng.integers(len(allowed_offsets)))]
            palette = tuple(map(int,rng.choice(10,4,replace=False))) if varied_palette else fixed_palette
            for retry in range(16):
                try:
                    pair = _build_pair(rng,kind,extent,code,offset,boundary,patches,palette,chord_demo,repair_axis)
                except ValueError:
                    # Geometry is resampled, never the rule or the requested contrasts.
                    extent = min(25,extent+1)
                    continue
                key = _fingerprint(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(mask, diagonal=False):
    cells = set(map(tuple,np.argwhere(mask).tolist()))
    result = []
    steps = OFFSETS if diagonal else ((1,0),(-1,0),(0,1),(0,-1))
    while cells:
        p = min(cells)
        cells.remove(p)
        stack,part = [p],{p}
        while stack:
            p = stack.pop()
            for d in steps:
                q = _add(p,d)
                if q in cells:
                    cells.remove(q)
                    stack.append(q)
                    part.add(q)
        result.append(part)
    return result


def _analyze_pair(pair):
    _require(isinstance(pair,dict) and set(pair) == {"input","output"},"pair schema")
    inp,out = _grid(pair["input"]),_grid(pair["output"])
    expected,info = _interpret(inp)
    _require(inp.shape == out.shape and np.array_equal(expected,out),"incorrect or partial stencil repair")
    selected = [g for g in info["gaps"] if g["selected"]]
    off = [g for g in info["gaps"] if g["matches"] and not g["on"] and g["free"]]
    uncued = [g for g in info["gaps"] if g["on"] and not g["matches"] and g["free"]]
    _require(bool(selected) and bool(off) and bool(uncued),"missing conjunctive contrasts")
    _require(all(len(g["matches"]) == 1 for g in info["gaps"] if g["matches"]),"ambiguous deviation")
    bg,line,marker = info["background"],info["line"],info["marker"]
    _require(np.all(out[inp == info["guide"]] == info["guide"]) and np.all(out[inp == marker] == marker),"registration moved")
    _require(np.any(info["erased"]),"no discarded scraps")
    for cells in _components(info["erased"],True):
        ordered = sorted(cells)
        _require(len(cells) in (2,3) and len({r-c for r,c in cells}) == 1
                 or len(cells) in (2,3) and len({r+c for r,c in cells}) == 1,"non-diagonal scrap")
        _require(all(ordered[i+1][0]-ordered[i][0] == 1 for i in range(len(ordered)-1)),"discontinuous scrap")
    new_patch = (out == marker) & (inp != marker)
    _require(np.count_nonzero(new_patch) == 3*len(selected),"overlapping patches")
    _require(np.all(inp[new_patch] == bg),"patch overwrites existing evidence")
    _require(np.all(out[(inp == line) & ~info["erased"]] == line),"sound linework changed")
    info.update({"selected":selected,"off":off,"uncued":uncued,"shape":inp.shape})
    return info


def _ablations(inp, info):
    """Diagnostic alternatives that omit exactly one gate or registration step."""
    source = info["contour"]["pixels"]
    all_links = source.copy()
    for i,j in info["contour"]["rejected"]:
        all_links.update(_segment(info["points"][i],info["points"][j]))
    all_links = {_add(p,info["offset"]) for p in all_links}
    stencil = info["stencil"]
    r0,r1 = min(r for r,c in stencil),max(r for r,c in stencil)
    c0,c1 = min(c for r,c in stencil),max(c for r,c in stencil)
    alternatives = {}
    for name in ("omit_A","omit_B_cue","untranslated","all_alignments","bounding_box"):
        out = inp.copy()
        out[info["erased"]] = info["background"]
        for gap in info["gaps"]:
            if not gap["free"]:
                continue
            cued = bool(gap["matches"])
            if name == "omit_A":
                use = cued
            elif name == "omit_B_cue":
                use = gap["on"]
            elif name == "untranslated":
                use = cued and all(p in source for p in gap["cells"])
            elif name == "all_alignments":
                use = cued and all(p in all_links for p in gap["cells"])
            else:
                use = cued and all(r0 <= r <= r1 and c0 <= c <= c1 for r,c in gap["cells"])
            if use:
                for p in gap["cells"]:
                    out[p] = info["marker"]
        alternatives[name] = out
    return alternatives


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {"train","test"},"puzzle schema")
    _require(all(isinstance(puzzle[s],list) for s in ("train","test")),"splits must be lists")
    n,t = len(puzzle["train"]),len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= t <= 4 and n > t,"illegal pair counts")
    seen,training = set(),[]
    for split in ("train","test"):
        for pair in puzzle[split]:
            info = _analyze_pair(pair)
            key = _fingerprint(_grid(pair["input"]))
            _require(key not in seen,"duplicate input")
            seen.add(key)
            if split == "train":
                training.append(info)
    # A rejected orthogonal chord must visibly lose to the selected topology.
    chord_witness = False
    untranslated_witness = False
    for info in training:
        rejected_pixels = set()
        for i,j in info["contour"]["rejected"]:
            rejected_pixels.update(_add(p,info["offset"]) for p in _segment(info["points"][i],info["points"][j]))
        chord_witness |= any(all(p in rejected_pixels for p in g["cells"]) for g in info["off"])
        untranslated_witness |= any(not all(p in info["contour"]["pixels"] for p in g["cells"]) for g in info["selected"])
    _require(chord_witness,"training lacks a rejected-chord contrast")
    _require(untranslated_witness,"training does not establish translation")
    _require({g["axis"] for info in training for g in info["selected"]} == {"h","v"},
             "training must show repairs on both axes")
    separated = set()
    for pair,info in zip(puzzle["train"],training):
        separated.update(name for name,alternative in _ablations(_grid(pair["input"]),info).items()
                         if not np.array_equal(alternative,pair["output"]))
    _require(separated == {"omit_A","omit_B_cue","untranslated","all_alignments","bounding_box"},
             "training does not separate plausible shortcuts")


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train","test"):
        for pair in puzzle[split]:
            for role in ("input","output"):
                a = _grid(pair[role])
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 host seeds, streaming diagnostics, no files or retained puzzle corpus."""
    shapes,structures,offsets,palettes,vertices,digests = set(),set(),set(),set(),set(),set()
    axes,counts,features = set(),set(),Counter()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digests.add(_digest(puzzle))
        for split in ("train","test"):
            for pair in puzzle[split]:
                a,b = pair["input"],pair["output"]
                info = _analyze_pair(pair)
                assert np.array_equal(solve(a),b)
                shapes.add(a.shape)
                offsets.add(info["offset"])
                palettes.add(tuple(info[k] for k in ("background","guide","line","marker")))
                vertices.add(len(info["points"]))
                counts.add(len(info["selected"]))
                for name,alternative in _ablations(a,info).items():
                    features[name+"_different_pairs"] += not np.array_equal(alternative,b)
                axes.update(g["axis"] for g in info["selected"])
                normalized = np.full(a.shape,0,np.int8)
                for role,k in enumerate(("guide","line","marker"),1):
                    normalized[a == info[k]] = role
                structures.add(hashlib.sha256(bytes(a.shape)+normalized.tobytes()).hexdigest())
                features["pairs"] += 1
                features["patches"] += len(info["selected"])
                features["preserved_cued"] += len(info["off"])
                features["preserved_uncued"] += len(info["uncued"])
                features["diagonal_sides"] += any(info["points"][i][0] != info["points"][j][0]
                                                  and info["points"][i][1] != info["points"][j][1]
                                                  for i,j in info["contour"]["edges"])
                features["line_components_total"] += len(_components(b == info["line"]))
                for turns,flip in ((1,False),(0,True)):
                    x,y = np.rot90(a,turns),np.rot90(b,turns)
                    if flip:
                        x,y = np.fliplr(x),np.fliplr(y)
                    assert np.array_equal(solve(x),y)
        if seed == HOST_SEEDS[0]:
            assert _digest(puzzle) == _digest(generate_puzzle(seed))
            # Reject an omitted patch, a preserved scrap, a closed intentional
            # gap, and a patched off-stencil gap, even if all other edits agree.
            base = puzzle
            info = _analyze_pair(base["train"][0])
            mutations = [(info["selected"][0]["cells"][0],info["background"]),
                         (tuple(np.argwhere(info["erased"])[0]),info["line"]),
                         (info["uncued"][0]["cells"][0],info["marker"]),
                         (info["off"][0]["cells"][0],info["marker"])]
            for p,color in mutations:
                broken = {s:[{k:v.copy() for k,v in pair.items()} for pair in base[s]] for s in base}
                broken["train"][0]["output"][p] = color
                try:
                    validate_generated_puzzle(broken)
                except ValueError:
                    pass
                else:
                    raise AssertionError("bad label accepted")
    assert len(digests) == 42 and len(structures) >= 190 and len(shapes) >= 35
    assert offsets == set(OFFSETS) and axes == {"h","v"} and counts == {1,2}
    assert len(vertices) >= 3 and len(palettes) >= 35 and features["diagonal_sides"] > 0
    for n in range(2,11):
        for t in range(1,min(4,n-1)+1):
            puzzle = generate_puzzle(918273,n,t)
            assert len(puzzle["train"]) == n and len(puzzle["test"]) == t
    for n,t in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3.0,1),(None,1),(4,None)):
        try:
            generate_puzzle(0,n,t)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds":42,"distinct_puzzles":len(digests),"shapes":len(shapes),
            "normalized_structures":len(structures),"offsets":len(offsets),
            "palettes":len(palettes),"vertex_counts":sorted(vertices),
            "patch_counts":sorted(counts),**features}


if __name__ == "__main__":
    print(json.dumps(self_test(),separators=(",", ":")))
