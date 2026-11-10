"""Assemble a marked categorical square, then expand its color-equality masks.

Only the standard library and NumPy are used. No parent module, saved example,
generation metadata, or random state is consulted by solve().
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


HOST_SEEDS = (0,627799099,667405986,748487631,809292723,841710441,902681290,486282094,614308896,964764805,422496506,14345868,720428526,884997649,438961598,279764705,146066772,276522700,240216900,284704701,21764748,656942072,936912012,655657348,178570719,5585270,271973334,15450656,608009289,923757870,745677731,292823774,202655672,440008498,358030815,629554280,101218880,147158528,807444202,713152599,644761159,388339639)


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "grid cells must be integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _neighbors(cell):
    r, c = cell
    return ((r-1,c),(r+1,c),(r,c-1),(r,c+1))


def _regions(cells):
    remaining = set(cells)
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        todo, found = [first], {first}
        while todo:
            for p in _neighbors(todo.pop()):
                if p in remaining:
                    remaining.remove(p)
                    found.add(p)
                    todo.append(p)
        result.append(found)
    return result


def _normal(cells):
    a = np.asarray(sorted(map(tuple,cells)), dtype=int)
    return a - a.min(axis=0)


def _scene(value):
    grid = _grid(value)
    markers = np.argwhere(grid == 9)
    _require(len(markers) == 4, "four marker corners required")
    r1, c1 = markers.min(axis=0)
    r2, c2 = markers.max(axis=0)
    n = int(r2-r1-1)
    _require(n in (3,4,5) and c2-c1-1 == n, "corners must delimit a 3..5 square")
    _require(set(map(tuple, markers)) == {(r1,c1),(r1,c2),(r2,c1),(r2,c2)}, "markers must be four rectangle corners")
    r0, c0 = int(r1+1), int(c1+1)
    bands = []
    for color in range(1,9):
        for axis in (0,1):
            if np.any(np.all(grid == color, axis=1-axis)):
                bands.append((color,axis))
    _require(len(bands) == 1, "one spanning band with a solid core is required")
    band, axis = bands[0]  # axis 0: horizontal, axis 1: vertical
    band_cells = set(map(tuple, np.argwhere(grid == band)))
    _require(len(_regions(band_cells)) == 1, "band must be connected")
    square = grid[r0:r0+n,c0:c0+n].copy()
    _require(np.all((square == 0) | (square == band)), "window must initially contain only band and gaps")
    _require(np.any(square == band) and np.any(square == 0), "window must intersect the band and contain gaps")
    canonical = square if axis == 0 else square.T
    solid = np.flatnonzero(np.all(canonical == band, axis=1))
    _require(len(solid) >= 1, "band core must cross the marked square")
    for c in range(n):
        rows = np.flatnonzero(canonical[:,c] == band)
        _require(np.array_equal(rows, np.arange(rows[0],rows[-1]+1)), "band columns must be intervals")
    outside = grid.copy()
    outside[r0:r0+n,c0:c0+n] = 0
    # Fixed geometry outside the square consists only of straight core lines.
    core = np.zeros_like(grid, dtype=bool)
    if axis == 0:
        core[r0+solid,:] = True
    else:
        core[:,c0+solid] = True
    outside_band = outside == band
    window_mask = np.zeros_like(core)
    window_mask[r0:r0+n,c0:c0+n] = True
    _require(np.array_equal(outside_band, core & ~window_mask), "band teeth must be inside the marked square")
    pieces = []
    occupied = set()
    for color in range(1,9):
        if color == band:
            continue
        for component in _regions(map(tuple,np.argwhere(grid == color))):
            _require(all(not (r1 <= r <= r2 and c1 <= c <= c2) for r,c in component), "pieces must start outside the marked well")
            _require(len(component) >= 2, "pieces must have at least two cells")
            for r,c in component:
                for dr in (-1,0,1):
                    for dc in (-1,0,1):
                        rr, cc = r+dr, c+dc
                        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                            _require(grid[rr,cc] not in (band,9), "piece touches fixed geometry")
                            _require((rr,cc) not in occupied, "pieces must be separated")
            occupied.update(component)
            pieces.append({"color":color,"source":np.array(sorted(component),dtype=int),"shape":_normal(component)})
    _require(2 <= len(pieces) <= 5, "expected 2..5 detached pieces")
    _require(sum(len(p["shape"]) for p in pieces) == np.count_nonzero(square == 0), "piece area must equal socket area")
    return grid, square, pieces, {"n":n,"origin":(r0,c0),"band":band,"axis":axis}


def _pack(square, pieces):
    """Enumerate every translation into the gaps; deduplicate colored covers.

    This is a general small-square exact cover, including alternative groupings,
    alignments and side-by-side placements. No intended cut or order is used.
    The search stops after two different colored completions, never after a
    fixed node budget. At most 25 board cells and five pieces are admitted.
    """
    n = len(square)
    hole_cells = [int(r*n+c) for r,c in np.argwhere(square == 0)]
    full = sum(1 << cell for cell in hole_cells)
    placements, by_cell, by_piece = [], {cell:[] for cell in hole_cells}, [[] for _ in pieces]
    for i,piece in enumerate(pieces):
        shape = piece["shape"]
        h,w = shape.max(axis=0)+1
        for r in range(n-int(h)+1):
            for c in range(n-int(w)+1):
                coords = shape + (r,c)
                if np.all(square[coords[:,0],coords[:,1]] == 0):
                    mask = sum(1 << int(rr*n+cc) for rr,cc in coords)
                    entry = (i,mask,coords)
                    j = len(placements)
                    placements.append(entry)
                    by_piece[i].append(j)
                    for rr,cc in coords:
                        by_cell[int(rr*n+cc)].append(j)
    answers = {}
    nodes = 0
    signatures = [(p["color"],tuple(map(tuple,p["shape"]))) for p in pieces]

    def search(used, filled, chosen):
        nonlocal nodes
        nodes += 1
        if len(answers) >= 2:
            return
        if filled == full:
            if used != (1 << len(pieces))-1:
                return
            result = square.copy()
            for j in chosen:
                i,_,coords = placements[j]
                result[coords[:,0],coords[:,1]] = pieces[i]["color"]
            answers.setdefault(result.tobytes(),(result,tuple(placements[j] for j in chosen)))
            return
        # A placement is available only if its component is unused and it
        # overlaps neither fixed band nor an already placed component.
        def available(ids):
            return [j for j in ids if not used & (1 << placements[j][0]) and not filled & placements[j][1]]
        choices = None
        for cell in hole_cells:
            if not filled & (1 << cell):
                options = available(by_cell[cell])
                if not options:
                    return
                if choices is None or len(options) < len(choices):
                    choices = options
        for i,ids in enumerate(by_piece):
            if not used & (1 << i):
                options = available(ids)
                if not options:
                    return
                if len(options) < len(choices):
                    choices = options
        equivalent = set()
        for j in choices:
            i,mask,_ = placements[j]
            key = (signatures[i],mask)
            if key in equivalent:
                continue
            equivalent.add(key)
            search(used | (1 << i),filled | mask,chosen+(j,))

    search(0,0,())
    return list(answers.values()), {"nodes":nodes,"candidate_placements":len(placements)}


def _assembled(value):
    grid,square,pieces,info = _scene(value)
    answers,stats = _pack(square,pieces)
    _require(len(answers) == 1, "marked square must have exactly one colored exact cover")
    packed,placements = answers[0]
    info.update(stats)
    info.update({"pieces":pieces,"placements":placements,"initial_square":square,"grid":grid})
    return packed,info


def _expand(packed):
    n = len(packed)
    selectors = np.repeat(np.repeat(packed,n,axis=0),n,axis=1)
    reference = np.tile(packed,(n,n))
    return np.where(selectors == reference,selectors,0).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only exact cover followed by the packed square's equality tensor."""
    packed,_ = _assembled(input_grid)
    return _expand(packed)


def _partition(rng, holes, count):
    components = _regions(holes)
    if len(holes) < 2*count or len(components) > count:
        return None
    for _ in range(35):
        seeds = [sorted(c)[int(rng.integers(len(c)))] for c in components]
        candidates = sorted(set(holes)-set(seeds))
        if count > len(seeds):
            seeds += [candidates[int(i)] for i in rng.choice(len(candidates),count-len(seeds),replace=False)]
        labels = {p:i for i,p in enumerate(seeds)}
        unassigned = set(holes)-set(seeds)
        while unassigned:
            frontier = sorted((p,labels[q]) for p in unassigned for q in _neighbors(p) if q in labels)
            if not frontier:
                break
            p,label = frontier[int(rng.integers(len(frontier)))]
            labels[p] = label
            unassigned.remove(p)
        groups = [{p for p in holes if labels.get(p) == i} for i in range(count)]
        if not unassigned and min(map(len,groups)) >= 2:
            return groups
    return None


def _band_square(rng, n, interior):
    if interior:
        k = int(rng.integers(1,n-1))
    else:
        k = int(rng.choice([0,n-1]))
    band = np.zeros((n,n),dtype=bool)
    band[k,:] = True
    # Teeth can reach two cells on side five, but leave an outer blank row on
    # each occupied side. This keeps each socket connected before subdivision.
    for direction,space in ((-1,k),(1,n-1-k)):
        depth = min(2,space-1)
        if depth > 0:
            heights = rng.integers(0,depth+1,size=n)
            heights -= heights.min()
            if len(set(map(int,heights))) == 1:
                heights[int(rng.integers(n))] = (int(heights[0])+1) % (depth+1)
            for c,height in enumerate(heights):
                for d in range(1,int(height)+1):
                    band[k+direction*d,c] = True
    return band,k


def _asymmetric(packed):
    return all(not np.array_equal(packed,t) for t in (packed.T,np.fliplr(packed),np.flipud(packed),np.rot90(packed)))


def _scatter(rng, base, groups, colors, origin):
    h,w = base.shape
    for _ in range(12):
        inp = base.copy()
        forbidden = base != 0
        r0,c0,n = origin
        forbidden[r0-1:r0+n+1,c0-1:c0+n+1] = True
        for coords,color in sorted(zip(groups,colors),key=lambda item:-len(item[0])):
            shape = _normal(coords)
            ph,pw = shape.max(axis=0)+1
            padded = np.pad(forbidden,1)
            blocked = np.zeros_like(forbidden)
            for dr in range(3):
                for dc in range(3):
                    blocked |= padded[dr:dr+h,dc:dc+w]
            valid = np.ones((h-int(ph)+1,w-int(pw)+1),dtype=bool)
            for r,c in shape:
                valid &= ~blocked[r:r+valid.shape[0],c:c+valid.shape[1]]
            choices = np.argwhere(valid)
            if not len(choices):
                break
            placed = shape + choices[int(rng.integers(len(choices)))]
            inp[placed[:,0],placed[:,1]] = color
            forbidden[placed[:,0],placed[:,1]] = True
        else:
            return inp
    return None


def _make_pair(rng, n, count, interior, axis, reuse, asymmetric=False):
    for _ in range(600):
        mask,k = _band_square(rng,n,interior)
        groups = _partition(rng,set(map(tuple,np.argwhere(~mask))),count)
        if groups is None:
            continue
        palette = rng.permutation(np.arange(1,9,dtype=np.int8))
        band = int(palette[0])
        colors = [int(v) for v in palette[1:count+1]]
        if reuse:
            # Prefer a repeated category split across the band, so its support
            # mask cannot be recovered from any single component's silhouette.
            pairs = [(i,j) for i in range(count) for j in range(i) if
                     (max(r for r,c in groups[i]) < k < min(r for r,c in groups[j])) or
                     (max(r for r,c in groups[j]) < k < min(r for r,c in groups[i]))]
            if interior and not pairs:
                continue
            if not pairs:
                pairs = [(i,j) for i in range(count) for j in range(i)]
            i,j = pairs[int(rng.integers(len(pairs)))]
            colors[i] = colors[j]
        square = np.where(mask,band,0).astype(np.int8)
        packed = square.copy()
        pieces = []
        for cells,color in zip(groups,colors):
            coords = np.array(sorted(cells),dtype=int)
            packed[coords[:,0],coords[:,1]] = color
            pieces.append({"shape":_normal(cells),"color":color})
        if asymmetric and not _asymmetric(packed):
            continue
        answers,_ = _pack(square,pieces)
        if len(answers) != 1 or not np.array_equal(answers[0][0],packed):
            continue
        height,width = map(int,rng.integers(max(15,n+10),31,size=2))
        r0 = int(rng.integers(2,height-n-1))
        c0 = int(rng.integers(2,width-n-1))
        base = np.zeros((height,width),dtype=np.int8)
        base[r0+k,:] = band
        base[r0:r0+n,c0:c0+n] = square
        base[r0-1,c0-1] = base[r0-1,c0+n] = 9
        base[r0+n,c0-1] = base[r0+n,c0+n] = 9
        inp = _scatter(rng,base,groups,colors,(r0,c0,n))
        if inp is None:
            continue
        if axis:
            inp,packed = inp.T.copy(),packed.T.copy()
        # Labels use the constructed completion, while acceptance above checks
        # all alternative translations without reference to that completion.
        return {"input":inp,"output":_expand(packed)}
    raise RuntimeError(f"bounded sampling exhausted for n={n}, pieces={count}, interior={interior}")


def _count(value, low, high, name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} through {high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    tr = _count(num_train,2,10,"num_train")
    te = _count(num_test,1,4,"num_test")
    _require(tr > te, "num_train must exceed num_test")
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    size_transfer = bool(rng.integers(2))
    orientation_mode = int(rng.integers(3))  # mixed, H->V, V->H
    puzzle = {"train":[],"test":[]}
    seen = set()
    for split,count in (("train",tr),("test",te)):
        for index in range(count):
            if split == "train":
                n = 4 if index == 0 or size_transfer else 5
                if index >= 2:
                    n = int(rng.choice([3,4] if size_transfer else [3,4,5]))
                interior = index == 1 or (index > 1 and n > 3 and bool(rng.integers(2)))
                piece_count = 2 if index == 0 or n == 3 else 3
                reuse = index == 1 or (index > 1 and piece_count >= 3 and bool(rng.integers(2)))
                axis = index % 2 if orientation_mode == 0 else orientation_mode-1
            else:
                n = 5 if size_transfer else int(rng.choice([4,5],p=[0.25,0.75]))
                piece_count = 4
                interior = bool(rng.integers(2))
                reuse = bool(rng.integers(2))
                axis = int(rng.integers(2)) if orientation_mode == 0 else 2-orientation_mode
            for _ in range(10):
                pair = _make_pair(rng,n,piece_count,interior,axis,reuse,asymmetric=index == 0)
                key = (pair["input"].shape,pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not generate a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_pair(pair):
    _require(isinstance(pair,dict) and set(pair) == {"input","output"}, "each pair must contain exactly input and output")
    inp,out = _grid(pair["input"]),_grid(pair["output"])
    packed,info = _assembled(inp)
    n = info["n"]
    _require(out.shape == (n*n,n*n), "output must be n squared by n squared")
    _require(np.all(packed != 0) and not np.any(packed == 9), "completed categories exclude blank and markers")
    # Decode from equality-tensor diagonal, independently of _expand.
    decoded = np.array([[out[r*n+r,c*n+c] for c in range(n)] for r in range(n)],dtype=np.int8)
    _require(np.array_equal(decoded,packed), "output diagonal does not recover the exact-cover completion")
    for r in range(n):
        for c in range(n):
            color = int(packed[r,c])
            tile = out[r*n:(r+1)*n,c*n:(c+1)*n]
            _require(np.array_equal(tile,np.where(packed == color,color,0)), "tile is not a whole-completed-square support mask")
    for color in np.unique(packed):
        _require(np.count_nonzero(out == color) == np.count_nonzero(packed == color)**2, "category frequency must square")
    _require(np.count_nonzero(packed == info["band"]) == np.count_nonzero(info["initial_square"] == info["band"]), "fixed band changed")
    used = set()
    for i,_,coords in info["placements"]:
        _require(i not in used, "component used twice")
        used.add(i)
        piece = info["pieces"][i]
        _require(np.array_equal(_normal(map(tuple,coords)),piece["shape"]), "piece rotated or reflected")
        _require(np.all(packed[coords[:,0],coords[:,1]] == piece["color"]), "piece recolored")
    _require(used == set(range(len(info["pieces"]))), "not every component was used")
    _require(not np.array_equal(out,_expand(info["initial_square"])), "packing omission did not change the answer")
    solid = np.repeat(np.repeat(packed,n,axis=0),n,axis=1)
    _require(not np.array_equal(out,solid), "mask omission did not change the answer")
    repeated = Counter(p["color"] for p in info["pieces"])
    info["global_repeat"] = any(repeated[int(color)] >= 2 and len(_regions(map(tuple,np.argwhere(packed == color)))) >= 2 for color in repeated)
    info["packed"] = packed
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {"train","test"}, "puzzle must have exactly train and test")
    _require(all(isinstance(puzzle[s],list) for s in ("train","test")), "splits must be lists")
    tr,te = len(puzzle["train"]),len(puzzle["test"])
    _require(2 <= tr <= 10 and 1 <= te <= 4 and tr > te, "illegal pair counts")
    seen,train_infos,test_infos = set(),[],[]
    for split in ("train","test"):
        for pair in puzzle[split]:
            info = _validate_pair(pair)
            key = (info["grid"].shape,info["grid"].tobytes())
            _require(key not in seen, "duplicate inputs")
            seen.add(key)
            (train_infos if split == "train" else test_infos).append(info)
    _require(any(i["global_repeat"] for i in train_infos), "training must show one category spanning detached pieces")
    _require(any(_asymmetric(i["packed"]) for i in train_infos), "training must anchor unrotated mask coordinates")
    _require(all(len(i["pieces"]) <= 3 for i in train_infos) and all(len(i["pieces"]) == 4 for i in test_infos), "expected demonstrated small covers then four-piece recombinations")


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train","test"):
        for pair in puzzle[split]:
            for role in ("input","output"):
                a = np.asarray(pair[role],dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded, streaming measurements over the 42 supplied seeds; no files."""
    structures,sizes,canvas_shapes,axes,bands,counts,category_counts,profiles,signatures = (set() for _ in range(9))
    total_pairs,global_repeats,max_nodes = 0,0,0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "seed must reproduce the whole puzzle")
        signatures.add(_digest(puzzle))
        for split in ("train","test"):
            for pair in puzzle[split]:
                info = _validate_pair(pair)
                packed = info["packed"]
                mapping = {}
                norm = bytes(mapping.setdefault(int(c),len(mapping)) for c in packed.flat)
                structures.add((info["n"],norm))
                sizes.add(info["n"])
                canvas_shapes.add(pair["input"].shape)
                axes.add(info["axis"])
                bands.add(info["band"])
                counts.add(len(info["pieces"]))
                category_counts.add(len(np.unique(packed)))
                profiles.add((info["n"],info["axis"],tuple((packed == info["band"]).ravel())))
                global_repeats += int(info["global_repeat"])
                total_pairs += 1
                max_nodes = max(max_nodes,info["nodes"])
        if seed == 0:
            # Coordinates belong to the assembled square, not to a canonical
            # band direction. Check all eight whole-scene square symmetries.
            anchor = puzzle["train"][1]
            original = anchor["input"].copy()
            for turns in range(4):
                for reflected in (False,True):
                    inp = np.rot90(original,turns)
                    out = np.rot90(anchor["output"],turns)
                    if reflected:
                        inp,out = np.fliplr(inp),np.fliplr(out)
                    _require(np.array_equal(solve(inp),out), "scene symmetry changed the rule")
            # Literal payload/band identities have no meaning. Keep only the
            # fixed background and extraction-marker roles unchanged.
            palette = np.array([0,8,7,6,5,4,3,2,1,9],dtype=np.int8)
            _require(np.array_equal(solve(palette[original]),palette[anchor["output"]]), "palette identities leaked into the answer")
            _require(np.array_equal(solve(original.tolist()),anchor["output"]), "nested-list inputs must obey the same rule")
            _require(np.array_equal(original,anchor["input"]), "solve modified its input")
            # Re-scatter precisely the same components: spatial origins do not
            # select tiles or determine the global support-mask coordinates.
            info = _validate_pair(puzzle["train"][1])
            base = info["grid"].copy()
            for piece in info["pieces"]:
                src = piece["source"]
                base[src[:,0],src[:,1]] = 0
            r0,c0 = info["origin"]
            moved = _scatter(np.random.default_rng(123),base,[p["shape"] for p in info["pieces"]],[p["color"] for p in info["pieces"]],(r0,c0,info["n"]))
            _require(moved is not None and not np.array_equal(moved,info["grid"]), "re-scatter fixture failed")
            _require(np.array_equal(solve(moved),puzzle["train"][1]["output"]), "scatter positions leaked into masks")
    _require(len(signatures) == 42 and sizes == {3,4,5}, "insufficient seeded/size variation")
    _require(axes == {0,1} and bands == set(range(1,9)), "orientation and band palettes must vary")
    _require(counts == {2,3,4} and len(category_counts) >= 3, "piece/category counts must vary")
    _require(len(structures) >= 100 and len(profiles) >= 70 and len(canvas_shapes) >= 60, "structural variation too narrow")
    _require(global_repeats >= 42, "global-mask repeats not adequately demonstrated")
    # The input-only domain permits five components, although ordinary puzzles
    # deliberately teach 2/3 and test 4. Check that broader advertised domain.
    for interior in (False,True):
        pair = _make_pair(np.random.default_rng(9301+int(interior)),5,5,interior,0,True)
        info = _validate_pair(pair)
        _require(len(info["pieces"]) == 5, "five-component fixture failed")
        _require(np.array_equal(solve(pair["input"]),pair["output"]), "five-component input-only solve failed")
    # Regression: two differently colored dominoes have two colored packings;
    # same-colored indistinguishable dominoes have one, despite permutations.
    square = np.array([[8,8,8],[8,0,0],[8,0,0]],dtype=np.int8)
    domino = np.array([[0,0],[0,1]],dtype=int)
    pieces = [{"shape":domino,"color":2},{"shape":domino,"color":3}]
    _require(len(_pack(square,pieces)[0]) == 2, "ambiguity must be detected")
    pieces[1]["color"] = 2
    _require(len(_pack(square,pieces)[0]) == 1, "equivalent component permutations must be deduplicated")
    for tr in range(2,11):
        for te in range(1,min(4,tr-1)+1):
            p = generate_puzzle(0,tr,te)
            _require((len(p["train"]),len(p["test"])) == (tr,te), "explicit pair counts failed")
    for tr,te in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(4.0,1),(None,1),(4,None)):
        try:
            generate_puzzle(0,tr,te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count accepted")
    broken = generate_puzzle(0)
    broken["train"][0]["output"][0,0] = 9
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong label accepted")
    return {"seeds":42,"pairs":total_pairs,"packed_structures":len(structures),"band_profiles":len(profiles),"canvas_shapes":len(canvas_shapes),"sizes":sorted(sizes),"piece_counts":sorted(counts),"category_counts":sorted(category_counts),"global_repeat_pairs":global_repeats,"maximum_search_nodes":max_nodes}


if __name__ == "__main__":
    print(json.dumps(self_test(),separators=(",",":")))
