"""A counter-powered comb press: finite travel, delayed contact, ordered clipping.

Only NumPy and the standard library are used. No files are read during generation
or prediction. Fixed roles are 0=background, 1=spent, 2=fuel, 3=register marks;
all other colors are inferred locally. Rotation preserves the counter's handedness.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


HOST_SEEDS = (0,103266074,288323621,226219937,944880205,742273741,404808964,
              37818013,57011239,317834241,594899198,276320405,124543988,
              81968635,861668054,148268533,449101479,16047726,524564020,
              440148383,873141112,614591461,369824414,742296588,398518209,
              908694848,738408367,958475476,977887545,409956577,561112920,
              354407054,782063026,867261979,675511001,111358033,881116976,
              279894278,372534799,456870405,631734944,298501036)


def _need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _need(np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _count(value, name, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in {lower}..{upper}")
    return int(value)


def _movement(f):
    return min(f["fuel"], f["distance"])


def _shifts(f, movement=None):
    m = _movement(f) if movement is None else movement
    return tuple(max(0, m - gap) for gap in f["gaps"])


def _prefix(seq, structure):
    return next(i for i, color in enumerate(seq) if color != structure)


def _draw(f, final=False, movement=None, pins=True):
    """Direct coordinate construction; the solver uses a separate pixel path."""
    a = np.zeros(f["shape"], dtype=np.int8)
    t,b,l,r = (f[k] for k in ("top","bottom","left","right"))
    s = f["structure"]
    a[t, l:r+1] = a[b, l:r+1] = s
    a[t:b+1, l] = a[t:b+1, r] = s
    m = (_movement(f) if movement is None else movement) if final else 0
    base,cl,cr,ch = f["base"]-m, f["cl"], f["cr"], f["ch"]
    a[base:base+ch+2, cl:cr+1] = s
    spent = f["ch"]*(cr-cl-1) - f["fuel"] + m
    counter = np.full((ch,cr-cl-1), 2, dtype=np.int8)
    counter.flat[:spent] = 1
    a[base+1:base+ch+1, cl+1:cr] = counter
    for i,(col,h,seq) in enumerate(zip(f["lanes"],f["heights"],f["stacks"])):
        a[base-h:base, col] = s
        shift = max(0,m-f["gaps"][i]) if final and pins else 0
        # The original stack starts immediately below the clipping wall.
        survivors = seq[shift:]
        a[t+1:t+1+len(survivors),col] = survivors
    a[f["base"]-f["distance"],l-2:l] = 3
    a[base,l-2:l] = 3
    return a


def _read_canonical(a, structure):
    """Explain every pixel, without relying on the modal color or grid padding."""
    sy,sx = np.nonzero(a == structure)
    _need(len(sy) > 0, "no structural cells")
    t,b,l,r = int(sy.min()),int(sy.max()),int(sx.min()),int(sx.max())
    _need(l >= 2 and b-t >= 6 and r-l >= 6, "housing too small")
    _need(np.all(a[t,l:r+1] == structure) and np.all(a[b,l:r+1] == structure)
          and np.all(a[t:b+1,l] == structure) and np.all(a[t:b+1,r] == structure),
          "housing is not a closed rectangular frame")
    yy,xx = np.nonzero((a == 1) | (a == 2))
    _need(len(yy) > 0, "counter missing")
    ct,cb,ci,cj = int(yy.min()),int(yy.max()),int(xx.min()),int(xx.max())
    counter = a[ct:cb+1,ci:cj+1]
    _need(np.all((counter == 1) | (counter == 2)), "counter is not solid")
    base,cl,cr,ch = ct-1,ci-1,cj+1,cb-ct+1
    _need(t < base < cb+1 < b and l < cl < cr < r, "carriage outside housing")
    _need(np.all(a[base,cl:cr+1] == structure)
          and np.all(a[cb+1,cl:cr+1] == structure)
          and np.all(a[base:cb+2,cl] == structure)
          and np.all(a[base:cb+2,cr] == structure), "counter frame broken")
    fuel = int(np.count_nonzero(counter == 2))
    _need(counter.ravel().tolist() == [1]*(counter.size-fuel)+[2]*fuel,
          "counter does not follow travel-up row-major order")
    marks = np.argwhere(a == 3)
    rows = sorted(set(map(int,marks[:,0]))) if len(marks) else []
    _need(len(rows) == 2 and rows[1] == base and len(marks) == 4,
          "registers must mark current base and a forward stop")
    target = rows[0]
    _need(t < target < base and np.all(a[rows,l-2:l] == 3), "bad stop alignment")
    distance = base-target
    lanes = tuple(int(c) for c in np.flatnonzero(a[base-1,cl:cr+1] == structure)+cl)
    _need(len(lanes) >= 2 and lanes[0] == cl+1 and lanes[-1] == cr-1,
          "teeth must span the counter interior")
    _need(all(v-u >= 2 for u,v in zip(lanes,lanes[1:])), "teeth merge")
    heights,stacks,gaps = [],[],[]
    for col in lanes:
        row = base-1
        while row > t and a[row,col] == structure:
            row -= 1
        height = base-1-row
        tooth_top = base-height
        _need(height >= 1 and tooth_top-distance >= t+2,
              "the permitted stroke leaves no room for a surviving pin")
        segment = a[t+1:tooth_top,col]
        nonzero = np.flatnonzero(segment != 0)
        _need(len(nonzero) >= 1 and np.array_equal(nonzero,np.arange(len(nonzero))),
              "pin must hang contiguously from roof")
        seq = tuple(map(int,segment[:len(nonzero)]))
        _need(all(c >= 4 for c in seq), "pin uses a reserved role color")
        p = next((i for i,c in enumerate(seq) if c != structure), len(seq))
        _need(p < len(seq) and all(c != structure for c in seq[p:]),
              "pin needs a structural prefix followed by an ordered payload")
        heights.append(height)
        stacks.append(seq)
        gaps.append(tooth_top-t-len(seq)-1)
    f = dict(shape=a.shape,structure=structure,top=t,bottom=b,left=l,right=r,
             base=base,cl=cl,cr=cr,ch=ch,lanes=lanes,heights=tuple(heights),
             stacks=tuple(stacks),gaps=tuple(gaps),fuel=fuel,distance=distance)
    _need(np.array_equal(a,_draw(f)), "unexplained pixels or overlapping parts")
    return f


def _parse(input_grid):
    a = _grid(input_grid)
    readings = []
    for turns in range(4):
        canonical = np.rot90(a,turns)
        for s in np.unique(canonical):
            if s < 4:
                continue
            try:
                f = _read_canonical(canonical,int(s))
            except (AssertionError,IndexError):
                continue
            f["turns"] = turns
            readings.append(f)
    _need(len(readings) == 1, "input must have one complete press interpretation")
    return readings[0]


def _pixel_solution(input_grid, f):
    """Move input pixels and update the counter one unit at a time."""
    a = np.rot90(np.asarray(input_grid), f["turns"])
    out = a.copy()
    base,cl,cr,ch,t = (f[k] for k in ("base","cl","cr","ch","top"))
    block = a[base:base+ch+2,cl:cr+1].copy()
    inner = block[1:-1,1:-1].copy()
    teeth = [(row,col) for col,h in zip(f["lanes"],f["heights"])
             for row in range(base-h,base)]
    # Clear originals first. Wall, fixed stop and all unrelated background stay.
    out[base:base+ch+2,cl:cr+1] = 0
    for p in teeth:
        out[p] = 0
    for col,seq in zip(f["lanes"],f["stacks"]):
        out[t+1:t+1+len(seq),col] = 0
    out[base,f["left"]-2:f["left"]] = 0
    moved = 0
    while moved < f["distance"]:
        pending = np.flatnonzero(inner.ravel() == 2)
        if not len(pending):
            break
        inner.flat[int(pending[0])] = 1
        moved += 1
    block[1:-1,1:-1] = inner
    out[base-moved:base-moved+ch+2,cl:cr+1] = block
    for row,col in teeth:
        out[row-moved,col] = a[row,col]
    for col,h,seq in zip(f["lanes"],f["heights"],f["stacks"]):
        pin = [(t+1+j,int(a[t+1+j,col])) for j in range(len(seq))]
        final_tip = base-moved-h
        while pin[-1][0] >= final_tip:
            pin = [(row-1,c) for row,c in pin if row-1 > t]
            _need(bool(pin), "a complete pin was clipped")
        for row,c in pin:
            out[row,col] = c
    out[base-moved,f["left"]-2:f["left"]] = 3
    return np.rot90(out,-f["turns"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Predict from this grid alone; no demonstrations, seed or cache are used."""
    return _pixel_solution(input_grid,_parse(input_grid))


def _profile(rng,n,nonflat=False):
    mode = str(rng.choice(("flat","rise","fall","valley","plateau","free")))
    high = int(rng.integers(2,6))
    h = rng.integers(1,high+1,size=n).tolist()
    if mode == "flat" and not nonflat:
        h = [int(rng.integers(1,high+1))]*n
    elif mode == "rise":
        h.sort()
    elif mode == "fall":
        h.sort(reverse=True)
    elif mode == "valley" and n >= 3:
        h[0],h[-1],h[n//2] = high,high,1
    elif mode == "plateau" and n >= 3:
        pos = int(rng.integers(n-1))
        h[pos:pos+2] = [high,high]
        h[(pos+2)%n] = 1
    if nonflat and len(set(h)) == 1:
        h[int(rng.integers(n))] = 1 if h[0] > 1 else high
    return tuple(map(int,h))


def _sample(rng,n,distance,structure,teaching=False,query=False):
    steps = rng.integers(2,5 if n <= 4 else 4,size=n-1).tolist()
    offsets = [1]
    for step in steps:
        offsets.append(offsets[-1]+int(step))
    iw = offsets[-1]
    ch = int(rng.integers(2 if teaching else 1,5))
    h = _profile(rng,n,nonflat=teaching)
    t = int(rng.integers(3))
    left = 2+int(rng.integers(3))
    cl = left+int(rng.integers(1,4))
    cr = cl+iw+1
    right = cr+int(rng.integers(1,4))
    base = t+max(h)+distance+2+int(rng.integers(1 if teaching else 0,3))
    bottom = base+ch+1+int(rng.integers(1,4))
    shape = (bottom+1+int(rng.integers(3)),right+1+int(rng.integers(3)))
    _need(max(shape) <= 30, "sampler exceeded canvas budget")
    lanes = tuple(cl+v for v in offsets)
    capacity = iw*ch
    choices = ["zero"] if not query else []
    if distance > 1:
        choices.append("fuel")
    if capacity >= distance:
        choices.append("exact")
    if capacity > distance:
        choices.append("stop")
    regime = str(rng.choice(choices))
    if regime == "zero":
        fuel = 0
    elif regime == "exact":
        fuel = distance
    elif regime == "stop":
        fuel = int(rng.integers(distance+1,capacity+1))
    else:
        fuel = int(rng.integers(1,min(distance-1,capacity)+1))
    gaps = [int(rng.integers(0,min(distance+2,base-height-t-2)+1)) for height in h]
    short = distance-2
    early = None
    if teaching:
        # The same hardware is shown with two fuel levels. One nonminimum tooth
        # contacts early, one at the short stroke, and one remains untouched.
        early = int(rng.choice([i for i in range(n) if h[i] > min(h)]))
        other = [i for i in range(n) if i != early]
        rng.shuffle(other)
        gaps[early],gaps[other[0]],gaps[other[1]] = 0,short,distance
        fuel = short
    elif query:
        # Queries require an observable interaction, with a pushed pin and an
        # unreached (or exactly touched) pin. Other lanes remain unconstrained.
        early,still = map(int,rng.choice(n,size=2,replace=False))
        gaps[early],gaps[still] = 0,distance
    payload_pool = [c for c in range(4,10) if c != structure]
    palette = rng.choice(payload_pool,size=int(rng.integers(1,6)),replace=False).tolist()
    stacks = []
    for i,(height,gap) in enumerate(zip(h,gaps)):
        length = base-height-t-gap-1
        prefix = int(rng.integers(length))
        if teaching and i == early:
            prefix = short+1
        size = length-prefix
        if rng.random() < .6:
            payload = [int(rng.choice(palette))]*size
        else:
            payload = list(map(int,rng.choice(palette,size=size)))
        if teaching and i == early:
            ca,cb,cc = sorted(map(int,rng.choice(payload_pool,size=3,replace=False)))
            # A valley with distinct endpoints distinguishes both sorting
            # directions and reversal, even in the two-demonstration case.
            payload = [cb]*(size-2)+[ca,cc]
        stacks.append((structure,)*prefix+tuple(payload))
    return dict(shape=shape,structure=structure,top=t,bottom=bottom,left=left,right=right,
                base=base,cl=cl,cr=cr,ch=ch,lanes=lanes,heights=h,stacks=tuple(stacks),
                gaps=tuple(gaps),fuel=fuel,distance=distance)


def _pair(f,turns):
    return {"input":np.rot90(_draw(f),turns).copy(),
            "output":np.rot90(_draw(f,True),turns).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train,"num_train",2,10)
    num_test = _count(num_test,"num_test",1,4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shift = str(rng.choice(("lanes","stroke")))
    orientation = int(rng.integers(4))
    mixed = bool(rng.integers(2))
    structure = int(rng.integers(4,10))
    recolor = bool(rng.integers(2))
    demo = _sample(rng,int(rng.integers(3,5)),int(rng.integers(4,6)),structure,True)
    cap = demo["ch"]*(demo["cr"]-demo["cl"]-1)
    high = dict(demo,fuel=int(rng.integers(demo["distance"]+1,cap+1)))
    result = {"train":[_pair(demo,orientation),_pair(high,orientation)],"test":[]}
    seen = {_key(p["input"]) for p in result["train"]}
    for split,count in (("train",num_train-2),("test",num_test)):
        for _ in range(count):
            for attempt in range(32):
                n = int(rng.integers(5,7) if shift == "lanes" and split == "test"
                        else rng.integers(2,5) if shift == "lanes" else rng.integers(2,7))
                d = int(rng.integers(6,9) if shift == "stroke" and split == "test"
                        else rng.integers(1,6) if shift == "stroke" else rng.integers(1,9))
                s = int(rng.integers(4,10)) if recolor else structure
                f = _sample(rng,n,d,s,query=split == "test")
                turns = int(rng.integers(4)) if mixed else orientation
                pair = _pair(f,turns)
                if _key(pair["input"]) not in seen:
                    result[split].append(pair)
                    seen.add(_key(pair["input"]))
                    break
            else:
                raise RuntimeError("32 bounded uniqueness attempts failed")
    rng.shuffle(result["train"])
    rng.shuffle(result["test"])
    return result


def _features(f):
    shifts = _shifts(f)
    prefixes = tuple(_prefix(seq,f["structure"]) for seq in f["stacks"])
    m = _movement(f)
    surviving_payloads = [seq[max(p,d):] for seq,p,d in zip(f["stacks"],prefixes,shifts)]
    return dict(lanes=len(f["lanes"]),distance=f["distance"],fuel=f["fuel"],movement=m,
                heights=f["heights"],gaps=f["gaps"],shifts=shifts,prefixes=prefixes,
                turns=f["turns"],shape=f["shape"],structure=f["structure"],
                counter_shape=(f["ch"],f["cr"]-f["cl"]-1),
                mixed_contact=0 in shifts and max(shifts)>0,
                moving_prefix=any(0 < d < p for d,p in zip(shifts,prefixes)),
                payload_clip=any(d > p for d,p in zip(shifts,prefixes)),
                multicolor_survives=any(len(set(seq[max(p,d):])) >= 2
                                       for seq,p,d in zip(f["stacks"],prefixes,shifts)),
                order_witness=any(seq != tuple(sorted(seq)) and seq != tuple(sorted(seq,reverse=True))
                                  and seq != seq[::-1] for seq in surviving_payloads),
                row_wrap=(f["ch"]*(f["cr"]-f["cl"]-1)-f["fuel"])//(f["cr"]-f["cl"]-1)
                         != (f["ch"]*(f["cr"]-f["cl"]-1)-f["fuel"]+m)//(f["cr"]-f["cl"]-1))


def _hardware_key(a,f):
    canonical = np.rot90(a,f["turns"]).copy()
    canonical[f["base"]+1:f["base"]+f["ch"]+1,f["cl"]+1:f["cr"]] = 1
    return _key(canonical)


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle,dict) and set(puzzle) == {"train","test"}, "bad schema")
    _need(all(isinstance(puzzle[s],list) for s in puzzle), "splits must be lists")
    _need(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
          and len(puzzle["train"]) > len(puzzle["test"]), "bad counts")
    seen,features,hardware = set(),{"train":[],"test":[]},{}
    without_a = without_b = without_height = False
    for split in ("train","test"):
        for pair in puzzle[split]:
            _need(isinstance(pair,dict) and set(pair) == {"input","output"}, "bad pair")
            a,b = _grid(pair["input"]),_grid(pair["output"])
            _need(a.shape == b.shape, "shape changed")
            _need(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            f = _parse(a)
            _need(np.array_equal(solve(a),b), "pixel solver disagrees")
            expected = np.rot90(_draw(f,True),-f["turns"])
            _need(np.array_equal(expected,b), "coordinate construction disagrees")
            # Family-specific conservation: each translated unit spends one cell;
            # every pushed pin loses exactly its displaced prefix at the roof.
            _need(np.count_nonzero(a == 2)-np.count_nonzero(b == 2) == _movement(f),
                  "fuel is not conserved one cell per step")
            _need(np.count_nonzero(b == 1)-np.count_nonzero(a == 1) == _movement(f),
                  "spent counter increment is wrong")
            _need(all(d < len(seq) for d,seq in zip(_shifts(f),f["stacks"])),
                  "a pin vanished completely")
            # Roof, floor and side walls cannot be consumed or overwritten.
            out = np.rot90(b,f["turns"])
            t,bt,l,r = (f[k] for k in ("top","bottom","left","right"))
            _need(np.all(out[t,l:r+1] == f["structure"])
                  and np.all(out[bt,l:r+1] == f["structure"])
                  and np.all(out[t:bt+1,l] == f["structure"])
                  and np.all(out[t:bt+1,r] == f["structure"]), "wall changed")
            features[split].append(_features(f))
            if split == "train":
                hardware.setdefault(_hardware_key(a,f),[]).append(f)
                if 0 < f["fuel"] < f["distance"]:
                    # Suppress fuel limiting, without inventing extra counter cells.
                    without_a |= _shifts(f,f["distance"]) != _shifts(f)
                without_b |= max(_shifts(f)) > 0
                flat_gaps = [g+h-min(f["heights"]) for g,h in zip(f["gaps"],f["heights"])]
                without_height |= tuple(max(0,_movement(f)-g) for g in flat_gaps) != _shifts(f)
    train = features["train"]
    _need(any(any(0 < f["fuel"] < f["distance"] for f in group)
              and any(f["fuel"] > f["distance"] for f in group)
              for group in hardware.values()), "missing counter-only controlled comparison")
    for name in ("mixed_contact","moving_prefix","payload_clip","multicolor_survives","order_witness"):
        _need(any(x[name] for x in train), f"training does not expose {name}")
    _need(without_a and without_b, "omitting a parent does not affect demonstrated pins")
    _need(without_height, "training cannot distinguish a flat pusher from a comb")
    max_n,max_d = max(x["lanes"] for x in train),max(x["distance"] for x in train)
    _need(all(x["lanes"] > max_n or x["distance"] > max_d for x in features["test"]),
          "missing lane-count or stroke transfer")
    _need(all(x["mixed_contact"] for x in features["test"]),
          "query does not require selective pin contact")


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train","test"):
        for pair in puzzle[split]:
            for role in ("input","output"):
                a = np.asarray(pair[role])
                h.update(bytes(a.shape))
                h.update(a.astype(np.uint8).tobytes())
    return h.hexdigest()


def self_test():
    """Stream 42 seeds; retain only hashes and small aggregate feature sets."""
    shapes,profiles,counters,colors,turns,lanes,distances,digests = (set() for _ in range(8))
    totals = dict(pairs=0,mixed_contact=0,moving_prefix=0,payload_clip=0,
                  multicolor_survives=0,order_witness=0,row_wrap=0,zero=0,exact=0,fuel=0,stop=0)
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digests.add(_digest(puzzle))
        _need(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterministic")
        for split in ("train","test"):
            for pair in puzzle[split]:
                f = _parse(pair["input"])
                d = _features(f)
                totals["pairs"] += 1
                shapes.add(d["shape"])
                profiles.add(d["heights"])
                counters.add(d["counter_shape"])
                colors.add(d["structure"])
                turns.add(d["turns"])
                lanes.add(d["lanes"])
                distances.add(d["distance"])
                for flag in ("mixed_contact","moving_prefix","payload_clip","multicolor_survives","order_witness","row_wrap"):
                    totals[flag] += int(d[flag])
                regime = "zero" if f["fuel"] == 0 else "fuel" if f["fuel"] < f["distance"] else "exact" if f["fuel"] == f["distance"] else "stop"
                totals[regime] += 1
        # Check covariance of the input-only solve under all rotations on one
        # pair per seed, with no reinterpretation of the color roles.
        pair = puzzle["train"][0]
        for k in range(4):
            _need(np.array_equal(solve(np.rot90(pair["input"],k)),
                                 np.rot90(pair["output"],k)), "rotation mismatch")
    _need(len(digests) == 42 and len(shapes) >= 60 and len(profiles) >= 65,
          "insufficient structural variety")
    _need(turns == set(range(4)) and lanes == set(range(2,7))
          and distances == set(range(1,9)) and colors == set(range(4,10)),
          "missing geometry/orientation/color coverage")
    _need(len(counters) >= 18 and all(totals[k] > 0 for k in totals), "missing boundary regime")
    # A small explicit fixture also checks literal surviving sequences. These
    # assertions do not derive their expected cells by calling the constructor.
    f = dict(shape=(17,13),structure=5,top=0,bottom=16,left=2,right=12,
             base=11,cl=4,cr=10,ch=2,lanes=(5,7,9),heights=(2,4,1),
             stacks=((5,5,4,6,4,6,4,6),(5,5,7,4),(8,9,8,9)),
             gaps=(0,2,5),fuel=3,distance=5)
    probe = _draw(f)
    out = solve(probe)
    _need(out[1:6,5].tolist() == [6,4,6,4,6], "ordered payload clipping regression")
    _need(out[1:4,7].tolist() == [5,7,4], "delayed prefix contact regression")
    _need(out[1:5,9].tolist() == [8,9,8,9] and int(out[5,9]) == 0,
          "unreached pin moved")
    _need(np.all(out[9:11,5:10] == 1), "carriage counter update regression")
    for fuel in (0,3,5,7,10):
        variant = dict(f,fuel=fuel)
        a = _draw(variant)
        b = solve(a)
        _need(np.array_equal(b,_draw(variant,True)), "fuel boundary mismatch")
        _need(fuel != 0 or np.array_equal(a,b), "exhausted press moved")
    mapping = np.array([0,1,2,3,7,9,4,6,5,8],dtype=np.int8)
    _need(np.array_equal(solve(mapping[probe]),mapping[out]), "structural recoloring mismatch")
    for y,x in ((12,5),(6,0),(1,5)):
        malformed = probe.copy()
        malformed[y,x] = 0 if (y,x) != (12,5) else 2
        try:
            solve(malformed)
        except AssertionError:
            pass
        else:
            raise AssertionError("malformed counter, marker or pin accepted")
    # Focused, literal feature probes: one empty gap, one delayed contact, one
    # uncontacted pin, with both a retained prefix and payload clipping.
    p = generate_puzzle(0,2,1)
    fs = [_parse(x["input"]) for x in p["train"]]
    low,high = sorted(fs,key=lambda f:f["fuel"])
    _need(_shifts(low) != _shifts(high), "fuel cannot change pin response")
    _need(any(g == low["fuel"] for g in low["gaps"]), "missing exact first-contact boundary")
    pair = p["train"][0]
    corrupt = {s:[{k:np.asarray(v).copy() for k,v in x.items()} for x in p[s]] for s in p}
    corrupt["test"][0]["output"][0,0] = (int(corrupt["test"][0]["output"][0,0])+1)%10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted corrupt output")
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            validate_generated_puzzle(generate_puzzle(20261023,a,b))
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=len(HOST_SEEDS),distinct_puzzles=len(digests),grid_shapes=len(shapes),
                profiles=len(profiles),counter_shapes=len(counters),totals=totals,
                independent_solver=False,blind_prediction=False)


if __name__ == "__main__":
    print(json.dumps(self_test(),separators=(",", ":")))
