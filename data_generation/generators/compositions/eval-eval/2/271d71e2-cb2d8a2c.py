"""Fuel reclaimed from routing instructions; standard library and NumPy only.

The frame is a reservoir, its port is the first cell of an unspooling route.
Background cells form the spent prefix in the travel-up raster. Route cells and
bar instruction cells are the same conserved material as the remaining fuel.
No parent modules, examples, stored puzzles, or generation state are consulted.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import numpy as np


HOST_SEEDS = (0,930906450,403531047,711521335,702590113,120129560,75230312,744348021,671020874,74829236,206917865,633753921,680661940,227045592,560007479,458287629,702897364,160036265,446529103,846061492,13344333,555092984,993765126,921882117,897078003,36193338,720442729,16094555,314355350,30607616,380452401,412134962,997367604,50492393,953902442,28844610,739612780,89438002,706773900,953866914,964980688,566560708)


def _need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(np.all((0 <= a) & (a <= 9)), "invalid color")
    return a.astype(np.int8, copy=True)


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train,"num_train",2,10),(num_test,"num_test",1,4)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, component = [start], {start}
        while stack:
            y,x = stack.pop()
            for p in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if p in unseen:
                    unseen.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(component)
    return result


def _border(t,b,l,r):
    return {(y,x) for y in range(t,b+1) for x in range(l,r+1)
            if y in (t,b) or x in (l,r)}


def _counter(t,b,l,r):
    # Downward travel viewed with its heading rotated upward: bottom-right first.
    return [(y,x) for y in range(b-1,t,-1) for x in range(r-1,l,-1)]


def _route(shape, port, bars):
    """Expand decoded geometric bends, with a pickup event at each row crossing."""
    height,width = shape
    path = [port]
    events = {}
    current_y,current_x = port
    previous = current_y
    for index,bar in enumerate(bars):
        row,lo,hi,n,side = (bar[k] for k in ("row","lo","hi","n","side"))
        d = n+1
        turn = row-d
        target = hi+d if side == 1 else lo-d
        _need(turn > previous and lo <= current_x <= hi, "bar misses incoming ray or turn overlaps prior bar")
        _need(0 <= target < width and row < height-1, "detour leaves grid")
        for y in range(current_y+1,turn+1):
            path.append((y,current_x))
        for x in range(current_x+side,target+side,side):
            path.append((turn,x))
        # Cross this bar now; later turns must be beyond this row.
        for y in range(turn+1,row+1):
            path.append((y,target))
        events[len(path)-1] = index
        current_y,current_x,previous = row,target,row
    for y in range(current_y+1,height):
        path.append((y,current_x))
    _need(len(path) == len(set(path)), "self-intersecting route")
    _need(all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a,b in zip(path,path[1:])), "disconnected route")
    return path,events


def _read_canonical(a, bg, structural, energy, turns):
    components = _components(a == structural)
    wide = [c for c in components if len({p[0] for p in c}) > 1 and len({p[1] for p in c}) > 1]
    _need(len(wide) == 1, "requires one reservoir")
    frame = wide[0]
    t,b = min(y for y,x in frame),max(y for y,x in frame)
    l,r = min(x for y,x in frame),max(x for y,x in frame)
    _need(b-t >= 2 and r-l >= 2 and 0 < l and r < a.shape[1]-1, "invalid reservoir shape")
    border = _border(t,b,l,r)
    missing = border-frame
    _need(len(missing) == 1 and frame == border-missing, "frame must have exactly one port")
    port = next(iter(missing))
    _need(port[0] == b and l < port[1] < r and a[port] == energy, "port is not on leading side")
    coordinates = _counter(t,b,l,r)
    values = [int(a[p]) for p in coordinates]
    spent = values.count(bg)
    _need(values == [bg]*spent+[energy]*(len(values)-spent), "counter is not a spent prefix")
    allowed = set(coordinates) | border
    bars = []
    for component in components:
        if component is frame:
            continue
        # Markers interrupt bar-color components, so parse complete rows below.
        _need(all(y > b for y,x in component), "structure behind reservoir")
    rows = np.flatnonzero(np.any(a[b+1:] == structural,axis=1))+b+1
    _need(1 <= len(rows) <= 5, "requires 1..5 bars")
    for raw_row in rows:
        row = int(raw_row)
        positions = np.flatnonzero(a[row] != bg)
        lo,hi = int(positions[0]),int(positions[-1])
        _need(len(positions) == hi-lo+1, "bar is not contiguous")
        marks = tuple(map(int,np.flatnonzero(a[row] == energy)))
        n = len(marks)
        _need(1 <= n <= 4 and hi-lo+1 >= max(2,2*n-1), "invalid unary marker count")
        _need(np.any(a[row,lo:hi+1] == structural), "bar has no permanent cells")
        low = marks == tuple(lo+2*j for j in range(n))
        high = marks == tuple(hi-2*j for j in reversed(range(n)))
        candidates = []
        if low and hi+n+1 < a.shape[1]:
            candidates.append(1)
        if high and lo-n-1 >= 0:
            candidates.append(-1)
        _need(len(candidates) == 1, "no unique feasible bypass")
        bars.append(dict(row=row,lo=lo,hi=hi,n=n,side=candidates[0],marks=marks,symmetric=low and high))
        allowed.update((row,x) for x in range(lo,hi+1))
    _need({tuple(p) for p in np.argwhere(a != bg)} <= allowed, "unexplained foreground")
    path,events = _route(a.shape,port,bars)
    _need(all(a[p] == bg for p in path[1:]), "route collides with input")
    return dict(a=a,turns=turns,bg=bg,structural=structural,energy=energy,
                tank=(t,b,l,r),port=port,coordinates=coordinates,capacity=len(coordinates),
                initial=len(coordinates)-spent,bars=bars,path=path,events=events)


def _decode(input_grid):
    g = _grid(input_grid)
    colors = set(map(int,np.unique(g)))
    _need(len(colors) == 3, "family uses three visible colors")
    bg = int(g[0,0])
    _need(all(int(g[p]) == bg for p in ((0,-1),(-1,0),(-1,-1))), "corners must identify background")
    candidates = []
    for structural in sorted(colors-{bg}):
        energy = next(iter(colors-{bg,structural}))
        for turns in range(4):
            try:
                candidates.append(_read_canonical(np.rot90(g,turns).copy(),bg,structural,energy,turns))
            except AssertionError:
                pass
    _need(len(candidates) == 1, "input lacks a unique complete explanation")
    return candidates[0]


def _execute(scene, reclaim=True, unlimited=False):
    """Unit-step input-only execution. Arrival/refuel precedes exhaustion testing."""
    output = scene["a"].copy()
    energy,structural,bg = (scene[k] for k in ("energy","structural","bg"))
    fuel = scene["initial"]
    reached,steps = [],0
    for step,p in enumerate(scene["path"][1:],1):
        if fuel == 0 and not unlimited:
            break
        if not unlimited:
            fuel -= 1
        output[p] = energy
        steps = step
        if step in scene["events"]:
            index = scene["events"][step]
            bar = scene["bars"][index]
            reached.append(index)
            output[bar["row"],list(bar["marks"])] = structural
            if reclaim and not unlimited:
                fuel += bar["n"]
                _need(fuel <= scene["capacity"], "reservoir overflow")
    if not unlimited:
        spent = scene["capacity"]-fuel
        for i,p in enumerate(scene["coordinates"]):
            output[p] = bg if i < spent else energy
    features = dict(steps=steps,fuel=fuel,reached=tuple(reached),
                    complete=steps == len(scene["path"])-1)
    return np.rot90(output,-scene["turns"]).copy(),features


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive roles, heading, commands, fuel, pickups, and output from input only."""
    return _execute(_decode(input_grid))[0]


def _span_options(width,current,n,side,allow_symmetric):
    d = n+1
    options = []
    for lo in range(current+1):
        for hi in range(max(current,lo+max(2,2*n-1)-1),width):
            length = hi-lo+1
            symmetric = length == 2*n-1
            if symmetric and (not allow_symmetric or n == 1):
                continue
            target = hi+d if side == 1 else lo-d
            other = lo-d if side == 1 else hi+d
            if 0 <= target < width and not (symmetric and 0 <= other < width):
                options.append((lo,hi,target,symmetric))
    return options


def _geometry(rng,width,initial,counts,sides,allow_symmetric):
    # At most 5*30 memoized states; one bounded permutation per visited state.
    failed = set()
    def search(index,current):
        if index == len(counts):
            return []
        if (index,current) in failed:
            return None
        choices = _span_options(width,current,counts[index],sides[index],allow_symmetric)
        order = list(map(int,rng.permutation(len(choices))))
        if allow_symmetric and rng.random() < .35:
            order.sort(key=lambda j:not choices[j][3])
        for j in order:
            lo,hi,target,sym = choices[j]
            tail = search(index+1,target)
            if tail is not None:
                return [(lo,hi,sym)]+tail
        failed.add((index,current))
        return None
    return search(0,initial)


def _ledger(path,events,bars,initial):
    """Bulk cost accounting used by construction; solver executes individual steps."""
    fuel,previous,reached = initial,0,[]
    for step,index in sorted(events.items()):
        cost = step-previous
        if fuel < cost:
            return previous+fuel,0,reached
        fuel = fuel-cost+bars[index]["n"]
        previous = step
        reached.append(index)
    cost = len(path)-1-previous
    used = min(fuel,cost)
    return previous+used,fuel-used,reached


def _make_pair(rng,colors,turns,counts,sides,lesson,allow_symmetric):
    bg,structural,energy = colors
    for attempt in range(96):
        a = int(rng.integers(2 if lesson in ("rescue","arrival_lesson") else 1,7))
        top = int(rng.integers(0,3))
        bottom = top+a+1
        minimum = bottom+sum(n+2 for n in counts)+2
        if minimum > 30:
            continue
        height = int(rng.integers(minimum,31))
        width = int(rng.integers(14,31)) if attempt < 64 else 30
        initial = int(rng.integers(2,width-2))
        spans = _geometry(rng,width,initial,counts,sides,allow_symmetric)
        if spans is None:
            continue
        slack = height-minimum
        if lesson in ("arrival_lesson","exact_exit"):
            # An empty arrival at the last pickup followed by a short exit
            # demonstrates event ordering and destination-limited stopping.
            if lesson == "arrival_lesson":
                exit_slack = int(rng.integers(min(slack,counts[-1]-2)+1))
            else:
                if slack < counts[-1]-1:
                    continue
                exit_slack = int(rng.integers(counts[-1]-1,slack+1))
            allocation = list(rng.multinomial(slack-exit_slack,np.full(len(counts),1/len(counts))))+[exit_slack]
        else:
            allocation = rng.multinomial(slack,np.full(len(counts)+1,1/(len(counts)+1)))
        previous = bottom
        bars = []
        for i,(n,side,(lo,hi,sym)) in enumerate(zip(counts,sides,spans)):
            row = previous+1+int(allocation[i])+n+1
            marks = tuple(lo+2*j if side == 1 else hi-2*j for j in range(n))
            bars.append(dict(row=row,lo=lo,hi=hi,n=n,side=side,marks=marks,symmetric=sym))
            previous = row
        port = (bottom,initial)
        path,events = _route((height,width),port,bars)
        crossings = sorted(events)
        length = len(path)-1
        total = sum(counts)
        if lesson == "rescue":
            # Die after finishing the second lateral leg, but before crossing
            # its bar row. Collecting at either bend would change this answer.
            low = max(crossings[0],crossings[1]-counts[1]-1-counts[0])
            high = crossings[1]-counts[0]-1
            if low > high:
                continue
            fuel = int(rng.integers(low,high+1))
        elif lesson == "arrival_lesson":
            fuel = crossings[-1]-sum(counts[:-1])
        elif lesson == "arrival":
            needed = max(step-sum(counts[:j]) for j,step in enumerate(crossings))
            fuel = max(needed,length-total+int(rng.integers(1,5)))
        elif lesson in ("exact_gate","chain"):
            # Arrive empty at a random gate; its collection must still occur.
            j = int(rng.integers(2 if lesson == "chain" else 0,len(counts)))
            fuel = crossings[j]-sum(counts[:j])
        elif lesson == "empty":
            fuel = 0
        elif lesson == "exact_exit":
            fuel = length-total
        else:
            fuel = int(rng.integers(1,length-total+5))
        # Select reservoir width after the route's energy requirement is known.
        # This avoids retries over huge arrays and permits thin/fresh reservoirs.
        minimum_width = max(2 if lesson in ("rescue","arrival_lesson") else 1,(fuel+a-1)//a)
        possible = []
        for w in range(minimum_width,width-3):
            if lesson in ("rescue","arrival_lesson") and a*w <= fuel:
                continue
            lefts = list(range(max(1,initial-w),min(initial-1,width-w-3)+1))
            if lefts:
                possible.append((w,lefts))
        if not possible:
            continue
        w,lefts = possible[int(rng.integers(len(possible)))]
        left = lefts[int(rng.integers(len(lefts)))]
        right = left+w+1
        coordinates = _counter(top,bottom,left,right)
        grid = np.full((height,width),bg,dtype=np.int8)
        for p in _border(top,bottom,left,right):
            grid[p] = structural
        grid[port] = energy
        for i,p in enumerate(coordinates):
            grid[p] = bg if i < len(coordinates)-fuel else energy
        for bar in bars:
            grid[bar["row"],bar["lo"]:bar["hi"]+1] = structural
            grid[bar["row"],list(bar["marks"])] = energy
        steps,final_fuel,reached = _ledger(path,events,bars,fuel)
        output = grid.copy()
        for p in path[1:steps+1]:
            output[p] = energy
        for j in reached:
            output[bars[j]["row"],list(bars[j]["marks"])] = structural
        for i,p in enumerate(coordinates):
            output[p] = bg if i < len(coordinates)-final_fuel else energy
        pair = {"input":np.rot90(grid,-turns).copy(),"output":np.rot90(output,-turns).copy()}
        # Frame completeness, collisions, and visible uniqueness are checked here.
        scene = _decode(pair["input"])
        predicted,features = _execute(scene)
        _need(np.array_equal(predicted,pair["output"]), "construction ledger disagrees with execution")
        if lesson == "rescue":
            _need(features["steps"] > fuel and 0 < len(reached) < len(bars) and final_fuel == 0,"uninformative rescue")
        return pair
    raise RuntimeError("bounded geometric construction exhausted")


def _key(g):
    a = np.asarray(g,dtype=np.int8)
    return a.shape,a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train,num_test = _counts(num_train,num_test)
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    colors = tuple(map(int,rng.choice(10,3,replace=False)))
    recolor = bool(rng.random() < .45)
    maximum = int(rng.integers(2,5))
    count_shift = bool(rng.integers(2))
    base = int(rng.integers(4))
    train_turns = (base,(base+1)%4)
    side = int(rng.choice((-1,1)))
    # The two mandatory lessons cover selective collection and boundary stopping.
    middle = list(range(2,maximum))
    arrival_counts = middle if middle else [int(rng.integers(1,maximum+1))]
    while len(arrival_counts) < 2:
        arrival_counts.append(int(rng.integers(2,maximum+1)))
    jobs = [("rescue",[1,maximum],[side,side],train_turns[0]),
            ("arrival_lesson",arrival_counts,[-side]*len(arrival_counts),train_turns[1])]
    for _ in range(num_train-2):
        k = int(rng.integers(1,3 if count_shift else 6))
        ns = list(map(int,rng.integers(1,maximum+1,size=k)))
        ss = list(map(int,rng.choice((-1,1),size=k)))
        mode = str(rng.choice(("random","arrival","exact_gate","exact_exit","empty"),p=(.4,.2,.2,.15,.05)))
        jobs.append((mode,ns,ss,int(rng.choice(train_turns))))
    puzzle,seen = {"train":[],"test":[]},set()
    for split,count in (("train",num_train),("test",num_test)):
        for i in range(count):
            if split == "train":
                mode,ns,ss,turns = jobs[i]
            else:
                k = int(rng.integers(3,6)) if count_shift else int(rng.integers(1,6))
                ns = list(map(int,rng.integers(1,maximum+1,size=k)))
                ss = list(map(int,rng.choice((-1,1),size=k)))
                turns = int(rng.choice(train_turns)) if count_shift else (base+int(rng.integers(2,4)))%4
                if count_shift:
                    mode = str(rng.choice(("arrival","exact_exit","chain"),p=(.4,.2,.4)))
                else:
                    mode = str(rng.choice(("arrival","exact_gate","exact_exit"),p=(.35,.45,.2)))
            # Very long high-clearance chains may not fit 30 cells: reduce only
            # unconstrained bars, never remove the taught maximum from lessons.
            if len(ns) >= 3:
                while sum(n+2 for n in ns) > 20:
                    choices = [j for j,n in enumerate(ns) if n > 1]
                    if not choices:
                        break
                    ns[int(rng.choice(choices))] -= 1
            for retry in range(12):
                pair_colors = tuple(map(int,rng.choice(10,3,replace=False))) if recolor else colors
                try:
                    symmetry_taught = split == "train" or any(
                        bar["symmetric"] for demo in puzzle["train"]
                        for bar in _decode(demo["input"])["bars"])
                    pair = _make_pair(rng,pair_colors,turns,ns,ss,mode,allow_symmetric=symmetry_taught)
                except RuntimeError:
                    if retry == 11:
                        raise
                    continue
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness construction exhausted")
    for split in puzzle:
        order = rng.permutation(len(puzzle[split]))
        puzzle[split] = [puzzle[split][int(j)] for j in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle,dict) and set(puzzle) == {"train","test"}, "incorrect puzzle schema")
    _need(all(isinstance(puzzle[s],list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]),len(puzzle["test"]))
    seen,records = set(),{"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            _need(isinstance(pair,dict) and set(pair) == {"input","output"}, "incorrect pair schema")
            grid,output = _grid(pair["input"]),_grid(pair["output"])
            _need(grid.shape == output.shape, "dimensions changed")
            key = _key(grid)
            _need(key not in seen,"duplicate input")
            seen.add(key)
            scene = _decode(grid)
            predicted,features = _execute(scene)
            _need(np.array_equal(predicted,output), "incorrect output")
            energy = scene["energy"]
            _need(np.count_nonzero(grid == energy) == np.count_nonzero(output == energy), "energy conservation failed")
            canonical = np.rot90(output,scene["turns"])
            # Physical structures and unvisited instructions must stay intact.
            for j,bar in enumerate(scene["bars"]):
                expected = scene["structural"] if j in features["reached"] else energy
                _need(np.all(canonical[bar["row"],list(bar["marks"])] == expected), "incorrect gate collection")
            coords = scene["coordinates"]
            expected_counter = [scene["bg"]]*(scene["capacity"]-features["fuel"])+[energy]*features["fuel"]
            _need([int(canonical[p]) for p in coords] == expected_counter,"counter update lost raster order")
            _need(features["steps"] == scene["initial"]+sum(scene["bars"][j]["n"] for j in features["reached"])-features["fuel"], "fuel ledger failed")
            records[split].append((scene,features))
    training = records["train"]
    _need(any(f["steps"] > s["initial"] and f["fuel"] == 0 and 0 < len(f["reached"]) < len(s["bars"]) for s,f in training), "training lacks selective fuel rescue")
    _need(any(f["complete"] and f["fuel"] > 0 for s,f in training), "training lacks destination stopping")
    _need(any(any(s["initial"]+sum(s["bars"][j]["n"] for j in f["reached"] if j < index) == step
                      and f["steps"] > step for step,index in s["events"].items() if index in f["reached"])
              for s,f in training), "training lacks empty-on-arrival refueling")
    _need(any(not f["complete"] and f["fuel"] == 0 and
              any(index not in f["reached"] and step-s["bars"][index]["n"]-1 <= f["steps"] < step
                  for step,index in s["events"].items()) for s,f in training),
          "training lacks a stop between lateral completion and collection")
    _need({s["bars"][j]["side"] for s,f in training for j in f["reached"]} == {-1,1}, "training lacks both bypass sides")
    _need(any(any(s["bars"][a]["side"] == s["bars"][b]["side"]
                      for a,b in zip(f["reached"],f["reached"][1:])) for s,f in training),
          "training lacks an executed repeated-side detour")
    _need(any(len({p[0] for p in s["coordinates"]}) > 1 and len({p[1] for p in s["coordinates"]}) > 1 and 0 < s["initial"] < s["capacity"] for s,f in training), "training lacks a readable 2D counter")
    taught = {b["n"] for s,f in training for b in s["bars"]}
    _need(len(taught) >= 2, "training lacks unary counting contrast")
    directions = {s["turns"] for s,f in training}
    train_max = max(len(s["bars"]) for s,f in training)
    for s,f in records["test"]:
        _need({b["n"] for b in s["bars"]} <= taught, "test uses undemonstrated marker count")
        _need(s["turns"] not in directions or len(s["bars"]) > train_max,"test lacks intended shift")
        _need(bool(f["reached"]),"test never uses a gate")
        if s["turns"] in directions:
            _need(len(f["reached"]) > train_max,"count shift does not affect the executed route")
        if any(b["symmetric"] for b in s["bars"]):
            _need(any(b["symmetric"] for demo,_ in training for b in demo["bars"]),"test symmetry was not demonstrated")
    # Explicit concept ablations: removing fuel or the collection feedback changes
    # a demonstrated answer. Ignoring B's detour also fails before the first bar.
    _need(any(not np.array_equal(_execute(s,unlimited=True)[0],_execute(s)[0]) for s,f in training), "ignoring A gives same training answers")
    _need(any(not np.array_equal(_execute(s,reclaim=False)[0],_execute(s)[0]) for s,f in training), "marker fuel has no demonstrated effect")
    _need(any(any(p[1] != s["port"][1] for p in s["path"][:f["steps"]+1]) for s,f in training), "ignoring B gives same route")


def _regressions():
    # Literal geometry: port (3,5), marker at (7,3), bypass x=8,
    # turn at y=5, pickup step 7. Eight initial units become nine steps
    # with one reclaimed unit, stopping at (9,8). The second bar is untouched.
    g = np.zeros((15,14),dtype=np.int8)
    g[0,2:10] = g[3,2:10] = 6
    g[0:4,2] = g[0:4,9] = 6
    g[3,5] = 2
    coords = [(2,x) for x in range(8,2,-1)]+[(1,x) for x in range(8,2,-1)]
    for p in coords[4:]:
        g[p] = 2
    g[7,3:7] = 6
    g[7,3] = 2
    g[12,6:11] = 6
    g[12,10] = g[12,8] = 2
    expected = g.copy()
    for p in coords:
        expected[p] = 0
    for p in ((4,5),(5,5),(5,6),(5,7),(5,8),(6,8),(7,8),(8,8),(9,8)):
        expected[p] = 2
    expected[7,3] = 6
    for turns in range(4):
        for palette in ((0,6,2),(7,1,9)):
            mapping = np.arange(10,dtype=np.int8)
            mapping[[0,6,2]] = palette
            _need(np.array_equal(solve(np.rot90(mapping[g],turns)),np.rot90(mapping[expected],turns)), "literal regression")
    # Exact-empty arrival at the first gate must refill before halting.
    exact = g.copy()
    exact[coords[4]] = 0  # seven units, seven steps to pickup
    output = solve(exact)
    _need(output[8,8] == 2 and output[9,8] == 0 and output[7,3] == 6,"refuel timing regression")
    # Fully alternating two-marker gate: only low bypass fits in width 14.
    symmetric = np.zeros((15,14),dtype=np.int8)
    symmetric[0,5:13] = symmetric[3,5:13] = 6
    symmetric[0:4,5] = symmetric[0:4,12] = 6
    symmetric[3,11] = 2
    sym_coords = [(2,x) for x in range(11,5,-1)]+[(1,x) for x in range(11,5,-1)]
    for p in sym_coords[4:]:
        symmetric[p] = 2
    symmetric[7,10:13] = (2,6,2)
    symmetric[12,6:11] = (6,6,2,6,2)
    expected_sym = symmetric.copy()
    for p in sym_coords:
        expected_sym[p] = 0
    for p in ((4,11),(4,10),(4,9),(4,8),(4,7),(5,7),(6,7),(7,7),(8,7),(9,7)):
        expected_sym[p] = 2
    expected_sym[7,10:13] = 6
    for turns in range(4):
        _need(np.array_equal(solve(np.rot90(symmetric,turns)),np.rot90(expected_sym,turns)),"one-exit symmetry regression")
    ambiguous = symmetric.copy()
    ambiguous[3,11],ambiguous[3,9] = 6,2
    ambiguous[7,10:13] = 0
    ambiguous[7,8:11] = (2,6,2)
    try:
        solve(ambiguous)
    except AssertionError:
        pass
    else:
        raise AssertionError("accepted a symmetric bar with two feasible exits")


def self_test():
    """42 fixed seeds, streaming measurements; no grids or traces are saved."""
    _regressions()
    signatures,shapes,structures,palettes = set(),set(),set(),set()
    directions,counts,markers,stops,phases,capacities = set(),set(),set(),set(),set(),set()
    transitions,symmetries,refill_cases,identity = set(),0,0,0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train","test"):
            for pair in puzzle[split]:
                s = _decode(pair["input"])
                out,f = _execute(s)
                digest.update(np.asarray(pair["input"]).tobytes())
                digest.update(np.asarray(pair["output"]).tobytes())
                shapes.add(tuple(pair["input"].shape))
                norm = np.zeros(s["a"].shape,dtype=np.int8)
                norm[s["a"] == s["structural"]] = 1
                norm[s["a"] == s["energy"]] = 2
                structures.add(_key(norm))
                palettes.add((s["bg"],s["structural"],s["energy"]))
                directions.add((split,s["turns"]))
                counts.add((split,len(s["bars"])))
                markers.update(b["n"] for b in s["bars"])
                capacities.add(s["capacity"])
                sides = [b["side"] for b in s["bars"]]
                transitions.update(zip(sides,sides[1:]))
                symmetries += sum(b["symmetric"] for b in s["bars"])
                refill_cases += f["steps"] > s["initial"]
                identity += np.array_equal(pair["input"],out)
                stops.add("surplus" if f["complete"] and f["fuel"] else "exact" if f["complete"] else "empty" if s["initial"] == 0 else "depleted")
                if 0 < f["steps"] < len(s["path"])-1:
                    p,q = s["path"][f["steps"]:f["steps"]+2]
                    phases.add("lateral" if p[0] == q[0] else "forward")
        signatures.add(digest.hexdigest())
    _need(len(signatures) == 42 and len(shapes) >= 45 and len(structures) >= 180,"insufficient structural variation")
    _need(len(palettes) >= 25 and len(capacities) >= 25,"insufficient palette/capacity variation")
    _need(directions == {(split,k) for split in ("train","test") for k in range(4)}, "missing orientation")
    _need(markers == {1,2,3,4} and phases == {"lateral","forward"},"missing marker/stop phases")
    _need(transitions == {(-1,-1),(-1,1),(1,-1),(1,1)} and symmetries > 0,"missing endpoint structures")
    _need({"surplus","exact","depleted"} <= stops and refill_cases >= 42,"missing stopping/refill behavior")
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            validate_generated_puzzle(generate_puzzle(177,a,b))
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p,q = generate_puzzle(19,2,1),generate_puzzle(19,2,1)
    _need(all(np.array_equal(x[k],y[k]) for split in p for x,y in zip(p[split],q[split]) for k in x),"nondeterministic")
    broken = {split:[{k:v.copy() for k,v in pair.items()} for pair in p[split]] for split in p}
    broken["test"][0]["output"][0,0] = (int(broken["test"][0]["output"][0,0])+1)%10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted corrupted output")
    return dict(seeds=42,pairs=210,distinct_puzzles=len(signatures),shapes=len(shapes),
                normalized_structures=len(structures),palettes=len(palettes),capacities=len(capacities),
                direction_split_cells=len(directions),bar_counts=sorted(counts),marker_counts=sorted(markers),
                stop_phases=sorted(phases),stops=sorted(stops),symmetric_bars=symmetries,
                pairs_extended_by_collection=refill_cases,identity_pairs=identity)


if __name__ == "__main__":
    print(json.dumps(self_test(),separators=(",",":")))
