# mdlARC Task Atlas

A landscape, zero-dependency visualiser for the JSON tasks in `assets_tmp`.

## Run

From the repository root:

```bash
python3 visualiser/server.py
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765).

Use left/right to move through tasks. Use up/down to move through pages of 12
pairs. The arrow keys, trackpad/mouse wheel, selectors, and on-screen controls
all provide the same navigation. Click either grid in a pair to enlarge it.

The server indexes filenames only at startup. A selected task is parsed on
demand, only its current 12-pair page is sent to the browser, and a four-task
LRU keeps recent tasks instantaneous. Directories containing `raw/` or `tasks/`
are detected automatically; other dataset folders may contain per-task JSON
files directly. This covers ARC, ConceptARC, RE-ARC, and NVARC-style list files.

Options:

```bash
python3 visualiser/server.py --help
python3 visualiser/server.py --port 9000 --cache-size 6
python3 visualiser/server.py --source /path/to/assets_tmp
```
