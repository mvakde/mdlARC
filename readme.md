# 27.5% on ARC-AGI in just $2 using a 28M transformer
Pareto frontier literally off the charts
<a href="https://mvakde.github.io/blog/new-pareto-frontier-arc-agi/"><img src="img/arc-prize-leaderboard.png"></a>

## How to run
- upload the `run-script.ipynb` file to google colab or modal
- choose A100
- Hit run all

## Self supervised compression on ARC

Every DL approach on ARC today trains a supervised algorithm[1]

This is dumb.  
A self-supervised compression step will obviously perform better:
- There is new information in the input grids and private puzzles that is currently uncompressed
- Test grids have distribution shifts. Compression will push these grids into distribution

Implementation details: [New pareto frontier on ARC-AGI](https://mvakde.github.io/blog/new-pareto-frontier-arc-agi/)
For more reasoning behind the approach, read my blog on **[Why all ARC solvers fail today](https://mvakde.github.io/blog/why-all-ARC-solvers-fail-today/)**

## Details
Performance - 27.5% on ARC-1 public eval
Total Compute cost - **$1.8**
- ~127min on 40GB A100 for training (1.2$)
- ~49min on 80GB A100 for inference (0.6$)


This is early performance. I was too GPU poor to do hyperparameter sweeps.

I should be able to push to 35% with just basic sweeps

I expect to hit 50% with a few obvious research ideas




[1]: CompressARC is an exception, but that compresses each task individually. Mine jointly compresses all tasks together. This gives better performance at lower cost, and is more "bitter lesson" pilled.