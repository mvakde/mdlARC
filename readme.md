Update 13 dec: ***27.5%*** now. Looking at the output grids, I expect 50% with scaling. Also gonna run this on ARC-2 today
~~***Update: 20% right now***~~
~~***(In progress, I expect improvements till 30%)***~~
# 10% on ARC-1 for less than a dollar using a 1M transformer

This already beats the pareto frontier btw

## Self supervised compression on ARC

Every DL approach on ARC today trains a supervised algorithm[1]

This is dumb.  
A self-supervised compression step will obviously perform better:
- There is new information in the input grids and private puzzles that is currently uncompressed
- Test grids have distribution shifts. Compression will push these grids into distribution


For more reasoning behind the approach, read **[My Blog](https://mvakde.github.io/blog/why-all-ARC-solvers-fail-today/)**

## Details
Performance - 10% on ARC-1 public eval  
Total compute cost - **$0.709**
- 52m on A100 for training (0.7$)
- 40s on A100 for inference (0.009$)

This is early performance. Haven't run all ablations yet

I should be able to push to 30% on ARC-1 and 8% on ARC-2




[1]: CompressARC is an exception, but that compresses each task individually. Mine jointly compresses all tasks together