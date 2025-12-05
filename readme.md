***In progress, I expect improvements till 30%***
# 10% on ARC-1 with a 1M transformer


## Self supervised compression on ARC

Every DL approach on ARC today trains a supervised algorithm

This is stupid.  
A self-supervised compression step will obviously perform better:
- There is new information in the input grids and private puzzles that is currently uncompressed
- Test grids have distribution shifts. Compression will push these grids into distribution

---

## Details
Performance - 10% on ARC-1 public eval
Total compute cost - **$0.709**
- 52m of A100 for training (0.7$)
- 40s of A100 for inference (0.009$)

This is early performance. Haven't run all ablations yet

I should be able to push to 30% on ARC-1 and 8% on ARC-2  

For the reasoining behind the approach, read **[My Blog](https://mvakde.github.io/blog/why-all-ARC-solvers-fail-today/)**