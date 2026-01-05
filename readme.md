# 39% on ARC-AGI-1 Pub in just ~$2 using a 60M transformer

Update: I am in the middle of refactoring the code. Performance has improved to 36% for the same cost. Doing a final push towards 50% now. The repo might be a bit unusable till I am done with refactoring and improving the model. Expect 1st week of Jan.

---

Deploy:
- Colab: upload one of the notebooks → select GPU runtime (A100 if available) → (optional) mount google drive to save runs -> “Run all”.
- Modal: upload the notebook to a Modal Notebook → select an A100 → (optional) attach a Volume and update the first cell’s `mount_folder`/volume name → “Run all”.
- If you’re running from a fork/branch, edit the `git clone ...` cell in the notebook.

---

Update:  
Wow this blew up. Pressure is on.  
Please bear with me as I want to do careful ablations.

## Self supervised compression on ARC

Every DL approach on ARC today trains a supervised algorithm (other than compressARC)

I think this is suboptimal.  
A self-supervised compression step will obviously perform better:
- There is new information in the input grids and private puzzles that is currently uncompressed
- Test grids have distribution shifts. Compression will push these grids into distribution

Implementation details: [New pareto frontier on ARC-AGI](https://mvakde.github.io/blog/new-pareto-frontier-arc-agi/)
For why I chose these specific implementations, read my blog on [Why all ARC solvers fail today](https://mvakde.github.io/blog/why-all-ARC-solvers-fail-today/)

## Details
Performance - 27.5% on ARC-1 public eval
Total Compute cost - $1.8
- ~127min on 40GB A100 for training (1.2$)
- ~49min on 80GB A100 for inference (0.6$)


This is early performance. I was too GPU poor to do hyperparameter sweeps.

I should be able to push to 35% with just basic sweeps

I expect to hit 50% with a few obvious research ideas
