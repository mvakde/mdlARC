# 44% on ARC-AGI-1: trained from scratch for just ~$0.67
- Takes 2hrs on a 5090
- Uses a standard tranformer
- 75M parameters

This is one of the **best non-LLM scores** in the world today (if not THE best).  
It is also the **cheapest**, by far, at that performance.   

Details: [Blog](https://mvakde.github.io/blog/44-on-arc-1/), [X thread](https://x.com/evilmathkid/status/2029519274835148829)   


<a href="https://mvakde.github.io/blog/44-on-arc-1/"><img src="graph.png"></a>



### New score
Performance: **44%** on ARC-1 public eval  
Total compute cost: **~$0.67**  (2hrs on a 5090 rented on vast.ai)

### Old score
Performance: 27.5% on ARC-1 public eval  
Total Compute cost: $1.8 (<3hrs on an A100 rented on Google Colab)
<!-- 
**Next goal:**  
50% should be possible with the next research ideas -->

## Deployment
1) Rent a 5090, ensure cuda >12.8, ideally >13.0   
2) Create a virtual environment and install `torch`, `numpy`, `numba`, `matplotlib` and `flash-attn`  
3) Download and build the dataset
4) (optional) delete raw data, solutions file and dataset scripts to prove no leakage
5) Run the training and inference script  

This script takes care of (3)-(5):  
```bash
git clone https://github.com/mvakde/mdlARC.git

# download and build the datasets
cd mdlARC/dataset_building_scripts
python download_and_group.py
python build_datasets.py arc1 --add-conceptarc --with-filtered
cd ..

# prove no data leakage (optional, uncomment to run)
# rm -r assets_tmp # deletes raw data
# rm assets/solutions.json # deletes solutions file
# rm -r dataset_building_scripts # deletes dataset related files

#run the training + inference script
python run_script.py high # Choose between 3 modes: low, medium, high
```
 Note: To get the best speed, I have disabled logging loss values. Feel free to add it back

## Citation

```bibtex
@misc{vakde2025mdlarc,
  author       = {Mithil Vakde},
  title        = {mdlARC},
  year         = {2025},
  url          = {https://github.com/mvakde/mdlARC},
}
```