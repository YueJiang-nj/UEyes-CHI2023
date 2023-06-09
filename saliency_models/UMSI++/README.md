# UMSI++


## Instructions to run the program

1. Create conda environment:
```bash
conda env create -f environment.yaml 
```

2. Activate the environment. 
```bash
conda activate umsi++
```

3. Install Jupyter notebook if you do not have it installed, and register the conda environment as a Jupyter kernel by running the following commands:
```bash
conda install ipykernel
python -m ipykernel install --user --name=umsipp
```

4. Run the Jupyter notebook and change kernel to 'umsipp' (Kernel -> Change Kernel -> umsipp) to train and evaluate the model.

5. change the following rows to the folders for input images and ground truth saliency maps.
```
img_filenames_ours = glob.glob('./images/*g')
imp_filenames_ours = glob.glob('./saliency_gt/*g')
```

### Note: remember to put umsi++.hdf5 to the folder weights.