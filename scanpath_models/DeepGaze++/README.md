# DeepGaze++


## Instructions to run the program

1. Create a conda environment
```bash
conda create -n deepgaze++
```

2. Install all the libraries.
```bash
conda install -c conda-forge imageio
conda install -c conda-forge matplotlib
conda install scipy
conda install pytorch -c pytorch
```

3. Put the input images into the input folder

4. Execute the prediction script, by defining the number of fixations you wish to obtain for each image.
```bash
python3 predict.py <numberOfFixationsToPredict>
```
For example, if you wish to predict 6 fixation points for each image, 
```bash
python3 predict.py 6
```

