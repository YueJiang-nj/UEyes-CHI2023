# PathGAN++

## Setup

1. Create the conda environment using environment.yml.

2. If the given environment.yml does not work, install libraries: numpy, tensorflow=2.9.1, sklearn, matplotlib

3. Put the input images into the **inputs** folder.

## Prediction

1. If your dataset includes fixation durations (CSV with 6 columns), then set reduced=False in line 158.
2. Place all the images for which you want to have a prediction in the **inputs** folder. Accepted formats: jpg, jpeg, or png.
3. Indicate the Generator weight file in **predict.py**:
```python
weights_generator = "weights/generator_PathGAN++"
```

4.  Execute the prediction:
```bash
python predict.py 
python generate_predicted_results.py
python subsample_seq.py
```

5. The results will be delivered as a csv file ./outputs/final_predicted_results.csv.

## Training

If your dataset includes the fixation durations (CSV with 6 columns), then set reduced=False in line 191 of **train.py**.
It is set to True by default.

### Training from scratch

Finally, execute the training script:
```bash
python train.py 
```

