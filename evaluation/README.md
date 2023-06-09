# Saliency evaluation

## Evaluate heatmaps

Prepare a folder with all reference (groundtruth) heatmaps and a folder with the predicted heatmaps.

IMPORTANT: The filenames should be the same in each folder, in order to automatically pair them for comparison.

The heatmaps must be saved as JPG or PNG images.

Run this command to get several evaluation metrics printed on the terminal: 
```sh
python3 eval.py --heatmaps --ref_dir /path/reference/heatmaps/ --pred_dir /path/predicted/heatmaps/
```

You can also compare a small group of files with [glob patterns](https://en.wikipedia.org/wiki/Glob_(programming)):
```sh
python3 eval.py --heatmaps --ref_files /path/reference/heatmap0?.jpg --pred_files /path/predicted/heatmap0?.jpg
```

Even you can compare individual heatmaps:
```sh
python3 eval.py --heatmaps --ref_files heatmap01_pred.jpg --pred_files heatmap01_pred.jpg
```


## Evaluate scanpaths 

Prepare a CSV file with all reference (groundtruth) scanpaths and another CSV file with the predicted scanpaths, 
both files having the following columns: `image, width, height, username, x, y, timestamp, duration`.

You can use this command to convert the eye tracking logs to the expected CSV format:
```sh
python3 gp3_logparser.py --log_dir /path/to/logs --img_dir /path/to/imgs > ref_scanpaths.csv
```

NOTE: The predicted scanpaths depend on the output of the computational model (e.g. PathGAN, DeepGaze, etc.)
so an ad-hoc post-processing step is necessary.

Finally, run this command to get several evaluation metrics printed on the terminal: 
```sh
python3 eval.py --scanpaths --ref_files ref_scanpaths.csv --pred_files pred_scanpaths.csv
```
