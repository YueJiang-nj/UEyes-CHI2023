# Data Processing

## Generate Heatmaps

```bash
python generate_heatmaps.py -ic ./eyetracker_logs -ib ./images -o ./output/ -s 1920x1200 -t data
python aggregate_data.py -w 1
```
where the folder ''eyetracker_logs'' is the path to the directory containing the CSV files in the form 'xx_KHyyy_fixations.csv', where xx is the block number and YYY is the participant ID, the folder ''images'' is the path to the directory containing the image blocks for the experiment.


## Generate Scanpath

```bash
python generate_heatmaps.py -ic ./eyetracker_logs -ib ./images -o ./output/ -s 1920x1200 -t path
```
where the folder ''eyetracker_logs'' is the path to the directory containing the CSV files in the form 'xx_KHyyy_fixations.csv', where xx is the block number and YYY is the participant ID, the folder ''images'' is the path to the directory containing the image blocks for the experiment.
