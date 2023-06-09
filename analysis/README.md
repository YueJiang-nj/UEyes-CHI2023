# Saliency analysis

## Generate fixations dataset

First of all, we need a consolidated CSV file with minimal but relevant information:
`image, width, height, username, x, y, timestamp, duration`.

Therefore we must parse all the eye-tracking logs. It may take some time:
```sh
python3 gp3_logparser.py --log_dir /path/to/logs --img_dir /path/to/imgs > dataset.csv
```

IMPORTANT: You can filter out fixations that occurred before of after some seconds, 
or even after of before a number of fixations.

**Example 1:** Consider only fixations that happended during the first 1 second of free viewing:
```sh
python3 gp3_logparser.py --log_dir /path/to/logs --img_dir /path/to/imgs --t_max 1 > dataset-until1s.csv
```

**Example 2:** Consider only fixations that happended after the first 3 seconds of free viewing:
```sh
python3 gp3_logparser.py --log_dir /path/to/logs --img_dir /path/to/imgs --t_min 3 > dataset-after3s.csv
```

**Example 3:** Consider only fixations that happended between the first 1 and 2 seconds of free viewing:
```sh
python3 gp3_logparser.py --log_dir /path/to/logs --img_dir /path/to/imgs --t_min 1 --t_max 2 > dataset-between1-2s.csv
```

**Example 4:** Consider only the first fixation of free viewing:
```sh
python3 gp3_logparser.py --log_dir /path/to/logs --img_dir /path/to/imgs --f_max 1 > dataset-first-fixation.csv
```

**Example 5:** Consider only images in the `web` category:
```sh
python3 gp3_logparser.py --log_dir /path/to/logs --img_dir /path/to/imgs --category web > web_dataset.csv
```

## Analyze color bias

Compute the frequency count, with pixel-level precision, of _all_ colors in an image or set of images:
```sh
python3 colorhist_image.py --img_dir /path/to/imgs > hist-img.json
```

Compute the frequency count, with pixel-level precision, of _fixated_ colors in an image or set of images:
```sh
python3 colorhist_fixations.py --csv_file dataset.csv --img_dir /path/to/imgs > hist-fix.json
```

Remember that we can analyze the fixated colors until 1s of free viewing, after 2 seconds, etc.
We just need to generate the right datasets with `gp3_logparser.py`.

Next, compute most frequent colors, both in visual and textual form:
```sh
python3 color_bias.py --json_file hist-img.json --save --frequency > hist-img.csv
python3 color_bias.py --json_file hist-fix.json --save --frequency > hist-fix.csv
```

If `--save` is provided, an image with the top-K color usage will be created. 
K can be changed with the `--num_colors` option (default: 16).

You can add an outline to the saved image with the `--outline` option.

If `--frequency` is provided, a CSV report will be output to stdout.
That's why it the previous example we redirected each command to a file.

Finally compute boxplots of image vs fixated colors:
```sh
Rscript color_bias.r --image hist-img.csv --fixated hist-fix.csv
```

### Bonus: Analyze popular colors

Extract the dominant color and the color palette from an image:
```sh
python3 popularcolors.py --img image.jpg [--outline --save]
```


## Analyze location bias

Generate a PDF plot with the distribution of fixations on screen:
```sh
Rscript location_bias.r -f dataset.csv
```

Again, `dataset.csv` must contain the following columns: 
`image, width, height, username, x, y, timestamp, duration`.


### Quadrant analysis

Generate first a CSV file with the fixations that happened on each screen quadrant:
```sh
python3 quadrants.py --log_dir ./et-logs/ --img_dir ./et-imgs/ > quadrants.csv
```

This CSV file has just three columns: `image, username, quadrant`.

You can use the same CLI options as in `gp3_logparser.py` above;
i.e. `--t_min`, `--t_max`, etc.

Then run statistical tests: 
```sh
Rscript quadrants.r -f quadrants.csv -t "Up to 7 seconds"
Rscript quadrants.r -f quadrants-first-fixation.csv -t "First fixation"
Rscript quadrants.r -f quadrants-first1s.csv -t "Up to 1 second"
Rscript quadrants.r -f quadrants-first3s.csv -t "Up to 3 seconds"
```

