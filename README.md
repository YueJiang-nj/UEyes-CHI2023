# UEyes

UEyes is a large eye-tracking-based dataset including 62 participants looking at 1,980 UI screenshots, covering four major UI types: webpage, desktop UI, mobile UI, and poster.


**Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems (CHI 2023)**

Authors: **Yue Jiang, Luis A. Leiva, Hamed R. Tavakoli, Paul R. B. Houssel, Julia Kylmälä1, Antti Oulasvirta**

**Project page:** https://yuejiang-nj.github.io/Publications/2023CHI_UEyes/project_page/main.html


## Dataset Overview

Our dataset is available here: https://zenodo.org/record/8010312

The dataset has the following file layout:

    images/
    eyetracker_logs/
    saliency_maps/
        fixmaps_1s/
        fixmaps_3s/
        fixmaps_7s/
        heatmaps_1s/
        heatmaps_3s/
        heatmaps_7s/
        overlay_heatmaps_1s/
        overlay_heatmaps_3s/
        overlay_heatmaps_7s/
    scanpaths/
        paths_1s/
        paths_3s/
        paths_7s/
    info.csv
    README.md
    
**images** contains all the UI screenshots used in the dataset.

**eyetracker_logs** contains all the raw logs outputed from the eye tracker. The csv file XX_hk0YY_fixations.csv includes the participant YY looking at the block XX. The detailed explanation of each column in the files is here: https://www.gazept.com/dl/Gazepoint_API_v2.0.pdf.

**sliency_maps** contains multi-duration fixation maps, heatmaps, and heatmaps overlaid with their corresponding UI images.

**scanapths** contains multi-duration scanpaths on UI images. Since each UI was looked at by multiple participants, we have one folder for each UI image. The folder name corresponds to the UI image name. Inside each folder, the image name N.png shows the scanpath of the participant N looking at this UI image.

**info.csv** contains the each image name, its category (ebpage, desktop UI, mobile UI, and poster), the block number it belongs to in our experiment, and whether it is in the trianing or test dataset.


## Code

This repo has following file layout:

    data_processing/
    evaluation/
    saliency_models/
        UMSI++/
    scanpath_models/
        DeepGaze++/
        PathGAN++/
    README.md
    
**data_processing** contains files to generate heatmaps and scanpaths from the raw eye tracking data.

**evaluation** contains the implementation all the evaluation metrics for heatmaps and scanpaths.

**saliency_models** contains our improved model UMSI++. 

**scanpath_models** contains our improved model DeepGaze++ and PathGAN++.


## Citation

If you use UEyes, please use the following citation:

- Jiang, Yue, Luis A. Leiva, Hamed Rezazadegan Tavakoli, Paul RB Houssel, Julia Kylmälä, and Antti Oulasvirta. "UEyes: Understanding Visual Saliency across User Interface Types." In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems, pp. 1-21. 2023.

```bib
@inproceedings{jiang2023ueyes,
    author = {Jiang, Yue and Leiva, Luis A. and Rezazadegan Tavakoli, Hamed and R. B. Houssel, Paul and Kylm\"{a}l\"{a}, Julia and Oulasvirta, Antti},
    title = {UEyes: Understanding Visual Saliency across User Interface Types},
    year = {2023},
    isbn = {9781450394215},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3544548.3581096},
    doi = {10.1145/3544548.3581096},
    booktitle = {Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems},
    articleno = {285},
    numpages = {21},
    keywords = {Human Perception and Cognition, Eye Tracking, Computer Vision, Deep Learning, Interaction Design},
    location = {Hamburg, Germany},
    series = {CHI '23}
}
```