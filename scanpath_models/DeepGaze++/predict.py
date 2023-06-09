from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import glob
from PIL import Image, ImageDraw
import sys, csv


import deepgaze_pytorch

DEVICE = 'cuda'

try:
    number_scanpaths = int(sys.argv[1])
except:
    print("Please define the number of scanpaths you wish to predict, it has to be an integer.")
    exit()

# the fixations parameters indicates the number of previous fixations to take into account
# to compute the probability of the next fixations
# it is set to 1 at the first recursive step
def prediction(image, fixations_x, fixations_y, centerbias, fixations, mask):

    # location of previous scanpath fixations in x and y (pixel coordinates), starting with the initial fixation on the image.
    fixation_history_x = np.array(fixations_x)
    fixation_history_y = np.array(fixations_y)

    model = deepgaze_pytorch.DeepGazeIII(fixations, pretrained=True).to(DEVICE)

    image_tensor = torch.tensor([image[:,:,:3].transpose(2, 0, 1)]).to(DEVICE)
    centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)
    x_hist_tensor = torch.tensor([fixation_history_x[model.included_fixations]]).to(DEVICE)
    y_hist_tensor = torch.tensor([fixation_history_x[model.included_fixations]]).to(DEVICE)

    log_density_prediction = (100 + model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)) \
                                * mask.to(DEVICE).unsqueeze(0).unsqueeze(0) - (1 - mask.to(DEVICE)) * 1000

    # Find the brightest pixel in the probaility map
    brightest_pixel = (log_density_prediction==torch.max(log_density_prediction)).nonzero()[0].detach().cpu().numpy()

    return brightest_pixel


def create_circular_mask(h, w, fixations_x, fixations_y, radius):

    # get the circular mask
    mask = torch.zeros(h, w)
    Y, X = np.ogrid[:h, :w]
    for i in range(len(fixations_x)):
        dist = np.sqrt((X - fixations_x[i])**2 + (Y - fixations_y[i])**2)
        mask = torch.maximum(mask, torch.from_numpy(dist <= radius) * (1 - 1/10 * (len(fixations_x) - i - 1)))

    return 1 - mask


# Obtain all images on which the prediction shall be applied in the input folder
extensions = ("*.png","*.jpg","*.jpeg",)
paths = []
for extension in extensions:
    paths.extend(glob.glob("inputs/*"+extension))



with open('./predicted_result.csv', 'w') as wfile:

    writer = csv.writer(wfile)
    writer.writerow(["image", "width", "height", "username", "x", "y", "timestamp"])

    for p in range(len(paths)):
        path = paths[p]

        # image = imread(path)
        image = Image.open(path).convert('RGB')
        w = image.size[0]
        h = image.size[1]

        image = image.resize((int(w/2.5), int(h/2.5)), Image.ANTIALIAS)
        w = image.size[0]
        h = image.size[1]
        image = np.array(image)
        

        image_file_name = path.split("/")[1].split(".")[0]

        # load precomputed centerbias log density (from MIT1003) over a 400x1024 image
        # you can download the centerbias from https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
        # alternatively, you can use a uniform centerbias via `centerbias_template = np.zeros((1024, 1024))`.
        centerbias_template = np.load('centerbias_mit1003.npy')
        
        # rescale to match image size
        centerbias = zoom(centerbias_template, (image.shape[0]/centerbias_template.shape[0], image.shape[1]/centerbias_template.shape[1]), order=0, mode='nearest')
        
        # renormalize log density
        centerbias -= logsumexp(centerbias)

        # Fixation history set as the center of the image
        center_height = image.shape[0] // 2
        center_width = image.shape[1] // 2

        # Result scanpath
        fixations_x = [center_width]
        fixations_y = [center_height]
        fixations = 1

        # Recursively predict the fixations until we have predicted 13 of them
        while len(fixations_x) < number_scanpaths:

            # Inhibition of Return (IOB)
            radius = int(0.2 * min(image.shape[1], image.shape[0]))
            mask = create_circular_mask(image.shape[0], image.shape[1], fixations_x, fixations_y, radius)
            brightest_pixel = prediction(image * mask.unsqueeze(2).numpy().astype('uint8'), fixations_x, fixations_y, centerbias, fixations, mask)

            # add the newly predicted pixel to the history of fixations
            fixations_x.append(brightest_pixel[3])
            fixations_y.append(brightest_pixel[2])

            # For each new fixation point we increase the fixations parameter by 1 until it hits 4
            if fixations <= 3:
                fixations += 1

        # write into the csv file
        for i in range(len(fixations_x)):
            x = fixations_x[i]
            y = fixations_y[i]
            writer.writerow([path.split("/")[1], w, h, 'test', x, y, 0.0])

