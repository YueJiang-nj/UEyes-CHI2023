##################################
# 
# Usage example:
# python generate_heatmaps.py -ic ./eyetracker_logs -ib ./images -o ./output/ -s 1920x1200 -t data
# python generate_heatmaps.py -ic ./eyetracker_logs -ib ./images -o ./output/ -s 1920x1200 -t heatmap
# python generate_heatmaps.py -ic ./eyetracker_logs -ib ./images -o ./output/ -s 1920x1200 -t path
#
##################################

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import pickle 


def get_info_from_path(csv_path: str) -> Dict[str, str]:
    """
    Gives information about the block number and participant ID
    from the name of the CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file
    """
    block, participant_id = os.path.basename(csv_path).split("_")[:2]
    return {"block": block, "participant_id": participant_id[2:]}


def get_image_path_from_csv(csv_path: str, images_dir: str, img_name: str) -> str:
    """
    Gives the path of the image from the path of the CSV file and the
    image directory containing the blocks.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file
    images_dir : str
        Path to the directory containing the blocks directory
    img_name : str
        Name of the image file

    Returns
    -------
    str:
        Full image path in the images_dir
    """
    info = get_info_from_path(csv_path)
    block = info["block"]
    block_str = f"block {int(block):d}" if block != "c" else "calibration"
    img_path = os.path.join(images_dir, f"{block_str}", img_name)
    return img_path


def padding(array: np.array, xx: int, yy: int) -> np.array:
    """
    Gets symmetric padding for an image array

    Parameters
    ----------
    array : np.array
        Array to be padded
    xx : int
        Horizontal padding
    yy : int
        Vertical padding

    Returns
    -------
    np.array:
        Padded array
    """
    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb), (0, 0)), mode="constant"), a, b


def get_optimal_padded_img(arr: np.array, aspect_ratio: float) -> np.array:
    """
    Gives an optimally padded image array for the specified aspect ratio

    Parameters
    ----------
    arr : np.array
        Array to be optimally padded
    aspect_ratio : float
        Desired aspect ratio

    Returns
    -------
    np.array:
        Padded array
    """
    x = arr.shape[0]
    y = arr.shape[1]

    xx = int(y / aspect_ratio)
    yy = int(aspect_ratio * x)

    if xx < x:
        xx = x
        yy = int(aspect_ratio * xx)
    if yy < y:
        yy = y
        xx = int(yy / aspect_ratio)

    return padding(arr, xx, yy)


@dataclass
class Coordinate:
    x: float
    y: float
    time: Optional[float] = None

    def transformation_matrix(self) -> np.array:
        """
        Gets the transformation matrix for a scaling transformation
        """
        return np.array([[self.x, 0], [0, self.y]])

    def inverse_transformation_matrix(self) -> np.array:
        """
        Gets the inverse transformation matrix for a scaling transformation
        """
        return np.array([[1 / self.x, 0], [0, 1 / self.y]])

    def scale(self, other: np.array) -> "Coordinate":
        """
        Scales the current coordinate with the specified scalar linear transformation

        Parameters
        ----------
        other : np.array
            Scaling transformation matrix

        Returns
        -------
        Coordinate:
            Scaled coordinates
        """
        result = other * np.array([self.x, self.y]).T
        return Coordinate(x=result[0][0], y=result[1][1], time=self.time)


def get_xyt_from_coordinates(coord: List[Coordinate]) -> np.array:
    """
    Extracts the position and time from a Coordinate object

    Parameters
    ----------
    coord : List[Coordinate]
        Coordinate object to be extracted

    Returns
    -------
    np.array:
        Extracted information with 0th position as X, 1st position as Y,
        and 2nd position as TIME
    """
    return np.array([(c.x, c.y, c.time) for c in coord]).T


class ImageData:
    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        img_name: str,
        screen_size: Tuple[int, int],
    ) -> None:
        """
        ImageData object contains everything related to an image seen by the participant

        From the gaze fixation points, the actual image, the displayed image, and their
        paths. It contains all the necessary information to process the data accordingly

        Parameters
        ----------
        csv_path : str
            Path of the CSV file
        images_dir : str
            Path of the directory containing the blocks directories
        img_name : str
            File name of the image file
        screen_size : Tuple[int, int]
            Size of the screen in (WIDTH, HEIGHT) format
        """
        self.df_path = csv_path
        self.images_dir = images_dir
        self.orig_df = pd.read_csv(self.df_path)
        self.orig_df.rename({self.orig_df.columns[3]: "TIME"}, axis=1, inplace=True)
        self.img_name = img_name
        self.screen_size = Coordinate(x=screen_size[0], y=screen_size[1])
        self.df = self.orig_df[self.orig_df["MEDIA_NAME"] == self.img_name].dropna(
            subset=["MEDIA_NAME", "TIME", "FPOGD", "BPOGV", "BPOGX", "BPOGY"], axis=0
        )
        self.img = np.array(
            Image.open(
                get_image_path_from_csv(self.df_path, self.images_dir, self.img_name)
            )
        )
        self.get_image_coordinates()


    def get_image_coordinates(self) -> None:
        bpogv_df = self.df[self.df["BPOGV"] == 1]
        bpog_x = bpogv_df["BPOGX"].clip(lower=0)
        bpog_y = bpogv_df["BPOGY"].clip(lower=0)
        time_duration = bpogv_df["FPOGD"]
        time = bpogv_df["TIME"]


        
        # get the padded image size
        self.screen_width = self.screen_size.x
        self.screen_height = self.screen_size.y
        self.image_width = self.img.shape[1]
        self.image_height = self.img.shape[0]
        screen_ratio = self.screen_width / self.screen_height
        image_ratio = self.image_width / self.image_height
        if screen_ratio >= image_ratio:
            size_x = self.image_height / self.screen_height * self.screen_width
            size_y = self.image_height
        else:
            size_x = self.image_width
            size_y = self.image_width / self.screen_width * self.screen_height

        # get paddings
        self.padx = (size_x - self.image_width) / 2
        self.pady = (size_y - self.image_height) / 2
        
        # get the image coordinates which is screen coords minus pad
        x = bpog_x * size_x - self.padx
        y = bpog_y * size_y - self.pady

        self.image_coords = [c for c in zip(x, y, time, time_duration)]
        self.image_coords = [c for c in self.image_coords
                if c[0] > 0 and c[0] < self.image_width 
                and c[1] > 0 and c[1] < self.image_height]
        self.image_coords_3s = self.get_coordinates(3)
        self.image_coords_1s = self.get_coordinates(1)


    def get_coordinates(self, duration=9999) -> List[Coordinate]:
        """
        Gets the coordinates for the current image's gaze fixation points
        from time t=0 to t=duration which can be specified in seconds.

        Returns
        -------
        List[Coordinate]:
            Coordinates for the fixation points
        """
        return [c for c in self.image_coords if c[2] <= duration]


def get_outpath_from_csv_path(output_dir: str, csv_path: str, img_name: str) -> str:
    """
    Generates the output path for the plotted image to be saved.

    Parameters
    ----------
    output_dir : str
        Main output directory
    csv_path : str
        Path to the CSV file
    img_name : str
        Name of the image to be saved

    Returns
    -------
    str:
        Full path for the saved image
    """
    info = get_info_from_path(csv_path)
    block = info["block"]
    block_str = f"block {int(block):d}" if block != "c" else "calibration"
    participant_id = info["participant_id"]
    return os.path.join(output_dir, f"kh{participant_id}", block_str, img_name)

# output binary fixmap and heatmap
def create_fixmap_and_heatmap(w, h, coords, img_name, outpath, sigma=25): 

    # output directories
    fixmaps_dir = outpath + "fixmaps"
    heatmaps_dir = outpath + "heatmaps"
    
    xs = tuple([elt[0] for elt in coords])
    ys = tuple([elt[1] for elt in coords])
    times = tuple([elt[2] for elt in coords])
    time_durations = tuple([elt[3] for elt in coords])
    bitmap = np.zeros((w, h))
    fixations = np.zeros((w, h))
    for c in coords: 
        x,y = int(c[0]),int(c[1])
        if x < w and y < h:
            fixations[x,y] += 1    
            bitmap[x,y] = 1
    heatmap = ndimage.gaussian_filter(fixations, [sigma, sigma])
    heatmap = 255*heatmap/float(np.max(heatmap))


    bitmap = ndimage.gaussian_filter(bitmap, [5, 5])
    bitmap = (bitmap > 0.001).astype(int)
    bitmap = 255 * np.ceil(bitmap/float(np.max(bitmap)))

    fixmap_img = Image.fromarray(np.uint8(np.transpose(bitmap)), "L")
    heatmap_img = Image.fromarray(np.uint8(np.transpose(heatmap)), "L")

    fixmap_img.save(os.path.join(fixmaps_dir, img_name))
    heatmap_img.save(os.path.join(heatmaps_dir, img_name))


def create_eye_path(w, h, coords, img_name, outpath, img, sigma=25): 

    # output directories
    paths_dir = outpath + "paths"
    
    xs = tuple([elt[0] for elt in coords])
    ys = tuple([elt[1] for elt in coords])
    ts = tuple([elt[2] for elt in coords])
    bitmap = np.zeros((w, h))
    fixations = np.zeros((w, h))
    for c in coords: 
        x,y = int(c[0]),int(c[1])
        if x < w and y < h:
            fixations[x,y] += 1    
            bitmap[x,y] = 1
    heatmap = ndimage.gaussian_filter(fixations, [sigma, sigma])
    heatmap = 255*heatmap/float(np.max(heatmap))


    bitmap = ndimage.gaussian_filter(bitmap, [1, 1])
    bitmap = (bitmap > 0.001).astype(int)
    bitmap = 255 * np.ceil(bitmap/float(np.max(bitmap)))

    fixmap_img = Image.fromarray(np.uint8(np.transpose(bitmap)), "L")
    heatmap_img = Image.fromarray(np.uint8(np.transpose(heatmap)), "L")

    if img.shape[2] == 3: 
        img = np.insert(
            img,
            3, #position in the pixel value [ r, g, b, a <-index [3]  ]
            255/2, # or 1 if you're going for a float data type as you want the alpha to be fully white otherwise the entire image will be transparent.
            axis=2, #this is the depth where you are inserting this alpha channel into
        )
    else:
        img[:,:,:4] = 255/2

    plt.gray()
    ax = plt.imshow(img)
    for i in range(len(xs)):
        if i > 0:
            ax.axes.arrow(
                xs[i - 1],
                ys[i - 1],
                xs[i] - xs[i - 1],
                ys[i] - ys[i - 1],
                width=3,
                color="yellow",
                alpha=0.5,
            )

    for i in range(len(xs)):
        cir_rad = 50
        circle = plt.Circle(
            (xs[i], ys[i]),
            radius=cir_rad,
            edgecolor="yellow",
            facecolor="black",
            alpha=0.5,
        )
        ax.axes.add_patch(circle)
        ax.axes.annotate(
            "{}".format(i + 1),
            xy=(xs[i], ys[i] + 3),
            fontsize=10,
            ha="center",
            va="center",
            color="white"
        )

    ax.figure.savefig(os.path.join(paths_dir, img_name), dpi=120, bbox_inches="tight")
    plt.close(ax.figure)


def output_result(img_data, outpath, TYPE):

    # output heatmaps
    if TYPE == "heatmap":
        create_fixmap_and_heatmap(img_data.image_width, img_data.image_height, 
                    img_data.image_coords, img_data.img_name, outpath, sigma=40)

    # output paths
    elif TYPE == "path":
        create_eye_path(img_data.image_width, img_data.image_height, 
                    img_data.image_coords, img_data.img_name, outpath, 
                    img_data.img, sigma=40)

    # output data
    elif TYPE == "data":
        file_pi = open(outpath + "data/" + img_data.img_name.split('.')[0] + '.pkl', 'wb') 
        pickle.dump(img_data, file_pi)

def main(args: Dict[str, str]) -> None:
    fixations_dir, images_dir, output_dir = (
        args["input_csv"],
        args["input_blocks"],
        args["output"],
    )

    # get the type of output, heatmap of binary or path
    TYPE = args["type"]

    # get screen size
    WIDTH, HEIGHT = [int(d) for d in args["screen_size"].lower().split("x")]
    SCREEN_SIZE = (WIDTH, HEIGHT)

    # Check for path validity
    for d in [fixations_dir, images_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(f"Path {d} does not exist.")

    os.makedirs(output_dir, exist_ok=True)

    # Get CSVs and blocks directories
    fixations = sorted(glob.glob(fixations_dir + "/*.csv"))
    images = sorted(glob.glob(images_dir + "/*"))


    # Check for defective CSVs (Incorrect number of columns for some data).
    # Skip them for processing.
    defective = list()
    for idx in range(len(fixations)):
        try:
            pd.read_csv(fixations[idx])
        except pd.errors.ParserError as e:
            print(f"[{idx}]{fixations[idx]}: {e}")
            defective.append(fixations[idx])

    # get the valid fixation points
    fixations_correct = [f for f in fixations if f not in defective]

    # For each CSV data file
    for csv_path in tqdm(
        fixations_correct, desc="Processing CSV Fixation Data", unit="csv"
    ):
        images = pd.read_csv(csv_path)["MEDIA_NAME"].unique()

        

        # Get the images in the CSV file, generate and plot heatmaps, and save them
        for img_name in tqdm(
            images, desc="Generating Heatmaps for CSV", unit="image", leave=False
        ):

          
            img_data = ImageData(
                csv_path,
                images_dir,
                img_name,
                SCREEN_SIZE,
            )

            outpath = get_outpath_from_csv_path(output_dir, csv_path, img_name)
            outpath = os.path.join(os.path.dirname(outpath), "")

            # create result directories
            if TYPE == "heatmap":
                os.makedirs(outpath + "fixmaps", exist_ok=True)
                os.makedirs(outpath + "heatmaps", exist_ok=True)
            elif TYPE == "path":
                os.makedirs(outpath + "paths", exist_ok=True)
            elif TYPE == "data":
                os.makedirs(outpath + "data", exist_ok=True)
            
            output_result(img_data, outpath, TYPE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate heatmaps and scanpaths from participant data."
    )
    parser.add_argument(
        "-ic",
        "--input-csv",
        help=(
            "Path to the directory containing the CSV files "
            "in the form 'xx_KHyyy_fixations.csv', "
            "where xx is the block number and YYY is the participant ID."
        ),
        type=str,
        required=True,
    )
    parser.add_argument(
        "-ib",
        "--input-blocks",
        help="Path to the directory containing the image blocks for the experiment.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--type",
        help="Type of output (heatmap / data / path).",
        choices=['heatmap', 'data', 'path'],
        default="heatmap",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path to store the generated heatmaps.",
        type=str,
        default="./output/",
    )
    parser.add_argument(
        "-s",
        "--screen-size",
        help="Screen size in WIDTHxHEIGHT format.",
        type=str,
        default="1920x1200",
    )
    args = vars(parser.parse_args())
    main(args)
