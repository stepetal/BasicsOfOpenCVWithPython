'''

Solving computer vision tasks with OpenCV library

'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import argparse

def argument_parser():
    '''
    Create and initialize parser of script arguments

    Returns
    -------
    ArgumentParser

    '''
    parser = argparse.ArgumentParser(description = __doc__,formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("image_file_path", type = str, help = "Path for test image")
    parser.add_argument("video_file_path", type = str, help = "Path for test video, 0 for capturing web - camera")
    parser.add_argument("current_mode",type = int,
                        help = "1 - Basic operations with images\n"
                               "2 - Basic drawing on canvas\n"
                                
                        )
    return parser

def basic_operations_with_images(image_path):
    image = cv2.imread(image_path,cv2.IMREAD_COLOR)
    image_mpl = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.figure(figsize = (8,8))
    plt.grid(False)
    plt.imshow(image_mpl)

if __name__ == "__main__":
    # create argument parser
    parser = argument_parser()
    args = parser.parse_args()
    print("Image file path: {}".format(args.image_file_path))
    print("Image file path: {}".format(args.image_file_path))
    print("Current mode is: {}".format(args.current_mode))
    image_path = args.image_file_path
    video_path = args.video_file_path
    if args.current_mode == 1:
        basic_operations_with_images("church.jpg")
    
    