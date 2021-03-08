'''

Solving computer vision tasks with OpenCV library

'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import argparse
from sklearn.datasets import load_sample_image

def argument_parser():
    '''
    Create and initialize parser of script arguments

    Returns
    -------
    ArgumentParser

    '''
    parser = argparse.ArgumentParser(description = __doc__,formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("--image_file_path", type = str, default = 'flower.jpg', help = "Path for test image")
    parser.add_argument("--video_file_path", type = str, default = '0', help = "Path for test video, by default 0  - capturing first web - camera")
    parser.add_argument("current_mode",type = int,
                        help = "1 - Basic operations with images\n" +
                               "2 - Basic operations with videos\n"
                                
                        )
    return parser

def basic_operations_with_images(image_path):
    '''
    Basic operations with images: image opening and writing, resizing, grayscaling,
    rotating,croping,putting text

    Parameters
    ----------
    image_path : str
        Path to image file

    Returns
    -------
    None.

    '''
    if image_path != 'flower.jpg':
        image = cv2.imread(image_path,cv2.IMREAD_COLOR)
        image_mpl = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    else:
        image_mpl = load_sample_image(image_path)
        #show original image
    plt.figure(figsize = (8,8))
    plt.title("Image")
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image_mpl)
    # create grayscale image from original matplotlib image
    grayscale_image = cv2.cvtColor(image_mpl,cv2.COLOR_RGB2GRAY)
    cv2.imwrite("grayscale_image.jpg",grayscale_image)

def basic_operations_with_videos(video_path):
    '''
    Basic operations with video: opening and saving
    
    Parameters
    ----------
    video_path : str
        Path to video file or index of camera 0 - 5

    Returns
    -------
    None.

    '''
    webcam_indexes = ['0','1','2','3','4','5']
    if video_path in webcam_indexes:
        # define video writer object with FourCC code
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output = cv2.VideoWriter('out_video.avi',fourcc,20.0,(640,480))
        # get video from the first available webcam
        cap = cv2.VideoCapture(webcam_indexes.index(video_path),cv2.CAP_DSHOW)
    else:
        # get video from file
        cap = cv2.VideoCapture(video_path)
    while True:
        _, frame = cap.read()
        cv2.imshow('Current video frame',frame)
        # write only capturing from webcam
        if video_path in webcam_indexes:
            output.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    if video_path in webcam_indexes:
        output.release()

def  main():
        # create argument parser
    parser = argument_parser()
    args = parser.parse_args()
    print("Image file path: {}".format(args.image_file_path))
    print("Video file path: {}".format(args.video_file_path))
    print("Current mode is: {}".format(args.current_mode))
    image_path = args.image_file_path
    video_path = args.video_file_path
    print(args)
    if args.current_mode == 1:
        basic_operations_with_images(args.image_file_path)
    elif args.current_mode == 2:
        basic_operations_with_videos(args.video_file_path)
    


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
    
    