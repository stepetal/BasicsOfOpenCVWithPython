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

def show_image_with_matplotlib(image,title="Image"):
    plt.figure(figsize = (8,8))
    plt.title(title)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)

def update_image_channels(image,value,ch):
    '''
    Update value for specified channel

    Parameters
    ----------
    image : numpy ndarray
        Input image
    value : int
        New value for channel
    ch : str
        Channel id

    Returns
    -------
    None.

    '''
    if ch == 'b':
        image[:,:,0] = np.ones(image.shape[:2],dtype = np.uint8) * value
    if ch == 'g':
        image[:,:,1] = np.ones(image.shape[:2],dtype = np.uint8) * value
    if ch == 'r':
        image[:,:,2] = np.ones(image.shape[:2],dtype = np.uint8) * value
    cv2.imshow("Image_window",image)


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
    show_image_with_matplotlib(image_mpl)
    # create grayscale image from original matplotlib image
    grayscale_image = cv2.cvtColor(image_mpl,cv2.COLOR_RGB2GRAY)
    cv2.imwrite("grayscale_image.jpg",grayscale_image)
    # create white canvas for drawing

    canvas = np.ones((600,600,3))
    width = canvas.shape[1]
    height = canvas.shape[0]
    # drawing line
    canvas = cv2.line(canvas,(0,0),(width,height),(0,0,255))
    # drawing rectangle
    canvas = cv2.rectangle(canvas,(0,0),(int(width/2),int(height/2)),(255,0,0))
    # drawing circle
    canvas = cv2.circle(canvas,(int(width/2),int(height/2)),20,(0,255,0))
    # drawing text
    canvas = cv2.putText(canvas,"Basic shapes",(10,int(height/2)),cv2.FONT_HERSHEY_PLAIN,0.5,(0,0,0),None,cv2.LINE_AA)
    show_image_with_matplotlib(canvas,"Basic primitives")
    # splitting and merging channels
    # get values of r,g,b and change image color
    hsv_image = cv2.cvtColor(image_mpl,cv2.COLOR_RGB2HSV)
    (h,s,v) = cv2.split(hsv_image)
    hsv_image = cv2.merge((h,s,v))
    # update channels values with trackbars
    cv2.namedWindow('Image_window',cv2.WINDOW_NORMAL);
    new_canvas = np.ones(image_mpl.shape,dtype = np.uint8)
    cv2.createTrackbar('R',
                       'Image_window',
                       0,
                       255,
                       lambda value: update_image_channels(new_canvas,value,'r')
                       )
    cv2.createTrackbar('G',
                       'Image_window',
                       0,
                       255,
                       lambda value: update_image_channels(new_canvas,value,'g')
                       )
    cv2.createTrackbar('B',
                       'Image_window',
                       0,
                       255,
                       lambda value: update_image_channels(new_canvas,value,'b')
                       )
    
    cv2.waitKey(0)


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
    
    