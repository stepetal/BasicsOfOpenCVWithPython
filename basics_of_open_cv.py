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
from numpy.random import Generator,PCG64




def show_image_with_matplotlib(image,title="Image"):
    plt.figure(figsize = (8,8))
    plt.title(title)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)
    plt.show()
    
def add_salt_and_pepper_noise(image,noise_percent):
    '''
    Add noise with type "salt and pepper"

    Parameters
    ----------
    image : numpy.ndarray
        Input image
    noise_percent : int
        Percentage of noise added to image

    Returns
    -------
    image : numpy.ndarray
        Image with noise

    '''
    noise_points = int((image.shape[0] * image.shape[1] * image.shape[2] * noise_percent) / 100)
    rand_gen = Generator(PCG64(seed = 1))
    rand_widths = rand_gen.integers(low = 0,high = image.shape[0] - 1,size = noise_points)
    rand_heights = rand_gen.integers(low = 0,high = image.shape[1] - 1,size = noise_points)
    rand_ch = rand_gen.integers(low = 0,high = image.shape[2] - 1,size = noise_points)
    rand_pix = np.hstack((rand_widths[:,np.newaxis],rand_heights[:,np.newaxis],rand_ch[:,np.newaxis]))
    noise = np.zeros(noise_points,dtype = np.uint8)
    rand_floats = rand_gen.random(size = noise_points)
    mask = rand_floats > 0.5
    noise[mask] = 255
    image[rand_widths[:,np.newaxis],rand_heights[:,np.newaxis],rand_ch[:,np.newaxis]] = noise[:,np.newaxis]
    return image

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

def drawing_basic_shapes():
    '''
    Demonstration of drawing circles, rectangles, text on canvas

    '''
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

def split_and_merge_channels(image_path):
    '''
    Splitting and merging channels of an image
    '''
    image_mpl = load_sample_image(image_path)
    hsv_image = cv2.cvtColor(image_mpl,cv2.COLOR_RGB2HSV)
    (h,s,v) = cv2.split(hsv_image)
    hsv_image = cv2.merge((h,s,v))

def create_trackbars_demo():
    '''
    Procedure for creating window with three trackbars used for mixing red,green an blue channels
    '''
    
    # update channels values with trackbars
    cv2.namedWindow('Image_window',cv2.WINDOW_NORMAL);
    new_canvas = np.ones(((300,200,3)),dtype = np.uint8)
    new_canvas = cv2.cvtColor(new_canvas,cv2.COLOR_BGR2HSV)
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

def operations_on_individual_pixels(image_mpl):
    '''
    Access and modification of pixels
    '''
    # access to individual pixel
    print("Pixel value is: {}".format(image_mpl[20,20]))
    # modify individual pixel
    px = image_mpl.item(20,20,0)
    image_mpl.itemset((20,20,0),33)
    print("Pixel value is: {}".format(image_mpl.item(20,20,0)))
    image_mpl.itemset((20,20,0),px)

def select_roi(image_mpl):
    '''
    Show region of interest
    '''
    # select ROI - region of interest
    roi = image_mpl[250:350,250:350]
    show_image_with_matplotlib(roi)

def show_border(image):
    '''
    Demonstrates addition of border to an image
    '''
    #show border
    replicated_image = cv2.copyMakeBorder(image,20,20,20,20,cv2.BORDER_REFLECT_101)
    show_image_with_matplotlib(replicated_image)
    
def arithmetic_and_bitwise_operations(image_mpl):
    '''
    Demonstration of basic arithmetic and bitwise operation on images
    '''
     # arithmetic operations on images
    sum_of_images = image_mpl + image_mpl
    show_image_with_matplotlib(sum_of_images,"Sum of images")
    # image blending
    blended_image = cv2.addWeighted(image_mpl,0.7,image_mpl,0.7,0)
    show_image_with_matplotlib(blended_image)
    # bitwise operations
    logo = cv2.imread("logo_white.png",cv2.IMREAD_UNCHANGED)
    logo_rgb = cv2.cvtColor(logo,cv2.COLOR_BGR2RGB)
    show_image_with_matplotlib(logo_rgb,"logo")
    gray_logo = cv2.cvtColor(logo,cv2.COLOR_BGR2GRAY)
    _,mask = cv2.threshold(gray_logo,50,255,cv2.THRESH_BINARY)
    inv_mask = cv2.bitwise_not(mask)
    show_image_with_matplotlib(mask,"mask")
    show_image_with_matplotlib(inv_mask,"inv_mask")
    show_image_with_matplotlib(cv2.bitwise_and(mask,inv_mask),"bitwise_and")
    show_image_with_matplotlib(cv2.bitwise_or(mask,inv_mask),"bitwise_or")

def show_image_properties(image_mpl):
    '''
    Procedure for showing shape and type of an image
    '''
    # show image properties
    print("Image shape is: {}".format(image_mpl.shape))
    print("Image size is: {}".format(image_mpl.size))
    print("Image type is: {}".format(image_mpl.dtype))
    
def show_execution_time():
    '''
    Procedure for demonstration of operation durations
    '''
    # measurement of performance
    t1 = cv2.getTickCount()
    print("Operation that cost some computation time...")
    t2 = cv2.getTickCount()
    delta_t = (t2 - t1) / cv2.getTickFrequency()
    print("Duration is: {} seconds".format(delta_t))
    
def gamma_correction(grayscale_image):
    '''
    Procedure for demonstration non-linear operation that adjusts pixel intensities
    '''
    image = grayscale_image.astype(np.float32) / 255
    gamma = 0.5
    corrected_image = np.power(image,gamma)
    print("Image type is: {}".format(image.dtype))
    #grayscale_image = np.array(grayscale_image * 255).astype(np.uint8)
    #show_image_with_matplotlib(grayscale_image,"Original image")
    #show_image_with_matplotlib(corrected_image,"Corrected image")
    cv2.imshow("original image",grayscale_image)
    cv2.imshow("Corrected image",corrected_image)
    
def plot_image_histogram(grayscale_image):
    '''
    Procedure for finding and plotting histogram of an image
    '''
    eq_hist_grayscale_image = cv2.equalizeHist(grayscale_image)
    eq_hist,eq_bins = np.histogram(eq_hist_grayscale_image,256,range = [0,255])
    
    hist,bins = np.histogram(grayscale_image,256,range = [0,255])
    
    plt.figure(figsize = (10,10))
    plt.title('Grayscale image histogram')
    plt.fill(hist)
    plt.xlabel('pixel value')
    
    plt.figure(figsize = (10,10))
    plt.title('Grayscale image normalized histogram')
    plt.fill_between(range(256),eq_hist,0)
    plt.xlabel('pixel value')
    
    

def image_thresholding(grayscale_image):
    '''
    Procedure for demonstration of applying different types of threshold to an image
    '''
    # image thresholding
    # simple thresholding
    _,thresh1 = cv2.threshold(grayscale_image,127,255,cv2.THRESH_BINARY)
    _,thresh2 = cv2.threshold(grayscale_image,127,255,cv2.THRESH_BINARY_INV)
    show_image_with_matplotlib(thresh1,"threshold binary")
    show_image_with_matplotlib(thresh2,"threshold binary inv")
    # adaptive thresholding
    image_for_thresh = cv2.imread("sudoku.png",cv2.IMREAD_GRAYSCALE)
    adaptive_thresh = cv2.adaptiveThreshold(image_for_thresh,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    cv2.imshow("adaptive threshold",adaptive_thresh)
    # otsu thresholding
    _, otsu_thresh = cv2.threshold(image_for_thresh,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("otsu threshold",otsu_thresh)


def geometric_transformations(image):
    '''
    Demonstration of geometric transformation of images
    '''
    # geometric transformations of images
    # resizing of image
    resized_image = cv2.resize(image,None,fx = 1/2,fy = 1/2,interpolation = cv2.INTER_CUBIC)
    cv2.imshow("resized image",resized_image)
    # translation of image
    M = np.float32([[1,0,20],[0,1,30]])
    translated_image = cv2.warpAffine(image,M,(image.shape[:2]))
    cv2.imshow("Translated image",translated_image)
    # rotation
    width,height = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width/2,height/2),90,1)
    rotated_image = cv2.warpAffine(image,rotation_matrix,(width,height))
    cv2.imshow("Roteted image",rotated_image)

def smoothin_techniques():
    '''
    Applying filters to smooth image
    '''
    logo = cv2.imread("logo_white.png",cv2.IMREAD_UNCHANGED)
    logo_rgb = cv2.cvtColor(logo,cv2.COLOR_BGR2RGB)
    # smoothing images
    # applying general filter (2D convolution)
    kernel = np.ones((5,5),np.float32)/5
    filtered_image = cv2.filter2D(logo,cv2.CV_8U,kernel)
    filtered_image = cv2.cvtColor(filtered_image,cv2.COLOR_BGR2RGB)
    show_image_with_matplotlib(filtered_image,"Image filtered with kernel np.ones * 1/5")
    # bluring image
    # averaging
    noisy_image = add_salt_and_pepper_noise(logo_rgb.copy(),25)
    blurred_image = cv2.blur(noisy_image,(5,5))
    show_image_with_matplotlib(blurred_image,"Blurred image with (5,5) kernel")
    gauss_blurred_image = cv2.GaussianBlur(noisy_image,(5,5),0)
    show_image_with_matplotlib(gauss_blurred_image,"Image blurred with gaussian kernel (5,5), std = 0")
    median_blurred_image = cv2.medianBlur(noisy_image,5)
    show_image_with_matplotlib(noisy_image,"Image with salt and pepper noise")
    show_image_with_matplotlib(median_blurred_image,"Image blurred with median blur, kernel 5x5")


def morphological_transformations():
    '''
    Demonstration of dilation,erosion,opening and closing operations
    '''
        # morphological transformations
    # erosion
    morph_image = cv2.imread("morpho_image_2.png",cv2.IMREAD_COLOR)
    morph_image = morph_image[:,:,::-1]
    #show_image_with_matplotlib(morph_image,"Original image")
    morph_kernel = np.ones((5,5),dtype = np.uint8)
    eroded_image = cv2.erode(morph_image,morph_kernel,iterations = 1)
    #show_image_with_matplotlib(eroded_image,"Eroded image")
    # dilation
    dilated_image = cv2.dilate(morph_image,morph_kernel,iterations = 1)
    #show_image_with_matplotlib(dilated_image,"Dilated image")
    # Opening
    open_image = cv2.morphologyEx(morph_image,cv2.MORPH_OPEN,morph_kernel)
    #show_image_with_matplotlib(open_image,"Eroded then dilated image")
    # Closing
    open_image = cv2.morphologyEx(morph_image,cv2.MORPH_CLOSE,morph_kernel)
    #show_image_with_matplotlib(open_image,"Dilated then eroded image")

def image_gradients():
    '''
    Demonstration of finding edges with Sobel operator,Canny algorithm and Laplacian opearator
    '''
    # Image gradients
    sudoku = cv2.imread("sudoku.png",cv2.IMREAD_GRAYSCALE)
    sudoku = cv2.cvtColor(sudoku,cv2.COLOR_BGR2RGB)
    show_image_with_matplotlib(sudoku,"Origian image")
    # Laplacian
    laplacian = cv2.Laplacian(sudoku,cv2.CV_64F)
    show_image_with_matplotlib(np.absolute(laplacian).astype(np.uint8),"Laplacian of the image")
    # Sobel OX
    sobel_x = cv2.Sobel(sudoku,cv2.CV_64F,1,0,ksize = 5)
    show_image_with_matplotlib(np.absolute(sobel_x).astype(np.uint8),"Sobel operator, axis OX")
    sobel_y = cv2.Sobel(sudoku,cv2.CV_64F,0,1,ksize = 5)
    show_image_with_matplotlib(np.absolute(sobel_y).astype(np.uint8),"Sobel operator, axis OY")
    # Canny
    sudoku = cv2.imread("sudoku.png",cv2.IMREAD_GRAYSCALE)
    sudoku = cv2.cvtColor(sudoku,cv2.COLOR_BGR2RGB)
    canny = cv2.Canny(sudoku,50,150)
    cv2.imshow("Canny edge detector",canny)
    show_image_with_matplotlib(canny,"Edges found by canny detector")
    
def contour_finding():
    '''
    Procedure demonstrates contour detection techniques
    '''
    # finding the contours
    
    image = load_sample_image('flower.jpg')
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(image_gray,127,255,cv2.THRESH_BINARY)
    contour_image = thresh.copy()
    contours,hierarchy = cv2.findContours(contour_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("Contour image",contour_image)
    cv2.imshow("Thresholded image",thresh)
    print("Number of contours found: {}".format(len(contours)))
    # draw all contours
    canvas = np.ones(thresh.shape,dtype = np.uint8)
    img = cv2.drawContours(canvas.copy(),contours,-1,(255,255,255),3)
    cv2.imshow("All contours of the image",img)
    # draw individual contour
    cnt = contours[5]
    img = cv2.drawContours(canvas.copy(),[cnt],0,(255,255,255),3)
    cv2.imshow("Specified contours of the image",img)
    
    

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
    #show_image_with_matplotlib(image_mpl)
    # create grayscale image from original matplotlib image
    grayscale_image = cv2.cvtColor(image_mpl,cv2.COLOR_RGB2GRAY)
    cv2.imwrite("grayscale_image.jpg",grayscale_image)
    
    plot_image_histogram(grayscale_image)
    



    
    
    
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
    
    