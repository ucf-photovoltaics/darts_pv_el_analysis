
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 12:09:58 2018

@authors: Eric, Dylan
"""

import os
import os.path
from os.path import basename
import decimal
from skimage.color import gray2rgb
from skimage import io
import matplotlib.pyplot as plt
import rawpy as rp
import numpy as np
import pandas as pd
import imageio as iio  # imageio.v3
import cv2
from PIL import Image, ImageFont, ImageDraw
import xarray as xr
# from pyhull import qconvex


def remove_exif(image_file, dst_dir=os.getcwd()):
    im = Image.open(image_file)
    im.getexif().clear()
    im.save(f"{dst_dir}/{basename(image_file)}")

# TODO see if I can convert to dict


def extract_exif(image_file, dst_dir=os.getcwd()):
    im = Image.open(image_file)
    exif_data = im.getexif()
    return exif_data


def choose_image_reader(image_filepath, for_analysis=False, image_32F=False):
    '''
    Choose the correct image reader based on the type of image.

    Parameters
    ----------
    image_filepath : str
        Filepath for the image.
    image_32F : bool, optional
        True only if using 32-bit float images.
        We've only had to use this for one type of InGaAs camera.
        The default is False.

    Returns
    -------
    image : Array of uint8 (image)
        Module image in machine-readable format.

    '''
    # Images for displaying cell grid.
    image_extension = image_filepath.split('.')[-1].upper()
    if image_extension in ['NEF', 'ARW']:
        image = rp.imread(image_filepath).raw_image
        if not for_analysis:
            image = rp.imread(image_filepath).postprocess()

    elif ('RAW.tif' in image_filepath) and (for_analysis is False):
        image = cv2.imread(image_filepath)
    elif image_extension == 'TXT':
        image = read_ir_image(image_filepath)
    else:
        image = io.imread(image_filepath)

    if image_32F:
        image = ((image/image.max())*255).astype(np.uint8)
        image = gray2rgb(image)

    return image


def normalize_image(image, method='max', ratio=1.0):
    """
    image: array
    method: str
        This defines how the image will be normalized. Available methods are
            'max', 'median', 'ratio'. If choosing 'ratio', the ratio variable
            must be set to a float.
    ratio: float
        Default = 1.0. A number greater than 0 which will multiply the image.
        E.g., if ratio is 2, the image will be multiplied by 2.
    """
    # uses only positive values for calculating median
    pixels = image.flatten()
    pixels = pixels[pixels > 0]

    if method == 'max':
        norm = pixels.max()
    elif method == 'median':
        norm = np.median(pixels)
    elif method == 'none':
        norm = 1

    # Turn zero values into 1 to prevent divide by zero
    image[image < 1] = 1
    image = (ratio*image)/norm

    # Take pixels that were forced to be 1 back to zero
    image[image == ratio/norm] = 0
    # image = image.astype(int)

    return image


def threshold_image(image, percent=0, value=1):
    '''
    image: array
    percent: float
        Default = 0. Sets any pixel value below percent*image.max() to 0.
    value: int
        Default = 1. Sets any pixel value below value to 0.
    '''
    # Sets values to 0 if they are below thresholds.
    percent = percent/100
    image[image < percent*np.max(image)] = 0
    image[image < value] = 0

    return image


def resize_image(image, width=None, height=None):

    # Initialize dimensions of image to be resized and grab image size
    dim = None
    (h, w) = image.shape[:2]

    # If both the width and height are None, then return the original image
    if width is None and height is None:
        return image

# calculate ratio of height and construct the dimensions if the width is None
    if width is None:
        r = height/float(h)
        dim = (int(w*r), height)
# calculate ratio of width, construct the dimensions if the height is None
    else:
        r = width/float(w)
        dim = (width, int(h*r))

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return resized


def order_module_corners(pts):
    # Initialize a list of coordiantes that will be ordered as follows:
    # top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype='float32')

    # Adding the x and y coordinates for each corner
    # The top-left point will have the smallest sum; bottom-right, largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right point will have the smallest difference; bottom-left, largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Return the ordered coordinates
    return rect


def flatten_corners(corners):
    # Keeps the 1st calculation of the corners

    # Uses 1st calculation of top boundary
    corners[1, 1] = corners[0, 1]
    # Uses 1st calculation of right boundary
    corners[2, 0] = corners[1, 0]
    # Uses 2nd calculation of bottom boundary
    corners[2, 1] = corners[3, 1]
    # Uses 1st calculation of left boundary
    corners[3, 0] = corners[0, 0]

    # initialize a list of coordinates that will be ordered as follows:
    # top-left, top-right, bottom-right, bottom-left
    # rect = np.zeros((4, 2), dtype='float32')

    # rect[0, 0] = rect[3, 0] = max(pts[:,0])
    # rect[1, 0] = rect[2, 0] = min(pts[:,0])

    # rect[0, 1] = rect[1, 1] = max(pts[:,1])
    # rect[2, 1] = rect[3, 1] = min(pts[:,1])


# TODO: make it local in select_module_corners
refPt = np.zeros([4, 2], dtype=np.float32)
event_num = 0


def get_boundary_corners(image):
    '''
    Converts the image corners into an array. Useful for cropping cells
    from a cropped image or perspective image with cropping.

    Parameters
    ----------
    image : Array of uint8 (image)
        Module image.

    Returns
    -------
    corners : Array of float32
        The (x,y) position of the module's corners.

    '''
    # top left, top right, bottom right, bottom left
    corners = np.array([[0, 0], [image.shape[1], 0],
                        [image.shape[1], image.shape[0]],
                        [0, image.shape[0]]], dtype='float32')

    return corners


def select_module_corners(display_image):
    '''
    Manually select the corners of the module instead of performing
    image processing.

    Parameters
    ----------
    display_image : Array of uint8 (image)
        Module image.

    Returns
    -------
    corners : Array of float32
        The (x,y) position of the module's corners.

    '''
    def click_edges(event, x, y, flags, param):
        # grab references to the global variables
        global refPt, event_num
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt[event_num, :] = [x, y]
            event_num += 1

    resize_height = 700
    ratio = display_image.shape[0]/resize_height
    image_reduced = resize_image(display_image, height=resize_height)

    clone = image_reduced.copy()
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', click_edges)

    global event_num
    event_num = 0
    while True:
        # display the image and wait for a keypress
        cv2.imshow('image', image_reduced)
        key = cv2.waitKey(1) & 0xFF
        cv2.circle(image_reduced, (int(
            refPt[event_num - 1, 0]), int(
                refPt[event_num - 1, 1])), 2, (0, 255, 0), thickness=2)
        # if the 'r' key is pressed, reset the cropping region
        if (key == ord('r')) or (key == ord('R')):
            image_reduced = clone.copy()
            event_num = 0
        # if the 'c' key is pressed, break from the loop and close window
        elif (key == ord('c')) or (key == ord('C')):
            cv2.destroyAllWindows()
            break

    corners = order_module_corners(refPt)*ratio
    # flatten_corners(corners)

    return corners


def detect_module_edges(image, blurring_steps=0):

    # Process image through a sequence of blurring steps
    for n in range(1, blurring_steps):
        image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply canny edge detection on blurred image
    edged = cv2.Canny(image, 1, 75)

    # finds the first and last bright pixel in the vertical and horizontal directions
    # deletes anything in between these two. Allows for a broader range of blurring steps to work properly
    new_edged = np.zeros([len(edged[:, 0]), len(edged[0, :])])

    for i1 in range(len(edged[0, :])):
        idx_first = np.argmax(edged[:, i1])
        idx_last = len(edged[:, i1]) - np.argmax(edged[:, i1][::-1]) - 1
        if idx_first != 0:
            new_edged[idx_first, i1] = 255
        if idx_last != len(edged[:, i1])-1:
            new_edged[idx_last, i1] = 255

    for i1 in range(len(edged[:, 0])):
        idx_first = np.argmax(edged[i1, :])
        idx_last = len(edged[i1, :]) - np.argmax(edged[i1, :][::-1]) - 1
        if idx_first != 0:
            new_edged[i1, idx_first] = 255
        if idx_last != len(edged[i1, :])-1:
            new_edged[i1, idx_last] = 255

    edged = new_edged.astype(np.uint8)

    # Initialize arrays for line detection
    strong_lines = np.zeros([4, 1, 2])
    strong_lines_2D = np.zeros([4, 1, 2])

    # Use HoughLines function to detect lines in the edge detection image
    min_line_length = 100
    max_line_gap = 10
    lines = cv2.HoughLines(edged, 1, np.pi/180, 10,
                           min_line_length, max_line_gap)

    # Try to verify that we did not detect two lines that essentially overlap
    # (Not always perfect!)
    n4 = 0
    n5 = 0
    for n3 in range(0, len(lines)):
        for rho, theta in lines[n3]:
            # Check is theta is near normal as we expect a close to square EL Image
            if rho < 0:
                rho *= -1
                theta -= np.pi
            parallel_theta = np.isclose(theta, 0, atol=np.pi/36)
            perpendicular_theta = np.isclose(theta, np.pi/2, atol=np.pi/36)
            if parallel_theta or perpendicular_theta:
                if n4 == 0:
                    strong_lines[n4] = lines[n3]
                    strong_lines_2D[n5] = lines[n3]
                    n4 = n4 + 1
                    n5 = n5 + 1

                else:
                    # is this line close in terms of rho
                    closeness_rho = np.isclose(
                        rho, strong_lines[0:n4, 0, 0], atol=100)

                    # is this line close in terms of theta
                    closeness_theta = np.isclose(
                        theta, strong_lines[0:n4, 0, 1], atol=np.pi/36)

                    # compare both theta and rho
                    closeness = np.all(
                        [closeness_rho, closeness_theta], axis=0)
                    directionality_theta = np.isclose(
                        theta, strong_lines_2D[0:2, 0, 1], atol=np.pi/9)

                    if not any(closeness) and n4 < 4:
                        strong_lines[n4] = lines[n3]
                        n4 = n4 + 1
                    if not any(closeness) and n5 < 2:
                        strong_lines_2D[n5] = lines[n3]
                        n5 = n5 + 1

                    elif not any(directionality_theta) and n5 == 2:
                        strong_lines_2D[n5] = lines[n3]
                        n5 = n5 + 1

                    elif not any(directionality_theta) and n5 == 3:
                        closeness_rho_2D = np.isclose(
                            rho, strong_lines_2D[2, 0, 0], atol=10)
                        if not closeness_rho_2D:
                            strong_lines_2D[n5] = lines[n3]
                            n5 = n5 + 1

    return strong_lines, image, edged


def find_module_corners(display_image, blurring_steps=None):
    def define_Intersection(L1, L2):
        # determines the intersection of two lines
        D = L1[0]*L2[1] - L1[1]*L2[0]
        Dx = L1[2]*L2[1] - L1[1]*L2[2]
        Dy = L1[0]*L2[2] - L1[2]*L2[0]
        if D != 0:
            x = Dx/D
            y = Dy/D
            return x, y
        else:
            return False

    def define_Line(p1, p2):
        # defines the parameters of a line from two points
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C

    # Resize image for more efficient processing
    resize_height = 500
    display_image_copy = display_image.copy()
    ratio = display_image_copy.shape[0]/resize_height
    image_hough_lines = resize_image(display_image_copy, height=resize_height)

    # Apply edge detection function ***Beware this does not always work!***
    if blurring_steps is None:
        lines, image_blurred, image_threshold = detect_module_edges(
            image_hough_lines)
    else:
        lines, image_blurred, image_threshold = detect_module_edges(
            image_hough_lines, blurring_steps=blurring_steps)

    # Convert edges into module corners
    lines_xy = np.zeros([4, 2, 2], dtype=np.float32)

    for i in range(0, 4):
        for rho, theta in lines[i]:
            try:
                # print(rho,theta)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 10000*(-b))
                y1 = int(y0 + 10000*(a))
                x2 = int(x0 - 10000*(-b))
                y2 = int(y0 - 10000*(a))

                cv2.line(image_hough_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

                lines_xy[i, 0, 0] = x1
                lines_xy[i, 0, 1] = y1
                lines_xy[i, 1, 0] = x2
                lines_xy[i, 1, 1] = y2

            except ValueError:
                pass

    corners = np.zeros([4, 2], dtype=np.float32)
    # TODO: fix line length
    corners[0, :] = define_Intersection(define_Line(
        lines_xy[0, 0, :], lines_xy[0, 1, :]), define_Line(
            lines_xy[2, 0, :], lines_xy[2, 1, :]))
    corners[1, :] = define_Intersection(define_Line(
        lines_xy[0, 0, :], lines_xy[0, 1, :]), define_Line(
            lines_xy[3, 0, :], lines_xy[3, 1, :]))
    corners[2, :] = define_Intersection(define_Line(
        lines_xy[1, 0, :], lines_xy[1, 1, :]), define_Line(
            lines_xy[2, 0, :], lines_xy[2, 1, :]))
    corners[3, :] = define_Intersection(define_Line(
        lines_xy[1, 0, :], lines_xy[1, 1, :]), define_Line(lines_xy[3, 0, :], lines_xy[3, 1, :]))

    corners = order_module_corners(corners)*ratio

    return corners


def get_module_corners(image, manual=False):
    '''
    Finds the corners of a module using multiple image processing functions.

    Parameters
    ----------
    image : Array of uint8 (image)
        Module image.

    Returns
    -------
    corners : Array of float32
        The (x,y) position of the module's corners.

    '''
    corners_lim = 0.9*max(len(image[0, :]), len(image[:, 0]))
    wi = 0
    wii = 0

    if manual:
        corners = select_module_corners(image)
    else:
        try:
            corners = find_module_corners(image, blurring_steps=0)

            wi = wi + 1

            # While corners are erroneous, increase blur steps
            while (corners.min() < 0 or abs(corners[0, 0] - corners[1, 0]) < 0.5*corners_lim or
                   abs(corners.max()) > corners_lim or 0 in corners or
                   (corners.max() - corners.min()) < 0.4*corners_lim):

                corners = find_module_corners(image, blurring_steps=wi)

                wi = wi + 1

                # After increasing blur step to 10, reset to 0 and start another loop
                # with a digital gain of 2 to brighten image

                while wi > 9:
                    corners = find_module_corners(image*2, blurring_steps=wii)

                    wii = wii + 1

                    # If auto cropping still fails, user manually marks corners
                    # Instructions printed in console
                    if wii == 10:
                        print(
                            f'Auto cropping attempted {str(wi+wii+1)} iteration(s).')
                        print(
                            'Error in automatic edge detection: switching to manual method.')
                        print(
                            'Click on module corners. Press \'r\' before clicking. Press \'r\' to reset and \'c\' when complete.')

                        corners = select_module_corners(image)

                        break
                if wii == 10:
                    break

            if wii < 10:
                print(f'Auto cropping took {str(wi+wii+1)} iteration(s).')
        except:
            print("Error occurred. Switching to manual.")
            corners = select_module_corners(image)

    # flatten_corners(corners)

    return corners


def crop_module(image, corners, border_percentage=0):
    '''
    Crops the module from the original image using the corners.
    Adds a border if specified.

    Parameters
    ----------
    image : array of uint8 (image)
        Module image.
    corners : Array of float32
        Holds the (x,y) position of the module's corners.
    border_percentage: float
        Specifies a percentage border around the cropped module.
        Defaults to 0 if not specified

    Returns
    -------
    array of uint8 (image)
        Cropped module image.

    '''
    corners = corners.astype('int')
    border_percentage = border_percentage/100

    # Add border to cropped module image
    q = int(len(image)*border_percentage)
    qq = int(len(image[0, :])*(border_percentage))

    return image[(corners[0, 1] - q):(corners[2, 1] + q),
                 (corners[0, 0] - qq):(corners[1, 0] + qq)]


def extract_gridpoints(corners, num_cells_x, num_cells_y, dtype=float):
    '''
    Finds the (x,y) coordinates for the corners of each cell on the module by
    creating a regular 2D grid from given corner points (4*(x0, y0))
    of the module

    Parameters
    ----------
    corners : Array of float32
        Holds the (x,y) position of the module's corners.
    num_cells_x : int
        Holds the number of cells per a row.
    num_cells_y : int
        Holds the number of cells per a column.
    dtype : type, optional
        Holds the datatype for calculating the corners of cells.
        The default is float.

    Returns
    -------
    p : Array of float64
        Holds all of the (x,y) coordinates for each corner of each cell in
        the module.
        Horizontal and vertical lines as (x0, y0, x1, y1)

    '''
    corners = order_module_corners(corners)
    sx, sy = num_cells_x + 1, num_cells_y + 1

    # Horizontal lines
    x0 = np.linspace(corners[0, 0], corners[3, 0], sy, dtype=dtype)
    x1 = np.linspace(corners[1, 0], corners[2, 0], sy, dtype=dtype)
    y0 = np.linspace(corners[0, 1], corners[3, 1], sy, dtype=dtype)
    y1 = np.linspace(corners[1, 1], corners[2, 1], sy, dtype=dtype)

    # Points
    p = np.empty(shape=(sx*sy, 2))
    n0 = 0
    n1 = sx

    for x0i, x1i, y0i, y1i in zip(x0, x1, y0, y1):
        p[n0:n1, 0] = np.linspace(x0i, x1i, sx)
        p[n0:n1, 1] = np.linspace(y0i, y1i, sx)
        n0 = n1
        n1 += sx

    return p, num_cells_x, num_cells_y


def generate_gridpoint_image(image, gridpoints, working_dir):
    '''
    Places points on the corners of each cell based on the gridpoint positions

    Parameters
    ----------
    image : array of uint8 (image)
        Module image.
    gridpoints : Array of float64
        Holds all of the (x,y) coordinates for each corner of each cell in
        the module.
    working_dir : str
        The directory of the selected module images.

    Returns
    -------
    gridpoint_image : array of uint8 (image)
        Module image with gridpoints marked

    '''
    gridpoint_image = image.copy()

    # Create gridpoints on original image
    for x0, y0 in gridpoints:
        cv2.circle(gridpoint_image, (int(x0), int(y0)),
                   2, (0, 255, 0), thickness=2)

    return gridpoint_image


def find_num_cells(gridpoints):
    '''
    Find the number of cells in each row and column based on the gridpoints.

    Parameters
    ----------
    gridpoints : Array of float64
        Holds all of the (x,y) coordinates for each corner of each cell
        in the module.

    Returns
    -------
    num_cells_x : int
        Holds the number of cells per a row.
    num_cells_y : int
        Holds the number of cells per a column.

    '''
    try:
        num_cells_x = 1
        for index in range(1, len(gridpoints)):
            if (gridpoints[index][0] < gridpoints[index - 1][0]):
                break
            else:
                num_cells_x += 1
        num_cells_y = int(len(gridpoints)/num_cells_x)
        return num_cells_x - 1, num_cells_y - 1

    except TypeError:
        num_cells_x = gridpoints[1]
        num_cells_y = gridpoints[2]
        return int(num_cells_x), int(num_cells_y)


def cell_parsing(image, gridpoints):
    '''
    Uses the gridpoints to parse the individual cells from
    the inputted module image

    Parameters
    ----------
    image : array of uint8 (image)
        Module image.
    gridpoints : Array of float64
        Holds all of the (x,y) coordinates for each corner of each cell
        in the module.

    Returns
    -------
    cell_images : list
        The parsed cell images from the inputted module image.

    '''
    # List to hold cell images
    cell_images = []

    num_cells_x = gridpoints[1]
    num_cells_y = gridpoints[2]
    gridpoints = gridpoints[0]

    for row in range(0, num_cells_y):
        for col in range(0, num_cells_x):
            point = num_cells_x*row + col + row
            cell_images.append(image[int(gridpoints[point, 1]):int(gridpoints[point + num_cells_x + 2, 1]),
                                     int(gridpoints[point, 0]):int(gridpoints[point + num_cells_x + 2, 0])])

    return cell_images


def create_3d_images(files):
    '''
    Resizes and appends multiple images, saving them in a
    3D numpy array.

    Parameters
    ----------
    files : str
        Module filepaths.

    Returns
    -------
    image_3d : Array of uint8
        Image files saved in 4d array (Index, Height, Width, Color channel)

    '''
    basewidth = 1000
    images_3d = []

    for file in files:
        # Convert NEF image to PIL image object
        if '.NEF' in file:
            img = rp.imread(file).postprocess()
            img = Image.fromarray(img)
        else:
            img = Image.open(file)

        # Expand image
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth, hsize))

        images_3d.append(img)

    images_3d = np.stack(images_3d, axis=0)

    return images_3d


def save_gif_image(images_3d, text, filename, duration=500):
    '''
    Adds text to multiple images and saves as a gif. For resizing images for
    consistency, run create_3d_images and use its return for images_3d here.

    Parameters
    ----------
    images_3d : Array of uint8
        Image files saved in 4d array (Index, Height, Width, Color channel).
        I.e., just use io.imread or cv2.imread for your jpgs and make a list.
    text : str, list, or tuple
        Text to write on image.
    filename :
        Base filename to save gif as. You do not need to add '.gif'
    duration : int or float
        Time spent on each image in ms. Default is 500 ms.
    Returns
    -------
    None.

    '''
    gif = []

    for i, img in enumerate(images_3d):
        if type(text) is list or tuple:
            image_text = text[i]
        else:
            image_text = text

        # Convert img to PIL image object
        img = Image.fromarray(img)

        # Write on PIL object

        # Add text to images
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", 35)
        draw.text((20, 10), image_text, font=font, fill=(255))

        gif.append(img)

    # Save the gif
    iio.imwrite(f"{filename}.gif", gif, duration=duration, loop=0)


#--------------------------------Image Saving---------------------------------#
def save_module(directory, filename, image, image_32F=False,
                ir_image=False, pltsave=False):
    '''
    Save the cropped module image.

    Parameters
    ----------
    working_dir : str
        The directory of the selected module images.
    cropped_image : array of uint8 (image)
        Cropped module image.

    Returns
    -------
    None.

    '''

    # Make directory to export images
    if not os.path.isdir(directory):
        os.mkdir(directory)

    if ir_image or pltsave:
        plt.imsave(f'{directory}\{filename}', image)
    elif image_32F:
        io.imsave(f'{directory}\{filename}', image)
    else:
        cv2.imwrite(f'{directory}\{filename}', image)


def save_cells(dst, cell_images):
    '''
    Save the cell images

    Parameters
    ----------
    cell_images : list
        Parsed cells from module.
    working_dir : str
        The directory of the selected module images.

    Returns
    -------
    None.

    '''

    # Make directory for cropped cell images
    if not os.path.isdir(dst):
        os.mkdir(dst)

    # Writing image files for cropped cells, cropped cells enhanced
    for index, cell in enumerate(cell_images):
        cell_num = str(index + 1).rjust(len(str(len(cell_images))), '0')
        cv2.imwrite(f'{dst}/cell_{cell_num}.tif', cell)
        #imageio.imsave(f'{filepath_cell_images}cell_{cell_num}.NEF', cell)


def save_cell_IV(dst, cell_IV):
    '''
    Save the dark cell I-V data.

    Parameters
    ----------
    working_dir : str
        The directory of the selected module images.
    cell_IV : Array of float64
        Dark I-V data for each cell in the module.

    Returns
    -------
    None.

    '''
    # Make directory for I-V .txt exports
    if not os.path.isdir(dst):
        os.mkdir(dst)

    for i in range(1, len(cell_IV[0, 0, :]) + 1):
        cell_num = str(i).rjust(len(str(len(cell_IV[0, 0, :]))), '0')
        np.savetxt(f'{dst}cell_{cell_num}.dat',
                   cell_IV[:, :, i-1], fmt='%.5f', delimiter='\t', comments="")


# -------------------------Additional Image Processing-------------------------#TODO

def perspective_correction(image, corners, orientation='landscape'):
    # ,resize_height=500):
    '''
    Transforms a module image to better conform with the commonly accepted
    distortions in constructed perspective. Makes the image appear as if 
    you are looking at the image straight on.

    Parameters
    ----------
    corners : Array of float32
        The (x,y) position of the module's corners.
    image_Original : Array of uint8 (image)
        Module image.
    orientation : str
        Orientation of the module.

    Returns
    -------
    image_warped : Array of float32
        The persepective corrected module image.

    '''
    # TEMPORARILY COMMENTED OUT 4/2/2024 - likely won't add it back in
    """
    or_ratio = {'portrait': [1, 2], 'landscape': [-1, 1]}

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    # ratio keeps aspect ratio of cells realistic
    # (right-left) / (bottom-top)
    ratio = (max(corners[:, 0])-min(corners[:, 0])) / \
        (max(corners[:, 1])-min(corners[:, 1]))
    ratio = ratio**or_ratio[orientation][0]
    """
    (tl, tr, br, bl) = corners.astype('float32')  # *ratio

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = int(max(int(widthA), int(widthB)))
    # / ratio/or_ratio[orientation][1])
    maxHeight = int(max(int(heightA), int(heightB)))  # /ratio)

    dst = np.float32(
        [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]])

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    corners = corners.astype('float32')
    M = cv2.getPerspectiveTransform(corners, dst)
    image_warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return image_warped


def find_highest_pixels(cell_images, percent):
    '''
    Finds the highest intensity pixel for each cell, filtering out the
    top percent.

    Parameters
    ----------
    cell_images : list
        Holds all of the cropped cell images
    percent : float
        Any cell pixel intensity values above this percentage are filtered out
        during analysis. Percentage cannot equal zero.
        E.g., for ``percentage = 0.01``, the top 1% of pixel intensities for
        each cell will be filtered out.

    Returns
    -------
    highest_pixel : list
        All of the highest intensity pixels for each cell.

    '''
    highest_pixel = []

    for cell in cell_images:
        image_Array = cell.reshape(-1, 1)
        image_Array = np.sort(image_Array, 0)
        # Applying filtering with percent
        highest_pixel.append(
            float(image_Array[round(len(image_Array) - 1 - len(image_Array)*percent)]))

    return highest_pixel


# Used to be in main EL_sweep anaylsis
# Calibration constant calculated for PVRs functions. Not used for EL sweep.
def calibration(calib_image, gridpoints, calibration_voltage, bit_depth=16):
    '''
    Calculates the calibration constant of a inputted module image and the highest
    intensity pixel for each cell in the same inputted module image.

    Parameters
    ----------
    calib_image : Array of uint8 (image)
        Module calibration image.
    gridpoints : Array of float64
        Holds all of the (x,y) coordinates for each corner of each cell
        in the module.
    bit_depth : int, optional
        Image bit depth used for filtering pixels that are noise, erroneous,
        or saturated. The default is 16.
    calibration_voltage : float
        The bias voltage of the calibration module.

    Returns
    -------
    calibration_constant : float
        Constant used to account for opitical and material properties of the
        module and camera system.
    cal_highest_pixels : Array of float64
        Highest intensity pixel for each cell in the calibration module.

    '''
    thermal_voltage = 25.85e-3

    num_cells_x = gridpoints[1]
    num_cells_y = gridpoints[2]
    #gridpoints = gridpoints[0]
    # Filtering saturated pixels
    # if l[0] has higher values than image, it counts back from 2**bit_depth
    calib_image[calib_image > int(0.99*2**bit_depth)] = 1

    cell_images = cell_parsing(calib_image, gridpoints)
    cal_highest_pixels = np.array(
        find_highest_pixels(cell_images, percent=0.01))

    calibration_constant = (decimal.Decimal(np.prod(cal_highest_pixels)))
    calibration_constant /= decimal.Decimal(
        calibration_voltage/thermal_voltage).exp()
    calibration_constant **= decimal.Decimal(1/(num_cells_x*num_cells_y))

    calibration_constant = float(calibration_constant)

    return calibration_constant, cal_highest_pixels


def map_parameter(gridpoints, parameter_array):
    '''
    Stores a value for each cell of a module based on a specified
    parameter (key).

    Parameters
    ----------
    gridpoints : Array of float64
        Holds all of the (x,y) coordinates for each corner of each cell
        in the module.
    parameter_array : Array of float64
        Holds the values of the specified parameter for each cell.

    Returns
    -------
    parameter_map : Array of float64
        Holds the parameter value of each cell in the module.

    '''
    # Translate original gridpoints to origin at top left corner of image
    # This defines a new coordinate system for the parameter map
    gridpoints[:, 0] = gridpoints[:, 0] - min(gridpoints[:, 0])
    gridpoints[:, 1] = gridpoints[:, 1] - min(gridpoints[:, 1])

    num_cells_x = len(list(set(gridpoints[:, 0]))) - 1
    num_cells_y = len(list(set(gridpoints[:, 1]))) - 1

    x_points = sorted([int(q) for q in set(gridpoints[:, 0])])
    y_points = sorted([int(q) for q in set(gridpoints[:, 1])])

    parameter_map = np.zeros(
        [int(max(gridpoints[:, 1])), int(max(gridpoints[:, 0]))])

    cell_idx = 0
    for row in range(num_cells_y):
        for col in range(num_cells_x):
            parameter_map[y_points[row]:y_points[row + 1],
                          x_points[col]:x_points[col + 1]] = parameter_array[cell_idx]
            cell_idx += 1

    return parameter_map


def correct_lens_distortion(image, coefficients=None):
    if coefficients == None:
        coefficients = [-1.0e-5, 0, 0, 0]
    # coefficients is list of 4 parameter [k1,k2,p1,p2]

    src = image
    width = src.shape[1]
    height = src.shape[0]
    distCoeff = np.zeros((4, 1), np.float64)

    k1 = coefficients[0]  # negative to remove barrel distortion
    k2 = coefficients[1]
    p1 = coefficients[2]
    p2 = coefficients[3]

    distCoeff[0, 0] = k1
    distCoeff[1, 0] = k2
    distCoeff[2, 0] = p1
    distCoeff[3, 0] = p2

    # assume unit matrix for camera
    cam = np.eye(3, dtype=np.float32)

    cam[0, 2] = width/2.0  # define center x
    cam[1, 2] = height/2.0  # define center y
    cam[0, 0] = 10.        # define focal length x
    cam[1, 1] = 10.        # define focal length y

    # here the undistortion will be computed
    dst = cv2.undistort(src, cam, distCoeff)

    # cv2.imshow('dst',dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return dst


def translate_image(image, corners, delta_x=0, delta_y=0):
    rows, cols = image.shape[:2]
    # positive values shift right, down
    M_t = np.float32([[1, 0, int(delta_x)], [0, 1, int(delta_y)]])
    translated_image = cv2.warpAffine(image, M_t, (cols, rows))

    return translated_image


def vignette(current_image, factor_a=0.5, factor_b=0.9):

    InputImage = current_image.copy()

    # Show on external window
    # cv2.imshow('image',InputImage)
    rows, cols = InputImage.shape[:2]
    # generating vignette mask using Gaussian kernels
    # sigma is second argument
    kernel_x = cv2.getGaussianKernel(cols, cols*factor_a)
    kernel_y = cv2.getGaussianKernel(rows, rows*factor_b)
    # kernel is a 2D array
    kernel = kernel_y*kernel_x.T

    kernel_curve_a = cv2.getGaussianKernel(cols, cols*factor_a)
    kernel_curve_b = cv2.getGaussianKernel(rows, rows*factor_b)

    mask = 1.0/(kernel/np.max(kernel))
    kernel_curve_a = 1.0/(kernel_curve_a/np.max(kernel_curve_a))
    kernel_curve_b = 1.0/(kernel_curve_b/np.max(kernel_curve_b))

    # create an image 0-255 of the mask
    v_mask = 255*np.copy(mask)/np.max(mask)
    v_mask = v_mask.astype(np.uint8)

    # If we have a grayscale image, only need to do ONCE
    # for color, all elements have to be multiplied (R, G, and B)
    output = InputImage.copy()
    output = np.where(output*mask <= 255, output*mask, 255)
    # make sure integer; otherwise, issues
    output = output.astype(np.uint8)

    return output.copy()


def convert_to_grayscale(image):
    '''
    Converts an image to grayscale

    Parameters
    ----------
    image : Array of uint8 (image)
        Color image.

    Returns
    -------
    Array of uint8 (image)
        Grayscale image.

    '''

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # *2#image#cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#*2


def histogram_stretch(image, percent_filter=0.05):
    '''
    Perform histogram stretching on an image to help increase contrast.

    Parameters
    ----------
    image : Array of uint8 (image)
        Image.
    percent_filter : float, optional
        Strips off this percentage of high pixels from the image.
        The default is 0.05.

    Returns
    -------
    hist_image : Array of uint8 (image)
        The increased contrast image.

    '''
    line = np.reshape(image, [1, -1])
    line_sort = np.sort(line[0])

    upper_limit = line_sort[-int(percent_filter*len(line[0]))]

    # TODO: look into using percent_filter here
    lower_limit = np.min(image)

    image[image > upper_limit] = upper_limit
    image[image < lower_limit] = lower_limit

    hist_image = (image - lower_limit)*(255/(upper_limit - lower_limit))
    hist_image = hist_image.astype(np.uint8)

    return hist_image


def use_metadata_corners(working_dir):
    '''
    Use the corners saved in the ModuleMetadata.txt file.

    Parameters
    ----------
    working_dir : str
        The directory of the selected module images.

    Returns
    -------
    corners : Array of float32
        The (x,y) position of the module's corners.

    '''
    module_metadata = pd.read_csv(
        f'{working_dir}ModuleMetadata.txt', sep='\t', header=None).set_index(0)
    corners = np.zeros([4, 2])
    corners[0, :] = [module_metadata.loc['Left Boundary', 1],
                     module_metadata.loc['Top Boundary', 1]]
    corners[1, :] = [module_metadata.loc['Right Boundary', 1],
                     module_metadata.loc['Top Boundary', 1]]
    corners[2, :] = [module_metadata.loc['Right Boundary', 1],
                     module_metadata.loc['Bottom Boundary', 1]]
    corners[3, :] = [module_metadata.loc['Left Boundary', 1],
                     module_metadata.loc['Bottom Boundary', 1]]

    return corners


def EL_sweep_analysis(EL_metadata,
                      gridpoints,
                      percent=0.01,
                      bit_depth=16,
                      image_32F=False):
    '''
    This function extracts dark current-voltage (I-V) data for each cell within 
    a module by analyzing a series of electroluminescence images. 
    Images are selected by the user and are passed to this function as a dictionary.

    Parameters
    ----------
    EL_metadata : dict
        Holds all metadata for selected modules in 5 dictionaries,
        with each dictionary holding a list. This includes filepaths,
        bias voltages (V) and currents (A), exposures (s), and temperatures (C).
        Each list is sorted based on the module with the highest current.
    gridpoints : Array of float64
        Holds all of the (x,y) coordinates for each corner of each cell in the module.
    percent : float, optional
        Any cell pixel intensity values above this percentage are
        filtered out during analysis. Percentage cannot equal zero.
        E.g., for ``percentage = 0.01``, the top 1% of pixel
        intensities for each cell will be filtered out. 
        The default is 0.01.
    bit_depth : int, optional
        Image bit depth used for filtering pixels that are
        noise, erroneous, or saturated.
        The default is 16.
    image_32F : bool, optional
        True only if using 32-bit float images.
        We've only had to use this for one type of InGaAs camera.
        The default is False.

    Returns
    -------
    cell_IV : Array of float64
        Holds all of the cell dark current_voltage(I-V) data.
    '''

    # Define Constants
    thermal_voltage = 25.85e-3

    try:
        num_modules = len(EL_metadata['filepath'])
        num_cells_x, num_cells_y = find_num_cells(
            gridpoints)  # Works as before
    except TypeError:
        print(gridpoints + "EL Sweep")
        num_modules = 2  # For use in pixel paremeters
        num_cells_x = gridpoints[1]
        num_cells_y = gridpoints[2]
        gridpoints = gridpoints[0]
    finally:
        num_cells = num_cells_x*num_cells_y

    # Initialize Arrays
    cell_resistance_factor = np.zeros([num_modules, num_cells])
    highest_pixel = np.zeros([num_modules, num_cells])

    #--------------Determining Operating Voltage for Each Cell---------------#
    for mod_idx in range(0, num_modules):
        image = choose_image_reader(EL_metadata['filepath'][mod_idx],
                                    for_analysis=True, image_32F=image_32F)

        # Filtering saturated pixels
        # if l[0] has higher values than image, it counts back from 2**bit_depth
        image[image > int(0.99*2**bit_depth)] = 1

        # Scale exposure time so all pixels are on the same scale
        # max exposure time / image exposure time is the scaling factor
        image = image*max(EL_metadata['exposure']) / \
            EL_metadata['exposure'][mod_idx]
        cell_images = cell_parsing(image, gridpoints)
        highest_pixel[mod_idx] = find_highest_pixels(cell_images, percent)

    # Sometimes, NumPy functions do not work because of very large numbers,
    # so values are divided by 100 prior to the calculations
    highest_pixel = highest_pixel/100

    if num_cells_x > 30:
        parallel_strings = num_cells_y
    elif (num_cells_x > 12) and (num_cells_x < 30):
        parallel_strings = 2
    else:
        parallel_strings = 1

    # Zeros(number of different module currents, cols for current and voltage, number of cells)
    cell_IV = np.zeros((num_modules, 2, num_cells))
    for mod_idx in range(0, num_modules):
        for cell_idx in range(0, num_cells):
            # Calculated for each cell in each module.
            cell_resistance_factor[mod_idx, cell_idx] = thermal_voltage * \
                np.log(highest_pixel[mod_idx, cell_idx])

        T_corr = 298.15  # 'Ideal' temperature, e.g., 25C
        T = EL_metadata['temperature'][mod_idx] + 273.15
        tempco_factor = (T - T_corr)/T

        # Determine Cell Operating Voltage
        for cell_idx in range(0, num_cells):
            cell_vop = cell_resistance_factor[mod_idx, cell_idx] + (
                EL_metadata['voltage'][mod_idx] - np.sum(cell_resistance_factor[mod_idx, :]))/num_cells
            cell_vop *= parallel_strings  # Accounts for parallel parallel_strings
            # Positive tempco_factor means voltage is added to hot cell_vop
            cell_vop = tempco_factor*cell_vop + cell_vop
            cell_IV[mod_idx][0][cell_idx] = EL_metadata['current'][mod_idx] / \
                parallel_strings
            cell_IV[mod_idx][1][cell_idx] = cell_vop

    return cell_IV


def crop_module_pipeline(img, manual=False, border_percentage=3):
    corners = get_module_corners(img, manual=manual)
    cropped = crop_module(img, corners, border_percentage)

    return cropped


def cell_parsing_cut_cells(image, num_cells_x=24, num_cells_y=6):
    """


    Parameters
    ----------
    image : Array of uint8 (image)
        Perspective corrected image.
    num_cells_x : int, optional
        Number of cells in x-axis, e.g., columns. The default is 24.
    num_cells_y : int, optional
        Number of cells in y-axis, e.g., rows. The default is 6.

    Returns
    -------
    all_cells_ordered : list of Arrays of unit8
        List of cell images in order shown on module image as read from
        left to right, top to bottom.

    """

    # As of 4/2/2024, this version is made for half cut cells. This should
    # be updated to work with shingled modules later on.

    # It is strongly recommended to perspective correct the image first,
    # as this code requires a perfect rectangle to function properly.

    # Automatically detect number of parallel strings - may come in handy
    '''
    if num_cells_x > 30:
        parallel_strings = num_cells_y
    elif (num_cells_x > 12) and (num_cells_x < 30):
        parallel_strings = 2
    else:
        parallel_strings = 1
    '''
    # treat each half like a module
    # get corners, gridpoints, crop cells, then organize cell list
    center_idx = int(image.shape[1]/2)

    # left half, then right half
    all_cells = []
    # for half in (image[:, :center_idx, :], image[:, center_idx:, :]):
    for half in (image[:, :center_idx], image[:, center_idx:]):
        corners = get_boundary_corners(half)
        gridpoints = extract_gridpoints(corners, int(
            num_cells_x/2), num_cells_y, dtype=float)
        all_cells.append(cell_parsing(half, gridpoints))

    # Indexes based on full module. This runs through the cell lists for each
    # half of the module. E.g., row 1 of left half, then row 1 of right half,
    # row 2 of left half, etc.
    all_cells_ordered = []
    for row in range(0, num_cells_y):
        for column in range(0, num_cells_x):
            print(row, column)
            # For the left half of the module
            if column < num_cells_x/2:
                all_cells_ordered.append(
                    all_cells[0][int(row*num_cells_x/2)+column])
            # For the right half of the module
            else:
                all_cells_ordered.append(
                    all_cells[1][int(row*num_cells_x/2)+column-int(num_cells_x/2)])

    return all_cells_ordered


# ========================== IR ============================#TODO
def convert_flir_ir(ir_file):
    '''
    Converts FLIR IR pixel values to temperature in C.

    Parameters
    ----------
    ir_file : filepath
        IR image filepath.

    Returns
    -------
    ir_image : Array of float64
        Converted image with pixel values equal to temperature in C.

    '''
    return io.imread(ir_file)*0.04 - 273.15

# TODO: add docstring


def read_ir_image(filename):
    data = pd.read_csv(filename, sep='\t', header=None)
    data = data[::-1][data.columns[::-1]]
    data = data.reset_index(drop=True)

    return data


def ir_module_filter(data, filter_cols=25, filter_rows=100, percent_buffer=0):
    ##data = pd.read_csv(filename, sep='\t', header=None)
    # the temperature data is rotated 180deg -this aligns it with image
    ##data = data[::-1][data.columns[::-1]]
    ##data = data.reset_index(drop=True)

    # filter all rows and columns based on sums of temperature values
    # temperatures below mean + standard deviation are filtered
    # 300 threshold for JPG
    testData = data
    if np.max(data.values) > 300:
        testData = data[data > (data.values.mean() + np.std(data.values))]
        # TODO: Needed?
        testData[data < (data.values.mean() + np.std(data.values))] = 0

    # empirical filter values found to work on about 200 test
    # images of 4 vintages
    # with <5 outliers
    cols = testData.sum(axis=0) > filter_cols*data.values.mean()
    rows = testData.sum(axis=1) > filter_rows*data.values.mean()
    cols = np.where(cols == True)[0]
    rows = np.where(rows == True)[0]
    rows_buffer = (percent_buffer/100)*len(rows)
    cols_buffer = (percent_buffer/100)*len(cols)

    col1 = cols[0] - cols_buffer
    col2 = cols[-1] + cols_buffer
    row1 = rows[0] - rows_buffer
    row2 = rows[-1] + rows_buffer

    (col1, col2, row1, row2) = [int(q) for q in [col1, col2, row1, row2]]
    (col1, row1) = [0 if q < 0 else q for q in [col1, row1]]
    if col2 > testData.shape[1]:
        col2 = testData.shape[1]
    if row2 > testData.shape[0]:
        row2 = testData.shape[0]

    cropped_module = data.iloc[row1:row2, col1:col2]

    return cropped_module


def save_ir_image(ir_data, ir_filename, temp_range=[]):
    # Set temperature range in figure based on image itself or user inputs
    if not temp_range:
        temp_range = [np.min(np.min(ir_data))-2, np.max(np.max(ir_data))+2]

    # Plot IR on color map with color bar label
    plt.imshow(ir_data, cmap='inferno', vmin=temp_range[0], vmax=temp_range[1])
    cbar = plt.colorbar()
    cbar.set_label('Temperature °C')
    plt.axis('off')
    # only difference between higher DPIs are sharpness of colorbar label
    # when zooming in. Increase DPI for publication quality images.
    plt.savefig(ir_filename, bbox_inches='tight', pad_inches=0.1, dpi=400)
    plt.close()

#-------------------------------------------------------------------------#


def sort_data(EL_metadata):
    '''
    Sorts all the data from lowest to highest current.

    Parameters
    ----------
    EL_metadata : dict
        Holds all the module information for each selected image.

    Returns
    -------
    None.

    '''
    # Retrieve indices for lists of values and apply to each key
    new_indices = np.argsort(EL_metadata['current'])
    for key in EL_metadata:
        EL_metadata[key] = [EL_metadata[key][i] for i in new_indices]


def FSEC_data_formatting(image_files):
    '''
    Formats the module information for each selected image into a 
    dictionary of lists.

    Parameters
    ----------
    image_files : tuple
        All the selected images

    Raises
    ------
    Exception
        Selected file types are not supported.

    Returns
    -------
    EL_metadata : dict
        Holds all the module information for each selected image.

    '''
    # Keys:
    # filepath
    # voltage (V)
    # current (A)
    # exposure (S)
    # temperature (C)
    # Dictionary of lists holding module metadata
    EL_metadata = {'filepath': [],
                   'voltage': [],
                   'current': [],
                   'exposure': [],
                   'temperature': []}

    for file in image_files:
        name = None
        # Exclude directories
        if os.path.isfile(file):
           # try to use these files
            if '.tif' in file:
                if 'RAW.tif' not in file:
                    pass
                else:
                    name = basename(file).split('_')
                    name.pop()  # Remove file name extension from list
            elif '.NEF' in file:
                name = basename(file.replace('.NEF', '')).split('_')
            else:
                pass
        if name:
            EL_metadata['filepath'].append(file)
            # Remove char from number value in file name
            EL_metadata['voltage'].append(float(name[-1].replace('V', '')))
            EL_metadata['current'].append(float(name[-2].replace('A', '')))
            EL_metadata['exposure'].append(float(name[-3].replace('s', '')))

            # Assign Temperature constant
            EL_metadata['temperature'].append(float(25))

    # Error if none of the files are allowed
    if not EL_metadata:
        raise Exception(
            'None of the files you selected are supported. Please only use .tif or .NEF files only.')

    sort_data(EL_metadata)

    return EL_metadata

# ========================== SCANNER ============================#TODO


def read_nc_file(nc_file, image_type='EL'):
    '''
    Options for image_type: EL, PL, UVF
    '''
    data_structure = xr.load_dataset(nc_file)
    image_metadata = data_structure.attrs

    cell_images = [data_structure.InGaAs[i].loc[image_type]
                   for i in range(len(data_structure.InGaAs))]
    cell_images = [cell_image.values for cell_image in cell_images]

    return cell_images, image_metadata


#-------------------------------------------------------------------------#
