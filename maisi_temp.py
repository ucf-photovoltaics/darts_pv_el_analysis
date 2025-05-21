# @ -0,0 +1,246 @@
# code written in 2024 by Jasen Szekely 〄
# maisi = modular alignment, isolation, subtraction imaging

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio as iio
import file_management as fm
import image as PVIM

###############################################################################
##
# Load Images
##
###############################################################################

# process for converting NEF files to png


def load_and_convert_nef(path_to_nef):
    import_image = PVIM.choose_image_reader(path_to_nef, for_analysis=True)
    # Normalize to uint8 range [0, 255]
    min_val = import_image.min()
    max_val = import_image.max()
    uint8_image = ((import_image - min_val) /
                   (max_val - min_val) * 255).astype(np.uint8)
    return uint8_image


def ensure_grayscale(image, file_name):
    if len(image.shape) == 3 and image.shape[2] == 3:  # Image is RGB
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        print(f"Converted '{file_name}' from RGB to grayscale.")
        return grayscale_image
    elif len(image.shape) == 2:  # Image is already grayscale
        print(f"'{file_name}' is already grayscale.")
        return image
    else:
        raise ValueError(
            f"Unexpected format for '{file_name}'. Cannot process.")


def crop_module(image_uint8, threshold_value=30):
    # Apply thresholding to create a binary mask
    _, binary_mask = cv2.threshold(
        image_uint8, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        raise ValueError(
            "No contours found. The object might not be visible or well-defined.")

    # Find the largest contour, assuming it's the module
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle for the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image using the bounding rectangle
    cropped_image = image_uint8[y:y+h, x:x+w]

    return cropped_image, binary_mask, (x, y, w, h)


# grab NEF files here. Need to be in the same directory.
#files_list = fm.get_files(title='Select 2 NEF files.')

if len(files_list) != 2:
    raise ValueError("Please select exactly two files.")
    files_list = fm.get_files(title='Select exactly 2 NEF files.')

path_to_nef1 = files_list[0]
path_to_nef2 = files_list[1]

# Load and convert NEF files to uint8 grayscale
# load_and_convert_nef(path_to_nef1)
image1 = PVIM.choose_image_reader(path_to_nef1, for_analysis=False)
# load_and_convert_nef(path_to_nef2)
image2 = PVIM.choose_image_reader(path_to_nef2, for_analysis=False)

# Check if images are grayscale
print(f"Image 1 shape (after conversion to uint8): {image1.shape}")
print(f"Image 2 shape (after conversion to uint8): {image2.shape}")

# =============================================================================
# ## print brightness statistics for debugging.
# mean_brightness_img1 = np.mean(f32_img1)
# mean_brightness_img2 = np.mean(f32_img2)
# print(f"Mean brightness Image 1: {mean_brightness_img1}")
# print(f"Mean brightness Image 2: {mean_brightness_img2}")
# =============================================================================

# =============================================================================
# ## Check if images are grayscale
# print(f"Image 1 shape (after grayscale check): {image1.shape}")
# print(f"Image 2 shape (after grayscale check): {image2.shape}")
# =============================================================================

# grab a png of image1 to show us later
iio.imsave('image1_normalized.jpg', cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY))

# =============================================================================
# storing as a PNG to feed the ORB monster
# the ORB monster will starve, take numpy array instead. Use png for ORB debug only.
#  image1 = iio.imsave('image1.png', loadedimage1)
#  image2 = iio.imsave('image2.png', loadedimage2)
#
#  ## Load the images with OpenCV and convert to grayscale
#  img1 = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)
#  img2 = cv2.imread('image2.png', cv2.IMREAD_GRAYSCALE)
#
#  # Normalize the images to make intensity values more consistent
#  img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)
#  img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)
# =============================================================================

###############################################################################
##
# Image Alignment
##
###############################################################################

# Use ORB for feature detection and matching
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# Matcher to find correspondences
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

# Check if we have enough matches
if len(matches) < 4:
    raise ValueError(
        "Not enough matches found. Make sure both images are of the same module, and that one isn't too dim.")

# Extract location of good matches
points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

# Find homography matrix and apply perspective transformation
H, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

# With the homography matrix found, we can apply this to the raw images
# and analyze those pixel values
raw_image1 = PVIM.choose_image_reader(path_to_nef1, for_analysis=True)
raw_image2 = PVIM.choose_image_reader(path_to_nef2, for_analysis=True)

# Normalize raw images to their maximum to account for imaging differences
raw_image1 = raw_image1/np.max(raw_image1)
raw_image2 = raw_image2/np.max(raw_image2)

aligned_image2 = cv2.warpPerspective(
    raw_image2, H, (raw_image1.shape[1], raw_image1.shape[0]))

# grab a png of image2 to show us later
iio.imsave('image2_normalized.jpg', (255*aligned_image2).astype(np.uint8))

###############################################################################
##
# ORB Debug
##
###############################################################################

# =============================================================================
# ## code to export keypoints from ORB for debugging, not acceptable for difference result due to possible camera-side postprocessing. comment out unless needed
#
#  ## Match features between the two images using a Brute-Force matcher
#  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#  matches = bf.match(descriptors1, descriptors2)
#
#  ## sort points by distance, lower is better
#  matches = sorted(matches, key = lambda x:x.distance)
# =============================================================================

###############################################################################
##
# Image Cropping
##
###############################################################################

# TODO - write image script, w/  thresholding, find contours, bounding rectangle, to find module.

###############################################################################
##
# Image Normalization
##
###############################################################################

# TODO - Write image normalization script of both, using each other's mean.

###############################################################################
##
# Image Subtraction
##
###############################################################################

#difference_image = cv2.absdiff(image1, aligned_image2)
difference_image = 100*cv2.absdiff(raw_image1, aligned_image2)

# grab a png of difference_image to show us later
iio.imsave('difference_image_output.jpg', difference_image.astype(np.uint8))

###############################################################################
##
# Statistics
##
###############################################################################

mean_diff = np.mean(difference_image)
std_diff = np.std(difference_image)

###############################################################################
##
# Visualization
##
###############################################################################

# blueprint for the plotting of the illustration to present to the user

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Image 1')
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Aligned Image 2')
plt.imshow(aligned_image2, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Difference Image')
plt.imshow(difference_image, cmap='gray')

plt.savefig('comparison.jpg', dpi=600)
# plt.show()

print(f"Mean difference: {mean_diff}")
print(f"Standard deviation of difference: {std_diff}")

# Histogram of the difference image
hist, bin_edges = np.histogram(difference_image)  # , bins=256, range=(0, 255))

plt.figure(figsize=(8, 4))
plt.title('Histogram of Difference Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.plot(bin_edges[0:-1], hist)  # bin_edges has one more value than hist
# plt.show()

# save png of histogram for refrence
plt.savefig('difference_histogram.jpg', dpi=600)

###############################################################################
##
# Debug
##
###############################################################################

# =============================================================================
# ## not needed onless you're trying to debug the ORB keypoints or just for fun, I guess
# img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# cv2.imwrite('orb_keypoints.png', img_matches)
# cv2.imshow('ORB keypoints', img_matches)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# =============================================================================
