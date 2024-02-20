from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

# Directory containing data and images
in_dir = "C:/Users/Christian/Documents/DTU/Billedanalyse/DTUImageAnalysis-main/DTUImageAnalysis-main/exercises/ex1-IntroductionToImageAnalysis/data/"

# X-ray image
im_name = "metacarpals.png"

# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org = io.imread(in_dir + im_name)
#print(im_org.shape)
#print(im_org.dtype)

# display image with original colour map
#io.imshow(im_org)
#plt.title('Metacarpal image')
#io.show()

# display image with different colour map
#io.imshow(im_org, cmap="terrain")
#plt.title('Metacarpal image (with colormap)')
#io.show()

# pixel scaling everyhting below vmin pixels gets darker and above lighter
#io.imshow(im_org, vmin=100, vmax=170)
#plt.title('Metacarpal image')
#io.show()

# pixel scaling everyhting automatically
#smallest = np.amin(im_org)
#biggest = np.amax(im_org)
#io.imshow(im_org, vmin=smallest, vmax=biggest)
#plt.title('Metacarpal image automatic pixel scaling')
#io.show()

# Compute and visualise the histogram of the image
#plt.hist(im_org.ravel(), bins=256) #ravel is called to convert the image into a 1D array
#plt.title('Image histogram')
#io.show()

# The value of a given bin can be found by
#h = plt.hist(im_org.ravel(), bins=256)
#bin_no = 100
#count = h[0][bin_no]
#print(f"There are {count} pixel values in bin {bin_no}")
#bin_left = h[1][bin_no]
#bin_right = h[1][bin_no + 1]
#print(f"Bin edges: {bin_left} to {bin_right}")

# ALTERNATIVE WAY OF CALLING IT
#y, x, _ = plt.hist(im_org.ravel(), bins=256)

# function to find the most common range of intensities
#bin_no = np.argmax(h[0])
#count = h[0][bin_no]
#print(f"{count} is the most common range")

### Exercise 9 - 13 ######################################################################################
#r = 100
#c = 90
#im_val = im_org[r, c]
#print(f"The pixel value at (r,c) = ({r}, {c}) is: {im_val}")

# CUT OF PIXELS FROM 0 TO 30
#im_org[:30] = 0
#io.imshow(im_org)
#io.show()

# MASK SHOWS COLOURS AS TRUE OR FALSE / BLACK OR WHITE
#mask = im_org > 150
#io.imshow(mask)
#io.show()

# EVERYTHING WITH A MASK VALUE OF 1 IS 255 WHITE, EVERYTHING ELSE GRAYSCALE
#im_org[mask] = 255
#io.imshow(im_org)
#io.show()

### EXERCISE 14 ################################################################################
# Directory containing data and images
in_dir2 = "C:/Users/Christian/Documents/DTU/Billedanalyse/DTUImageAnalysis-main/DTUImageAnalysis-main/exercises/ex1-IntroductionToImageAnalysis/data/"

# X-ray image
im_name2 = "ardeche.jpg"

# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
#im_org2 = io.imread(in_dir2 + im_name2)
#print(im_org2.shape)
#print(im_org2.dtype)

# display image with original colour map
#io.imshow(im_org2)
#plt.title('Metacarpal image')
#io.show()

#r = 100
#c = 90
#im_val = im_org2[r, c]
#print(f"The pixel value at (r,c) = ({r}, {c}) is: {im_val}")

#im_org2[:300] = [0,255,0]
#io.imshow(im_org2)
#io.show()

### EXERCISE 17 ################################################################################

in_dir3 = "C:/Users/Christian/Documents/DTU/Billedanalyse/DTUImageAnalysis-main/DTUImageAnalysis-main/exercises/ex1-IntroductionToImageAnalysis/data/"
# X-ray image
im_name3 = "jafar.jpg"

# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org3 = io.imread(in_dir3 + im_name3)
#print(im_org3.shape)
#print(im_org3.dtype)

# display image with original colour map
#io.imshow(im_org3)
#plt.title('JAFAR')
#io.show()

image_rescaled = rescale(im_org3, 0.5, anti_aliasing=True, channel_axis=2)
#io.imshow(image_rescaled)
#io.show()
#print(image_rescaled.shape)
#print(image_rescaled.dtype)

#plt.hist(im_org.ravel(), bins=256) #ravel is called to convert the image into a 1D array
#plt.title('Image histogram')
#io.show()

r = 100
c = 90
im_val = im_org3[r, c]
#print(f"The pixel value at (r,c) = ({r}, {c}) is: {im_val}")

# scale the height and width of the image with different scales.
image_resized = resize(im_org3, (im_org3.shape[0] // 4,
                       im_org3.shape[1] // 6),
                       anti_aliasing=True)
#io.imshow(image_resized)
#io.show()

#automatically scale your image so the resulting width (number of columns) is always equal to 400
f = 400/im_org3.shape[1]
image_rescaled2 = rescale(im_org3, f, anti_aliasing=True, channel_axis=2)
#io.imshow(image_rescaled2)
#io.show()

im_gray = color.rgb2gray(im_org3)
im_byte = img_as_ubyte(im_gray)

# Compute and visualise the histogram of the image
#plt.hist(im_gray.ravel(), bins=256) #ravel is called to convert the image into a 1D array
#plt.title('Image histogram')
#io.show()

#io.imshow(im_gray)
#io.show()

### EXERCISE 22 ################################################################################

in_dir = "C:/Users/Christian/Documents/DTU/Billedanalyse/DTUImageAnalysis-main/DTUImageAnalysis-main/exercises/ex1-IntroductionToImageAnalysis/data/"
# X-ray image
im_name4 = "DTUSign1.jpg"

# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org4 = io.imread(in_dir + im_name4)
#print(im_org4.shape)
#print(im_org4.dtype)

# display image with original colour map
#io.imshow(im_org4)
#plt.title('SKILT')
#io.show()

r_comp = im_org4[:, :, 0]
#io.imshow(r_comp)
#plt.title('DTU sign image (Red)')
#io.show()

# Cut out a part of the image an choose colour (2 equal blue)
im_org4[1500:1700, 2200:2800, :] = 0
im_org4[1500:1700, 2200:2800, 2] = 255
#io.imshow(im_org4)
#io.show()

# Save the image
#io.imsave('ex1-IntroductionToImageAnalysis/data/DTUSign1-marked.jpg', im_org4)

### EXERCISE 27 ################################################################################
# convert the image to grayscale
rgb_image = color.gray2rgb(im_org)

# Create a zero matrix with the same dimensions as the image
blue_filter = np.zeros_like(rgb_image)

# Set the blue channel to the values of the original image
blue_filter[:,:,2] = rgb_image[:,:,2]

#io.imshow(blue_filter)
#io.show()

# MASK SHOWS COLOURS AS TRUE OR FALSE / BLACK OR WHITE
mask = im_org > 150
#io.imshow(mask)
#io.show()

# Create a mask for white color
# Assuming that white color in the image is represented by values close to 255 for uint8 images
mask_white = np.all(rgb_image > 150, axis=-1)  # You can adjust the threshold as needed

# Apply the blue color to the masked areas
rgb_image[mask_white] = [0, 0, 255]  # Blue color in RGB is represented by [0, 0, 255] for uint8 images

#io.imshow(rgb_image)
#io.show()

### EXERCISE 28 Advanced Image Visualisation ################################################################################
# The tool profile_line can be used to sample a profile across the bone:
p = profile_line(im_org, (342, 77), (320, 160))
plt.plot(p)
plt.ylabel('Intensity')
plt.xlabel('Distance along line')
#plt.show()

# An image can also be viewed as a landscape, where the height is equal to the grey level:
im_name = "road.png"
im_org = io.imread(in_dir + im_name)
im_gray = color.rgb2gray(im_org)
ll = 200
im_crop = im_gray[40:40 + ll, 150:150 + ll]
xx, yy = np.mgrid[0:im_crop.shape[0], 0:im_crop.shape[1]]
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xx, yy, im_crop, rstride=1, cstride=1, cmap=plt.cm.jet,
                       linewidth=0)
fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()


### DICOM images ################################################################################
im_name = "1-442.dcm"
ds = dicom.dcmread(in_dir + im_name)
print(ds)

im = ds.pixel_array
plt.imshow(im, vmin=-1000, vmax=1000, cmap='gray')
plt.show()


