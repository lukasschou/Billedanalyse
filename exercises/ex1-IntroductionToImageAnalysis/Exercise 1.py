from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

# Directory containing data and images
in_dir = "C:/Users/lukas/OneDrive/Uni/Billedanalyse/DTUImageAnalysis-main/DTUImageAnalysis-main/exercises/ex1-IntroductionToImageAnalysis/data/"

# X-ray image
im_name = "metacarpals.png"

# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org = io.imread(in_dir + im_name)

print(im_org.shape)

print(im_org.dtype)

#io.imshow(im_org)
#plt.title('Metacarpal image')
#io.show()

#io.imshow(im_org, cmap="jet")
#plt.title('Metacarpal image (with colormap)')
#io.show()

min_pix = print(np.amin(im_org))
max_pix = print(np.amax(im_org))

#io.imshow(im_org, vmin=min_pix, vmax=max_pix)
#plt.title('Metacarpal image (with colormap)')
#io.show()

#plt.hist(im_org.ravel(), bins=256)
#plt.title('Image histogram')
#io.show()

h = plt.hist(im_org.ravel(), bins=256)
bin_no = 100
count = h[0][bin_no]
print(f"There are {count} pixel values in bin {bin_no}")

bin_left = h[1][bin_no]
bin_right = h[1][bin_no + 1]
print(f"Bin edges: {bin_left} to {bin_right}")

bin_max = np.argmax(h[0])
bin_max_value = max(h[0])
print(f"Bin number {bin_max} has the most common range of intensities with {bin_max_value} pixel values")

r = 110
c = 90
im_val = im_org[r, c]
print(f"The pixel value at (r,c) = ({r}, {c}) is: {im_val}")

im_org[:30] = 0
#io.imshow(im_org)
#io.show()

mask = im_org > 150
#io.imshow(mask)
#io.show()

im_org[mask] = 255
#io.imshow(im_org)
#io.show()

im2_name = "ardeche.jpg"
im2_org = io.imread(in_dir + im2_name)
print(im2_org.shape)
print(im2_org.dtype)

#io.imshow(im2_org)
#plt.title('Ardeche image')
#io.show()

r = 110
c = 90
im_val = im2_org[r, c]
print(f"The pixel value at (r,c) = ({r}, {c}) is: {im_val}")

rows = im2_org.shape[0]
columns = im2_org.shape[1]
r_2 = int(rows / 2)

im2_org[0:r_2] = [0, 255, 0]
#io.imshow(im2_org)
#io.show()

im3_name = "test.jpg"
im3_org = io.imread(in_dir + im3_name)
print(im3_org.shape)
print(im3_org.dtype)

#io.imshow(im3_org)
#io.show()

image_rescaled = rescale(im3_org, 0.25, anti_aliasing=True, channel_axis=2)
#io.imshow(image_rescaled)
#io.show()
print(image_rescaled.shape)
print(image_rescaled.dtype)

im_val = image_rescaled[r, c]
print(f"The pixel value at (r,c) = ({r}, {c}) is: {im_val}")

column = (400 / im3_org.shape[1]) * im3_org.shape[0]

image_resized = resize(im3_org, (column, 400), anti_aliasing=True)
#io.imshow(image_resized)
#io.show()
print(image_resized.shape)

im3_gray = color.rgb2gray(im3_org)
im_byte = img_as_ubyte(im3_gray)

#plt.hist(im3_gray.ravel(), bins=256)
#plt.title('Image histogram')
#io.show()

im4_org = io.imread(in_dir + "DTUSign1.jpg")

r_comp = im4_org[:, :, 0]
#io.imshow(r_comp)
#plt.title('DTU sign image (Red)')
#io.show()

#im4_org[1500:1700, 2200:2800, :] = 0
#im4_org[1500:1700, 2200:2800, 2] = 255
#io.imshow(im4_org)
#io.show()

#io.imsave('ex1-IntroductionToImageAnalysis/data/DTUSign1-marked.jpg', im4_org)
#io.imsave('ex1-IntroductionToImageAnalysis/data/DTUSign1-marked.png', im4_org)


io.imshow(im_org)
io.show()

def highlight_pixels_above(image, threshold):
    # Convert the grayscale image to RGB
    rgb_image = color.gray2rgb(image)

    # Create a mask for pixels above the threshold
    mask = image > threshold

    # Create a blue version of the image
    blue_image = np.zeros_like(rgb_image)
    blue_image[..., 2] = 255

    # Apply the mask to the original image and the blue image
    result = np.where(mask[..., None], blue_image, rgb_image)

    return result

highlighted_image = highlight_pixels_above(im_org, 150)
#io.imshow(highlighted_image)
#io.show()

p = profile_line(im_org, (342, 77), (320, 160))
#plt.plot(p)
#plt.ylabel('Intensity')
#plt.xlabel('Distance along line')
#plt.show()

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

im_name = "1-442.dcm"
ds = dicom.dcmread(in_dir + im_name)
print(ds)

im = ds.pixel_array
io.imshow(im, vmin=-1000, vmax=1000, cmap='gray')
io.show()
