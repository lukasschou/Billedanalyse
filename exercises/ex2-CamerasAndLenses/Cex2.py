import math
import numpy as np

y = 3.0
x = 10.0
angle = math.atan2(y, x)

#print(angle)
#print(np.rad2deg(angle))
#print(math.degrees(angle))
#angle_degrees = 180.0 / math.pi * angle_radians

# function to calculate the distance between focal length and object distance
def camera_b_distance(f, g):
    """
    camera_b_distance returns the distance (b) where the CCD should be placed
    when the object distance (g) and the focal length (f) are given
    :param f: Focal length
    :param g: Object distance
    :return: b, the distance where the CCD should be placed
    """
    b = 1 / ((1 / f) - (1 / g))
    return b

f = 15/1000 # focal length in meters
g = 0.1     # object distance in meters
#print(camera_b_distance(f, g))

# A focused image of Thomas is formed inside the camera. At which distance from the lens?
f = 5/1000 # focal length in meters
g = 5     # object distance in meters
print(camera_b_distance(f, g))

#How tall (in mm) will Thomas be on the CCD-chip?
#Thomas is 1.8 mm tall
thomas_height = 1.8
thomas_camera_height = thomas_height/(g/f)*1000 #in mm
print(thomas_camera_height)

#What is the size of a single pixel on the CCD chip? (in mm)?
a = 6.4 # width of the CCD-chip in mm
b = 4.8 # height of the CCD-chip in mm
pixel_width = 640
pixel_height = 480
print(a/pixel_width) #in mm
print(b/pixel_height) #in mm

#How tall (in pixels) will Thomas be on the CCD-chip?
thomas_camera_height_pixels = thomas_camera_height/(b/pixel_height)
print(thomas_camera_height_pixels)
# thomas is 180 pixels tall

#What is the horizontal field-of-view (in degrees)?
ah = 5
bh = (6.4/2)
h_field = math.atan2(bh, ah) *2
print(f"Horizontal field of view {math.degrees(h_field)}")

#What is the vertical field-of-view (in degrees)?
av = 5
bv = (4.8/2)
v_field = math.atan2(bv, av) *2
print(f"Vertical field of view {math.degrees(v_field)}")

