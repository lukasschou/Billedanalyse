import math

#########################################
# Exercise 1
a = 10
b = 3

angle = math.degrees(math.atan2(b, a))
print(angle)

#########################################
# Exercise 2
def camera_b_distance(f, g):
    """
    camera_b_distance returns the distance (b) where the CCD should be placed
    when the object distance (g) and the focal length (f) are given
    :param f: Focal length
    :param g: Object distance
    :return: b, the distance where the CCD should be placed
    """
    return 1 / (1 / f - 1 / g)

print(camera_b_distance(15/1000, 15))

#########################################
# Exercise 3
# 1
lens_dist = camera_b_distance(5/1000, 5)
print(lens_dist)

# 2
h = 1.8/(5/0.005)
print(h)

# 3
chip_width = 6.4 
chip_height = 4.8
resolution_width_pixels = 640
resolution_height_pixels = 480
pixel_width = chip_width / resolution_width_pixels
pixel_height = chip_height / resolution_height_pixels

print(f"Size of a single pixel: {pixel_width}mm x {pixel_height}mm")

# 4
h_p = h / pixel_height*1000
print(f"He is: {h_p} pixels high")

# 5
fov_rad_h = 2 * math.atan((chip_width / 2) / 5)
print(f"The horizontal field of view is {math.degrees(fov_rad_h)} degrees")

# 6
fov_rad_v = 2 * math.atan((chip_height / 2) / 5)
print(f"The vertical field of view is {math.degrees(fov_rad_v)} degrees")
