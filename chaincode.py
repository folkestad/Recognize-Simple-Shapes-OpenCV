from __future__ import division
import cv2
import numpy
import math
import matplotlib.pyplot as plt


__author__ = 'Eirik'


def find_start_pos(img):
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] == 255:
                return i, j
    return 0, 0


def get_new_pos(current, direction):
    if direction == 0:
        return current[0] + 0, current[1] - 1
    elif direction == 1:
        return current[0] - 1, current[1] - 1
    elif direction == 2:
        return current[0] - 1, current[1] + 0
    elif direction == 3:
        return current[0] - 1, current[1] + 1
    elif direction == 4:
        return current[0] + 0, current[1] + 1
    elif direction == 5:
        return current[0] + 1, current[1] + 1
    elif direction == 6:
        return current[0] + 1, current[1] + 0
    else:
        return current[0] + 1, current[1] - 1


def get_next_neighbor(current, prev, img):
    neighbors = [
        img[current[0] + 0][current[1] - 1],
        img[current[0] - 1][current[1] - 1],
        img[current[0] - 1][current[1] + 0],
        img[current[0] - 1][current[1] + 1],
        img[current[0] + 0][current[1] + 1],
        img[current[0] + 1][current[1] + 1],
        img[current[0] + 1][current[1] + 0],
        img[current[0] + 1][current[1] - 1]
    ]
    direction = (prev + 1) % 8
    while direction != prev:
        if neighbors[direction] == 255:
            return get_new_pos(current, direction), (direction + 4) % 8
        direction = (direction + 1) % 8
    return None, (direction + 4) % 8


def first_difference(list):
    fd_list = []
    c = list[-1]
    list.insert(0, c)
    for x in range(len(list)-1):
        counter = 0
        pointer = list[x]
        while True:
            if list[x+1] == pointer:
                fd_list.append(counter)
                break
            else:
                counter += 1
                pointer = (pointer + 1) % 8
    return fd_list


def subsample_set(sample_list):
    subsample = []
    sample_counter = 0
    for sample in sample_list:
        if (sample_counter % 50) == 0:
            subsample.append(sample)
        sample_counter += 1
    return subsample


def major_axis(points):
    max_diam = None
    for p1 in points:
        for p2 in points:
            if max_diam is None:
                max_diam = p1, p2
            elif dist(max_diam[0], max_diam[1]) < dist(p1, p2):
                max_diam = p1, p2
    return max_diam


def minor_axis(points, major_axis):
    max_diam = None
    desired_slope = des_slope(major_axis)
    error = 0.00001
    for p1 in points:
        for p2 in points:
            if p1 is not major_axis[0] and p1 is not major_axis[1] and p2 is not major_axis[1] and p2 is not major_axis[0]:
                if desired_slope + error >= slope(p1, p2) >= desired_slope - error:
                    if max_diam is None:
                        max_diam = p1, p2
                    elif dist(max_diam[0], max_diam[1]) < dist(p1, p2):
                        max_diam = p1, p2
    print slope(major_axis[0], major_axis[1])
    print slope(max_diam[0], max_diam[1])
    return max_diam


def dist(p1, p2):
    return math.sqrt(math.pow(p2[1] - p1[1], 2) + math.pow(p2[0] - p1[0], 2))


def des_slope(axis):
    #print axis
    if slope(axis[0], axis[1]) == 0:
        return 0.0000001
    else:
        return -1/(slope(axis[0], axis[1]))


def slope(p1, p2):
    max_x = p1
    min_x = p2
    if p1[1] < p2[1]:
        max_x = p2
        min_x = p1
    if (max_x[1] - min_x[1]) == 0:
        return 0
    else:
        return (max_x[0] - min_x[0])/(max_x[1] - min_x[1])


def boundary_and_chain(img):
    """
    :rtype : tuple
    """
    chain = []
    boundary = []
    prev = 0
    current = find_start_pos(binary_img)
    start = current
    boundary.append(current)
    start_counter = 0

    while start_counter < 2:
        if start == current:
            start_counter += 1
        current, prev = get_next_neighbor(current, prev, img)
        if current is None:
            break
        boundary.append(current)
        print len(boundary)
        chain.append((prev + 4) % 8)
    return boundary, chain


image = cv2.imread('images/ellipse.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
(thresh, im_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
thresh = 127
binary_img = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]

boundary_points, chain_code = boundary_and_chain(binary_img)

boundary_img = numpy.zeros(image.shape, numpy.uint8)
for i in boundary_points:
    boundary_img[i[0]][i[1]] = 255

subsample = subsample_set(boundary_points)
subsample_img = numpy.zeros(image.shape, numpy.uint8)
for i in subsample:
    subsample_img[i[0]][i[1]] = 255

print chain_code
print first_difference(chain_code)
print subsample
major = major_axis(boundary_points)
minor = minor_axis(boundary_points, major)
print major
print minor
#plt.plot([major[0][1], major[1][1], minor[0][1], minor[1][1]], [major[0][0], major[1][0], minor[0][0], minor[1][0]])
#plt.axis([0, 400, 0, 400])
#plt.show()

cv2.line(boundary_img, (major[1][1], major[1][0]), (major[0][1], major[0][0]), (255, 0, 0), 1)
cv2.line(boundary_img, (minor[1][1], minor[1][0]), (minor[0][1], minor[0][0]), (255, 0, 0), 1)


# creating bounding box
temp = []
for i in boundary_points:
    temp.append((i[1], i[0]))
numpy_boundary = numpy.array(temp)

rect = cv2.minAreaRect(numpy_boundary)
box = cv2.cv.BoxPoints(rect)
box = numpy.int0(box)
cv2.drawContours(image, [box], 0, (255, 0, 0), 1)

cv2.circle(boundary_img, (major[1][1], major[1][0]), 3, (255,0,255), -1)
cv2.circle(boundary_img, (major[0][1], major[0][0]), 3, (255,0,255), -1)
cv2.circle(boundary_img, (minor[1][1], minor[1][0]), 3, (255,0,255), -1)
cv2.circle(boundary_img, (minor[0][1], minor[0][0]), 3, (255,0,255), -1)

cv2.imshow('original', image)
cv2.imshow('boundary', boundary_img)
cv2.imshow('subsample', subsample_img)
cv2.waitKey(0)
cv2.destroyAllWindows()




