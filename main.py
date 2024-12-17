import heapq
import scipy.linalg
import tifffile
import numpy as np
import scipy.signal
import cv2 as cv
import cv2.ximgproc as ximgproc
import time
import meshio

import config


class Point:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y
    
    def __str__(self):
        return "Point(" + str(self.x) + ", " + str(self.y) +")"

    def __repr__(self):
        return "Point(" + str(self.x) + ", " + str(self.y) +")"


# 28 - 2632 seem to have good sections
#filename = "../scroll001.rec_1150/SPTV1168.tif"
#data = tifffile.imread(files=filename)


def smooth1(image, denoise_threshold = 0.00075, aftergauss_threshold = 0.0005, gauss_sigma = 5.0, erosion_size = 5):
    copy = image
    copy = (copy > denoise_threshold) * copy
    copy = cv.GaussianBlur(copy, (int(gauss_sigma * 3.0), int(gauss_sigma * 3.0)), sigmaX=gauss_sigma, sigmaY=gauss_sigma)
    copy = (copy > aftergauss_threshold) * np.ones_like(copy)
    copy = cv.erode(copy, cv.getStructuringElement(cv.MORPH_ELLIPSE, (erosion_size, erosion_size)))
    return copy


# old stuff, tried to find "starting point for the scroll"
def find_starting_point_old(image):
    kernel_size = 40
    kernel = np.reshape(np.tile(np.linspace(0.0, 1.0, kernel_size), kernel_size), (kernel_size, kernel_size))
    kernel = (kernel >= 0.5) * np.ones_like(kernel) * 2.0 - np.ones_like(kernel)
    kernel = kernel / (kernel_size * kernel_size)

    result1 = scipy.signal.convolve2d(image, kernel, mode='full')[:-kernel_size, :-kernel_size]
    kernel = kernel.transpose()
    result2 = scipy.signal.convolve2d(image, kernel, mode='full')[:-kernel_size, :-kernel_size]
    kernel = -kernel
    result3 = scipy.signal.convolve2d(image, kernel, mode='full')[kernel_size:, kernel_size:]
    kernel = kernel.transpose()
    result4 = scipy.signal.convolve2d(image, kernel, mode='full')[kernel_size:, kernel_size:]
    
    result1 = (result1 >= 0.1) * result1
    result2 = (result2 >= 0.1) * result2
    result3 = (result3 >= 0.1) * result3
    result4 = (result4 >= 0.1) * result4

    tifffile.imwrite("convolution_result1.tif", result1)
    tifffile.imwrite("convolution_result2.tif", result2)
    tifffile.imwrite("convolution_result3.tif", result3)
    tifffile.imwrite("convolution_result4.tif", result4)
    result_diag_1 = (result1 * result2 + result3 * result4) * 10.0
    result_diag_2 = (result1 * result3 + result2 * result4) * 10.0
    tifffile.imwrite("result_diag_1.tif", result_diag_1)
    tifffile.imwrite("result_diag_2.tif", result_diag_2)
    result = result_diag_1 * result_diag_2
    tifffile.imwrite("result.tif", result)
    return image


def find_nearest_white(img, target):
    nonzero = cv.findNonZero(img)
    distances = np.sqrt((nonzero[:,:,0] - target[0]) ** 2 + (nonzero[:,:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)
    return (nonzero[nearest_index][0][0], nonzero[nearest_index][0][1])


def find_starting_point(image):
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            if image[y, x] > 0:
                return Point(x, y)
    raise Exception("Uhm, where are the pixels? I haven't found a single positive value in your 'image'.")


unvisited = 1
visited = 0
remember = 2  # do not add such neighbors

class Neighbor:
    def __init__(self, dist: float, point: Point):
        self.dist = dist
        self.point = point
    
    def __lt__(self, other):
        return self.dist < other.dist


def point_distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def build_broken_line(image, start: Point, recursion_depth = 0):
    unvisited_neighbors = [Neighbor(0, start)]
    result = [unvisited_neighbors[0].point]
    points_added_to_result = 0
    points_visited = 0
    
    # 1. while have unvisited "neighbors", take one and visit
    while not len(unvisited_neighbors) == 0:        
        neighb = heapq.heappop(unvisited_neighbors)

        if image[neighb.point.y, neighb.point.x] != visited:
            for offset in [Point(1, 0), Point(0, 1), Point(-1, 0), Point(0, -1)]:
                pos = (neighb.point.y + offset.y, neighb.point.x + offset.x)
                
                if image[pos] != visited:  # assume there's a black border around the white pixels
                    point = Point(pos[1], pos[0])
                    dist = point_distance(start, point)
                    if neighb.dist > config.line_building['distance_threshold']:
                        if points_added_to_result == 0:
                            #print("from", start, "got to", neighb.point)
                            result = result + [neighb.point] + build_broken_line(image, neighb.point, recursion_depth+1)
                            points_added_to_result += 1
                        elif points_added_to_result == 1:
                            #print("from", start, "ALSO got to", neighb.point)
                            result = build_broken_line(image, neighb.point, recursion_depth+1)[::-1] + [neighb.point] + result
                            points_added_to_result += 1
                        else:
                            raise Exception("Found third point", neighb.point, "starting from", start)
                    elif image[pos] < remember + recursion_depth:
                        image[pos] = remember + recursion_depth
                        heapq.heappush(unvisited_neighbors, Neighbor(dist, point))
                
            image[neighb.point.y, neighb.point.x] = visited
            points_visited += 1
    
    #print("starting from", start, "visited", points_visited, "points")
    return result


def get_line(data, calc_with_line = False):
    with_line = np.zeros_like(data)
    data = smooth1(data,
                   denoise_threshold=config.smooth1_configs['denoise_threshold'],
                   aftergauss_threshold=config.smooth1_configs['aftergauss_threshold'],
                   gauss_sigma=config.smooth1_configs['gauss_sigma'],
                   erosion_size=config.smooth1_configs['erosion_size'])
    data = ximgproc.thinning((data > 0.0) * np.ones_like(data, np.uint8) * 255)
    data = cv.dilate(data, cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2)))
    data = data > 0 * np.ones_like(data)
    starting_point = find_starting_point(data)
    data = np.array(data, np.int32)
    line = build_broken_line(data, starting_point)
    if calc_with_line:
        for i in range(len(line))[:-1]:
            cv.line(with_line, (line[i].x, line[i].y), (line[i + 1].x, line[i + 1].y), 1, 12)
    return line, with_line


def get_line_for(path: str, calc_with_line = False):
    data = tifffile.imread(path)
    return get_line(data, calc_with_line)
    

def make_mesh():
    delta_z = config.distance_between_layers
    z = 0
    filenames = config.good_files_names
    #print(filenames)
    line1 = None
    number = 0
    line2, _ = get_line_for(filenames[0])
    points = [[p.x, p.y, 0] for p in line2]
    z += delta_z
    triangles = []
    #print("file", filenames[0], "gives", len(line2), "points")


    for name in filenames[1:]:
        number += 1
        #print("progress: ", number / len(filenames) * 100, "%", sep="")
        line1 = line2
        line2, _ = get_line_for(name)
        if point_distance(line2[0], line1[0]) > point_distance(line2[-1], line1[0]):
            line2 = line2[::-1]
        #print("file", name, "gives", len(line2), "points")
        points.extend([[p.x, p.y, z] for p in line2])
        z += delta_z
        index1 = 0
        index2 = 0
        while index1 < len(line1) - 1 and index2 < len(line2) - 1:
            index1_cur = len(points) - len(line1) - len(line2) + index1
            index2_cur = len(points) - len(line2) + index2

            dist1 = point_distance(line1[index1 + 1], line2[index2])
            dist2 = point_distance(line1[index1], line2[index2 + 1])
            if dist1 < dist2:
                triangles.append([index1_cur, index1_cur + 1, index2_cur])
                index1 += 1
            else:
                triangles.append([index1_cur, index2_cur + 1, index2_cur])
                index2 += 1

        # only one of the following loops will execute

        while index1 < len(line1) - 1:
            index1_cur = len(points) - len(line1) - len(line2) + index1
            index2_cur = len(points) - len(line2) + index2
            triangles.append([index1_cur, index1_cur + 1, index2_cur])
            index1 += 1

        while index2 < len(line2) - 1:
            index1_cur = len(points) - len(line1) - len(line2) + index1
            index2_cur = len(points) - len(line2) + index2
            triangles.append([index1_cur, index2_cur + 1, index2_cur])
            index2 += 1
        
    return points, triangles


if False:
    points, triangles = make_mesh()

    with open("mesh.obj", "w") as outfile:
        for p in points:
            outfile.write("v {} {} {}\n".format(p[0], p[1], p[2]))
        for t in triangles:
            outfile.write("f {} {} {}\n".format(t[0] + 1, t[1] + 1, t[2] + 1))
