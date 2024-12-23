"""Алгоритмы бинаризации свитков."""

import heapq
import tifffile
import numpy as np
import cv2 as cv
import cv2.ximgproc as ximgproc
import config


class Point:
    """Информация о точке на плоскости. Заменитель 2-тьюплов.
    Предпочтителен там, где используется, потому что поля имеют имена.
    """

    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y
    
    def __str__(self):
        return "Point(" + str(self.x) + ", " + str(self.y) +")"

    def __repr__(self):
        return "Point(" + str(self.x) + ", " + str(self.y) +")"


def smooth1(image: np.ndarray,
            denoise_threshold: float|None = None,
            aftergauss_threshold: float|None = None,
            gauss_sigma: float|None = None,
            erosion_size: int|None = None):
    """Грубо бинаризировать чёрно-белое изображение сечения свитка.
    Основа более продвинутых алгоритмов.
    В модуле config содержатся хорошие значения параметров. Если какой-то параметр указан как None,
    то значение берётся из модуля config.
    
    |  denoise_threshold - ниже какого порога следует отсечь значения в начале, чтобы удалить шум.
    |  gauss_sigma - параметр сигма для фильтра Гаусса, применяемого для сглаживания. Размер окна вычисляется по этой сигме.
    |  aftergauss_threshold - по какому порогу бинаризировать изображение после применения фильтра Гаусса.
    |  erosion_size - какого размера взять фильтр эрозии в конце.
    """

    if denoise_threshold is None:
        denoise_threshold = config.smooth1_configs['denoise_threshold']
    
    if aftergauss_threshold is None:
        aftergauss_threshold = config.smooth1_configs['aftergauss_threshold']

    if gauss_sigma is None:
        gauss_sigma = config.smooth1_configs['gauss_sigma']
    
    if erosion_size is None:
        erosion_size = config.smooth1_configs['erosion_size']

    # cv2 принимает только эти 3 значения.
    assert(gauss_sigma == 3.0 or gauss_sigma == 5.0 or gauss_sigma == 7.0)

    copy = image
    copy = (copy > denoise_threshold) * copy
    copy = cv.GaussianBlur(copy, (int(gauss_sigma * 3.0), int(gauss_sigma * 3.0)), sigmaX=gauss_sigma, sigmaY=gauss_sigma)
    copy = (copy > aftergauss_threshold) * np.ones_like(copy)
    copy = cv.erode(copy, cv.getStructuringElement(cv.MORPH_ELLIPSE, (erosion_size, erosion_size)))
    return copy


def point_distance(p1: Point, p2: Point):
    """Евклидово расстояние между двумя точками на плоскости."""
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def find_starting_point(image: np.ndarray):
    """Найти какую-нибудь белую точку на чёрно-белом изображении."""
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            if image[y, x] > 0:
                return Point(x, y)
    raise Exception("Failed to find a single positive value in the image.")


def build_broken_line(image: np.ndarray, start: Point, distance_threshold: int):
    """Построить ломаную, проходящую вдоль линии, нарисованной на бинарном изображении.
    Ожидается, что на изображении линия присутствует, и что она не касается краёв изображения.

    |  image - 2d массив, изображение тонкой (~3 пикселя - идеальная толщина) белой линии без разрывов и самопересечений.
    |  start - Point, семя алгоритма. Построение ломаной начинается в этой точке.
    |  distance_threshold - расстояние, которое выдерживается между вершинами ломаной.
    """

    visited = 0
    remember = 2  # Эта клетка уже добавлена в список, больше добавлять её не надо.

    class Neighbor:
        def __init__(self, dist: float, point: Point):
            self.dist = dist
            self.point = point

        def __lt__(self, other):
            return self.dist < other.dist

    def build_broken_line_recursive(image: np.ndarray, start: Point, recursion_depth: int):
        unvisited_neighbors = [Neighbor(0, start)]
        result = [unvisited_neighbors[0].point]
        points_added_to_result = 0
        points_visited = 0

        while not len(unvisited_neighbors) == 0:        
            neighb = heapq.heappop(unvisited_neighbors)

            if image[neighb.point.y, neighb.point.x] != visited:
                for offset in [Point(1, 0), Point(0, 1), Point(-1, 0), Point(0, -1)]:
                    pos = (neighb.point.y + offset.y, neighb.point.x + offset.x)

                    if image[pos] != visited:
                        point = Point(pos[1], pos[0])
                        dist = point_distance(start, point)
                        if neighb.dist > distance_threshold:
                            if points_added_to_result == 0:
                                result = result + [neighb.point] + build_broken_line_recursive(image, neighb.point, recursion_depth+1)
                                points_added_to_result += 1
                            elif points_added_to_result == 1:
                                result = build_broken_line_recursive(image, neighb.point, recursion_depth+1)[::-1] + [neighb.point] + result
                                points_added_to_result += 1
                            else:
                                # Скорее всего самопересечение.
                                raise Exception("Found third point", neighb.point, "starting from", start)
                        elif image[pos] < remember + recursion_depth:
                            image[pos] = remember + recursion_depth
                            heapq.heappush(unvisited_neighbors, Neighbor(dist, point))

                image[neighb.point.y, neighb.point.x] = visited
                points_visited += 1

        return result
    
    return build_broken_line_recursive(image, start, 0)


def get_line(data: np.ndarray, distance_threshold: int, should_make_copy_with_line: bool = False, **smooth1_args):
    """Построить линию, описывающую форму сечения свитка, запечатлённого на изображении.
    
    |  data - 2d массив, сырое изображение сечения свитка.
    |  distance_threshold - расстояние, которое выдерживается между вершинами ломаной.
    |  should_make_copy_with_line - bool, если True то вторым аргументом возвращается изображение с нарисованной на нём результирующей линией.
    |  smooth1_args - какие параметры передать при вызове smooth1. Смотрите описание smooth1.
    """
    if should_make_copy_with_line:
        with_line = np.zeros_like(data)
    else:
        with_line = None

    data = smooth1(data, **smooth1_args)
    data = ximgproc.thinning((data > 0.0) * np.ones_like(data, np.uint8) * 255)
    data = cv.dilate(data, cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2)))
    data = data > 0 * np.ones_like(data)
    starting_point = find_starting_point(data)
    data = np.array(data, np.int32)
    line = build_broken_line(data, starting_point, distance_threshold)

    if should_make_copy_with_line:
        for i in range(len(line))[:-1]:
            cv.line(with_line, (line[i].x, line[i].y), (line[i + 1].x, line[i + 1].y), 1, 12)

    return line, with_line


def get_line_for(path: str, distance_threshold: int, should_make_copy_with_line: bool = False, **smooth1_args):
    """Применить алгоритм get_line к изображению, записанному в файл.
    
    |  path - путь к файлу изображения.
    |  distance_threshold - расстояние, которое выдерживается между вершинами ломаной.
    |  should_make_copy_with_line - bool, если True то вторым аргументом возвращается изображение с нарисованной на нём результирующей линией.
    |  smooth1_args - какие параметры передать при вызове smooth1. Смотрите описание smooth1.
    """
    data = tifffile.imread(path)
    return get_line(data, distance_threshold, should_make_copy_with_line)
    

def get_scroll_shape_as_mesh(filenames: list|None = None, delta_z: int|None = None, distance_threshold: int|None = None, **smooth1_args):
    """Создать 3d-модель, описывающую форму свитка.
    Возвращает 2 списка - точек и треугольников.
    Список точек - список списков из 3 элементов. 3 элемента - координаты точки в 3d.
    Список треугольников - список списков из 3 элементов. 3 элемента - индексы (начинаются с 0) точек,
    образующих треугольник.

    |  filenames - список строк, пути к tiff-файлам, из которых следует читать слои томографии свитка. Если None, то значение берётся из модуля config.
    |  delta_z - каким считать росстояние между слоями. Если None, то значение берётся из модуля config.
    |  distance_threshold - расстояние, которое выдерживается между вершинами ломаной.
    |  smooth1_args - какие параметры передать при вызове smooth1. Смотрите описание smooth1.
    """

    if filenames is None:
        filenames = config.good_files_names

    if delta_z is None:
        delta_z = config.distance_between_layers
    
    if distance_threshold is None:
        distance_threshold = config.line_building['distance_threshold']

    z = 0
    line1 = None
    number = 0
    line2, _ = get_line_for(filenames[0], distance_threshold)
    points = [[p.x, p.y, 0] for p in line2]
    z += delta_z
    triangles = []

    for name in filenames[1:]:
        number += 1
        line1 = line2
        line2, _ = get_line_for(name, distance_threshold)
        if point_distance(line2[0], line1[0]) > point_distance(line2[-1], line1[0]):
            line2 = line2[::-1]

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

        # Только один из следующих двух циклов будет выполнен.

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


def save_as_obj(points: list, triangles: list, filename: str):
    """Сохранить 3d модель, описанную точками и треугольниками, в формате obj.
    Предполагаемое использование: сохранять результат работы get_scroll_shape_as_mesh.

    |  points - список списков из 3 элементов. 3 элемента - координаты точки в 3d.
    |  triangles - список списков из 3 элементов. 3 элемента - индексы (начинаются с 0) точек, образующих треугольник.
    |  filename - путь к результирующему файлу, например "my mesh.obj"
    """
    with open(filename, "w") as outfile:
        for p in points:
            outfile.write("v {} {} {}\n".format(p[0], p[1], p[2]))
        for t in triangles:
            outfile.write("f {} {} {}\n".format(t[0] + 1, t[1] + 1, t[2] + 1))


def delete_all_but_biggest_region(bw_image: np.ndarray, connectivity=4):
    """Заполнить все регионы, состоящие из числа 255, кроме самого большого, числом 0.
    Предполагается, что такой регион один.
    
    |  bw_image - 2d массив, чёрно-белое (0 и 255) восьмибитное изображение.
    """
    count, labeled, stats, centroids = cv.connectedComponentsWithStats(bw_image, connectivity=connectivity)
    max_area = 0
    for i in range(count)[1:]:
        max_area = max(max_area, stats[i][4])
    max_label = None
    for i in range(count)[1:]:
        area = stats[i][4]
        if area == max_area:
            max_label = i
    
    result = np.array(np.isin(labeled, [max_label]), np.uint8) * 255
    return result


def fill_small_regions(bw_image: np.ndarray, min_area: int, connectivity=4):
    """Заполнить все регионы, состоящие из числа 255, меньше по площади чем min_area, числом 0.
    
    |  bw_image - 2d массив, чёрно-белое (0 и 255) восьмибитное изображение.
    """
    count, labeled, stats, centroids = cv.connectedComponentsWithStats(bw_image, connectivity=connectivity)
    big_labels = set()
    for i in range(count)[1:]:
        area = stats[i][4]
        if area >= min_area:
            big_labels.add(i)
    result = np.array(np.isin(labeled, list(big_labels)), np.uint8) * 255
    return result


def smooth_from_canny(image_after_canny: np.ndarray, dilate1_size: int = 5, noise_area_threshold: int = 200,
                      inside_area_threshold: int = 20000, dilate2_size: int = 4):
    """Получить сглаженную бинаризацию, используя результат работы фильтра Кэнни.
    
    |  image_after_canny - 2d массив, результат применения фильтра Кэнни к исходному изображению.
    |  dilate1_size - какого размера брать первое раздутие, до первого утоньшения.
    |  noise_area_threshold - какая площадь считается максимальной площадью кусочков шума, которые будут удалены после первого утоньшения.
    |  dilate2_size - какого размера брать второе раздутие, после удаления шума.
    |  inside_area_threshold - порог площади внутренности. Если площадь чёрного региона меньше этого порога, то он регион считается частью внутренности сечения свитка и закрашивается в белый цвет.
    """
    image_after_canny = cv.dilate(image_after_canny, cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilate1_size, dilate1_size)))
    image_after_canny = ximgproc.thinning(image_after_canny)

    canny_clean = fill_small_regions(image_after_canny, noise_area_threshold, connectivity=8)

    dilated = cv.dilate(canny_clean, cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilate2_size, dilate2_size)))
    thinned = ximgproc.thinning(dilated)
    
    filled = 255 - fill_small_regions(255 - thinned, inside_area_threshold, connectivity=4)
    result = delete_all_but_biggest_region(filled, connectivity=4)
    return result


def canny_based_segmentation(data: np.ndarray, gauss_sigma: float = 5.0, canny_thresh1: float = 0.1,
                             canny_thresh2: float = 10.0, **smooth_from_canny_args):
    """Получить бинаризацию сечения свитка, запечатлённого на изображении, с помощью алгоритма,
    основанного на фильтре Кэнни.
    
    |  data - 2d массив, сырое (только что из файла) изображение сечения свитка.
    |  gauss_sigma - какой параметр сигма брать для сглаживающего фильтра Гаусса. Размер окна вильтра вычисляется автоматически.
    |  canny_thresh1, canny_thresh2 - пороги для фильтра Кэнни.
    |  smooth_from_canny_args - какие параметры передать при вызове smooth_from_canny. Смотрите описание smooth_from_canny.
    """
    canny_input = data
    canny_input = cv.GaussianBlur(canny_input, (int(gauss_sigma * 3.0), int(gauss_sigma * 3.0)),
                                  sigmaX=gauss_sigma, sigmaY=gauss_sigma)
    canny_input = np.array((canny_input - np.min(canny_input)) / np.max(canny_input) * 255.0, np.uint8)
    canny = cv.Canny(canny_input, canny_thresh1, canny_thresh2)
    smooth_filled = smooth_from_canny(canny, **smooth_from_canny_args)
    return smooth_filled
