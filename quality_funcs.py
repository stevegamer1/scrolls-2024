"""Функции оценки качества и тесты для них."""

import numpy as np
import scipy.signal as signal
import scipy.spatial.distance as distance
import scipy.spatial as spatial


def calc_IoU(arr1: np.ndarray, arr2: np.ndarray):
    """Вычислить intersection over union двух булевых массивов."""
    assert(arr1.dtype == bool)
    assert(arr2.dtype == bool)
    assert(arr1.shape == arr2.shape)
    intersection = arr1 * arr2
    union = arr1 + arr2
    return intersection.sum() / float(union.sum())


def calc_relative_area_diff(arr1: np.ndarray, arr2: np.ndarray):
    """Вычислить relative area difference двух булевых массивов. Функция асимметрична. arr2 обычно ground truth."""
    assert(arr1.dtype == bool)
    assert(arr2.dtype == bool)
    assert(arr1.shape == arr2.shape)
    area1 = float(arr1.sum())
    area2 = float(arr2.sum())
    return (area1 - area2) / area2


def get_surface_cKDtree(image: np.ndarray):
    """Получить координаты пикселей, образующих границы (8-соседство) белых регионов на булевом изображении.
    Возвращает массив с координатами и scipy.spatial.cKDtree, построенное на нём.
    
    |  image - изображение, на котором надо найти поверхностные пиксели.
    """
    assert(image.dtype == bool)
    image8bit = np.array(image, np.int8)
    surface_kernel = np.array([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]], np.int8)
    surface = np.column_stack(np.nonzero(np.logical_and(image8bit, signal.convolve2d(1 - image8bit, surface_kernel, mode='same'))))
    return surface, spatial.cKDTree(surface)


def calc_ASD(arr1_surface: np.ndarray, arr1_kdtree: spatial.cKDTree, arr2_surface: np.ndarray,  arr2_kdtree: spatial.cKDTree):
    """Вычислить Average Symmetric Distance. Если оно больше, то массивы более 'разные'.
    
    |  arr1_surface, arr2_surface - двумерные массивы. По оси 0 - номер точки, по оси 1 - координаты точки.
    |  arr1_kdtree, arr2_kdtree - scipy.spatial.cKDtree, построенные на arr1_surface и arr2_surface.
    """
    arr1_to_arr2 = np.sum(arr1_kdtree.query(arr2_surface, 1)[0])  # counting starts from 1
    arr2_to_arr1 = np.sum(arr2_kdtree.query(arr1_surface, 1)[0])  # counting starts from 1
    result = (arr1_to_arr2 + arr2_to_arr1) / (arr1_surface.shape[0] + arr2_surface.shape[0])
    return result


def calc_RMSD(arr1_surface: np.ndarray, arr1_kdtree: spatial.cKDTree, arr2_surface: np.ndarray,  arr2_kdtree: spatial.cKDTree):
    """Вычислить Root Mean Symmetric Distance. Если оно больше, то массивы более 'разные'.
    Более чувствительна к выбросам, чем ASD.

    |  arr1_surface, arr2_surface - двумерные массивы. По оси 0 - номер точки, по оси 1 - координаты точки.
    |  arr1_kdtree, arr2_kdtree - scipy.spatial.cKDtree, построенные на arr1_surface и arr2_surface.
    """
    arr1_to_arr2 = arr1_kdtree.query(arr2_surface, 1)[0]
    arr1_to_arr2 = np.dot(arr1_to_arr2, arr1_to_arr2)
    arr2_to_arr1 = arr2_kdtree.query(arr1_surface, 1)[0]
    arr2_to_arr1 = np.dot(arr2_to_arr1, arr2_to_arr1)
    result = np.sqrt((arr1_to_arr2 + arr2_to_arr1) / (arr1_surface.shape[0] + arr2_surface.shape[0]))
    return result


def calc_Hausdorff(arr1_surface: np.ndarray, arr1_kdtree: spatial.cKDTree, arr2_surface: np.ndarray,  arr2_kdtree: spatial.cKDTree):
    """Хаусдорфово расстояние. Maximum Symmetric Distance.
    Если оно больше, то массивы более 'разные'.

    |  arr1_surface, arr2_surface - двумерные массивы. По оси 0 - номер точки, по оси 1 - координаты точки.
    |  arr1_kdtree, arr2_kdtree - scipy.spatial.cKDtree, построенные на arr1_surface и arr2_surface.
    """
    arr1_to_arr2 = np.max(arr1_kdtree.query(arr2_surface, 1)[0])
    arr2_to_arr1 = np.max(arr2_kdtree.query(arr1_surface, 1)[0])
    result = max(arr1_to_arr2, arr2_to_arr1)
    return result


def calc_Hausdorff95(arr1_surface: np.ndarray, arr1_kdtree: spatial.cKDTree, arr2_surface: np.ndarray,  arr2_kdtree: spatial.cKDTree):
    """Вычислить почти Хаусдорфово расстояние, но вместо максимума берётся 95-й перцентиль.
    Менее чувствительна к выбросам, чем Хаусдорфово расстояние.

    |  arr1_surface, arr2_surface - двумерные массивы. По оси 0 - номер точки, по оси 1 - координаты точки.
    |  arr1_kdtree, arr2_kdtree - scipy.spatial.cKDtree, построенные на arr1_surface и arr2_surface.
    """
    arr1_to_arr2 = np.percentile(arr2_kdtree.query(arr1_surface, 1)[0], 95)
    arr2_to_arr1 = np.percentile(arr2_kdtree.query(arr1_surface, 1)[0], 95)
    result = max(arr1_to_arr2, arr2_to_arr1)
    return result


def quality_functions_tests():
    """Проверка правильной работы функций оценки качества."""
    example_arr1 = np.array([[1, 1, 1, 0, 0],
                             [1, 1, 1, 0, 0],
                             [1, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]], bool)
    example_arr2 = np.array([[0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 1],
                             [0, 0, 1, 1, 1],
                             [0, 0, 1, 1, 1],
                             [0, 0, 1, 1, 1]], bool)

    example_arr1_surf, example_arr1_kdtree = get_surface_cKDtree(example_arr1)
    example_arr2_surf, example_arr2_kdtree = get_surface_cKDtree(example_arr2)
    assert(abs(calc_ASD(example_arr1_surf, example_arr1_kdtree, example_arr2_surf, example_arr2_kdtree) - 1.128564) < 0.0001)
    assert(calc_ASD(example_arr1_surf, example_arr1_kdtree, example_arr1_surf, example_arr1_kdtree) == 0)
    assert(calc_RMSD(example_arr1_surf, example_arr1_kdtree, example_arr1_surf, example_arr1_kdtree) == 0)
    assert(calc_Hausdorff(example_arr1_surf, example_arr1_kdtree, example_arr1_surf, example_arr1_kdtree) == 0)
    assert(calc_Hausdorff95(example_arr1_surf, example_arr1_kdtree, example_arr1_surf, example_arr1_kdtree) == 0)


    assert(calc_IoU(np.array([True], bool), np.array([True], bool)) == 1.0)
    assert(calc_IoU(np.array([True], bool), np.array([False], bool)) == 0.0)
    assert(calc_IoU(np.array([False], bool), np.array([True], bool)) == 0.0)
    assert(calc_IoU(np.array([False, True], bool), np.array([True, True], bool)) == 0.5)
    assert(calc_IoU(np.array([False, True, True], bool), np.array([True, True, False], bool)) == (1.0 / 3.0))


    assert(calc_relative_area_diff(np.array([True], bool), np.array([True], bool)) == 0.0)
    assert(calc_relative_area_diff(np.array([False], bool), np.array([True], bool)) == -1.0)
    assert(calc_relative_area_diff(np.array([True, True], bool), np.array([True, False], bool)) == 1.0)


quality_functions_tests()
