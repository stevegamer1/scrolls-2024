"""Примеры предполагаемого использования функций оценки качества и алгоритмов бинаризации."""

import functools
import itertools
import os
import cv2 as cv
import nrrd
import numpy as np
import trimesh
import algorithms as algos
import config
import quality_funcs as quality
import tifffile



def get_scroll1_ground_truth():
    """Получить ground truth для свитка-1 как numpy-массив."""
    ground_truth, header = nrrd.read(os.path.abspath(config.scroll001_ground_truth_path))
    ground_truth = np.swapaxes(ground_truth, 0, 2)
    return ground_truth


def do_quality_functions_experiments(output_filename: str|None, should_print: bool = True):
    """Вычислить функции оценки качества для ground truth и её искусственно искажённых версий для изучения
    свойств функций оценки качества.

    |  output_filename - путь к файлу, в который нужно записать результаты, или None, если не нужно писать результаты в файл.
    |  should_print - True, если нужно выводить результаты в консоль.
    """

    if output_filename is None and not should_print:
        print("Warning: do_quality_functions got both output_filename=None and should_print=False, the function\
 call is ineffectual.")

    ground_truth = get_scroll1_ground_truth()

    chosen_slices = ground_truth[range(0, len(ground_truth), 400)]

    def noise_border(arr: np.ndarray, p: float):
        """Инвертировать случайные p * 100% пикселей, но только среди находящихся 'на краю'."""
        dilated = cv.dilate(np.array(arr, np.uint8), cv.getStructuringElement(cv.MORPH_RECT, (4, 4)))
        eroded = cv.erode(np.array(arr, np.uint8), cv.getStructuringElement(cv.MORPH_RECT, (4, 4)))
        border_mask = (dilated + eroded) % 2  # xor
        rng = np.random.default_rng(42)
        noise = np.logical_and(border_mask, np.array(rng.binomial(1, p, border_mask.shape), bool))
        return np.logical_xor(arr, noise)
    
    def noise_inside(arr: np.ndarray, p: float):
        """Инвертировать случайные p * 100% пикселей, но только среди находящихся глубоко внутри белого региона."""
        eroded = cv.erode(np.array(arr, np.uint8), cv.getStructuringElement(cv.MORPH_RECT, (4, 4)))
        rng = np.random.default_rng(42)
        noise = np.logical_and(eroded, np.array(rng.binomial(1, p, eroded.shape), bool))
        return np.logical_xor(arr, noise)

    def erode_bool(arr, k_size):
        return np.array(cv.erode(np.array(arr, np.uint8), cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))), bool)

    def dilate_bool(arr, k_size):
        return np.array(cv.dilate(np.array(arr, np.uint8), cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))), bool)

    experiments = {
        "Shift-X-1" : functools.partial(np.roll, shift=1, axis=1),
        "Shift-X-10" : functools.partial(np.roll, shift=10, axis=1),
        "Noise 5%, border" : functools.partial(noise_border, p=0.05),
        "Noise 70%, inside" : functools.partial(noise_inside, p=0.70),
        "Erode 3 pixels" : functools.partial(erode_bool, k_size=3),
        "Dilate 3 pixels" : functools.partial(dilate_bool, k_size=3)
    }
    
    results = {}
    for exp_name in experiments.keys():
        results[exp_name] = {"IoUs":[], "RADs":[], "ASDs":[], "RMSDs":[], "Hausdorffs":[], "Hausdorffs-95":[]}
    
    for slice_float in chosen_slices:
        slice = slice_float > 0
        for exp_name, experiment in experiments.items():
            exp_slice = experiment(slice)
            results[exp_name]["IoUs"].append(quality.calc_IoU(slice, exp_slice))
            results[exp_name]["RADs"].append(quality.calc_relative_area_diff(slice, exp_slice))
            slice_surf, slice_kdtree = quality.get_surface_cKDtree(slice)
            exp_slice_surf, exp_slice_kdtree = quality.get_surface_cKDtree(exp_slice)
            results[exp_name]["ASDs"].append(quality.calc_ASD(slice_surf, slice_kdtree, exp_slice_surf, exp_slice_kdtree))
            results[exp_name]["RMSDs"].append(quality.calc_RMSD(slice_surf, slice_kdtree, exp_slice_surf, exp_slice_kdtree))
            results[exp_name]["Hausdorffs"].append(quality.calc_Hausdorff(slice_surf, slice_kdtree, exp_slice_surf, exp_slice_kdtree))
            results[exp_name]["Hausdorffs-95"].append(quality.calc_Hausdorff95(slice_surf, slice_kdtree, exp_slice_surf, exp_slice_kdtree))

    if should_print:
        for exp_name, experiment in experiments.items():
            print("Experiment: " + exp_name)
            for key, values in results[exp_name].items():
                print(key + " mean:", np.mean(values))
            print()

    if output_filename is not None:
        with open(output_filename, 'w') as f:
            for exp_name, experiment in experiments.items():
                f.write("Experiment: " + exp_name + "\n")
                for key, values in results[exp_name].items():
                    f.write(key + ": " + str(values) + "\n")
                f.write("\n")


def assess_algorithms(output_filename: str|None, should_print: bool = True):
    """Вычислить функции оценки качества для каждого алгоритма, результаты вывести на экран и записать в файл.
    
    |  output_filename - путь к файлу, в который нужно записать результаты.
    |  should_print - True, если нужно выводить результаты в консоль.
    """

    if output_filename is None and not should_print:
        print("Warning: assess_algorithms got both output_filename=None and should_print=False, the function\
 call is ineffectual.")

    def smooth1(slice: np.ndarray, slice_index: int):
        return algos.smooth1(slice,
            denoise_threshold=config.smooth1_configs['denoise_threshold'],
            aftergauss_threshold=config.smooth1_configs['aftergauss_threshold'],
            gauss_sigma=config.smooth1_configs['gauss_sigma'],
            erosion_size=config.smooth1_configs['erosion_size'])

    def canny_based(slice: np.ndarray, slice_index: int):
        return algos.canny_based_segmentation(slice)

    mesh_raw = algos.get_scroll_shape_as_mesh()
    mesh = trimesh.Trimesh(vertices=mesh_raw[0], faces=mesh_raw[1])
    def segmentation_from_mesh(slice: np.ndarray, slice_index: int):
        section = mesh.section(plane_origin=[0, 0, slice_index], plane_normal=[0, 0, 1])
        section.explode()
        result = np.zeros(slice.shape, np.uint8)
        for i in range(len(section.vertices)):
            start = np.array(section.vertices[section.entities[i].end_points[0]][:2], np.int32)
            end = np.array(section.vertices[section.entities[i].end_points[1]][:2], np.int32)
            result = cv.line(result, start, end, 255, 13)
        return result

    algorithms = {
        "smooth1" : smooth1,
        "Canny filter-based" : canny_based,
        "make mesh & dilate" : segmentation_from_mesh
    }

    ground_truth = get_scroll1_ground_truth()

    results = {}
    for algo_name, algorithm in algorithms.items():
        results[algo_name] = {"IoUs":[], "RADs":[], "ASDs":[], "RMSDs":[], "Hausdorff's 95":[], "Hausdorff's":[]}

    chosen_slices = range(15, len(ground_truth), 400)
    for slice_i in chosen_slices:
        grtruth_slice = ground_truth[slice_i] > 0
        data_slice = tifffile.imread(config.filenames[slice_i])

        for algo_name, algorithm in algorithms.items():
            segmentation = np.array(algorithm(data_slice, slice_i), bool)
            results[algo_name]["IoUs"].append(quality.calc_IoU(grtruth_slice, segmentation))
            results[algo_name]["RADs"].append(quality.calc_relative_area_diff(grtruth_slice, segmentation))
            grtruth_surf, grtruth_kdtree = quality.get_surface_cKDtree(grtruth_slice)
            segm_surf, segm_kdtree = quality.get_surface_cKDtree(segmentation)
            results[algo_name]["ASDs"].append(quality.calc_ASD(grtruth_surf, grtruth_kdtree, segm_surf, segm_kdtree))
            results[algo_name]["RMSDs"].append(quality.calc_RMSD(grtruth_surf, grtruth_kdtree, segm_surf, segm_kdtree))
            results[algo_name]["Hausdorff's 95"].append(quality.calc_Hausdorff95(grtruth_surf, grtruth_kdtree, segm_surf, segm_kdtree))
            results[algo_name]["Hausdorff's"].append(quality.calc_Hausdorff(grtruth_surf, grtruth_kdtree, segm_surf, segm_kdtree))

    if should_print:
        for algo_name, algorithm in algorithms.items():
            print("Algorithm: " + algo_name)
            for func_name, result in results[algo_name].items():
                print(func_name + " mean:", np.mean(result))
            print()
    
    if output_filename is not None:
        with open(output_filename, 'w') as f:
            for algo_name, algorithm in algorithms.items():
                f.write("Algorithm: " + algo_name + "\n")
                for func_name, result in results[algo_name].items():
                    f.write(func_name + ": " + str(result) + "\n")
                f.write("\n")


if __name__ == "__main__":
    assess_algorithms("algorithms results.txt")
