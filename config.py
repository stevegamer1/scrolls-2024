# writing this in python was the simplest way
line_building = dict(
    distance_threshold = 50,
)

distance_between_layers = 100
filenames = ["../scroll001.rec_1150/SPTV{:04d}.tif".format(i) for i in range(0, 2687)]
good_files_names = filenames[28:2632:distance_between_layers]

smooth1_configs = dict(
    denoise_threshold = 0.00075,
    aftergauss_threshold = 0.0006,
    gauss_sigma = 3.0,
    erosion_size = 5
)