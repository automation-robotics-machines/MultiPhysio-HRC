import os
import json
import pandas as pd
import numpy as np
from typing import Union, List, Set
from functools import wraps
import time


def flatten_nested_list(lst: List) -> Set:
    return set([tuple(flatten_nested_list(x)) if isinstance(x, list) else x for x in lst])


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r took: %2.4f sec' % (f.__name__, te - ts))
        return result

    return wrap


def save_data(
    data: pd.DataFrame,
    filename: str
) -> None:
    data.to_csv(filename)


def video_to_numpy(video_cap):
    frames = []
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break
        frames.append(frame)

    video_cap.release()
    video_array = np.array(frames)
    return video_array


def load_config(file_path: str) -> dict:
    with open(file_path, 'r') as config_file:
        return json.load(config_file)


def import_filenames(
    directory_path: str
) -> tuple[List[str], List[str]]:
    """
    Import all file and folder names of a directory
    """
    filename_list = []
    dir_list = []
    for root, dirs, files in os.walk(directory_path, topdown=False):
        filename_list = files
        dir_list = dirs
    return filename_list, dir_list


def create_windows(
    array: np.array,
    sec_window: int,
    sampling_rate: int,
    step: Union[int, None] = None
) -> np.array:

    if array.ndim == 1:
        array = array[:, np.newaxis]
    if step is None:
        step = int(sec_window * sampling_rate)

    window = sec_window * sampling_rate
    n, m = array.shape
    n_windows = int((n - window) // step + 1)
    windowed_array = np.empty((n_windows, window, m))

    for i in range(n_windows):
        start = i * step
        windowed_array[i] = array[start:start + window, :]

    return windowed_array
