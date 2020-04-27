import numpy as np
cimport numpy as np
cimport cython

def time_shit_pixel(np.ndarray[np.uint8_t, ndim=4] frame_history,
                    cython.uint frame_history_pointer,
                    np.ndarray[np.uint_t, ndim=2] time_shit_matrix):

    cdef cython.uint width = frame_history.shape[2]
    cdef cython.uint height = frame_history.shape[1]
    cdef cython.uint channel = 3
    cdef np.ndarray[np.uint8_t, ndim=3] frame

    frame = np.empty((height, width, channel), dtype=np.uint8)
    for r in range(height):
        for c in range(width):
            t_shift = time_shit_matrix[r,c]
            t = frame_history_pointer - t_shift
            frame[r, c] = frame_history[t, r, c]
    return frame
