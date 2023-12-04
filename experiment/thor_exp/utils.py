import numpy as np


def get_cmat(fov=90, resolution=(64, 64)):
    width, height = resolution 
    assert width == height # in thor we use square images
    f = (1./np.tan(np.deg2rad(fov)/2)) * width / 2.0 # focal length
    cmat = np.eye(4)
    cmat[0, 0] = f
    cmat[1, 1] = f
    cmat[0, 2] = (width - 1) / 2.0
    cmat[1, 2] = (height - 1) / 2.0
    return cmat


