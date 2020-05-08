import numpy as np
from collections import namedtuple
from scipy import ndimage
Point = namedtuple('Point', 'x y')

array = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]])
# print(array)
# index,uniques = np.unique(array,return_index=True)
# print(index)
# print(uniques)

blobs = array==1
print(blobs)
labels, nlabels = ndimage.label(blobs)

print(labels)
print(nlabels)