import numpy as np

class tools:
    def __init__(self):
        return

    def setdiffn(self, array1, array2):
        """Version of Matlab/R's 2D SETDIFF for ND arrays only. Reduces ND array down to 2D and returns 2D array (last two dims of ND arrays)
        (c) GNU GPL2, 2016, T.Kam """
        ncols = array1.shape[-1]
        a = array1.ravel()
        b = array2.ravel()
        # Compare element-wise for similarity
        c = np.in1d(a,b).reshape(array1.shape).sum(axis=-1,keepdims=True)
        # index of array1 rows NOT contained in array 2
        d = np.nonzero(c.ravel() != ncols)
        # map index into 2D subset of 2D-reshaped version of array1:
        return a.reshape(-1,ncols)[d], d

    def setintersectn(self,array1, array2):
        """Simpler version of Matlab/R's INTERSECT for ND arrays. Reduces ND array down to 2D and returns 2D array (last two dims of ND arrays).
        (c) GNU GPL2, 2016, T.Kam """
        ncols = array1.shape[-1]
        a = array1.ravel()
        b = array2.ravel()
        # Compare element-wise for similarity
        c = np.in1d(a,b).reshape(array1.shape).sum(axis=-1,keepdims=True)
        # index of array1 rows contained in array 2
        d = np.nonzero(c.ravel() == ncols)
        # map index into 2D subset of 2D-reshaped version of array1:
        return a.reshape(-1,ncols)[d], d
