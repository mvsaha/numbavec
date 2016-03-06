import numpy as np
import numba


def Vector(numba_type):
    """Generate an instance of a dynamically sized vector numba jitclass."""

    class _Vector:
        """Dynamically sized arrays in nopython=True mode."""

        def __init__(self, n):
            """Initialize with space to hold 1 value."""
            self.n = n
            self.m = n
            self.full_arr = np.empty(self.n, dtype=numba_type)


        @property
        def size(self):
            """The number of valid valued."""
            return self.n

        @property
        def arr(self):
            """Return the subarray."""
            return self.full_arr[:self.n]


        def append(self, val):
            """Add a value to the end of the Vector, expanding it if necessary."""
            if self.n == self.m:
                self._expand()
            self.full_arr[self.n] = val
            self.n += 1


        def reserve(self, n):
            """Reserve a number of elements.
            This funciton ensures no resize overhead when appending values 0 to n-1."""
            if n > self.m:  # Only change size if we are
                temp = np.zeros(int(n), dtype=numba_type)
                temp[:self.n] = self.arr
                self.full_arr = temp
                self.m = n


        def consolidate(self):
            if self.n < self.m:
                self.full_arr = self.arr
                self.m = self.n


        def __array__(self):
            """Array inteface for Numpy compatibility."""
            return self.full_arr[:self.n]


        def extend(self):
            pass


        def _expand(self):
            """Internal function that handles the resizing of the array."""
            self.m = int(self.m * 2) + 1
            temp = np.empty(self.m, dtype=numba_type)
            temp[:self.n] = self.full_arr[:self.n]
            self.full_arr = temp

    if numba_type not in Vector._saved_type:
        spec = [("n", numba.int32),
                ("m", numba.int32),
                ("full_arr", numba_type[:])]
        Vector._saved_type[numba_type] = numba.jitclass(spec)(_Vector)

    return Vector._saved_type[numba_type]


Vector._saved_type = dict()


VectorUint8 = Vector(numba.uint8)
VectorUint16 = Vector(numba.uint16)
VectorUint32 = Vector(numba.uint32)
VectorUint64 = Vector(numba.uint64)

VectorInt8 = Vector(numba.int8)
VectorInt16 = Vector(numba.int16)
VectorInt32 = Vector(numba.int32)
VectorInt64 = Vector(numba.int64)

VectorFloat32 = Vector(numba.float32)
VectorFloat64 = Vector(numba.float64)

VectorComplex64 = Vector(numba.complex64)
VectorComplex128 = Vector(numba.complex128)
