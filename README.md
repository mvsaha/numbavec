# numbavec
Bare bones dynamically sized vector numba.jitclass for use in nopython mode.

## Installation
On the command line:
    ```pip install https://github.com/mvsaha/numbavec/zipball/master```

## Usage
Typical imports
```
import numpy as np
import numba as nb
import numbavec
```
### Vector Types
The Following vector types are exposed:

+ VectorUint8
+ VectorUint16
+ VectorUint32
+ VectorUint64
+ VectorInt8
+ VectorInt16
+ VectorInt32
+ VectorInt64
+ VectorFloat32
+ VectorFloat64
+ VectorComplex64
+ VectorComplex128

### Vector Creation
Each vector type (e.g. `VectorInt16`) is a constructor that takes a single value indicating the initial size of the array. These initial values are filled with garbage data.
```python
# Create a vector of floats with 5 garbage values.
v = numbavec.VectorFloat64(5)
```
The values in the vector are put in the `.arr` member variable, which can be referenced inside or outside of nopython mode.
```python
>>>v.arr.size == 5
True
```
Vector creation can be done in nopython mode:
```python
@nb.njit
def build_vec_and_loop(vec_size):
    # Create a vector of a certain size
    vec = numbavec.VectorInt32(vec_size)
    
    # Loop through elements of the vector
    for val in vec.arr:
        pass # Do something here...
    return vec
```
```
>>>vec = build_vec_and_loop(10)
>>>vec
<numba.jitclass.boxing._Vector object at 0x10a341dd0>
```
### Appending to a Vector
Tack values onto the end of a vector using `.append()`. Works in nopython mode
```
@nb.njit
def count_down(vec, start):
    for i in range(start,-1,-1):
        vec.append(i)
```
```
>>>vec = VectorFloat64(0)
>>>vec
<numba.jitclass.boxing._Vector object at 0x10a341dd0>
>>>vec.append(0); vec.append(1); vec.append(2)  # Regular append in the interpreter
>>>count_down(vec, 7)  # Appending in a nopython function
>>>print(vec.arr)
[0., 1., 2., 7., 6., 5., 4., 3., 2., 1., 0.]
```

### Bulk Appending
Add more than one value to an vector at once using the `.extend` method. Extend takes a numpy array and copies the values to the end of the vector:
```
>>>v = VectorUint16(3)
>>>v.arr[:] = 2
>>>v.extend(np.arange(5, 10))
>>>print(v.arr)
[2, 2, 2, 5, 6, 7, 8, 9]
```
If we don't want to *add* to existing values we can use `set_to` and `set_to_copy`. `set_to` simply points `.arr` to an array already exists in memory. Be careful: any changes that you impose on `.arr` will be reflected in the input array.
```python
>>>vec = VectorInt64(0)
>>>x = np.array([6, 7, 8, 9], dtype=np.int64)  # The dtype must match exactly or an error will be raised
>>>vec.set_to(x)
>>>print(vec.arr)
[6, 7, 8, 9]
>>>vec.arr[1:3] = -3
>>>print(vec.arr)
[ 6, -3, -3, 9]
>>>print(x)  # x has been changed!
[ 6, -3, -3, 9]
```

You can prevent these side effects with `set_to_copy`, which will copy the input array first.
```python
>>>vec = VectorInt64(0)
>>>x = np.array([6, 7, 8, 9], dtype=np.int64)  # The dtype must match exactly or an error will be raised
>>>vec.set_to_copy(x)
>>>print(vec.arr)
[6, 7, 8, 9]
>>>vec.arr[1:3] = -3
>>>print(vec.arr)
[ 6, -3, -3, 9]
>>>print(x)  # x has not been changed
[6, 7, 8, 9]
```

### Convenience
`.first` and `.last` give the first and last values of the vector, or raise an `IndexError` if the vector is empty.

The Vectors define the `__array__` method, which allows Vector objects to be passed into numpy functions that expect a 1d numpy array:
```python
>>>vec = VectorFloat32(0)
>>>vec.set_to(np.array([4., 8., 15., 16., 23., 42.]))
>>>np.mean(vec)
15.5
```
