import numpy as np

# Create two NumPy arrays
array1 = np.array([1.43590843590843, 2, 3])
array2 = np.array([4.42342, 5, 6])

# Stack arrays as two columns
stacked_arrays = np.column_stack((array1, array2))
print(stacked_arrays)

np.savetxt("try.txt", stacked_arrays, delimiter=' ')