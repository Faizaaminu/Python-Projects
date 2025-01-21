
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

    # Creating python List
python_list = [1,2,3,4,5]

#creating numpy array from tuple
#craeting python tuple
python_tuple = (1,2,3,4,5)
print(type (python_tuple))


    # Checking data types
print('Type:', type (python_list)) # <class 'list'>
    
print(python_list)

two_dimensional_list = [[0,1,2], [3,4,5], [6,7,8]]

print(two_dimensional_list) 

three_dimensional_list = [[[0,1,2], [3,4,5], [6,7,8]]]
print(three_dimensional_list)

# getting the size

numpy_array_from_list = np.array([1, 2, 3, 4, 5])
two_dimensional_list = np.array([[0, 1, 2],[3, 4, 5], [6, 7, 8]])
three_dimensional_list =np.array ([[[0,1,2], [3,4,5], [6,7,8]]])

print('The size:', numpy_array_from_list.size) # 5
print('The size:', two_dimensional_list.size)  # 9
print('The size:', three_dimensional_list.size)  # 9


#check dimension of array
n = np.array([1, 2, 3, 4, 5])
two_dimensional_list = np.array([[0, 1, 2], [3, 4, 5],[6, 7, 8]])
three_dimensional_list =np.array ([[[0,1,2], [3,4,5], [6,7,8]]])
print(n.ndim)
print(two_dimensional_list.ndim )
print(three_dimensional_list.ndim)

#accessing elements
n = np.array([1, 2, 3, 4, 5])
print(n[0])
wo_dimensional_list = np.array([[0, 1, 2], [3, 4, 5],[6, 7, 8]])
print('2nd element on first row:', two_dimensional_list[0,1])


# getting the shape
nums = np.array([1, 2, 3, 4, 5])
print(nums)
print('shape of nums: ', nums.shape)
print(two_dimensional_list)
print('shape of numpy_two_dimensional_list' )
three_by_three_array = np.array([[0, 1, 2],
        [3,4,5],
        [6,7,8]])
print(three_by_three_array.shape)

#slicing numpy array
n = np.array([1, 2, 3, 4, 5])
print(n[-3:-1])
two_dimension_array = np.array([[0, 1, 2], [3, 4, 5],[6, 7, 8]])
first_two_rows_and_columns = two_dimension_array[0:2, 0:2]
print(first_two_rows_and_columns)
hree_dimensional_array =np.array ([[[0,1,2], [3,4,5], [6,7,8]]])
three_by_three_array[::-1][::-1]  # reverse row and column position


#  Creating Numpy(Numerical Python) array from python list

numpy_array_from_list = np.array(python_list)
print(type (numpy_array_from_list))   # <class 'numpy.ndarray'>
print(numpy_array_from_list) # array([1, 2, 3, 4,5])


#  convert array to  list to list().
np_to_list = numpy_array_from_list.tolist()
print(type (np_to_list))
print('one dimensional array:', np_to_list)
print('two dimensional array: ', two_dimensional_list.tolist())
print('three dimensional array:',three_dimensional_list.tolist())



numpy_bool_array = np.array([0, 1, -1, 0, 0], dtype=bool)
print(numpy_bool_array) # array([False,  True, True, False, False]])

  

# Addition
numpy_array_from_list = np.array([1, 2, 3, 4, 5])
print('original array: ', numpy_array_from_list)
ten_plus_original = numpy_array_from_list  + 10
print(ten_plus_original)


##convert int to float
numpy_int_arr = np.array([1,2,3,4], dtype = 'float')
numpy_int_arr

#convert float to int
numpy_int_arr = np.array([1., 2., 3., 4.], dtype = 'int')
numpy_int_arr


# checking for data type
int_lists = [-3, -2, -1, 0, 1, 2,3]
int_array = np.array(int_lists)
float_array = np.array(int_lists, dtype=float)
bool_array =np.array(int_lists, dtype='bool')
str_array=np.array(int_lists, dtype='str')

print(int_array)
print(int_array.dtype)
print(float_array)
print(float_array.dtype)
print(bool_array)
print(bool_array.dtype)
print(str_array)
print(str_array.dtype)



 #Generate random numbers

normal_array = np.random.normal(79, 15, 80)
normal_array

# Gnerate random data
random_data = np.random.randint(0, 100, size=50)  # 50 random integers between 0 and 100
print(random_data)

# Generate data following a normal distribution
normal_data = np.random.normal(loc=50, scale=10, size=100)  # Mean 50, std 10
print(normal_data)
#numpy statistical functions
np_normal_dis = np.random.normal(5, 0.5, 100)
np_normal_dis
## min, max, mean, median, sd
print('min: ', two_dimension_array.min())
print('max: ', two_dimension_array.max())
print('mean: ',two_dimension_array.mean())
 #print('median: ',two_dimension_array.median())
print('sd: ', two_dimension_array.std())


#Creating pandas series
nums = [1, 2, 3, 4,5]
s = pd.Series(nums)
print(s)


fruits = ['Orange','Banana','Mango']
fruits = pd.Series(fruits, index=[1, 2, 3])  # with index
print(fruits)


# creating dataframe
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [24, 27, 22, 32],
    'Score': [ 90, 88,40, 60]
}
df = pd.DataFrame(data, columns=['Name','Age', 'Score'])
print(df)
print(df.loc[[0,1]])
Grade = ['A', 'A', 'D','C']
df['Grade'] = Grade
print(df)
print(df.columns)
print(df.columns.dtype)
print(df['Age'].dtype)

# Generate larger datasets
large_data = pd.DataFrame({
    'A': np.random.randint(0, 100, 100),
    'B': np.random.random(100),
    'C': np.random.normal(0, 1, 100)
})
print(large_data)



# Line plot
plt.plot(random_data)
plt.title('Line Plot')
plt.show()

# Histogram
plt.figure(figsize=(8, 6))
plt.hist(normal_data, bins=10, color='blue', alpha=0.5)
plt.title('Histogram')
plt.show()



# Font dictionary
colour = {
    'color': 'blue',
    'fontname': 'Times New Roman',
    'size': 18
}

x_axis = np.array([10, 50])
y_axis = np.array([5, 400])

# First plot with markers
plt.plot(x_axis, y_axis, marker='o', ms=10, mec='red', mfc='k')

# Second plot (adding a straight line for illustration)
plt.plot(x_axis, y_axis)

# Title
plt.title('DATA VISUALIZATION', fontdict=colour, loc='center')

# Labels with correct font dictionary
plt.ylabel('Height', fontdict={'color': 'blue', 'size': 14})
plt.xlabel('Age', fontdict={'color': 'red', 'size': 13})

# Show the plot
plt.show()

temp = np.array([1,2,3,4,5])
pressure = temp * 2 + 5
pressure

plt.plot(temp,pressure)
plt.xlabel('Temperature in oC')
plt.ylabel('Pressure in atm')
plt.title('Temperature vs Pressure')
plt.xticks(np.arange(0, 6, step=0.5))
plt.show()























