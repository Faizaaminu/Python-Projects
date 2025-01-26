### DATA GENERATION AND VISUALIZATION


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

# generate uniform data
random_data=np.random.uniform(20,30)
print(random_data)

#generate a random element from non empty sequnce using random.choice()
items=['one', 'two', 'three' 'four' ,'five']
items=np.random.choice(items)
print(items)

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


#DATA VISUALIZATION
# Line plot
x=[1,2,3,4,5]
y=[18,37,57,39,58]
plt.plot(x ,y)


x=[1,2,3,4,5]
y=[10,40,20,40,50]
plt.plot(x ,y)
plt.title('Line Plot')
plt.show()

x=[1,2,3,4,5]
y=[18,37,57,39,58]
plt.subplot(2,2,1)
plt.plot(x ,y)
plt.title("axis 1")



x=[1,2,3,4,5]
y=[10,40,20,40,50]
plt.subplot(2,2,2)
plt.plot(x, y)
plt.title("axis 2")

x=[1,2,3,4,5]
y=[10,30,10,40,30]
plt.subplot(2,2,3)
plt.scatter(x,y)
plt.title("axis 3")


x=[1,2,3,4,5]
y=[37,46,38,49,10]
plt.subplot(2,2,4)
plt.bar(x, y)
plt.title("axis 4")

plt.suptitle("Super Title")
plt.show()

# Histogram
plt.figure(figsize=(8, 6))
plt.hist(normal_data, bins=10, color='blue', alpha=0.5)
plt.title('Histogram')
plt.show()

import seaborn as sns

# Scatter plot
sns.scatterplot(x=large_data['A'], y=large_data['B'])
plt.title('Scatter Plot')
plt.show()


from numpy import array

font_create = {
    'color': 'blue',
    'font': 'Times New Roman',
    'size': 10
}

x_axis = np.array([10, 50])
y_axis = np.array([25, 400])

# plt.plot(x_axis, y_axis '*', ms=20, mec='red', mfc='black)
plt.subplot(2,3,4)
plt.plot(x_axis, y_axis, ls='dashdot', c='r', lw=3.0)
plt.title('DATA VISUALIZATION', fontdict=font_create, loc='center')
# Labels with correct font dictionary
plt.ylabel('Height', fontdict={'color': 'blue', 'size': 13})
plt.xlabel('Age', fontdict={'color': 'green', 'size': 13})
plt.grid(lw=2.0,c='k')


# Data for the first plot
x = np.array([20, 150])
y = np.array([50, 350])

# Data for the second plot
xdata = np.array([20, 40, 14, 2])
ydata = np.array([11, 20, 10, 50])

# Subplot configuration
plt.subplot(2, 3, 1)  # Subplot position in a 2x3 grid
plt.plot(x, y, label='Line 1', color='blue', lw=2.0)  # Plot 1: Line with custom style
plt.plot(xdata, ydata, marker='*', label='Line 2', color='green', linestyle='dashed')  # Plot 2: Line with markers
plt.title('PLOT 1')  # Title of the subplot
plt.xlabel('X-Axis')  # X-axis label
plt.ylabel('Y-Axis')  # Y-axis label
plt.grid(axis='x', c='red', lw=2.0)  # Grid lines for the x-axis
plt.legend()  # Add legend for better interpretation
plt.tight_layout()  # Adjust layout to prevent overlaps
plt.show()


#plot 3
x_point=np.array([20,150])
y_point=np.array([50,320])
plt.subplot(2,3,1)
plt.plot(x_point,y_point, marker='*')
plt.title('PLOT 2')

#plot 4
x2=np.array([20,150])
y1=np.array([50,320])
plt.subplot(2,3,2)
plt.plot(x2,y1, ls='dotted', lw=2.0, marker='*', c='green')
plt.title('PLOT 3')


#plot 5
#A student grade and total number of student as a bar chart
student_grade=np.array(['A','B','C','E','F'])
number_of_student=np.array([50,320,15,24,2])
plt.subplot(2,3,4)
plt.bar(student_grade, number_of_student, color='red')
plt.xlabel('student grade')
plt.ylabel('number of students')
plt.show()

# plot 6
pie_y = [40, 30, 20, 10]
label_data = ['English', 'Maths ', 'Physics ', 'Computer Science']
# Pie chart
plt.pie(pie_y, labels=label_data, autopct='%1.1f%%', startangle=90)

# Title and display
plt.title('Pie Chart')
plt.suptitle('DATA VISUALIZATION', c='r')
plt.show()





















