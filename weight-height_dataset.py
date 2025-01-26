import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('c:/Users/MINE/Downloads/weight-height.csv')
print(df)
print(df.head(10)) # display the first 10 records
print(df.info())   #display the information about the dataset
print(df.tail())  #display the last few row
print(df.shape)  #display the number of rows and coluums
print(df.columns)   #display the column names
print(df.loc[:,['Height','Weight']])  #display the height and weight columns
print(len('Height') == len('Weight'))  #check if the records in the height and weight columns are equal
print(df['Gender'])   #display the gender column

# Data processing: Calculate basic statistics for the dataset

#group data by
# Group data by Gender and calculate mean height and weight
gender_means = df.groupby('Gender')[['Height', 'Weight']].mean()  #group the data 
# Display mean values
print(gender_means)

# Group the data by gender
grouped_data = df.groupby('Gender')
# Calculate the mean height and weight for each gender
avg_values = grouped_data[['Height', 'Weight']].mean()
print("Average Height and Weight by Gender:")
print(avg_values)

# calculate the media height and weight
median_values=grouped_data[['Height', 'Weight', ]].median()
print('median Height and weight by Gender:')
print(median_values)

#calculate standard deviation of Height and weight
std_values=grouped_data[['Height','Weight']].std()
print('standard deviation of Height and Weight:')
print(std_values)

#calculate the variance of Height and Weight
variance_values=grouped_data[['Height', 'Weight']].var()
print('variance of Height and Weight:')
print(variance_values)

#Calculate the min and max height and weight
min_values=grouped_data[['Height', 'Weight']].min()
max_values=grouped_data[['Height', 'Weight']].max()
print('min values of Height and Weight:')
print(min_values)
print('max values of Height and Weight:')
print(max_values)

#calculate the range of Height and Weight
range_values=max_values-min_values
print('range of Height and Weight:')
print(range_values)


# Visualize: Gender distribution
gender_counts = df['Gender'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='viridis')
plt.title('Gender Distribution')
plt.ylabel('Count')
plt.xlabel('Gender')
plt.show()

# Visualize: Distribution of Height and Weight
# Histograms for height and weight by gender
plt.figure(figsize=(12, 5))
sns.histplot(data=df, x='Height', hue='Gender', kde=True, bins=20, alpha=0.5)
plt.title('Distribution of Height by Gender')
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 5))
sns.histplot(data=df, x='Weight', hue='Gender', kde=True, bins=20, alpha=0.5)
plt.title('Distribution of Weight by Gender')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.show()

# Boxplot for Height
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Gender', y='Height')
plt.title('Boxplot of Height by Gender')
plt.xlabel('Gender')
plt.ylabel('Height')
plt.show()

# Boxplot for Weight
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Gender', y='Weight')
plt.title('Boxplot of Weight by Gender')
plt.xlabel('Gender')
plt.ylabel('Weight')
plt.show()

# Calculate IQR for height
q1_height = df['Height'].quantile(0.25)
q3_height = df['Height'].quantile(0.75)
iqr_height = q3_height - q1_height
lower_bound_height = q1_height - 1.5 * iqr_height
upper_bound_height = q3_height + 1.5 * iqr_height

# Calculate IQR for weight
q1_weight = df['Weight'].quantile(0.25)
q3_weight = df['Weight'].quantile(0.75)
iqr_weight = q3_weight - q1_weight
lower_bound_weight = q1_weight - 1.5 * iqr_weight
upper_bound_weight = q3_weight + 1.5 * iqr_weight

# Detect outliers

outliers_height = df[(df['Height'] < lower_bound_height) | (df['Height'] > upper_bound_height)]
outliers_weight = df[(df['Weight'] < lower_bound_weight) | (df['Weight'] > upper_bound_weight)]

print(f"Outliers in Height: {len(outliers_height)}")
print(f"Outliers in Weight: {len(outliers_weight)}")


