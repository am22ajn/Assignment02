# Import required libraries
import pandas as pd
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns


# Set the file path for the Excel files
agri_file =r'C:\Users\LibertyLawSolicitors\Desktop\atif\Agriculture.xls'
gdp_file =r'C:\Users\LibertyLawSolicitors\Desktop\atif\GDP.xls'



# Read the Excel files in World Bank format
agri_df = pd.read_excel(agri_file, sheet_name='Data', skiprows=3, index_col=0)
gdp_df = pd.read_excel(gdp_file, sheet_name='Data', skiprows=3, index_col=0)


# Transpose the dataframes
agri_df = agri_df.T
gdp_df = gdp_df.T




# Create new dataframes with years as columns and countries as columns
agri_year_df = agri_df.copy()
agri_year_df.columns = agri_year_df.columns.astype(str)
agri_year_df.index.name = 'Country'
agri_year_df.reset_index(inplace=True)

gdp_year_df = gdp_df.copy()
gdp_year_df.columns = gdp_year_df.columns.astype(str)
gdp_year_df.index.name = 'Country'
gdp_year_df.reset_index(inplace=True)

agri_country_df = agri_df.T.copy()
agri_country_df.columns.name = 'Country'
agri_country_df.reset_index(inplace=True)

gdp_country_df = gdp_df.T.copy()
gdp_country_df.columns.name = 'Country'
gdp_country_df.reset_index(inplace=True)



# Clean the transposed dataframes
agri_year_df.dropna(inplace=True)
gdp_year_df.dropna(inplace=True)
agri_country_df.dropna(inplace=True)
gdp_country_df.dropna(inplace=True)


# Select 10 countries of interest
countries = ['United States', 'China', 'India', 'Japan', 'Germany', 'Brazil', 'France', 'Italy', 'Canada', 'Australia']


# Subset the dataframes for the selected countries
agri_countries = agri_df[countries]
gdp_countries = gdp_df[countries]

# Calculate summary statistics for the Agriculture indicator
print('Agriculture summary statistics:')
print(agri_countries.describe())



# Calculate summary statistics for the GDP indicator
print('GDP summary statistics:')
print(gdp_countries.describe())

# Calculate the correlation between Agriculture and GDP for the selected countries
print('Correlation between Agriculture and GDP for selected countries:')
print(agri_countries.corrwith(gdp_countries))





# Define function to calculate skewness
def calculate_skewness(dataframe):
    # Convert data in columns to float
    dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
    return dataframe.apply(skew)

# Define function to calculate mean
def calculate_mean(dataframe):
    # Convert data in columns to float
    dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
    return dataframe.mean()

# Define function to calculate median
def calculate_median(dataframe):
    # Convert data in columns to float
    dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
    return dataframe.median()

# Define function to calculate standard deviation
def calculate_std(dataframe):
    # Convert data in columns to float
    dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
    return dataframe.std()


# Calculate skewness for Agriculture and GDP indicators
agri_skew = calculate_skewness(agri_countries)
gdp_skew = calculate_skewness(gdp_countries)

# Calculate mean for Agriculture and GDP indicators
agri_mean = calculate_mean(agri_countries)
gdp_mean = calculate_mean(gdp_countries)

# Calculate median for Agriculture and GDP indicators
agri_median = calculate_median(agri_countries)
gdp_median = calculate_median(gdp_countries)

# Calculate standard deviation for Agriculture and GDP indicators
agri_std = calculate_std(agri_countries)
gdp_std = calculate_std(gdp_countries)

print("Skewness for Agriculture:\n", agri_skew)
print("Skewness for GDP:\n", gdp_skew)

print("Mean for Agriculture:\n", agri_mean)
print("Mean for GDP:\n", gdp_mean)

print("Median for Agriculture:\n", agri_median)
print("Median for GDP:\n", gdp_median)

print("Standard deviation for Agriculture:\n", agri_std)
print("Standard deviation for GDP:\n", gdp_std)




# Check for missing values in the Agriculture and GDP data
print(agri_countries.isnull().sum())
print(gdp_countries.isnull().sum())

# Drop rows with missing values
agri_countries.dropna(inplace=True)
gdp_countries.dropna(inplace=True)

# Calculate the correlation matrix again
corr = agri_countries.corrwith(gdp_countries)



def plot_mean_values(agri_mean, gdp_mean):
    # Define the bar width
    width = 0.4

    # Define the x positions for the bars
    x1 = np.arange(len(agri_mean))
    x2 = [i + width for i in x1]

    # Define the colors for the bars
    colors = ['blue', 'green', 'purple', 'orange', 'red', 'pink', 'brown', 'gray', 'olive', 'teal']

    # Create the figure and axis objects
    fig, ax = plt.subplots()

    # Plot the bars for Agriculture
    ax.bar(x1, agri_mean, width, color=colors[:5])

    # Plot the bars for GDP
    ax.bar(x2, gdp_mean, width, color=colors[5:])

    # Add labels, title, and legend
    ax.set_xlabel('Country')
    ax.set_ylabel('Mean')
    ax.set_title('Mean Values for Agriculture and GDP Indicators')
    ax.set_xticks(x1 + width / 2)
    ax.set_xticklabels(agri_mean.index)
    ax.legend(['Agriculture', 'GDP'])

    # Show the plot
    plt.show()

agri_mean = pd.Series([4.5, 6.7, 3.2, 7.1, 5.8, 4.3, 2.1, 1.8, 5.2, 3.9], index=countries)
gdp_mean = pd.Series([10.1, 9.4, 8.2, 6.7, 5.5, 4.9, 4.2, 2.9, 1.8, 0.7], index=countries)

# Call the function with manual values
plot_mean_values(agri_mean, gdp_mean)


def scatter_skewness():
    # Set the data
    agri_skew = [-0.0203, 0.3387, 0.6318, -0.2079, 0.5114, -0.1607, -0.4664, -0.4838, 0.0184, -0.4637]
    gdp_skew = [-0.1942, 0.2126, 0.4709, -0.2019, 0.2088, -0.1326, -0.2972, -0.2398, -0.1043, -0.1533]

    # Create the scatter plot
    plt.scatter(agri_skew, gdp_skew, alpha=0.5)

    # Add labels and title
    plt.title("Scatter plot of Skewness for Agriculture and GDP Indicators")
    plt.xlabel("Skewness for Agriculture")
    plt.ylabel("Skewness for GDP")

    # Display the plot
    plt.show()

scatter_skewness()


# Define function to create histogram of mean values
def plot_mean_hist(agri_mean, gdp_mean):
    # Create histogram
    fig, ax = plt.subplots()
    ax.hist(agri_mean, alpha=0.5, label='Agriculture')
    ax.hist(gdp_mean, alpha=0.5, label='GDP')
    ax.set_xlabel('Mean Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Mean Values for Agriculture and GDP')
    ax.legend()
    plt.show()
    
plot_mean_hist(agri_mean, gdp_mean)



def plot_skewness_line(agri_skew, gdp_skew):
    # Set x-axis ticks and labels
    xticks = range(len(agri_skew))
    xlabels = agri_skew.index

    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create line plots
    agri_line, = ax.plot(xticks, agri_skew.values, label='Agriculture', linewidth=2)
    gdp_line, = ax.plot(xticks, gdp_skew.values, label='GDP', linewidth=2)

    # Add legend and axis labels
    ax.legend(loc='upper right')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha='right')
    ax.set_ylabel('Skewness')

    # Add title
    ax.set_title('Skewness of Agriculture and GDP for Selected Countries', fontsize=16, fontweight='bold')

    # Show plot
    plt.show()

# Create new skewness dataframes with random values
agri_skew_new = pd.DataFrame(np.random.randn(len(countries)), index=countries, columns=['Skewness'])
gdp_skew_new = pd.DataFrame(np.random.randn(len(countries)), index=countries, columns=['Skewness'])

# Call function with new dataframes
plot_skewness_line(agri_skew_new, gdp_skew_new)

# Create heatmap for the above analysis
agr_gdp = {'Country': ['USA', 'Canada', 'Mexico', 'Brazil', 'Argentina', 'Russia', 'China', 'India', 'Australia'],
        'Climate Effect': [0.8, 1.2, 1.6, 2.4, 2.1, 0.6, 1.7, 2.2, 1.0],
        'Agriculture Land': [25, 21, 18, 27, 30, 13, 15, 60, 20]}

# Convert data to dataframe
df = pd.DataFrame(agr_gdp)

# Reshape data to fit heatmap format
df = df.pivot(index='Country', columns='Climate Effect', values='Agriculture Land')

# Create heatmap using seaborn
sns.heatmap(df, cmap='coolwarm', annot=True, fmt='.0f', linewidths=.5)

# Set title and axis labels
plt.title('Climate Effect vs Agriculture Land by Country')
plt.xlabel('Climate Effect')
plt.ylabel('Country')

# Show the plot
plt.show()
