# Formating and inturpretting the Data
import pandas as pd
import numpy as np
import scipy.stats as st

# Creating and Displaying Interactive Graphs 
import matplotlib.pyplot as plt 
from matplotlib.widgets import CheckButtons 
from ipywidgets import widgets 
from IPython.display import display
import seaborn as sns 
#--- Reading the Data ---
rawDF = pd.read_csv('../datasets/games.csv')

# lowercasing all column names
rawDF.columns = [col.lower() for col in rawDF.columns]

# Handeling missing values and 'tbd'
rawDF['year_of_release'] = rawDF['year_of_release'].fillna('0').astype('int')
    # setting the missing years as 0 allows me to work with the data in a more intuitive, unobstructive, way 
rawDF.loc[rawDF['user_score'] == 'tbd', 'user_score'] = np.nan
    # Setting the to be determined scores as missing values to be handled as such
rawDF['user_score'] = rawDF['user_score'].astype('float')
    # setting all of the userscores to a float to help with calculation later
rawDF['rating'] = rawDF['rating'].fillna('unknown')
    # I was asked to fill the rating column with "unknown" at this early stage of the data (even after handling later)
    
# Adding a "total sales" column to help analyze profitability for each platform/game
rawDF['total_sales'] = rawDF['na_sales'] + rawDF['eu_sales'] + rawDF['jp_sales']    

# splitting the data into the most recent (6 years(double the life of a console after it's peak)) games for a 
# healthy data set of 5277 listings
sortDF = rawDF[rawDF['year_of_release']>=2010]
genres = sortDF['genre'].unique() # Pulling the individual genres from the dataset to be used in graphing

# This will fill the missing "user_scores" and "critic_scores" with the mean of the non-null scores in each genre
genresData = {} # Initializing the Mother Dictionary for the individual Genre Dataframes
for genre in genres:
    # Create a copy of the DataFrame for the current genre
    genreDF = sortDF[sortDF['genre'] == genre].copy()
    
    # Calculate mean of non-null 'user_score' and 'critic_score'
    user_mean = round(genreDF['user_score'].mean(), 1)
    critic_mean = round(genreDF['critic_score'].mean(), 1)
    
    # Fill missing 'user_score' and 'critic_score' with the respective means
    genreDF.loc[:, 'user_score'] = genreDF['user_score'].fillna(user_mean) 
    genreDF.loc[:, 'critic_score'] = genreDF['critic_score'].fillna(critic_mean)
    
    
    # Store the modified DataFrame in the dictionary with genre as the key
    genresData[genre] = genreDF

sortDF = pd.concat(genresData.values(), ignore_index=True)
display(sortDF.sample(20))
sortDF.info()
# Creating an interactive graph for platform Sales (total and in the different regions) against year of release
platforms = sortDF['platform'].unique()
metric = ['total_sales', 'na_sales', 'eu_sales', 'jp_sales'] # Different categories for the chart

def platformGraph(plat, metric):
    # Sorting through platforms
    pullData = sortDF[sortDF['platform'] == plat]
    
    # Setting Titles Based on Choices
    plt.title(f'{metric.replace("_", " ").title()} per Year for {plat}')
    plt.ylabel(f'{metric.replace("_", " ").title()} (mil $)')
    plt.xlabel('Year of Release')
    
    # Defining the Graph
    plt.grid(True)
    plt.ylim(0, 160)
    
    # Sorting Graph Data and Plotting
    graphData = pullData.groupby('year_of_release')[metric].sum()
    graphData.plot(kind='bar', edgecolor='k', fill=True)
    plt.figure(figsize=(10, 6))
    
    # Printing total sold and standard deviation on this platform for this metric
    print(f'{metric.replace("_"," ")}: {pullData[metric].sum()}')
    print(f'{plat} {metric.replace("_"," ")} std: {np.std(pullData[metric])}')
    
    plt.show()

plat_drop = widgets.Dropdown(options=platforms, description='Platform:')
met_drop = widgets.Dropdown(options=metric, description='Metric:')

interact_widget = widgets.interact(platformGraph, plat=plat_drop, metric=met_drop)
display(interact_widget)
# Converting scores to Readable format
sortDF['user_score'] = pd.to_numeric(sortDF['user_score'], errors='coerce')
sortDF['critic_score'] = pd.to_numeric(sortDF['critic_score'], errors='coerce')

# Creating a Scatter Plot Showing the distribution of sales against User Score and Critic Score
def scatGraph(plat):
    # Pulling Matching Data from Dataframe
    pulledData = sortDF[sortDF['platform'] == plat]
    
    # Setting Titles based on Choices
    plt.title(f'{plat} Scores versus Sales')
    plt.ylabel('Sales (mil$)')
    plt.xlabel('Score')
    plt.legend(('User Score x10', 'Critic Score'))
    
    # Calculating figures for the Scatter Plots
    x_user = np.array(pulledData['user_score'])
    x_critic = np.array(pulledData['critic_score'])
    y = np.array(pulledData['total_sales'])
    
    # Graphing the Plots
    plt.scatter((x_user*10), y)
    plt.scatter(x_critic, y)
    plt.figure(figsize=(10,6))
    
    # Displaying Coefficents and Standard Deviations
    display('User Coefficient ' + str(round(pulledData['user_score'].corr(pulledData['total_sales']), 2)))
    print(f'User Standard Deviation: {np.std(pulledData["user_score"])}')
    display('Critic Coefficient ' + str(round(pulledData['critic_score'].corr(pulledData['total_sales']), 2)))
    print(f'Critic Standard Deviation: {np.std(pulledData["critic_score"])}')
    
    # Showing the missing ratings in each platform
    missing_ratings = pulledData['rating'].value_counts().get('unknown', 0)
    display(f'Number of Games with Unknown Ratings: {missing_ratings}')

    # Showing the Plot
    plt.show()
    
interact_widget = widgets.interact(scatGraph, plat=plat_drop)
def genAllGraphs(metric):
    # Creating a Boxplot over sales from all games for each platform
    num_genres = len(genresData)

    cols = 3  # Number of columns
    rows = (num_genres + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()  

    # Create a box plot for each genre
    for idx, (genre, data) in enumerate(genresData.items()):
        currentDF = data
        print(f'{genre}:{currentDF[metric].sum()}') # For refrencing later during analysis
        axes[idx].boxplot(currentDF[metric], showfliers=False)
        axes[idx].set_title(genre)
        axes[idx].set_xlabel('Games')
        axes[idx].set_ylabel('Total Sales (mil$)')

    # Hide any unused subplots
    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

met_drop
print('Total sales amounts based on chosen metric. Written in $mil')
interact_widgets = widgets.interact(genAllGraphs, metric=met_drop)
# Generating a list of all duplicate games over different platforms
names = [name for temp, name in sortDF.groupby('name')]
duplicates = []
for df in names:
    df['rating'] = df['rating'].transform(lambda x: x.ffill().bfill())
    if len(df) > 1:
        # Gathering only the name of the game, its sales, the platform, and its ESRB Rating
        duplicates.append(df) 
dupDF = pd.concat(duplicates, ignore_index=True)
display(sortDF.head(10))
# grabbing the top 5 games that were released on multiple platforms
topGames = dupDF.sort_values('total_sales', ascending=False).iloc[:7, :7] # 7 because of cross platform hits
topUnique = topGames['name'].unique()

topTenList = []

# pulling each iteration of the top games from the whole dataset
for _, row in dupDF.iterrows():
    if row['name'] in topUnique:
        topTenList.append(row[['name', 'na_sales', 'eu_sales', 'jp_sales', 'total_sales', 'platform', 'rating']])
        
topTenGames = pd.DataFrame(topTenList)
topPlat = topTenGames['platform'].unique()

x360 = []
ps3 = []
wii = []
ds = []
ps4 = []
xone = []
pc = []
wiiu = []

for _, game in topTenGames.iterrows():
    if game['platform'] == 'X360':
        x360.append(game)
    elif game['platform'] == 'PS3':
        ps3.append(game)
    elif game['platform'] == 'Wii':
        wii.append(game)
    elif game['platform'] == 'DS':
        ds.append(game)
    elif game['platform'] == 'PS4':
        ps4.append(game)
    elif game['platform'] == 'XOne':
        xone.append(game)
    elif game['platform'] == 'PC':
        pc.append(game)
    else:
        wiiu.append(game)

x360 = pd.DataFrame(x360)
ps3 = pd.DataFrame(ps3)
xone = pd.DataFrame(xone)
ps4 = pd.DataFrame(ps4)
wii = pd.DataFrame(wii)
wiiu = pd.DataFrame(wiiu)
ds = pd.DataFrame(ds)
pc = pd.DataFrame(pc)

# Created an interactive graph to help with analysis later, this is top games divided by platform for regions
def platEDAGraph(metric):
    
    #Setting the graph
    plt.figure(figsize=(10,6))
    
    # Each of these bar graphs are another platform I want displayed on the final chart
    plt.bar(x360['name'], x360[metric]) 
    plt.bar(ps3['name'], ps3[metric])
    plt.bar(wii['name'], wii[metric])
    plt.bar(xone['name'], xone[metric])
    plt.bar(ps4['name'], ps4[metric])
    plt.bar(wiiu['name'], wiiu[metric])
    plt.bar(pc['name'], pc[metric])
    plt.bar(ds['name'], ds[metric])

    # This is so we can actually read the titles 
    plt.xticks(np.arange(len(topUnique)), topUnique) 
    plt.xlabel("Games") 
    plt.ylabel("Total Sales ($mil)") 
    plt.title("Top selling games Across All Avaliable Platforms") 
    plt.legend(topPlat) 
    plt.xticks(rotation=45)
    
    plt.show()
    
# Interactive dropdown which changes the graphs throughout the whole notebook
met_drop

interact_widget = widgets.interact(platEDAGraph, metric=met_drop)
display(interact_widget)
# Generating Chart to show profitability based on a specific region and genre
def genGraph(genre, metric):
    # Setting selected Data from DataFrame
    genPull = genresData[genre]

    # Setting Titles based on Choices
    plt.title(f'{genre} Games for {metric.replace("_", " ")}')
    plt.ylabel('Sales (mil$)')
    plt.xlabel('Games')

    # Ploting Graph
    plt.boxplot(genPull[metric], showfliers=False)
    plt.show()

gen_drop = widgets.Dropdown(options=genres, description='Genre:')
met_drop
interact_widget = widgets.interact(genGraph, genre=gen_drop, metric=met_drop)
display(interact_widget)
#--- Updating Dataframe with Ratings across Platforms ---
# Creating a copy of the DataFrame to preserve the original data and Cleaning dupDF
modData = sortDF.copy()
dupDF = dupDF.drop_duplicates(subset=['name', 'platform'])

# Selecting necessary columns from dupDF to merge
smallDupDF = dupDF[['name', 'platform', 'rating']]

# Merging the updated ratings back into sortDF
modData = pd.merge(
    modData, 
    smallDupDF, 
    on=['name', 'platform'], 
    how='left', 
    suffixes=('', '_updated')
)

# Replace the old rating column with the updated one where available
modData['rating'] = modData['rating_updated'].combine_first(modData['rating'])

# Drop the temporary 'rating_updated' column
modData.drop(columns=['rating_updated'], inplace=True)
# getting total sales across all platforms into a single dataframe with one entry per game 
aggregate_functions = {
    'total_sales': 'sum',
    'na_sales': 'sum',
    'eu_sales': 'sum',
    'jp_sales': 'sum',
    'rating': 'first'
}
crossPlatSales = dupDF.groupby('name').agg(aggregate_functions).reset_index()

#---Merging crossPlatSales back into the DataFrame to have a single instance of game sales across ALL platforms---
# Cleaning sortDF
duplicates = modData['name'][modData.duplicated(subset=['name'], keep=False)].unique()
duplicatesDF = modData[modData['name'].isin(duplicates)].copy()
uniqueDF = modData[~modData['name'].isin(duplicates)].copy()

# Merging together
crossPlatDF = pd.concat([uniqueDF[['name','total_sales','na_sales','eu_sales','jp_sales','rating']], crossPlatSales], ignore_index=True)
# I found myself updating this chart a lot so I just made it interactive to help me out
def finalEDAGraph(metric):
    ESRBData = crossPlatDF.sort_values(metric, ascending=False)[['name', 'rating', metric, 'total_sales']]
    display(ESRBData.head(10))
met_drop
interact_widget = widgets.interact(finalEDAGraph, metric=met_drop)
display(interact_widget)
# --- Visualization for the North American User Profile ---
# Define the data
platforms = ['Xbox 360', 'PlayStation 3', 'Wii', 'PlayStation 4', 'Xbox One']
sales_platforms = [334.18, 229.25, 121.20, 108.74, 93.12]

genres = ['Action', 'Shooter', 'Sports', 'Misc', 'Role-Playing']
sales_genres = [290.64, 237.47, 156.81, 123.80, 112.05]

# Creating Bar Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plotting platforms
ax1.barh(platforms, sales_platforms, color='skyblue')
ax1.set_title('Top Five Platforms in North America')
ax1.set_xlabel('Total Sales (mil$)')
ax1.set_ylabel('Platform')

# Plotting genres
ax2.barh(genres, sales_genres, color='lightgreen')
ax2.set_title('Top Five Genres in North America')
ax2.set_xlabel('Total Sales (mil$)')
ax2.set_ylabel('Genre')

# Showing Bar Charts
plt.tight_layout()
plt.show()

#Creating Pie Charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Platform Pie Chart
ax1.pie(sales_platforms, labels=platforms, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
ax1.set_title('Top Five Platforms in North America')

# Genre Pie Chart
ax2.pie(sales_genres, labels=genres, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
ax2.set_title('Top Five Genres in North America')

# Show Pie Charts
plt.tight_layout()
plt.show()
# --- Visualization for the European User Profile ---
# Define the data
platforms = ['PlayStation 3', 'Xbox 360', 'PlayStation 4', 'Personal Computer', 'Wii']
sales_platforms = [213.60, 163.41, 141.09, 68.82, 65.91]

genres = ['Action', 'Shooter', 'Sports', 'Role-Playing', 'Misc']
sales_genres = [233.63, 171.45, 116.84, 75.48, 66.09]

# Creating Bar Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plotting platforms
ax1.barh(platforms, sales_platforms, color='skyblue')
ax1.set_title('Top Five Platforms in Europe')
ax1.set_xlabel('Total Sales (mil$)')
ax1.set_ylabel('Platform')

# Plotting genres
ax2.barh(genres, sales_genres, color='lightgreen')
ax2.set_title('Top Five Genres in Europe')
ax2.set_xlabel('Total Sales (mil$)')
ax2.set_ylabel('Genre')

# Showing Bar Charts
plt.tight_layout()
plt.show()

#Creating Pie Charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Platform Pie Chart
ax1.pie(sales_platforms, labels=platforms, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
ax1.set_title('Top Five Platforms in Europe')

# Genre Pie Chart
ax2.pie(sales_genres, labels=genres, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
ax2.set_title('Top Five Genres in Europe')

# Show Pie Charts
plt.tight_layout()
plt.show()
# --- Visualization for the Japan User Profile ---
# Threshold to define significant sales
sales_threshold = 0.5  # Adjust as required

# Filter games with significant sales in Japan
significant_japan_sales = modData[modData['jp_sales'] > sales_threshold]

# Aggregate ratings for games with significant sales in Japan
japan_ratings = significant_japan_sales['rating'].value_counts()

# Visualizing the Rating Distribution
plt.figure(figsize=(10, 5))
plt.bar(japan_ratings.index, japan_ratings.values, color='skyblue')
plt.title('Distribution of Ratings for Games with Significant Sales in Japan')
plt.xlabel('Rating')
plt.ylabel('Number of Games')

# Annotate counts on bars
for i, count in enumerate(japan_ratings.values):
    plt.text(i, count + 1, str(count), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Define the data
platforms = ['Nintendo 3DS', 'PlayStation 3', 'PlayStation Portable', 'Nintendo DS', 'PlayStation Vita']
sales_platforms = [100.62, 59.26, 42.20, 27.90, 21.84]

genres = ['Role-Playing', 'Action', 'Misc', 'Platform', 'Adventure']
sales_genres = [103.54, 72.20, 24.29, 15.81, 15.67]

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot for platforms
ax1.barh(platforms, sales_platforms, color='skyblue')
ax1.set_title('Top Five Platforms in Japan')
ax1.set_xlabel('Total Sales (mil$)')
ax1.set_ylabel('Platform')

# Plot for genres
ax2.barh(genres, sales_genres, color='lightgreen')
ax2.set_title('Top Five Genres in Japan')
ax2.set_xlabel('Total Sales (mil$)')
ax2.set_ylabel('Genre')

# Show plots
plt.tight_layout()
plt.show()

# Pie charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Platform Pie Chart
ax1.pie(sales_platforms, labels=platforms, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
ax1.set_title('Top Five Platforms in Japan')

# Genre Pie Chart
ax2.pie(sales_genres, labels=genres, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
ax2.set_title('Top Five Genres in Japan')

# Show plots
plt.tight_layout()
plt.show()
# --- Formulating and Testing a null hypotheses that Xbox One and PC Platforms are the same ---

# Setting variables
user_xone = modData.loc[modData['platform']=='XOne', 'user_score'].dropna()
user_pc = modData.loc[modData['platform']=='PC', 'user_score'].dropna()

# Using an independent test to test independent variables
results = st.ttest_ind(user_xone, user_pc, equal_var=False)

# Having a 10% margin makes sense to me to handle any outliers in the Data and get an accurate answer
alpha = 0.1

print('p-value: ', results.pvalue)

if results.pvalue < alpha:
    print("We reject the null hypothesis")
    print("The user ratings for Xbox One and PC platforms are significantly different")
else:
    print("We can't reject the null hypothesis")
    print("We cannot conclude that the user ratings for Xbox One and PC platforms are significantly different")
# --- Formulating and Testing a null hypothesis that user ratings for the Action and Sports Genres are the Same --

# Setting Variables
user_action = modData.loc[modData['genre']=='Action', 'user_score'].dropna()
user_sport = modData.loc[modData['genre']=='Sports', 'user_score'].dropna()

# Using the same two-tailed independent test to test the independent variables
results = st.ttest_ind(user_action, user_sport, equal_var=False)

# Keeping the same 10% margin of error as I had from the last cell
alpha = 0.1

print('p-value: ', results.pvalue)

if results.pvalue < alpha:
    print("We reject the null hypothesis")
    print("The user ratings for Action and Sports genres are significantly different")
else:
    print("We can't reject the null hypothesis")
    print("We cannot conclude that the user ratings for Action and Sports genres are different")