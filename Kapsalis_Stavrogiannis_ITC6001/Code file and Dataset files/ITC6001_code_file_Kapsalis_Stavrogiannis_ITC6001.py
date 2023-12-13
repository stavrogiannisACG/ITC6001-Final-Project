import numpy as np
import pandas as pd 
import json
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from datetime import datetime
import plotly.express as px
import matplotlib.pyplot as plt
plt.ioff()

# reading all data files into pandas dataframes
artist_info = pd.read_csv('artists.dat', sep='\t', encoding=None, names=None, header=0, index_col=None)
tag_info = pd.read_csv('tags.dat', sep='\t', encoding='ISO-8859-1', names=None, header=0, index_col=None)
user_artists = pd.read_csv('user_artists.dat', sep='\t', encoding=None, names=None, header=0, index_col=None)
user_friends = pd.read_csv('user_friends.dat', sep='\t', encoding=None, names=None, header=0, index_col=None)
user_tags_ts = pd.read_csv('user_taggedartists-timestamps.dat', sep='\t', encoding=None, names=None, header=0, index_col=None)
user_tags = pd.read_csv('user_taggedartists.dat', sep='\t', encoding=None, names=None, header=0, index_col=None)


###################################################################
print("****** Q1 - Data Description ******")

### Frequency plot of the listening frequency of artists by users
res1a_raw = pd.merge(artist_info, user_artists, left_on='id', right_on='artistID', how='inner')
print(f"The first 5 rows of the dataframe containing the data we used to plot the listening frequency of artists by users:\n\n {res1a_raw.head()}")
print(f"The number of duplicate values in this dataframe is: {user_artists[user_artists.duplicated(subset=['userID','artistID'])].shape[0]}.")  # or res1a[res1a.duplicated(subset=['userID','artistID'])]

print("Plotting the total listening count per artist.\nThe sum of all ‘weight’ values corresponding to a single artistID gives the total count of streams their songs have got.\n\n")
temp = pd.pivot_table(data=res1a_raw, index='name', values='weight', aggfunc='sum')
temp = temp.sort_values(by='weight', ascending=False)
temp.rename(columns={'weight': 'listening count'}, inplace=True)
res1a_final = temp.reset_index()  # reset_index converts the pivot's index into a column, aka I transform the
# pivot table back to a pandas dataframe
res1a_final[:10].plot(x='name', y='listening count', kind='bar', rot=45)  
plt.title('Total Listening Count per Artist')
plt.show()



### Frequency of tag usage
# For this analysis, we used the 'user_tags' dataframe. 
# Making sure there are no duplicate rows based on the ['userID', 'artistID', 'tagID'] variables pair - a user can apply many 
# tages to a single user, but each unique tag (based on its ID) only once. 
print(f'The number of duplicated (userID, artistID, tagID) triplets in the "user_taggedartists-timestamps" csv and df is: {user_tags[user_tags.duplicated(subset=["userID","artistID","tagID"])].shape[0]}')

# Showing the top 10 most used tags (by their values, as their IDs don't convey any meaning in presentation).
# I only show the top 10 because otherwise the plot gets over-crowded.

# First, in order to display tag values and not just IDs on the x-axis, I join the user_artists df with that containing further info on the tags.
res1b_raw = pd.merge(tag_info, user_tags, on='tagID')
print("Plotting the total frequency of uses of each tag using data from the user_tags dataframe.\n\n")
# Then, I just create a bar plot of the frequencies with which the various values of the 'tagValue' column appear in the df's records.
fig, ax = plt.subplots()
res1b_raw = res1b_raw['tagValue'].value_counts().sort_values(ascending=False)
plt.title('Top 10 most used tags')
plt.xlabel('Tag Value')
plt.ylabel('Use Frequency')
res1b_raw[:10].plot(ax=ax, kind='bar', rot=45)  # res1b_raw is just an array of named observations - streaming count per ARTIST
plt.show()



print("****** Q1 - Outlier Detection Using the Z-Score Method ******")
# Detecting the outliers in each distribution under study 

### total listening count for artists
print(f"The first 5 rows of the dataframe containing data on the total listening count for each artist have as follows:\n\n {res1a_final.head()}")

res1a_final['Z-score'] = round((res1a_final['listening count'] - res1a_final['listening count'].mean()) / res1a_final['listening count'].std(), 2)
outliers_artists = res1a_final[abs(res1a_final['Z-score']) > 3]
print(f"There are {outliers_artists.shape[0]} outlier values in the distribution of artists' listening count.")  # 85 outlier values; too many to show in a single table, we will include a csv file with the 
# results in our deliverable folder for the project.
 
outliers_artists.to_csv('Z-outliers_artists.csv', sep=',', header=True, index=False)  # code to write the results in a csv

# Graphically checking the validity of the normality assumption.
print("Checking the validity of the Z-Score Method's normality assumption using violin plots.\n\n")
fig1 = px.violin (res1a_final['listening count'], box=True, points='all')
fig1.show()

# Positive asymmetry is detected, due to relatively few values that affect the mean upwards. 
# Thus, the use of this method in our example gives irrelevant results, since the normality assumption doesn't hold. 



### total listening time for users 
# for each user, I'll add together the views they've made to artists' songs
res1_d_raw = pd.pivot_table(data=user_artists, index=['userID'], values=['weight'], aggfunc=np.sum)
res1_d_raw = res1_d_raw.sort_values(by='weight', ascending=False)
res1_d_final = res1_d_raw.reset_index()
print(f"The first 5 rows of the dataframe containing data on the total listening time of each user have as follows:\n\n {res1_d_final.head()}")

res1_d_final['Z-score'] = round((res1_d_final['weight'] - res1_d_final['weight'].mean()) / res1_d_final['weight'].std(), 2)
outliers_users = res1_d_final[abs(res1_d_final['Z-score']) > 3]
print(f"There are {outliers_users.shape[0]} outlier values in the dataframe.")  # 45 outlier values; too many to show in a single table, we will include a csv file with the 
# results in our deliverable folder for the project.

outliers_users.to_csv('Z-outliers_users.csv', sep=',', header=True, index=False)  # code to write the results in a csv file 

# Graphically checking the validity of the normality assumption.
print("Checking the validity of the Z-Score Method's normality assumption using violin plots.\n\n")
fig2 = px.violin (res1_d_final['weight'], box=True, points='all')
fig2.show()

# Even higher positive asymmetry than in the previous case, the same remarks are true. 



### frequency of tag usage 
# in the 'res1b_raw' series, we already have the count of usage for each tag (by the tag name) for all tags in the dataset 
# Thus, our raw data are ready to use to create bins and show the distribution for the random variable representing the 
# count under study
res1_b_final = res1b_raw.to_frame().reset_index()
res1_b_final.rename(columns={'tagValue': 'count', 'index': 'tagValue'}, inplace=True)
res1_b_final['Z-score'] = round((res1_b_final['count'] - res1_b_final['count'].mean()) / res1_b_final['count'].std(), 2)
outliers_tags = res1_b_final[abs(res1_b_final['Z-score']) > 3]
print(f"There are {outliers_tags.shape[0]} values in the dataframe containing data on the number of times each tag has been used.")  
# 68 outlier values; too many to show in a single table, we will include a csv file with the 
# results in our deliverable folder for the project.

outliers_tags.to_csv('Z-outliers_tags.csv', sep=',', header=True, index=False)  # code to write the results in a csv file

# Graphically checking the validity of the normality assumption.
print("Checking the validity of the Z-Score Method's normality assumption using violin plots.\n\n")
fig3 = px.violin (res1_b_final['count'], box=True, points='all')
fig3.show()

# The same holds true for the normality distribution as before.



print("****** Q1 - Outlier Detection Using the IQR Method ******")

### total listening count for artists

q3_ = res1a_final['listening count'].quantile(q=0.75)
q1_ = res1a_final['listening count'].quantile(q=0.25)
iqr_ = q3_ - q1_ 

res1a_final['IQR_meth'] = (res1a_final['listening count'] >= q3_ + 1.5 * iqr_) | (res1a_final['listening count'] <= q1_ - 1.5 * iqr_)

outliers_artists_IRQ = res1a_final[res1a_final['IQR_meth'] == True]
print(f"There are {outliers_artists_IRQ.shape[0]} in the distribution of the total listening count for artists.")  # we see that this method gives many more outliers so it's much more effective,
# as the threshold values it sets don't rely on distribution assumptions that need to hold

outliers_artists_IRQ.to_csv('IQR-outliers_artists.csv', sep=',', header=True, index=False)

res1a_final.boxplot('listening count', vert=False, showfliers=False)
plt.title('Distribution of Total Streaming Count of Artists (not showing outliers)')
plt.axvline(res1a_final['listening count'].mean(), color='red')
plt.show()
# the median is much closer to q1 than q3, and the left-side whisker is much shorter than the right-side whisker
# this shows that most values are concentrated at the left-side end of the distribution, yet there are some 
# (outliers_users.shape[0] in number) extremely high values that influence the mean of the distribution upwards

# thus, the distribution is far from normal, and the thresholds of the z-score method here are irrelevant
# So, the use of this method in this case doesn't bring up trustworthy resutls 



### total listening time for users 
q3_ = res1_d_final['weight'].quantile(q=0.75)
q1_ = res1_d_final['weight'].quantile(q=0.25)
iqr_ = q3_ - q1_ 

res1_d_final['IQR_meth'] = (res1_d_final['weight'] >= q3_ + 1.5 * iqr_) | (res1_d_final['weight'] <= q1_ - 1.5 * iqr_)

outliers_users_IRQ = res1_d_final[res1_d_final['IQR_meth'] == True]
print(f"There are {outliers_users_IRQ.shape[0]} outliers in the distribution of the total listening time for users.")# we see that this method gives many more outliers so it's much more effective,
outliers_users_IRQ.to_csv('IQR-outliers_users.csv', sep=',', header=True, index=False)

res1_d_final.boxplot('weight', vert=False, showfliers=False)
plt.title('Distribution of Total Streaming Count for Users (not showing outliers)')
plt.axvline(res1_d_final['weight'].mean(), color='red')
plt.show()
# the median is much closer to q1 than q3, and the left-side whisker is much shorter than the right-side whisker
# this shows that most values are concentrated at the left-side end of the distribution, yet there are some 
# (outliers_users.shape[0] in number) extremely high values that influence the mean of the distribution upwards

# thus, the distribution is far from normal, and the thresholds of the z-score method here are irrelevant
# So, the use of this method in this case doesn't bring up trustworthy resutls 



### frequency of tag usage 
q3_ = res1_b_final['count'].quantile(q=0.75)
q1_ = res1_b_final['count'].quantile(q=0.25)
iqr_ = q3_ - q1_ 
res1_b_final['IQR_meth'] = (res1_b_final['count'] >= q3_ + 1.5 * iqr_) | (res1_b_final['count'] <= q1_ - 1.5 * iqr_)

outliers_tags_IRQ = res1_b_final[res1_b_final['IQR_meth'] == True]
print(f"There are {outliers_tags_IRQ.shape[0]} outliers in the distribution of the frequency of usage for each tag.")# we see that this method gives many more outliers so it's much more effective,
outliers_tags_IRQ.to_csv('IQR-outliers_tags.csv', sep=',', header=True, index=False)

res1_b_final.boxplot('count', vert=False, showfliers=False)
plt.title('Distribution of Total Usage Count for Tags (not showing outliers)')
plt.axvline(res1_b_final['count'].mean(), color='red')
plt.show()
# the median is much closer to q1 than q3, and the left-side whisker is much shorter than the right-side whisker
# this shows that most values are concentrated at the left-side end of the distribution, yet there are some 
# (outliers_users.shape[0] in number) extremely high values that influence the mean of the distribution upwards

# thus, the distribution is far from normal, and the thresholds of the z-score method here are irrelevant
# So, the use of this method in this case doesn't bring up trustworthy resutls 




##################################### Q2 Part 1 ###############################################

#Read the data and store to dataframe
df = pd.read_csv('user_artists.dat', sep='\t', encoding=None, names=None, header=0, index_col=None)


# Create a pivot table without filling missing values
pivot_table = df.pivot_table(index='userID', columns='artistID', values='weight')

# Fill missing values with 0
pivot_table.fillna(0, inplace=True)

# Calculate cosine similarity directly using dot product and np.linalg.norm
cosine_sim = pivot_table.dot(pivot_table.T) / (np.linalg.norm(pivot_table, axis=1)[:, None] * np.linalg.norm(pivot_table.T, axis=0))


######################################Q2 Part 1 using  sklearn#########################################

# Calculate cosine similarity directly on the pivot table
#cosine_sim = cosine_similarity(pivot_table)
#######################################################################################################



# Create a DataFrame from the similarity matrix
cosine_sim_df = pd.DataFrame(cosine_sim, index=pivot_table.index, columns=pivot_table.index)

cosine_sim_df_with_id = cosine_sim_df.copy()
cosine_sim_df_with_id.insert(0, 'user_id', cosine_sim_df.index)

# Save the cosine similarity matrix with user IDs to a CSV file
cosine_sim_df_with_id.to_csv('user-pairs-similarity.data', index=False)


##################################### Q2 Part 2 ###############################################

# Function to find k-nearest neighbors for a user
def find_neighbors(user_id, k):
    # Get the user's similarity scores with all other users
    user_similarities = cosine_sim_df.loc[user_id]

    # Sort the users based on similarity and select the top k (excluding the user itself)
    neighbors = user_similarities.sort_values(ascending=False).index[1:k+1].tolist()

    return neighbors

# Dictionary to store neighbors for each user
neighbors_dict = {}

# Find and store neighbors for each user for k=3 and k=10
for user_id in pivot_table.index:
    neighbors_dict[user_id] = {
        'k=3': find_neighbors(user_id, 3),
        'k=10': find_neighbors(user_id, 10),
    }

# Save the neighbors dictionary to a JSON file
with open('neighbors-k-users.data', 'w') as json_file:
    json.dump(neighbors_dict, json_file, indent=2)


########## Providing some Barplots calculating the average similarity for k=3 neighbors #####################
cosine_sim_df = pd.read_csv('user-pairs-similarity.data', index_col='user_id') 

#Changing both user_id instancves (in the excel of cosine and the k-nearest neighbors JSON file to string in ordder to compare)
cosine_sim_df.index = cosine_sim_df.index.astype(str)
cosine_sim_df.columns = cosine_sim_df.columns.astype(str)

with open('neighbors-k-users.data', 'r') as json_file:
    neighbors_data = json.load(json_file)

numeric_to_string_mapping = {str(user_id): user_id for user_id in cosine_sim_df.index}

# Function to calculate the average similarity for a user's neighbors
def average_similarity(user_id, neighbors):
    try:
        # Convert user_id and neighbors to strings
        user_id_str = str(user_id)
        neighbors_str = list(map(str, neighbors))
        
        # Ensure neighbors_str is a list of strings
        neighbors_str = list(map(str, neighbors_str))

        # Check if the user_id is in the index of cosine_sim_df
        if user_id_str in cosine_sim_df.index and all(neighbors_user_str in cosine_sim_df.index for neighbors_user_str in neighbors_str):
            return cosine_sim_df.loc[user_id_str, neighbors_str].mean()
        else:
            return None
    except Exception as e:
        print(e)
        return None
    
# Calculate average similarity for each user with their k=3 neighbors
user_avg_similarity_k3 = {user_id: average_similarity(user_id, neighbors_data[user_id]['k=3'])
                       for user_id in neighbors_data}

# Filter out users where average similarity is None
user_avg_similarity_k3 = {k: v for k, v in user_avg_similarity_k3.items() if v is not None}


#Find top ten users based on their average similarity
top_ten_users = sorted(user_avg_similarity_k3.items(), key=lambda x: x[1], reverse=True)[:10]

#print top ten users_id as well as their average similarity with their 3 neighbors
for user_id, avg_similarity in top_ten_users:
    print(f"User {user_id}: Average Similarity with k=3 Neighbors = {avg_similarity}")
    
user_ids, avg_similarities = zip(*top_ten_users)

#create the barplots in logarithmic form (in normal form the difference is minimal and is not properly displayed in the bar chart)
plt.bar(user_ids, avg_similarities, color='blue')
plt.xlabel('User IDs')
plt.ylabel('Average Similarity (Log Scale)')
plt.title('Top Ten Users Based on Average Similarity')
plt.yscale('log')  # Set the y-axis to a logarithmic scale
plt.show()




######Same for k=10###################
user_avg_similarity_k10 = {user_id: average_similarity(user_id, neighbors_data[user_id]['k=10'])
                       for user_id in neighbors_data}

# Filter out users where average similarity is None
user_avg_similarity_k10 = {k: v for k, v in user_avg_similarity_k10.items() if v is not None}


#Find top ten users based on their average similarity
top_ten_users_10 = sorted(user_avg_similarity_k10.items(), key=lambda x: x[1], reverse=True)[:10]

#print top ten users_id as well as their average similarity with their 10 neighbors
for user_id, avg_similarity in top_ten_users_10:
    print(f"User {user_id}: Average Similarity with k=10 Neighbors = {avg_similarity}")
    
user_ids, avg_similarities = zip(*top_ten_users_10)

#create the barplots in logarithmic form (in normal form the difference is minimal and is not properly displayed in the bar chart)
plt.bar(user_ids, avg_similarities, color='blue')
plt.xlabel('User IDs')
plt.ylabel('Average Similarity (Log Scale)')
plt.title('Top Ten Users Based on Average Similarity with their k neighbors (k=10)')
plt.yscale('log')  # Set the y-axis to a logarithmic scale
plt.show()





################################ Q2 Part 1 Barplot ##########################################
# Calculate average similarity for each user with all other users
user_avg_similarity_all = {user_id: average_similarity(user_id, cosine_sim_df.index)
                            for user_id in cosine_sim_df.index}

# Filter out users where average similarity is None
user_avg_similarity_all = {k: v for k, v in user_avg_similarity_all.items() if v is not None}

# Find top ten users based on their average similarity with all other users
top_ten_users_all = sorted(user_avg_similarity_all.items(), key=lambda x: x[1], reverse=True)[:10]

# Print top ten users_id as well as their average similarity with all other users
for user_id, avg_similarity in top_ten_users_all:
    print(f"User {user_id}: Average Cosine Similarity with All Users = {avg_similarity}")

# Create a bar plot for the top ten users with all other users
user_ids_all, avg_similarities_all = zip(*top_ten_users_all)

max_similarity_user_all = max(user_avg_similarity_all, key=user_avg_similarity_all.get)
print(f"\nUser with Highest Average Cosine Similarity: {max_similarity_user_all}, Average Similarity Value: {user_avg_similarity_all[max_similarity_user_all]}")

# Find and print the user with the lowest average similarity
min_similarity_user_all = min(user_avg_similarity_all, key=user_avg_similarity_all.get)
print(f"User with Lowest Average Cosine Similarity: {min_similarity_user_all}, Average Similarity Value: {user_avg_similarity_all[min_similarity_user_all]}")

# Calculate and print the mean average similarity for all users
mean_avg_similarity_all = sum(user_avg_similarity_all.values()) / len(user_avg_similarity_all)
print(f"Mean Average Cosine Similarity for All Users: {mean_avg_similarity_all}")

plt.bar(user_ids_all, avg_similarities_all, color='green')
plt.xlabel('User IDs')
plt.ylabel('Average Similarity')
plt.title('Top Ten Users Based on Average Similarity (All Users)')
plt.show() 






###################################################################
print("****** Q3 ******\n\n")
print(f"The first 5 rows of the dataframe containing the data used in the following analysis have as follows:\n\n {user_tags_ts.head()}")

# divide the 'timestmap' column's values by 1000 so as to get timestamp expressed in seconds and not milliseconds
user_tags_ts['timestamp'] = (user_tags_ts['timestamp'] / 1000).astype('int64')
dates = [datetime.fromtimestamp(ts) for ts in user_tags_ts['timestamp']]
user_tags_ts['date_'] = dates

# Creating a 'quarters' column in the df, in which we will have a tuple of the form (year, quarter), so that
# it is sortable. 
quarters = [(t.year, t.quarter) for t in user_tags_ts['date_']]
user_tags_ts['quarters'] = quarters

# this I do in the previous question (q3-a) as well, and I group by this column


print(f"We added a 'quarters' column containing tuples representing the year and quarter an observation falls in.\nNow, the first 5 rows of the dataframe have as follows:\n\n {user_tags_ts.head()}")

# Checking for wrong timestamp values based on the 'quarters' column; we cannot have datapoints corresponding to 
# time prior to 2005, as the platform under study did not exist back then! 
print(f"The unique 'quarter' values in our dataset are the following:/n/n {user_tags_ts['quarters'].unique()}")

print("We see that there are observations going back as much as 1956, 1957, and 1979. This is evidence of measurement error and so we proceed to drop the following rows.\n\n") 
wrong = user_tags_ts[user_tags_ts['quarters'].isin([(1956, 3), (1956, 2), (1979, 2), (1957, 1)])]
print(wrong)

# dropping the problematic rows & sorting the dataframe in ascending order by 'quarters'
user_tags_ts.drop(wrong.index, axis=0, inplace=True)
user_tags_ts.sort_values(by='quarters', inplace=True)



print("****** Q3 - a ******\n\n")
# I will find the distinct count of occurence of each item (user/artist/tag) per interval - I select this interval to be a quarter of the year

print(f"There are {len(user_tags_ts['quarters'].unique())} intervals (quarters) in the time period under study.")

res3_users = user_tags_ts['userID'].groupby([user_tags_ts['date_'].dt.year, user_tags_ts['date_'].dt.quarter]).nunique()
res3_users.to_csv('Q3a_users.csv', sep=',', header=True, index=False)  # writing the results to a csv file
res3_users.plot(kind='line')
plt.title('Users Active per Quarter')
plt.xlabel('Quarter')
plt.ylabel('(Unique) Users Count')
plt.show()
# There was a clear uptrend from the beginning of the platform up to the third quarter of 2010, which 
# coincided with the great expansion of internet usage in general, and specifically social media. Then there was a 
# sudden drop, which coincided with the removal of many features central to the platform's offering, and there also was 
# a redesign of its look that was welcomed with hostility by many users; namely, after its freely offerred features like
# usings its radio streaming service were limited to subscription payers in some countries, growing a lot of content 
# among the platform's user base.


res3_tags = user_tags_ts['tagID'].groupby([user_tags_ts['date_'].dt.year, user_tags_ts['date_'].dt.quarter]).nunique()
res3_tags.to_csv('Q3a_tags.csv', sep=',', header=True, index=False)  # writing the results to a csv file
res3_tags.plot(kind='line')
plt.title('Number of (Unique) Tags Used per Quarter')
plt.xlabel('Quarter')
plt.ylabel('(Unique) Tags Count')
plt.show()
# The fall in tag usage coincides with the drop in the number of (unique) active users.


res3_artists = user_tags_ts['artistID'].groupby([user_tags_ts['date_'].dt.year, user_tags_ts['date_'].dt.quarter]).nunique()
res3_artists.to_csv('Q3a_artists.csv', sep=',', header=True, index=False)  # writing the results to a csv file
res3_artists.plot(kind='line')
plt.title('Number of (Unique) Artists Listened to per Quarter')
plt.xlabel('Quarter')
plt.ylabel('(Unique) Artists Listened to Count')
plt.show()
# Drop in streaming activity closely following the general reduction of the user base of the platform. 



print("****** Q3 - b ******\n\n")

# example of how we approach the problem of finding the top 5 most prevalent artists in a given quarter
# for a given quarter, we find all corresponding rows/records of the dataframe; then, we count the no of occurrence of 
# each artist; finally, we sort that resulting series in descending order by the no of occurrences, and select only the top 
# 5 artists
# temp = user_tags_ts['artistID'][user_tags_ts['quarters'] == user_tags_ts['quarters'].unique()[0]].groupby(user_tags_ts['artistID']).count()
# temp.sort_values(ascending=False)[:5]
# the series has artistID as indexes and their count of occurrence as values

# It might be the case for some quarters that there were less than 5 artists that got assigned a tag; in that case, 
# we'll create a series of length 5, filling in the blanks with NaN values.

# We add the (quarter, list_of_top_5_artists) pairs in a dedicated list, called 'top_artists'.

# Then, we want to extract 3 lists out of the 'top_artists' pd.Series object; the quarters (5 times each), the ids of the top 5 artists in each 
# quarter, and the frequencies of those artists in each quarter.

# We build a dataframe using those lists, and drop rows with NaN values (which correspond to quarters in which we didn't have
# tags assigned to as many as 5 artists)

# The final result is a dataframe showing each quarter, and then, in descending order (by their frequency) it shows 
# the top artistID values and their corresponding freqs

def top_list(item_considered, working_df): 
    top_list = [] 
    for i in range(len(working_df['quarters'].unique())):
        q = working_df['quarters'].unique()[i]  # the df is sorted in ascending order based on this column, so the corresponding values will be considered in ascending order as well
        fin = working_df[item_considered][working_df['quarters'] == q].groupby(user_tags_ts[item_considered]).count().sort_values(ascending=False)[:5]
        
        if len(fin) < 5:
            fin += [np.nan] * (5 - len(fin))        
    
        top_list.append([q, fin])
    
    
    top_5_ids = [] 
    top_5_freqs = [] 
    qqs = []  # quarters considered
    for sublist in top_list:
        for i in range(len(sublist[1].values)):
            qqs.append(sublist[0])
            top_5_ids.append(sublist[1].index[i])
            top_5_freqs.append(sublist[1].values[i])       
        
    res_df = pd.DataFrame([qqs, top_5_ids, top_5_freqs]).T
    res_df.columns = ['quarter', 'id', 'freq']
    res_df.sort_values(by=['quarter', 'freq'], ascending=[True,False])
    
    return res_df 


res_artists = top_list('artistID', user_tags_ts)
res_artists.to_csv('Q3b_artists.csv', sep=',', header=True, index=False)  # writing the results to a csv file
print(f"The first 5 rows of the dataframe containing details on the 5 most listened to artists per quarter have as follows:\n\n {res_artists.head()}")

res_tags = top_list('tagID', user_tags_ts)
res_tags.to_csv('Q3b_tags.csv', sep=',', header=True, index=False)  # writing the results to a csv file
print(f"Similarly for the 5 most frequently used tags per quarter:\n\n {res_tags.head()}")

# We see that in all of the quarters there exist at least 5 unique artists that have been assigned at least one tag. 
print(f"There are {res_artists['id'].isna().sum()} NaN artists in the dataframe, meaning that in each quarter there were at least 5 unique artists being assigned at least one tag.")


print(f"There are {res_tags['id'].isna().sum()} NaN tags in the dataframe, meaning that in each quarter there were at least 5 unique tags being used.")



print("****** Studying the Frequency in which the most Famous Artists show up at the top of Quarterly Rankings ******")
# The dataframe 'res_artists' is sorted in ascending order based on the 'quarter' column, and in descending order based on the 'freq' for the top
# Artists in each quarter - if we drop duplicates, we will only keep the artists most listened to in each quarter; 

# We created a dictionary using the resulting dataframe, in order to get the number that each of those top artists was seen at the 
# top of a quarter:
top_artists = dict(res_artists.drop_duplicates('quarter')['id'].value_counts().sort_values(ascending=False))
print(f"The top artist id - frequency pairs are the following: {top_artists}")

# We turned this into a df and merged it with the 'artists_info' df to get their names
top_artists_full = pd.merge(pd.DataFrame(zip(top_artists.keys(), top_artists.values()),
            columns=['artistID', 'count']), artist_info[['id','name']], left_on='artistID', right_on='id', how='inner')
top_artists_full.drop('id', axis=1, inplace=True)

# Writing the results to a .csv file to include in our deliverables. 
top_artists_full.to_csv('dominating_artists.csv', sep=',', header=True, index=False)
print(f"The 5 artists figuring at the top of quarterly rankings the most are the following:\n\n {top_artists_full.head()}")



print(f"We also performed the same analysis from the most widely used tags.")
top_tags = dict(res_tags.drop_duplicates('quarter')['id'].value_counts().sort_values(ascending=False))
print(f"The top tag id - frequency pairs are the following: {top_tags}")

print("We see that a tag dominates in 21 out of the 24 intervals (i.e. 87.5% of them) under study. It is then meaningful to also get the value of each tag to try and extract some useful observations out of them.") 

# I'll turn this into a df and merge it with the 'artists_info' df to get their names
top_tags_full = pd.merge(pd.DataFrame(zip(top_tags.keys(), top_tags.values()),
            columns=['tagID', 'count']), tag_info, on='tagID', how='inner')
top_tags_full.to_csv('dominating_tags.csv', sep=',', header=True, index=False)
print(f"The tags figuring most often at the top of quarterly rankings have as follows:\n\n {top_tags_full}")

# We see that there's no unique artist dominating in the quarterly rankings, and those that have appeared on the top of them are 
# mixed among the musical genres; 2 of them are pop musicians (Christina Aguilera, Britney Spears) and the other 2 (Depeche Mode, Blood Ruby) venture mainly in the rock genre.


# However, the tag that has dominated the most, with an astounding difference from the rest, has to do with rock music, and other 2 of those dominating tags also have to do
# with genres other than pop (college rock, alternative). 

# This tells us that, even though pop artists have extremely loyal fans that stream their songs much more than fans of other 
# music genres (as Figure 1 (link to it) shows us), the rock fans user cohort of last.fm is far more active, interacting with 
# the profiles of their loved artists (by applying them tags) much more than pop fans do. 






###################################################################

#Creating the dataframes for friends and users AND for artists and users
df_artists= pd.read_csv('user_artists.dat', sep='\t', encoding=None, names=None, header=0, index_col=None)
df_friends= pd.read_csv('user_friends.dat', sep='\t', encoding=None, names=None, header=0, index_col=None)


# Dropping weigth column from df_artists
df_artist_dropped_weight=df_artists.drop(columns=['weight'])

#creating dataframe holding the count of artists for each user
df_artists_count=df_artist_dropped_weight.groupby('userID').count().rename(columns={'artistID':'count_of_artists'})


#creating dataframe holding the count of friends for each user
df_friends_count=df_friends.groupby('userID').count().rename(columns={'friendID':'count_of_friends'})

# Merge dataframes based on 'userID'
merged_df = pd.merge(df_friends_count, df_artists_count, on='userID')



######################################### Part a #################################################


#calculate correlation between two columns  using corr() function
corr = merged_df["count_of_friends"].corr(merged_df["count_of_artists"])
corr2=merged_df.corr(method='pearson')

#plotting the scatterplot for the correlation
sns.lmplot(x="count_of_artists", y="count_of_friends", data=merged_df);

## plotting the correlation matrix
plt.figure(figsize=(10,10))
plt.show()
sns.heatmap(corr2, cmap="Greens",annot=True)

print("\n\n################## Part a ##########################\n\n")
print(merged_df)
print("\n Correlation Part A (pearson): ",corr)
print("\n Correlation Matrix Part A (pearson): \n", corr2)

######################################### Part a Alegebraic #################################################


# Extract the two columns
x = merged_df['count_of_friends']
y = merged_df['count_of_artists']

# Calculate mean of each column
mean_x = x.mean()
mean_y = y.mean()

# Calculate the numerator and denominators
numerator = sum((x_i - mean_x) * (y_i - mean_y) for x_i, y_i in zip(x, y))
denominator_x = sum((x_i - mean_x) ** 2 for x_i in x)
denominator_y = sum((y_i - mean_y) ** 2 for y_i in y)

correlation = numerator / (denominator_x**0.5 * denominator_y**0.5)
print(f"\n Alegebraically calculated Correlation for part A between count_of_friends and count_of_artists: {correlation}")


######################################### Part b #################################################

#Dropping the artistID , hold only the weight column which represents the number of streams/listenings
df_artist_dropped_artistsID=df_artists.drop(columns=['artistID'])

#creating dataframe with user id and the SUM of all their streams/listening instances
df_listening_count=df_artist_dropped_artistsID.groupby('userID').sum().rename(columns={'weight':'sum_of_listening'})

# merge the two dataframes (one holding the count of friends, one holding the sum of weights)
merged_df_part_b = pd.merge(df_friends_count, df_listening_count, on='userID')

#calculate correlation between the two columns using the corr() function
corr3 = merged_df_part_b["sum_of_listening"].corr(merged_df_part_b["count_of_friends"])
corr4=merged_df_part_b.corr(method='pearson')

#plotting the scatterplot for the correlation
sns.lmplot(x="count_of_friends", y="sum_of_listening", data=merged_df_part_b);

#plotting the correlation matrix
plt.figure(figsize=(14,10))
sns.heatmap(corr4, cmap="Blues",annot=True)
plt.show()

print("\n\n#################### Part b #############################\n\n")
print(merged_df_part_b)
print("\n Correlation part B (pearson): ",corr3)
print("\n Correlation Matrix Part B (pearson): \n", corr4)


######################################### Part b Alegebraic #################################################

# Extract the two columns
x1 = merged_df_part_b['count_of_friends']
y1 = merged_df_part_b['sum_of_listening']

# Calculate mean of each column
mean_x1 = x1.mean()
mean_y1 = y1.mean()

# Calculate the numerator and denominators
numerator1 = sum((x_i - mean_x1) * (y_i - mean_y1) for x_i, y_i in zip(x1, y1))
denominator_x1 = sum((x_i - mean_x1) ** 2 for x_i in x1)
denominator_y1 = sum((y_i - mean_y1) ** 2 for y_i in y1)

correlation1 = numerator1 / (denominator_x1**0.5 * denominator_y1**0.5)
print(f"\n Alegebraically calculated Correlation for part B between count_of_friends and sum_of_listening: {correlation1}")
