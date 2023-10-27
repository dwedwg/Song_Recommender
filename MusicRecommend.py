import pandas as pd
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


df=pd.read_csv("data.csv")

feature_cols=['acousticness', 'danceability', 'duration_ms', 'energy',
              'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
              'speechiness', 'tempo', 'time_signature', 'valence',]

# output graphs for each feature col
def show_graphs(feature_cols):
    for category in feature_cols:
        feature = df[category]
        plt.hist(feature, bins = 50)
        plt.title(str(category))
        plt.show()

#normalize dataframe
scaler = MinMaxScaler()
normalized_df =scaler.fit_transform(df[feature_cols])

# Create a pandas series with song titles as indices and indices as series values 
indices = pd.Series(df.index, index=df['song_title']).drop_duplicates()

# Create cosine similarity matrix based on given matrix
cosine = cosine_similarity(normalized_df)

def generate_recommendation(song_title, model_type=cosine ):
    """
    Purpose: Function for song recommendations 
    Inputs: song title and type of similarity model
    Output: Pandas series of recommended songs
    """
    # Get song indices
    index=indices[song_title]
    # Get list of songs for given songs
    score=list(enumerate(model_type[indices[song_title]]))
    # Sort the most similar songs
    similarity_score = sorted(score,key = lambda x:x[1],reverse = True)
    # Select the top-10 recommend songs
    similarity_score = similarity_score[1:11]
    top_songs_index = [i[0] for i in similarity_score]
    # Top 10 recommende songs
    top_songs=df['song_title'].iloc[top_songs_index]

    return top_songs

    
if __name__ == "__main__":
    
    show_graphs(feature_cols)

    print("SONG RECOMMENDATION SYSTEM")
    print("SONG CHOICE HAS TO BE FROM DATA LIST AND IN THE CORRECT CASE")
    print("**************************************************************************")
    song_choice = input("Input Song name that is in the data list that you like:")

    #generate cosine recommendations
    print("Recommended Songs from cosine:")
    cosine_list = generate_recommendation(song_choice,cosine).values
    for i in range (len(cosine_list)):
        print( str(i+1) + ": " + cosine_list[i])
    
    #generate sigmoid recommendations
    sig_kernel = sigmoid_kernel(normalized_df)
    print("Recommended Songs from sigmoid :")
    sigmoid_list = generate_recommendation(song_choice,sig_kernel).values
    for i in range (len(sigmoid_list)):
        print( str(i+1) + ": " + cosine_list[i])
