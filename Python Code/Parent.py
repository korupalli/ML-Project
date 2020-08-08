import Preprocessor
import pandas as pd

frame = pd.concat([pd.read_csv('/content/drive/My Drive/Colab Notebooks/TweetsCOV19.tsv',sep = '\t',skiprows = list(range(4467891,8935782)),usecols=range(11),header=None),pd.read_csv('/content/drive/My Drive/Colab Notebooks/TweetsCOV19.tsv',sep = '\t',skiprows = list(range(4467891)),usecols=range(11),header=None)])
frame.columns = ['Tweet Id', 'Username', 'Timestamp', 'Followers', 'Friends', 'Retweets', 'Favorites', 'Entities', 'Sentiment', 'Mentions', 'Hashtags']

train, test = Preprocessor.Preprocess(frame)

#Processed data
Processed_frame = pd.read_csv('/content/drive/My Drive/Colab Notebooks/data.csv')
Preprocessor.Undersampling(frame)

