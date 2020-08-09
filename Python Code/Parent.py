import Preprocessor
import pandas as pd
from algoSelection import chi_square,select_k_best,correlation,classification_compare
import HyperTuning
from HyperTuning import ensembler,Hyper_Tuning
'''
frame = pd.concat([pd.read_csv('/content/drive/My Drive/Colab Notebooks/TweetsCOV19.tsv',sep = '\t',skiprows = list(range(4467891,8935782)),usecols=range(11),header=None),pd.read_csv('/content/drive/My Drive/Colab Notebooks/TweetsCOV19.tsv',sep = '\t',skiprows = list(range(4467891)),usecols=range(11),header=None)])
frame.columns = ['Tweet Id', 'Username', 'Timestamp', 'Followers', 'Friends', 'Retweets', 'Favorites', 'Entities', 'Sentiment', 'Mentions', 'Hashtags']

train, test = Preprocessor.Preprocess(frame)

#Processed data
Processed_frame = pd.read_csv('/content/drive/My Drive/Colab Notebooks/data.csv')
Preprocessor.Undersampling(frame)
'''
#############################################################################################
train_set=pd.read_csv('train.csv')
test_set=pd.read_csv('test.csv')

train_set.drop(['Tweet Id'],axis=1,inplace=True)
test_set.drop(['Tweet Id'],axis=1,inplace=True)

#feature selection
# chi_square(train_set)
# correlation(train_set)

#algo selection
# classification_compare(train_set,test_set)

#hyper-tuning and prediction
Hyper_Tuning(train_set, test_set)
