import pandas as pd

#dataset is in Gdrive
#importing as two parts and merging

data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/TweetsCOV19.tsv',sep = '\t',skiprows = list(range(4467891,8935782)),usecols=range(11),header=None)

data2 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/TweetsCOV19.tsv',sep = '\t',skiprows = list(range(4467891)),usecols=range(11),header=None)

frame = pd.concat([data,data2])
