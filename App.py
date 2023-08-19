import pandas as pd
import random as r
import re
import nltk
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import seaborn as sns
from wordcloud import WordCloud

data_df = pd.read_csv('train.csv')
data_df.head()

data_df.info()

print("Random Tweets:\n")
print(data_df['tweet'].iloc[r.randint(1,31962)],"\n")
print(data_df['tweet'].iloc[r.randint(1,31962)],"\n")
print(data_df['tweet'].iloc[r.randint(1,31962)],"\n")
print(data_df['tweet'].iloc[r.randint(1,31962)],"\n")
print(data_df['tweet'].iloc[r.randint(1,31962)],"\n")

def data_processing(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"https\S+|www\S+http\S+", '', tweet, flags = re.MULTILINE)
    tweet = re.sub(r'\@w+|\#','', tweet)
    tweet = re.sub(r'[^\w\s]','',tweet)
    tweet = re.sub(r'ð','',tweet)
    tweet_tokens = word_tokenize(tweet)
    filtered_tweets = [w for w in tweet_tokens if not w in stop_words]
    return " ".join(filtered_tweets)

data_df['tweet'] = data_df['tweet'].apply(data_processing)

data_df = data_df.drop_duplicates('tweet')

lemmatizer = WordNetLemmatizer()
def lemmatizing(data):
    tweet = [lemmatizer.lemmatize(word) for word in data]
    return data

data_df['tweet'] = data_df['tweet'].apply(lambda x: lemmatizing(x))

print("Processed Tweets:\n")
print(data_df['tweet'].iloc[0],"\n")
print(data_df['tweet'].iloc[1],"\n")
print(data_df['tweet'].iloc[2],"\n")
print(data_df['tweet'].iloc[3],"\n")
print(data_df['tweet'].iloc[4],"\n")

data_df.info()

data_df['label'].value_counts()


fig = plt.figure(figsize=(5,5))
sns.countplot(x='label', data = data_df)

fig = plt.figure(figsize=(7,7))
colors = ("red", "gold")
wp = {'linewidth':2, 'edgecolor':"black"}
tags = data_df['label'].value_counts()
explode = (0.1, 0.1)
tags.plot(kind='pie',autopct = '%1.1f%%', shadow=True, colors = colors, startangle =90, 
         wedgeprops = wp, explode = explode, label='')
plt.title('Distribution of sentiments')

non_hate_tweets = data_df[data_df.label == 0]
non_hate_tweets.head()

text = ' '.join([word for word in non_hate_tweets['tweet']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in non hate tweets', fontsize = 20)

neg_tweets = data_df[data_df.label == 1]
neg_tweets.head()

text = ' '.join([word for word in neg_tweets['tweet']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in hate tweets', fontsize = 20)

plt.show()