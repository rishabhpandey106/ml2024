import numpy as np
import pandas as pd

df = pd.read_csv('sms_spam_classifier\spam.csv', encoding='latin-1')

# data cleaning
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder
df['v1'] = LabelEncoder().fit_transform(df['v1'])


df = df.drop_duplicates(keep='first')

# #eda

import nltk 

df['v2_len'] = df['v2'].apply(len)

df['v2_words_count'] = df['v2'].apply(lambda x:len(nltk.word_tokenize(x)))

df['v2_sentence_count'] = df['v2'].apply(lambda x:len(nltk.sent_tokenize(x)))

# print(df.head())

import seaborn as sns
import matplotlib.pyplot as plt

# sns.histplot(df['v2_len'][df['v1'] == 0], color='red')
# sns.histplot(df['v2_len'][df['v1'] == 1], color='green')
# plt.show()


import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    
    text = y[:]
    y.clear()
    return " ".join(text)

df['transformed_text'] = df['v2'].apply(transform)
print(df.head())

# from wordcloud import WordCloud

# wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

# spam = wc.generate(df[df['v1'] == 1]['transformed_text'].str.cat(sep=" "))
# ham = wc.generate(df[df['v1'] == 0]['transformed_text'].str.cat(sep=" "))
# plt.figure(figsize=(10,10))
# plt.imshow(spam)
# plt.imshow(ham)
# plt.show()

# spam_corpus = []
# for msg in df[df['v1'] == 1]['transformed_text'].tolist():
#     for word in msg.split():
#         spam_corpus.append(word)

# from collections import Counter

# sns.barplot(x=pd.DataFrame(Counter(spam_corpus).most_common(30))[0],y=pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
# plt.xticks(rotation='vertical')
# plt.show()


# Assuming 'df' is your modified DataFrame
df.to_csv('sms_spam_classifier/new_spam.csv', index=False)

