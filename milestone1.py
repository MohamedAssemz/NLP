import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from bs4 import BeautifulSoup
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Exploratory Data Analysis EDA

# Importing data
data_train = pd.read_csv('AVI Dataset/train.csv.zip')

# Displaying first few rows
print("Dimensions of the dataset:", data_train.shape)
print("First few rows of the dataset:")
print(data_train.head())

# Check data types and null values
print("Data types and null values:")
print(data_train.info())

# Summary statistics for numerical columns
print("Summary statistics for numerical columns:")
print(data_train.describe())

# Distribution of labels
label_distribution = data_train['Label'].value_counts()
print("Distribution of labels:")
print(label_distribution)

# Pie chart of label distribution
plt.figure(figsize=(6, 6))
plt.pie(label_distribution, labels=label_distribution.index, autopct='%1.1f%%')
plt.title('Distribution of Labels')
plt.show()

# Check for duplicates
duplicate_rows = data_train[data_train.duplicated()]
print("Number of duplicate rows:", len(duplicate_rows))

# Histogram of 'Score' column
plt.hist(data_train['Score'], bins=20, range=(data_train['Score'].min(), data_train['Score'].max()))
plt.title('Distribution of Score')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()

# Remove outliers based on IQR
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

plt.figure(figsize=(8, 6))
sns.boxplot(x=data_train['Score'])
plt.title('Distribution of Score (Before Removing Outliers)')
plt.xlabel('Score')
plt.show()

data_no_outliers = remove_outliers_iqr(data_train, 'Score')

plt.figure(figsize=(8, 6))
sns.boxplot(x=data_no_outliers['Score'])
plt.title('Distribution of Score (After Removing Outliers)')
plt.xlabel('Score')
plt.show()

# Visualize distribution of 'Score' after removing outliers
plt.figure(figsize=(8, 6))
sns.histplot(data_no_outliers['Score'], kde=True)
plt.title('Distribution of Score (After Removing Outliers)')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()

# Compute text length
data_train['Title_Length'] = data_train['Title'].apply(lambda x: len(x.split()))
data_train['Body_Length'] = data_train['Body'].apply(lambda x: len(x.split()))

print("Summary statistics for text length:")
print(data_train[['Title_Length', 'Body_Length']].describe())

# Word Cloud for Title Column
titles_text = ' '.join(data_train['Title'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(titles_text)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Title Column')
plt.axis('off')
plt.show()

# Histogram and Scatter plot using Plotly
fig_histogram = px.histogram(data_train, x='Score', range_x=[-5, 50], color='Label')
fig_histogram.show()

fig_scatter = px.scatter(data_train, x='Score', y='ViewCount', color='Label', log_y=True, marginal_y='box', range_x=[-20, 200])
fig_scatter.show()

# Textual Analysis

# Generate word clouds for Android and iOS titles
android_data = data_train[data_train['Label'] == 'android']
ios_data = data_train[data_train['Label'] == 'ios']
android_titles_text = ' '.join(android_data['Title'])
ios_titles_text = ' '.join(ios_data['Title'])

android_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(android_titles_text)
ios_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(ios_titles_text)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(android_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Android Titles')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(ios_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for iOS Titles')
plt.axis('off')

plt.show()

# Apply LDA to Title column
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(data_train['Title'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic {topic_idx+1}:")
    print(" ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]))
    print()

# Sentiment Analysis
data_train['Title_Sentiment'] = data_train['Title'].apply(lambda x: TextBlob(x).sentiment.polarity)

plt.figure(figsize=(8, 6))
sns.histplot(data_train['Title_Sentiment'], bins=20)
plt.title('Distribution of Title Sentiment Polarity')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.show()

# Data Preprocessing

# Lowercasing
data_train['Title'] = data_train['Title'].str.lower()
data_train['Body'] = data_train['Body'].str.lower()

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_and_tokenize(text):
    cleaned_text = BeautifulSoup(text, "html.parser").get_text(separator=" ")

    tokens = []
    current_word = ""
    for char in cleaned_text:
        if char.isalnum() or char in ["'", ".", "-"]:
            current_word += char
        else:
            if current_word:
                tokens.append(current_word)
                current_word = ""
            if char.strip():
                tokens.append(char)
    if current_word:
        tokens.append(current_word)
    return tokens

data_train['Title_tokens'] = data_train['Title'].apply(preprocess_and_tokenize)
data_train['Body_tokens'] = data_train['Body'].apply(preprocess_and_tokenize)

# Removing Punctuation
data_train['Title_tokens'] = data_train['Title_tokens'].apply(lambda x: [word for word in x if word not in string.punctuation])
data_train['Body_tokens'] = data_train['Body_tokens'].apply(lambda x: [word for word in x if word not in string.punctuation])

# Removing Stopwords
stop_words = set(stopwords.words('english'))
data_train['Title_tokens'] = data_train['Title_tokens'].apply(lambda x: [word for word in x if word not in stop_words])
data_train['Body_tokens'] = data_train['Body_tokens'].apply(lambda x: [word for word in x if word not in stop_words])

# Lemmatization
lemmatizer = WordNetLemmatizer()
data_train['Title_tokens'] = data_train['Title_tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
data_train['Body_tokens'] = data_train['Body_tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Convert tokens back to strings
data_train['Title_cleaned'] = data_train['Title_tokens'].apply(lambda x: ' '.join(x))
data_train['Body_cleaned'] = data_train['Body_tokens'].apply(lambda x: ' '.join(x))

# Word Frequency Analysis
title_words = ' '.join(data_train['Title_cleaned']).split()
body_words = ' '.join(data_train['Body_cleaned']).split()

title_word_freq = Counter(title_words)
body_word_freq = Counter(body_words)

print("Top 20 frequent words in title:")
print(title_word_freq.most_common(20))
print("\nTop 20 frequent words in body:")
print(body_word_freq.most_common(20))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.barplot(x=[pair[0] for pair in title_word_freq.most_common(20)], y=[pair[1] for pair in title_word_freq.most_common(20)], ax=axes[0])
axes[0].set_title('Top 20 Frequent Words in Title')
axes[0].set_xlabel('Words')
axes[0].set_ylabel('Frequency')

sns.barplot(x=[pair[0] for pair in body_word_freq.most_common(20)], y=[pair[1] for pair in body_word_freq.most_common(20)], ax=axes[1])
axes[1].set_title('Top 20 Frequent Words in Body')
axes[1].set_xlabel('Words')
axes[1].set_ylabel('Frequency')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
