# Natural Language Processing & Text Mining

'''
Natural Language Processing and Text Mining Steps:

    - Text Preprocessing
        - Upper/Lower case
        - Removing punctuation characters
        - Removing numbers
        - Removing stopwords
        - Removing rare words
        - Tokenization
        - Stemming / Lemmatization
    - NLP Applications
        - N-Grams
        - Part of speech tagging and visualization
        - Named entity resolution
    - Mathematical Operations and Simple Feature Exraction
        - Letter/Character Count
        - Word Count
        - Catching Numbers and Counting
        -Text Visualization
            - Term Frequency and Bar Plot
            - Word Cloud
    - Sentiment Analysis / Modeling
        - Text Preprocessing
        - Feature Engineering
            - Count Vectors
            - TF-IDF Vectors (words, characters, n-grams)
            - Word Embeddings (word2vec, bert, fasttext)
        # TF(t) = (Frequency of a t term in a document) / (total number of terms in the document)
        # IDF(t) = log_e(Total number of documents / number of documents with t term in them)
        - Machine Learning for Sentiment Analysis
'''

# Import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn import ensemble
import xgboost

import nltk
from nltk.corpus import stopwords

from textblob import TextBlob
from textblob.blob import Word
from textblob import TextBlob

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from warnings import filterwarnings
filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Text Preprocessing

text = """ 
A Scandal in Bohemia! 01 
The Red-headed League,2 
A Case, of Identity 33 
The Boscombe Valley Mystery4 
The Five Orange Pips1 
The Man with? the Twisted Lip 
The Adventure of the Blue Carbuncle 
The Adventure of the Speckled Band 
The Adventure of the Engineer's Thumb 
The Adventure of the Noble Bachelor 
The Adventure of the Beryl Coronet 
The Adventure of the Copper Beeches"""

text
text.split()

v_text = text.split("\n")
type(v_text)
v_text = v_text[1:len(v_text)]
v_text

# Create a dataframe
mdf = pd.DataFrame(v_text, columns=["Stories"])

# Upper/Lower Case
mdf["Stories"].apply(lambda x: " ".join(x.lower() for x in x.split()))
d_mdf = mdf["Stories"].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Removing punctuation characters
d_mdf = d_mdf.str.replace("[^\w\s]", "")

# Removing numbers
d_mdf = d_mdf.str.replace("\d", "")

# Removing stopwords
nltk.download("stopwords")
sw = stopwords.words("english")
sw

type(d_mdf)

d_mdf = pd.DataFrame(d_mdf, columns=["Stories"])
d_mdf = d_mdf["Stories"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

# Removing words with rare frequencies
d_mdf = pd.DataFrame(d_mdf, columns=["Stories"])
pd.Series(" ".join(d_mdf["Stories"]).split()).value_counts()
# Deleting a definite list
sil = pd.Series(" ".join(d_mdf["Stories"]).split()).value_counts()[-3:]
d_mdf["Stories"].apply(lambda x: " ".join(i for i in x.split() if i not in sil))


# TOKENIZATION
nltk.download("punkt")
TextBlob(d_mdf["Stories"][1]).words
d_mdf["Stories"].apply(lambda x: TextBlob(x).words)

# STEMMING & LEMMATIZATION
from nltk.stem import PorterStemmer
st = PorterStemmer()
d_mdf["Stories"].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

nltk.download("wordnet")
d_mdf["Stories"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# NLP APPLICATIONS

# N-Gram

a = """Bu örneği anlaşılabilmesi için daha uzun bir metin üzerinden göstereceğim. 
N-gram'lar birlikte kullanılan kelimelerin kombinasyolarını gösterir"""

a

TextBlob(a).ngrams(2)


# Part of speech tagging (POS)

nltk.download("averaged_perceptron_tagger")
TextBlob(d_mdf["Stories"][6]).tags
d_mdf["Stories"].apply(lambda x: TextBlob(x).tags)

pos = d_mdf["Stories"].apply(lambda x: TextBlob(x).tags)

sentence = "R and Python are useful data science tools for the new or old data scientists who eager to do efficent data science task"
pos = TextBlob(sentence).tags

# POS tag list:
# CC	coordinating conjunction
# CD	cardinal digit
# DT	determiner
# EX	existential there (like: "there is" ... think of it like "there exists")
# FW	foreign word
# IN	preposition/subordinating conjunction
# JJ	adjective	'big'
# JJR	adjective, comparative	'bigger'
# JJS	adjective, superlative	'biggest'
# LS	list marker	1)
# MD	modal	could, will
# NN	noun, singular 'desk'
# NNS	noun plural	'desks'
# NNP	proper noun, singular	'Harrison'
# NNPS	proper noun, plural	'Americans'
# PDT	predeterminer	'all the kids'
# POS	possessive ending	parent\'s
# PRP	personal pronoun	I, he, she
# PRP$	possessive pronoun	my, his, hers
# RB	adverb	very, silently,
# RBR	adverb, comparative	better
# RBS	adverb, superlative	best
# RP	particle	give up
# TO	to	go 'to' the store.
# UH	interjection	errrrrrrrm
# VB	verb, base form	take
# VBD	verb, past tense	took
# VBG	verb, gerund/present participle	taking
# VBN	verb, past participle	taken
# VBP	verb, sing. present, non-3d	take
# VBZ	verb, 3rd person sing. present	takes
# WDT	wh-determiner	which
# WP	wh-pronoun	who, what
# WP$	possessive wh-pronoun	whose
# WRB	wh-abverb	where, when

# Visualization of POS Tagging

reg_exp = "NP: {<DT>?<JJ>*<NN>}"
rp = nltk.RegexpParser(reg_exp)
results = rp.parse(pos)
print(results)
results.draw()


# Named Entity Recognition (NER)
from nltk import word_tokenize, pos_tag, ne_chunk
nltk.download('maxent_ne_chunker')
nltk.download('words')
sentence = "Sinan is a creative person who work for R Studio AND he attented conference at New york last year"
print(ne_chunk(pos_tag(word_tokenize(sentence))))


# Mathematical Operations and Simple Feature Exraction

# Letter/Character Count

d_mdf["Stories"].str.len()
d_mdf["letter_count"] = d_mdf["Stories"].str.len()

d_mdf

# Word Count

d_mdf["Stories"].apply(lambda x: len(str(x).split(" ")))
d_mdf["word_count"] = d_mdf["Stories"].apply(lambda x: len(str(x).split(" ")))
d_mdf

# Catching Special Characters and Counting
d_mdf["adv_bool"] = d_mdf["Stories"].apply(lambda x: len([x for x in x.split() if x.startswith("adventure")]))
d_mdf

# Catching Numbers and Counting
mdf
mdf["Stories"].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

d_mdf["sayi_sayisi"] = mdf["Stories"].apply(lambda x: len([x for x in x.split() if x.isdigit()]))


# TEXT VISUALIZATION

data = pd.read_csv("datasets/train.tsv", sep="\t")
data.columns
data.head()

# Lower/Upper Transformation
data['Phrase'] = data['Phrase'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# Removing punctuation characters
data['Phrase'] = data['Phrase'].str.replace('[^\w\s]', '')
# Removing Numbers
data['Phrase'] = data['Phrase'].str.replace('\d', '')
# Removing stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words('english')
data['Phrase'] = data['Phrase'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
# Removing rares
sil = pd.Series(' '.join(data['Phrase']).split()).value_counts()[-1000:]
data['Phrase'] = data['Phrase'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))

# Lemmatization
# nltk.download('wordnet')
data['Phrase'] = data['Phrase'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data['Phrase'].head(10)

type(data["Phrase"])

# Term Frequency and Bar Plot
tf1 = (data["Phrase"][0:10000]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf1.columns = ["words", "tf"]
tf1.head(50)
tf1[tf1["tf"] > 100]
a = tf1[tf1["tf"] > 100]
a.plot.bar(x="words", y="tf")
plt.show()


# WORDCLOUD
text = data["Phrase"][0]
text

# Creating and showing WordCloud
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Saving the word cloud
# wordcloud.to_file("kelime_bulutu.png")

# For all the dataset

text = " ".join(i for i in data.Phrase)

wordcloud = WordCloud(max_font_size=50, background_color="white").generate(text)
plt.figure(figsize=[20, 20])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# WORDCLOUD according to a template

vbo_mask = np.array(Image.open("datasets/tr.png"))
vbo_mask

wc = WordCloud(background_color="white", max_words=1000, mask=vbo_mask, contour_width=3, contour_color="firebrick")
wc.generate(text)
wc.to_file("tr.png")
plt.figure(figsize=[10, 10])
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()


# SENTIMENT MODELING

# Load the dataset
data = pd.read_csv("datasets/train.tsv", sep="\t")
data.head()
# See the frequencies for sentiment labels
data["Sentiment"].value_counts()
# Rename sentiment labels, create binary sentiments --> 0,1 : 'negative', 3,4: 'positive'
data["Sentiment"].replace(0, value="negative", inplace=True)
data["Sentiment"].replace(1, value="negative", inplace=True)
data["Sentiment"].value_counts()
data["Sentiment"].replace(3, value="positive", inplace=True)
data["Sentiment"].replace(4, value="positive", inplace=True)
data["Sentiment"].value_counts()
data = data[(data.Sentiment == "negative") | (data.Sentiment == "positive")]
# See new frequencies for sentiment labels
data["Sentiment"].value_counts()
# Shape of the dataset
data.shape
# Create a dataframe
df = pd.DataFrame()
df["text"] = data["Phrase"]
df["label"] = data["Sentiment"]
df.head()


# Text Preprocessing

# Lower/Upper Transformation
df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# Removing punctuation characters
df['text'] = df['text'].str.replace('[^\w\s]', '')
# Removing numbers
df['text'] = df['text'].str.replace('\d', '')
# Removing stopwords
sw = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
# Removing rare words
sil = pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))
# Lemmatization
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df.head()


# Feature Engineering

# * Count Vectors
# * TF-IDF Vectors (words, characters, n-grams)
# * Word Embeddings (word2vec, bert, fasttext)

# TF(t) = (Frequency of a t term in a document) / (total number of terms in the document)
# IDF(t) = log_e(Total number of documents / number of documents with t term in them)

df.head()

# Train Test Split
train_x, test_x, train_y, test_y = model_selection.train_test_split(df["text"], df["label"], random_state=1, test_size=0.30)
train_x[0:5]
train_y[0:5]
train_x.shape, test_x.shape

# Label Encoding
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)

# Count Vectors
vectorizer = CountVectorizer()
vectorizer.fit(train_x)
x_train_count = vectorizer.transform(train_x)
x_test_count = vectorizer.transform(test_x)
type(x_train_count) # scipy.sparse.csr.csr_matrix
vectorizer.get_feature_names()[0:5]
len(vectorizer.get_feature_names()) # 13532
x_train_count.toarray()
x_train_count.toarray().shape # (53534, 13532)


# TF-IDF

# word level
tf_idf_word_vectorizer = TfidfVectorizer()
tf_idf_word_vectorizer.fit(train_x)
tf_idf_word_vectorizer.get_feature_names()[0:5]
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

# ngram level tf-idf
tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
tf_idf_ngram_vectorizer.fit(train_x)
x_train_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(train_x)
x_test_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(test_x)

# characters level tf-idf
tf_idf_chars_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
tf_idf_chars_vectorizer.fit(train_x)
x_train_tf_idf_chars = tf_idf_chars_vectorizer.transform(train_x)
x_test_tf_idf_chars = tf_idf_chars_vectorizer.transform(test_x)


# MACHINE LEARNING

# Logistic Regression

log = linear_model.LogisticRegression()
log_model = log.fit(x_train_count, train_y)
accuracy = model_selection.cross_val_score(log_model, x_test_count, test_y, cv=10).mean()
print("Count Vectors Accuracy Ratio :", accuracy) # 0.8464969333660036

loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_word, train_y)
accuracy = model_selection.cross_val_score(loj_model, x_test_tf_idf_word, test_y, cv=10).mean()
print("Word-Level TF-IDF Accuracy Ratio:", accuracy) # 0.8417022335428406

loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_ngram, train_y)
accuracy = model_selection.cross_val_score(loj_model, x_test_tf_idf_ngram, test_y, cv=10).mean()
print("N-GRAM TF-IDF Accuracy Ratio::", accuracy) # 0.7655600002279319

loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_chars, train_y)
accuracy = model_selection.cross_val_score(loj_model, x_test_tf_idf_chars, test_y, cv=10).mean()
print("CHARLEVEL Accuracy Ratio:", accuracy) # 0.7857817399942637


# Naive Bayes

nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_count, train_y)
accuracy = model_selection.cross_val_score(nb_model, x_test_count, test_y, cv = 10).mean()
print("Count Vectors Accuracy Ratio:", accuracy) # 0.8329497907949792

nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_word, train_y)
accuracy = model_selection.cross_val_score(nb_model, x_test_tf_idf_word, test_y, cv = 10).mean()
print("Word-Level TF-IDF Accuracy Ratio:", accuracy) # 0.8349372384937238

nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_ngram, train_y)
accuracy = model_selection.cross_val_score(nb_model, x_test_tf_idf_ngram, test_y, cv = 10).mean()
print("N-GRAM TF-IDF Accuracy Ratio:", accuracy) # 0.7686715481171549

nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_chars, train_y)
accuracy = model_selection.cross_val_score(nb_model, x_test_tf_idf_chars, test_y, cv = 10).mean()
print("CHARLEVEL Accuracy Ratio:", accuracy) # 0.7565899581589958


# Random Forests

rf = ensemble.RandomForestClassifier()
rf_model = rf.fit(x_train_count, train_y)
accuracy = model_selection.cross_val_score(rf_model, x_test_count, test_y, cv=10).mean()
print("Count Vectors Accuracy Ratio:", accuracy) # 0.8317211708862564

rf = ensemble.RandomForestClassifier()
rf_model = rf.fit(x_train_tf_idf_word, train_y)
accuracy = model_selection.cross_val_score(rf_model, x_test_tf_idf_word, test_y, cv=10).mean()
print("Word-Level TF-IDF Accuracy Ratio:", accuracy)

rf = ensemble.RandomForestClassifier()
rf_model = loj.fit(x_train_tf_idf_ngram, train_y)
accuracy = model_selection.cross_val_score(rf_model, x_test_tf_idf_ngram, test_y, cv=10).mean()
print("N-GRAM TF-IDF Accuracy Ratio:", accuracy)

rf = ensemble.RandomForestClassifier()
rf_model = loj.fit(x_train_tf_idf_chars, train_y)
accuracy = model_selection.cross_val_score(rf_model, x_test_tf_idf_chars, test_y, cv=10).mean()
print("CHARLEVEL Accuracy Ratio:", accuracy)


# XGBoost

xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_count,train_y)
accuracy = model_selection.cross_val_score(xgb_model, x_test_count, test_y, cv = 10).mean()
print("Count Vectors Accuracy Ratio:", accuracy)

xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_tf_idf_word,train_y)
accuracy = model_selection.cross_val_score(xgb_model, x_test_tf_idf_word, test_y, cv = 10).mean()
print("Word-Level TF-IDF Accuracy Ratio:", accuracy)

xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_tf_idf_ngram,train_y)
accuracy = model_selection.cross_val_score(xgb_model, x_test_tf_idf_ngram, test_y, cv = 10).mean()
print("N-GRAM TF-IDF Accuracy Ratio:", accuracy)

xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_tf_idf_chars,train_y)
accuracy = model_selection.cross_val_score(xgb_model, x_test_tf_idf_chars, test_y, cv = 10).mean()
print("CHARLEVEL Accuracy Ratio:", accuracy)


# Making Predictions

log_model

# We need to vectorize/convert the comment into CountVectorizer form to be able make predictions
comment = pd.Series("Chess is cerebral. This series is not. Scott Frank has made films from Logan to Wolverine. This one is somewhere in the middle. The direction and writing is clichéd to the point you can guess their dialogues.It juggles too many ideas; genius to madness, woman in male dominated field, orphan dealing with past. But everything is mismanaged. Also what's up with like 4 montages in every episode. THAT'S SO LAZY. Anya Taylor Joy is the only saving grace. Her and the production design make the show watchable. Also the chess moves and techniques are pretty accurate that's kinda cool.")
v = CountVectorizer()
v.fit(train_x)
new_comment = v.transform(comment)
# Now we can predict!
log_model.predict(new_comment)




