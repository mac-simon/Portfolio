#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


# In[2]:


#importing data frame for df1
df =  pd.read_csv('amazon_cells_labelled.txt', sep='\t' , engine='python',names=['review','label'])


# In[3]:


#importing data frame for df2
df2 = pd.read_csv('yelp_labelled.txt', sep='\t' , engine='python',names=['review','label'])


# In[4]:


#importing data frame for df3
df3 =  pd.read_csv('imdb_labelled.txt', sep='\t' , quoting=3, engine='python',names=['review','label'])


# In[5]:


#appending the three dataframes
df = df.append([df2,df3])


# In[6]:


df


# In[7]:


#viewing head
df.head()


# In[8]:


#finding unique messages
# 1493 unique positive reviews, 1490 unique negative reviews
df.groupby('label').describe()


# In[9]:


#adding message length as column
df['length'] = df['review'].apply(len)


# In[10]:


df.head()


# In[11]:


#65 word sentence reviews - before processing 
df.length.describe()


# In[12]:


#visualizing message length - before processing 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.title("Review Length")
plt.xlabel("Word Sentence Length")
plt.ylabel("Frequency")
df['length'].plot.hist(bins=30)


# In[13]:


# Preprocessing
#removing puncuation
import re
import string

def remove_punct(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)
string.punctuation
df["review"] = df.review.map(remove_punct)


# In[14]:


from collections import Counter

# Count unique words
def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count


counter = counter_word(df.review)


# In[15]:


len(counter)


# In[16]:


# Stop Words: A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine
# has been programmed to ignore, both when indexing entries for searching and when retrieving them 
# as the result of a search query.
stop = set(stopwords.words("english"))
def remove_stopwords(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)


# In[17]:


df['review'] = df.review.map(remove_stopwords)


# In[18]:


#message length info - post processing 
df['length'].describe()
df['length'] = df['review'].apply(len)


# In[19]:


#processed dataframe
df.head()


# In[20]:


#processed dataframe stats 
df.length.describe()


# In[21]:


#writing processed data to csv
df.to_csv('prepared_dataset_d213.csv')


# In[22]:


#visualizing message length
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.title("Word Sentence Length")
plt.xlabel("Word Frequency")
plt.ylabel("Sentence Length")
df['length'].plot.hist(bins=30)


# In[23]:


from collections import Counter

# Count unique words
def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count
counter = counter_word(df.review)


# In[24]:


#number of unique words
len(counter)


# In[25]:


#top 5 most used words
counter.most_common(5)


# In[26]:


num_unique_words = len(counter)


# In[27]:


# Split dataset into training and validation set
train_size = int(df.shape[0] * 0.8)

train_df = df[:train_size]
val_df = df[train_size:]

# split text and labels
train_sentences = train_df.review.to_numpy()
train_labels = train_df.label.to_numpy()
val_sentences = val_df.review.to_numpy()
val_labels = val_df.label.to_numpy()


# In[28]:


train_sentences.shape, val_sentences.shape


# In[29]:


from tensorflow.keras.preprocessing.text import Tokenizer

# vectorize a text corpus by turning each text into a sequence of integers
tokenizer = Tokenizer(num_words=num_unique_words)
tokenizer.fit_on_texts(train_sentences) # fit only to training


# In[30]:


# each word has unique index
word_index = tokenizer.word_index


# In[31]:


word_index


# In[32]:


train_sequences = tokenizer.texts_to_sequences(train_sentences)
val_sequences = tokenizer.texts_to_sequences(val_sentences)


# In[33]:


print(train_sentences[10:15])
print(train_sequences[10:15])


# In[34]:


# Pad the sequences to have the same length
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Max number of words in a sequence
max_length = 41

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding="post", truncating="post")
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding="post", truncating="post")
train_padded.shape, val_padded.shape


# In[35]:


train_padded[10]


# In[36]:


print(train_sentences[10])
print(train_sequences[10])
print(train_padded[10])


# In[37]:


# Check reversing the indices

# flip (key, value)
reverse_word_index = dict([(idx, word) for (word, idx) in word_index.items()])


# In[38]:


reverse_word_index


# In[39]:


def decode(sequence):
    return " ".join([reverse_word_index.get(idx, "?") for idx in sequence])


# In[40]:



decoded_text = decode(train_sequences[10])

print(train_sequences[10])
print(decoded_text)


# In[41]:


# Create LSTM model
from tensorflow.keras import layers

# Embedding: https://www.tensorflow.org/tutorials/text/word_embeddings
# Turns positive integers (indexes) into dense vectors of fixed size. (other approach could be one-hot-encoding)

# Word embeddings give us a way to use an efficient, dense representation in which similar words have 
# a similar encoding. Importantly, you do not have to specify this encoding by hand. An embedding is a 
# dense vector of floating point values (the length of the vector is a parameter you specify).

model = keras.models.Sequential()
model.add(layers.Embedding(num_unique_words, 64, input_length=max_length))

# The layer will take as input an integer matrix of size (batch, input_length),
# and the largest integer (i.e. word index) in the input should be no larger than num_words (vocabulary size).
# Now model.output_shape is (None, input_length, 32), where `None` is the batch dimension.


model.add(layers.LSTM(64, dropout=0.1))
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()


# In[42]:


loss = keras.losses.BinaryCrossentropy(from_logits=False)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)


# In[45]:


history = model.fit(train_padded, train_labels, epochs=30, validation_data=(val_padded, val_labels), verbose=2, callbacks=[callback])


# In[46]:


predictions = model.predict(train_padded)
predictions = [1 if p > 0.5 else 0 for p in predictions]


# In[47]:


print(train_sentences[10:20])

print(train_labels[10:20])
print(predictions[10:20])


# In[48]:


import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (8.0, 5.0)
plt.style.use("dark_background")
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(history.epoch,history.history['loss'])
history.history['loss']


# In[49]:


history.epoch


# In[50]:


model.save_weights('./checkpoints/my_checkpoint')


# In[51]:


import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (8.0, 5.0)
plt.style.use("dark_background")
plt.title("Val_accuracy vs Loss")
plt.xlabel("Val_acuracy")
plt.ylabel("Loss")
plt.plot(history.history['val_accuracy'],history.history['loss'])
history.history['loss']


# In[ ]:





# In[ ]:





# In[ ]:




