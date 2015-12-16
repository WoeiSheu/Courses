
# coding: utf-8

# In[1]:

import graphlab as gl


# In[2]:

people = gl.SFrame('people_wiki.gl/')
people.head()


# In[3]:

len(people)


# In[4]:

obama = people[people['name'] == 'Barack Obama']


# In[6]:

obama['text']


# In[7]:

clooney = people[people['name'] == 'George Clooney']
clooney


# In[8]:

obama['word_count'] = gl.text_analytics.count_words(obama['text'])


# In[12]:

obama_word_count_table = obama[['word_count']].stack('word_count', new_column_name=['word','count'])
print obama_word_count_table
print obama_word_count_table.sort('count', ascending=False)


# In[13]:

people['word_count'] = gl.text_analytics.count_words(people['text'])
people.head()


# In[15]:

tfidf = gl.text_analytics.tf_idf(people['word_count'])
people['tfidf'] = tfidf
people.head()


# In[16]:

obama = people[people['name'] == 'Barack Obama']


# In[17]:

obama[['tfidf']].stack('tfidf', new_column_name=['word','tfidf']).sort('tfidf',ascending=False)


# In[19]:

clinton = people[people['name'] == 'Bill Clinton']
beckham = people[people['name'] == 'David Beckham']


# In[20]:

gl.distances.cosine(obama['tfidf'][0],clinton['tfidf'][0])


# In[21]:

gl.distances.cosine(obama['tfidf'][0],beckham['tfidf'][0])


# In[23]:

knn_model = gl.nearest_neighbors.create(people, features=['tfidf'], label='name')


# In[25]:

knn_model.query(obama)


# In[28]:

swift = people[people['name'] == 'Taylor Swift']
knn_model.query(swift)


# In[29]:

jolie = people[people['name'] == 'Angelina Jolie']
knn_model.query(jolie)


# In[30]:

arnold = people[people['name'] == 'Arnold Schwarzenegger']
knn_model.query(arnold)


# # Homework

# In[36]:

elton = people[people['name'] == 'Elton John']
elton[['word_count']].stack('word_count', new_column_name=['word','count']).sort('count',ascending=False)


# In[41]:

import operator
sorted(elton['tfidf'][0].iteritems(),key=operator.itemgetter(1),reverse = True)


# In[43]:

victoria = people[people['name'] == 'Victoria Beckham']
paul = people[people['name'] == 'Paul McCartney']
print gl.distances.cosine(elton['tfidf'][0],victoria['tfidf'][0])
print gl.distances.cosine(elton['tfidf'][0],paul['tfidf'][0])


# In[49]:

knn_model = gl.nearest_neighbors.create(people, features=['tfidf'], label='name', distance='cosine')
knn_model_wordcount = gl.nearest_neighbors.create(people, features=['word_count'], label='name', distance='cosine')


# In[50]:

knn_model.query(elton)


# In[46]:

knn_model_wordcount.query(elton)


# In[47]:

knn_model_wordcount.query(victoria)


# In[51]:

knn_model.query(victoria)


# In[ ]:



