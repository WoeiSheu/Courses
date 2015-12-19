
# coding: utf-8

# In[1]:

import graphlab as gl
song_data = gl.SFrame('song_data.gl/')


# In[2]:

song_data.head()


# In[3]:

gl.canvas.set_target('ipynb')
song_data['song'].show()


# In[4]:

len(song_data)


# In[5]:

users = song_data['user_id'].unique()


# In[6]:

len(users)


# In[7]:

train_data,test_data = song_data.random_split(0.8,seed=0)


# In[8]:

popularity_model = gl.popularity_recommender.create(train_data,
                                                   user_id='user_id',
                                                   item_id='song')


# In[9]:

popularity_model.recommend(users=[users[0]])


# In[10]:

popularity_model.recommend(users=[users[1]])


# In[11]:

personalized_model = gl.item_similarity_recommender.create(train_data,
                                                          user_id='user_id',
                                                          item_id='song')


# In[13]:

personalized_model.recommend(users=[users[0]])


# In[14]:

personalized_model.recommend(users=[users[1]])


# In[15]:

personalized_model.get_similar_items(['With Or Without You - U2'])


# In[29]:

get_ipython().magic(u'matplotlib inline')


# In[30]:

model_performance = gl.recommender.util.compare_models(test_data,
                                                      [popularity_model,personalized_model],
                                                      user_sample=0.05)


# In[31]:

song_data.head()


# In[33]:

len(song_data[song_data['artist']=='Kanye West']['user_id'].unique())


# In[34]:

len(song_data[song_data['artist']=='Foo Fighters']['user_id'].unique())


# In[35]:

print len(song_data[song_data['artist']=='Taylor Swift']['user_id'].unique())
print len(song_data[song_data['artist']=='Lady GaGa']['user_id'].unique())


# In[47]:

artists_listen_count = song_data.groupby(key_columns='artist', operations={'total_count': gl.aggregate.SUM('listen_count')})


# In[54]:

artists_listen_count.sort('total_count',ascending=False)


# In[55]:

artists_listen_count.sort('total_count',ascending=True)


# In[56]:

subset_test_users = test_data['user_id'].unique()[0:10000]


# In[66]:

rec_song = personalized_model.recommend(subset_test_users,k=1)


# In[65]:

print subset_test_users.head()
print rec_song


# In[67]:

rec_song.groupby(key_columns='song', operations={'count': gl.aggregate.COUNT()}).sort('count',ascending=False)


# In[ ]:



