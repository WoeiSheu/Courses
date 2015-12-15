
# coding: utf-8

# In[1]:

import graphlab as gl


# In[2]:

products = gl.SFrame('amazon_baby.gl/')


# In[3]:

products.head()


# In[4]:

products['word_count'] = gl.text_analytics.count_words(products['review'])


# In[5]:

products.head()


# In[6]:

gl.canvas.set_target('ipynb')
products['name'].show()


# In[7]:

giraffe_reviews = products[products['name']=='Vulli Sophie the Giraffe Teether']


# In[8]:

len(giraffe_reviews)


# In[10]:

giraffe_reviews['rating'].show(view='Categorical')


# In[11]:

products['rating'].show(view='Categorical')


# In[12]:

products_exclude_3stars = products[products['rating']!=3]


# In[13]:

products_exclude_3stars['sentiment'] = products_exclude_3stars['rating'] >= 4
products_exclude_3stars.head()


# In[19]:

train_data,test_data = products_exclude_3stars.random_split(0.8,seed=0)


# In[20]:

sentiment_model = gl.logistic_classifier.create(train_data,
                                                target='sentiment',
                                                features=['word_count'],
                                                validation_set=test_data)


# In[81]:

sentiment_model.evaluate(test_data)
# sentiment_model.evaluate(test_data,metric='roc_curve')


# In[22]:

sentiment_model.show(view='Evaluation')


# In[23]:

giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews,output_type='probability')
giraffe_reviews.head()


# In[27]:

giraffe_reviews = giraffe_reviews.sort('predicted_sentiment',ascending=False)


# In[28]:

giraffe_reviews.head()


# In[29]:

giraffe_reviews[0]['review']


# In[30]:

giraffe_reviews[-1]['review']


# In[31]:

selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']


# In[76]:

products = products[products['rating']!=3]
products['sentiment'] = products['rating'] >= 4

sums = {}
def create_new(singleword):
    def singleword_count(word_count):
        if singleword in word_count:
            return word_count[singleword]
        else:
            return 0
    
    products[singleword] = products['word_count'].apply(singleword_count)
    sums[singleword] = products[singleword].sum()
    
map(create_new,selected_words)

print sums


# In[79]:

import operator
print sorted(sums.iteritems(),key=operator.itemgetter(1))
products.head()


# In[55]:

train_data,test_data = products.random_split(.8, seed=0)

selected_words_model = gl.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=selected_words,
                                                     validation_set=test_data)


# In[92]:

# selected_words_model['coefficients'].sort('value',ascending=True)
selected_words_model['coefficients'].sort('value',ascending=True).print_rows(num_rows=12)


# In[83]:

# selected_words_model.evaluate(test_data)
selected_words_model.evaluate(test_data,metric='roc_curve')

selected_words_model.show(view='Evaluation')


# In[88]:

diaper_champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ']
diaper_champ_reviews['predicted_sentiment'] = sentiment_model.predict(diaper_champ_reviews,output_type='probability')
diaper_champ_reviews = diaper_champ_reviews.sort('predicted_sentiment',ascending=False)
diaper_champ_reviews.head()


# In[89]:

diaper_champ_reviews['predicted_sentiment'] = selected_words_model.predict(diaper_champ_reviews, output_type='probability')
diaper_champ_reviews.head()
# diaper_champ_reviews.sort('predicted_sentiment',ascending=False)


# In[90]:

def printSelected(selected_word):
    print diaper_champ_reviews[0][selected_word]

map(printSelected,selected_words)

diaper_champ_reviews[0]['word_count']


# In[73]:

diaper_champ_reviews.head()


# In[ ]:



