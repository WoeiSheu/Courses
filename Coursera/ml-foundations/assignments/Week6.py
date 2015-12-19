
# coding: utf-8

# # Deep Features for Image Classification

# In[1]:

import graphlab as gl


# In[2]:

image_train = gl.SFrame('image_train_data/')
image_test = gl.SFrame('image_test_data/')


# In[3]:

print image_train.head()
print image_test.head()


# In[4]:

gl.canvas.set_target('ipynb')


# In[5]:

image_train['image'].show()


# In[6]:

raw_pixel_model = gl.logistic_classifier.create(image_train,
                                               target='label',
                                               features=['image_array'])


# In[7]:

image_test[0:3]['image'].show()


# In[8]:

image_test[0:3]['label']


# In[9]:

raw_pixel_model.predict(image_test[0:3])


# In[10]:

raw_pixel_model.evaluate(image_test)


# In[11]:

len(image_train)


# In[12]:

deep_features_model = gl.logistic_classifier.create(image_train,
                                                         features=['deep_features'],
                                                         target='label')


# In[13]:

image_test[0:3]['image'].show()
deep_features_model.predict(image_test[0:3])


# In[14]:

deep_features_model.evaluate(image_test)


# # Deep Features for Image Retrieval

# In[15]:

knn_model = gl.nearest_neighbors.create(image_train,
                                       features=['deep_features'],
                                       label='id')


# In[16]:

cat = image_train[18:19]
cat['image'].show()


# In[17]:

knn_model.query(cat)


# In[18]:

def get_images_from_id(query_result):
    return image_train.filter_by(query_result['reference_label'],'id')


# In[19]:

cat_neighbors = get_images_from_id(knn_model.query(cat))


# In[20]:

cat_neighbors['image'].show()


# In[21]:

car = image_train[8:9]
car['image'].show()


# In[22]:

car_neighbors = get_images_from_id(knn_model.query(car))


# In[23]:

car_neighbors['image'].show()


# In[24]:

show_neighbors = lambda i: get_images_from_id(knn_model.query(image_train[i:i+1]))['image'].show()


# In[25]:

show_neighbors(8)


# In[26]:

show_neighbors(64)


# In[27]:

image_train['label'].sketch_summary()


# In[28]:

automobile_data = image_train[image_train['label'] == 'automobile']
dog_data = image_train[image_train['label'] == 'dog']
cat_data = image_train[image_train['label'] == 'cat']
bird_data = image_train[image_train['label'] == 'bird']


# In[29]:

automobile_model = gl.nearest_neighbors.create(automobile_data,
                                              features=['deep_features'],
                                              label='id')
dog_model = gl.nearest_neighbors.create(dog_data,
                                              features=['deep_features'],
                                              label='id')
cat_model = gl.nearest_neighbors.create(cat_data,
                                              features=['deep_features'],
                                              label='id')
bird_model = gl.nearest_neighbors.create(bird_data,
                                              features=['deep_features'],
                                              label='id')


# In[30]:

get_images_from_id(cat_model.query(image_test[0:1]))


# In[31]:

def get_image_from_model(model):
    get_images_from_id(model.query(image_test[0:1])[0])['image'].show()

get_image_from_model(cat_model)
get_image_from_model(dog_model)
get_image_from_model(automobile_model)
get_image_from_model(bird_model)


# In[32]:

sum(cat_model.query(image_test[0:1])[0:5]['distance'])/5


# In[33]:

sum(dog_model.query(image_test[0:1])[0:5]['distance'])/5


# In[34]:

image_test_cat = image_test[image_test['label'] == 'cat']
image_test_dog = image_test[image_test['label'] == 'dog']
image_test_bird = image_test[image_test['label'] == 'bird']
image_test_automobile = image_test[image_test['label'] == 'automobile']


# In[35]:

dog_cat_neighbors = cat_model.query(image_test_dog, k=1)
dog_dog_neighbors = dog_model.query(image_test_dog, k=1)
dog_bird_neighbors = bird_model.query(image_test_dog, k=1)
dog_automobile_neighbors = automobile_model.query(image_test_dog, k=1)


# In[36]:

dog_cat_neighbors.head()


# In[37]:

dog_distances = gl.SFrame({'dog-dog': dog_dog_neighbors['distance'],
                          'dog-cat': dog_cat_neighbors['distance'],
                          'dog-bird': dog_bird_neighbors['distance'],
                          'dog-automobile': dog_automobile_neighbors['distance']})


# In[38]:

dog_distances.head()


# In[43]:

# it's confusing that: min(dog_distances[0]) print dog-automobile, while max(dog_distances[0]) print dog-dog
# the reason is dog_distances[0]) is a dict, and it sort the dict by the key, but not value.
def is_dog_correct(row):
    if min(row,key=row.get) == 'dog-dog':
        return 1
    else:
        return 0

dog_distances.apply(is_dog_correct)


# In[44]:

min(dog_distances[10].items(), key=lambda x: x[1]) # another way to get the min.


# In[45]:

sum(dog_distances.apply(is_dog_correct))


# In[ ]:



