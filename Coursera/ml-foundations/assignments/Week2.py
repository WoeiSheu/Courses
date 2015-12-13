
# coding: utf-8

# In[5]:

import graphlab as gl


# In[6]:

sales = gl.SFrame('home_data.gl/')


# In[7]:

sales


# In[8]:

gl.canvas.set_target('ipynb')
sales.show(view="Scatter Plot", x="sqft_living", y="price")


# In[9]:

training_data,test_data = sales.random_split(0.8,seed=0)


# In[10]:

sqft_model = gl.linear_regression.create(training_data,target='price',features=['sqft_living'])


# In[11]:

print test_data['price'].mean()


# In[13]:

print sqft_model.evaluate(test_data)


# In[20]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.plot(test_data['sqft_living'],test_data['price'],'.',
        test_data['sqft_living'],sqft_model.predict(test_data),'-')


# In[22]:

sqft_model.get('coefficients')


# In[24]:

my_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']
sales[my_features].show()


# In[33]:

sales.show(view='BoxWhisker Plot',x='zipcode',y='price')


# In[35]:

my_features_model = gl.linear_regression.create(training_data,target='price',features=my_features)


# In[38]:

print sqft_model.evaluate(test_data)
print my_features_model.evaluate(test_data)


# In[39]:

house1 = sales[sales['id']=='5309101200']


# In[40]:

house1


# In[49]:

get_ipython().run_cell_magic(u'HTML', u'', u'<img src="http://info.kingcounty.gov/Assessor/eRealProperty/MediaHandler.aspx?Media=2916871">')


# In[53]:

print house1['price'],sqft_model.predict(house1),my_features_model.predict(house1)


# In[55]:

house2 = sales[sales['id']=='1925069082']
house2


# In[57]:

print house2['price'],sqft_model.predict(house2),my_features_model.predict(house2)


# In[58]:

bill_gates = {'bedrooms':[8], 
              'bathrooms':[25], 
              'sqft_living':[50000], 
              'sqft_lot':[225000],
              'floors':[4], 
              'zipcode':['98039'], 
              'condition':[10], 
              'grade':[10],
              'waterfront':[1],
              'view':[4],
              'sqft_above':[37500],
              'sqft_basement':[12500],
              'yr_built':[1994],
              'yr_renovated':[2010],
              'lat':[47.627606],
              'long':[-122.242054],
              'sqft_living15':[5000],
              'sqft_lot15':[40000]}


# In[62]:

print sqft_model.predict(gl.SFrame(bill_gates)),my_features_model.predict(gl.SFrame(bill_gates))


# In[84]:

my_zipcode = sales['zipcode'].unique().sort()
maxprice = sales[sales['zipcode']==my_zipcode[0]]['price'].mean()
index = my_zipcode[0]
my_zipcode = my_zipcode[1:]
for i in my_zipcode:
    if maxprice < sales[sales['zipcode']==i]['price'].mean():
        maxprice = sales[sales['zipcode']==i]['price'].mean()
        index = i

print index,maxprice


# In[104]:

filter_data = sales[(sales['sqft_living']>2000) & (sales['sqft_living']<4000)]
1.0*len(filter_data)/len(sales)


# In[91]:

my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house				
'grade', # measure of quality of construction				
'waterfront', # waterfront property				
'view', # type of view				
'sqft_above', # square feet above ground				
'sqft_basement', # square feet in basement				
'yr_built', # the year built				
'yr_renovated', # the year renovated				
'lat', 'long', # the lat-long of the parcel				
'sqft_living15', # average sq.ft. of 15 nearest neighbors 				
'sqft_lot15', # average lot size of 15 nearest neighbors 
]
my_model = gl.linear_regression.create(training_data,target='price',features=my_features,validation_set=None)
advanced_model = gl.linear_regression.create(training_data,target='price',features=advanced_features,validation_set=None)


# In[92]:

print my_model.evaluate(test_data),advanced_model.evaluate(test_data)


# In[105]:

179542.43331269047-156831.11680200775


# In[ ]:



