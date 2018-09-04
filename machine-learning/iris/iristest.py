# References:
# http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

# Load Iris Data
iris = datasets.load_iris()

#print(iris.data) #print(iris.data.shape) #print(iris.data.target)

#select only species 0
sp_select=iris.target==0
idata=iris.data[sp_select,:]

#Get Petal.Width (x) and Petal.Height (y)
x=idata[:,0]
y=idata[:,1]
#print(x) #print(y)

# reshape the array so that linear model fit function takes
x=x.reshape((x.shape[0],-1)) #150,-1
y=y.reshape((y.shape[0],-1)) #150,-1

#print(x) #print(y)

#Build Model
lm = linear_model.LinearRegression()
lm.fit(x, y)

#Draw graph
plt.cla()
plt.scatter(x,y)
plt.plot(x, lm.predict(x), color='red',linewidth=3)
plt.show()