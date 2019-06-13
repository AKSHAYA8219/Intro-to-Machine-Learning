import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#sample data set
X=np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
Y=np.array([1,1,1,2,2,2])

clf=GaussianNB()  #setting the classifier
clf.fit(X, Y)   #train the model using fit function

#Test data
pred=clf.predict([[-0.8,-1]])

acc=accuracy_score([1],pred)
print(acc)
