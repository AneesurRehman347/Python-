import dataset
import numpy as np
def GradientDesc(x,y):
    m = c =0
    iter =1000
    n = len(x)
    learning_rate = 0.08
    for i in range(iter):
        y_pred = m*x+c
        mse = (1/n) * sum(val**2 for val in (y-y_pred))
        dm = -(2/n) * sum(x*(y -y_pred))
        dc =  -(2/n) * sum(y -y_pred)
        m = m - learning_rate *dm
        c = c - learning_rate * dc
        print("m={},c={},iteration={},CostMinimum={}".format(m,c,i,mse))

x =[1,2,3,4,5]
x = np.array(x)
y =[5,9,11,13,15]
y = np.array(y)

GradientDesc(x,y)


