import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import statistics as st
import random as rd
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression 


data_file =  pd.read_csv("c115/data3.csv")

data_file_Velocity =  data_file["Velocity"].tolist()
data_file_Escaped = data_file["Escaped"].tolist()

data_file_plot_graph = px.scatter(x=data_file_Velocity , y = data_file_Escaped)
# data_file_plot_graph.show()

# to plot the line of regressions
Velocity_array = np.array(data_file_Velocity)
Escaped_array = np.array(data_file_Escaped)

m,c =  np.polyfit(Velocity_array,Escaped_array,1)

y =[]

for x in Velocity_array:
    y_value =  m*x+c 
    y.append(y_value)
    
regression_plot_graph = px.scatter(x = Velocity_array , y= Escaped_array )
regression_plot_graph.update_layout(shapes=[dict(type="line", y0=min(y), y1=max(
     y), x0=min(Velocity_array), x1=max(Velocity_array))])

# regression_plot_graph.show()

# to reshape the array from 3x3 matrix to 1 array 
X = np.reshape(data_file_Velocity,(len(data_file_Velocity),1))
Y = np.reshape(data_file_Escaped,(len(data_file_Escaped),1))

# to fit the data in the model 
lr = LogisticRegression()
lr.fit(X,Y)

# to create a scater plot using matplotlib
plt.figure()
plt.scatter(X.ravel(),Y,color = "teal", zorder = 20 )
# to define the sigmoid formula 
def model(x):
    return 1/(1+np.exp(-x))

# to shape the dots evenly 
x_test = np.linspace(0,100,200)
# to create the x_test as single array
chances = model(x_test*lr.coef_+lr.intercept_).ravel()
# to plot the different value of y with different values 
plt.plot(x_test,chances,color = "orange", linewidth =3)
plt.axhline( y = 0 ,color="red" ,linestyle ="-")
plt.axhline( y = 1 ,color="green" ,linestyle ="-")
plt.axhline( y = 0.5 ,color="blue" ,linestyle ="--")

plt.show()


# Velocity = int(input("enter add your marks"))
# admit = model(Velocity*lr.coef_+lr.intercept_).ravel()[0]

# if admit <= 0.01:
#     print("\033[1;31;40m REJECTED")
# else :
#     print("\033[1;32;40m Escaped")