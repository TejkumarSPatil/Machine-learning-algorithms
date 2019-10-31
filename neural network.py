                             NEURAL NETWORK

- Neural networks work in very similar manner. 
- It takes several input, processes it through multiple neurons from multiple hidden layers and
  returns the result using an output layer. 
- This result estimation process is technically known as “Forward Propagation“.

-  the basic forming unit of a neural network is a perceptron.
- A perceptron can be understood as anything that takes multiple inputs and produces one output.


import numpy as np

#Input array
x=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
x
x.shape

#Output
y=np.array([[1],[1],[0]])
y
y.shape


#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))
print(sigmoid(x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)
print(derivatives_sigmoid(x))


#Variable initialization
epoch=10000 #Setting training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons = x.shape[1] #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer


#weight and bias initialization of hidden layers
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))

print(wh)
print(bh)
print(wout)
print(bout)


for i in range(epoch):      #Forward Propogation
    hidden_layer_input1=np.dot(x,wh)
    hidden_layer_input=hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1=np.dot(hiddenlayer_activations,wout)
    output_layer_input= output_layer_input1+ bout
    output = sigmoid(output_layer_input)
    
    
 #Backpropagation
E = y-output
slope_output_layer = derivatives_sigmoid(output)
slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
d_output = E * slope_output_layer
Error_at_hidden_layer = d_output.dot(wout.T)
d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
wout += hiddenlayer_activations.T.dot(d_output) *lr
bout += np.sum(d_output, axis=0,keepdims=True) *lr
wh += x.T.dot(d_hiddenlayer) *lr
bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr 

print(output)  

########################################################################################




    

