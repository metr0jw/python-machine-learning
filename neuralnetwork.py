import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.arange(-10, 10, 0.01)
plt.plot(x, relu(x))
plt.show()

def identity_function(x):
    return x

def init_network() :
    network = {} # dictionary data type declare
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]]) # 0->1 weight change
    network['b1'] = np.array([0.1,0.2,0.3]) # 0->1 bias change
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]]) # 1->2 weight change
    network['b2'] = np.array([0.1,0.2]) # 1->2 bias change
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]]) # 2->3 weight change
    network['b3'] = np.array([0.1, 0.2]) # 2->3 bias change

    return network

def forward(network,x) :
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # Save value which is stored in the variable

    a1 = np.dot(x,W1) + b1 # 0->1 calculate weight
    z1 = relu(a1) # 0->1 activated
    a2 = np.dot(z1,W2) + b2 # 1=>2 calculate weight
    z2 = relu(a2) # 1->2 activated
    a3 = np.dot(z2,W3) + b3 # 2->3 calculate weight
    y = identity_function(a3) # 2->3 activated

    return y

network = init_network()
x = np.array([1.0, 0.5]) # input
y = forward(network, x) # output
print(y)