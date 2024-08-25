import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    #Activation function
    return 1/(1+np.exp(-x))

def deriv_sigmoid(x):
    #derivative of sigmoig, used to calculate gradient descent
    fx = sigmoid(x)
    return fx *(1-fx)

def mse_loss(y_true, y_pred):
    #This is essentially our cost function
    #y_true and y_pred are numpy arrays of the same length
    return ((y_pred - y_true)**2).mean()

class NeuralNetwork:
    '''
    A neural network with:
        - 2 inputs
        - a hidden layer consisting of 2 neurons (h1 and h2)
        - an output layer with one neuron (o1) that takes h1 and h2 as inputs
    '''
    def __init__(self):
        #Weights
        self.w1 = np.random.normal() #x1 to h1
        self.w2 = np.random.normal() #x2 to h1
        self.w3 = np.random.normal() #x1 to h2
        self.w4 = np.random.normal() #x2 to h2
        self.w5 = np.random.normal() #h1 to o1
        self.w6 = np.random.normal() #h2 to o1

        #Biases 
        self.b1 = np.random.normal() #h1 bias
        self.b2 = np.random.normal() #h2 bias
        self.b3 = np.random.normal() #o1 bias

    def feedforward(self,x):
        h1 = sigmoid(self.w1*x[0] + self.w2*x[1] + self.b1)
        h2 = sigmoid(self.w3*x[0] + self.w4*x[1] + self.b1)
        o1 = sigmoid(self.w5*h1 + self.w6*h2 + self.b3)
        return o1
    
    def train(self, data, all_y_trues):
        '''
        - data is a (n x 2) numpy array, where n = # of samples in the dataset
        - all_y_trues is a numpy array with n elements. Elements in all_y_trues correspond to those in data
        '''

        learn_rate = 0.1
        epochs = 1000 #number of times to loop through the entire dataset
        losses = []

        for epoch in range(epochs):
            for x,y_true in zip(data, all_y_trues):
                #---Do a feed forward
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)

                y_pred = o1

                #---Calculate partial derivatives
                d_L_d_ypred = -2 *(y_true-y_pred)

                #Neuron o1
                #we are essentially taking the derivative of f(w5*h1 + w6*h2 + b3)
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)   #product rule
                d_ypred_d_w6 = h2* deriv_sigmoid(sum_o1)    #product rule
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5*deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6*deriv_sigmoid(sum_o1)

                #Neuron h1
                d_h1_d_w1 = x[0] *deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] *deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                #Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                #---Update weights and biases 
                #Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate *d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                #Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                #Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6  -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

                #---Calculate total loss at the end of each epoch
                if epoch % 10 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1, data)
                    loss = mse_loss(all_y_trues, y_preds)
                    losses.append(loss)
                    print("Epoch %d loss: %.3f" % (epoch,loss))

        plt.figure(figsize=(10, 6))
        plt.plot(range(0, 400), losses, marker='o', linestyle='-')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()    
        


#Define dataset
data = np.array([
[-2,-1], #Alice
[25,6],  #Bobert
[17,4],  #Charlie
[-15,-6], #Diana
])

#0 is male, 1 is female
all_y_trues = np.array([
1, #Alice
0, #Bobert
0, #Charlie
1,  #Diana
])

#Train the Neural Network
network = NeuralNetwork()
network.train(data, all_y_trues)

name = input("Please input the name of the person: ")
height = int(input('Please input the height of the person in inches to the nearest whole number: ')) - 66
weight = int(input('Please input the weight of the person in pounds to the nearest whole number: ')) - 135

input = np.array([weight, height])
output = network.feedforward(input)

print(output)

if output <= 0.5:
    print(name, "is likely a male")

elif output > 0.5:
    print(name, "is likely a female")




