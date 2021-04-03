import analyseImage
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



X_matrix,Y_matrix = analyseImage.run()

def sigmoid(z):

    gz = 1/(1+np.exp(-z))

    return gz

def costFunctionReg(theta, X, y, Lambda):
    #number of training examples
    m = y.shape[0]

    #vector of model predictions for all training examples
    h= sigmoid(np.dot(X,theta))

    error = (-y* np.log(h)) - ((1-y)*np.log(1-h))

    #cost function without regularization term
    cost = sum(error)/m

    #add regularization term to the cost function
    regCost = cost + Lambda/(2*m) * sum(theta[1:]**2)

    #gradient of theta
    grad_theta = (1/m) * np.dot(X.transpose(),(h - y))[0]
    #vector of gradients of theta_j from j=1:n (adding the regularization term of the gradient)
    grad = (1/m) * np.dot(X.transpose(),(h - y))[1:] + (Lambda/m)* theta[1:]
       
    # all gradients in a column vector shape
    grad_all=np.append(grad_theta,grad)
    grad_all = grad_all.reshape((len(grad_all), 1))
    
    return regCost[0], grad_all

def gradientDescent(X,y,theta,alpha,num_iters,Lambda):
    #print("NEW ITERATION")
    #print(y)
    J_history = []
    
    for i in range(num_iters):
        
        #call CostFunctionReg 
        cost, grad = costFunctionReg(theta, X, y, Lambda)
        
        #update theta
        descent = alpha * grad ### <---- no need to multiply by 1/m since its already done in the cost function
        theta = theta - descent
        
        J_history.append(cost)
    
    return theta , J_history

def oneVsAll(X, y, initial_theta, alpha, num_iters, Lambda, K):
    all_theta = []
    all_J=[]
    
    #number of training examples
    m= X.shape[0]
   
    #number of features
    n= X.shape[1] # e.g. number of pixels
    
    # k = len(list(set(y[:, 0]))) # find more efficient solution
    
    # add an extra column of 1´s corresponding to xo=1 (aka intercept term)
    X = np.append(np.ones((m,1)), X, axis=1)
    
    for i in range(1,K+1):
        # IMPORTANT TO SEE np.where(y==i,1,0)
        theta , J_history = gradientDescent(X,np.where(y==i,1,0),initial_theta,alpha,num_iters,Lambda)
        
        #update (extend)
        all_theta.extend(theta)
        
        #update (extend)
        all_J.extend(J_history)

        
    return np.array(all_theta).reshape(K,n+1), all_J # reshape just to be sure

def predictOneVsAll(all_theta, X):
    """
    Using all_theta, compute the probability of image X(i) for each class and predict the label
    
    return a vector of prediction
    """
    #number of training examples
    m = X.shape[0]
    
    # add an extra column of 1´s corresponding to xo=1 (aka intercept term)
    X = np.append(np.ones((m,1)), X, axis=1)
    
    predictions = np.dot (X, all_theta.T) # predictions.shape =(5000,10)
    #np.argmax returns indices of the max element of the array in a particular axis.
    #+1 in order to label 0 as 10. 
    #print(predictions)

    return np.argmax(predictions,axis=1)+1

K = 10
m= X_matrix.shape[0]
#Inicialize vector theta =0
initial_theta = np.zeros((X_matrix.shape[1] + 1, 1))
#Optimization hyper-parameters 
alpha=1 #learning rate
num_iters=300
Lambda=0.1

all_theta, all_J = oneVsAll(X_matrix, Y_matrix, initial_theta, alpha, num_iters, Lambda, K)

plt.plot(all_J)  #All classifiers
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
#plt.show()
image_to_predict=analyseImage.analyseImage()
pred = predictOneVsAll(all_theta, image_to_predict)
#Check that pred.shape  = (5000,) => rank 1 array. You need to reshape it !!!
#print(pred.shape)
pred= pred.reshape(pred.shape[0], 1)
print(pred.shape)
print("My guess is :")
print(pred)
print("Training Set Accuracy:",sum(pred==Y_matrix)[0]/m*100,"%")  # IMPORTANT FORMULA