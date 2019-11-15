import quandl
import numpy as np 
import matplotlib.pyplot as plt 

X = 2*np.random.rand(100,1)
y = 4+3*X+np.random.randn(100,1)
plt.scatter(X,y)
plt.show()


def cal_cost(theta,X,y):
	m = len(y)
	predictions = X.dot(theta)
	cost = (1/2*m)*np.sum(np.square(predictions - y))
	return cost

def grad_descent(X,y,theta,learning_rate=0.01,iterations=1000):
	m = len(y)
	cost_history = np.zeros(iterations)
	theta_history = np.zeros((iterations,2))
	for it in range(iterations):
		predictions = np.dot(X,theta)
		theta = theta - (1/m)*learning_rate*(X.T.dot((predictions-y)))

		#theta_history[it,:]=theta.T
		#cost_history[it] = cal_cost(theta,X,y)

	return theta, cost_history, theta_history

theta = np.random.randn(2,1)

X_b = np.c_[np.ones((len(X),1)),X]
theta, cost_history, theta_history = grad_descent(X_b,y,theta)

print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))
print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))