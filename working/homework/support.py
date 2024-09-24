"""Support file with classes/functions which are used in multiple notebooks"""
#8.7 pylint yay
import numpy as np


class Neuron:
    """Simple realization of Neuron (w1 * x1 + w2 * x2 + b) with gradients types: "sgd", "classic"
    and optimizers: "classic", "adam" are included inside.\\
    Training process is based on sklearn estimator fit model."""
    def __init__(self,
                initial_weights:np.ndarray[float],
                initial_bias:float = 0,
                learning_rate:float = 0.3,
                epochs:int = 10,
                loss_function:callable = None,
                grad:str = "classic",
                optimizer:str = "classic",
                with_bias:bool = True, **kwargs) -> None:
        self.weights = initial_weights
        self.bias = initial_bias
        self.learning_rate = learning_rate
        self.iter = epochs
        self.metadata = {"loss":[], "grad":[]}
        self.with_bias = with_bias
        self.optimizer = optimizer

        if loss_function is None:
            self.loss_function = self.__mse
        else:
            self.loss_function = loss_function
        if grad == "classic":
            self.grad = self.__subgrad
        else:
            self.grad = self.__stochsubgrad
        #params for adam (were created to fetch weights at each step of gradient)
        if optimizer == "adam":
            self.beta1 = 0
            self.beta2 = 0
            if "beta1" in kwargs:
                self.beta1 = kwargs["beta1"]
            if "beta2" in kwargs:
                self.beta2 = kwargs["beta2"]
            self.first_moment = 0
            self.second_moment = 0


    def fit(self, data:np.ndarray, y:np.ndarray) -> dict:
        """fitting model params by X, y"""
        #classical learning rate optimizer
        if self.optimizer == "classic":
            for i in range(self.iter):
                grad = self.grad(data,y, 0.001)
                for o,item in enumerate(self.weights):
                    self.weights[o] -= grad[o] * self.learning_rate
                if self.with_bias is True:
                    self.bias -= grad[-1] * self.learning_rate
                else:
                    self.bias = 0
                theta = list(self.weights)
                theta.append(self.bias)
                loss = self.loss_function(y, self.__predict(data, theta))

                self.metadata["loss"].append(np.round(loss,4))
                self.metadata["grad"].append(np.round(grad,4))
            return self.metadata
        #adam optimizer
        if self.optimizer == "adam":
            for i in range(1,self.iter+1):
                grad = self.grad(data, y, 0.001)
                self.first_moment = self.beta1*self.first_moment + (1 - self.beta1) * grad
                self.second_moment = self.beta2*self.second_moment + (1-self.beta2) * (grad**2)
                bias_corrected_fm = self.first_moment/(1-self.beta1**i)
                bias_corrected_sm = self.second_moment/(1-self.beta2**i)

                #update step
                for o,item in enumerate(self.weights):
                    self.weights[o] -= self.learning_rate * bias_corrected_fm[o] \
                    / (np.sqrt(bias_corrected_sm[o]) + 0.00000001)
                if self.with_bias is True:
                    self.bias -=  self.learning_rate * bias_corrected_fm[-1] \
                        / (np.sqrt(bias_corrected_sm[-1]) + 0.00000001)
                else:
                    self.bias = 0
                theta = list(self.weights)
                theta.append(self.bias)
                loss = self.loss_function(y, self.__predict(data, theta))

                self.metadata["loss"].append(np.round(loss,4))
                self.metadata["grad"].append(np.round(grad,4))


    def predict(self, data:np.ndarray) -> np.ndarray:
        if len(data.shape) == 1:
            data = data.reshape(-1,1)
        y_pred = []
        theta = list(self.weights)
        theta.append(self.bias)
        for i,item in enumerate(data):
            x = list(item)
            x.append(1)
            y_pred.append(self.__sigmoid(np.dot(x, theta)))
        return np.array(y_pred)

    def __sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    def __mse(self, y_true, y_pred):
        return np.mean((y_true-y_pred)**2)
    
    def __predict(self, data:np.ndarray, theta) -> np.ndarray:
        if len(data.shape) == 1:
            data = data.reshape(-1,1)
        y_pred = []
        for i,item in enumerate(data):
            x = list(item)
            x.append(1)
            y_pred.append(self.__sigmoid(np.dot(x, theta)))
        return np.array(y_pred)
    
    # classic grad
    def __subgrad(self, data, y_true, alpha) -> np.ndarray:
        grad = []
        theta = self.weights.copy()
        theta.append(self.bias)
        for i,item in enumerate(theta):
            theta_hat = theta.copy()
            theta_hat[i] += alpha
            y = self.__predict(data, theta)
            y_hat = self.__predict(data, theta_hat)
            grad.append((self.loss_function(y_true, y_hat) - self.loss_function(y_true, y))/alpha)
        return np.array(grad)
    
    # sgd
    def __stochsubgrad(self, data, y_true, alpha) -> np.ndarray:
        grad = []
        theta = self.weights.copy()
        theta.append(self.bias)
        #batch
        choice = np.random.randint(0,len(data),int(len(data)*0.67)).astype(int)
        data = np.array(data)[choice]
        y_true = np.array(y_true)[choice]
        for i,item in enumerate(theta):
            theta_hat = theta.copy()
            theta_hat[i] += alpha
            y = self.__predict(data, theta)
            y_hat = self.__predict(data, theta_hat)
            grad.append((self.loss_function(y_true, y_hat) - self.loss_function(y_true, y))/alpha)
        return np.array(grad)
