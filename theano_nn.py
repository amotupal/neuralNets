import theano
import theano.tensor as T
from theano import shared
import numpy as np
import sklearn.datasets as skds
import matplotlib.pyplot as plt
import timeit

X = T.matrix('X') # matrix of doubles
y = T.lvector('y') # vector of int64

train_X, train_y = skds.make_moons(5000, noise=0.20)


epsilon = np.float32(0.01)
reg_lambda = np.float32(0.01)




# Dimension definitions
num_examples = len(train_X)
nn_input_dim = 2
nn_ouput_dim = 2
nn_hdim = 1000

# Shared variables with initial values
X = theano.shared(train_X.astype('float32'), name = 'X')
y = theano.shared(train_y.astype('int32'), name = 'y')
W1 = theano.shared(np.random.randn(nn_input_dim, nn_hdim).astype('float32'), name = 'W1')
b1 = theano.shared(np.zeros(nn_hdim).astype('float32'), name='b1')
W2 = theano.shared(np.random.randn(nn_hdim, nn_ouput_dim).astype('float32'), name = 'W2')
b2 = theano.shared(np.zeros(nn_ouput_dim).astype('float32'), name = 'b2')


# Forward Propagation Algorithm in theano
z1 = X.dot(W1) + b1
a1 = T.tanh(z1)
z2 = a1.dot(W2) + b2
y_hat = T.nnet.softmax(z2) 

loss_reg = 1./num_examples * reg_lambda/2 * (T.sum(T.sqr(W1)) + T.sum(T.sqr(W2)))

loss = T.nnet.categorical_crossentropy(y_hat, y).mean() + loss_reg

prediction = T.argmax(y_hat, axis=1)

forward_prop = theano.function([], y_hat)
calculate_loss = theano.function([],loss)
predict = theano.function([], prediction)


# calculating derivatives for gradient
dW2 = T.grad(loss, W2)
db2 = T.grad(loss, b2)
dW1 = T.grad(loss, W1)
db1 = T.grad(loss, b1)

gradient_step = theano.function(inputs=[],updates=((W2, W2 - epsilon * dW2),
                                               (W1, W1 - epsilon * dW1),
                                               (b2, b2 - epsilon * db2),
                                               (b1, b1 - epsilon * db1)))

def build_model(num_passes=20000, print_loss=False):
    
    np.random.seed(0)
    W1.set_value(np.random.randn(nn_input_dim, nn_hdim).astype('float32') /np.sqrt(nn_input_dim))
    b1.set_value(np.zeros(nn_hdim).astype('float32'))
    W2.set_value(np.random.randn(nn_hdim, nn_ouput_dim).astype('float32') / np.sqrt(nn_hdim))
    b2.set_value(np.zeros(nn_ouput_dim).astype('float32'))
    
    for i in range(0, num_passes):

        gradient_step()

        if(print_loss and (i % 1000 == 0)):
            print("Loss after iteration " + str(i) + ": " + str(calculate_loss()))



def plot_decision_boundary(pred_func):

    # Set min and max values and give it some padding

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    # Generate a grid of points with distance h between them

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid
    X.set_value((np.c_[xx.ravel(), yy.ravel()]).astype('float32'))
    #Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = pred_func()
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


#print(timeit.timeit(build_model,number=1))
# plot_decision_boundary(lambda x: predict(model, x))   
# plt.title("Decision Boundary for hidden layer size 3")
#plt.figure(figsize=(16,32))
#hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]
#for i, nn_hdim in enumerate(hidden_layer_dimensions):
#    plt.subplot(5, 2, i+1)
#    plt.title('Hidden Layer size ' + str(nn_hdim))
#    model = build_model(nn_hdim)
#    plot_decision_boundary(lambda x: predict(model, x))
#plt.show()