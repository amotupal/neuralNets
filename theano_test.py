# import theano.tensor as T
# from theano import function
# import numpy
# # x = T.dscalar('x')
# # y = x
# # f = function([x],y)

# x = T.dmatrix('x')
# y = T.dmatrix('y')
# z = x + y
# f = function([x,y],z)
# print(f(numpy.array([[1, 2], [3, 4]]), numpy.array([[10, 20], [30, 40]])))
# # print(f([[1, 2], [3, 4]], [[10, 20], [30, 40]]))

# import theano
# a = theano.tensor.vector()
# b = theano.tensor.vector()
# # out = a + a ** 10
# out = a ** 2 + b ** 2 + 2 * a  * b 

# f = theano.function([a,b], out)
# print(f([0,1,2], [1,2,3]))
