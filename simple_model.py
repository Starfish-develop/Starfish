import numpy as np
import copyreg
import types

def _pickle_method(method):
    # Author: Steven Bethard
    # http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    cls_name = ''
    if func_name.startswith('__') and not func_name.endswith('__'):
        cls_name = cls.__name__.lstrip('_')
    if cls_name:
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    # Author: Steven Bethard
    # http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)

m = 2.0
b = 1.0
x = np.linspace(0, 10, num=20)
y = m * x + b + np.random.normal(size=(20,))


class Data:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Model:
    def __init__(self, x, params):
        self.params = params
        self.x = x
        self.model()

    def model(self):
        self.y = self.params[0] * self.x + self.params[1]

    def set_params(self,params):
        self.params = params
        self.model()

class Lnprob:
    def __init__(self, data, model):
        self.data = data
        self.model = model

    def lnprob(self, params):
        self.model.set_params(params)
        return np.sum((self.data.y - self.model.y)**2)


myData = Data(x,y)
myModel = Model(x, np.array([1.5, 1.5]))
myLnprob = Lnprob(myData, myModel)

#print(myLnprob.lnprob(np.array([1.5, 1.5])))
