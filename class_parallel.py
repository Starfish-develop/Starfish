import multiprocessing as mp
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

class Parallel:
    def __init__(self):
        #self.pool = mp.Pool(4)
        #self.process = process
        pass

    def process(self, parameters):
        print(parameters)

    def do_parallel(self):
        pool = mp.Pool(4)
        param_list = np.arange(30)

        list(pool.map(self.process, param_list))

copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)

class MyClass(object):

    def __init__(self):
        self.my_args = [1,2,3,4]
        self.output  = {}

    def my_single_function(self, arg):
        return arg**2

    def my_parallelized_function(self):
        # Use map or map_async to map my_single_function onto the
        # list of self.my_args, and append the return values into
        # self.output, using each arg in my_args as the key.

        # The result should make self.output become
        # {1:1, 2:4, 3:9, 4:16}
        self.output = dict(zip(self.my_args,
                               pool.map(self.my_single_function, self.my_args)))

pool = mp.Pool()
foo = MyClass()
foo.my_parallelized_function()
print(foo.output)

#Class parallel methods will work
#1) If the function is defined at the top level of the module, or the class does not include Pool
#2) The pool object is not defined as a class object but instead as a local variable (it means NO pool object may be
# referenced in the class
#3) They will also work if you define the self.process method as a reference to a TOP level module which is
#outside the class
#4) If the pool object is defined as part of the class, you can still do map as long as process is defined (or referenced
# to outside the scope).

#If the function is bound to the class, it is a METHOD __class__
#If the function is referenced to something outside the class, it is a FUNCTION __class__

#EMCEE works because lnprob is defined somewhere else

#Basically, we need to separate the HDF5 file from the actual processing of the data. It sounds like if you really
#want to use parallel functions with Python, then it's better to use MPI. Will we have to use MPI for our
#cached interpolator? Probably.

#They will not work if you attempt to do both at the same time, because when it passes self.function, self also includes
#the pool object.
#Things can also be complicated by HDF5 files, which probably do not like to be pickled.

#def process(parameters):
#    print(parameters)
#
#class _function_wrapper(object):
#    """
#    This is a hack to make the likelihood function pickleable when ``args``
#    are also included.
#
#    """
#    def __init__(self, f, args):
#        self.f = f
#        self.args = args
#
#    def __call__(self, x):
#        try:
#            return self.f(x, *self.args)
#        except:
#            import traceback
#            print("emcee: Exception while calling your likelihood function:")
#            print("  params:", x)
#            print("  args:", self.args)
#            print("  exception:")
#            traceback.print_exc()
#            raise


#if __name__=="__main__":
#par = Parallel()
#par.do_parallel()

