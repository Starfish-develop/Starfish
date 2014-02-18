import numpy as np

class Data:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class LineModel:
    def __init__(self, x, params):
        self.params = params
        self.x = x
        self.model()

    def model(self):
        self.y = self.params[0] * self.x + self.params[1]

    def set_params(self,params):
        self.params = params
        self.model()

class LineModelFlag:
    def __init__(self, x, params):
        self.params = params
        self.x = x
        self.flags = np.ones_like(self.x, dtype='b') #boolean flags to mark a data point as "good" (=1) or "bad" (=0)
        self.model()                                 #start with all points considered good.

    def model(self):
        self.y = self.params[0] * self.x + self.params[1]

    def set_params(self,params):
        self.params = params
        self.model()

    def set_flags(self,flags):
        self.flags = flags