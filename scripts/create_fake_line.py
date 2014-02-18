import numpy as np

'''Create a fake data set with a line in order to test bad data model.'''

@np.vectorize
def gaussian(x, mu=5150, sigma=2):
    return - 1/np.sqrt(2. * np.pi) * np.exp(-0.5 * (x - mu)**2/sigma**2)

xs = np.linspace(5100, 5200,num=2000)
ys = gaussian(xs) + 1.0 + np.random.normal(scale=0.02, size=2000)

np.save("data/Fake/line.wls.npy", xs)
np.save("data/Fake/line.fls.npy", ys)

#import matplotlib.pyplot as plt
#plt.plot(xs,ys)
#plt.show()
