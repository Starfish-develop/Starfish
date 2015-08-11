
# coding: utf-8

# In[1]:

from matplotlib import rc
rc("font", family="serif", size=10)
rc("text", usetex=True)

import matplotlib.pyplot as plt
import daft


# In[2]:

# Instantiate the PGM.
pgm = daft.PGM([4.0, 2.3], origin=[0.3, 0.3])

# Parameters.
pgm.add_node(daft.Node("star", r"$\mathbf{\theta}_\star$", 0.9, 2))
pgm.add_node(daft.Node("w", r"$\mathbf{w}$", 1.8, 2, plot_params={"fill":True, "fc":"w"}))

pgm.add_node(daft.Node("model", r"$\mathbf{\mathsf{M}}$", 2.7, 2, plot_params={"fill":True, "fc":"w"}))

pgm.add_node(daft.Node("eig", r"$\mathbf{\Xi}$", 2.0, 1, fixed=True))
pgm.add_node(daft.Node("obs", r"$\mathbf{\theta}_\textrm{ext}$", 2.55, 1))
pgm.add_node(daft.Node("phi", r"$\phi_{\mathsf{P}}$", 3.1, 1))

pgm.add_node(daft.Node("data", r"$\mathbf{\mathsf{D}}$", 3.7, 2, observed=True))
pgm.add_node(daft.Node("nuis", r"$\phi_\mathsf{C}$", 3.7, 1))



# Add in the edges.
pgm.add_edge("star", "w")
pgm.add_edge("w", "model")
pgm.add_edge("model", "data")



pgm.add_edge("eig", "model")
pgm.add_edge("obs", "model")
pgm.add_edge("phi", "model")
pgm.add_edge("nuis", "data")

pgm.render()
pgm.figure.savefig("../../plots/PGM.svg")
pgm.figure.savefig("../../plots/PGM.pdf")
pgm.figure.savefig("../../plots/PGM.png")
plt.close()



# In[ ]:



