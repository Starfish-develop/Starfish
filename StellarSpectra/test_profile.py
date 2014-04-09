import cProfile
import pstats

#Your module here
from StellarSpectra.tests import profile_evaluate


cProfile.run("profile_evaluate.main()", "prof")
#cProfile.run("plot_MCMC.main()", "prof")


def display_stats(pfile):
    p = pstats.Stats(pfile)
    p.sort_stats('cumulative').print_stats(.9)
    p.sort_stats('time').print_stats(.9)


display_stats('prof')
