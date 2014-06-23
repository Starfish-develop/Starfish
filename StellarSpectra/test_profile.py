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


# Additionally, if you want to do memory profiling using memory_profiler,
# put the decorator `@profile` over the function you want to profile
# And run with
#	python -m memory_profiler model.py
