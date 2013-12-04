import cProfile
import pstats

#Your module here
import model
#import plot_MCMC


cProfile.run("model.main()", "prof")
#cProfile.run("plot_MCMC.main()", "prof")


def display_stats(pfile):
    p = pstats.Stats(pfile)
    p.sort_stats('cumulative').print_stats(.9)
    p.sort_stats('time').print_stats(.9)


display_stats('prof')
