import cProfile
import pstats
#import trim_PHOENIX
#import model
import MCMC



cProfile.run("MCMC.main()","prof_MCMC_parallel_32")

def display_stats(pfile):
    p = pstats.Stats(pfile)
    p.sort_stats('cumulative').print_stats(.9)
    #p.sort_stats('name').print_stats()
    p.sort_stats('time').print_stats(.9)

display_stats('prof_MCMC_parallel_32')
