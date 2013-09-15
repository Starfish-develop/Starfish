import cProfile
import pstats
#import trim_PHOENIX
#import model
#import MCMC
#import synthphot
import synth_homebrew



cProfile.run("synth_homebrew.main()","prof_home")

def display_stats(pfile):
    p = pstats.Stats(pfile)
    p.sort_stats('cumulative').print_stats(.9)
    #p.sort_stats('name').print_stats()
    p.sort_stats('time').print_stats(.9)

display_stats('prof_home')
