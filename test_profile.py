import cProfile
import pstats
#import trim_PHOENIX
import model
#import MCMC
#import synthphot
#import synth_homebrew
#import test_multi
#import fft_interpolate

cProfile.run("model.main()", "prof")


def display_stats(pfile):
    p = pstats.Stats(pfile)
    p.sort_stats('cumulative').print_stats(.9)
    #p.sort_stats('name').print_stats()
    p.sort_stats('time').print_stats(.9)


display_stats('prof')
