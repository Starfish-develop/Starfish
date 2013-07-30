import cProfile
import pstats
#import trim_PHOENIX
import model


cProfile.run("model.main()","prof2")

def display_stats(pfile):
    p = pstats.Stats(pfile)
    p.sort_stats('cumulative').print_stats(.1)
    #p.sort_stats('name').print_stats()
    p.sort_stats('time').print_stats(.1)

display_stats('prof2')
