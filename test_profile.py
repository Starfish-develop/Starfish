import cProfile
import pstats

#Your module here
import model


cProfile.run("model.main()", "prof")


def display_stats(pfile):
    p = pstats.Stats(pfile)
    p.sort_stats('cumulative').print_stats(.9)
    p.sort_stats('time').print_stats(.9)


display_stats('prof')
