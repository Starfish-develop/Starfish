#!/usr/bin/env python
# encoding: utf-8
# filename: myprofile.py

import pstats, cProfile

import test_cov_wrap

cProfile.runctx("test_cov_wrap.update()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
