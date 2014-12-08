#!/bin/bash

#Download the Castelli and Kurucz models from the STScI website
wget -cr -nH --cut-dirs=3 'ftp://ftp.stsci.edu/cdbs/grid/ck04models/*'
