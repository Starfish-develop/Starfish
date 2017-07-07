#!/bin/bash
#
# download PHOENIX HiRes model spectra (enlightened by Ian Czekala)
# could work on the bigmac (bigmac => Dropbox => local)
#
# PHOENIX: ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/
#
##########################################
## Author: ZJ Zhang (zhoujian@hawaii.edu)
## Jul. 06th, 2017
##########################################

####### MODIFICATION HISTORY:
#
#############################

### 1. Setting Download directory 
Directory="/Users/zhoujian/Dropbox/PHOENIX/"
# Check to see if libraries/raw/ directory exists, if not, make one - 
if [ ! -d "$DIRECTORY" ]; then
  echo $DIRECTORY does not exist, creating.
  mkdir -p $DIRECTORY
fi
# --

### 2. Start downloading...
# 2.1 Preparation
cd $DIRECTORY
# 2.2 Wavelength File
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
# 2.3 Model Spectra
wget -r --no-parent ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/
# --

### 3. Move PHOENIX folder to the target place
# could do it by hand
