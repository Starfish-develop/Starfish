#!/bin/bash

# Invoke this script from the top directory.

DIRECTORY="libraries/raw/PHOENIX/"

# Check to see if libraries/raw/ directory exists, if not, make it.
if [ ! -d "$DIRECTORY" ]; then
  echo $DIRECTORY does not exist, creating.
  mkdir -p $DIRECTORY
fi

cd $DIRECTORY

# Download the wavelength file
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits

# Create the Z=0.0, Z=-0.5, and Z=-1.0 directories, and download the appropriate spectra
Z0="Z-0.0"
Z05="Z-0.5"
Z10="Z-1.0"

mkdir $Z0
cd $Z0
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte06000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte06000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte06000-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte06100-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte06100-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte06100-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte06200-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte06200-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte06200-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits


cd ..
mkdir $Z05
cd $Z05

wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.5/lte06000-4.00-0.5.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.5/lte06000-4.50-0.5.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.5/lte06000-5.00-0.5.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.5/lte06100-4.00-0.5.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.5/lte06100-4.50-0.5.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.5/lte06100-5.00-0.5.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.5/lte06200-4.00-0.5.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.5/lte06200-4.50-0.5.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.5/lte06200-5.00-0.5.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits


cd ..
mkdir $Z10
cd $Z10

wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-1.0/lte06000-4.00-1.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-1.0/lte06000-4.50-1.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-1.0/lte06000-5.00-1.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-1.0/lte06100-4.00-1.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-1.0/lte06100-4.50-1.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-1.0/lte06100-5.00-1.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-1.0/lte06200-4.00-1.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-1.0/lte06200-4.50-1.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
wget ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-1.0/lte06200-5.00-1.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
