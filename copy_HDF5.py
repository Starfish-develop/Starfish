#!python

#Check to make sure the file has been copied to the node /scratch properly.

#invoke with the following command line arguments
#check_HDF5.sh existing.hdf5 destination.hdf5

import sys
import os
import subprocess

existing = sys.argv[1]
destination = sys.argv[2]

#compute MD5 sum of existing.hdf5
def get_md5(file):
    proc = subprocess.Popen(["md5sum", file], stdout=subprocess.PIPE)
    out, err = proc.communicate()
    out = out.decode('UTF-8')
    #strip the filename and keep the code
    return out.split()[0]

md_existing = get_md5(existing)
print("md5sum of {} is {}".format(existing, md_existing))

#check if destination.hdf5 exists
if os.path.exists(destination):
    print("{} exists".format(destination))

    md_destination = get_md5(destination)
    print("md5sum of {} is {}".format(existing, md_destination))
    #Do the md5sums match up?
    if md_existing == md_destination:
        print("md5 sums are equal")
    else:
        import time
        tries = 0
        while (md_existing != md_destination) and tries < 5:
            print("md5 sums are not equal, waiting for 2 minutes")
            time.sleep(120)
            md_destination = get_md5(destination)
            print("md5sum of {} is {}".format(existing, md_destination))
            tries += 1
        if md_existing != md_destination:
            print("Failed to copy new file.")
            raise IOError("Unable to transfer grid.")

else:
    print("{} does not exist, copying".format(destination))
    subprocess.call(["cp", existing, destination])