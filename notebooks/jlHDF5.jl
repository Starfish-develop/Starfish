"""
jlHDF5.jl use the Julia language to read in spectral data stored in HDF5 format. Each wls, fls, and sigmas should be stored as a 2D array.

Load and return a "Dataset" object.

"""
module jlHDF5


import HDF5 

export DataSpectrum, openData

type DataSpectrum
    wls::Array{Float64, 2}
    fls::Array{Float64, 2}
    sigmas::Array{Float64, 2}
end

#Allow any 2D, floating point arrays to be given, but convert these Float64 for computation.
DataSpectrum(wls::Array{FloatingPoint, 2}, fls::Array{FloatingPoint, 2}, sigmas::Array{FloatingPoint, 2}) = DataSpectrum(float64(wls), float64(fls), float64(sigmas))

"""
Initialize a DataSpectrum type from a filename.
"""
function openData(filename::String)
    fid = HDF5.h5open(filename, "r")
    #Julia is indexed by column, row
    wls = fid["wls"][:,:]
    fls = fid["fls"][:,:]
    sigmas = fid["sigmas"][:,:]
    HDF5.close(fid)
    return DataSpectrum(wls, fls, sigmas)
end


##Example code
#mySpec = openData("../data/WASP14/WASP14-2009-06-14.hdf5")
#println(mySpec.wls[1:10,1])
#println(typeof(mySpec))
#println(typeof(mySpec.fls))

end
