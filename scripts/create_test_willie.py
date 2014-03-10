from StellarSpectra.grid_tools import HDF5Interface, Interpolator, KPNO, TRES, MasterToFITSIndividual


myHDF5Interface = HDF5Interface("/n/holyscratch/panstarrs/iczekala/master_grids/PHOENIX_master.hdf5")

myInterpolator = Interpolator(myHDF5Interface, avg_hdr_keys=["air", "PHXLUM", "PHXMXLEN",
                 "PHXLOGG", "PHXDUST", "PHXM_H", "PHXREFF", "PHXXI_L", "PHXXI_M", "PHXXI_N", "PHXALPHA", "PHXMASS",
                                                         "norm", "PHXVER", "PHXTEFF"])


outKPNO = "/n/home07/iczekala/StellarSpectra/libraries/willie/KPNO/"
outKPNOfnu = "/n/home07/iczekala/StellarSpectra/libraries/willie/KPNOfnu/"
outTRES = "/n/home07/iczekala/StellarSpectra/libraries/willie/TRES/"
outTRESfnu = "/n/home07/iczekala/StellarSpectra/libraries/willie/TRESfnu/"

params_hot = {"temp":6000, "logg":4.5, "Z":0.0, "vsini":8}
params_cool = {"temp":4000, "logg":4.5, "Z":0.0, "vsini":4}

KPNOcreator = MasterToFITSIndividual(interpolator=myInterpolator, instrument=KPNO())
KPNOcreator.process_spectrum(params_hot, out_unit="f_nu_log", out_dir=outKPNO)
KPNOcreator.process_spectrum(params_cool, out_unit="f_nu_log", out_dir=outKPNO)
KPNOcreator.process_spectrum(params_hot, out_unit="f_nu", out_dir=outKPNOfnu)
KPNOcreator.process_spectrum(params_cool, out_unit="f_nu", out_dir=outKPNOfnu)


TREScreator = MasterToFITSIndividual(interpolator=myInterpolator, instrument=TRES())
TREScreator.process_spectrum(params_hot, out_unit="f_nu_log", out_dir=outTRES)
TREScreator.process_spectrum(params_cool, out_unit="f_nu_log", out_dir=outTRES)
TREScreator.process_spectrum(params_hot, out_unit="f_nu", out_dir=outTRESfnu)
TREScreator.process_spectrum(params_cool, out_unit="f_nu", out_dir=outTRESfnu)