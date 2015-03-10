from emcee import GibbsSampler, ParallelSampler

class Sampler(GibbsSampler):
    '''
    Subclasses the GibbsSampler in emcee

    :param cov:
    :param starting_param_dict: the dictionary of starting parameters
    :param cov: the MH proposal
    :param revertfn:
    :param acceptfn:
    :param debug:

    '''

    def __init__(self, **kwargs):
        self.dim = len(self.param_tuple)
        #p0 = np.empty((self.dim,))
        #starting_param_dict = kwargs.get("starting_param_dict")
        #for i,param in enumerate(self.param_tuple):
        #    p0[i] = starting_param_dict[param]

        kwargs.update({"dim":self.dim})
        #self.spectra_list = kwargs.get("spectra_list", [0])

        super(Sampler, self).__init__(**kwargs)

        #Each subclass will have to overwrite how it parses the param_dict into the correct order
        #and sets the param_tuple

        #SUBCLASS here and define self.param_tuple
        #SUBCLASS here and define self.lnprob
        #SUBCLASS here and do self.revertfn
        #then do super().__init__() to call the following code

        self.outdir = kwargs.get("outdir", "")

    def startdict_to_tuple(self, startdict):
        raise NotImplementedError("To be implemented by a subclass!")

    def zip_p(self, p):
        return dict(zip(self.param_tuple, p))

    def lnprob(self):
        raise NotImplementedError("To be implemented by a subclass!")

    def revertfn(self):
        raise NotImplementedError("To be implemented by a subclass!")

    def acceptfn(self):
        raise NotImplementedError("To be implemented by a subclass!")

    def write(self):
        '''
        Write all of the relevant sample output to an HDF file.

        Write the lnprobability to an HDF file.

        flatchain
        acceptance fraction
        tuple parameters as an attribute in the header from self.param_tuple

        The actual HDF5 file is structured as follows

        /
        stellar parameters.flatchain
        00/
        ...
        22/
        23/
            global_cov.flatchain
            regions/
                region1.flatchain

        Everything can be saved in the dataset self.fname

        '''

        filename = self.outdir + "flatchains.hdf5"
        self.logger.debug("Opening {} for writing HDF5 flatchains".format(filename))
        hdf5 = h5py.File(filename, "w")
        samples = self.flatchain

        self.logger.debug("Creating dataset with fname:{}".format(self.fname))
        dset = hdf5.create_dataset(self.fname, samples.shape, compression='gzip', compression_opts=9)
        self.logger.debug("Storing samples and header attributes.")
        dset[:] = samples
        dset.attrs["parameters"] = "{}".format(self.param_tuple)
        dset.attrs["acceptance"] = "{}".format(self.acceptance_fraction)
        dset.attrs["acor"] = "{}".format(self.acor)
        dset.attrs["commit"] = "{}".format(C.get_git_commit())
        hdf5.close()

        #lnprobability is the lnprob at each sample
        filename = self.outdir + "lnprobs.hdf5"
        self.logger.debug("Opening {} for writing HDF5 lnprobs".format(filename))
        hdf5 = h5py.File(filename, "w")
        lnprobs = self.lnprobability

        dset = hdf5.create_dataset(self.fname, samples.shape[:1], compression='gzip', compression_opts=9)
        dset[:] = lnprobs
        dset.attrs["commit"] = "{}".format(C.get_git_commit())
        hdf5.close()

    def plot(self, triangle_plot=False):
        '''
        Generate the relevant plots once the sampling is done.
        '''
        samples = self.flatchain

        plot_walkers(self.outdir + self.fname + "_chain_pos.png", samples, labels=self.param_tuple)

        if triangle_plot:
            import triangle
            figure = triangle.corner(samples, labels=self.param_tuple, quantiles=[0.16, 0.5, 0.84],
                                     show_titles=True, title_args={"fontsize": 12})
            figure.savefig(self.outdir + self.fname + "_triangle.png")

            plt.close(figure)

class PSampler(ParallelSampler):
    '''
    Subclasses the GibbsSampler in emcee

    :param cov:
    :param starting_param_dict: the dictionary of starting parameters
    :param cov: the MH proposal
    :param revertfn:
    :param acceptfn:
    :param debug:

    '''

    def __init__(self, **kwargs):
        self.dim = len(self.param_tuple)
        #p0 = np.empty((self.dim,))
        #starting_param_dict = kwargs.get("starting_param_dict")
        #for i,param in enumerate(self.param_tuple):
        #    p0[i] = starting_param_dict[param]

        kwargs.update({"dim":self.dim})
        #self.spectra_list = kwargs.get("spectra_list", [0])

        super(PSampler, self).__init__(**kwargs)

        #Each subclass will have to overwrite how it parses the param_dict into the correct order
        #and sets the param_tuple

        #SUBCLASS here and define self.param_tuple
        #SUBCLASS here and define self.lnprob
        #SUBCLASS here and do self.revertfn
        #then do super().__init__() to call the following code

        self.outdir = kwargs.get("outdir", "")

    def startdict_to_tuple(self, startdict):
        raise NotImplementedError("To be implemented by a subclass!")

    def zip_p(self, p):
        return dict(zip(self.param_tuple, p))

    def lnprob(self):
        raise NotImplementedError("To be implemented by a subclass!")

    def revertfn(self):
        raise NotImplementedError("To be implemented by a subclass!")

    def acceptfn(self):
        raise NotImplementedError("To be implemented by a subclass!")

    def write(self):
        '''
        Write all of the relevant sample output to an HDF file.

        Write the lnprobability to an HDF file.

        flatchain
        acceptance fraction
        tuple parameters as an attribute in the header from self.param_tuple

        The actual HDF5 file is structured as follows

        /
        stellar parameters.flatchain
        00/
        ...
        22/
        23/
            global_cov.flatchain
            regions/
                region1.flatchain

        Everything can be saved in the dataset self.fname

        '''

        filename = self.outdir + "flatchains.hdf5"
        self.logger.debug("Opening {} for writing HDF5 flatchains".format(filename))
        hdf5 = h5py.File(filename, "w")
        samples = self.flatchain

        self.logger.debug("Creating dataset with fname:{}".format(self.fname))
        dset = hdf5.create_dataset(self.fname, samples.shape, compression='gzip', compression_opts=9)
        self.logger.debug("Storing samples and header attributes.")
        dset[:] = samples
        dset.attrs["parameters"] = "{}".format(self.param_tuple)
        dset.attrs["acceptance"] = "{}".format(self.acceptance_fraction)
        dset.attrs["acor"] = "{}".format(self.acor)
        dset.attrs["commit"] = "{}".format(C.get_git_commit())
        hdf5.close()

        #lnprobability is the lnprob at each sample
        filename = self.outdir + "lnprobs.hdf5"
        self.logger.debug("Opening {} for writing HDF5 lnprobs".format(filename))
        hdf5 = h5py.File(filename, "w") #creates if doesn't exist, otherwise read/write
        lnprobs = self.lnprobability

        dset = hdf5.create_dataset(self.fname, samples.shape[:1], compression='gzip', compression_opts=9)
        dset[:] = lnprobs
        dset.attrs["commit"] = "{}".format(C.get_git_commit())
        hdf5.close()

    def plot(self, triangle_plot=False):
        '''
        Generate the relevant plots once the sampling is done.
        '''
        samples = self.flatchain

        plot_walkers(self.outdir + self.fname + "_chain_pos.png", samples, labels=self.param_tuple)

        if triangle_plot:
            import triangle
            figure = triangle.corner(samples, labels=self.param_tuple, quantiles=[0.16, 0.5, 0.84],
                                     show_titles=True, title_args={"fontsize": 12})
            figure.savefig(self.outdir + self.fname + "_triangle.png")

            plt.close(figure)

class StellarSampler(PSampler):
    """
    Subclasses the Sampler specifically for stellar parameters



    """
    def __init__(self, **kwargs):
        '''
        :param pconns: Collection of parent ends of the PIPEs
        :type pconns: dict

        :param starting_param_dict:
            the dictionary of starting parameters

        :param cov:
            the MH proposal

        :param fix_logg:
            fix logg? If so, to what value?

        :param debug:

        :param args: []
        '''

        self.fix_logg = kwargs.get("fix_logg", False)
        starting_pram_dict = kwargs.get("starting_param_dict")
        self.param_tuple = self.startdict_to_tuple(starting_pram_dict)
        print("param_tuple is {}".format(self.param_tuple))
        self.p0 = np.array([starting_pram_dict[key] for key in self.param_tuple])

        kwargs.update({"p0":self.p0, "revertfn":self.revertfn, "acceptfn": self.acceptfn, "lnprobfn":self.lnprob})
        super(StellarSampler, self).__init__(**kwargs)

        #self.pconns is a dictionary of parent connections to each PIPE connecting to the child processes.
        self.spectrum_ids = sorted(self.pconns.keys())
        self.fname = "stellar"

    def startdict_to_tuple(self, startdict):
        tup = ()
        for param in C.stellar_parameters:
            #check if param is in keys, if so, add to the tuple
            if param in startdict:
                tup += (param,)
        return tup

    def reset(self):
        super(StellarSampler, self).reset()

    def revertfn(self):
        '''
        Revert the model to the previous state of parameters, in the case of a rejected MH proposal.
        '''
        self.logger.debug("reverting stellar parameters")
        self.prior = self.prior_last

        #Decide we don't want these stellar params. Tell the children to reject the proposal.
        for pconn in self.pconns.values():
            pconn.send(("DECIDE", False))

    def acceptfn(self):
        '''
        Execute this if the MH proposal is accepted.
        '''
        self.logger.debug("accepting stellar parameters")
        #Decide we do want to keep these stellar params. Tell the children to accept the proposal.
        for pconn in self.pconns.values():
            pconn.send(("DECIDE", True))

    def lnprob(self, p):
        # We want to send the same stellar parameters to each model,
        # but also send the different vz and logOmega parameters
        # to the separate spectra, based upon spectrum_id.
        #self.logger.debug("StellarSampler lnprob p is {}".format(p))

        #Extract only the temp, logg, Z, vsini parameters
        if not self.fix_logg:
            params = self.zip_p(p[:4])
            others = p[4:]
        else:
            #Coming in as temp, Z, vsini, vz, logOmega...
            params = self.zip_p(p[:3])
            others = p[3:]
            params.update({"logg": self.fix_logg})

        # Prior
        self.prior_last = self.prior

        logg = params["logg"]
        self.prior = -0.5 * (logg - 5.0)**2/(0.05)**2

        #others should now be either [vz, logOmega] or [vz0, logOmega0, vz1, logOmega1, ...] etc. Always div by 2.
        #split p up into [vz, logOmega], [vz, logOmega] pairs that update the other parameters.
        #mparams is now a list of parameter dictionaries

        #Now, pack up mparams into a dictionary to send the right stellar parameters to the right subprocesses
        mparams = {}
        for (spectrum_id, order_id), (vz, logOmega) in zip(self.spectrum_ids, grouper(others, 2)):
            p = params.copy()
            p.update({"vz":vz, "logOmega":logOmega})
            mparams[spectrum_id] = p

        self.logger.debug("updated lnprob params: {}".format(mparams))

        lnps = np.empty((self.nprocs,))

        #Distribute the calculation to each process
        self.logger.debug("Distributing params to children")
        for ((spectrum_id, order_id), pconn) in self.pconns.items():
            #Parse the parameters into what needs to be sent to each Model here.
            pconn.send(("LNPROB", mparams[spectrum_id]))

        #Collect the answer from each process
        self.logger.debug("Collecting params from children")
        for i, pconn in enumerate(self.pconns.values()):
            lnps[i] = pconn.recv()

        self.logger.debug("lnps : {}".format(lnps))
        s = np.sum(lnps)
        self.logger.debug("sum lnps {}".format(s))
        return s + self.prior

class NuisanceSampler(Sampler):
    def __init__(self, **kwargs):
        '''

        :param OrderModel: the parallel.OrderModel instance

        :param starting_param_dict: the dictionary of starting parameters

        :param cov:
            the MH proposal

        :param debug:

        :param args: []

        '''

        starting_param_dict = kwargs.get("starting_param_dict")
        self.param_tuple = self.startdict_to_tuple(starting_param_dict)
        print("param_tuple is {}".format(self.param_tuple))
        #print("param_tuple length {}".format(len(self.param_tuple)))

        chebs = [starting_param_dict["cheb"][key] for key in self.cheb_tup]
        covs = [starting_param_dict["cov"][key] for key in self.cov_tup]
        regions = starting_param_dict["regions"]
        #print("initializing {}".format(regions))
        regs = [regions[id][kk] for id in sorted(regions) for kk in C.cov_region_parameters]
        #print("regs {}".format(regs))

        self.p0 = np.array(chebs + covs + regs)

        kwargs.update({"p0":self.p0, "revertfn":self.revertfn, "lnprobfn":self.lnprob})
        super(NuisanceSampler, self).__init__(**kwargs)

        self.model = kwargs.get("OrderModel")
        spectrum_id, order_id = self.model.id
        order = kwargs.get("order", order_id)
        #self.fname = "{}/{}/{}".format(spectrum_id, order, "nuisance")
        self.fname = "nuisance"
        self.params = None
        self.prior_params = kwargs.get("prior_params", None)
        if self.prior_params:
            self.sigma0 = self.prior_params["regions"]["sigma0"]
            self.mus = self.prior_params["regions"]["mus"]
            self.mu_width = self.prior_params["regions"]["mu_width"]
            self.sigma_knee = self.prior_params["regions"]["sigma_knee"]
            self.frac_global = self.prior_params["regions"]["frac_global"]

    def startdict_to_tuple(self, startdict):
        #This is a little more tricky than the stellar parameters.
        #How are the keys stored and passed in the dictionary?
        #{"cheb": [c0, c1, c2, ..., cn], "cov": [sigAmp, logAmp, l],
        #        "regions":{0: [logAmp, ], 1: [], N:[] }}

        #Serialize the cheb parameters
        self.ncheb = len(startdict["cheb"])
        self.cheb_tup = ("logc0",) + tuple(["c{}".format(i) for i in range(1, self.ncheb)])

        #Serialize the covariance parameters
        self.ncov = 3
        cov_tup = ()
        for param in C.cov_global_parameters:
            #check if param is in keys, if so, add to the tuple
            if param in startdict["cov"]:
                cov_tup += (param,)
        self.cov_tup = cov_tup

        regions_tup = ()
        self.regions = startdict.get("regions", None)
        if self.regions:
            self.nregions = len(self.regions)
            for key in sorted(self.regions.keys()):
                for kk in C.cov_region_parameters:
                    regions_tup += ("r{:0>2}-{}".format(key,kk),)
            self.regions_tup = regions_tup
        else:
            self.nregions = 0
            self.regions_tup = ()


        tup = self.cheb_tup + self.cov_tup + self.regions_tup
        #This should look like
        #tup = ("c0", "c1", ..., "cn", "sigAmp", "logAmp", "l", "r00_logAmp", "r00_mu", "r00_sigma",
        # "r01_logAmp", ..., "rNN_sigma")
        return tup

    def zip_p(self, p):
        '''
        Convert the vector to a dictionary
        '''
        cheb = dict(zip(self.cheb_tup, p[:self.ncheb]))
        cov = dict(zip(self.cov_tup, p[self.ncheb:self.ncheb+self.ncov]))
        regions = p[-self.nregions*3:]
        rdict = {}
        for i in range(self.nregions):
            rdict[i] = dict(zip(("logAmp", "mu", "sigma"), regions[i*3:3*(i+1)]))

        params = {"cheb":cheb, "cov":cov, "regions":rdict}
        return params

    def revertfn(self):
        self.logger.debug("reverting model")
        self.model.prior = self.prior_last
        self.params = self.params_last
        self.model.revert_nuisance()

    def lnprob(self, p):
        self.params_last = self.params
        params = self.zip_p(p)
        self.params = params
        self.logger.debug("Updating nuisance params {}".format(params))

        # Nuisance parameter priors implemented here
        self.prior_last = self.model.prior
        # Region parameter priors implemented here
        if self.nregions > 0:
            regions = params["regions"]
            keys = sorted(regions)

            #Unpack the region parameters into a vector of mus, amps, and sigmas
            amps = 10**np.array([regions[key]["logAmp"] for key in keys])
            cov_amp = 10**params["cov"]["logAmp"]

            #First check to make sure that amplitude can't be some factor less than the global covariance
            if np.any(amps < (cov_amp * self.frac_global)):
                return -np.inf

            mus = np.array([regions[key]["mu"] for key in keys])
            sigmas = np.array([regions[key]["sigma"] for key in keys])

            #Make sure the region hasn't strayed too far from the original specification
            if np.any(np.abs(mus - self.mus) > self.sigma0):
                # The region has strayed too far from the original specification
                return -np.inf

            #Use a Gaussian prior on mu, that it keeps the region within the original setting.
            # 1/(sqrt(2pi) * sigma) exp(-0.5 (mu-x)^2/sigma^2)
            #-ln(sigma * sqrt(2 pi)) - 0.5 (mu - x)^2 / sigma^2
            #width = 0.04
            lnGauss = -0.5 * np.sum(np.abs(mus - self.mus)**2/self.mu_width**2 -
                                    np.log(self.mu_width * np.sqrt(2. * np.pi)))

            # Use a ln(logistic) function on sigma, that is flat before the knee and dies off for anything
            # greater, to prevent dilution into global cov kernel
            lnLogistic = np.sum(np.log(-1./(1. + np.exp(self.sigma_knee - sigmas)) + 1.))

            self.model.prior = lnLogistic + lnGauss

        try:
            self.model.update_nuisance(params)
            lnp = self.model.evaluate() # also sets OrderModel.lnprob to proposed value. Includes self.model.prior
            return lnp
        except C.ModelError:
            return -np.inf
