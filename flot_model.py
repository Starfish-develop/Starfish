#!/usr/bin/env python

#Use flot JavaScript plotting library to visualize a single order plot

import numpy as np
import json
import jinja2

def np_to_json(arr0, arr1):
    '''
    Take two numpy arrays, as in a plot, and return the JSON serialization for flot.
    '''
    data = np.array([arr0, arr1]).T #Transpose the arrays
    listdata = data.tolist() #Convert numpy array to a list
    return json.dumps(listdata) #Serialize to JSON

def order_json(wl, fl, sigma, mask, flm, cheb):
    '''
    Given the quantities from a fit, create the JSON necessary for flot.
    '''

    residuals = fl - flm

    # create the three lines necessary for the plot, in JSON
    # data = [[wl0, fl0], [wl1, fl1], ...]
    # model = [wl0, flm0], [wl1, flm1], ...]
    # residuals = [[wl0, residuals0], [wl1, residuals1], ...]
    plot_data = {"data":np_to_json(wl[mask], fl[mask]), "model":np_to_json(wl, flm), "residuals": np_to_json(wl[mask],
                residuals[mask]), "sigma": np_to_json(wl[mask], sigma[mask]), "cheb": np_to_json(wl, cheb) }
    return plot_data


def render_template(base, plot_data):

    templateLoader = jinja2.FileSystemLoader(searchpath=base + "templates")
    # An environment provides the data necessary to read and
    # parse our templates.  We pass in the loader object here.
    templateEnv = jinja2.Environment(loader=templateLoader)
    template = templateEnv.get_template('flot_plot.jinja')

    templateVars = {"base": base}
    templateVars.update(plot_data)

    #Render plot using plot_data
    outputText = template.render(templateVars)
    f = open('index_flot.html', 'w')
    f.write(outputText)
    f.close()


def main():
    #Use argparse to determine if we've specified a config file
    import argparse
    parser = argparse.ArgumentParser(prog="flot_model.py", description="Plot the model and residuals using flot.")
    parser.add_argument("json", help="*.json file describing the model.")
    parser.add_argument("params", help="*.yaml file specifying run parameters.")
    # parser.add_argument("-o", "--output", help="*.html file for output")
    args = parser.parse_args()

    import json
    import yaml

    if args.json: #
        #assert that we actually specified a *.json file
        if ".json" not in args.json:
            import sys
            sys.exit("Must specify a *.json file.")

    if args.params: #
        #assert that we actually specified a *.yaml file
        if ".yaml" in args.params:
            yaml_file = args.params
            f = open(args.params)
            config = yaml.load(f)
            f.close()

        else:
            import sys
            sys.exit("Must specify a *.yaml file.")
            yaml_file = args.params

    from StellarSpectra.model import Model
    from StellarSpectra.spectrum import DataSpectrum
    from StellarSpectra.grid_tools import TRES, HDF5Interface

    #Figure out what the relative path is to base
    import StellarSpectra
    base = StellarSpectra.__file__[:-26]

    myDataSpectrum = DataSpectrum.open(base + config['data'], orders=config['orders'])
    myInstrument = TRES()
    myHDF5Interface = HDF5Interface(base + config['HDF5_path'])

    myModel = Model.from_json(args.json, myDataSpectrum, myInstrument, myHDF5Interface)

    for model in myModel.OrderModels:

        #If an order has regions, read these out from model_final.json
        region_dict = model.get_regions_dict()
        print("Region dict", region_dict)
        #loop through these to determine the wavelength of each
        wl_regions = [value["mu"] for value in region_dict.values()]

        #Make vertical markings at the location of the wl_regions.

        #Get the data, sigmas, and mask
        wl, fl, sigma, mask = model.get_data()

        #Get the model flux
        flm = model.get_spectrum()

        #Get chebyshev
        cheb = model.get_Cheb()

        name = "Order {}".format(model.order)

        plot_data = order_json(wl, fl, sigma, mask, flm, cheb)
        plot_data.update({"wl_regions":wl_regions})
        print(plot_data['wl_regions'])

        render_template(base, plot_data)

        #Get the covariance matrix
        # S = model.get_Cov()

if __name__=="__main__":
    main()
