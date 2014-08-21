#!/usr/bin/env python

import hdfutils

hdfutils.GR_list(hdfutils.flatchainTreeList, burn=hdfutils.args.burn, thin=hdfutils.args.thin)