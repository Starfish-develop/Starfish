====
MCMC
====

Driver scripts.

Creating the parameter script
=============================

All parameters are specified in YAML. The specification is [here](http://www.yaml.org/spec/1.2/spec.html) and the
 python interface is documented [here](http://pyyaml.org/wiki/PyYAMLDocumentation). YAML is a very powerful configuration
 language but it's also very easy to use, so don't be intimidated. It's a highly worthwhile format to learn,
 and you can probably figure out how it works from the example scripts.

Additionally, you can also configure the fitting runs by directly editing the python script ``scripts/stars/base_lnrob``
If you wanted to, you could also remove the YAML dependencey and just declare your variables in the script, but I think
it is nice to keep them separate for organization's sake.
