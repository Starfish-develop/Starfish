import yaml


class Config:
    """
    This is a class to allow lazy-loading of config files. This means that the atributes are only read from the file
    when accessed in the code, allowing for changes in the file without having to restart the python instance
    """

    def __init__(self, filename):
        self.filename = filename
        # Format string for saving/reading orders
        self.specfmt = "s{}_o{}"

    def __getitem__(self, item):
        with open(self.filename) as f:
            base = yaml.safe_load(f)
            return base[item]

    def __getattr__(self, item):
        with open(self.filename) as f:
            base = yaml.safe_load(f)
            return base[item]

    def __contains__(self, item):
        if item == 'filename' or item == 'specfmt':
            return True

        with open(self.filename) as f:
            base = yaml.safe_load(f)
            return item in base