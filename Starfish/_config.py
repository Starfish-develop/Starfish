import os
import shutil

import oyaml as yaml

from . import DEFAULT_CONFIG_FILE


class Config:
    def __contains__(self, key):
        return key in self._config

    def __delitem__(self, key):
        del self._config[key]

    def __eq__(self, other):
        return self._config.__eq__(other)

    def __getitem__(self, key):
        return self._config[key]

    def __init__(self, path):
        self._path = path
        self._protect_rewrites = True

        with open(self._path, 'r') as fd:
            # self._config = yaml.safe_load(fd.read(), YamlLoader(self))
            self._config = yaml.safe_load(fd)

        self._protect_rewrites = os.path.abspath(path) == DEFAULT_CONFIG_FILE

    def __setitem__(self, key, value):
        ret = self._config.__setitem__(key, value)
        return ret

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            return super().__getattribute__(key)

    def __setattr__(self, key, value):
        if key not in ['_path', '_protect_rewrites', '_config']:
            if key in self:
                self.__setitem__(key, value)
                self._rewrite()

        super().__setattr__(key, value)

    def _rewrite(self):
        if self._protect_rewrites:
            raise RuntimeError("The default file is not allowed to be overwritten. Please copy a file using "
                               "config.copy_file(<path>) for your use.")
        with open(self._path, 'w') as fd:
            yaml.safe_dump(self._config, fd)

    def update(self, d={}, **kwargs):
        protected_rewrites = self._protect_rewrites
        self._protect_rewrites = True

        self._config.update(d, **kwargs)

        self._protect_rewrites = protected_rewrites

        self._rewrite()

    def change_file(self, filename):
        self._path = filename
        with open(self._path, 'r') as fd:
            self._config = yaml.safe_load(fd)

    def copy_file(self, directory, switch=True):
        outname = os.path.join(directory, 'config.yaml')
        shutil.copy(DEFAULT_CONFIG_FILE, outname)
        if switch:
            self.change_file(outname)
