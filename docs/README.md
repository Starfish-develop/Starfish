# Building the documentation

After [dfm/python-fsps](https://github.com/dfm/python-fsps/tree/master/docs)

To build and deploy the documentation, follow these steps:

On the master branch, in this directory, run:

    make clean
    make html

Confirm that the results in `_build/html` look sensible.

Change to the root directory of the repository and run (changing `VERSION` to the correct version number):

    git checkout gh-pages
    mkdir VERSION
    cp -r docs/_build/html/* VERSION/
    git add VERSION
    git commit -m "Updated docs for version VERSION"
    git push

If you want to release this version as the default (stable) version, run:

    rm current
    ln -s VERSION current
    git add current
    git commit -m "Updating stable version to VERSION"
    git push

