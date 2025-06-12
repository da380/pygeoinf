# pygeoinf

This is a package for solving inverse and inference problems with an emphasis on problems posed on infinite-dimensional Hilbert spaces. Currently, only methods for linear problems have been implemented, but the addition of functionality for non-linear problems is planned for the future. 

## Installation


### Using pip

To be done...

### Using poetry

This package is most easily installed using poetry. It can also be included within another poetry project diretly from github. 

Having cloned the repository, and assuming you have poetry installed, all that is needed to to enter the main directly and type:

```
poetry install
```
If you want to include the tutorials you add:

```
poetry install --with tutorials
```

and it you want the tests as well:

```
poetry install --with tutorials --with tests
```

Once installed, the virtual environment can be activated by typing:

```
$(poetry env activate)
```

Alternative, you can use ```poetry run ...``` to directly run commands without starting up the virtual environment. For example, typing

```
poetry run jupyter notebook
```

will launch in a browser a juptyer notebook from which you can navigate to the tutorials. 

### Getting started

To understand what this package can do it is recommended that you work through the tutorials. These cover most of the functionality of the package. Beyond this, you can use the help function to read the documentation linked to the various classes and their methods. 

