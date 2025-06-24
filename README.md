# pygeoinf

This is a package for solving inverse and inference problems with an emphasis on problems posed on infinite-dimensional Hilbert spaces. Currently, only methods for linear problems have been implemented, but the addition of functionality for non-linear problems is planned for the future. 

## Installation


### Using pip

To be done...

### Using poetry

This package is most easily installed using poetry (https://python-poetry.org/). Clone the repository and from within that director type:

```
poetry install
```

The optional tutorials and tests can be added with the ```--with``` option. For example to install with the tutorials we need:

```
poetry install --with tutorials
```


Once installed, the virtual environment can be activated by typing:

```
$(poetry env activate)
```

Alternative, you can use ```poetry run``` to directly execute commands without starting up the virtual environment.

The library can be added as a dependency within a separate poetry project using:

```
poetry add git+https://github.com/da380/pygeoinf
```




