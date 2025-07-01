# pygeoinf

This is a package for solving inverse and inference problems with an emphasis on problems posed on infinite-dimensional Hilbert spaces. Currently the methods are restricted to linear problems.


## Installation


### Using pip

The package can be installed using pip through:

```
pip install pygeoinf
```

### Using poetry

This package can also be installed using poetry (https://python-poetry.org/). Clone the repository and from within that director type:

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
poetry add pygeoinf
```

or using 

```
poetry add git+https://github.com/da380/pygeoinf
```

to link directly to the git repository.



## Tutorials

You can run the interactive tutorials directly in Google Colab:

| Tutorial Name          | Link to Colab                                                                                                                                                                                                                                    |
| :--------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tutorial 1 - A first example       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/t1.ipynb)                                                  |
| Tutorial 2 - Hilbert spaces       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/t2.ipynb)                                       |
| Tutorial 3 - Dual spaces       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/t3.ipynb)                                       |
| Tutorial 4 - Linear operators       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/t4.ipynb)                                       |
| Tutorial 5 - Gaussian measures       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/t5.ipynb)                                       |
| Tutorial 6 - Minumum norm inversions      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/t6.ipynb)                                       |
| Tutorial 7 - Bayesian inversions     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/main/tutorials/t7.ipynb)                                       |


Alternatively, you can install the package locally and run the notebooks.


