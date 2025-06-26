# pygeoinf

This is a package for solving inverse and inference problems with an emphasis on problems posed on infinite-dimensional Hilbert spaces. Currently, only methods for linear problems have been implemented, but the addition of functionality for non-linear problems is planned for the future. 

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

You can run our interactive tutorials directly in Google Colab:

| Tutorial Name          | Link to Colab                                                                                                                                                                                                                                    |
| :--------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tutorial 1        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pygeoinf/blob/develop/tutorials/t1.ipynb)                                                  |
| Advanced Features      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your_github_username/my_python_package/blob/main/examples/advanced_tutorial.ipynb)                                            |

Alternatively, you can install the package locally and run the notebooks.


