Installation
============

All the examples are written in Python 3, within a virtualenv. External dependencies are managed by pip. The details 
about installation of these tools, may be found on their website. For the purposes of the introduction, we will assume 
that all necessary applications have been already installed.

Setting up virtualenv and dependencies
--------------------------------------
The following command creates a virtualenv with Python 3 interpreter in the `.virtualenv` directory.

```
virtualenv -ppython3 --system-site-packages .virtualenv
```

Installation of the TensorFlow cannot be done using standard pip-based approach. Instead of, a package needs to be 
installed manually. The process is described in details on the official website of the library: 
[TensorFlow installation][tensorflow_pip]. Please take care to follow this description and choose appropriate version of 
the library for your OS before going further.

All the dependencies will be kept in `requirements.txt` file, created in the main directory of the project.

```
tensorflow==0.11.0
```

Installation of necessary dependencies may be done using the following command:

```
pip3 install -r requirements.txt
```

[tensorflow_pip]: https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#pip-installation