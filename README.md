
We would like to use virtualenv to ensure that the projects are self contained.
For Linux:
http://friendly-101.readthedocs.io/en/latest/pipvirtualenv.html
https://tutorials.technology/tutorials/10-whats-is-a-virtualenv-and-how-to-install-it.html
For MacOSX:
Installing pip and virtualenv
We need to install pip. Following the instructions from https://pip.pypa.io/en/stable/installing/ . Note that while we can simply download get-pip.py, there is a concern that python 2.7 in MacOSX is built into the system and we may face errors. The safest route is to install python 3 as they recommended.
Install python3 from https://www.python.org/downloads/
Then, download get-pip.py from https://pip.pypa.io/en/stable/installing/
Navigate to the download directory and install pip via:
 python3 get-pip.py
Now. Install virtualenv using pip:
 sudo pip install virtualenv
Creating a Virtual Environment
Here is a nice introduction to pip and virtualenv for python: http://www.dabapps.com/blog/introduction-to-pip-and-virtualenv-python/ The key commands are the following:
To make a new virtual environment with env as the environemnt name:
 virtualenv env
In this case, we want to specify the python version number to be 2.7 (for using python 3: http://stackoverflow.com/questions/23842713/using-python-3-in-virtualenv)
 virtualenv -p python2.7 env
Activating the environment
From the env directory, run:
source bin/activate
Validating the environment
to ensure that we are using the correct environment, run
 which python
and confirm that the environment is in the correct directory
Installing external libraries to the python environment.
We will now install 3rd party libraries to our python environment. This makes it self contained so that we do not affect MacOSx's system-wide python installation.
We follow the general format:
bin/pip install *libraryname*
install matplotlib for plots and animations
bin/pip install matplotlib
install scipy for matrices
bin/pip install scipy
install mpmath for floats (and since sympy depends on it)
bin/pip install mpmath
install sympy
Download the tar file from https://github.com/sympy/sympy. Extract the tar file in the env directory.
ensure that we are in the correct environment. Navigate to the sympy-x.x.x.x. folder and run
python setup.py install
installing scikit-learn (for machine learning)
http://scikit-learn.org/stable/install.html
bin/pip install -U scikit-learn
 
Note that -U is placed for performing recursive update since scikit-learn has external dependencies
Scikit-learn requires:
 
Python (>= 2.6 or >= 3.3),
NumPy (>= 1.6.1),
SciPy (>= 0.9).
MacOSX Framework issues
There are some issues with using virtualenv on Mac as summarized here: http://wiki.wxpython.org/wxPythonVirtualenvOnMac. Still, we do this tedious method so that we ensure that we are using consistent version numbers across computers.
To enable the use of matplotlib in our python environment, create a new script file called "fwpy.sh" in the bin folder of the python virtual environment.
#!/bin/bash

# what real Python executable to use
PYVER=2.7
PYTHON=/Library/Frameworks/Python.framework/Versions/$PYVER/bin/python$PYVER

# find the root of the virtualenv, it should be the parent of the dir this script is in
ENV=`$PYTHON -c "import os; print os.path.abspath(os.path.join(os.path.dirname(\"$0\"), '..'))"`

# now run Python with the virtualenv set as Python's HOME
export PYTHONHOME=$ENV 
exec $PYTHON "$@"
Executing Python Scripts
From now on, execute python scripts by running the following command on the env/ directory.
 sh bin/fwpy.sh *pythonScript.py*
Try the following example matplotlib file to ensure that the frameworks problem has been fixed double_pendulum_animated.py
