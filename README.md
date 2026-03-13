# Finite Element Modelling framework for structural modelling of Leading-Edge Inflatable Kites
- Makes use of non-compressible spring elements to represent the bridle lines, pulleys and canopy
- Timoshenko beam elements are tuned to match the properties of inflatable beams.

### Finite Element Model of the TU Delft V3 Kite
![Finite Element Model of the TU Delft V3 Kite](https://github.com/awegroup/kite_fem/blob/main/docs/images/kitemodel.svg)


### V3 kite with shortened bridle lines under gravity load
![V3 kite with shortened bridle lines under gravity load](https://github.com/awegroup/kite_fem/blob/main/docs/images/hangingkite.svg)


## Installation Instructions

### Building on Linux: Fedora/RHEL

`pyfe3d` uses Cython to build C++ extensions and links against the static C++ runtime.  
On Fedora/RHEL systems, you need to install the following packages before building:

```bash
sudo dnf install gcc gcc-c++ python3-devel libgomp libstdc++-static
```

Without these, the build may fail with errors like:

```bash
/usr/bin/ld: cannot find -lstdc++
collect2: error: ld returned 1 exit status
```

### Building on Ubuntu/Debian

**this has not been tested yet...**

On Debian/Ubuntu systems, the static libstdc++ is not shipped as a package.
In that case, it is usually better to remove the -static flags in the pyfe3d build configuration
or use a prebuilt wheel if available. You will still need the standard development toolchain:

```bash
sudo apt update
sudo apt install build-essential python3-dev g++ libgomp1
```

### Building on Windows

....?


1. Clone the repository:
    ```bash
    git clone https://github.com/awegroup/kite_fem
    ```

2. Navigate to the repository folder:
    ```bash
    cd kite_fem
    ```
    
3. Create a virtual environment:
   
   Linux or Mac:
    ```bash
    python3 -m venv venv
    ```
    
    Windows:
    ```bash
    python -m venv venv
    ```
    
4. Activate the virtual environment:

   Linux or Mac:
    ```bash
    source venv/bin/activate
    ```

    Windows
    ```bash
    .\venv\Scripts\activate
    ```

5. Install the required dependencies:

   For users:
    ```bash
    pip install .
    ```
        
   For developers:
    ```bash
    pip install -e .[dev]
    ```
    
    For ubuntu add:
    ```
    pip instal pyqt5
    sudo apt install cm-super
    sudo apt install dvipng
   ```

6. To deactivate the virtual environment:
    ```bash
    deactivate
    ```
### Failed to build installable wheels for some pyproject.toml based on projects: Pyfe3D
If the above error occurs, it means that your system does not have a c compiler installed. Follow the instructions for your OS (taken from https://cython.readthedocs.io/en/stable/src/quickstart/install.html):

Linux: The GNU C Compiler (gcc) is usually present, or easily available through the package system. On Ubuntu or Debian, for instance, it is part of the build-essential package. Next to a C compiler, Cython requires the Python header files. On Ubuntu or Debian, the command sudo apt-get install build-essential python3-dev will fetch everything you need.

Mac OS X: To retrieve gcc, one option is to install Apple’s XCode, which can be retrieved from the Mac OS X’s install DVDs or from https://developer.apple.com/.

Windows: The CPython project recommends building extension modules (including Cython modules) with the same compiler that Python was built with. This is usually a specific version of Microsoft Visual C/C++ (MSVC) - see https://wiki.python.org/moin/WindowsCompilers. Visual studio build tools can be found on https://gist.github.com/Mr-Precise/9967e3fcf03f2df0282b76841d2f3876 MSVC is the only compiler that Cython is currently tested with on Windows. If you’re having difficulty making setuptools detect MSVC then PyMSVC aims to solve this.

A possible alternative is the open source MinGW (a Windows distribution of gcc). See the appendix for instructions for setting up MinGW manually. Enthought Canopy and Python(x,y) bundle MinGW, but some of the configuration steps in the appendix might still be necessary.



### Dependencies
- matplotlib>=3.7.1
- numpy
- scipy
- pyfe3d

## Usages
?

## Contributing Guide
Please report issues and create pull requests using the URL:
```
https://github.com/awegroup/kite_fem/
```

We welcome contributions to this project! Whether you're reporting a bug, suggesting a feature, or writing code, here’s how you can contribute:

1. **Create an issue** on GitHub
2. **Create a branch** from this issue
   ```bash
   git checkout -b issue_number-new-feature
   ```
3. --- Implement your new feature---
4. Verify nothing broke using **pytest**
```
  pytest
```
5. **Commit your changes** with a descriptive message
```
  git commit -m "#<number> <message>"
```
6. **Push your changes** to the github repo:
   git push origin branch-name
   
7. **Create a pull-request**, with `base:develop`, to merge this feature branch
8. Once the pull request has been accepted, **close the issue**

## Citation
If you use this project in your research, please consider citing it. 
Citation details can be found in the [CITATION.cff](CITATION.cff) file included in this repository.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## WAIVER
Technische Universiteit Delft hereby disclaims all copyright interest in the package written by the Author(s).
Prof.dr. H.G.C. (Henri) Werij, Dean of Aerospace Engineering

### Copyright
Copyright (c) 2025 Patrick Roeleveld
