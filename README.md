# PyWFDeconv
Original Git: https://github.com/meravstr/Wide-Field-Deconvolution  
Paper: https://www.biorxiv.org/content/10.1101/2020.02.01.930040v1  
Translating (and enhancing) code from Matlab into Python..   


## 1. Installation  
Put the PyWFDeconv folder into your Python Project.  
Use "import PyWFDeconv as wfd" for default usage. This contains function wrappers that execute most of the code.  
If you want to get into the individual functions, you'll have to import sub-packages (e.g. as in "import PyWFDeconv.convar").  
### Notes  
If you want to use experimental features, you'll most likely have to downgrade to Python 3.7.  
Numba doesn't support newer Python Version atm (Mar 2021).  
However, basic PyWFDeconv functionalities work with Python 3.9.

## 2. Usage
Use wrappers.