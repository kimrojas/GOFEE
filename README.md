# A subtle modification to GOFEE

Find the original code here:
https://gitlab.au.dk/au480665/gofee/-/blob/GOFEE2/gofee/gofee.py
and the documentation here:
http://grendel-www.cscaa.dk/mkb/.

How to use this modified code:

a. Put the gofee_modified.py file inside the gofee-folder. For instance: /home/msamuel/gofee2/gofee.

b. In the input file use
```
from gofee.gofee_modified import GOFEE
```
INSTEAD OF
```
from gofee import GOFEE
```

##################################

##################################
**Modification details:**

1. Decaying kappa

I add an opption to turn kappa into a gaussian function which dacaying to 1. The kappa value for every iteration is printed in the log file. 

```
kappa: float or string
        Default: 2
        "How much to weigh predicted uncertainty in the acquisition
        function. 
        Set to "decay" for using the decaying kappa."
```

2. Similarity check

During a parralel gofee run, a similarity check is performed to ensure the verry similar structure won't be evaluated using DFT repeatedly. To activate this option, plase name the calculation folder as "run0, run1, run2, etc." and name the trajectory file as "structures.traj".

```
similarity_thr: float
        Default: 0.999
        "Threshold to skip DFT evaluation of very similar structure. The structure will be just copied from the already evaluated structure instead. The similarity check is performed by calculating the kerner value of the two structure's feature vectors, K(X1, X2). "
```

Please refer to the example for more details. 