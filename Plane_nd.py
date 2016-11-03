"""
The Plane_nd class defines functions for 
fitting an n-dimensional plane to data values 
in an n-dimensional independent variable parameter space.

```
# Example
xvec = np.linspace(0, 10)
yvec = np.linspace(0, 10)
zvec = np.linspace(0, 10)
xx, yy, zz = np.meshgrid(xvec, yvec, zvec)
dvars = 5 + 2*xx + 3*yy + 4*zz
xflat = xx.flatten()
yflat = yy.flatten()
zflat = zz.flatten()
ivflat = np.transpose(np.array([xflat, yflat, zflat]))
dvflat = dvars.flatten()

fitter = Plane_nd(ivflat, dvflat, 3)

fitter.dolsq()
```
"""    

import numpy as np
from scipy.optimize import leastsq

class Plane_nd(object):
    def __init__(self, ivals, dvals, dm):
        self.ivals = ivals
        self.dvals = dvals
        self.dm = dm
        self.npars = self.dm+1
        self.npts = len(self.dvals)
    
    def fplane(self, x, *cpars):
        cpars = np.array(cpars)
        xp = np.empty(self.npars)
        xp[1:self.npars] = x[0:self.dm]
        xp[0] = 1.0
        return np.sum(xp * cpars)

    def objfun(self, pars):
        fvals = np.array([dv - self.fplane(iv,pars) for dv, iv in zip(self.dvals, self.ivals)])
        return fvals

    def dolsq(self, guess=None, do_print=False):
        if guess:
            # If guess is provided, it should provide a vector for the initial values xini
            xini = np.array(guess)
        else:
            xini = np.zeros(self.npars)
            xini[0] = np.average(self.dvals)
            # Find independent values at average of dependent var
            iave = np.abs(self.dvals-xini[0]).argmin()
            ivt = np.transpose(self.ivals)
            for i in np.arange(1, self.npars):
                ivti = ivt[i-1]
                # Find maximum value along independent axis
                iv_max_i = ivti.argmax()
                iv_max = ivti[iv_max_i]
                # Find corresponding dependent variable value
                dv_max = self.dvals[iv_max_i]
                # Find minimum value along independent axis
                iv_min_i = ivti.argmin()
                iv_min = ivti[iv_min_i]
                # Find corresponding dependent variable value
                dv_min = self.dvals[iv_min_i]
                # Estimate slope
                xini[i] = (dv_max - dv_min)/(iv_max - iv_min)/self.dm
        popt, pcov, idict, mesg, ierr = leastsq(self.objfun, xini, full_output=True, xtol=1.e-20, ftol=1.e-16)
        if do_print:
            print(popt)
            print(pcov)
            for k in idict.keys():
                print('{}: {}'.format(k, idict[k]))
            print(mesg)
            print(ierr)
        return popt, pcov



