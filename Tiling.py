"""
Tile an N-Dimensional Domain containing Point objects depending on a Tile decision function.

Copyright (c) 2016, Donald E. Willcox
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from Plane_nd import Plane_nd

class OutputWriter(object):
    # Provides a convenience function which can either write to stdout
    # or open and write a file.
    def __init__(self, otype=None, null=False):
        """
        If otype==None, then OutputWriter will print to stdout.
        Otherwise, open a file named otype for writing.
        If null, write immediately returns and does nothing.
        """
        self.ofile = None
        self.null = null
        if otype:
            self.ofile = open(otype, 'w')
            
    def close(self):
        """
        If OutputWriter has a file open, then close it.
        """
        if self.ofile:
            self.ofile.close()
            
    def write(self, content):
        """
        Print content to stdout if no file object is available in ofile.
        If there is a file object in ofile, then write content to the file.
        
        When writing content to a file, append a newline.
        """
        if self.null:
            return
        if self.ofile:
            self.ofile.write(content + '\n')
        else:
            print(content)

class BCTypes(object):
    # Boundary Types
    up    = +1
    none  = None
    down  = -1
    tile  = +2
    point = +3
    all_types = +4

class BCDim(object):
    # Boundary data structure along a dimension
    def __init__(self):
        self.lo_bc = BCTypes.none
        self.lo_bc_type = BCTypes.none
        self.lo_bc_object = BCTypes.none
        self.hi_bc = BCTypes.none
        self.hi_bc_type = BCTypes.none
        self.hi_bc_object = BCTypes.none

class TETypes(object):
    """Tiling Error Types"""
    # Tiling cannot find a point to start tiling at
    cannot_start_point = 0
    # Tiling cannot enclose enough points to constrain the fit
    cannot_enclose_enough_points = 1
    # Too few points remain in domain to constrain a fit on a new tile
    few_points_remain = 2

class DMCycle(object):
    # Dimension Cycler
    def __init__(self, dm=None):
        if not dm:
            print('ERROR: Must provide number of dimensions to DMCycle!')
            exit()
        self.dm = dm
        self.dims = [i for i in range(self.dm)]

    def cycle(self):
        t = self.dims.pop(0)
        self.dims.append(t)
        return t

class TilingError(Exception):
    """Error class for various kinds of tiling errors that can arise."""
    def __init__(self, err_type, err_tile=None, scratch_points=None, message=''):
        # err_tile is the Tile object we were attempting to extend
        self.err_type = err_type
        self.err_tile = err_tile
        self.scratch_points = scratch_points
        self.message = message

class Point(object):
    def __init__(self, r=[], v=None):
        # Position in n-D space
        self.r = np.array(r, copy=True)
        # Position in n-D space normalized by domain extents
        self.rnorm = np.copy(r)
        # Value of point (Scalar)
        self.v = v
        self.dm = len(r)
        # n-D Mask: Boundary Edge represented by Point
        self.bedge = [None for i in range(self.dm)]
        # n-D Mask: Boundary Type represented by Point
        self.btype = [BCTypes.none for i in range(self.dm)]

    def norm_dist_to_pt(self, b):
        # Get distance between this Point and Point b
        dr = np.array(self.rnorm) - np.array(b.rnorm)
        return np.sqrt(np.sum(dr**2))

    def order_nn(self, plist=[]):
        # Orders the points in plist by nearest neighbors to self.
        if not list(plist):
            return None
        retlist = sorted(plist, key=(lambda p: self.norm_dist_to_pt(p)))
        return retlist

    def get_average_dist_nn(self, plist=[], num_neighbors=1):
        # Given the point list plist, find the average distance
        # of the nearest num_neighbors to self.
        if not list(plist) or num_neighbors > len(plist):
            return None
        ordered_plist = self.order_nn(plist)
        which_nn = ordered_plist[:num_neighbors]
        dist_nn = np.array([self.norm_dist_to_pt(nn) for nn in which_nn])
        ave_dist_nn = np.mean(dist_nn)
        return ave_dist_nn

class Plane(object):
    def __init__(self, points=None, fit_guess=[], dm=None, lo=[], hi=[], writer=None):
        # fit_guess should provide, well, a guess for the fit parameters
        # if fit_guess isn't supplied, it will be estimated from the points.
        self.cpars = None # Length n+1 for n-D space
        self.cerrs = None # Length n+1 for n-D space, std err on cpars
        self.dm = dm
        self.resd  = None
        self.ss_res = None
        self.ave_dvar = None
        self.ss_tot = None
        self.coeff_det = None
        self.norm_resd = None
        self.L2_norm_resd = None
        self.tilde_resd = None
        self.lo = np.copy(lo)
        self.hi = np.copy(hi)
        self.center = None
        if list(self.lo) and list(self.hi):
            # Determine center and width from [lo, hi]
            self.center = 0.5*(self.lo + self.hi)
            self.width = self.hi - self.lo
        self.compute_pars(points, fit_guess)
        if writer:
            self.writer = writer
        else:
            self.writer = OutputWriter()

    def close(self):
        """
        Manually cleanup. Used for closing open file handles.
        """
        self.writer.close()

    def print_fit_report(self, writer=None):
        """Prints report of the fit to the writer. If writer==None, use self.writer."""
        if not writer:
            writer = self.writer
        writer.write('--- FIT COEFFICIENTS REPORT ---')
        writer.write('--- CONSTANT: {} +/- {}'.format(self.cpars[0], self.cerrs[0]))
        for di in range(self.dm):
            writer.write('--- DIMENSION {}: {} +/- {}'.format(di, self.cpars[di+1], self.cerrs[di+1]))
            
    def compute_pars(self, points, fit_guess=[]):
        if not points:
            return
        ivars = np.array([p.r for p in points])
        dvars = np.array([p.v for p in points])
        fitter = Plane_nd(ivars, dvars, self.dm)
        popt, perr = fitter.dolsq(fit_guess)
        self.cpars = popt
        self.cerrs = perr
        dpfit = np.array([fitter.fplane(ivr, popt) for ivr in ivars])
        if not list(self.center):
            self.center = np.mean(ivars, axis=0)
        if not list(self.width):
            self.width = np.amax(ivars, axis=0) - np.amin(ivars, axis=0)

        ## GOODNESS OF FIT STATISTICS
        self.resd = dvars - dpfit
        
        self.abs_delta_pos = np.absolute(np.array([ivr-self.center for ivr in ivars]))
        self.tilde_resd = np.absolute(np.sum(self.resd * np.transpose(self.abs_delta_pos), axis=1))
        # normalize tilde_resd by the central plane value and the tile dimensions
        self.tilde_resd = self.tilde_resd/(self.center * self.width)

        # Sum of squares of residuals
        self.ss_res = np.sum(self.resd**2)
        # Mean of scalar values
        self.ave_dvar = np.mean(dvars)
        # Total sum of squares
        self.ss_tot = np.sum((dvars - self.ave_dvar)**2)
        # Coefficient of Determination R^2 = 1 - SS(residuals)/SS(total)
        self.coeff_det = 1.0 - self.ss_res/self.ss_tot
        
        self.norm_resd = self.resd/dvars # Residuals normalized by dependent variable samples
        self.L2_norm_resd = np.sqrt(np.sum(self.norm_resd**2)) # L-2 norm of normalized residuals
        
class Tile(object):
    def __init__(self, points=[], lo=[], hi=[], fit_guess=[], dm=None, smask=None,
                 virtual=False, writer=None):
        self.points = points
        self.fit_guess = fit_guess
        self.plane_fit = None
        self.previous_tilde_resd = None # Value of tilde_resd on the previous plane fit
        self.fresh_plane_fit = False # Is the plane fit current?
        self.virtual = virtual # True if this tile represents empty space
        self.lo = np.copy(lo)
        self.hi = np.copy(hi)
        self.smask = None
        self.dm = dm
        if writer:
            self.writer = writer
        else:
            self.writer = OutputWriter()

        # Try to initialize values that weren't passed to the Tile
        if not dm and list(self.lo):
            self.dm = len(self.lo)
        if points and not dm:
            self.dm = points[0].dm
        if points and (not list(self.lo) or not list(self.hi)):
            self.writer.write('init tile lo = {}'.format(self.lo))
            self.writer.write('init tile hi = {}'.format(self.hi))
            self.writer.write('Making tile, calling boundary_minimize')
            self.boundary_minimize()
            self.writer.write('init tile lo = {}'.format(self.lo))
            self.writer.write('init tile hi = {}'.format(self.hi))

        # Setup surface mask smask
        # smask[di] is none if the tile is thick in that dimension
        # smask[di] is down or up if the tile is a lower or upper surface of any tile
        if type(smask) == list:
            self.smask = smask[:]
        else:
            self.smask = [BCTypes.none for j in range(self.dm)]

    def close(self):
        """
        Manually cleanup. Used for closing open file handles.
        """
        self.writer.close()
        # Close Fit object
        if self.plane_fit:
            self.plane_fit.close()
            
    def extend_dimension(self, di, dx, surface, direction):
        sfvec = None
        if surface == BCTypes.up:
            sfvec = self.hi
        elif surface == BCTypes.down:
            sfvec = self.lo
        else:
            self.writer.write('ERROR: SURFACE NOT DEFINED!')
            exit()
        dirsign = None
        if direction == BCTypes.up:
            dirsign = +1
        elif direction == BCTypes.down:
            dirsign = -1
        else:
            self.writer.write('ERROR: DIRECTION NOT DEFINED!')
            exit()
        sfvec[di] += dirsign * dx

    def get_dim_thickness(self, di):
        """
        Given the dimension di, return the thickness
        of this tile along di.
        """
        return self.hi[di] - self.lo[di]

    def get_thinnest_dimension(self):
        """
        Find the dimension di in which this Tile is thinnest.

        Also find the thickness dx along dimension di.

        Return (di, dx)
        """
        di_min = None
        dx_min = None
        for di in range(self.dm):
            dx = self.hi[di] - self.lo[di]
            if not di_min or dx < dx_min:
                di_min = di
                dx_min = dx
        return di_min, dx_min

    def order_thinnest_dimensions(self):
        """
        Return the dimensions of this Tile in a list
        ordered by the Tile thickness in each dimension
        from smallest to largest.
        """
        dims = [di for di in range(self.dm)]
        dims.sort(key=(lambda di: self.get_dim_thickness(di)))
        return dims

    def colocated_with(self, btile, di=-1):
        """
        Determine whether self and btile are colocated.

        If the optional argument di is provided,
        determines whether self and btile are colocated
        only considering dimension di.
        """
        if not (list(self.lo) and list(self.hi) and
                list(btile.lo) and list(btile.hi)):
            return False
        else:
            if di == -1:
                colocated_lo = self.lo == btile.lo
                colocated_hi = self.hi == btile.hi
                colocated = np.all(np.logical_and(colocated_lo, colocated_hi))
                return colocated
            else:
                colocated_lo = self.lo[di] == btile.lo[di]
                colocated_hi = self.hi[di] == btile.hi[di]
                colocated = colocated_lo and colocated_hi
                return colocated

    def gen_vertices(self):
        """
        Return a generator for the vertices of this Tile.
        """
        if not list(self.lo) or not list(self.hi):
            return [None]
        # Re-arrange [lo, hi] by dimension
        dimbcs = [[self.lo[i], self.hi[i]] for i in range(self.dm)]
        return itertools.product(*dimbcs)

    def create_surface(self, di, surface):
        """
        Make and return the surface of this Tile
        defined by the constant dimension (di) where
        the surface normal of Tile along di on this 
        surface lies in the direction given by (surface)
        """
        lo = np.copy(self.lo)
        hi = np.copy(self.hi)
        sm = self.smask[:]
        bvec = None
        if surface == BCTypes.up:
            bvec = self.hi
            sm[di] = BCTypes.up
        elif surface == BCTypes.down:
            bvec = self.lo
            sm[di] = BCTypes.down
        lo[di] = bvec[di]
        hi[di] = bvec[di]
        sface = Tile(lo=lo, hi=hi, smask=sm, virtual=True, writer=self.writer)
        self.writer.write('MADE A SURFACE, HERE IS THE REPORT::::')
        sface.print_tile_report()
        return sface
    
    def get_surfaces(self, dj=-1):
        """
        Return the surfaces for this Tile as Tile objects.

        The distinguishing feature of the surface relative to this Tile
        is that, although the surface and Tile are of the same
        dimensionality, there is at least one dimension di
        in which the surface tile has lo[di] == hi[di] == constant.

        In case this tile is already a surface, then only set the 
        constraint lo[di] == hi[di] == constant if lo[di] != hi[di].

        If dj is provided, only return surfaces for which
        lo[dj] == hi[dj] == constant. 

        Otherwise return all such surfaces if dj is not
        provided (-1). I'm using -1 here because 0 is
        a valid dimension but tests as a boolean False.

        If dj is provided and it is not a nonconstant dimension,
        then return an empty list.
        """
        self.writer.write('GENERATING SURFACES FOR TILE ALONG DIM {}'.format(dj))
        stiles = []
        for di in self.get_nonconstant_dimensions():
            if dj == -1 or di == dj:
                for surface in [BCTypes.down, BCTypes.up]:
                    sface = self.create_surface(di, surface)
                    stiles.append(sface)
        return stiles

    def get_constant_dimensions(self):
        """
        Find all dimensions di for which lo[di] == hi[di] == constant

        Return a list of tuples [(di, constant), ...] satisfying that condition.
        """
        constant_dimensions = []
        for di in range(self.dm):
            if self.lo[di] == self.hi[di]:
                constant_dimensions.append((di, self.lo[di]))
        return constant_dimensions

    def get_nonconstant_dimensions(self):
        """
        Find all dimensions di for which lo[di] != hi[di]

        Return a list of such dimensions di.
        """
        non_constant_dimensions = []
        for di in range(self.dm):
            if self.lo[di] != self.hi[di]:
                non_constant_dimensions.append(di)
        return non_constant_dimensions

    def print_tile_report(self, tile_number=None, writer=None):
        """Prints report of this Tile to the writer. If writer==None, use self.writer."""
        if not writer:
            writer = self.writer
        writer.write('--- TILE {} REPORT ---'.format(tile_number))
        writer.write('--- VIRTUAL TILE = {} ---'.format(self.virtual))
        writer.write('--- LO = {} ---'.format(self.lo))
        writer.write('--- HI = {} ---'.format(self.hi))
        writer.write('--- SURFACE MASK = {} ---'.format(self.smask))
        if not self.virtual:
            writer.write('--- NPTS = {} ---'.format(len(self.points)))
            writer.write('--- L2 NORMALIZED RESD. = {} ---'.format(self.get_L2_norm_resd()))
            writer.write('--- COEFFICIENT OF DETERMINATION = {} ---'.format(self.get_coeff_det()))
            writer.write('--- TILDE RESD. = {} ---'.format(self.get_tilde_resd()))
            self.print_fit_report(writer)
            writer.write('------- POINTS ------')
            for p in self.points:
                writer.write('{}: {}'.format(p.r, p.v))

    def print_fit_report(self, writer=None):
        """Prints report of the fit on this Tile to the writer. If writer==None, use self.writer."""
        if not writer:
            writer = self.writer
        if not self.fresh_plane_fit:
            self.do_plane_fit()
        self.plane_fit.print_fit_report(writer)

    def get_volume(self, dom_lo=[], dom_hi=[]):
        """
        Computes volume of the Tile. If [lo, hi] is undefined, return None.
        If dom_lo and dom_hi are supplied, normalize the tile dimensions
        by the dimensions of domain lo and hi first before computing
        a normalized volume.
        """
        if not list(self.lo) or not list(self.hi):
            return None
        if list(dom_lo) and list(dom_hi):
            dom_lo = np.copy(dom_lo)
            dom_hi = np.copy(dom_hi)
            dom_width = dom_hi - dom_lo
        else:
            dom_width = np.ones(self.dm)
        dr = (np.array(self.hi) - np.array(self.lo))/dom_width
        return np.prod(dr)

    def boundary_minimize(self):
        """
        Given the points in the Tile, set the boundary
        defined by [lo, hi] to the minimum surface enclosing the points.
        """
        if self.points:
            self.lo = self.points[0].r
            self.hi = self.points[0].r
            for p in self.points:
                self.lo = np.minimum(self.lo, p.r)
                self.hi = np.maximum(self.hi, p.r)

    def extend_points(self, plist=[]):
        """
        Given the list of points (plist), extends the Tile if
        necessary to enclose them.

        Set the Tile boundaries to the minimum volume enclosing
        the provided points.

        Do nothing if no points are provided.
        """
        if not plist:
            return
        self.points += plist
        self.boundary_minimize()
        self.fresh_plane_fit = False

    def overlaps_point_dimension(self, refpoint, di):
        """
        Checks to see if self overlaps refpoint in the dimension di:

        refpoint must be a Point object

        di must be an integer in range(self.dm)
        """
        refp_olap = True
        if (refpoint.r[di] >= self.hi[di] or
            refpoint.r[di] <= self.lo[di]):
            # tiles do not overlap
            refp_olap = False
        return refp_olap

    def get_point_occlusions(self, points, di):
        """
        Given a list of points (points), find the points 
        which will occlude self along dimension di and thus can set bounds.

        Return a list of such points.
        """
        # Find the points which self does not overlap in dimension di
        # but does overlap in every other dimension.
        # These points set the bounds on extensions along dimension di.
        # (If there is an additional dimension along which tile and points
        # do not overlap, then no constraint can be made along di.)
        opoints = []
        for p in points:
            if self.overlaps_point_dimension(p, di):
                # p overlaps along di, so can't constrain di
                continue
            kandidate = True
            for dj in range(self.dm):
                if dj==di:
                    continue
                if not self.overlaps_point_dimension(p, dj):
                    # p doesn't overlap along dj, dj =/= di
                    # so can't constrain di
                    kandidate = False
                    break
            if kandidate:
                opoints.append(p)
        return opoints

    def get_point_constraints(self, points, di, bcdi=None):
        """
        Get point based [lo, hi] constraints along dimension di for this Tile.

        Return boundary conditions along dimension di in bcdi.

        Calculates point occlusions from the list of points (points)
        """
        if not bcdi:
            bcdi = BCDim()
        opoints = self.get_point_occlusions(points, di)
        # Get point constraint on di for [lo, hi]
        for p in opoints:
            # Check if p can constrain lo along di
            if p.r[di] <= self.lo[di]:
                phalf = 0.5*(p.r[di] + self.lo[di])
                if bcdi.lo_bc == BCTypes.none or phalf > bcdi.lo_bc:
                    bcdi.lo_bc = phalf
                    bcdi.lo_bc_type = BCTypes.point
                    bcdi.lo_bc_object = p

            # Check if p can constrain hi along di
            if p.r[di] >= self.hi[di]:
                phalf = 0.5*(p.r[di] + self.hi[di])
                if bcdi.hi_bc == BCTypes.none or phalf < bcdi.hi_bc:
                    bcdi.hi_bc = phalf
                    bcdi.hi_bc_type = BCTypes.point
                    bcdi.hi_bc_object = p
        return bcdi
    
    def overlaps_tile_dimension(self, reftile, di):
        """
        Checks to see if self overlaps reftile in the dimension di:

        reftile must be a Tile object

        di must be an integer in range(self.dm).
        """
        reft_olap = True
        if (reftile.lo[di] >= self.hi[di] or
            reftile.hi[di] <= self.lo[di]):
            # tiles do not overlap
            reft_olap = False
        return reft_olap

    def overlaps_tiles(self, tlist=[]):
        """
        Checks to see if self overlaps any of the tiles in tlist

        Returns list of tiles in tlist which overlap self

        Returns the empty list if no tiles in tlist overlap self
        """
        if not tlist:
            return False
        olap = []
        for reft in tlist:
            # Check overlap between self and reft
            reft_olap = True
            for di in range(self.dm):
                reft_olap = self.overlaps_tile_dimension(reft, di)
                if not reft_olap:
                    # tiles do not overlap
                    break
            if reft_olap:
                olap.append(reft)
        return olap

    def get_tile_occlusions(self, tiles, di):
        """
        Given a list of tile objects (tiles), find the tiles in tiles
        which are not self which will occlude self along dimension di 
        and thus can set bounds.

        Return a list of such tiles.
        """
        otiles = []
        for ktile in tiles:
            if self.whether_occludes_tile(ktile, di):
                otiles.append(ktile)
        return otiles

    def whether_occludes_tile(self, atile, di):
        """
        Determine whether self and atile occlude along dimension di.

        Return True if they occlude, False otherwise.

        By design, occlusion is false if the tiles
        overlap along dimension di or if they are the same tile.
        """
        if self.colocated_with(atile) and self.overlaps_tile_dimension(atile, di):
            return False
        else:
            # self and atile can occlude along dimension di
            # if they do not overlap in dimension di
            # but do overlap in every other dimension.
            # Only in such a case can atile provide a constraint
            # on the extent of self along dimension di.
            candidate = True
            for dj in range(self.dm):
                if (dj==di or (self.smask[dj] != BCTypes.none and
                               atile.smask[dj] != BCTypes.none)):
                    continue
                if not self.overlaps_tile_dimension(atile, dj):
                    # atile doesn't overlap self along dj, dj =/= di
                    # so atile and self can't occlude along di
                    candidate = False
                    break
            return candidate

    def whether_osculates_tile(self, atile, di, direction=BCTypes.none):
        """
        Determine whether self and atile osculate on any surface in dimension di.

        If direction is supplied, then the osculation surface relative to self
        should have the surface normal mask equal to direction.

        Find the surface of self which is osculated (sface)

        Also find the ctile which is the intersection of the osculating surfaces
        of self and atile.

        Return (sface, ctile) in case atile osculates self.

        Otherwise return (None, None)
        """
        negative = (None, None)
        
        # First determine whether atile occludes self along di
        if not self.whether_occludes_tile(atile, di):
            return negative
        
        # Now determine if atile and self osculate
        self_surfaces_di = self.get_surfaces(di)
        atile_surfaces_di = atile.get_surfaces(di)
        for isface, sface in enumerate(self_surfaces_di):
            for iaface, aface in enumerate(atile_surfaces_di):
                if (sface.hi[di] == aface.hi[di] and
                    sface.lo[di] == aface.lo[di] and
                    (direction == BCTypes.none or
                     sface.smask[di] == direction)):
                    # sface and aface osculate
                    if direction == BCTypes.none:
                        self.writer.write('identified osculation between sface {} and aface {} for di = {}'.format(isface, iaface, di))
                    else:
                        self.writer.write('identified osculation between sface {} and aface {} along direction {} for di = {}'.format(isface, iaface, direction, di))
                    # now find the intersection tile between sface and aface
                    ctile = sface.get_tile_intersection(aface)
                    # set smask of ctile to that of sface
                    ctile.smask = sface.smask[:]
                    self.writer.write('OSCULATION INTERSECTION REPORT::::')
                    ctile.print_tile_report()
                    return (sface, ctile)
                
        # Found nothing so atile and self don't osculate
        return negative
                    
    def get_tile_intersection(self, atile):
        """
        Return the tile which is the intersection of self and atile.
        """
        lo = np.maximum(self.lo, atile.lo)
        hi = np.minimum(self.hi, atile.hi)
        return Tile(lo=lo, hi=hi, virtual=True, writer=self.writer)

    def get_tile_constraints(self, tiles, di, bcdi=None):
        """
        Get tile based [lo, hi] constraints along dimension di for this Tile.

        Return boundary conditions along dimension di in bcdi.

        Calculates tile occlusions from the list tiles.
        """
        if not bcdi:
            bcdi = BCDim()
        otiles = self.get_tile_occlusions(tiles, di)
        # Get tile constraint on di for [lo, hi]
        for btile in otiles:
            # Check if btile can constrain lo along di
            if btile.hi[di] <= self.lo[di]:
                if bcdi.lo_bc == BCTypes.none or btile.hi[di] > bcdi.lo_bc:
                    bcdi.lo_bc = btile.hi[di]
                    bcdi.lo_bc_type = BCTypes.tile
                    bcdi.lo_bc_object = btile

            # Check if btile can constrain hi along di
            if btile.lo[di] >= self.hi[di]:
                if bcdi.hi_bc == BCTypes.none or btile.lo[di] < bcdi.hi_bc:
                    bcdi.hi_bc = btile.lo[di]
                    bcdi.hi_bc_type = BCTypes.tile
                    bcdi.hi_bc_object = btile
        return bcdi

    def get_hypothetical_extend(self, points=[], avoid_tiles=None, greedy_absorb_points=[]):
        # Returns a hypothetical tile with points consisting of self.points + points
        # Extension will avoid tiles in avoid_tiles
        # If greedy_absorb_points is a list of Points,
        # Tile will absorb as many points in greedy_absorb_points as
        # it encloses where the Tile extent is determined by the list `points`.
        # Returns the new Tile, in_pts, out_pts
        # in_pts  : points in greedy_absorb_points now within the Tile
        # out_pts : points in greedy_absorb_points which Tile couldn't absorb
        stile = Tile(points=self.points + points, fit_guess=self.fit_guess, writer=self.writer)
        if stile.overlaps_tiles(avoid_tiles):
            return None, [], greedy_absorb_points
        else:
            # Absorb as many points as allowed from greedy_absorb_points
            in_pts, out_pts = stile.which_points_within(greedy_absorb_points)
            stile.extend_points(in_pts)
            return stile, in_pts, out_pts

    def extend_min_volume(self, plist=[], avoid_tiles=None, decision_fun=None, dom_lo=[], dom_hi=[]):
        """
        Given the list of points (plist), extends the Tile
        by adding one point from plist to Tile where the point 
        is selected from plist such that it minimizes the normalized volume
        of Tile. The normalized volume is the product of tile dimensions,
        each normalized by the extent of the domain in each dimension,
        passed via dom_lo and dom_hi arguments. 
        If either of those arguments are not supplied, then do not normalize the volume.

        Returns plist where the selected point is popped from the list.

        If avoid_tiles is passed, it should be a list of Tile objects.
        The current Tile will then only be extended such that it does
        not intersect the tiles in avoid_tiles.

        If a function is passed as decision_fun, this Tile will be passed to
        the 'decision function' to determine whether to extend the tile.
        decision_fun should take a single Tile argument and return True or False
        """
        if list(dom_lo):
            dom_lo = np.copy(dom_lo)
        if list(dom_hi):
            dom_hi = np.copy(dom_hi)
        min_vol_point_i = None
        min_vol_point = None
        min_vol = None
        min_in_points = None
        min_out_points = None
        for i, p in enumerate(plist):
            other_points = plist[:]
            other_points.pop(i)
            pext = [p]
            stile, in_spts, out_spts = self.get_hypothetical_extend(points=pext, avoid_tiles=avoid_tiles,
                                                                    greedy_absorb_points=other_points)
            if not stile:
                continue
            svol = stile.get_volume(dom_lo=dom_lo, dom_hi=dom_hi)
            dbool = True
            if callable(decision_fun):
                dbool = decision_fun(stile)
                # self.writer.write('dbool = {}'.format(dbool))
            if (((not min_vol) or svol < min_vol)
                and
                dbool):
                min_vol = svol
                min_vol_point = p
                min_vol_point_i = i
                min_in_points = in_spts[:]
                min_out_points = out_spts[:]
        if not min_vol_point:
            # If the above could find no point to extend, then do nothing
            # Return plist and False, indicating no extension
            return plist, False
        else:
            # Else, extend this Tile
            self.extend_points([min_vol_point] + min_in_points)
            # Return reduced point list and True, indicating extension
            plist = min_out_points[:]
            return plist, True

    def which_points_within(self, pointlist=[], lo=[], hi=[]):
        # Determines which Points of the list pointlist fall on or within the
        # boundaries of this tile. Does not care whether they are members.
        if not list(lo) or not list(hi):
            lo = self.lo
            hi = self.hi
        inpts = []
        outpts = []
        for pt in pointlist:
            pt_in = True
            for di in range(self.dm):
                if pt.r[di] < lo[di] or pt.r[di] > hi[di]:
                    pt_in = False
                    break
            if pt_in:
                inpts.append(pt)
            else:
                outpts.append(pt)
        return inpts, outpts
        
    def get_subtile(self, lo, hi):
        # Return a Tile object corresponding to a subtile of self.
        # Return None if Error.        
        # First, check domain partitioning
        if len(lo) != self.dm:
            return None
        if len(hi) != self.dm:
            return None
        for di in range(self.dm):
            if lo[di] < self.lo[di] or lo[di] > self.hi[di]:
                return None
            if hi[di] < self.lo[di] or hi[di] > self.hi[di]:
                return None
        # Now get the points within [lo, hi]
        inpts, outpts = self.which_points_within(self.points, lo, hi)
        # Create and return sub-Tile
        stile = Tile(inpts, lo, hi, self.dm, writer=self.writer)
        return stile

    def do_plane_fit(self):
        if not len(self.points) > self.dm:
            # Don't attempt a fit if there are too few points
            return
        if self.plane_fit:
            self.previous_tilde_resd = self.plane_fit.tilde_resd
        p = Plane(points=self.points, fit_guess=self.fit_guess,
                  dm=self.dm, lo=self.lo, hi=self.hi, writer=self.writer)
        self.plane_fit = p
        self.fit_guess = np.copy(p.cpars)
        self.fresh_plane_fit = True

    def get_coeff_det(self):
        # Returns coefficient of determination
        # between the points in the Tile and a Plane fit.
        if not self.fresh_plane_fit:
            self.do_plane_fit()
        return self.plane_fit.coeff_det

    def get_L2_norm_resd(self):
        # Returns L-2 norm of normalized residuals
        # between the points in the Tile and a Plane fit.
        if not self.fresh_plane_fit:
            self.do_plane_fit()
        return self.plane_fit.L2_norm_resd

    def get_tilde_resd(self):
        # Returns residual-weighted sum of central deviances
        # between the points in the Tile and a Plane fit.
        if not self.fresh_plane_fit:
            self.do_plane_fit()
        return self.plane_fit.tilde_resd

class Domain(object):
    def __init__(self, points=[], lo=[], hi=[], dm=None,
                 plot_lo=[], plot_hi=[],
                 point_normalize=True,
                 plot_dimfrac=0.9, last_domain_slice=(None, None),
                 dlabels=[], ilabel=None,
                 logfile=None, summaryfile=None):
        # The Domain is just a set of Point objects
        # and functions for tiling them into a set of Tile objects.        
        self.tiles = []         # Tiles contain points
        self.virtual_tiles = [] # virtual Tiles are empty
        self.lo = np.copy(lo)
        self.hi = np.copy(hi)
        self.dm = dm
        self.points = points
        self.scratch_points = []
        self.plot_num = 0
        self.last_domain_slice = last_domain_slice # (fig, ax) tuple
        self.dlabels = dlabels # Labels for the Domain dimensions
        self.ilabel = ilabel # Label for the scalar value (independent var)
        if logfile:
            self.logwriter = OutputWriter(logfile)
        else:
            self.logwriter = OutputWriter(null=True)
        if summaryfile:
            self.sumwriter = OutputWriter(summaryfile)
        else:
            self.sumwriter = OutputWriter()

        set_photogenic_lims = False
        if not list(plot_lo):
            self.plot_lo = np.copy(self.lo)
            set_photogenic_lims = True
        else:
            self.plot_lo = np.copy(plot_lo)
            
        if not list(plot_hi):
            self.plot_hi = np.copy(self.hi)
            set_photogenic_lims = True
        else:
            self.plot_hi = np.copy(plot_hi)

        if set_photogenic_lims:
            self.photogenic_plot_limits(plot_dimfrac)

        # Check and set dimensionality
        if list(self.lo) and list(self.hi):
            if len(self.lo) != len(self.hi):
                self.logwriter.write('ERROR: lo and hi supplied with incongruous dimensions.')
                exit()
            else:
                if not self.dm:
                    self.dm = len(self.lo)

        # Check and unset dependent variable labels if needed
        if self.dlabels and len(self.dlabels) != self.dm:
            self.dlabels = []

        # Set up boundary masks for points
        if list(points) and list(self.lo) and list(self.hi):
            self.bc_init_mask_points(self.points)

        if point_normalize:
            # Normalize point positions by domain extents
            width = self.hi - self.lo
            for p in self.points:
                for di in range(self.dm):
                    p.rnorm[di] = (p.rnorm[di]-self.lo[di])/width[di] + self.lo[di]

    def close(self):
        """
        Manually cleanup. Used for closing open file handles.
        """
        self.logwriter.close()
        self.sumwriter.close()
        # Close Tiles
        for atile in self.tiles + self.virtual_tiles:
            atile.close()

    def photogenic_plot_limits(self, dimfrac=0.9):
        """
        Compute the appropriate plot limits [self.plot_lo, self.plot_hi]
        for which 2-D domain slices will plot the domain [self.lo, self.hi]
        such that the domain extents in each dimension occupy
        the fraction dimfrac of their corresponding axis.
        """
        center = 0.5 * (self.lo + self.hi)
        width = self.hi - self.lo
        plot_width = width/dimfrac
        self.plot_lo = center - 0.5 * plot_width
        self.plot_hi = center + 0.5 * plot_width
            
    def plot_domain_slice(self, dimx=0, dimy=1, save_num=None, show_tile_id=True,
                          save_last_figure=False, underlay_figure_axis=None, show_plot=False):
        if self.dm < 2:
            return
        # Plot 2-D Slice of Domain
        if underlay_figure_axis:
            fig, ax = underlay_figure_axis
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        # Plot Tile outlines
        linestyle_options = ['-', '--', '-.', ':']
        ls_cycler = DMCycle(len(linestyle_options))
        for i, t in enumerate(self.tiles + self.virtual_tiles):
            # Plot Tile outline
            if t.virtual:
                self.logwriter.write('plotting a virtual tile {}'.format(i))
                # Plotting options for a virtual tile
                edgecolor_value = 'red'
                hatch_value = '/'
                linestyle_value = '-' # linestyle_options[ls_cycler.cycle()]
            else:
                self.logwriter.write('plotting a real tile {}'.format(i))
                # Plotting options for a real tile
                edgecolor_value = 'orange'
                hatch_value = None
                linestyle_value = '-' # linestyle_options[ls_cycler.cycle()]
            ax.add_patch(Rectangle((t.lo[dimx], t.lo[dimy]), t.hi[dimx]-t.lo[dimx], t.hi[dimy]-t.lo[dimy], facecolor='None',
                                   edgecolor=edgecolor_value,
                                   hatch=hatch_value,
                                   linewidth=1.5,
                                   linestyle=linestyle_value))
            tile_center_x = 0.5 * (t.lo[dimx] + t.hi[dimx])
            tile_center_y = 0.5 * (t.lo[dimy] + t.hi[dimy])
            # Plot Text in the center of the Tile
            if show_tile_id:
                ax.text(x=tile_center_x, y=tile_center_y, s='{}'.format(i))
        # Plot points inside Tiles
        points_x = []
        points_y = []
        points_v = []
        for j, t in enumerate(self.tiles):
            for i, p in enumerate(t.points):
                points_x.append(p.r[dimx])
                points_y.append(p.r[dimy])
                points_v.append(p.v)
        if self.points:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cmap = mpl.cm.viridis
            point_scalar_range = np.array([p.v for p in self.points])
            bounds = np.linspace(np.amin(point_scalar_range), np.amax(point_scalar_range), num=5)
            norm = mpl.colors.Normalize(vmin=np.amin(bounds),vmax=np.amax(bounds))
            img = ax.scatter(points_x, points_y, c=points_v, cmap=cmap, norm=norm)
            if self.ilabel:
                cblabel = self.ilabel
            else:
                cblabel = 'Scalar Value'
            fig.colorbar(img, cmap=cmap, cax=cax, ticks=bounds, norm=norm, label=cblabel)
        # Plot points outside Tiles
        if self.scratch_points:
            for i, p in enumerate(self.scratch_points):
                ax.scatter(p.r[dimx], p.r[dimy], color='red')
        if not underlay_figure_axis:
            ax.set_xlim([self.plot_lo[dimx], self.plot_hi[dimx]])
            ax.set_ylim([self.plot_lo[dimy], self.plot_hi[dimy]])
            if self.dlabels:
                ylabel = self.dlabels[dimy]
                xlabel = self.dlabels[dimx]
            else:
                ylabel = 'Dimension {}'.format(dimy)
                xlabel = 'Dimension {}'.format(dimx)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            fig.tight_layout()
        if not save_num:
            self.plot_num += 1
            this_plot_num = self.plot_num
        else:
            this_plot_num = save_num
        self.logwriter.write('SAVING PLOT NUMBER {}'.format(this_plot_num))
        if type(this_plot_num) == int:
            nstr = '{0:04d}'.format(this_plot_num)
        else:
            nstr = str(this_plot_num)
#        fig.savefig('tiled_domain_{}.eps'.format(nstr))
        fig.savefig('tiled_domain_{}.png'.format(nstr))
        if show_plot:
            plt.show()
        if save_last_figure:
            self.last_domain_slice = (fig, ax)
        elif not underlay_figure_axis:
            plt.close(fig)

    def plot_domain_slice_scratch(self, stile, dimx=0, dimy=1, save_num=None, show_tile_id=True,
                                  save_last_figure=False, underlay_figure_axis=None, show_plot=False):
        """
        Given a scratch tile, add it to the tiles in this domain, plot the domain slice,
        and then pop the tile from the domain tiles so the domain is unaffected.
        """
        if stile.virtual:
            self.virtual_tiles.append(stile)
        else:
            self.tiles.append(stile)
        self.plot_domain_slice(dimx=dimx, dimy=dimy, save_num=save_num, show_tile_id=show_tile_id,
                               save_last_figure=save_last_figure,
                               underlay_figure_axis=underlay_figure_axis, show_plot=show_plot)
        if stile.virtual:
            self.virtual_tiles.pop()
        else:
            self.tiles.pop()

    def print_domain_report(self, writer=None):
        # Prints full (scratch) domain data to the given writer
        # If none, use the logwriter
        if not writer:
            writer = self.logwriter
        writer.write('---DOMAIN REPORT---')
        writer.write('DOMAIN LO = {}'.format(self.lo))
        writer.write('DOMAIN HI = {}'.format(self.hi))
        writer.write('-------POINTS------')
        for p in self.scratch_points:
            writer.write('{}: {}'.format(p.r, p.v))
        writer.write('--------TILES------')
        for i, t in enumerate(self.tiles + self.virtual_tiles):
            t.print_tile_report(tile_number=i, writer=writer)
        
    def bc_init_mask_points(self, plist):
        # Set initial boundary masks for list of points plist given domain boundaries
        # Initial because uses domain lo, hi
        dm_hi_pts = [None for i in range(self.dm)]
        dm_hi_val = np.copy(self.lo)
        dm_lo_pts = [None for i in range(self.dm)]
        dm_lo_val = np.copy(self.hi)
        for p in plist:
            for di in range(self.dm):
                # Check high bc
                if p.r[di] > dm_hi_val[di]:
                    # Point is high, use point as bc
                    dm_hi_val[di] = p.r[di]
                    dm_hi_pts[di] = p
                if p.r[di] < dm_lo_val[di]:
                    # Point is low, use point as bc
                    dm_lo_val[di] = p.r[di]
                    dm_lo_pts[di] = p
                
        # Mask points in upper and lower bc lists
        for di, p in enumerate(dm_hi_pts):
            # self.logwriter.write('BC HI PT {}'.format(di))
            # self.logwriter.write('position: {}'.format(p.r))
            p.bedge[di] = self.hi[di]
            p.btype[di] = BCTypes.up
            # self.logwriter.write('bedge: {}'.format(p.bedge[di]))
            # self.logwriter.write('btype: {}'.format(p.btype[di]))
            
        for di, p in enumerate(dm_lo_pts):
            # self.logwriter.write('BC LO PT {}'.format(di))
            # self.logwriter.write('position: {}'.format(p.r))
            p.bedge[di] = self.lo[di]
            p.btype[di] = BCTypes.down
            # self.logwriter.write('bedge: {}'.format(p.bedge[di]))
            # self.logwriter.write('btype: {}'.format(p.btype[di]))

    def propagate_tile_perturbation(self, from_tile, di, dx,
                                    surface, direction,
                                    ignore_tiles=[], dry_run=False):
        """
        Extend the boundaries of from_tile along dimension di by length dx.

        Find the other tiles in the domain which extending this boundary
        would either interfere with or pull away from if it already osculates
        them along dimension di. Correct those tile dimensions and propagate
        the changes throughout the domain recursively. Never shrink a tile
        by more than dx if it's of width dx or smaller. In that case, 
        return False to indicate the propagation could not succeed.

        The surface mask smask of this Tile will be checked and
        if it is not equal to BCTypes.none to indicate this Tile is a surface
        along dimension di, then expand the surface outwards in the 
        direction of smask.

        Surface and Direction should be either BCTypes.up or BCTypes.down
        to indicate whether lo or hi is to be shifted and in what direction.

        If ignore_tiles, then ignore propagating the perturbation to
        the tiles in ignore_tiles. This is an aid for recursively propagating
        throughout the domain.

        If dry_run == True, then do not actually perform any propagation
        but do check the domain tiles recursively to see if the propagation
        is allowed.

        Returns False if the perturbation could not be applied to from_tile,
        returns True otherwise.
        """
        # First just check whether from_tile is thicker than dx if
        # we are asked to shrink from_tile. Return False otherwise.
        if surface == -direction and from_tile.get_dim_thickness(di) <= dx:
            return False
        
        # Otherwise ...
        # Extend the surface of from_tile in the direction (direction)
        # from its up or down (surface) in dimension di
        # by the width dx.
        sface = from_tile.create_surface(di, surface)
        
        # Find the tiles in the domain which sface osculates along di in direction
        osct_tiles = self.get_osculating_tiles(sface, di, direction=direction,
                                               get_other_sface=True, return_other_tile=True)
        
        # Extend sface to get overlaps
        sface.extend_dimension(di, dx, direction, direction)
        # Find the tiles in the domain which sface overlaps.
        olap_tiles = sface.overlaps_tiles(self.tiles + self.virtual_tiles)

        # Find the tiles in the domain which the sface osculated but does not
        # intersect now that it has been extended. These are the newly un-osculated tiles.
        unosct_tiles = []
        for otile in osct_tiles:
            if not otile in olap_tiles:
                unosct_tiles.append(otile)

        # Create next ignore_tiles by including form_tile
        ignore_tiles_next = ignore_tiles[:]
        ignore_tiles_next.append(from_tile)

        # Check or do propagation to newly overlapped or unosculated tiles given dry_run
        for otile in olap_tiles + unosct_tiles:
            colocated_ignore = any([otile.colocated_with(igt) for igt in ignore_tiles_next])
            if not colocated_ignore:
                # Perturb otile in dimension di
                # in the same direction as the extension direction
                # from the oppositely oriented surface
                could_otile_prop = self.propagate_tile_perturbation(otile, di, dx,
                                                                    -surface, direction,
                                                                    ignore_tiles=ignore_tiles_next,
                                                                    dry_run=dry_run)
                if not could_otile_prop:
                    self.logwriter.write('COULD NOT SHRINK TILE ALONG DIMENSION {}'.format(di))
                    self.logwriter.write('--- OBSTRUCTING TILE ---')
                    otile.print_tile_report()
                    return could_otile_prop
        # We didn't return with False in the otile loop so the propagation must have succeeded.

        # Did we shrink a real from_tile?
        if (not from_tile.virtual) and surface == -direction:
            # Experimentally shrink a virtual tile of the same size as from_tile.
            # Check to make sure it still contains enough points.
            scratch_from_tile = Tile(lo=from_tile.lo, hi=from_tile.hi, virtual=True, writer=self.logwriter)
            scratch_from_tile.extend_dimension(di, dx, surface, direction)
            # Check to make sure that if from_tile were to shrink, then it will still contain
            # the minimum number of points required to constrain a N-D plane fit.
            inpts, outpts = scratch_from_tile.which_points_within(self.points)
            if len(inpts) <= self.dm:
                self.logwriter.write('SHRINKING TILE ALONG DIMENSION {} NOT POSSIBLE. TILE WOULD NOT CONTAIN AT LEAST {} POINTS'.format(di, self.dm+1))
                return False
            else:
                self.logwriter.write('VERIFIED THAT A SHRUNK TILE ALONG DIMENSION {} WOULD STILL ENCLOSE AT LEAST {} POINTS.'.format(di, self.dm+1))
            
        # Perturb the dimension di of form_tile accordingly if this wasn't a dry run.
        if not dry_run:
            self.logwriter.write('SHRINKING TILE ALONG DIMENSION {}'.format(di))
            from_tile.extend_dimension(di, dx, surface, direction)
            from_tile.print_tile_report()
        else:
            self.logwriter.write('COULD SHRINK TILE ALONG DIMENSION {} BUT THIS WAS A DRY RUN SO I DIDN\'T'.format(di))            
        return True

    def multi_propagate_tile_perturbation(self, tosc, di, dx,
                                          surface, direction,
                                          ignore_tiles=[], dry_run=False):
        """
        Wrapper for propagate_tile_perturbation that takes a list tosc
        of (btile, sface, ctile) where btile is to be perturbed and
        propagated in direction if sface's surface mask matches surface.

        tosc contents are as returned by self.get_osculating_tiles.

        Returns True if the propagation succeeded, False otherwise.

        Only actually applies the propagation if dry_run == False.
        """
        can_propagate_direction = True
        this_ignore_tiles = ignore_tiles[:]
        for btile, sface, ctile in tosc:
            if sface.smask[di] != surface:
                continue
            can_propagate = self.propagate_tile_perturbation(btile, di, dx, surface, direction,
                                                             ignore_tiles=this_ignore_tiles, dry_run=dry_run)
            this_ignore_tiles.append(btile)
            can_propagate_direction = can_propagate_direction and can_propagate
        return can_propagate_direction
            
    def get_tile_boundaries(self, atile, di, allow_bc_types=[BCTypes.all_types]):
        """
        Given atile and a dimension di, return the [lo, hi] boundaries in a BCDim object.

        Account for the types of boundary conditions listed in allow_bc_types.
        """
        bcdi = BCDim()

        if (BCTypes.tile in allow_bc_types or
            BCTypes.all_types in allow_bc_types):
            # Get tile constraints
            bcdi = atile.get_tile_constraints(self.tiles, di, bcdi)

        if (BCTypes.point in allow_bc_types or
            BCTypes.all_types in allow_bc_types):
            # Get point constraints
            bcdi = atile.get_point_constraints(self.scratch_points, di, bcdi)

        # If neither point nor tile constraint, use domain [lo, hi]
        if bcdi.lo_bc == BCTypes.none:
            bcdi.lo_bc = self.lo[di]
        if bcdi.hi_bc == BCTypes.none:
            bcdi.hi_bc = self.hi[di]

        # Return BCDim object
        return bcdi
        
    def set_tile_boundaries(self, atile, allow_bc_types=[BCTypes.all_types]):
        """
        Given atile, sets its [lo, hi] boundaries in each dimension.

        Also updates the boundary masks for adjacent points
        in the tiling list scratch_points.
        """
        revised_tile = False
        # Expand Tile in each dimension as possible
        for di in range(self.dm):
            # Get tile boundaries in dimension di from points and tiles
            bcdi = self.get_tile_boundaries(atile, di,
                                            allow_bc_types=allow_bc_types)

            if not (atile.lo[di] == bcdi.lo_bc and
                    atile.hi[di] == bcdi.hi_bc):
                revised_tile = True
                
            # Now implement [lo_bc, hi_bc] for this tile and dimension di
            atile.lo[di] = bcdi.lo_bc
            atile.hi[di] = bcdi.hi_bc

            # If point constraint, make that point a boundary along di
            if bcdi.lo_bc_type == BCTypes.point:
                # Set point bc masks wrt points remaining in domain
                bcdi.lo_bc_object.bedge[di] = bcdi.lo_bc
                bcdi.lo_bc_object.btype[di] = BCTypes.up
            if bcdi.hi_bc_type == BCTypes.point:
                # Set point bc masks wrt points remaining in domain
                bcdi.hi_bc_object.bedge[di] = bcdi.hi_bc
                bcdi.hi_bc_object.btype[di] = BCTypes.down

            # Explain Yourself!
            # self.logwriter.write('Setting Tile Boundaries along dimension {} for Reasons:'.format(di))
            # self.logwriter.write('lo reason: {}'.format(lo_bc_type))
            # self.logwriter.write('hi reason: {}'.format(hi_bc_type))
            atile.print_tile_report()
            # Go to next dimension
        return revised_tile # True if in some dimension we changed the tile bounds

    def tiling_decision_function(self, L2r_thresh=None, coeff_det_thresh=None,
                                 tilde_resd_thresh=None, tilde_resd_factor=None):
        def dfun(atile):
            accept_tile = True
            if coeff_det_thresh:
                # self.logwriter.write('checking coeff_det_thresh')
                accept_tile = (accept_tile and
                               (atile.get_coeff_det() > coeff_det_thresh))
            if L2r_thresh:
                # self.logwriter.write('checking L2r_thresh')
                accept_tile = (accept_tile and
                               (atile.get_L2_norm_resd() < L2r_thresh))
            if tilde_resd_thresh:
                # self.logwriter.write('checking tilde_resd_thresh')
                accept_tile = (accept_tile and
                               all(atile.get_tilde_resd() < tilde_resd_thresh))
            if tilde_resd_factor:
                # self.logwriter.write('checking tilde_resd_factor')
                if atile.previous_tilde_resd:
                    accept_tile = (accept_tile and
                                   all(atile.get_tilde_resd()/atile.previous_tilde_resd < tilde_resd_factor))
            return accept_tile
        return dfun

    def form_tile(self, decision_function=None, plot_intermediate=False):
        # self.logwriter.write('Executing Domain.form_tile()')
        # self.logwriter.write('Number of points in domain: {}'.format(len(self.scratch_points)))
        # self.logwriter.write('Number of tiles in domain: {}'.format(len(self.tiles)))

        # Select starting point
        # Choose the point with the closest minimum required nearest neighbors
        p_start = None
        pi_start = None
        min_dist_nn = None
        for pi, p in enumerate(self.scratch_points):
            other_points = self.scratch_points[:]
            other_points.pop(pi)
            dnn = p.get_average_dist_nn(plist=other_points,
                                        num_neighbors=self.dm)
            if not min_dist_nn or dnn < min_dist_nn:
                min_dist_nn = dnn
                p_start = p
                pi_start = pi
        if not p_start and self.scratch_points:
            self.logwriter.write('Points remain but no starting point could be found!')
            self.logwriter.write('Number of Domain Points {}'.format(len(self.scratch_points)))
            self.logwriter.write('Number of Domain Tiles {}'.format(len(self.tiles)))
            raise TilingError(err_type=TETypes.cannot_start_point,
                              err_tile=atile,
                              scratch_points=self.scratch_points, 
                              message='Could not find a starting point!')

        # self.logwriter.write('Found starting point')
        # self.logwriter.write('Start index: {}'.format(pi_start))
        # self.logwriter.write('Start position: {}'.format(p_start.r))
            
        # Form tile with starting point
        atile = Tile(points=[p_start], lo=p_start.r, hi=p_start.r, dm=self.dm, writer=self.logwriter)
        self.scratch_points.pop(pi_start)

        # self.logwriter.write('Getting at least n+1 points')
        # Extend to enclose a total n+1 points for n-D parameter space.
        # More points may be enclosed if exactly n+1 isn't possible
        canex = True
        while len(atile.points) < self.dm+1 and canex:
            self.scratch_points, canex = atile.extend_min_volume(plist=self.scratch_points,
                                                                 avoid_tiles=self.tiles,
                                                                 dom_lo=self.lo, dom_hi=self.hi)
            if plot_intermediate:
                self.plot_domain_slice_scratch(atile)
                
            # self.logwriter.write('Attempted tile has {} points'.format(len(atile.points)))
        # self.logwriter.write('Obtained {} points'.format(len(atile.points)))
        
        # Check the number of points, if it's less than n+1,
        # return partially tiled points and raise an exception!
        if len(atile.points) < self.dm+1:
            # First return atile.points to self.scratch_points
            self.scratch_points = self.scratch_points + atile.points
            # Then raise a tiling error
            raise TilingError(err_type=TETypes.cannot_enclose_enough_points,
                              err_tile=atile,
                              scratch_points=self.scratch_points, 
                              message='Could not enclose n+1 points!')
            
        # extend Tile checking the fit decision function decision_function
        # self.logwriter.write('Extending initial tile')
        canex = True
        while canex:
            self.scratch_points, canex = atile.extend_min_volume(plist=self.scratch_points,
                                                                 avoid_tiles=self.tiles,
                                                                 decision_fun=decision_function,
                                                                 dom_lo=self.lo, dom_hi=self.hi)
            if plot_intermediate:
                self.plot_domain_slice_scratch(atile)
            # self.logwriter.write('Attempted tile has {} points'.format(len(atile.points)))
                        
        # set boundaries of atile and update point boundary masks
        # self.logwriter.write('Updating point boundary masks')
        did_update = self.set_tile_boundaries(atile)

        # Add atile to tiles in this domain
        # self.logwriter.write('Adding tile to domain')
        self.tiles.append(atile)

        # Check that at least n+1 points remain in the domain, otherwise warn user!
        if self.scratch_points and len(self.scratch_points) < self.dm+1:
            raise TilingError(err_type=TETypes.few_points_remain,
                              err_tile=atile,
                              scratch_points=self.scratch_points,
                              message='Fewer than n+1 points remain!')

    def extend_existing_tiles(self, decision_function=None):
        """
        Extends all existing tiles, gobbling up scratch_points as possible.

        Do this by figuring out which tile can best include each of the remaining scratch_points,
        given the decision_function constraint.
        """
        canex_tiles = False
        for j, p in enumerate(self.scratch_points):
            other_points = self.scratch_points[:]
            other_points.pop(j)
            min_L2r = None
            min_L2r_stile = None
            min_L2r_atile_i = None
            min_out_spts = None
            for i, atile in enumerate(self.tiles):
                other_tiles = self.tiles[:]
                other_tiles.pop(i)
                try_points = [p]
                stile, in_spts, out_spts = atile.get_hypothetical_extend(points=try_points, avoid_tiles=other_tiles,
                                                                         greedy_absorb_points=other_points)
                if stile:
                    dbool = True
                    if callable(decision_function):
                        dbool = decision_function(stile)
                    if dbool and (not min_L2r or stile.get_L2_norm_resd() < min_L2r):
                        min_L2r = stile.get_L2_norm_resd()
                        min_L2r_stile = stile
                        min_L2r_atile_i = i
                        min_out_spts = out_spts[:]
            if min_L2r_stile:
                canex_tiles = True
                # set boundaries of min_L2r_stile and update point boundary masks
                # self.logwriter.write('Updating point boundary masks and replacing atile=>stile')
                self.scratch_points = min_out_spts[:]
                self.tiles.pop(min_L2r_atile_i)
                did_update = self.set_tile_boundaries(min_L2r_stile)
                self.tiles.append(min_L2r_stile)
                break
        return canex_tiles

    def bound_existing_tiles(self):
        """
        Given the tiles in self.tiles, update all their tile-based boundaries
        until no further updates can be made.
        """
        revised_tiles = True
        while revised_tiles:
            revised_tiles = False
            for atile in self.tiles:
                revised_atile = self.set_tile_boundaries(atile,
                                                         allow_bc_types=[BCTypes.tile])
                revised_tiles = revised_tiles or revised_atile

    def get_osculating_tiles(self, atile, di, direction=BCTypes.none,
                             get_other_sface=False, return_other_tile=False):
        """
        Get the tiles in Domain which osculate atile
        along the dimension di. Return them as a list
        tosc = [(sface, ctile), ...]
        where sface and ctile are as in Tile.whether_osculates_tile

        If direction == BCTypes.up or direction == BCTypes.down
        then only return the tiles which osculate atile such that the
        osculating surface of atile has a surface normal oriented along direction.
        Otherwise, if direction == BCTypes.none, return tiles which osculate
        atile in any direction along di. Note that this direction 
        should be relative to atile regardless the value of get_other_sface.

        If get_other_sface, then sface will correspond to the 
        surface of the Tile which osculates self.

        If return_other_tile, will return the tuples [(stile, sface, ctile), ...]
        where stile is the Tile of which sface is the surface.
        """
        if direction != BCTypes.none and get_other_sface:
            rel_direction = -direction
        else:
            rel_direction = direction
        tosc = []
        for ibtile, btile in enumerate(self.tiles + self.virtual_tiles):
            if not atile.colocated_with(btile):
                self.logwriter.write('CHECKING OSCULATION with domain tile {}'.format(ibtile))
                if get_other_sface:
                    (sface, ctile) = btile.whether_osculates_tile(atile, di,
                                                                  direction=rel_direction)
                else:
                    (sface, ctile) = atile.whether_osculates_tile(btile, di,
                                                                  direction=rel_direction)
                if ctile and not sface.colocated_with(ctile):
                    self.logwriter.write('FOUND OSCULATION along direction {}, dimension {} with domain tile {}'.format(direction,di,ibtile))
                    if return_other_tile:
                        tosc.append((btile, sface, ctile))
                    else:
                        tosc.append((sface, ctile))
                else:
                    self.logwriter.write('NO PROPER SUBSET OSCULATION with domain tile {}'.format(ibtile))
        return tosc
                
    def do_empty_tiling(self, plot_tile_surfaces=False):
        """
        Creates empty virtual subtiles to cover the domain dom.

        Adds the created virtual subtiles to the domain dom.

        Returns True if virtual Tiles were created.

        Returns False if no virtual Tiles could be created.

        Because the Domain Tile loop doesn't update itself
        as Tiles are added to the Domain, you should loop over
        this function until it returns None to indicate
        the entire Domain has been Tiled.
        """
        self.logwriter.write('Entered DO_EMPTY_TILING')
        dom_tile = Tile(lo=self.lo, hi=self.hi, virtual=True, writer=self.logwriter)
        ncdim = dom_tile.get_nonconstant_dimensions()
        self.logwriter.write('nonconstant dimensions: {}'.format(ncdim))
        self.logwriter.write('self.dm: {}'.format(self.dm))
        self.logwriter.write('dom_tile.dm: {}'.format(dom_tile.dm))
        created_virtual_tiles = False
        # Loop over Tiles in Domain
        for iatile, atile in enumerate(self.tiles + self.virtual_tiles):
            self.logwriter.write('examining ATILE {}'.format(iatile))
            atile.print_tile_report()
            # Find the dimensions in which Tile has
            # a nonzero extent and can exhibit
            # a surface osculation with another Tile.
            # The Point Tiling passes over the Domain
            # will have ensured all point-containing
            # fully-dimensional Tiles will have extent
            # in every dimension because of the
            # Point-constrained boundary conditions.
            # For each nonconstant dimension in ncdim,
            # Find the Surfaces of atile which osculate the
            # other Tiles in Domain as (sface, ctile)
            # where sface is a Surface of atile and
            # where ctile is the intersection of sface
            # and another Tile in Domain.
            # Create a list of objects (sface, ctile): tosc
            # such that sface.lo != ctile.lo and sface.hi != ctile.hi
            for di in ncdim:
                self.logwriter.write('dimension {}'.format(di))
                tosc = self.get_osculating_tiles(atile, di)
                self.logwriter.write('TOSC LENGTH: {}'.format(len(tosc)))
                self.logwriter.write('atile.smask: {}'.format(atile.smask))
                for iaface, aface in enumerate(atile.get_surfaces(di)):
                    self.logwriter.write('examining AFACE'.format(iaface))
                    aface.print_tile_report()
                    # For each unique surface in atile (aface):
                    # Get a list of all (sface, ctile) for which surface==sface.
                    # If there is a surface of atile with no entries in tosc,
                    # then it is osculated by no other tile and
                    # a virtual tile can be extended from that entire surface.
                    surface_tiles = []
                    for (sface, ctile) in tosc:
                        if aface.colocated_with(sface):
                            self.logwriter.write('appending ctile')
                            surface_tiles.append(ctile)
                    if not surface_tiles:
                        self.logwriter.write('no surface_tiles, using aface')
                        surface_tiles.append(aface)

                    # Create Domain sdom from aface
                    sdom = Domain(lo=aface.lo, hi=aface.hi,
                                  plot_lo=self.plot_lo, plot_hi=self.plot_hi,
                                  last_domain_slice=self.last_domain_slice)
                    
                    if len(ncdim) == 1:
                        self.logwriter.write('ONE DIMENSIONAL')
                        sdom.virtual_tiles.append(aface)
                    else:
                        sdom.tiles = surface_tiles
                        self.logwriter.write('CALLING DO_EMPTY_TILING RECURSIVELY')
                        sdom.do_empty_tiling(plot_tile_surfaces=plot_tile_surfaces)

                    # Extend the virtual tiles on sdom along di,
                    # using the Tile constraints of Domain self.
                    # Include both real and virtual tiles to set
                    # the constraints.
                    # Add the virtual tiles to Domain self as virtual tiles
                    # ONLY IF YOU COULD DO A NONZERO EXTENSION ALONG DI
                    for vtile in sdom.virtual_tiles:
                        self.logwriter.write('vtile smask: {}'.format(vtile.smask))
                        bcdi = vtile.get_tile_constraints(tiles=(self.tiles +
                                                                 self.virtual_tiles),
                                                          di=di)
                        vtile_could_extend = False
                        if vtile.smask[di] == BCTypes.down:
                            if bcdi.lo_bc == BCTypes.none:
                                # Use the domain wall if no tile constraint
                                self.logwriter.write('using domain wall')
                                vtile_lo = self.lo[di]
                            else:
                                # Use the tile constraint
                                self.logwriter.write('using tile constraint')
                                vtile_lo = bcdi.lo_bc
                            if vtile_lo < vtile.lo[di]:
                                # This is an extension of vtile
                                vtile_could_extend = True
                            vtile.lo[di] = vtile_lo
                        elif vtile.smask[di] == BCTypes.up:
                            if bcdi.hi_bc == BCTypes.none:
                                # Use the domain wall if no tile constraint
                                self.logwriter.write('using domain wall')
                                vtile_hi = self.hi[di]
                            else:
                                # Use the tile constraint
                                self.logwriter.write('using tile constraint')
                                vtile_hi = bcdi.hi_bc
                            if vtile_hi > vtile.hi[di]:
                                # This is an extension of vtile
                                vtile_could_extend = True
                            vtile.hi[di] = vtile_hi
                        else:
                            # Something went horribly wrong setting the virtual tile smask
                            self.logwriter.write('THATS IMPOSSIBLE!!')
                            exit()
                            
                        # If you can extend vtile along di, update vtile.smask
                        # and add it to self.virtual_tiles
                        if vtile_could_extend:
                            self.logwriter.write('could extend vtile. appending to domain self.')
                            vtile.smask[di] = BCTypes.none
                            self.virtual_tiles.append(vtile)
                            # Update boolean created_virtual_tiles
                            created_virtual_tiles = True
                            
                    # Plot/Print sdom domain report
                    self.logwriter.write('PLOTTING DOMAIN SDOM')
                    if plot_tile_surfaces:
                        sdom.plot_domain_slice(show_tile_id=False,
                                               save_num='ncd-{}_iat-{}_iaf-{}_di-{}'.format(len(ncdim),
                                                                                            iatile, iaface, di),
                                               underlay_figure_axis=self.last_domain_slice)
                    self.logwriter.write('PRINTING SDOM REPORT')
                    sdom.print_domain_report()
            # Return before continuing to the next atile
            # You want the loop to consider the
            # virtual tiles you just added to this Domain
            # so call this function again if created_virtual_tiles == True.
            if created_virtual_tiles:
                break
        return created_virtual_tiles

    def shrink_virtual_tiles(self):
        """
        Shrink a virtual tile V in the domain to zero volume by
        rearranging neighboring tiles.

        The algorithm is outlined below:
        Pop a virtual tile V off the Domain's list of virtual_tiles.
        Shrink virtual tile V by identifying its thinnest dimension di 
        (of width W) and finding the tiles B of maximum volume with surfaces S 
        which V osculates along di (B may be up or down relative to di, but not both).
        Take the surfaces S and form a virtual tile (SW) of thickness W extending from S
        in the direction of V. Find all tiles T, T != V, which SW overlaps.
        Shrink all tiles T away from B in the dimension di by length W.
        Expand tiles B into the volume of SW.
        Remove virtual tile V from domain.
        Return and Repeat until no virtual tiles remain.

        Real or virtual tiles will have a problem if they osculate
        the virtual tile but are thinner than the
        virtual tile in the osculating dimension.
        To get around that, shrink_virtual_tiles should
        check to see if its smallest dimension 
        is not thicker than its osculating tiles along
        that dimension. If that's not true, 
        then it will not be possible
        to reduce such a virtual tile.

        Reallocate points to real tiles and repeat fitting to update stats.
        This has to be done in whatever code calls this function.
        """
        for ivtile, vtile in enumerate(self.virtual_tiles):
            # Find the list of dimensions of vtile ordered
            # from thinnest to thickest.
            self.logwriter.write('EXAMINING VTILE {} FOR SHRINKING'.format(ivtile))
            for di in vtile.order_thinnest_dimensions():
                self.logwriter.write('EXAMINING DIMENSION {}'.format(di))
                dx = vtile.get_dim_thickness(di)
                # Find the domain tiles which osculate vtile along di
                # This includes real and virtual tiles.
                tosc = self.get_osculating_tiles(vtile, di, get_other_sface=True, return_other_tile=True)
                if not tosc:
                    self.logwriter.write('ERROR: VIRTUAL TILE {} DOES NOT OSCULATE A DOMAIN TILE ALONG DIMENSION {}'.format(ivtile, di))
                    exit()
                else:
                    self.logwriter.write('--- VTILE OSCULATES THE FOLLOWING DOMAIN TILES: ---')
                    for btile, sface, ctile in tosc:
                        btile.print_tile_report()

                # Find the direction in which to collapse vtile 
                # which allows expanding the nearby tiles of largest volume
                volume_up = 0.0
                volume_down = 0.0
                for btile, sface, ctile in tosc:
                    if sface.smask[di] == BCTypes.up:
                        volume_up += btile.get_volume(dom_lo=self.lo, dom_hi=self.hi)
                    else:
                        volume_down += btile.get_volume(dom_lo=self.lo, dom_hi=self.hi)
                if volume_up > volume_down:
                    start_direction = BCTypes.up
                else:
                    start_direction = BCTypes.down

                could_propagate = True
                for direction in [start_direction, -start_direction]:
                    self.logwriter.write('CHECKING DIRECTION {}'.format(direction))
                    could_propagate = True
                    can_propagate_direction = self.multi_propagate_tile_perturbation(tosc, di, dx,
                                                                                     direction, direction,
                                                                                     ignore_tiles=[vtile],
                                                                                     dry_run=True)
                    could_propagate = could_propagate and can_propagate_direction
                    if can_propagate_direction:
                        self.logwriter.write('CAN PROPAGATE VTILE SHRINK FOR VTILE {}, DIMENSION {}, DIRECTION {}'.format(ivtile, di, direction))
                        could_propagate = self.multi_propagate_tile_perturbation(tosc, di, dx,
                                                                                 direction, direction,
                                                                                 ignore_tiles=[vtile],
                                                                                 dry_run=False)
                        break
                    else:
                        self.logwriter.write('CANNOT PROPAGATE VTILE SHRINK FOR VTILE {}, DIMENSION {}, DIRECTION {}'.format(ivtile, di, direction))
                        continue
                    
                # Determine if I did a propagation above and pop vtile and return if so.
                if could_propagate:
                    # Pop vtile from self.virtual_tiles
                    self.logwriter.write('>>>SUCCESSFULLY SHRUNK VIRTUAL TILE {}'.format(ivtile))
                    self.virtual_tiles.pop(ivtile)
                    # Return to the calling code so
                    # the loop over self.virtual_tiles
                    # can be restarted to avoid indexing
                    # errors due to the pop.
                    return True # True to indicate we eliminated a vtile
        return False # Return False if no virtual tiles could be eliminated.

    def create_virtual_tiles(self, make_plots=False, plot_tile_surfaces=False):
        """
        Tile all untiled space in the domain into virtual tiles
        and add them to self.virtual_tiles.

        Returns True if a new virtual tile was created. 
        Returns False otherwise.
        """
        created_new_virtual = False
        created_virtual_tiles = True
        while created_virtual_tiles:
            self.logwriter.write('>>>CALLING DO_EMPTY_TILING')
            created_virtual_tiles = self.do_empty_tiling(plot_tile_surfaces=plot_tile_surfaces)
            created_new_virtual = created_new_virtual or created_virtual_tiles
            if make_plots and created_virtual_tiles:
                self.plot_domain_slice(show_tile_id=False)
        return created_new_virtual

    def static_tile_assign_points(self):
        """
        Statically assign points to the tiles in the Domain
        and update the fits on those tiles. Existing points
        in the tiles are reset to only those points
        assigned here.
        """
        for iatile, atile in enumerate(self.tiles):
            in_pts, out_pts = atile.which_points_within(self.points)
            if set(in_pts) != set(atile.points):
                self.logwriter.write('TILE {} UPDATE CONTAINED POINTS!! PREVIOUS NUMPTS = {}, NEW NUMPTS = {}'.format(iatile, len(atile.points), len(in_pts)))
            # Set tile points to the points it now contains
            atile.points = in_pts[:]
            # Update the fit on the tile
            atile.do_plane_fit()
            # If any of in_pts were in scratch_points (i.e., they remained untiled)
            # then remove them from scratch_points
            for p in in_pts:
                try:
                    ip = self.scratch_points.index(p)
                    self.scratch_points.pop(ip)
                except:
                    continue
                
    def do_domain_tiling(self, L2r_thresh=None, coeff_det_thresh=None,
                         tilde_resd_thresh=None, tilde_resd_factor=None,
                         attempt_virtual_shrink=False,
                         plot_tile_surfaces=False, plot_intermediate=False,
                         plot_tiling=False, plot_final=True):
        # Initialize a list of scratch points for tiling
        self.scratch_points = self.points[:]
        # Clear current tiling
        self.tiles = []
        # Get the decision function
        decision_function = self.tiling_decision_function(L2r_thresh=L2r_thresh,
                                                          coeff_det_thresh=coeff_det_thresh,
                                                          tilde_resd_thresh=tilde_resd_thresh,
                                                          tilde_resd_factor=tilde_resd_factor)
        # Tile the domain given the decision function
        try:
            while self.scratch_points:
                self.form_tile(decision_function=decision_function,
                               plot_intermediate=plot_intermediate)
                if plot_tiling:
                    self.plot_domain_slice(show_tile_id=True)
        except TilingError as terr:
            self.logwriter.write(terr.message)
            self.logwriter.write('Number of points in attempted tile: {}'.format(
                len(terr.err_tile.points)))
            self.logwriter.write('Number of points remaining in domain: {}'.format(
                len(terr.scratch_points)))
            if (terr.err_type == TETypes.few_points_remain or
                terr.err_type == TETypes.cannot_enclose_enough_points):
                # Distribute remaining points among existing tiles
                # decision_function constraint is ignored, so
                # tile overlap is the only constraint.
                canex = True
                while self.scratch_points and canex:
                    self.logwriter.write('EXTENDING EXISTING TILES')
                    canex = self.extend_existing_tiles()
                    if plot_tiling:
                        self.plot_domain_slice(show_tile_id=True)
            else:
                raise

        # Update the boundaries of existing tiles to help eliminate empty untiled space
        self.bound_existing_tiles()
        if plot_tiling:
            self.plot_domain_slice(show_tile_id=True, save_last_figure=True)
                    
        # Tile any remaining empty untiled space into virtual tiles
        created_virtual_tiles = self.create_virtual_tiles(make_plots=plot_tiling,
                                                          plot_tile_surfaces=plot_tile_surfaces)
        
        if attempt_virtual_shrink:
            # Shrink virtual tiles to zero volume by shifting neighboring real tiles as possible
            num_virtual_tiles = len(self.virtual_tiles)
            could_shrink_virtual = True
            while could_shrink_virtual:
                self.logwriter.write('>>>CALLING SHRINK_VIRTUAL_TILES')
                could_shrink_virtual = self.shrink_virtual_tiles()
                self.plot_domain_slice(show_tile_id=True)
            if len(self.virtual_tiles) == num_virtual_tiles and num_virtual_tiles != 0:
                self.logwriter.write('>>>WARNING: COULD NOT SHRINK ALL VIRTUAL TILES -- {} REMAIN'.format(len(self.virtual_tiles)))
            # Reallocate points to real tiles and repeat fitting to update stats.
            self.static_tile_assign_points()

            # See if there exists any new empty untiled space that should be a virtual tile
            # that isn't already in self.virtual_tiles. Complain if there is, that's a bug.
            created_virtual_tiles = self.create_virtual_tiles(make_plots=plot_tiling,
                                                              plot_tile_surfaces=plot_tile_surfaces)
            if created_virtual_tiles:
                self.logwriter.write('ERROR: VIRTUAL TILE CREATED AFTER SHRINK_VIRTUAL_TILES!!!')
                exit()
        
        # Output Results
        if plot_final:
            self.plot_domain_slice()
        self.print_domain_report(self.logwriter)
        if self.sumwriter.ofile:
            self.print_domain_report(self.sumwriter)
        self.logwriter.write('COMPLETED DOMAIN TILING!')
