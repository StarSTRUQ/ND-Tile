"""
Tile an n-D domain containing Point objects depending on a decision function.

Donald Willcox
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from Plane_nd import Plane_nd

class BCTypes(object):
    # Boundary Types
    up    = +1
    none  = None
    down  = -1
    tile  = +2
    point = +3

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
        self.r = np.array(r)
        # Value of point (Scalar)
        self.v = v
        self.dm = len(r)
        # n-D Mask: Boundary Edge represented by Point
        self.bedge = [None for i in range(self.dm)]
        # n-D Mask: Boundary Type represented by Point
        self.btype = [BCTypes.none for i in range(self.dm)]

    def dist_to_pt(self, b):
        # Get distance between this Point and Point b
        dr = np.array(self.r) - np.array(b.r)
        return np.sqrt(np.sum(dr**2))

    def order_nn(self, plist=[]):
        # Orders the points in plist by nearest neighbors to self.
        if not list(plist):
            return None
        retlist = sorted(plist, key=(lambda p: self.dist_to_pt(p)))
        return retlist

    def get_average_dist_nn(self, plist=[], num_neighbors=1):
        # Given the point list plist, find the average distance
        # of the nearest num_neighbors to self.
        if not list(plist) or num_neighbors > len(plist):
            return None
        ordered_plist = self.order_nn(plist)
        which_nn = ordered_plist[:num_neighbors]
        dist_nn = np.array([self.dist_to_pt(nn) for nn in which_nn])
        ave_dist_nn = np.mean(dist_nn)
        return ave_dist_nn

class Plane(object):
    def __init__(self, points=None, fit_guess=None, dm=None, lo=[], hi=[]):
        # fit_guess should provide, well, a guess for the fit parameters
        # if fit_guess isn't supplied, it will be estimated from the points.
        self.cpars = None # Length n+1 for n-D space
        self.dm = dm
        self.resd  = None
        self.norm_resd = None
        self.geom_norm_resd = None
        self.tilde_resd = None
        self.lo = lo
        self.hi = hi
        self.center = None
        if list(self.lo) and list(self.hi):
            # Determine center and width from [lo, hi]
            self.center = 0.5*(self.lo + self.hi)
            self.width = self.hi - self.lo
        self.compute_pars(points, fit_guess)
        
    def compute_pars(self, points, fit_guess):
        if not points:
            return
        ivars = np.array([p.r for p in points])
        dvars = np.array([p.v for p in points])
        fitter = Plane_nd(ivars, dvars, self.dm)
        popt, pcov = fitter.dolsq(fit_guess)
        dpfit = np.array([fitter.fplane(ivr, popt) for ivr in ivars])
        if not list(self.center):
            self.center = np.mean(ivars, axis=0)
        if not list(self.width):
            self.width = np.amax(ivars, axis=0) - np.amin(ivars, axis=0)
        self.resd = dvars - dpfit
        
        self.abs_delta_pos = np.absolute(np.array([ivr-self.center for ivr in ivars]))
        self.tilde_resd = np.absolute(np.sum(self.resd * np.transpose(self.abs_delta_pos), axis=1))
        # normalize tilde_resd by the central plane value and the tile dimensions
        self.tilde_resd = self.tilde_resd/(self.center * self.width)
        
        self.norm_resd = self.resd/dvars
        self.geom_norm_resd = np.sqrt(np.sum(self.norm_resd**2))
        
class Tile(object):
    def __init__(self, points=[], lo=[], hi=[], fit_guess=None, dm=None):
        self.points = points
        self.fit_guess = fit_guess
        self.plane_fit = None
        self.previous_tilde_resd = None # Value of tilde_resd on the previous plane fit
        self.fresh_plane_fit = False # Is the plane fit current?
        self.lo = lo
        self.hi = hi
        self.dm = dm
        if all(points) and not dm:
            self.dm = points[0].dm
        if not list(self.lo) or not list(self.hi):
            self.boundary_minimize()

    def print_tile_report(self, tile_number=None):
        """Prints report of this Tile"""
        print('---TILE {} REPORT---'.format(tile_number))
        print('lo = {}, hi = {}, npts = {}'.format(self.lo, self.hi, len(self.points)))
        print('---GEOM. MEAN NORM RESD. = {} ---'.format(self.get_geom_norm_resd()))
        print('---TILDE RESD. = {} ---'.format(self.get_tilde_resd()))
        print('-------POINTS------')
        for p in self.points:
            print('{}: {}'.format(p.r, p.v))

    def get_volume(self):
        """
        Computes volume of the Tile. If [lo, hi] is undefined, return None.
        """
        if not list(self.lo) or not list(self.hi):
            return None
        dr = np.array(self.hi) - np.array(self.lo)
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

    def overlaps_tile_dimension(self, reftile, di):
        """
        Checks to see if self overlaps reftile in the dimension di:

        reftile must be a Tile object

        di must be an integer in range(self.dm)
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

    def get_hypothetical_extend(self, points=[], avoid_tiles=None, greedy_absorb_points=[]):
        # Returns a hypothetical tile with points consisting of self.points + points
        # Extension will avoid tiles in avoid_tiles
        # If greedy_absorb_points is a list of Points,
        # Tile will absorb as many points in greedy_absorb_points as
        # it encloses where the Tile extent is determined by the list `points`.
        # Returns the new Tile, in_pts, out_pts
        # in_pts  : points in greedy_absorb_points now within the Tile
        # out_pts : points in greedy_absorb_points which Tile couldn't absorb
        stile = Tile(points=self.points + points, fit_guess=self.fit_guess)
        if stile.overlaps_tiles(avoid_tiles):
            return None, [], greedy_absorb_points
        else:
            # Absorb as many points as allowed from greedy_absorb_points
            in_pts, out_pts = stile.which_points_within(greedy_absorb_points)
            stile.extend_points(in_pts)
            return stile, in_pts, out_pts

    def extend_min_volume(self, plist=[], avoid_tiles=None, decision_fun=None):
        """
        Given the list of points (plist), extends the Tile
        by adding one point from plist to Tile where the point 
        is selected from plist such that it minimizes the volume
        of Tile.

        Returns plist where the selected point is popped from the list.

        If avoid_tiles is passed, it should be a list of Tile objects.
        The current Tile will then only be extended such that it does
        not intersect the tiles in avoid_tiles.

        If a function is passed as decision_fun, this Tile will be passed to
        the 'decision function' to determine whether to extend the tile.
        decision_fun should take a single Tile argument and return True or False
        """
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
            svol = stile.get_volume()
            dbool = True
            if callable(decision_fun):
                dbool = decision_fun(stile)
                # print('dbool = {}'.format(dbool))
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
        stile = Tile(inpts, lo, hi, self.dm)
        return stile

    def do_plane_fit(self):
        if self.plane_fit:
            self.previous_tilde_resd = self.plane_fit.tilde_resd
        p = Plane(points=self.points, fit_guess=self.fit_guess,
                  dm=self.dm, lo=self.lo, hi=self.hi)
        self.plane_fit = p
        self.fit_guess = p.cpars
        self.fresh_plane_fit = True

    def get_geom_norm_resd(self):
        # Returns geometric mean of normalized residuals
        # between the points in the Tile and a Plane fit.
        if not self.fresh_plane_fit:
            self.do_plane_fit()
        return self.plane_fit.geom_norm_resd

    def get_tilde_resd(self):
        # Returns residual-weighted sum of central deviances
        # between the points in the Tile and a Plane fit.
        if not self.fresh_plane_fit:
            self.do_plane_fit()
        return self.plane_fit.tilde_resd

class Domain(object):
    def __init__(self, points=[], lo=[], hi=[], dm=None):
        # The Domain is just a set of Point objects
        # and functions for tiling them into a set of Tile objects.        
        self.tiles = []
        self.lo = lo
        self.hi = hi
        self.dm = dm
        self.points = points
        self.plot_num = 0

        if lo and hi and len(lo) != len(hi):
            print('ERROR: lo and hi supplied with incongruous dimensions.')
            exit()
        else:
            if not self.dm:
                self.dm = len(lo)

        # Set up dimension cycling
        self.dm_cycle = DMCycle(self.dm)

        # Set up boundary masks for points
        if list(points) and list(lo) and list(hi):
            self.bc_init_mask_points(self.points)

    def plot_domain_slice(self, dimx=0, dimy=1, save_num=None, show_tile_id=True):
        if self.dm < 2:
            return
        # Plot 2-D Slice of Domain
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Plot Tile outlines
        linestyle_options = ['-', '--', '-.', ':']
        ls_cycler = DMCycle(len(linestyle_options))
        for i, t in enumerate(self.tiles):
            # Plot Tile outline
            ax.add_patch(Rectangle((t.lo[dimx], t.lo[dimy]), t.hi[dimx]-t.lo[dimx], t.hi[dimy]-t.lo[dimy], facecolor='None',
                                   edgecolor='orange',
                                   linewidth=1.5,
                                   linestyle=linestyle_options[ls_cycler.cycle()]))
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
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cmap = mpl.cm.viridis
        if list(self.points):
            point_scalar_range = np.array([p.v for p in self.points])
        else:
            point_scalar_range = np.array([p.v for p in self.scratch_points])            
        bounds = np.linspace(np.amin(point_scalar_range), np.amax(point_scalar_range), num=5)
        norm = mpl.colors.Normalize(vmin=np.amin(bounds),vmax=np.amax(bounds))
        img = ax.scatter(points_x, points_y, c=points_v, cmap=cmap, norm=norm)
        plt.colorbar(img, cmap=cmap, cax=cax, ticks=bounds, norm=norm, label='Scalar Value')
        # Plot points outside Tiles
        for i, p in enumerate(self.scratch_points):
            ax.scatter(p.r[dimx], p.r[dimy], color='red')
        ax.set_ylabel('Dimension {}'.format(dimy))
        ax.set_xlabel('Dimension {}'.format(dimx))
        plt.tight_layout()
        if not save_num:
            self.plot_num += 1
            this_plot_num = self.plot_num
        else:
            this_plot_num = save_num
        print('SAVING PLOT NUMBER {}'.format(this_plot_num))
        plt.savefig('tiled_domain_{}.eps'.format(this_plot_num))
        plt.savefig('tiled_domain_{}.png'.format(this_plot_num))
        plt.close()

    def print_domain_report(self):
        # Prints full (scratch) domain data
        print('---DOMAIN REPORT---')
        print('DOMAIN LO = {}'.format(self.lo))
        print('DOMAIN HI = {}'.format(self.hi))
        print('-------POINTS------')
        for p in self.scratch_points:
            print('{}: {}'.format(p.r, p.v))
        print('--------TILES------')
        for i, t in enumerate(self.tiles):
            t.print_tile_report(tile_number=i)
        
    def bc_init_mask_points(self, plist):
        # Set initial boundary masks for list of points plist given domain boundaries
        # Initial because uses domain lo, hi
        dm_hi_pts = [None for i in range(self.dm)]
        dm_hi_val = self.lo[:]
        dm_lo_pts = [None for i in range(self.dm)]
        dm_lo_val = self.hi[:]
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
            # print('BC HI PT {}'.format(di))
            # print('position: {}'.format(p.r))
            p.bedge[di] = self.hi[di]
            p.btype[di] = BCTypes.up
            # print('bedge: {}'.format(p.bedge[di]))
            # print('btype: {}'.format(p.btype[di]))
            
        for di, p in enumerate(dm_lo_pts):
            # print('BC LO PT {}'.format(di))
            # print('position: {}'.format(p.r))
            p.bedge[di] = self.lo[di]
            p.btype[di] = BCTypes.down
            # print('bedge: {}'.format(p.bedge[di]))
            # print('btype: {}'.format(p.btype[di]))

    def set_tile_boundaries(self, atile):
        """
        Given atile, sets its [lo, hi] boundaries in each dimension.

        Also updates the boundary masks for adjacent points
        in the tiling list scratch_points.
        """
        # Expand Tile in each dimension as possible
        for di in range(self.dm):
            # Find the tiles which atile does not overlap in dimension di
            # but does overlap in every other dimension.
            # These tiles set the bounds on extensions along dimension di.
            # (If there is an additional dimension along which the tiles
            # do not overlap, then no constraint can be made along di.)
            otiles = []
            for ktile in self.tiles:
                if atile.overlaps_tile_dimension(ktile, di):
                    # ktile overlaps along di, so can't constrain di
                    continue
                kandidate = True
                for dj in range(self.dm):
                    if dj==di:
                        continue
                    if not atile.overlaps_tile_dimension(ktile, dj):
                        # ktile doesn't overlap along dj, dj =/= di
                        # so can't constrain di
                        kandidate = False
                        break
                if kandidate:
                    otiles.append(ktile)

            # Find the points which atile does not overlap in dimension di
            # but does overlap in every other dimension.
            # These points set the bounds on extensions along dimension di.
            # (If there is an additional dimension along which tile and points
            # do not overlap, then no constraint can be made along di.)
            opoints = []
            for p in self.scratch_points:
                if atile.overlaps_point_dimension(p, di):
                    # p overlaps along di, so can't constrain di
                    continue
                kandidate = True
                for dj in range(self.dm):
                    if dj==di:
                        continue
                    if not atile.overlaps_point_dimension(p, dj):
                        # p doesn't overlap along dj, dj =/= di
                        # so can't constrain di
                        kandidate = False
                        break
                if kandidate:
                    opoints.append(p)
            
            # Setup bc data structures for figuring out boundaries
            lo_bc = BCTypes.none
            lo_bc_type = BCTypes.none
            lo_bc_object = BCTypes.none
            hi_bc = BCTypes.none
            hi_bc_type = BCTypes.none
            hi_bc_object = BCTypes.none

            # Get tile constraint on di for [lo, hi]
            for btile in otiles:
                # Check if btile can constrain lo along di
                if btile.hi[di] <= atile.lo[di]:
                    if lo_bc == BCTypes.none or btile.hi[di] > lo_bc:
                        lo_bc = btile.hi[di] + np.finfo(float).eps
                        lo_bc_type = BCTypes.tile
                        lo_bc_object = btile

                # Check if btile can constrain hi along di
                if btile.lo[di] >= atile.hi[di]:
                    if hi_bc == BCTypes.none or btile.lo[di] < hi_bc:
                        hi_bc = btile.lo[di] - np.finfo(float).eps
                        hi_bc_type = BCTypes.tile
                        hi_bc_object = btile

            # Get point constraint on di for [lo, hi]
            for p in opoints:
                # Check if p can constrain lo along di
                if p.r[di] <= atile.lo[di]:
                    phalf = 0.5*(p.r[di] + atile.lo[di])
                    if lo_bc == BCTypes.none or phalf > lo_bc:
                        lo_bc = phalf
                        lo_bc_type = BCTypes.point
                        lo_bc_object = p

                # Check if p can constrain hi along di
                if p.r[di] >= atile.hi[di]:
                    phalf = 0.5*(p.r[di] + atile.hi[di])
                    if hi_bc == BCTypes.none or phalf < hi_bc:
                        hi_bc = phalf
                        hi_bc_type = BCTypes.point
                        hi_bc_object = p
            
            # If neither point nor tile constraint, use domain [lo, hi]
            if lo_bc == BCTypes.none:
                lo_bc = self.lo[di]
            if hi_bc == BCTypes.none:
                hi_bc = self.hi[di]

            # Now implement [lo_bc, hi_bc] for this tile and dimension di
            atile.lo[di] = lo_bc
            atile.hi[di] = hi_bc

            # If point constraint, make that point a boundary along di
            if lo_bc_type == BCTypes.point:
                # Set point bc masks wrt points remaining in domain
                lo_bc_object.bedge[di] = lo_bc
                lo_bc_object.btype[di] = BCTypes.up
            if hi_bc_type == BCTypes.point:
                # Set point bc masks wrt points remaining in domain
                hi_bc_object.bedge[di] = hi_bc
                hi_bc_object.btype[di] = BCTypes.down

            # Explain Yourself!
            # print('Setting Tile Boundaries along dimension {} for Reasons:'.format(di))
            # print('lo reason: {}'.format(lo_bc_type))
            # print('hi reason: {}'.format(hi_bc_type))
            atile.print_tile_report()
            # Go to next dimension

    def tiling_decision_function(self, gnr_thresh=None, tilde_resd_thresh=None,
                                 tilde_resd_factor=None):
        def dfun(atile):
            accept_tile = True
            if gnr_thresh:
                # print('checking gnr_thresh')
                accept_tile = (accept_tile and
                               (atile.get_geom_norm_resd() < gnr_thresh))
            if tilde_resd_thresh:
                # print('checking tilde_resd_thresh')
                accept_tile = (accept_tile and
                               all(atile.get_tilde_resd() < tilde_resd_thresh))
            if tilde_resd_factor:
                # print('checking tilde_resd_factor')
                if atile.previous_tilde_resd:
                    accept_tile = (accept_tile and
                                   all(atile.get_tilde_resd()/atile.previous_tilde_resd < tilde_resd_factor))
            return accept_tile
        return dfun

    def form_tile(self, decision_function=None):
        # Cycle through dimensions
        di = self.dm_cycle.cycle()

        # print('Executing Domain.form_tile()')
        # print('Number of points in domain: {}'.format(len(self.scratch_points)))
        # print('Number of tiles in domain: {}'.format(len(self.tiles)))

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
            print('Points remain but no starting point could be found!')
            print('Dimension {}'.format(di))
            print('Number of Domain Points {}'.format(len(self.scratch_points)))
            print('Number of Domain Tiles {}'.format(len(self.tiles)))
            raise TilingError(err_type=TETypes.cannot_start_point,
                              err_tile=atile,
                              scratch_points=self.scratch_points, 
                              message='Could not find a starting point!')

        # print('Found starting point')
        # print('Start index: {}'.format(pi_start))
        # print('Start position: {}'.format(p_start.r))
            
        # Form tile with starting point
        atile = Tile(points=[p_start], lo=p_start.r, hi=p_start.r, dm=self.dm)
        self.scratch_points.pop(pi_start)

        # print('Getting at least n+1 points')
        # Extend to enclose a total n+1 points for n-D parameter space.
        # More points may be enclosed if exactly n+1 isn't possible
        canex = True
        while len(atile.points) < self.dm+1 and canex:
            self.scratch_points, canex = atile.extend_min_volume(plist=self.scratch_points,
                                                                 avoid_tiles=self.tiles)
            # print('Attempted tile has {} points'.format(len(atile.points)))
        # print('Obtained {} points'.format(len(atile.points)))
        
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
        # print('Extending initial tile')
        canex = True
        while canex:
            self.scratch_points, canex = atile.extend_min_volume(plist=self.scratch_points,
                                                                 avoid_tiles=self.tiles,
                                                                 decision_fun=decision_function)
            # print('Attempted tile has {} points'.format(len(atile.points)))
                        
        # set boundaries of atile and update point boundary masks
        # print('Updating point boundary masks')
        self.set_tile_boundaries(atile)

        # Add atile to tiles in this domain
        # print('Adding tile to domain')
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
            min_gnr = None
            min_gnr_stile = None
            min_gnr_atile_i = None
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
                    if dbool and (not min_gnr or stile.get_geom_norm_resd() < min_gnr):
                        min_gnr = stile.get_geom_norm_resd()
                        min_gnr_stile = stile
                        min_gnr_atile_i = i
                        min_out_spts = out_spts[:]
            if min_gnr_stile:
                canex_tiles = True
                # set boundaries of min_gnr_stile and update point boundary masks
                # print('Updating point boundary masks and replacing atile=>stile')
                self.scratch_points = min_out_spts[:]
                self.tiles.pop(min_gnr_atile_i)
                self.set_tile_boundaries(min_gnr_stile)
                self.tiles.append(min_gnr_stile)
                break
        return canex_tiles

    def do_domain_tiling(self, gnr_thresh=None, tilde_resd_thresh=None,
                         tilde_resd_factor=None):
        # Initialize a list of scratch points for tiling
        self.scratch_points = self.points[:]
        # Clear current tiling
        self.tiles = []
        # Get the decision function
        decision_function = self.tiling_decision_function(gnr_thresh=gnr_thresh,
                                                          tilde_resd_thresh=tilde_resd_thresh,
                                                          tilde_resd_factor=tilde_resd_factor)
        # Tile the domain given the decision function
        try:
            while self.scratch_points:
                self.form_tile(decision_function)
                self.plot_domain_slice(show_tile_id=False)
        except TilingError as terr:
            print(terr.message)
            print('Number of points in attempted tile: {}'.format(
                len(terr.err_tile.points)))
            print('Number of points remaining in domain: {}'.format(
                len(terr.scratch_points)))
            if (terr.err_type == TETypes.few_points_remain or
                terr.err_type == TETypes.cannot_enclose_enough_points):
                # Distribute remaining points among existing tiles
                # decision_function constraint is ignored, so
                # tile overlap is the only constraint.
                canex = True
                while self.scratch_points and canex:
                    print('EXTENDING EXISTING TILES')
                    canex = self.extend_existing_tiles()
                    self.plot_domain_slice(show_tile_id=False)
            else:
                raise

        # Output Results
        self.plot_domain_slice()
        self.print_domain_report()

