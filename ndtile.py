"""
Do Tiling for an N-dimensional data set given an input CSV file
containing one point per row. Each point is specified by a set 
of independent parameter values followed by the dependent scalar value.

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
import numpy as np
import argparse
from Tiling import Point, Domain

parser = argparse.ArgumentParser()
parser.add_argument('infile', type=str,
                    help='Name of the input csv file containing (x1, x2, ..., y) data series for scalar data of the form y=f(x1, x2, ...). One line of header will be skipped.')
parser.add_argument('-L2rt', '--L2resthresh', type=float, 
                    help='Upper threshold for tiling constraint: L-2 norm of normalized residuals.')
parser.add_argument('-cfdt', '--cdetthresh', type=float, 
                    help='Lower threshold for tiling constraint: coefficient of determination.')
parser.add_argument('-tsym', '--tilesymmetry', type=float, 
                    help='Threshold on normalized residual symmetry across a tile.')
parser.add_argument('-fsym', '--factortilesymmetry', type=float, 
                    help='Threshold on growth factor for normalized residual symmetry across a tile.')
parser.add_argument('-ptsf', '--plotsurfaces', action='store_true',
                    help='If supplied, plot tile surfaces when searching for empty space to create virtual tiles.')
parser.add_argument('-pint', '--plotintermediate', action='store_true',
                    help='If supplied, plot the domain at intermediate steps whenever a new point is added to a tile.')
parser.add_argument('-ptil', '--plottiling', action='store_true',
                    help='If supplied, plot the domain whenever a new tile is added to the domain.')
parser.add_argument('-pfin', '--plotfinal', action='store_true',
                    help='If supplied, plot the domain when tiling is complete.')
parser.add_argument('-dlab', '--dimlabels', type=str, nargs='*', help='If supplied, will collect a series of strings specifying, in order, the label for each dimension. If the number of dimension labels is not exactly equal to the dimensionality of the dataset, then the supplied labels will be ignored.')
parser.add_argument('-ilab', '--independentlabel', type=str, help='Takes a string argument to set the label for the independent scalar value corresponding to this dataset.')
parser.add_argument('-noshrink', '--noshrink', action='store_true',
                    help='If supplied, virtual tiles containing empty space will not be shrunk after point tiling.')
parser.add_argument('-log', '--logfile', type=str,
                    help='Name of the log file in which to write the status of intermediate steps. If --logfile is not supplied, no intermediate printing will be done.')
parser.add_argument('-o', '--outfile', type=str,
                    help='Name of the summary file in which to print the final tiling result.')
args = parser.parse_args()

# Read Data
raw_data = np.genfromtxt(args.infile, delimiter=',', skip_header=1)
# Each element of data is a row from the csv file, so convert to columns
data = np.transpose(raw_data)
# data[0:-1] = Independent Parameter values
pt_ivals = np.transpose(data[0:-1])
# data[-1] = Scalar Dependent Parameter values
pt_dvals = data[-1]

# Create list of Points
pointlist = []
for r, v in zip(pt_ivals, pt_dvals):
    p = Point(r, v)
    pointlist.append(p)

# Get bounds on the independent parameter domain
lo = np.amin(pt_ivals, axis=0)
hi = np.amax(pt_ivals, axis=0)

# Form Domain
dom = Domain(points=pointlist, lo=lo, hi=hi,
             dlabels=args.dimlabels, ilabel=args.independentlabel,
             logfile=args.logfile, summaryfile=args.outfile)

# Tile Domain
dom.do_domain_tiling(L2r_thresh=args.L2resthresh, coeff_det_thresh=args.cdetthresh, tilde_resd_thresh=args.tilesymmetry,
                     tilde_resd_factor=args.factortilesymmetry, attempt_virtual_shrink=(not args.noshrink),
                     plot_tile_surfaces=args.plotsurfaces, plot_intermediate=args.plotintermediate,
                     plot_tiling=args.plottiling, plot_final=args.plotfinal)

# Cleanup, closing open file handles
dom.close()
