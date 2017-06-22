# Info

Please have a look on the demo.m-file. There you can see how to smooth an image. If you want to change the smoothing, simply change the sigma, N, and fact parameters.. Or let me know if something should be different ;-)

If this is helpful, I will write it that all images are preprocessed and saved as satImage_001_smoothed.png for example.

Paper: http://www.di.ens.fr/~aubry/llf.html

# Readme from the paper-code
This is an implementation of the algorithm of edge-aware detail
manipulation described in the paper:

Fast Local Laplacian Filters: Theory and Applications. 
Mathieu Aubry, Sylvain Paris, Samuel W. Hasinoff, Jan Kautz, and Fredo Durand. 
ACM Transactions on Graphics 2014


The key scripts and functions are:
  llf.m                  - the algorithm of the standard version of the filter described in the paper
  llf_general.m          - the algorithm of the general version of the filter
  demo.m                 - example of smoothing image satImage_001.png

Includes Laplacian pyramid routines adapted from Tom Mertens'
Matlab implementation, modified by Samuel W. Hasinoff.

mathieu.aubry@m4x.org, March 2014
