# CluMPR_DESI
Python implementation of CluMPR galaxy cluster-finding algorithm, optimized for DESI Legacy Survey (photometric redshifts + stellar masses) and Cori computer (NERSC). As a bonus, I have included a modified version of CluMPR which finds widely-separated gravitationally lensed quasars.

#### Paper decribing the algorithms and cluster+lensed quasar catalogs: The CluMPR Galaxy Cluster-Finding Algorithm and DESI Legacy Survey Galaxy Cluster catalogue (M. J. Yantovski-Barth et al.)

## Directory structure:

Backgrounds: contains code for calculating mass and galaxy count backgrounds (see Appendix A in paper above).

Main algorithm: contains the code of the main cluster-finding algorithm in two files (one for north region of DESI Legacy, one for south region).

Mass Calibration: contains code for calculating the mass correction factor (see Appendix B in paper above).

Miscellaneous: random files which may contain useful code.

Postprocessing: contains code for handling the outputs of the main algorithm and turning them into a good catalog (flagging objects, etc).

Quasars: contains code for finding gravitationally lensed quasar candidates (see Appendix C in paper above).

BEWARE: This code was written by me when I was (mostly) still an undergraduate....it is neither pretty nor efficient. It is done entirely in Jupyter notebooks and is not being actively maintained. However, do not hesitate to reach out to me if you are interested in running this algorithm on your dataset. This code can be a good starting point for your own implementation.

Contact: michael.barth (at) umontreal.ca
