# Finite-Difference Modelling of Wave Propagation in Fractured Media

Implementation of the finite-difference elastic wave solvers from the manuscript:

Turner, E. H. H., Mitchell, T. R., Onederra, I. A., and Leonardi, C. R. "Finite difference modelling of wave propagation in fractured media with imperfect contact interface conditions." Computers and Geotechnics (2027). DOI: [TBC]

## Contents
- `SH-Pseudo2D-2layer.py`: Solves the pseudo-two-dimensional SH elastic wave problem in Section 4.1 of the manuscript.
- `AW-1D-2layer.py`: Solves the one-dimensional acoustic wave problem in Section 4.1, for comparison with the two-layer SH fracture model.
- `PSV-2D-2layer.py`: Solves the two-dimensional P-SV elastic wave problem in Section 4.4 of the manuscript.
- `requirements.txt`: Python dependencies.

## Setup
Tested with Python 3.9.6

    pip install -r requirements.txt

## Citation
If you use this code, please cite the paper above.

## Contact
For further details, please contact Euan Turner at [e.turner@uq.edu.au](mailto:e.turner@uq.edu.au).
