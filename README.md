# Finite-Difference Modelling of Wave Propagation in Fractured Media

Implementation of the finite-difference elastic wave solvers from the manuscript:

Turner, E. H. H., Mitchell, T. R., Onederra, I. A., and Leonardi, C. R. "Finite difference modelling of wave propagation in fractured media with imperfect contact interface conditions." Computers and Geotechnics (2026). DOI: [TBC]

## Contents
- `Sh-1Dprob-2layer-hfrac.py`: Solves the SH elastic wave model for the two-layer horizontal fracture problem in Section 4.1 of the manuscript.
- `Psv-model-2layer.py`: Solves the two-dimensional P-SV elastic wave model for the problem setup in Section 4.4 of the manuscript.
- `Psv-model-Schoenberg.py`: Solves the same P-SV problem using the Schoenberg and Coates effective-medium model, for comparison with the two-layer fracture model.
- `requirements.txt`: Python dependencies.

## Setup
Tested with Python 3.9.6

    pip install -r requirements.txt

## Citation
If you use this code, please cite the paper above.

## Contact
For further details, please contact Euan Turner at [e.turner@uq.edu.au](mailto:e.turner@uq.edu.au).
