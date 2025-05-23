# HolomorphicMappings
Holomorphic Mappings for Integrated Garment and Motof Design
In: Proceedings of Bridges 2025, forthcoming
by  Loe Feijs and Rong-Hao Liang and Holly Krueger and Marina Toeters
(C) Loe Feijs 2025.
This GitHub repository contains the core of the custom software in Python mentioned in the paper.

# Abstract of the paper
Today's fashion design is based on a separation of concerns such that the geometry of the fabric motif and the geometry of the pattern cut are unrelated. We challenge this approach by developing mathematical tools to morph the motif. The core innovation lies in the mathematical framework, based on holomorphic mappings and harmonic conjugate functions, to map motifs onto arbitrarily shaped panels. This approach, implemented via custom software, allows for seamless motif continuation across complex garment shapes, avoiding cutting through repeating designs. We demonstrate the technique's application through several garment examples, showcasing its potential for creative pattern design and efficient manufacturing. 

# Which libraries are needed?
Python 3.9 (www.python.org) 
Numpy (https://numpy.org) 
Matplotlib (https://matplotlib.org) 
Numba (https://numba.pydata.org/)
NumPy version: 2.0.2,
Numba version: 0.60.0,
Matplotlib version: 3.9.2.

# How to run? 
C:\Users\LFEIJS\AppData\Local\Programs\Python\Python39\python.exe garment.py\
(this is Loe's example; choose your own python folder):

# Where to find more information?
See the Bridges2025 paper (forthcoming)
and the summplement in the online 2025  Bridges archive (forthcoming).
From july 16th, 2025 onward, the paper and the supplement should be on https://archive.bridgesmathart.org/2025

# Limitations
No options for adding anchors or control points are provided, the contour stays four-sided (in the code this is mentioned as "VIERKANT").
Inside the software, one can see preparations for multiple bundles of field lines and equipotential lines.
For "VIERKANT", there is only one bundle needed.
This version only does a four-sided panel,
the six-sided garment, the "extra conductor" sleeve and the equidistant scaffolding are not released (as these were very, very experimental).
Yet, these experiments have made the code extra complex, for example the field lines and equipots are organised in bundles, whereas this code only uses one such bundle.

# User interface
Once the inital contour is shown, use the mouse for dragging the
end points and the handles of the Bézier lines of the contour.
If the contour lines are changed, do not forget to put the ends of the scaffold lines back so they are ON the contour.
Computing potentials etc. should begin after that.

The user interface has 9 buttons, in practice these have
to be operated in the presented order, from top to bottom:
Pot, Fieldlines, Con, Equipots, Grid, Curves, Clear, Motif, and Svg.
Some patience is needed, especially the first button, Pot, takes about one minute or so.

Initially, all buttons, except Pot, are disabled (red when hoover-over).
Once the previous action is completed, the hoover-over turns green,
which means that the button is enabled. After the last button, i.e. Svg, a new file will be produced called "garment.svg", which can be opened by chrome, firefox, Adobe Illustrator or Inkscape.

# Overview of button functions
* Pot: calculate the potential using the contour as boundary condition, also do scaffolding
* Fieldlines: the red lines are what emerges from the steepest descent procedure
* Con: shows the conjugate harmonic (it was already computed  by Pot)
* Equipots: these are equipontial lines, also red
* Grid: now the field lines and the equipot lines are intersected
* Curves: the segments of the red lines are being converted to Béziers, shown in Blue
* Clear: removes the red elements, keeps the blue
* Motif: a motif is put inside each blue grid cell
* Svg: all the motifs are written to a newly created svg file.

# Code highlights

* The polynomial which defines cubic Béziers is _polynom inside bezier.py.
* The predefined contour is defined in vierkant_define inside contour.py.
* The interpolation to get pot values everywhere from a discrete array is in intpol2 inside field.py.
* The relaxation algorithm is in the function compute_pot inside field.py (for the Neumann condition, thanks Yohan Yee, github.com/yohanyee/laplace-relaxation).
* The subdivision of scaffold lines based on potential is electrify_aut in scaffold.py.
* The steepest descent procedure is in _find_fieldline_half in field.py.
* The intersection of a found field line and found equipotential line is intersect in polyline.py (thanks Paul Bourke, paulbourke.net/geometry).
* The turtles which draw the motifs are defined in a class Turtle in turtle.py. 
* The initialisation of turtles is __init__(self,axes,z_ini,morph) where morph is a morphing function.
* The two motifs supported are laurentius_lab and pied-de-poule (blocks 1,b,c, and d) are defined in turtle language inside motif.py. To change the motif, go to lines 515 and 517 in garment.py, where the True and False of the two walrus flags could be interchanged.
* The bilinear interpolation to map motifs into grid cells is grid2morph inside grid.py.

