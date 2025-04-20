from numba import njit
from math import sqrt,pi
from cmath import phase, rect

from scaffold import Scaffold
from polyline import lerp,intersectQ
from bezier import Bezier


""" A Contour objects describes a panel of fabric to be cut. Seamallowace is to be added later.     """ 
""" A combination of a few contours describes a garment (not implemented for now).                  """
""" There are a number of segments araound the contour, always denoted clockwise, eg. neck, rshld.  """
""" The names of the segments depend on the contour's name ("VIERKANT", ..).                        """
""" To generate morphed motifs we shall generate so-called fieldlines, borrowing terrminology from 2D electrostatics. """
""" We also borrow theory from complex function theory, e.g. conductors are treated as Dirichlet boundary conditions. """
""" A pair of transducers, adjacent in the list, implements a dart (Dutch: figuurnaad). They must be equally long and """
""" and be equally oriented, and have one point in common, which we call the center of the transducer pair.           """
""" Inside the dart a special field shall be generated which goes straight from the first to the second transducer.   """
""" Darts and transducers not needed for "VIERKANT"                                                                   """


@njit
def aux_is_outside(long_line,inner_point, z):
    """ Auxilliary, isolated from main program to allow for njit, slow algorithm:( """
    n = 0
    for i in range(len(long_line) - 1):
        if intersectQ(long_line[i],long_line[i + 1],inner_point, z):
            n += 1
    return (n % 2) == 1

# HELPERS FOR VIERKANT:

def vierkant_define(ax):
    A = Bezier(ax, [ 50 + 250j, 120 + 275j, 180 + 275j, 250 + 250j]) # convex, arc length is 204.5 
    B = Bezier(ax, [250 + 250j, 225 + 180j, 225 + 120j, 250 +  50j]) # concave, arc length is 204.5 
    C = Bezier(ax, [250 +  50j, 180 +  25j, 120 +  25j,  50 +  50j]) # convex, arc length is 204.5 
    D = Bezier(ax, [ 50 +  50j,  75 + 120j,  75 + 180j,  50 + 250j]) # concave, arc is 204.5 

    conductors  = [A,C]       
    insulators  = [B,D]  
    transducers = []   
    return (conductors,insulators,transducers)

def vierkant_electrify(A,C): 
    A.voltage = 1.0 # The fieldlines go from to to bottom
    C.voltage = 0.0 # The fieldlines go from to to bottom

def vierkant_scaffold(ax): 
    def lerp(lo,hi,alpha): 
        return (1 - alpha)*lo + alpha*hi
    def controls(m1,m2):
        return [m1, lerp(m1,m2,.3), lerp(m1,m2,.7), m2]

    m1 = 150 + 268.5j  # On the edge, mast goes downward
    m2 = 150 +  31.5j  # Adjust so it is on A and C
    mast = Bezier(ax, controls(m1,m2))

    y1 =  68.8 + 150j   # In the middle
    y2 = 231.2 + 150j   
    yard = Bezier(ax, controls(y1,y2)) 

    return Scaffold(ax, "VIERKANT", [mast], [yard]) # TODO: 

def vierkant_is_outside(conductors,insulators,transducers,z):
    A,C = conductors
    B,D = insulators # No transducers
    top = A.pixel(.5)
    down = C.pixel(.5)
    TINYSHIFT = 17 + 23j # Avoid intersecting at contour corners
    inner_point = (top + down) / 2 + TINYSHIFT
    segments = [A,B,C,D] # TZT J1,J2 erbij
    long_line = A.line() + B.line() + C.line() + D.line()
    return aux_is_outside(long_line,inner_point,z)


# HELPERS FOR WORKING WITH TRANSDUCERS (DARTS):

# A dart consist of a straight bezier and a next straight bezier of equal length
# Such that the  a2 control point of both co-incide, usually the lines are transducers
# This version of the code does not include darts.

@njit
def heron(a : float,b : float, c : float) -> float:
    """ Heron's theorem for area of triangle with sides a,b,c """
    """ https://en.wikipedia.org/wiki/Heron%27s_formula       """
    """ s is called the semiperimeter, half the circumference """
    s = (a + b + c)/2 
    return sqrt(s*(s - a)*(s - b)*(s - c))

@njit
def is_outside_dart(a1,a2,a1_next,a2_next, z):
    """ Boolean which gives true if z is outside the dart defined by the a1,a2,a1_nex,a2_next"""
    assert a2 == a2_next                                  # where the transducers meet
    a3 = a1_next                                          # a1,a2,a3 is the dart's triangle
    l1 = abs(a1 - a3)                                     # length of its "short" side
    l2 = abs(a1 - a2)                                     # length of transducer
    l3 = abs(a3 - a2)                                     # length of next
    a1a2a3 = heron(l1,l2,l3)                              # area of the dart's triangle
    za1a2 = heron(abs(z - a1),abs(z - a2),abs(a1 - a2))   # another triangle, involving z
    za2a3 = heron(abs(z - a2),abs(z - a3),abs(a2 - a3))   # idem 
    za1a3 = heron(abs(z - a1),abs(z - a3),abs(a1 - a3))   # idem
    sum3  = za1a2 + za2a3 + za1a3                         # if z inside then this sum equals a1a2a3 
    MARGIN = 0.0001                                       # Do not remove this - been there :(
    test = (sum3 > a1a2a3 + MARGIN)                       # old trick, avoids case-by-case analysis, see                        
    return test                                           # www.jeffreythompson.org/


@njit
def is_inside_dart(a1,a2,a1_next,a2_next, z): 
    b = is_outside_dart(a1,a2,a1_next,a2_next, z)                  
    return not b


@njit
def wormjump(a1,a2,a1_next,a2_next, z):
    """ Jump across the wormhole, should work in a dart line's neighbourhood      """
    """ Note, it does not flip, it rather is a rotation                           """
    """ From a point near the first side it moves to somewhere near the next side """
    """ It is meant for wormholing neighbours during the relaxation of pot        """
    """ The dartlines are supposed to be equally long                             """
    if not (a2 == a2_next):
        print("a2,a2_next ",a2,a2_next)   
    assert(a2 == a2_next)                             # Where the lines meet
    theta = phase(a1_next - a2_next) - phase(a1 - a2) # Compare the angles of the two lines
    newz =  a2  + (z - a2) * rect(1.,theta)           # Rotate counter clockwise over theta ()
    return newz


@njit
def wormjump_retour(a1,a2,a1_next,a2_next, z):
    # from the "next" line to the first
    assert(a2 == a2_next)                             # Where the lines meet
    theta = phase(a1_next - a2_next) - phase(a1 - a2) # Compare the angles of the two lines
    minus_theta = -theta                              # This is opposite varaint
    newz = a2  + (z - a2) * rect(1.,minus_theta)      # Rotate in the other direction
    return newz


@njit
def cross_the_dart(a1,a2,a1_next,a2_next, z):
    """ Transducers must be straight lines, not curved. Aux for transfield           """
    """ Test whether z is inside the triangle of the dart (otherwise return None)    """
    """ The transfer_field in the direction from the first to the second transducer. """
    """ Do not store in pixellated field array, use directly (for more precision).   """
    if is_outside_dart(a1,a2,a1_next,a2_next, z):
        return None                                 
    else:
        a3 = a1_next                                # a1,a2,a3 is the dart's triangle
        mid = (a1 + a3)/2                           # Middle point of the short side
        center = a2                                 # center is where the transducers meet
        biss = center - mid                         # Bissector (middle line)           
        goto = 1j * biss                            # Turn 90 degrees counter clockwise
        goto = goto / abs(goto)                     # Field of unit size
        return goto                                 # E.g. go from point on G1 toward point on G6

#CLASS:

class Contour():
    """wrapper class for conductors, insulators, transducers around and inside a panel"""
    def __init__(self,ax,name):
        self._ax = ax # as Bezier needs that
        self.name = name
        self.conductors = []  # To be filled during define
        self.insulators = []  # Idem
        self.transducers = [] # Idem
        self.vbeams = []      # To be filled during propose_scaffold
        self.hbeams = []
        #### self.central_darts_point = 119.2 + 509.7*1j # Hack for VOORPAND
        if not name in ["VIERKANT"]:
            print("OOPS, unknown contour type")

    def define(self):
        if self.name == "VIERKANT":
            conductors, insulators, transducers = vierkant_define(self._ax)
            self.conductors = conductors
            self.insulators = insulators
            self.transducers = transducers
            for b in conductors + insulators + transducers:
                b.permanent = True
 
    def electrify(self):
        if self.name == "VIERKANT":
            A,C = self.conductors   
            vierkant_electrify(A,C)


    def scaffold(self):
        if self.name == "VIERKANT":
            return vierkant_scaffold(self._ax)

    def is_outside_darts(self, z):
        """ Boolean test whether z is outside ALL darts """
        it = iter(self.transducers) # thx Jochen Ritzel
        tst = True
        for tt in zip(it,it):       # stackoverflow.com/questions/4628290/pairs-from-single-list
            t,t_next = tt           # Each successive pair of transducers defines a dart
            a1,_,_,a2 = t.controls                             # unpack just what we need
            a1_next,_,_,a2_next = t_next.controls              # controls as in a Bezier object
            tst = tst and is_outside_dart(a1, a2, a1_next, a2_next, z)
        return tst

    def is_inside_somedart(self, z):
        """ Boolean test whether z is outside ALL darts """
        it = iter(self.transducers) # thx Jochen Ritzel
        for tt in zip(it,it):       # stackoverflow.com/questions/4628290/pairs-from-single-list
            t,t_next = tt           # Each successive pair of transducers defines a dart
            a1,_,_,a2 = t.controls                             # unpack just what we need
            a1_next,_,_,a2_next = t_next.controls              # controls as in a Bezier object
            if is_inside_dart(a1, a2, a1_next, a2_next, z):
                return True
        else: return False

    def transfield_dummy(self,z):
        # just go rightward
        if self.is_outside_darts(z):
            return None
        else: return 1 + 0j

    def transfield(self,linetype,z):
        """ The result of transfield is a field vector which can be used for going across the dart      """
        """ For equipots (linetype is "E") we flip the 2nd dart, GRRR adhoc                             """
        """ BLIJ: HIER KUN JE ALLE LIJNEN PATCHEN DIE NIET DOOR DE DART WILLEN                          """
        return None

    def is_outside(self, z):
        if self.name == "VIERKANT":
            return vierkant_is_outside(self.conductors,self.insulators,self.transducers, z)
