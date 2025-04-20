from numba import jit,njit
import numpy as np 
import math

from bezier import Bezier
from scaffold import *
from polyline import *
from contour import *

""" A field can be thought of as a 2D electric field, i.e. a vector at each point in the plane.              """
""" One field has four different representations, here named pot, field, con, and confield.                  """
""" pot is a potential, field negative its gradient, con the complex conjugate of pot, confield -gradient.   """
""" Later perhaps implement Field object, containing all 4 representations. So far we just have its helpers. """ 

ABX = 300        # Art Board (eg TSHIRT 512x512 or VOORPAND ACHTERPAND 300x800 or 350x800 MOUWPAND or 300x300 VIERKANT, or 400x300 HOLLYPAND)
ABY = 300        # Idem

def set_abx_aby(abxy): 
    """ call this from main file """
    global ABX,ABY
    abx,aby = abxy
    ABX = abx
    ABY = aby

FREE = 1         # No conditions
DIRICHLET = 2    # Dirichlet boundary condition
NEUMANN = 3      # Neumann boundary condition

#BOUNDARY CONDITIONS SETTING ROUTINES: 
def ini_bnd():
    f = np.full((ABY, ABX), FREE)
    return f 


def none_bnd(bnd):
    #clear it after usage
    for j in range(ABY):
        for i in range(ABX):
            bnd[j,i] = FREE


def ini_wmh():
    # wormhole table, Gaußian integers only
    # used for transducers (2 transducers define a dart)
    w = np.full((ABY, ABX), 0 + 0j)
    for j in range(ABY):
        for i in range(ABX):   
            w[j,i] = j*1j + i # default, stay here
    return w 


def c_nearest(conductors,z):
    """ find which conductor (Bezier) is nearest """
    d = ABX + ABY
    aha = conductors[0]
    for c in conductors:
        t = c.t_nearest_raw(z)
        p = c.pixel(t)
        newd = taxicab(z,p)
        if newd <= d: # Try Walrus? thx Guido
            d = newd
            aha = c
    return aha 


def apply_dirichlet(pot,bnd,b : Bezier):
    """ Apply the b.voltage (or voltages) and they will be fixed  """
    """ Please, please stay way from the edge of the field        """
    STEPS = 400 # Was 200
    dt = 1/STEPS
    v0 = b.voltage
    if bis := b.voltage_bis:
        v1 = bis
    else: 
        v1 = v0
    for t in np.arange(-0.05,1.05,dt):
        x,y = z2xy(b.pixel(t))
        for i in range(round(x) - 1, round(x) + 2):     
            for j in range(round(y) - 1, round(y) + 2): 
                pot[j,i] = lerp(v0,v1,t)
                bnd[j,i] = DIRICHLET


def apply_neumann(pot,bnd,b : Bezier):
    """ Bezier b could actually be a linear, work one one side (eg right) """
    dz = b.tangent(.5)
    phi = np.degrees(np.atan2(dz.imag,dz.real))  # phi <= 0 means downward edge
    sx = 1 if phi <= 0 else -3                   #  was 0, -2 # Shift in x direction
    sy = 1 if -90 <= phi and phi <= 90 else -3   # Iwas 0, -2 # dem
    x0,y0 = z2xy(b.pixel(0))
    x1,y1 = z2xy(b.pixel(1))
    pot0 = pot[round(y0),round(x0)]
    pot1 = pot[round(y1),round(x1)]
    STEPS = 1000 # GRRRR, just in case we have a long side (F), leave no gaps
    dt = 1/STEPS
    for t in np.arange(-.05,1.05,dt): 
        x,y = z2xy(b.pixel(t))
        rxrange = range(round(x) + sx, round(x) + 3 + sx)
        for i in rxrange:  
            ryrange = range(round(y) + sy, round(y) + 3 + sy) 
            for j in ryrange:
                pot[j,i] = lerp(pot0,pot1,t) # ????
                bnd[j,i] = NEUMANN


def pairwise(t):  # Thx Jochen Ritzel
    it = iter(t)  # Stackoverflow.com/questions/4628290/pairs-from-single-list
    return zip(it,it)

def apply_boundaries(cnt,pot,bnd,wmh,conductors,insulators,transducers):
    """ DEP pot,cnt MOD bnd,wmh                                                               """
    """ draw the curved lines and impose them as Dirichtlet conditions, insulators as Neumanns"""
    """ no iniPot!!!, proceed with precomputed or previous potential pot                      """
    """ return the XMIN etc. which can be used later to make computations more efficient      """
    """ We need contour cnt to know its name and hack accordingly                             """
    XMIN = ABX; YMIN = ABY
    XMAX = 0;  YMAX = 0
    for c in conductors + insulators + transducers:
    	if c.xmin() < XMIN: XMIN = c.xmin() 
    	if c.ymin() < YMIN: YMIN = c.ymin() 
    	if c.xmax() > XMAX: XMAX = c.xmax() 
    	if c.ymax() > YMAX: YMAX = c.ymax() 
    none_bnd(bnd)
    b : Bezier
    for b in conductors:
        apply_dirichlet(pot,bnd,b) 
        b.draw("solid")
    for b in insulators:
        apply_neumann(pot,bnd,b)   
        b.draw("solid")
 
    return map(round,[XMIN,YMIN,XMAX,YMAX])


# COMPUTING POTENTIALS:
def ini_pot():
    p = np.empty([ABY,ABX])
    for j in range(ABY):
        for i in range(ABX):
            p[j,i] = j / ABY
    return p


@jit("f8[:,:](i8[:,:])", nopython=True, nogil=True)
def ini_con(bnd):
    """ Prep for harmonic conjugate of pot """
    p = np.full((ABY,ABX),-0.1)
    for j in range(ABY):
        for i in range(ABX):
            p[j,i] = 0.5 - i/ABX
    return p


def ini_out(cnt):
    """ out is a memoized version of the is_outside test for contour cnt """
    """ Use in computation of con                                        """
    p = np.full((ABY, ABX), True)                                        
    for j in range(ABY):
        for i in range(ABX):
            z = i + j*1j
            p[j,i] = cnt.is_outside_darts(z)
    return p


@njit
def rasterize(pot):
    LEVELS = 44
    rpot = np.empty((ABY,ABX))
    for i in range(1, ABX):
        for j in range(1, ABY):
            rpot[j,i] = np.floor(LEVELS*pot[j,i]) / LEVELS
    return rpot


@njit
def rasterize_bis(pot):
    """ Shifted by +0.5 so negatives are visible, use for con """
    LEVELS = 36
    SHIFT = 0.5
    rpot = np.empty((ABY,ABX))
    for i in range(1, ABX):
        for j in range(1, ABY):
            rpot[j,i] = SHIFT + np.floor(LEVELS*pot[j,i]) / LEVELS
    return rpot


def precompute_pot(cnt,pot,conductors):
    #MOD pot array contents, we need cnt for its name
    pass


@jit("(f8[:,:], c16)", nopython=True, nogil=True)
def intpol(pot,z : complex) -> float:
    #like pot[y,x], but interpolates in (real-valued) pot array
    x,y = z.real,z.imag
    if x <= 0 or y <= 0 or x >= ABX-1 or y >= ABY-1: return.5
    y0 = math.floor(y)
    y1 = math.ceil(y)
    x0 = math.floor(x)
    x1 = math.ceil(x)
    dy = y - y0
    dx = x - x0
    return lerp(lerp(pot[y0,x0],pot[y1,x0],dy),lerp(pot[y0,x1],pot[y1,x1],dy),dx)


@jit("(f8[:,:], i8[:,:], c16[:,:], b1[:,:], i8, i8, i8, i8, i8)", nopython=True, nogil=True)
def compute_pot(pot, bnd, wmh, out, XMIN, YMIN, XMAX, YMAX, howoften):
    """ Apply Laplace relaxation steps. For the Neumann condition, we used the ideas from: """
    """ github.com/yohanyee/laplace-relaxation/blob/main/laplacerelaxation/relaxation.py   """
    """ out says whether a point is outside of the dart                                    """
    """ the real and imag values in wmh must have been rounded beforehand!!!!!             """
    """ the workhole table has not just boundary points, but also points near the boundary """
    for h in range(howoften):
        for i in range(XMIN,XMAX):
            for j in range(YMIN,YMAX):
                condition = bnd[j,i]
                if condition == FREE:
                    pot[j,i] = (pot[j + 1,i] + pot[j - 1,i] + pot[j,i + 1] + pot[j,i - 1]) / 4
 
                if condition == DIRICHLET:  
                    pass
                if condition == NEUMANN:
                    g = 0.
                    n = 0
                    if bnd[j - 1,i] == FREE:
                        g += pot[j - 1,i]
                        n += 1
                    if bnd[j + 1,i] == FREE:
                        g += pot[j + 1,i]
                        n += 1
                    if bnd[j,i - 1] == FREE:
                        g += pot[j,i - 1]
                        n += 1
                    if bnd[j,i + 1] == FREE:
                        g += pot[j,i + 1]
                        n += 1
                    if n >= 1:
                        pot[j,i] = g / n


#FIELD COMPUTATION, HARMONIC CONJUGATE AND ITS (CON) FIELD:
def ini_field():
    f = np.empty((ABY,ABX), dtype=complex)
    for j in range(ABY):
        for i in range(ABX):
            f[j,i] = -(0.000001 + 0.000001j)  
    return f


def ini_confield():
    f = np.empty((ABY,ABX), dtype=complex)
    for j in range(ABY):
        for i in range(ABX):
            f[j,i] = -(0.00001 + 0.00001j)  
    return f

@njit("c16(c16[:,:],c16)")
def intpol2(field,z: complex) -> complex:
    #like field[y,x], but interpolates in (complex) field array
    x, y = z.real, z.imag
    if x <= 0 or y <= 0 or x >= ABX-1 or y >= ABY-1: return.5
    y0 = math.floor(y)
    y1 = math.ceil(y)
    x0 = math.floor(x)
    x1 = math.ceil(x)
    dy = y - y0
    dx = x - x0
    return lerp2(lerp2(field[y0,x0],field[y1,x0],dy),lerp2(field[y0,x1],field[y1,x1],dy),dx)


def super_intpol2(gap_jumper,field,z: complex) -> complex:
    """ NIET MEER IN GEBRUIK """
    """ Like intpol2, yet different when z is inside a transducer's gap       """
    """ gap_jumper: let the field point accross the gap, or None when outside """
    if gap_jumper(z) == None:
        return intpol2(field,z)
    else: return gap_jumper(z)


@jit(nopython=True, nogil=True) 
def compute_field(pot,field):
    """PRE: pot is sufficiently relaxed, POST: field = -grad pot """
    for j in range(1, ABY - 1):
        for i in range(1, ABX-1):
            re = (pot[j,i + 1] - pot[j,i - 1])/2
            im = (pot[j + 1,i] - pot[j - 1,i])/2
            field[j,i] = -(re + im*1j)


@njit
def compute_confield_old(con,confield):
    """ PRE: con is the harmonic conjugate of pot              """
    """ DEF: just the same formula, POST: confield = -grad con """
    compute_field(con,confield) 

@njit
def compute_confield(field,confield):
    """ New approach (not differentiate con, obtained by !@#$ integration) """
    """ Instead use field and turn that by 90 degrees (PRE: field defined) """
    for j in range(1, ABY - 1):
        for i in range(1, ABX-1):
            fji = field[j,i]
            confield[j,i] = 1j * fji

@jit("(c16[:,:], f8[:,:], i8[:,:], c16[:,:], i8, i8, i8, i8, c16)", nopython=True, nogil=True)
def compute_con(field,con,bnd,wmh,XMIN, YMIN, XMAX, YMAX, greenwich):
    """ MOD con                                                                    """
    """ From given field, prepare con and conbnd by "conjugate" travel             """
    """ Travel along flood paths (constrain flooding in disk of increasing radius) """
    """ But integrate zero when travelling over points inside darts                """
    """ Intuition of conjugate harmonic as longitude (eg fieldlines top to bottom) """

    ready = np.full((ABY,ABX),False) # What has been calculated so far

    x0 = int(greenwich.real)         # Integers needed for addressing array
    y0 = int(greenwich.imag)         # Idem
    con[y0,x0 - 25] = 0              # Arbitrarily, the longitude at Greenwich (-25) is said to be zero
    ready[y0,x0 - 25] = True         # Idem
                                     # First flood the left half, then transduce, then flood the rest
    for radius in range(5,1500,2): 
        change = False               # Set to False, so we can see if change happens
        for x in range(XMIN,x0):     # Do not pass beyond the middle of the darts
            for y in range(YMIN,YMAX):
                if ready[y,x] and euclidean2([x0 - 25,y0],[x,y]) <= radius:
                    f : complex
                    f = field[y,x]
                    ddy = -f.real
                    ddx = f.imag
                    for dx in [-1,+1]:
                        if  not ready[y,x + dx] and bnd[y,x + dx] in [FREE]:
                            con[y,x + dx] = con[y,x] + (dx * ddx )
                            ready[y,x + dx] = True
                            change = True
                    for dy in [-1,+1]:
                        if  not ready[y + dy,x] and bnd[y + dy,x] in [FREE]:
                            con[y + dy,x] = con[y,x] + (dy * ddy)
                            ready[y + dy,x] = True
                            change = True
        if not change: 
            break

    for radius in range(5,1500,1):                # Don't know how to check for change, just do it all:(
        for x in range(XMIN,XMAX):                # How do Forth and back??
            for y in range(YMIN,YMAX):            # Idem
                if ready[y,x] and euclidean2([x0 - 25,y0],[x,y]) <= radius:
                    for dx in [+1,-1]:
                        for dy in [+1,-1]:
                            if  not ready[y + dy,x + dx] and bnd[y + dy,x + dx] in [FREE]:
                                f : complex
                                f = field[y,x]
                                g : complex
                                g = field[y + dy,x + dx]
                                fg = (f + g) /2
                                ddy = -fg.real
                                ddx = fg.imag
                                con[y + dy,x + dx] = con[y,x] + (dx * ddx ) + (dy * ddy)
                                ready[y + dy,x + dx] = True



#FINDING FIELD LINES:

DELTA = .05 # determines tile size, GRRRR
def _find_fieldline_half(cnt,field,conductors,z,flip,long_jump,linetype):
    """ cnt is the contour, we need it for the transfer field in the darts """
    """ we need the conductors: if we approach one, we jump towards it..   """
    """ The jumping will arrive orthogonally at the conductor..            """
    """ which is better: the inpol2(field) is inaccurate near the boundary."""
    """ from x,y till bezier, go from hi to low voltage if flip is -1      """
    """ linetype is either "F" (fieldlines) or E" (equipots)               """
    global ax
    JUMP = 2.5 # was 2.5, typically 3
    MAXJUMP = 8 if long_jump else 3 # Euclidean distance (not voltage)
    c = None
    points = [z]                      # Begin the line's construction
    jumping = True                    # When done, end the line by jumping towards nearest Bézier
    f = cnt.transfield(linetype,z)    # Was f = cnt.transfield_new(estd(cnt.transducers,field, z), z)
    if f is None:
        f = intpol2(field,z)
    for i in range(1500):             # Develop line in at most .. steps (was 1000)
        c = c_nearest(conductors,z)   # Formerly known as b_nearest
        t = c.t_nearest(z)            # Was t_nearest_raw
        p = c.pixel(t)     
        adaptive_maxjump = MAXJUMP if abs(f) > 0.001 else 3 * MAXJUMP # Bah, en dat allemaal voor die eerste lijn van sllr   
        if cnt.is_inside_somedart(z):
            adaptive_maxjump /= 2     # Better crossing the dart than jumping from within
        if abs(p - z) <= adaptive_maxjump:
            break 
        f = cnt.transfield(linetype,z) # As before
        if f is None:
            f = intpol2(field,z) 
        a = abs(f) 
        if a == 0:
            break
        z = z - flip * JUMP * f/a
        points.append(z)

    t = c.t_nearest(z) 
    zt = c.pixel(t) # target on b
    if jumping:
        if long_jump:   # interpolate, enable smooth
            for a in [0.2,0.4,0.6,0.8]:
                zi = lerp2(z,zt,a)
                points.append(zi)
        points.append(zt)
    return points


def find_fieldline(cnt,field,conductors,z,long_jump,linetype):
    """ cnt is the contour, we need it for the transfer field in the darts """
    """ Full line from hi to low voltage                                   """
    def rm_dup(old_list): #remove adjacent duplicates
        prev = None
        new_list = [] 
        for p in old_list:
            if p != prev:
                new_list.append(p)
                prev = p
        return new_list
    up = _find_fieldline_half(cnt,field,conductors,z,1,long_jump,linetype)
    down = _find_fieldline_half(cnt,field,conductors,z,-1,long_jump,linetype) # HERSTEL
    all = up.copy()
    all.reverse()
    all.extend(down)
    return rm_dup(all)


def find_equipotline_OBSOLETE(cnt,con,confield,insulators,z,long_jump):
	#use the field line technology again
    find_fieldLine(cnt,con,confield,insulators,z,long_jump,"E")


def find_fieldlines(cnt,scf,field,conductors):
    lower_yard_left_bis = None  # Some panels do not have this
    lower_yard_right_bis = None # Some panels do not have this

    if cnt.name == "VIERKANT":
        yard, = scf.yards_electrified # GRRR, komma niet vergeten
        lower_yard_left = None
        lower_yard_right = None
        LL_TEST = 2000 # i.e. we all consider them long
    
    verts = [] # ("vertical") field lines:
    if yard:
        for p in yard:
            verts.append(find_fieldline(cnt,field,conductors,p,False,"F"))

    # sort them into longs and two groups of shorts (use constant LL_TEST):                    
    long_lines = [line for line in verts if line[-1].imag < LL_TEST] 
    
    short_lines_left = []
    for v in verts:
        if not v in long_lines:
            short_lines_left.append(v)
        else: break

    short_lines_right = []
    for v in verts:
        if not v in short_lines_left and not v in long_lines:
            short_lines_right.append(v)
    
    short_lines_lower_left = []
    if lower_yard_left: # VOORPAND does not have this
        for p in lower_yard_left: # WAS lower_yard_left[:-1]: # DIE !@#$% electrify_aut is niet goed
            short_lines_lower_left.append(find_fieldline(cnt,field,conductors,p,False,"F"))

    short_lines_lower_left_bis = []
    if lower_yard_left_bis: 
        for p in lower_yard_left_bis: 
            short_lines_lower_left_bis.append(find_fieldline(cnt,field,conductors,p,False,"F"))

    short_lines_lower_right  = []
    if lower_yard_right: # VOORPAND, ACHTERPAND does not have this
        for p in lower_yard_right: 
            short_lines_lower_right.append(find_fieldline(cnt,field,conductors,p,False,"F"))

    short_lines_lower_right_bis  = []
    if lower_yard_right_bis: # VOORPAND, ACHTERPAND does not have this
        for p in lower_yard_right_bis: 
            short_lines_lower_right_bis.append(find_fieldline(cnt,field,conductors,p,False,"F"))

    return short_lines_left, long_lines, short_lines_right, short_lines_lower_left, short_lines_lower_left_bis, short_lines_lower_right, short_lines_lower_right_bis


def equipot_singular(ax,pot,field,con,confield,conductors,insulators,zINI,NH,singularity):
    #let singularity be the index in the mast which gives the most singular equipot
    raw = find_fieldline(cnt,pot,field,conductors,zINI,False,"E")
    mast = electrify(pot,raw,1.0,0.0,NH)
    z = mast[singularity]
    eqp = find_fieldline(cnt,con,confield,insulators+conductors,z,True,"E")

    return eqp

def equipot_singular_vernieuwd(ax,cnt,scf,confield,conductors,singularity):
    #let singularity be the index in the mast which gives the most singular equipot
    mast, = scf.yards_electrified
    eqp = find_fieldline(cnt,confield,conductors,mast[singularity],True,"E")
    return eqp


#POLISHING FIELD LINES AND EQUIPOTLINES:
def insert_nodes(pot,fieldline,nodes):
    """ Each fieldline is a polyline, that is a list of (small-step) points.                """
    """ Usually, some essential points (eg crossing nodes) are not included in the polyline."""
    """ Here, we combine them and sort them by decreasing pot voltage.                      """
    """ for fieldlines use pot, for equipots use con                                        """
    def pot_key(z : complex) -> float:
        return intpol(pot,z)
    return sorted(fieldline + nodes, key=pot_key, reverse=True)


def smooth(e):      #use this for equipots
    FF = 0.66       #filter factor in 0..1
    s =  e[1]-e[0]  #speed vector averaged
    f = [e[0],e[1]] #filtered version of e
                    #higher FF is smoother
    for i in range(2,len(e)):
        s = FF*s + (1 - FF)*(e[i] - f[i-1])
        f.append(f[i-1] + s) 
    f[-1] = e[-1]   #the end goes unfiltered
    return f

def bend_up(line,correction):           # in units e.g. 2
    b = []                              # new bended line
    ll = len(line)                      # for 1st  equipot 
    for i in range(ll):                 # if not    @ neck
        if i == 0: 
            d = (line[3] - line[0]) / 3 # longdistance avg
        else: d = line[i] - line[i-1]
        d /= abs(d) #normalize
        howmuch = 2*abs(i - ll/2) / ll  # 0@middle, 1@ends
        howmuch *= correction           
        howmuch += correction/4         # middle offset
        shift = howmuch * d * 1j        # most bend @ ends
        b.append(line[i] + shift)
    return b


def find_equipots(ax,cnt,scf,confield,conductors,insulators,singularity):
    """" "F" is for fieldlines, "E" for traditional equipots, and "EE" for the epaulet-encircling ïn the sleeve """
    uppers = []
    singls = []
    lowers = []

    if cnt.name in ["VIERKANT"]:
        mast, = scf.masts_electrified # Just one mast
        for i in range(len(mast)): 
            eqp = find_fieldline(cnt,confield,insulators,mast[i],True,"E") # Was False
            uppers.append(eqp)
        return uppers, singls, lowers
