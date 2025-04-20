from numba import njit
from polyline import *
import numpy as np

# A Scaffold is a set of horizontal and vertical beziers to be used as launching points """
# for field lines and equipotlines. Each contour will have its own scaffold.          """   

# HELPERS:

@njit("f8(f8[:,:], c16)")
def _intpol(pot,z : complex) -> float:
    #like pot[y,x], but interpolates in (real-valued) pot array
    #WARNING: this local version is UNSAFE near artboard boundar!!
    x, y = z.real, z.imag
    y0 = math.floor(y)
    y1 = math.ceil(y)
    x0 = math.floor(x)
    x1 = math.ceil(x)
    dy = y - y0
    dx = x - x0
    return lerp(lerp(pot[y0,x0],pot[y1,x0],dy),lerp(pot[y0,x1],pot[y1,x1],dy),dx)


#@njit("c16(f8[:,:], c16, c16, f8)") 
def _bin_search(pot,p0,p1,v):
    #aux to findPoint4Volt, find a point on the straight line segment p0-p1, where voltage is decreasing
    #see medium.com/dscjssstu/generalising-binary-search-b58f4bf4e176
    EPSILON = 0.001 # the precision to which we want the v to be approximated
    SAFETY = 25 #prevent infinite loop
    left = 0
    right = 1
    cnt = 0
    while right - left > EPSILON and cnt < SAFETY:
        cnt += 1
        mid = left + (right - left)/2
        p = lerp2(p0,p1,mid)
        if _intpol(pot,p) <= v:
            right = mid
        else:
            left = mid
    return lerp2(p0,p1,left + (right - left)/2)


#@njit("c16(f8[:,:], c16[:], f8)")
def find_point_by_volt(pot,line,v):
    #PRE: 0<=v<=1 and pot along line must be monotonically decreasing from above to below or equal v
    #POST result is a position whose pot is close to v. Locate the appropriate segment first. 
    prev = 0
    aha = 0
    for i in range(len(line)):
        z = line[i]
        if _intpol(pot,z) <= v: 
            prev = i-1
            aha = i
            break
    if prev == -1:
        return line[0]
    p = _bin_search(pot,line[prev],line[aha],v)
    return p 


def electrify(pot,rawline,fromV,toV,N):
    """ PRE: voltage along line decreasing from fromV to toV eg 1 to 0.5 """
    """ N segments between N+1 points, so we distribute the points       """
    p = rawline[0]
    line = [p]
    if N == 0:
        print("N is 0 and fromV,toV,rawline is ",fromV,toV,rawline)
    dv = (fromV - toV) / N
    for i in range(1,N):
        p = find_point_by_volt(pot,rawline,fromV - i*dv)
        line.append(p)
    line.append(rawline[-1]) # Last
    return line   


def electrify_linear(rawline,N):
    """ Just deliver a straight line and divide it in N segments """
    """ Forget about pot, also forget any curving in rawline     """
    z_ini = rawline[0]
    z_end = rawline[-1]
    line = []
    d = 1.0 / N
    alpha : float
    for alpha in np.arange(0, 1 + d, d):
        z = lerp2(z_ini, z_end, alpha)
        line.append(z)
    return line

@njit
def electrify_equidistant(rawline,N):
    """ Follow a curved polyline and divide it in N segments                     """
    """ Forget about pot, but it still is a curve. Typical N between 10 and 100  """
    """ Aim at accuracy of .1, assume each step is at most of length 5           """
    """ The newline steplength is thus at most .05. Naive and inefficient, sorry """
    """ This is used for flat yet curved seams (Holly's idea) although it goes against the holomorphic function theory """
    D = 0.05
    def refine(raw):
        z = raw[0]
        new = [z]
        for i in range(len(raw) - 1):
            for alpha in np.arange(D, 1 + D, D):
                z = lerp2(raw[i],raw[i + 1],alpha)
                new.append(z)
        return new
    def sumlen(line):
        s = 0
        for i in range(len(line) - 1):
            s += abs(line[i + 1] - line[i])
        return s
    def find_approximate(refined,s):      # Find a list where s is the desired sumlength
        for i in range(len(refined) + 1): # Try all prefix lists of refined
            sl = sumlen(refined[0:i])     # Length of the prefix upto i
            if abs(sl - s) < 1.5*D:       # Better not skip a candidate
                return refined[i]         # Done when we have something close
        else: print("OOPS, find_approximate in electrify_equidistant failed")
    refined = refine(rawline)
    total = sumlen(refined)
    step = total / N
    e = []
    for s in np.arange(0, total, step):
        z = find_approximate(refined,s)
        e.append(z)
    e.append(rawline[-1])
    return e # TODO: TEST IT

def electrify_aut(pot,rawline,N):
    # automatic finding of fromV and toV, tricky repairs.
    # not for verticals, but for (almost) horizontal line serving as yard.
    # ?? works for TSHIRT, not elsewhere
    def monotonify(pot,line):
        def m1(pot,line,op):               # The m1 is the essential monotonification step
            z0 = line[0]                   # Tip for debugging: it crashes when the voltage is reversed (better flip yard)
            z1 = line[1]
            v0 = _intpol(pot,z0)
            v1 = _intpol(pot,z1)
            if op(v0,v1): 
                return m1(pot,line[1:],op) # Skip any out-of-order v0
            else: return line              # Note the recursion
        def less(x,y): return x < y
        def more(x,y): return x > y
        new = m1(pot,line,less)            # Work from left to right
        new.reverse()                      # Prepare for right to left sweep
        newnew = m1(pot,new,more)          # Do it again
        newnew.reverse()
        return newnew
    def hilo(pot,line):
        hi = _intpol(pot,line[1])          # Was [0], better not use very first
        lo = _intpol(pot,line[-2])         # Was [-1], idem very last
        return hi,lo
    pFirst = rawline[0]                    # Good positions but..
    pLast  = rawline[-1]                   # Probably bad voltages
    rawline = monotonify(pot,rawline)
    hi,lo = hilo(pot,rawline)              # Good voltages
    extra_lo = 0.015                       # Was 0.015 (pray)
    extra_hi = 0.025                       # Idem
    fhi = lerp(hi,lo,0 - extra_lo)        # Extrapolate the voltage
    flo = lerp(hi,lo,1 + extra_hi)        # At both ends
    line = electrify(pot,rawline,fhi,flo,N)
    line[0] = pFirst                       # Put it back
    line[-1] = pLast                       # Idem
    return line


def electrify_aut_bis(pot,rawline,N):
    # UNTESTED automatic finding of fromV and toV, tricky repairs.
    # not for verticals, but for equipot serving as yard.
    # This  untested version is intended for VOORPAND (wrong con value at edge),
    # where work with the right edge (and don't care about lower_yard_left)
    def hilo(pot,line):
        hi = _intpol(pot,line[0])
        lo = _intpol(pot,line[-1])
        return hi,lo
    pFirst = rawline[0]                        # let's assume this is good
    pLast  = rawline[-1]                       # but this isn't: perhaps bad voltages
    fraction = (N - 1)/N                       # relative length of the supposedly good part
                                               # GRRRR there is much more not good
    upto = round(fraction * len(rawline))      # howfar we go along the line
    goodpart = rawline[0:upto]                 # we do not go till edge
    hi,lo = hilo(pot,goodpart)                 # good voltages
    line = electrify(pot,goodpart,hi,lo,N - 1)  # safe to resample
    line.append(pLast)                         # put the bad guy back
    return line

# CLASS:

class Scaffold(object):
    def __init__(self, axes, name, mast_beziers, yard_beziers): # ship language: mast, yard
        self._ax = axes                                # For matplotlib
        self.name = name                               # eg "TSHIRT"
        self.mast_beziers = mast_beziers               # Bezier for mast
        self.yard_beziers = yard_beziers               # Beziers for various yards
        self.masts   = []                              # Keep polyline versions as well
        self.yards   = []                              # Use _update to keep them in sync                    
        self.masts_electrified = []                    # optional electrified versions too
        self.yards_electrified = []                    # optional electrified versions too
        self.update()                                  # polylines should match the beziers
                          
    def update(self): 
        # call update() whenever the beziers are edited
        b : Bezier
        self.masts = [b.line() for b in self.mast_beziers] 
        self.yards = [b.line() for b in self.yard_beziers]

 
    def draw(self):
        # focus on points (when editing, the bezier will redraw itself)
        self.update()  
        for v in self.masts:
            draw_points_black(self._ax,v,True)
        for h in self.yards:
            draw_points_black(self._ax,h,True)

    def electrify(self,pot,con,NHLIST,NVLIST):
        # Remember, masts are  used to launch ("horizontal") equipots
        # De voltages for this are usually 1.0 to 0.0 V, but are different for the sleeve
        # The yards are used to launch ("vertical") fieldlines
        # for these we have an automatic con-voltage finding mechanism
        # The n values from HLIST or VLIST tell how to subdivide the masts and yards, respectively

        print("scf.electrify")
        self.update()  

        if self.name in ["VIERKANT"]:
            vmaxvmin = [1.0, 0.0]
            voltage_ranges = [vmaxvmin]

        nlist = NHLIST
        self.masts_electrified = []
        assert len(self.masts) == len(nlist) and len(nlist) == len(voltage_ranges)
        for i in range(len(self.masts)):  # No, not Pythonic
            print("ANOTHER MAST")
            v = self.masts[i]             # Raw mast line
            n = nlist[i]                  # Subdivide into ..
            vmax, vmin = voltage_ranges[i]
            if self.name in ["VIERKANT"]:
                r = electrify(pot,v,vmax,vmin,n) 
            self.masts_electrified.append(r)
        
        nlist = NVLIST
        self.yards_electrified = []     
        for h,n in zip(self.yards, nlist): 
            print("ANOTHER YARD")
            if self.name in ["VIERKANT"]:
                r = electrify_aut(con,h,n) 
            self.yards_electrified.append(r)

    def show_points(self):
        for v in self.masts_electrified:
            draw_points_black(self._ax,v,True)
        for h in self.yards_electrified:
            draw_points_black(self._ax,h,True)
