import math
from numba import jit,njit

""" Polylines, sometimes just called lines are lines built from short linear segments.        """
""" We usually represent them as a list of complex numbers. They appear as fieldlines,        """
""" equipotential lines, scaffolding lines and other auxilliary lines.                        """
""" The module also contains general purpose helpers for complex points, linear interpolation """
""" And a variety of distance functions.                                                      """

# GLOBAL:
plotted = [] #for removal of old lines

# HELPER FUNCTIONS:
def xy2z(x : float,y : float) -> complex:
    return x + y*1j

def z2xy(z : complex) -> tuple[float,float]:
    return z.real, z.imag

@jit("f8(f8, f8, f8)", nopython=True, nogil=True) 
def lerp(lo,hi,alpha):
    return (1 - alpha)*lo + alpha*hi

@jit("c16(c16, c16, f8)", nopython=True, nogil=True)
def lerp2(lo : complex,hi : complex,alpha : float) -> complex:
    # for complex lo, hi, copy of same function in field module
    return (1 - alpha)*lo + alpha*hi

# Manhattan distance function
# Hopefully fast, perhaps use for mouse actions
INFINITY = 1000000
#@jit(nopython=True, nogil=True) 
def taxicab(p : complex,q : complex) -> float:
    if p is None:
        return INFINITY
    else: 
        px, py = p.real, p.imag
        qx, qy = q.real, q.imag
        return abs(px - qx) + abs(py - qy)

@jit(nopython=True, nogil=True) 
def euclidean2(p,q):
    # p a pair
    if p is None:
        return INFINITY
    else: 
        px, py = p
        qx, qy = q
        dx = px - qx
        dy = py - qy
        return math.sqrt(dx*dx + dy*dy)

# HELPERS FOR PLOTTING:
def clear_plotted(ax):
    global plotted
    if  plotted != []:
        for handle in plotted:
            handle.remove()
    plotted = []

def draw_polyline(ax,polyline):
    global plotted
    x_values = []
    y_values = []
    for p in polyline:
        x,y = z2xy(p)
        x_values.append(x)
        y_values.append(y)
        h, = ax.plot(x_values, y_values, 'red', linestyle="solid", linewidth=.15)
        plotted.append(h)

def draw_polyline_orange(ax,polyline):
    # drawing a polyline in orange
    global plotted
    x_values = []
    y_values = []
    for i in range(len(polyline)):
        x,y = z2xy(polyline[i])
        x_values.append(x)
        y_values.append(y)
        h, = ax.plot(x_values, y_values, 'orange', linestyle="solid", linewidth=.3)
        plotted.append(h)

def draw_polyline_green(ax,polyline):
    # drawing a polyline in green
    global plotted
    x_values = []
    y_values = []
    for i in range(len(polyline)):
        x,y = z2xy(polyline[i])
        x_values.append(x)
        y_values.append(y)
        h, = ax.plot(x_values, y_values, 'green', linestyle="solid", linewidth=.3)
        plotted.append(h)

def draw_points_black(ax,polyline,big):
    # use for resampled scaffold lines
    global plotted
    x_values = []
    y_values = []
    for i in range(len(polyline)):
        x,y = z2xy(polyline[i])
        ms = 3 if big else 1
        h, = ax.plot(x,y,marker='o', markersize=ms, color='black', markerfacecolor="None", linewidth=.1)
        plotted.append(h)

def draw_points_red(ax,polyline,big):
    # use for resampled scaffold lines with big True
    global plotted
    x_values = []
    y_values = []
    for i in range(len(polyline)):
        x,y = z2xy(polyline[i])
        ms = 3 if big else 1
        h, = ax.plot(x,y,marker='o', markersize=ms, color='red', markerfacecolor="None", linewidth=.1)
        plotted.append(h)

def draw_point(ax,point,big):
    global plotted
    x,y = z2xy(point)
    x_values = [x]
    y_values = [y]
    ms = 3 if big else 1
    h, = ax.plot(x,y,marker='o', markersize=ms, color='red', markerfacecolor="None", linewidth=.1)
    plotted.append(h)

def draw_point_orange(ax,point,big):
    global plotted
    x,y = z2xy(point)
    x_values = [x]
    y_values = [y]
    ms = 3 if big else 1
    h, = ax.plot(x,y,marker='o', markersize=ms, color='orange', markerfacecolor="None", linewidth=.1)
    plotted.append(h)

def lengths_cumulative(line):
    """ auxiliary for marinapand distance calibrations """
    sums = [0]
    for i in range(len(line) - 1):
        sums.append(abs(line[i + 1] - line[i]))
    return sums


# HELPERS FOR INTERSECTING POLYLINES
# adapted from P. Bourke's collection

@njit
def nearest_point_on_line(p1,p2,p3):
    """ Minimum Distance between a Point and a Line, Written by Paul Bourke, October 1988  """ 
    """ https://paulbourke.net/geometry/pointlineplane/ (p1 p2 define the line)            """
    """ Tested: p1 = 1+1j; p2 = 1 + 100j; p3 = 2 + 2j; z = nearest_point_on_line(p1,p2,p3) """
    """ Result: (1+2j) (correct).                                                          """
    x1, y1 = p1.real, p1.imag
    x2, y2 = p2.real, p2.imag
    x3, y3 = p3.real, p3.imag
    denominator = abs(p1 - p2) ** 2
    u = ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1)) / denominator
    x = x1 + u * (x2 - x1)
    y = y1 + u * (y2 - y1)
    return x + y*1j

@njit
def intersectQ(p1 : complex,p2 : complex,p3 : complex,p4 : complex) -> bool:
    #local.wasp.uwa.edu.au/~pbourke/geometry/lineline2d/
    #true if segment edge p1..p2 intersects segment edge p3..p4
    #previously: if segment edge (xy1-xy2) intersects segment xy3-xy4 (s)
    x1, y1 = p1.real, p1.imag 
    x2, y2 = p2.real, p2.imag 
    x3, y3 = p3.real, p3.imag 
    x4, y4 = p4.real, p4.imag
    d =  ((y4 - y3)*(x2 - x1) - (x4 - x3)*(y2 - y1))     # d is "denominator"
    if d == 0: return False                              # Do not divide by 0
    f1 = ((x4 - x3)*(y1 - y3) - (y4 - y3)*(x1 - x3)) / d # n_a
    f2 = ((x2 - x1)*(y1 - y3) - (y2 - y1)*(x1 - x3)) / d # n_b
    return 0 <= f1 and f1 <= 1 and 0 <= f2 and f2 <= 1   # Was  mixed < and <=

@njit
def _intersection_point(p1 : complex,p2 : complex,p3 : complex,p4 : complex) -> complex:
    x1, y1 = p1.real, p1.imag 
    x2, y2 = p2.real, p2.imag 
    x3, y3 = p3.real, p3.imag 
    x4, y4 = p4.real, p4.imag
    d =  ((y4 - y3)*(x2 - x1) - (x4 - x3)*(y2 - y1))  #"denominator"
    if d == 0:
        return None
    f1 = ((x4 - x3)*(y1 - y3) - (y4 - y3)*(x1 - x3)) / d #n_a
    f2 = ((x2 - x1)*(y1 - y3) - (y2 - y1)*(x1 - x3)) / d #n_b
    isects = 0 <= f1 and f1 <= 1 and  0 <= f2 and f2 <= 1 
    re = x1 + f1*(x2 - x1)
    im = y1 + f1*(y2 - y1)
    return re + im*1j if isects else None
 
# HERSTEL @njit
def _intersection_point_l(p1 : complex,p2 : complex,line : list[complex]) -> complex:
    #let s is the segment from point p1 to p2
    #let s0,s1,s... be the segments in between the points in line
    #does s cross one of s0,s1,..?
    segments = []
    for i in range(len(line) - 1):
        segments.append([line[i],line[i+1]])
    for s in segments:
         q1,q2 = s
         p = _intersection_point(p1,p2,q1,q2) 
         if not p is None:
             return p
    else: return None

# HERSTEL @njit
def intersect(line1 : list[complex],line2 : list[complex]) -> complex:
    segments1 = []
    for i in range(len(line1) - 1):
        segments1.append([line1[i],line1[i+1]])
    for s in segments1:
         p1,p2 = s
         p = _intersection_point_l(p1,p2,line2) 
         if not p is None:
             if math.isnan(p.real) or math.isnan(p.imag): print("OOPS, nan from intersect(line1 : list[complex],line2 : list[complex]) -> complex:")
             return p
    return None
  