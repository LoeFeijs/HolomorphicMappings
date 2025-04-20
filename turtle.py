from math import sin,cos,radians,isnan
import matplotlib.pyplot as plt
from numba import jit,njit
import numpy as np

from field import ABX,ABY
from bezier import Bezier

#GLOBALS:
paths = [] #accumulated strings for svg file by all turtles

#HELPERS:  
def get_paths():
    global paths
    return paths


def prelude():
    x, y = str(ABX), str(ABY) 
    xmm, ymm = x + "mm", y + "mm"
    return ("<svg version=\"1.1\"\n "
            "width=\"" + xmm + "\" "
            "height=\"" + ymm + "\"\n "
            "viewbox=\"0 0 " + x + " " + y + "\" "
            "xmlns=\"http://www.w3.org/2000/svg\">\n" 
           )


def postlude():
    return "</svg>"


@jit("c16(c16,c16,c16,c16,f8)", nopython=True, nogil=True)
def _bezier(a1,c1,c2,a2, t : float) -> complex:
    #parameter t, 0<= t <=1
    return a1*(1-t)*(1-t)*(1-t) + c1*3*(1-t)*(1-t)*t + c2*3*(1-t)*t*t + a2*t*t*t  


def scale(units):
    FACTOR = 2.8349944627 # experimental via adobe illustrator
    return(FACTOR * units)


def mk_path(d : str, color : str):
    # eg   <path d="M100,200 C100,100 250,100 250,200 S400,300 400,200" />
    return "<path fill=\"black\" stroke=\"" + color + "\" stroke-width=\".1\" d=" + d + "/>\n"


def mk_path_nofill(d : str, color : str):
    # eg   <path d="M100,200 C100,100 250,100 250,200 S400,300 400,200" />
    return "<path fill=\"none\" stroke=\"" + color + "\" stroke-width=\".1\" d=" + d + "/>\n"


def mk_m(a1x,a1y):
    m = "M"
    m += format(scale(a1x),".2f") + "," + format(scale(a1y),".2f") + " "
    return m


def mk_c3(c1x,c1y,c2x,c2y,a2x,a2y):
    c3 = "C"
    c3 += format(scale(c1x),".2f") + "," + format(scale(c1y),".2f") + " "
    c3 += format(scale(c2x),".2f") + "," + format(scale(c2y),".2f") + " "
    c3 += format(scale(a2x),".2f") + "," + format(scale(a2y),".2f") + " "
    return c3


def z2xy(z):
    return z.real, z.imag


#CLASS DEFINITION:
class Turtle(object):
    #note, we work in the complex plane
    #don't forget penUp when done (otherwise nothing is shown)
    #note the holomorphic function parameter morph: complex -> complex
    #use only for the motif, not for fieldlines or bezierified versions thereof
    def __init__(self,axes,z_ini,morph):
        self._ax = axes            # for matplotlib
        self._x = z_ini.real       # unmorphed turtle space
        self._y = z_ini.imag       # idem
        self._morph = morph        # holomorphic mapping function
        self._heading = 0          # degrees
        self._filling = False       # screen only, not svg
        self._fillcolor = "pink"   # screen only, not svg
        self._plotcolor = "purple" # idem
        x4plot, y4plot = z2xy(morph(z_ini)) # a1, then successive c1,c2,a2 points
        self._x_points = [x4plot]           # this is an a1 (already morphed)
        self._y_points = [y4plot]           # later add triples c1,c2,a3

    def forward(self,amount):
        # seems straight, yet prepared for bezier morphing
        dx = cos(radians(self._heading)) * amount
        dy = sin(radians(self._heading)) * amount
        dx /= 3 # store line as bezier curve 
        dy /= 3 # having in-line control points
        for i in [1,2,3]:
            self._x += dx
            self._y += dy
            x4plot, y4plot = z2xy(self._morph(self._x + self._y*1j))
            self._x_points.append(x4plot) # the x_point list contains morphed points
            self._y_points.append(y4plot) # although forward goes straight, beziers needed for render in view of morph

    def curveto(self,c1,c2,a2):
        # absolute coordinates, do not mix with forward and left right commands
        for point in [c1,c2,a2]:
            x4plot, y4plot = z2xy(point)
            self._x_points.append(x4plot) # x_points list contains morphed points
            self._y_points.append(y4plot) # y_points idem 
        self._x, self._y = z2xy(a2)

    def right(self,amount):
        self._heading -= amount

    def left(self,amount):
        self._heading += amount

    def pen_down(self):
        x4plot, y4plot = z2xy(self._morph(self._x + self._y*1j))
        self._x_points = [x4plot]
        self._y_points = [y4plot]

    def pen_up(self):
        global paths
        # first a rude screen view (just hop through points)
        if self._filling:
            self._ax.fill(self._x_points,self._y_points,self._fillcolor,linewidth=.3)
        self._ax.plot(self._x_points,self._y_points,self._plotcolor,linewidth=.3)
        PRINTING = True
        x = self._x_points[0]
        y = ABY - self._y_points[0] 
        if PRINTING and not x is None and not isnan(x):
            d = mk_m(x,y) 
            i = 1
            while i < len(self._x_points):
                c1x = self._x_points[i]
                c1y = ABY - self._y_points[i]     # svg works upside down
                c2x = self._x_points[i + 1]
                c2y = ABY - self._y_points[i + 1] #idem
                a2x = self._x_points[i + 2]
                a2y = ABY - self._y_points[i + 2] # idem
                i += 3
                d += mk_c3(c1x,c1y,c2x,c2y,a2x,a2y)
            if self._filling:
                path = mk_path("\"" + d + "\"",self._plotcolor) 
            else: 
                path = mk_path_nofill("\"" + d + "\"",self._plotcolor)
            paths.append(path) # accumulate in global paths for svg
        self._xPoints = []
        self._yPoints = []

    def pd(self):
        self.pen_down()

    def pu(self):
        self.pen_up()
