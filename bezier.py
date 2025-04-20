from matplotlib.pyplot import plot, show
from numpy import arange
from math import sqrt
from numba import jit

from polyline import lerp, lerp2, taxicab

#GLOBAL (similar as in grid, yet another variable)
plotted = []

# HELPERS:
@jit(nopython=True, nogil=True)
def xy2z(x : float,y : float) -> complex:
    return x + y*1j

def clear_bezier_lines(ax):
    global plotted
    if  plotted != []:
        for handle in plotted:
            handle.remove()
    plotted = []

# CLASS DEFINITION:
class Bezier(object):
    """Only for arms and hem, do not use for grid curves or fashion motifs"""
    DIST = 10                             # distance for selecting and dragging
    def __init__(self, axes, controls):   # constructor
        self._ax = axes                   # for matplotlib
        self.controls = controls         # [anchor,control,control,anchor] Bézier
        self._dragged = None              # otherwise an index in controls
        self._plotted = []                # for removal when plotting new point or line
        self._INFINITY = 1000000
        self.voltage = 0.0                # voltage for entire line (or at t = 0 for variable voltage)
        self.voltage_bis = None           # voltage at t = 1 (for variable voltage only, None otherwise)
        self._cubic_or_linear = "cubic" if not controls[1] is None and not controls[2] is None else "linear"
        self._filling = False
        self._fillcolor = "Black"
        self._linestyle = "solid"

    def xmin(self):#for saving computations
        a1, c1, c2, a2 = self.controls[0:4]
        if c1 == None or c2 == None:
            return min(a1.real, a2.real)
        else: return min(a1.real, c1.real, c2.real, a2.real)

    def ymin(self):
        a1, c1, c2, a2 = self.controls[0:4]
        if c1 == None or c2 == None:
            return min(a1.imag, a2.imag)
        else: return min(a1.imag, c1.imag, c2.imag, a2.imag)

    def xmax(self):
        a1, c1, c2, a2 = self.controls[0:4]
        if c1 == None or c2 == None:
            return max(a1.real, a2.real)
        else: return max(a1.real ,c1.real ,c2.real, a2.real)

    def ymax(self):
        a1, c1, c2, a2 = self.controls[0:4]
        if c1 == None or c2 == None:
            return min(a1.imag, a2.imag)
        else: return max(a1.imag ,c1.imag ,c2.imag ,a2.imag)
    
    def set_linear(self,a1,a2):
        self._cubic_or_linear = "linear"
        #c points in the middle attention, eg wrt derivatives for these
        self.controls = [a1,lerp(a1,a2,.5),lerp(a1,a2,.5),a2]

    def begin(self):
        return self.controls[0]

    def last(self):
        return self.controls[3]

    def _polynom(self,z0, z1, z2, z3,t):
        return z0*(1-t)*(1-t)*(1-t) + z1*3*(1-t)*(1-t)*t + z2*3*(1-t)*t*t + z3*t*t*t

    def _dpoldt(self,z0, z1, z2, z3,t):
        #differentiate polynom
        return z0*(-3*(1-t)*(1-t)) + z1*(3*(1-t)*(1-t)-6*(1-t)*t) + z2*(6*(1-t)*t-3*t*t) + z3*(3*t*t) 

    def tangent(self,t):
        #return tangent as a vector
        a1, c1, c2, a2 = self.controls[0:4]
        if self._cubic_or_linear == "linear": 
            dz = a2 - a1
            return dz
        else: 
            dz = self._dpoldt(a1,c1,c2,a2,t)
            return dz / abs(dz)

    def normaal(self,t):
        #return normalized vector orthonormal to tangent vector (turn rightward)
        dz = self.tangent(t)
        nx = dz.imag
        ny = -dz.real
        return nx + ny*1j

    def _curve(self,a1,c1,c2,a2):
        global plotted
        x_values = []
        y_values = []
        steps = 40 # was 40
        dt = 1.0/steps
        for i in range(steps + 1):
            t = i * dt
            z = self._polynom(a1,c1,c2,a2,t)
            x_values.append(z.real)
            y_values.append(z.imag)
            if self._filling:
                h, = self._ax.fill(self.x_values,self.y_values,self._fillcolor,linewidth=.1)
            h, = self._ax.plot(x_values, y_values, 'black', linestyle=self._linestyle, linewidth=.5)
            self._plotted.append(h)
            plotted.append(h)

    def _curve_raw(self,a1,c1,c2,a2):
        global plotted
        x_values = []
        y_values = []
        steps = 6  #cf 40 in _curve
        dt = 0.166 #must be 1/steps
        for i in range(steps + 1):
            t = i * dt
            z = self._polynom(a1,c1,c2,a2,t)
            x_values.append(z.real)
            y_values.append(z.imag)
            if self._filling:
                h, = self._ax.fill(x_values,y_values,self._fillcolor,linewidth=.1)
            h, = self._ax.plot(x_values, y_values, 'blue', linestyle=self._linestyle, linewidth=.5)
            self._plotted.append(h)
            plotted.append(h)

    def pixel(self,t):
        #let t run from 0 to 1 (or slightly beyond)
        a1,c1,c2,a2 = self.controls[0:4]
        if self._cubic_or_linear == "cubic":
            return self._polynom(a1,c1,c2,a2,t)
        else: return lerp2(a1,a2,t)

    def length_fout_fout(self):
        STEPS = 10
        dt = 1.0 / STEPS
        sum = 0
        t = 0
        for i in range(STEPS):
            p = self.pixel(t)
            q = self.pixel(t + dt)
            sum += abs(p - q)
        return sum

    def length(self):
        DT = 0.01
        sum = 0
        for t in arange(0, 1 - DT, DT):
            p = self.pixel(t)
            q = self.pixel(t + DT)
            sum += abs(q - p)
        return sum 

    def length_very_accurate(self):
        """ Use during panel design so panels will match during sewing """
        DT = 0.0001
        sum = 0
        for t in arange(0, 1 - DT, DT):
            p = self.pixel(t)
            q = self.pixel(t + DT)
            sum += abs(q - p)
        return sum 

    def t_nearest(self,z):
        """ t which gives nearest position at Bézier curve                     """
        """ Hopefully accurate but slow, untested: look a bit beyond both ends """
        d = self._INFINITY
        steps = 100 # Was 50
        dt = 1.0/steps
        aha = -1
        for t in arange(-0.1, 1.1, dt):
            p = self.pixel(t)
            newd = abs(z - p) 
            if newd < d:
                d = newd
                aha = t
        lo = aha - 2*dt # Refine search in small area around aha
        hi = aha + 2*dt # Area being from lo to ho
        d = self._INFINITY 
        steps = 100 # Was 50 
        dt = (hi - lo)/steps
        for t in arange(lo, hi, dt):
            p = self.pixel(t)
            newd = abs(z - p)
            if newd < d:
                d = newd
                aha = t
        return aha

    def t_nearest_raw(self,z):
        # t which gives nearest position at Bézier curve
        # big steps and taxicab metric, faster yet no so accurate
        d = self._INFINITY
        steps = 30 # Was 20, was 10
        dt = 1.0/steps
        aha = -1
        for t in arange(-0.02, 1.02, dt): # was -0.1, 1.1
            p = self.pixel(t)
            newd = taxicab(z,p)
            if newd < d:
                d = newd
                aha = t
        return aha

    def clear(self):
        global plotted
        if  self._plotted != []:
            for handle in self._plotted:
                handle.remove()
                if handle in plotted:
                    plotted.remove(handle)
        self._plotted = []

    def draw_raw(self):
        global plotted
        self.clear()
        a1, c1, c2, a2 = self.controls[0:4]
        self._curve_raw(a1,c1,c2,a2)
        h1, = self._ax.plot(a1.real, a1.imag, marker='o', markersize=4, color="blue")
        h2, = self._ax.plot(c1.real, c1.imag, marker='o', markersize=3, color="blue", markerfacecolor="None")
        h3, = self._ax.plot(c2.real, c2.imag, marker='o', markersize=3, color="blue", markerfacecolor="None")
        h4, = self._ax.plot(a2.real, a2.imag, marker='o', markersize=4, color="blue")
        self._plotted += [h1,h2,h3,h4]
        x_values = [a1.real, c1.real]
        y_values = [a1.real, c1.real]
        h5, = self._ax.plot(x_values, y_values, 'blue', linestyle="solid", linewidth=.5)
        x_values = [c2.real, a2.real]
        y_values = [c2.real, a2.real]
        h6, = self._ax.plot(x_values, y_values, 'blue', linestyle="solid", linewidth=.5)
        self._plotted += [h5,h6]
        plotted += [h1,h2,h3,h4,h5,h6]

    def draw(self,linestyle):
        global plotted
        if self._cubic_or_linear == "cubic":
            self.clear()
            a1, c1, c2, a2 = self.controls[0:4]
            self._linestyle = linestyle
            self._curve(a1,c1,c2,a2)
            h1, = self._ax.plot(a1.real, a1.imag, marker='o', markersize=4, color="blue")
            h2, = self._ax.plot(c1.real, c1.imag, marker='o', markersize=3, color="blue", markerfacecolor="None")
            h3, = self._ax.plot(c2.real, c2.imag, marker='o', markersize=3, color="blue", markerfacecolor="None")
            h4, = self._ax.plot(a2.real, a2.imag, marker='o', markersize=4, color="blue")
            self._plotted += [h1,h2,h3,h4]
            x_values = [a1.real, c1.real]
            y_values = [a1.imag, c1.imag]
            h5, = self._ax.plot(x_values, y_values, 'blue', linestyle="solid", linewidth=.5)
            x_values = [c2.real, a2.real]
            y_values = [c2.imag, a2.imag]
            h6, = self._ax.plot(x_values, y_values, 'blue', linestyle="solid", linewidth=.5)
            self._plotted += [h5,h6]
            plotted += [h1,h2,h3,h4,h5,h6]
        else: self.draw_linear(linestyle)

    def draw_linear(self,linestyle):
        global plotted
        self.clear()
        self._linestyle = linestyle
        a1,_,_,a2 = self.controls[0:4]
        x_values = [a1.real, a2.real]
        y_values = [a1.imag, a2.imag]
        h, = self._ax.plot(x_values, y_values, 'black', linestyle=self._linestyle, linewidth=.5)
        self._plotted.append(h)
        plotted.append(h)

    def line(self):
        #produce a polyline, i.e. list of points on the line, from t = 0 to t = 1
        HOWMANY = 200 # was 100, opgevoerd voor yard van marinapand
        dt = 1.0 / HOWMANY
        line = []
        for t in arange(0,1,dt):
            p = self.pixel(t)
            line.append(p)
        line.append(self.pixel(1))
        return line

    def _lpr(self):
        print(self.controls)
    
    def _cpi_nearest(self,p):
        #nearby controlpoint index(if any)
        #hopefully fast, use for mouse actions selecting and dragging control points
        aha = -1
        d = self._INFINITY
        nn_controls = [x for x in self.controls if x != None]
        for i in range(len(nn_controls)):
            q = nn_controls[i]
            newd = taxicab(p,q)
            if newd < d:
                d = newd
                aha = i # perhaps better use pythonic enumerate?
        return aha if aha >= 0 and d < self.DIST else None

    def button_press(self,event):
        x = event.xdata
        y = event.ydata
        z = xy2z(x,y)
        i = self._cpi_nearest(z)
        if i is not None:
            self.controls[i] = z
            self._dragged = i

    def motion_notify(self,event):
        if self._dragged is not None:
            x = event.xdata #was round
            y = event.ydata #idem
            self.controls[self._dragged] = xy2z(x,y)
            self.draw(self._linestyle) #was raw

    def button_release(self,event):
        self._dragged = None
    #end class Bezier
