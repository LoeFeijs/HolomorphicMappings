from   matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
from   numba import jit,njit
from   math import isnan,radians,degrees
import numpy as np 

if VERBOSE := True:
    print("NumPy version:", np.__version__)
    import numba as nb
    print("Numba version:", nb.__version__)
    import matplotlib as plt
    print("Matplotlib version:", plt.__version__)

from bezier import Bezier,clear_bezier_lines
from turtle import Turtle
from scaffold import Scaffold
from contour import *

from timer import Timer
from polyline import *
from turtle import *
from field import *
from motif import *
from grid  import *


### Use python39, C:\Users\LFEIJS\AppData\Local\Programs\Python\Python39\python.exe
### April 2025, coding goals and conventions:
### 1. use complex numbers instead of x,y pairs, except when addressing an array
### 2. potentials etc are in numpy array [j,i], j being a rounded y, i a rounded x
### 3. avoid nested lists, as that blocks numba jit, lists of complex are probably fine
### 4. p and z are points in the complex plane, x,y coords, but better do z instead if x,y
### 5. t is a real number (float) parameterizing conductors and insulators, d is distance
### 6. r may be a float range, s some shift or offset. Constants are capitalized. 
### 7. art board size must be integer, garment coordinates can be reals (and a pair of them a complex)

#GLOBAL CONSTANTS:
XMIN = 0                # restrict area of relaxation
YMIN = 0                # idem
ABXY = [300,300]        # all definitions of this should match
XMAX = ABXY[0]          # sorry, must match ABX,ABY globals from field.py
YMAX = ABXY[1]          # vierkant 300x300 
CONTOUR = "VIERKANT"    # Dutch for four-sided contour, the only type supported in this software

if CONTOUR in ["VIERKANT"]:
    set_abx_aby(ABXY)
    NH = 22              # Number of Horizontal strips
    NV = 20              # Idem vertical
    NHLIST = [22]        # The same again
    NVLIST = [20]        # The same again
    SNG = 10             # not appliccable
    GREENWICH = 150 + 150j  # Starting point of path integration for con

# GLOBAL VARIABLES:

cnt = None      # Contour for main panel
scf = None      # Scaffold based on the given contour
fig = None      # For matplotlib gui
ax  = None      # Idem (many more ax* variants, butons, sliders)
pot = None      # Various electrostatics-like components
con = None      # Eg potential,its harmonic conjugate
bnd = None      # Pixelated boundary conditions
out = None      # Outside of darts test
wmh = None      # Wormhole table
field = None    # The (electric) field
confield = None # And its harmonic conjugate

#GUI LAYOUT AND CONNECTIONS:
def gui_create():
    global fig,ax
    global axpot,axfieldlines,axcon,axgrid,axmotif,axclear
    global button_pot,button_fieldlines,button_con,button_equipots
    global button_grid,button_curves,button_clear,button_clear,button_motif,button_svg

    axpot =        fig.add_subplot([0.88, 0.885, 0.1, 0.075])   #buttons, params: x,y,b,h
    axfieldlines = fig.add_subplot([0.88, 0.785, 0.1, 0.075])   #2nd button
    axcon =        fig.add_subplot([0.88, 0.685, 0.1, 0.075])   #harmonic conjugate of pot
    axequipots =   fig.add_subplot([0.88, 0.585, 0.1, 0.075])   #..
    axgrid =       fig.add_subplot([0.88, 0.485, 0.1, 0.075])   #..
    axcurves =     fig.add_subplot([0.88, 0.385, 0.1, 0.075])   #..
    axclear =      fig.add_subplot([0.88, 0.285, 0.1, 0.075]) 
    axmotif =      fig.add_subplot([0.88, 0.185, 0.1, 0.075])   #
    axsvg =        fig.add_subplot([0.88, 0.085, 0.1, 0.075])   #
    
    button_pot =        Button(axpot, 'Pot')               #creating the widgets
    button_fieldlines = Button(axfieldlines, 'Fieldlines') #another widget
    button_con =        Button(axcon, 'Con')               #another widget 
    button_equipots =   Button(axequipots, 'Equipots')     #..
    button_grid =       Button(axgrid, 'Grid')             #..
    button_curves =     Button(axcurves, 'Curves')
    button_clear =      Button(axclear, 'Clear')
    button_motif =      Button(axmotif, 'Motif')
    button_svg =        Button(axsvg, 'Svg')

def gui_connect():
    fig.canvas.mpl_connect('button_press_event', button_press)
    fig.canvas.mpl_connect('motion_notify_event', motion_notify)
    fig.canvas.mpl_connect('button_release_event', button_release)

    button_pot.on_clicked(pot_clicked)
    button_con.on_clicked(con_clicked)
    button_fieldlines.on_clicked(fieldlines_clicked)
    button_equipots.on_clicked(equipots_clicked)
    button_grid.on_clicked(grid_clicked)
    button_curves.on_clicked(curves_clicked)
    button_clear.on_clicked(clear_clicked)
    button_motif.on_clicked(motif_clicked)
    button_svg.on_clicked(svg_clicked)
    
    button_pot.hovercolor = 'green'
    button_fieldlines.hovercolor = 'red'
    button_con.hovercolor = 'red'
    button_equipots.hovercolor = 'red'
    button_grid.hovercolor = 'red'
    button_curves.hovercolor = 'red'
    button_clear.hovercolor = 'red'
    button_motif.hovercolor = 'red'
    button_svg.hovercolor = 'red'

#EVENT HANDLER ROUTINES:
def button_press(event):
    global cnt, scf
    if event.inaxes != ax: return
    #so this is not for sliders 
    b : Bezier
    for b in cnt.conductors + cnt.insulators + cnt.transducers:
        b.button_press(event)
    if scf:
    	for b in scf.mast_beziers + scf.yard_beziers:
            if b:
                b.button_press(event)

def motion_notify(event):
    global ax,cnt,scf
    if event.inaxes != ax: return
    b : Bezier
    for b in cnt.conductors + cnt.insulators + cnt.transducers:     
        b.motion_notify(event)
    if scf:
        for b in scf.mast_beziers + scf.yard_beziers:
            if b:
                b.motion_notify(event)
    plt.draw()

def button_release(event):
    global ax,cnt,scf,conductors,pot,bnd,field,fig,im
    if event.inaxes != ax: return
    b : Bezier
    for b in cnt.conductors + cnt.insulators + cnt.transducers:
        b.button_release(event)
    for b in cnt.conductors + cnt.insulators:
        b.draw("solid")  
    for b in cnt.transducers:
        b.draw("dashed")   
    if scf:
    	for b in scf.mast_beziers + scf.yard_beziers:
            if b:
                b.button_release(event)

    button_pot.hovercolor = 'green'
    button_con.hovercolor = 'red'
    button_fieldlines.hovercolor = 'red'
    button_clear.hovercolor = 'red'
    plt.draw()

def pot_clicked(event):
    """ Compute the field components                                                          """
    """ Also needed to (re-)electrify the scaffold, for example after adjusting the sliders!! """
    global pot,bnd,wmh,field,con,confield,cnt,XMIN,YMIN,XMAX,YMAX

    plt.draw()
    XMIN,YMIN,XMAX,YMAX = apply_boundaries(cnt,pot,bnd,wmh,cnt.conductors,cnt.insulators,cnt.transducers) 
    precompute_pot(cnt,pot,cnt.conductors)
    howmany = 80 
    for i in range(howmany):
        with Timer("compute_pot"):
            compute_pot(pot, bnd, wmh, out, XMIN-5, YMIN-5, XMAX+5, YMAX+5, 2000)
        im.set_data(rasterize(pot))
        plt.draw()
        plt.pause(0.01)

    im.set_data(rasterize(pot))
    if VERBOSE := True:
        print("lengths of conductors follow (upper, lower line):")
        for b in cnt.conductors:
            bl = b.length_very_accurate()
            print(f"    {bl:.1f}")
        print("lengths of insulators follow (right, left line):")
        for b in cnt.insulators:
            bl = b.length_very_accurate()
            print(f"    {bl:.1f}")

    plt.draw()
    plt.pause(0.01)

    compute_field(pot,field) 
    print("COMPUTED: pot, field")
    con = ini_con(bnd)
    with Timer("compute_con"):
        compute_con(field,con,bnd,wmh,XMIN, YMIN, XMAX, YMAX, GREENWICH)
    
    compute_confield(field,confield) 
    print("COMPUTED: con, confield")

    with Timer("scaffolding"):
        scf.update()
        scf.electrify(pot,con,NHLIST,NVLIST)
        scf.show_points()

    button_pot.hovercolor = 'grey'
    button_fieldlines.hovercolor = 'green'
    button_con.hovercolor = 'green'
    button_clear.hovercolor = 'red'
    plt.draw()
    return

# MORE GLOBALS:
fieldlines = None # fill by fieldlines_clicked(event)
equipots   = None # fill by equipots_clicked(event)

def fieldlines_clicked(event):
	# This version is prepared for multiple bundles of field lines and equipot lines
	# For "VIERKANT", there is only one bundle needed, viz. the ll (long lines)
    global ax,scf,fieldlines

    with Timer("find_fieldlines"):
        fieldlines = find_fieldlines(cnt,scf,field,cnt.conductors) 
        # Results: short_lines_left, long_lines, short_lines_right, short_lines_lower_left, short_lines_lower_right
        # Aka (packed): sll,ll,slr,slll,sllr
        # Attention: fieldlines are directed

    if cnt.name == "VIERKANT":                         
        _,ll,_,_,_,_,_ = fieldlines                     
        print("len(ll)",len(ll))
        B,D = cnt.insulators           
  
        ll[-1] =  B.line() 
        ll[0] = D.line()[::-1]   
       
    sll, ll, slr, slll, slll_bis, sllr, sllr_bis = fieldlines  # Lots of bundles, for "VIERKANT", one would be enough
    for f in sll + ll + slr + slll + slll_bis + sllr + sllr_bis:
        draw_polyline(ax,f) 
        
    plt.draw()
    button_fieldlines.hovercolor = 'grey'
    button_clear.hovercolor = 'green'
    return


def con_clicked(event):
    #con from field must be ready, now show it
    global plt,field,con,confield,bnd,XMIN,YMIN,XMAX,YMAX
    im.set_data(rasterize_bis(con))
    plt.draw()
    button_con.hovercolor = 'grey'
    button_equipots.hovercolor = 'green'
    print("SHOWING FIELD COMPONENT: con")
    return


def equipots_clicked(event):
    global ax,equipots,SNG

    equipots = find_equipots(ax,cnt,scf,confield,cnt.conductors,cnt.insulators,SNG)
    uppers, singls, lowers = equipots # Equipots must run right to left (for mouwpand some don't flip later)

    def nearest(arm,z):
    	# find point on arm line closest to point z
    	return min(arm, key=lambda x:abs(x - z))

    def split(arm,split_point):
    	# PRE:  arm line runs somewhat vertically, i.e. by decreasing .imag
    	above = [p for p in arm if p.imag >= split_point.imag]
    	below = [p for p in arm if p.imag <= split_point.imag]
    	return above, below

    if cnt.name == "VIERKANT":                                         
        A,C = cnt.conductors
        uppers[0] = A.line()
        uppers[-1] = C.line()[::-1]  
        _stretch(uppers[0])     
        _stretch(uppers[-1])    

    for e in uppers: 
        draw_polyline(ax,e) 

    plt.draw()
    button_fieldlines.hovercolor = 'grey'
    button_grid.hovercolor = 'green'
    button_clear.hovercolor = 'green'
    return


def _stretch(e):
    """ Typically e is an equipot line (or a field line)       """
    """ Make the ends stick out for not missing end-intersects """
    e[0] = e[0] - 5*(e[1] - e[0])      # Was 1*   
    e[-1] = e[-1] + 5*(e[-1] - e[-2])  # Idem


def _micro_stretch(e):
    """ Typically e is an equipot line (or a field line)       """
    """ Make the ends stick out for not missing end-intersects """
    e[0] = e[0] - 1.0*(e[1] - e[0])      # Was  5*Was 1*   
    e[-1] = e[-1] + 1.0*(e[-1] - e[-2])  # Idem

def grid_clicked(event):
    global ax,grid, spec
    # First, let's count what we have:
    sll, ll, slr, slll, slll_bis, sllr, sllr_bis = fieldlines
    print("len(sll,ll,slr)",len(sll),len(ll),len(slr))
    print("len(slll,slll_bis,sllr,sllr_bis)",len(slll),len(slll_bis),len(sllr),len(sllr_bis))
    uppers, singls, lowers = equipots
    print("len(uppers) incl copy single",len(uppers))
    print("len(lowers) incl copy single",len(lowers))
    for group in fieldlines:
        for line in group:
            _micro_stretch(line) # WAS _stretch RE-TEST THIS FOR VOORPAND, ACHTERPAND
    for group in equipots:
        for line in group:
            _micro_stretch(line)
    do_grid_anchor_points(cnt,grid,fieldlines,[uppers,lowers])
    do_grid_control_points(cnt,grid)
 
    if cnt.name == "VIERKANT": 
        draw_grid(ax,grid)

    plt.draw()
    button_curves.hovercolor = 'green'
    button_clear.hovercolor = 'green'    
    print("Grid done")
    return


def curves_clicked(event):
    global ax,grid,equipots,fieldlines
    insert_bezierifieds_into_grid(ax,cnt,grid,equipots,fieldlines)
    button_motif.hovercolor = 'green'
    button_clear.hovercolor = 'green'
    plt.draw()    
    print("Curves done")
    return


def clear_clicked(event):
	# Do not clear the bezier lines
    clear_plotted(ax)
    plt.draw()
    return


def morph(t):
    global grid
    pos,a1,c1,c2,a2,a1v,c1v,c2v,a2v = grid
    return grid2morph(pos,a1,c1,c2,a2,a1v,c1v,c2v,a2v,t)

def redraw_curves(ax,grid,equipots,fieldlines): # Huge auxiliary for motif_clicked
    pos,a1,c1,c2,a2,_,_,_,_ = grid
    uppers,_,lowers = equipots
    sll, ll, slr, slll, _, sllr, _ = fieldlines

    # Redraw the contour curves, use turtle, do not morph, colour is red
    for b in cnt.conductors + cnt.insulators + cnt.transducers:
        a1_,c1_,c2_,a2_ = b.controls
        if c1_ is None: c1_ = lerp2(a1_,a2_,0.3)
        if c2_ is None: c2_ = lerp2(a1_,a2_,0.7)
        t = Turtle(ax,a1_,lambda x:x)
        t._filling = False
        t._plotcolor = "red"
        t.pd()    
        t.curveto(c1_,c2_,a2_)
        t.pu() 

    # Redraw the equipot-based curves stored in a1,c1,c2,a2 of grid, use turtle, do not morph
    for j in range(len(uppers)): 
        i_ini = 0
        z = a1[j,i_ini]
        t = Turtle(ax,z,lambda x:x)
        t._filling = False
        t._plotcolor = "blue"
        t.pd()    
        for i in range(len(sll) + len(ll) + len(slr) - 1):
            t.curveto(c1[j,i],c2[j,i],a2[j,i])
        t.pu()       
    plt.draw()
    plt.pause(0.01)

    off_i = len(sll) - len(slll) # Offset
    off_j = len(uppers)          # Idem
    for j in range(off_j, off_j + len(lowers)): 
        i_ini = off_i
        z = a1[j,i_ini]
        t = Turtle(ax,z,lambda x:x)
        t._filling = False
        t._plotcolor = "black"
        t.pd()    
        for i in range(off_i, off_i + len(slll) + len(ll) + len(sllr) - 1):
            t.curveto(c1[j,i],c2[j,i],a2[j,i])
        t.pu()       
    plt.draw()
    plt.pause(0.01)

    _,_,_,_,_,a1v,c1v,c2v,a2v = grid

    # Redraw the upper fieldline-based curves (going from a2v via c2v,c1v to a1v)
    off_i = 0 # Offset
    off_j = 0 # Idem
    howmany_i = len(sll)
    for i in range(off_i, off_i + howmany_i):
        z = a2v[1 + off_j,i]
        t = Turtle(ax,z,lambda x:x)
        t._filling = False
        t._plotcolor = "black"
        t.pd()
        for j in range(1 + off_j, off_j + len(uppers)):
            t.curveto(c2v[j,i],c1v[j,i],a1v[j,i])
        t.pu()       
    plt.draw()
    plt.pause(0.01)

    off_i = len(sll)
    off_j = 0 
    howmany_i = len(ll)
    for i in range(off_i, off_i + howmany_i):
        z = a2v[1 + off_j,i]
        t = Turtle(ax,z,lambda x:x)
        t._filling = False
        t._plotcolor = "black"
        t.pd()
        for j in range(1 + off_j, off_j + len(uppers)):
            t.curveto(c2v[j,i],c1v[j,i],a1v[j,i])
        t.pu()       
    plt.draw()
    plt.pause(0.01)

    off_i = len(sll) + len(ll)
    off_j = 0 
    howmany_i = len(slr)
    for i in range(off_i, off_i + howmany_i):
        z = a2v[1 + off_j,i]
        t = Turtle(ax,z,lambda x:x)
        t._filling = False
        t._plotcolor = "black"
        t.pd()
        for j in range(1 + off_j, off_j + len(uppers)):
            t.curveto(c2v[j,i],c1v[j,i],a1v[j,i])
        t.pu()       
    plt.draw()
    plt.pause(0.01)

    off_i = len(sll) - len(slll)
    off_j = len(uppers)
    howmany_i = len(slll)
    for i in range(off_i, off_i + howmany_i):
        z = a2v[1 + off_j,i]
        t = Turtle(ax,z,lambda x:x)
        t._filling = False
        t._plotcolor = "black"
        t.pd()
        for j in range(1 + off_j, off_j + len(lowers)):
            t.curveto(c2v[j,i],c1v[j,i],a1v[j,i])
        t.pu()       
    plt.draw()
    plt.pause(0.01)

    off_i = len(sll) 
    off_j = len(uppers)
    howmany_i = len(ll)
    for i in range(off_i, off_i + howmany_i):
        z = a2v[1 + off_j,i]
        t = Turtle(ax,z,lambda x:x)
        t._filling = False
        t._plotcolor = "black"
        t.pd()
        for j in range(1 + off_j, off_j + len(lowers)):
            t.curveto(c2v[j,i],c1v[j,i],a1v[j,i])
        t.pu()       
    plt.draw()
    plt.pause(0.01)

    off_i = len(sll) + len(ll)
    off_j = len(uppers)
    howmany_i = len(sllr)
    for i in range(off_i, off_i + howmany_i):
        z = a2v[1 + off_j,i]
        t = Turtle(ax,z,lambda x:x)
        t._filling = False
        t._plotcolor = "black"
        t.pd()
        for j in range(1 + off_j, off_j + len(lowers)):
            t.curveto(c2v[j,i],c1v[j,i],a1v[j,i])
        t.pu()       
    plt.draw()
    plt.pause(0.01)

    return # End of huge redraw_curves

def motif_clicked(event):
    global ax,grid,equipots,fieldlines
    #pos,a1,c1,c2,a2,_,_,_,_ = grid
    #uppers,_,lowers = equipots
    #sll, ll, slr, slll, sllr = fieldlines
    redraw_curves(ax,grid,equipots,fieldlines)

    uppers,_,_ = equipots
    pos,_,_,_,_,_,_,_,_ = grid
    
    if PRINTING_MOTIFS := True:
        # Now do the fashion motifs, also deploy turtle
        for j in range(len(pos) - 1):              # On screen: j from top to bottom
            for i in range(len(pos[0]) - 1):       # On screen: i from left to right
                if pos[j,i] is not None:           # Eg in lower left region some unused j,i (tshirt)
                    k = len(pos) - 1 - j           # Apply ~upside down conversion
                    z = i + k*1j                   # z: as in "ax" plotting system 
                    if LAURENTIUS_INSTEAD_OF_POULES := True: # Choose one of the three options True, two False
                        laurentius_lab(ax,z,morph) # For test use lambda x:3*x (instead of morph)
                    if PIED_DE_POULES := False: 
                        if j % 2 == 0 and i % 2 == 0: pied_de_poule_block_a(ax,z,morph) 
                        if j % 2 == 0 and i % 2 == 1: pied_de_poule_block_b(ax,z,morph) 
                        if j % 2 == 1 and i % 2 == 0: pied_de_poule_block_c(ax,z,morph) 
                        if j % 2 == 1 and i % 2 == 1: pied_de_poule_block_d(ax,z,morph) 
     
    block10x10(ax,0          +          0*1j, lambda x:x) # calibration blocks
    block10x10(ax,0          + (ABY - 10)*1j, lambda x:x) # calibration
    block10x10(ax,(ABX - 10) + (ABY - 10)*1j, lambda x:x) # calibration
    block10x10(ax,(ABX - 10) +          0*1j, lambda x:x) # calibration

    plt.draw()
    print("Motif done") 
    button_svg.hovercolor = 'green'
    return


def svg_clicked(event):
    global svg
    print("Svg clicked")
    with open("garment.svg","w") as svg:
        svg.write(prelude())
        for p in get_paths(): # from global path in turtle
            svg.write(p) 
        svg.write(postlude())
    return


if __name__ == '__main__':
    #FIGURE, AXES, ART BOARD

    print("DRAG CONTROL POINT.....")
    print("OR COMPUTE POTENTIALS:)")
    print("AFTER THAT: FIELD LINES")
    # set_abx_aby(ABXY) # ABX and ABY are in field
    fig = plt.figure("Analytic Garment by LAURENTIUS LAB.")
    fig.patch.set_facecolor('pink')
    ax = fig.add_subplot(111)
    plt.tight_layout()

    gui_create()
    gui_connect()

    #GARMENT CONSTRUCTION
    cnt = Contour(ax, CONTOUR)
    scf = cnt.scaffold()
    for vh in scf.mast_beziers + scf.yard_beziers:
        if vh:
            vh.draw("solid")
    plt.draw()
      
    if cnt.name == "VIERKANT":
        print("VIERKANT 300x300")
        cnt.define()
        cnt.electrify()  

    ax.set_xlim([0,ABX - 1])
    ax.set_ylim([0,ABY - 1])

    #GLOBAL INITIALISATION AND FIRST DRAWING:
    pot = ini_pot()
    bnd = ini_bnd()
    wmh = ini_wmh()
    field = ini_field()
    confield = ini_confield()
    ## precompute_pot(pot,cnt.conductors)
    if HERSTEL := True:
        compute_field(pot,field) 

    im = ax.imshow(rasterize(pot))
    if PRISM_COLOR_SCHEMA := False:
        im.set_cmap("prism") # was "prism" or "flag"

    XMIN,YMIN,XMAX,YMAX = apply_boundaries(cnt,pot,bnd,wmh,cnt.conductors,cnt.insulators,cnt.transducers)

    show_wmh = np.full((ABY,ABX),0.)
    for j in range(ABY):
    	for i in range(ABX):
    		show_wmh[j,i] = (wmh[j,i].imag % 100) / 100

    show_bnd = np.full((ABY,ABX),0.)
    for j in range(ABY):
    	for i in range(ABX):
    		show_bnd[j,i] = 0

    con = np.full((ABY,ABX),-0.1)
    con = ini_con(bnd)
    out = ini_out(cnt)

    grid = ini_grid(NH,NV)
    im.set_data(rasterize(pot))

    plt.draw()
    plt.pause(0.01)
    plt.show()
    if CHECK_BACKEND := False: 
        print("backend: ", plt.get_backend())
