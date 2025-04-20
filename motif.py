from math import sin,cos,radians
import numpy as np

from turtle import Turtle

def block10x10(ax,z,morph):
    """ this is a 10x10 box for calibration"""
    t = Turtle(ax,z,morph)
    t._filling = True
    t._fillcolor = "yellow"
    t._plotcolor = "black"
    t.pd()
    t.forward(10)
    t.left(90)
    t.forward(10)
    t.left(90) 
    t.forward(10)
    t.left(90) 
    t.forward(10)
    t.pu()

def block1x1(ax,z,morph):
    """ this is a 1x1 box for calibration"""
    t = Turtle(ax,z,morph)
    t._filling = True
    t._fillcolor = "yellow"
    t._plotcolor = "black"
    t.pd()
    t.forward(1)
    t.left(90)
    t.forward(1)
    t.left(90) 
    t.forward(1)
    t.left(90) 
    t.forward(1)
    t.pu()

def laurentius_lab(ax,z,morph):
    """ this LL logo fits in a 1x1 box"""
    t = Turtle(ax,z,morph)
    t._filling = True
    t._fillcolor = "black"
    t._plotcolor = "blue"
    t.forward(.1)
    t.left(90)
    t.forward(.1)
    t.right(90) 
    t.pd() # start making L
    t.forward(.5)
    t.left(90)
    t.forward(.2)
    t.left(90)
    t.forward(.3)
    t.right(90)
    t.forward(.6)
    t.left(90)
    t.forward(.2)
    t.left(90)
    t.forward(.8) # L done
    t.pu()
    t.forward(.1)
    t.right(90)
    t.forward(.1)
    t.left(180) # back home
    t.forward(.7)
    t.left(90)
    t.forward(.1)
    if SECOND_L_INCLUDED := True:
        t.pd() # begin 2nd L
        t.forward(.6)
        t.left(90)
        t.forward(.3)
        t.right(90)
        t.forward(.2)
        t.right(90)
        t.forward(.5)
        t.right(90)
        t.forward(.8)
        t.right(90)
        t.forward(.2) #2nd L done
        t.pu()

def pied_de_poule_block_a(ax,z,morph):
    """ this thingy fits in a 1x1 box          """
    """ entirely black                         """
    """ we start left down, turtle points east """
    t = Turtle(ax,z,morph)
    t._filling = True
    t._fillcolor = "black"
    t._plotcolor = "blue"
    t.forward(0.01)
    t.left(90)
    t.forward(0.01) 
    t.right(90) # Now we moved away from the edge
    t.pd()
    t.forward(0.98)
    t.left(90)
    t.forward(0.98)
    t.left(90) 
    t.forward(0.98)
    t.left(90) 
    t.forward(0.98)
    t.pu()

def pied_de_poule_block_b(ax,z,morph):
    t = Turtle(ax,z,morph) # home 0,0 point east
    t._filling = True
    t._fillcolor = "black"
    t._plotcolor = "blue"
    t.forward(0.01)
    t.left(90)
    t.forward(0.01) 
    t.right(90) # Now we moved away from the edge
    STEP = 0.1225 # Eight steps make 0.98
    t.forward(3 * STEP)
    t.left(90) 
    t.pd() # Make long black tail
    for i in range(3):
        t.forward(STEP)
        t.left(90)
        t.forward(STEP)
        t.right(90)
    for i in range(4):
        t.forward(STEP)
    t.right(90)
    for i in range(7):
        t.forward(STEP)
        t.right(90)
        t.forward(STEP)
        t.left(90)
    t.right(90)
    t.right(90)
    for i in range(4):
        t.forward(STEP)
    t.pu()
    t.forward(3 * STEP)
    t.right(90)
    t.right(90) # Back at starting point 0.01,0.01
    t.forward(8 * STEP)
    t.left(90)
    t.forward(8 * STEP)
    t.pd() # big-horn triangle
    t.left(90)
    for i in range(4):
        t.forward(STEP)
    t.left(90)
    for i in range(4):
        t.forward(STEP)
        t.left(90)
        t.forward(STEP)
        t.right(90)
    t.left(90)
    t.left(90)
    for i in range(4):
        t.forward(STEP)
    t.pu()

def pied_de_poule_block_c(ax,z,morph):
    t = Turtle(ax,z,morph)
    t._filling = True
    t._fillcolor = "black"
    t._plotcolor = "blue"
    t.forward(0.01)
    t.left(90)
    t.forward(0.01) 
    t.right(90) # Now we moved away from the edge
    STEP = 0.1225 # Eight steps make 0.98
    t.pd() # small horn triangle
    for i in range(3):
        t.forward(STEP)
    t.left(90)
    for i in range(3):
        t.forward(STEP)
        t.left(90)
        t.forward(STEP)
        t.right(90)
    t.left(90)
    t.left(90)
    for i in range(3):
        t.forward(STEP)
    t.pu()
    t.left(90) # back home
    t.forward(8 * STEP)
    t.left(90) 
    t.pd() # long tail, start from lower tip, go ccw
    for i in range(4):
        t.forward(STEP)
    t.left(90)
    for i in range(4):
        t.forward(STEP)
        t.right(90)
        t.forward(STEP)
        t.left(90)
    for i in range(4):
        t.forward(STEP)
    t.left(90)
    for i in range(8):
        t.forward(STEP)
        t.left(90)
        t.forward(STEP)
        t.right(90)
    t.pu()

def pied_de_poule_block_d(ax,z,morph):
    pass # it is white, haha
    