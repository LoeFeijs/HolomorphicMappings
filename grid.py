import math
import numpy as np
from numba import jit


from turtle import Turtle
from polyline import *
from field import * 

# grid definition, storage and computation for the mapping t2z, which is a holomorphic mapping
# from an NV*NH area, where a turtle lives in cartesian coords, to the garment area, where we have equipots and fieldlines.
# This module should have been a class MyGrid with attributes pos, a1,c1,c2,a2,a1v,c1v,c2v,a2v but Numba does work for classes.
# the c1,c2 are given relative to the a1,a2. The a1,c1,c2,a2 describes a horizontal aka equipot bezier segment, a1v,c1v,c2v,a2v vertical aka fieldline.
# There is redundancy: a1 is pos and a2v[j,i] = a1[j - 1,i] etc. The [j,i] counting goes top-down (j) and left-right (i). The a2v is above a1v = a1

def ini_grid(NH,NV):  
    pos = np.empty([NH + 4, NV + 1], dtype = complex) 
    a1  = np.empty([NH + 4, NV + 1], dtype = complex)
    c1  = np.empty([NH + 4, NV + 1], dtype = complex)
    c2  = np.empty([NH + 4, NV + 1], dtype = complex)
    a2  = np.empty([NH + 4, NV + 1], dtype = complex)
    a1v = np.empty([NH + 4, NV + 1], dtype = complex)
    c1v = np.empty([NH + 4, NV + 1], dtype = complex)
    c2v = np.empty([NH + 4, NV + 1], dtype = complex)
    a2v = np.empty([NH + 4, NV + 1], dtype = complex)
    def noneify(a):
        for j in range(len(a)):
            for i in range(len(a[0])):
                a[j,i] = None
    for np_array in [pos,a1,c1,c2,a2,a1v,c1v,c2v,a2v]:
        noneify(np_array)
    return pos,a1,c1,c2,a2,a1v,c1v,c2v,a2v


def do_grid_anchor_points(cnt,grid, fieldlines, uppers_lowers):
    """ MOD grid, notably its pos, contour cnt is needed for its name: case distinction                                    """
    """ Index positions (they are going left-to-right, top-to-bottom) """
    pos,a1,c1,c2,a2,a1v,c1v,c2v,a2v = grid 
    sll, ll, slr, slll, slll_bis, sllr, sllr_bis = fieldlines
    uppers, lowers = uppers_lowers
    def cross(fgroup, egroup, j_offset ,i_offset):
        nonlocal pos
        for j in range(len(egroup)):     
            for i in range(len(fgroup)): 
                z = intersect(egroup[j], fgroup[i])
                if z is not None:
                    pos[j + j_offset,i + i_offset] = z 
    if cnt.name in ["VIERKANT"]:
        cross(ll,uppers,0,0) 


def do_grid_control_points(cnt,grid):
    """ MOD grid, notably its a1,c1,c2,a2,a1v,c1v,c2v,a2v                             """
    """ Note that pos indexes are going left-to-right, top-to-bottom                  """
    """ These are preliminary c1,c2 values, which will work, yet are not realy curved """
    pos,a1,c1,c2,a2,a1v,c1v,c2v,a2v = grid 
    for j in range(len(pos)):
        for i in range(len(pos[0])):
            z = pos[j,i]
            # Later use -field for control points of horizontal segments
            # .. idem the confield for the vertical segments
            # e.g. f = f/abs(f) so normalize them before use 
            if z is not None:
                if i + 1 < len(pos[0]) and pos[j,i + 1] is not None:
                    a1[j,i] = pos[j,i]                  # to be overwritten by bezierify routines
                    a2[j,i] = pos[j,i + 1]              # idem 
                    c1[j,i] = lerp2(a1[j,i],a2[j,i],.3) # default, idem, 
                    c2[j,i] = lerp2(a1[j,i],a2[j,i],.7) # idem
                if j - 1 >= 0 and pos[j - 1,i] is not None:
                    a1v[j,i] = pos[j,i]
                    a2v[j,i] = pos[j - 1,i] 
                    c1v[j,i] = lerp2(a1v[j,i],a2v[j,i],.3) 
                    c2v[j,i] = lerp2(a1v[j,i],a2v[j,i],.7)


def bezierify(equipot,anchors):
    """ The equipot line is a polyline, and its segments near each anchor give a tangent vector   """
    """ This direction provided the esssential information for choosing non-anchor control points """ 
    """ The result consists of four lists: one for the a1 points, one for the c1 points and so on """
    nn_anchors = [x for x in anchors if x is not None and not math.isnan(x.real)]
    if len(nn_anchors) == 0:
        return [],[],[],[]
    a1 = [] # anchor points
    c1 = [] # control points
    c2 = [] 
    a2 = []
    def index_near(z):
        nonlocal equipot
        here = None
        minimum = 1000000
        for i in range(len(equipot)):
            d = abs(z - equipot[i])
            if d < minimum:
                minimum = d
                here = i
        return here
    a = nn_anchors[0]
    i = index_near(a)
    assert not i is None
    de = equipot[1] - equipot[0] if i == 0 else equipot[i + 1] - equipot[i - 1] 
    da = abs(nn_anchors[1] - nn_anchors[0]) # distance between anchors
    de = de / abs(de)                       # normalised tangent at equipot
    a1.append(a)                            # get the first segment started
    c1.append(a + .3*da*de)                 # idem
    prev_a = a                              # previous anchor
    for a in nn_anchors[1:]:                # The first one is done already
        i = index_near(a)                   # We have probably many more i's than a's (we hope so)
        de = equipot[i + 1] - equipot[i - 1] if i + 1 < len(equipot) else equipot[-1] - equipot[-2]
        da = abs(a - prev_a)                # distance between anchors
        if abs(de) > 0:                     # Avoid division by zero
            de = de / abs(de)               # If possible, normalise tangent again
        c2.append(a - .3*da*de)             # Finish previous bezier segment
        a2.append(a)                        # for values of .3 see /bezierCircleResearch
        a1.append(a)                        # launch new segment
        c1.append(a + .3*da*de)             # idem
        prev_a = a                          # previous anchor
    return a1,c1,c2,a2                      # four lists, as in grid


def draw_bezierified(ax,a1,c1,c2,a2):
    if len(c1) == 0: return
    for i in range(len(c2)): #let op c2,a2 korter dan a1,c1
        b = Bezier(ax,[a1[i],c1[i],c2[i],a2[i]])
        if HANDLES := False:
            b.draw_raw()
        else: b._curve_raw(a1[i],c1[i],c2[i],a2[i]) #bah


def insert_bezierified_equipot(grid,a1_,c1_,c2_,a2_,j,i_ini):
    # We have a1_ ie list of a1 points, c1_ a list of c1 points, etc.
    # They must me transferred to the a1, c1, etc of the grid, begin at j_ini,i cf pos[j_ini,i]
    # MOD grid, notably the a1,c1,c2,a2
    _,a1,c1,c2,a2,_,_,_,_ = grid 
    if len(a1_) == 0: 
        return
    a1[j,i_ini] = a1_[0]
    if math.isnan(a1_[0].real): println("insert_bezierified_equipot nan error, h = 0") 
    c1[j,i_ini] = c1_[0]
    c2[j,i_ini] = c2_[0]
    VERBOSE = False
    if VERBOSE and a2[j,i_ini] != a2_[0]:
        print("j,i = ",j,i_ini)
        print("a2[j,i_ini] = ",a2[j,i_ini])
        print("a2_[0]      = ",a2_[0])
 
    a2[j,i_ini] = a2_[0]
    for h in range(1,len(c2_)): # attention: a2,c2 shorter than a1,c1
        if math.isnan(a1_[h].real): println("insert_bezierified_equipot nan error, h =",h) 
        a1[j,i_ini + h] = a1_[h]
        c1[j,i_ini + h] = c1_[h]
        c2[j,i_ini + h] = c2_[h]
        a2[j,i_ini + h] = a2_[h]


def insert_bezierified_fieldline(grid,a1v_,c1v_,c2v_,a2v_,j_ini,i):
    # MOD grid, notably a1v,c1v,c2v,a2v
    # The line has been bezierified moving downward, so a1v was above a2v (here we reverse them)
    _,_,_,_,_,a1v,c1v,c2v,a2v = grid 
    if len(a1v_) == 0: 
        return
    a2v[j_ini,i] = a1v_[0]
    if math.isnan(a1v_[0].real): println("insert_bezierified_fieldline nan error, h = 0") 
    c2v[j_ini,i] = c1v_[0]
    c1v[j_ini,i] = c2v_[0]
    a1v[j_ini,i] = a2v_[0]
    for h in range(1,len(c2v_) + 1): # attention: a2v,c2v shorter than a1v,c1v
        a2v[j_ini + h,i] = a1v_[h - 1]
        if math.isnan(a1v_[h].real): println("insert_bezierified_fieldline nan error, h =",h) 
        c2v[j_ini + h,i] = c1v_[h - 1]
        c1v[j_ini + h,i] = c2v_[h - 1]
        a1v[j_ini + h,i] = a2v_[h - 1]

def pairwise(t):  # Thx Jochen Ritzel
    it = iter(t)   # Stackoverflow.com/questions/4628290/pairs-from-single-list
    return zip(it,it)

def insert_bezierifieds_into_grid(ax, cnt, grid, equipots, fieldlines):
    # MOD grid Version for TSHIRT, VOORPAND, ACHTERPAND, VIERKANT (not MOUWPAND)
    pos,*_ = grid
    uppers,singls,lowers = equipots
    sll, ll, slr, slll, _, sllr, _ = fieldlines
    eqpts = uppers + lowers # singls already gone into these, once in each
    print("bezierifying uppers")
    for j in range(len(uppers)):
        a1,c1,c2,a2 = bezierify(uppers[j],pos[j])
        draw_bezierified(ax,a1,c1,c2,a2)
        insert_bezierified_equipot(grid,a1,c1,c2,a2,j,0)

    for j in range(len(lowers)):
        # print("j",j) 
        offset_j = len(uppers) 
        offset_i = len(sll) - len(slll)
        a1,c1,c2,a2 = bezierify(lowers[j],pos[j + offset_j])
        draw_bezierified(ax,a1,c1,c2,a2)
        insert_bezierified_equipot(grid,a1,c1,c2,a2, j + offset_j, offset_i)
    def uppers_pos_(i):
        nonlocal uppers
        slice = [pos[j,i] for j in range(len(uppers))]
        return slice
    def lowers_pos_(i):
        nonlocal uppers,lowers
        slice = [pos[len(uppers) + j,i] for j in range(len(lowers))]
        return slice

    print("bezierifying long lines upstairs")
    for i in range(len(ll)):
        offset_i = len(sll)
        if DEBUGGING := False:
            print("ll[i] ",ll[i])
        a1v,c1v,c2v,a2v = bezierify(ll[i],uppers_pos_(i + offset_i))
        draw_bezierified(ax,a1v,c1v,c2v,a2v)
        insert_bezierified_fieldline(grid,a1v,c1v,c2v,a2v,0,i + offset_i)

    print("bezierifying short lines right:")
    for i in range(len(slr)):
        offset = len(sll) + len(ll)
        a1v,c1v,c2v,a2v = bezierify(slr[i],uppers_pos_(i + offset))
        draw_bezierified(ax,a1v,c1v,c2v,a2v)
        insert_bezierified_fieldline(grid,a1v,c1v,c2v,a2v,0,i + offset)

    print("bezierifying long lines downstairs:")
    for i in range(len(ll)):
        offset_j = len(uppers)
        offset_i = len(sll)
        a1v,c1v,c2v,a2v = bezierify(ll[i],lowers_pos_(i + offset_i))
        draw_bezierified(ax,a1v,c1v,c2v,a2v)
        insert_bezierified_fieldline(grid,a1v,c1v,c2v,a2v,offset_j,i + offset_i)

    print("bezierifying short lower lines right:")
    for i in range(len(sllr)):
        offset_j = len(uppers)
        offset_i = len(sll) + len(ll)
        a1v,c1v,c2v,a2v = bezierify(sllr[i],lowers_pos_(i + offset_i))
        draw_bezierified(ax,a1v,c1v,c2v,a2v)
        insert_bezierified_fieldline(grid,a1v,c1v,c2v,a2v,offset_j,i + offset_i)


def do_specials_tshirt(singls,larm,rarm):
    s1 = intersect(larm.line(),singls[0])
    s2 = intersect(rarm.line(),singls[0])
    return s1,s2


def do_specials_voorpand(singls,c1,c2):
    s = intersect(c1.line() + c2.line(),singls[0])
    return [s]


def draw_grid(ax, grid):
    pos,a1,c1,c2,a2,a1v,c1v,c2v,a2v = grid
    # show the intersection points:
    for j in range(len(pos)): 
        for i in range(len(pos[0])):
            z = pos[j,i]
            if z is not None:
                draw_point(ax,z,True)
        

def draw_specials(ax, specials):
    # show the special points as orange
    for s in specials:
        if s is not None:
            draw_point_orange(ax,s,True)


@jit("c16(c16,c16,c16,c16,f8)", nopython=True, nogil=True)
def _bezier(a1,c1,c2,a2, t : float) -> complex:
    # Parameter t, 0<= t <=1, could also work outside 0..1 range, but it may ben inward again (not good)
    return a1*(1-t)*(1-t)*(1-t) + c1*3*(1-t)*(1-t)*t + c2*3*(1-t)*t*t + a2*t*t*t

@jit("c16(c16,c16,c16,c16,f8)", nopython=True, nogil=True)
def _protruding_bezier(a1,c1,c2,a2, t : float) -> complex:
    # Parameter t, 0<= t <=1 means just _bezier
    # Otherwise protrude in the a1-a2 direction (avoid bending inward)
    if t < 0:
        aa = a2 - a1
        protrude = -t
        extreme_left = a1 - aa
        z = lerp2(a1,extreme_left,protrude)
    if t > 1:
        aa = a2 - a1
        protrude = t - 1
        extreme_right = a2 + aa
        z = lerp2(a2,extreme_right,protrude)
    if 0 <= t and t <= 1:
        z = a1*(1-t)*(1-t)*(1-t) + c1*3*(1-t)*(1-t)*t + c2*3*(1-t)*t*t + a2*t*t*t
    return z


def stretch(e):
    #typically e is an equipot line (or a field line)
    #make the ends stick out for not missing end-intersects
    e[0] = e[0] - (e[1] - e[0])   
    e[-1] = e[-1] + (e[-1] - e[-2])


#@jit("c16(c16[:,:],c16[:,:],c16[:,:],c16[:,:],c16[:,:],c16[:,:],c16[:,:],c16[:,:],c16[:,:],c16)", nopython=True, nogil=True)
def grid2morph(pos,a1,c1,c2,a2,a1v,c1v,c2v,a2v, t : complex) -> complex:
    # this takes a grid and defines the holomorphic mapping
    # use integers j,ix for array indexing
    # use tx,ty in 0..1 to bezier&lerp2 inside cell
    i : int = math.floor(t.real)
    j : int = math.floor(t.imag)
    tx : float = t.real - i
    ty : float = t.imag - j

    _a1 : complex = a1[j,i] # the horizontal bezier segment 
    _c1 : complex = c1[j,i]
    _c2 : complex = c2[j,i]
    _a2 : complex = a2[j,i]
    _a1v : complex = a1v[j,i] # the vertical bezier segment
    _c1v : complex = c1v[j,i]
    _c2v : complex = c2v[j,i]
    _a2v : complex = a2v[j,i]
    _a1_up : complex = a1[j - 1,i] # the horizontal at the cell's ceiling
    _c1_up : complex = c1[j - 1,i]
    _c2_up : complex = c2[j - 1,i]
    _a2_up : complex = a2[j - 1,i]
    _a1v_rhs : complex = a1v[j,i + 1] # the vertical at the cell's right-hand side vertical
    _c1v_rhs : complex = c1v[j,i + 1]
    _c2v_rhs : complex = c2v[j,i + 1]
    _a2v_rhs : complex = a2v[j,i + 1]

    #old algorithm
    a1_lifted = _bezier(_a1v,_c1v,_c2v,_a2v,ty)                       # the vertical bezier through t has worked
    a2_lifted = _bezier(_a1v_rhs,_c1v_rhs,_c2v_rhs,_a2v_rhs,ty)       # the rhs vertical has worked
    c1_lifted = lerp2(_c1,_c1_up,ty)                                  # still doubts on the lerping
    c2_lifted = lerp2(_c2,_c2_up,ty)                                  # now we have a lifted horizontal bezier
    z : complex = _bezier(a1_lifted,c1_lifted,c2_lifted,a2_lifted,tx) # put that into action

    # new algorithm (bilinear interpolation):
    # two lines of lerp after (protruding) bezier, then intersect 
    horizontal = []
    for t in np.arange(-.5, 1.5, 0.1):
        bt = _protruding_bezier(_a1,_c1,_c2,_a2, t)
        bt_up = _protruding_bezier(_a1_up,_c1_up,_c2_up,_a2_up,t)
        p = lerp2(bt, bt_up, ty)
        horizontal.append(p)

    vertical = []
    for t in np.arange(-.5, 1.5, 0.1):
        bt = _protruding_bezier(_a1v,_c1v,_c2v,_a2v, t)
        bt_rhs = _protruding_bezier(_a1v_rhs,_c1v_rhs,_c2v_rhs,_a2v_rhs,t)
        q = lerp2(bt, bt_rhs, tx)
        vertical.append(q)

    zz : complex = intersect(vertical,horizontal)
    
    if zz is None or math.isnan(z.real): 
        zz = _a1 #GRRRR
        ##print("grid2morph: missed intersect")
    return zz
