### Some routines to check whether a point is within a polygon.
### Works for convex polygons only. Be Warned.


import numpy as np

class Point(object):
    ''' A class for points, having an x and y coordinate.

    supports adding, subtracting of other points
    dividing/multiplying by a scalar (for scaline)
    and calculating the length
    
    '''
    def __init__(self, x,y):
        self.x = x
        self.y = y

    def __repr__(self):
        s = "Point(%f, %f)"%(self.x, self.y)
        return s
        
    def __sub__(self, p):
        delta = Point(self.x - p.x,
                      self.y - p.y)
        return delta

    def __add__(self, p):
        p = Point(self.x + p.x,
                  self.y + p.y)
        return p

    def __mul__(self, f):
        p = Point(self.x*f, self.y*f)
        return p
    
    def __truediv__(self, r):
        p = Point(self.x/r, self.y/r)
        return p
    
    @property
    def modulus(self):
        r = self.x**2 + self.y**2
        return r**0.5

class LineSegment(object):
    ''' Class that defines a line segment based on a parameterisation

    p0 is base point
    p1 is second vertex

    self.delta is Point-like and denotes the direction vector
    
    A linesegment can be intersected with another line segment
    '''
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1
        self.delta = self.__compute_delta()
        self.length = (p1-p0).modulus

    def __repr__(self):
        s = "LineSegment %s - > %s"%(self.p0, self.p1)
        return s
    
    def __compute_delta(self):
        delta = self.p1-self.p0
        delta=delta/delta.modulus
        return delta

    def get_point(self, Lambda):
        p = self.p0 + self.delta*Lambda*self.length
        return p
    
    def solve(self, ls):
        try:
            lambda_1 = (ls.p0.y-self.p0.y) + ls.delta.y/ls.delta.x*(self.p0.x-ls.p0.x)
            lambda_1 /= self.delta.y-ls.delta.y/ls.delta.x*self.delta.x
            lambda_2 = (self.p0.x-ls.p0.x)/ls.delta.x + lambda_1*self.delta.x/ls.delta.x
        except ZeroDivisionError:
            try:
                lambda_1 = (self.p0.x-ls.p0.x)/self.delta.x
                lambda_2 = (self.p0.y-ls.p0.y+lambda_1*self.delta.y)/ls.delta.y
            except ZeroDivisionError:
                lambda_1=None
                lambda_2=None
        return lambda_1, lambda_2
    
    def intersect_with_linesegment(self, linesegment):
        lambda_1, lambda_2 = self.solve(linesegment)
        if lambda_1 is None and lambda_2 is None:
            return None, None
        else:
            return lambda_1/self.length, lambda_2/linesegment.length

    

    
class Triangle(object):
    def __init__(self, v0, v1, v2):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        #pl.plot([v0.x, v1.x, v2.x,v0.x],
        #        [v0.y, v1.y, v2.y,v0.y])
        #pl.draw()
        #input()
                

    def contains(self, p):
        # the idea is as follows:
        # for each edge of a triangle:
        #     the endpoints of the edge define the line segment
        #     a second linesegemnt is defined by the centre of gravity and the point p
        #     if lambda_2, the running parameter of the second line > 1, then
        #     p is on the same side of the edge as the centre of gravity
        #     if this is true for all edges, p is inside the triangle
        points0 = [(self.v0, self.v1),
                   (self.v1, self.v2),
                   (self.v2, self.v0)]
        points1 = [(self.v2, p),
                   (self.v0, p),
                   (self.v1, p)]

        inside=True
        for p,q in zip(points0, points1):
            ls1 = LineSegment(*p)
            ls2 = LineSegment(*q)
            s,r = ls1.intersect_with_linesegment(ls2)
            if r is None:
                continue
            if r<1:
                inside=False
        return inside
    
class Polygon(object):
    def __init__(self, *v):
        self.vertices = v
        self.cog = np.mean(v)
        self.triangles = self.__create_triangles(v)

    def __create_triangles(self, v):
        V = [_v for _v in v] + [v[0]]
        t = []
        for i in range(len(V)-1):
            t.append(Triangle(V[i], V[i+1], self.cog))
        return t
    
    def contains(self, p):
        # p is inside a convex polygon if it is at least in 1 of the triangles
        # triangles are defined by the edges of the polygon and the centre of gravity
        # this means it works for convec polygons only.
        # this is not checked...
        s = [t.contains(p) for t in self.triangles]
        return sum(s)>=1
    
        
if __name__ == "__main__":
    import pylab as pl
    import random
    p0=Point(1,1)
    p1=Point(1,3)
    p2=Point(4,4)
    p3=Point(4,1)
    p4=Point(3,0.5)

    P = Polygon(p0, p1, p2, p3, p4)

    x = [_p.x for _p in [p0, p1, p2, p3, p4, p0]]
    y = [_p.y for _p in [p0, p1, p2, p3, p4, p0]]
    pl.plot(x,y,'o-')

    for i in range(1000):
        x=random.random()*5
        y=random.random()*5
        if P.contains(Point(x,y)):
            pl.plot([x],[y],'g.')
        else:
            pl.plot([x],[y],'r.')
    pl.draw()
