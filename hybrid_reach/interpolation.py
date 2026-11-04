"""Local CPU multilinear values and interpolated nodal finite differences.

Only the containing cell's corners and derivative stencils are gathered. Large
memory-mapped value histories need no materialized full-grid gradient arrays.
"""
from itertools import product
import numpy as np


class LocalInterpolator:
    def __init__(self,coordinates):
        self.coordinates=tuple(np.asarray(v,dtype=float) for v in coordinates)
        if any(v.ndim!=1 or len(v)<3 or not np.isfinite(v).all() or np.any(np.diff(v)<=0) for v in self.coordinates):
            raise ValueError('Expected increasing finite axes with at least three nodes')
        self.shape=tuple(len(v) for v in self.coordinates)
        self.corners=np.array(list(product((0,1),repeat=len(coordinates))),dtype=int)

    def sample(self,field,point,derivative_axes=()):
        if field.shape!=self.shape:raise ValueError('Field and axes do not match')
        point=np.asarray(point,dtype=float)
        if point.shape!=(len(self.coordinates),) or not np.isfinite(point).all():raise ValueError('Expected a finite point')
        if any(x<v[0] or x>v[-1] for x,v in zip(point,self.coordinates)):raise ValueError('Query outside computational domain')
        lower=np.array([np.clip(np.searchsorted(v,x,side='right')-1,0,len(v)-2) for x,v in zip(point,self.coordinates)])
        fraction=np.array([(x-v[i])/(v[i+1]-v[i]) for x,v,i in zip(point,self.coordinates,lower)])
        indices=lower+self.corners
        weights=np.prod(np.where(self.corners,fraction,1-fraction),axis=1)
        corners=field[tuple(indices.T)]
        value=float(weights@corners)
        derivatives=[]
        for axis in derivative_axes:
            v=self.coordinates[axis];i=indices[:,axis]
            # Quadratic three-node derivative weights at each corner. This is
            # np.gradient(..., edge_order=2), including nonuniform axis spacing.
            center=np.clip(i,1,len(v)-2);nodes=np.stack([center-1,center,center+1],axis=-1)
            x=v[i];a,b,c=(v[nodes[:,k]] for k in range(3))
            coefficients=np.stack([(2*x-b-c)/((a-b)*(a-c)),
                (2*x-a-c)/((b-a)*(b-c)),(2*x-a-b)/((c-a)*(c-b))],axis=-1)
            nodal=np.zeros(len(indices))
            for k in range(3):
                stencil=indices.copy();stencil[:,axis]=nodes[:,k]
                # Subtract the corner first: constant fields have exactly zero
                # derivative, without cancellation of large absolute values.
                nodal+=coefficients[:,k]*(field[tuple(stencil.T)]-corners)
            derivatives.append(float(weights@nodal))
        return value,np.array(derivatives)
