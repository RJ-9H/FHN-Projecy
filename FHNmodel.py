import numpy as np
from abc import ABC, abstractmethod 

#Base class to be inherited by regular FHN and mass conserved FHN
class FHNBase(ABC):
    def __init__(self, a= 0.7 ,b = 0.8, epsilon = 0.08, Du =1.0 , Dv = 0.5):
        self.a = a
        self.b = b 
        self.epsilon = epsilon 
        self.Du = Du 
        self.Dv = Dv

    def f(self, u,v):
        #equation 10 
        return u - u**3/3 -v
    
    def g(self, u, v):
        #equation 11
        return u + self.a - self.b * v 
    
    def laplacian(self, field, dx):
        #numerical solution of laplacian, finite difference approx, 1D, periodic 
        #d2u/dx2 = (u_{i+1} -2 u_i + u_{i-1} / dx^2)
        return (np.roll(field, -1) -2 * field + np.roll(field, 1) / dx**2)
    
    @abstractmethod
    def step(self, dx, dt, x, t):
        #to be inherited by other methods 
        pass
 
    #This for naming and displaying information when doing plots
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"a= {self.a}, b = {self.b}, epsilon = {self.epsilon,}, "
                f"Du = {self.Du}, Dv = self.Dv)")
     

class RegularFHN(FHNBase):
    #du/dt (partial derivative)= f(u,v) + Du * laplacian u
    #dv/dt (partial derivative) =epsilon g(u,v) + Dv * lapclacian v
    def step(self, u, v, dx, dt):
        lap_u = self.laplacian(u, dx)
        lap_v = self.laplacian(v, dx)

        dudt = self.f(u,v) + self.Du * lap_u

        dvdt = self.epsilon * self.g(u,v) + self.Dv * lap_v

        u_new = dudt * dt + u
        v_new = dvdt * dt + v
        return u_new, v_new
    
class MassConservedFHN(FHNBase):
    def DoubleLapl(self, field, dx):
        #taking the laplaciantwice
        return self.laplacian(self.laplacian(field, dx))
    
    def step(self, u, v, dx, dt):
        lap_u = self.laplacian(u, dx)
        lap_v = self.laplacian(v, dx)

        lap_f = self.laplacian(self.f(u,v), dx)
        lap_g = self.laplacian(self.g(u,v), dx)

        bilap_u = self.DoubleLapl(u, dx)
        bilap_v = self.DoubleLapl(v, dx)

        dudt = -(lap_f + self.Du * bilap_u)
        dvdt = -(self.epsilon * lap_g + self.Dv * bilap_v)

        u_new = u + dudt * dt
        v_new = v + dvdt * dt