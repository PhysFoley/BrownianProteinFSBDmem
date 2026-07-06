import numpy as np
from scipy.stats import norm
from scipy import fft
import hoomd
import json

class Membrane:
    def __init__(self,simulation,N,bonded_part_inds,dt,kbond_pm,pm_rest_len,kappa=20.0,sigma=0.0,eta=24.0):
        if N%2 == 0:
             print("Error: membrane lattice N must be odd")
             quit()
        self.sim = simulation
        self.L = simulation.state.box.Lx # assumes Lx=Ly
        self.N = N
        self.a = self.L/N
        self.kappa = kappa
        self.sigma = sigma
        self.kbond_pm = kbond_pm
        self.pm_rest_len = pm_rest_len
        self.bonded_part_inds = bonded_part_inds
        self.dt = dt

        # list of independent complex mode indices
        self.ci = []
        self.ci += [[m,n] for m in np.arange((1-N)/2,1+(N-1)/2) for n in np.arange(1,1+(N-1)/2)]
        self.ci += [[m,0] for m in np.arange(1,1+(N-1)/2) ]

        # this array will contain 1 in each entry corresponding
        # to the complex degs of freedom above, and a zero everywhere else
        self.zero_block = np.zeros((N,N),dtype=np.complex128)
        for inds in self.ci:
            self.zero_block[self.index(*inds)] = 1.

        # the mode grid
        self.k = (2*np.pi/self.L)*np.array([ [[m,n] for n in np.arange((1-N)/2,1+(N-1)/2)] for m in np.arange((1-N)/2,1+(N-1)/2)])

        # |k|
        magk = (2*np.pi/self.L)*np.array([ [np.sqrt((m**2) + (n**2)) for n in np.arange((1-N)/2,1+(N-1)/2)] for m in np.arange((1-N)/2,1+(N-1)/2)])

        # k^2
        self.k2 = np.power(magk,2.)

        # k^4
        self.k4 = np.power(magk,4.)

        # the actual amplitudes h_k (dynamical variables to be updated throughout simulation)
        self.hk = np.zeros((N,N),dtype=np.complex128)

        orig_settings = np.seterr(divide="ignore") # prevent next line from printing div by zero warning, this is handled immediately after
        # Lambda oseen parameter, see Lin & Brown PRL
        self.lamk = 1./(4.*eta*magk)
        np.seterr(**orig_settings) # restore original error settings

        self.lamk[self.index(0,0)] = 0.0 # don't blow up the trivial zero mode NOTE: this could be non-zero, but should not matter

        self.ext_force = np.zeros((self.N,self.N))

        self.h = np.zeros((self.N,self.N))

        self.mem_force = MembraneForce(self)
        simulation.operations.integrator.forces.append(self.mem_force)

    # for indexing into np array using m,n values
    def index(self,m,n):
        return int(m+((self.N-1)/2)),int(n+((self.N-1)/2))

    # get the indices of the spatial bin
    # corresponding to the membrane patch at x,y
    def get_h_inds(self,x,y):
        xind = int(((x+(self.a/2)+(self.L/2))%self.L)/self.a)
        yind = int(((y+(self.a/2)+(self.L/2))%self.L)/self.a)
        return (xind,yind)

    # propagate one membrane timestep
    def update(self):
        # forces from the hamiltonian: very simple (diagonal) for small gradient Helfrich!
        Fk = -self.kappa*self.k4*self.hk - self.sigma*self.k2*self.hk
        
        # add in fourier transform of exeternal forces
        Fk += fft.fftshift(fft.fft2(self.ext_force))*self.L/(self.N*self.N)
        
        # now, for the noise, which is unfortunately not as simple
        # first, just making an NxN matrix of complex Gaussians (all real and imag parts are iid)
        noise = (norm.rvs(size=(self.N,self.N),scale=np.sqrt(self.lamk*self.dt)) + (1.j)*norm.rvs(size=(self.N,self.N),scale=np.sqrt(self.lamk*self.dt)))
        
        # update the modes
        self.hk += self.lamk*Fk*self.dt + noise
        
        # now, remove the excess dofs
        self.hk *= self.zero_block
        # and finally fill in the negative modes with the complex conjugates
        self.hk += np.flip(np.flip(np.conj(self.hk),axis=1),axis=0)
    
    def calc_forces(self,snap):
        # get real-space h
        self.h = fft.ifft2(fft.ifftshift(self.hk))*self.N*self.N/self.L
        self.ext_force = np.zeros((self.N,self.N))
        forces = np.zeros((self.sim.state.N_particles,3)) # for storing calculated forces on bonded particles

        for ind in self.bonded_part_inds:
            i = snap.particles.rtag[ind] # reverse tag lookup, since hoomd rearranges particles in memory
            pos = snap.particles.position[i]
            h_inds = self.get_h_inds(pos[0],pos[1])
            # force on the particle
            disp = pos[2]-np.real(self.h[*h_inds])
            forces[i,:] = np.array([0.0, 0.0, -np.sign(disp)*self.kbond_pm*(np.abs(disp) - self.pm_rest_len)])
            #             mem_force = -part_force / patch area
            self.ext_force[*h_inds] += -forces[i,2] / (self.a*self.a)
        return forces

class MembraneForce(hoomd.md.force.Custom):
    def __init__(self,membrane):
        super().__init__() # aniso=True if we want to apply torques
        self.mem = membrane
    
    def set_forces(self, timestep):
        with self.cpu_local_force_arrays as force_arrays:
            with self._state.cpu_local_snapshot as snap:
                force_arrays.force[:] = self.mem.calc_forces(snap)
        self.mem.update()

class MembraneWriter(hoomd.custom.Action):
    def __init__(self,membrane,filename):
        self.mem = membrane
        self.fname = filename
        self.file = open(filename,'w')
        metadata = {
            "L": membrane.L,
            "N": membrane.N,
            "a": membrane.a,
            "dt": membrane.dt
        }
        self.file.write("{\n")
        self.file.write(f'"metadata": {json.dumps(metadata)},\n')
        self.file.write('"data": [\n')
        self.firstWrite = True # to tell the writer not to put a comma on first write
    
    def finalize(self):
        self.file.write("\n]\n}\n")
        self.file.flush()
        self.file.close()
    
    def act(self,timestep):
        if not self.firstWrite:
            self.file.write(',\n')
        m_str = json.dumps(np.real(self.mem.h).tolist())
        self.file.write(m_str)
        self.file.flush()
        self.firstWrite = False
