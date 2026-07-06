import json
import numpy as np
import hoomd
import gsd.hoomd
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from membrane import MembraneForce

from pprint import pprint

# flat clath coords: ccw from above
monomer_coords = np.array([
    [0.0000,  0.0000, 0.0000], # com
    [-5.1962,-3.0000, 0.0000], # cd1
    [5.1962, -3.0000, 0.0000], # cd2
    [0.0000,  6.0000, 0.0000], # cd3
    [0.0000,  0.0000, 1.0000]  # normal vector
])

m_leg = 0.0834 # mass unit ~2.578 MDa, 215 kDa per heavy chain (190) + light chain (25)

# moment of inertia for flat clath
I = np.zeros((3,3))
for leg in monomer_coords[1:4,:]:
        I += m_leg*(np.dot(leg,leg)*np.identity(3) - np.outer(leg,leg))

def find_bonds(mols,bssites,bond_cutoff=6.0):
    bs_tree = KDTree(bssites)

    bonds = []

    pairs = bs_tree.query_pairs(bond_cutoff)

    for pair in pairs:
        mol_ind = pair[0] // 3
        site = bssites[pair[0]]
        com = mols[mol_ind][0]

        mol2_ind = pair[1] // 3
        site2 = bssites[pair[1]]
        com2 = mols[mol2_ind][0]

        if mol_ind == mol2_ind:
            continue

        v1 = site - com
        s1 = site2 - site
        cos1 = np.dot(v1,s1)/(np.linalg.norm(v1)*np.linalg.norm(s1))

        v2 = site2 - com2
        s2 = -s1
        cos2 = np.dot(v2,s2)/(np.linalg.norm(v2)*np.linalg.norm(s2))

        relsite1_ind = pair[0]%3
        relsite2_ind = pair[1]%3

        # the commented-out if was to prevent the creation of highly strained
        # bonds when postulating bonds from already existing structures. It
        # doesn't apply here
        # if (cos1 > np.sqrt(2.)/2.0) and (cos2 > np.sqrt(2.)/2.0):
        bonds.append([(mol_ind,relsite1_ind),(mol2_ind,relsite2_ind)])
    
    return bonds

def pucker_coords(pucker_angle):
    rot_angle = (pucker_angle-90.0)*np.pi/180.0
    zhat = np.array([0.,0.,1.])
    new_coords = np.copy(monomer_coords)
    # rotate bs site positions based on pucker angle
    for i in range(1,4):
        axis = np.cross(zhat,monomer_coords[i])
        axis /= np.linalg.norm(axis)
        Rp = Rotation.from_rotvec(rot_angle*axis)
        new_coords[i] = Rp.apply(monomer_coords[i])
    return new_coords

# note: user is responsible for including z-offset in initial coords
def create_arbitrary_initial_frame(pucker_angle,cutoff,com_coords,rot_vectors,box_l,verbose=True):
    tri_coords = pucker_coords(pucker_angle)

    coms = []
    constituents = []
    mols = []
    bssites = []
    
    for c,r in zip(com_coords,rot_vectors):
        coms.append(c)
        R = Rotation.from_rotvec(r)
        mols.append([c + R.apply(pos) for pos in tri_coords])
        for k in range(1,4):
            bssites.append(mols[-1][k])
        for k in range(1,5):
            constituents.append(mols[-1][k])

    bssites = np.array(bssites)
    bonds = find_bonds(mols,bssites,bond_cutoff=cutoff)

    frame = gsd.hoomd.Frame()
    frame.particles.types = ["clat","bs","n"]

    rot_quats = [Rotation.from_rotvec(r).as_quat(scalar_first=True) for r in rot_vectors]

    N_clat = len(mols)
    N_cons = len(constituents)

    frame.particles.N = N_clat + N_cons
    frame.particles.position = coms + constituents
    frame.particles.typeid = [0]*N_clat + [1,1,1,2]*N_clat
    frame.particles.mass = [3*m_leg]*N_clat + [1]*N_clat*4 # masses of constituent particles are not used
    frame.particles.moment_inertia = [np.diag(I)]*N_clat*5 # same for moments of inertia
    frame.particles.orientation = rot_quats + [(1,0,0,0)]*N_clat*4
    frame.configuration.box = [*box_l, 0, 0, 0]

    # the body index for all constituent particles needs to be set to the index of its corresponding central particle
    frame.particles.body = [i for i in range(N_clat)] + [i//4 for i in range(N_clat*4)]

    if verbose:
        print(f"Number of clathrin triskelia: {N_clat}")
        print(f"Number of particles: {frame.particles.N}\n")
        print(f"Particle positions:\n{frame.particles.position}\n")
        print(f"Particle type ids: {frame.particles.typeid}\n")
        print(f"Particle masses: {frame.particles.mass}\n")
        print(f"Particle body indices:\n{frame.particles.body}\n")
        print(f"Particle inertia tensors: {frame.particles.moment_inertia}\n")
        print(f"Particle orientations:\n{frame.particles.orientation}\n")
        print(f"Simulation box geometry: {frame.configuration.box}")

    # assign bonds/angles/dihedrals
    frame.bonds.N = len(bonds)
    frame.bonds.types = ["sigma"]
    bond_groups = []
    bond_ids = []

    frame.angles.N = len(bonds)*2
    frame.angles.types = ["theta1","theta2"]
    angle_groups = []
    angle_ids = []

    frame.dihedrals.N = len(bonds)
    frame.dihedrals.types = ["omega"]
    dihedral_groups = []
    dihedral_ids = []

    for b in bonds:
        # sigma harmonic bond:
        #   start from N_clat since 0 to N_clat-1 are the com particles
        #   then, add on the offset for which clathrin we're in,
        #   then finally add the site number offset
        particle0_ind = N_clat + b[0][0]*4 + b[0][1]
        particle1_ind = N_clat + b[1][0]*4 + b[1][1]
        bond_groups.append([particle0_ind,particle1_ind])
        bond_ids.append(0) # sigma

        # theta angle potentials:
        # similar logic to above, but three-body potential
        # involving the CoMs
        particle0_ind = b[0][0] # CoM
        particle1_ind = N_clat + b[0][0]*4 + b[0][1]
        particle2_ind = N_clat + b[1][0]*4 + b[1][1]
        angle_groups.append([particle0_ind,particle1_ind,particle2_ind])
        angle_ids.append(0) # theta1

        particle0_ind = b[1][0] # CoM
        particle1_ind = N_clat + b[1][0]*4 + b[1][1]
        particle2_ind = N_clat + b[0][0]*4 + b[0][1]
        angle_groups.append([particle0_ind,particle1_ind,particle2_ind])
        angle_ids.append(1) # theta2

        # omega dihedral
        # four-body potential involving normal vector "particles"
        particle0_ind = N_clat + b[0][0]*4 + 3
        particle1_ind = b[0][0]
        particle2_ind = b[1][0]
        particle3_ind = N_clat + b[1][0]*4 + 3
        dihedral_groups.append([particle0_ind,particle1_ind,particle2_ind,particle3_ind])
        dihedral_ids.append(0) # omega

    frame.bonds.typeid = bond_ids
    frame.bonds.group = bond_groups

    frame.angles.typeid = angle_ids
    frame.angles.group = angle_groups

    frame.dihedrals.typeid = dihedral_ids
    frame.dihedrals.group = dihedral_groups

    return frame

def setup_clathrin_integrator(simulation,params):
    tri_coords = pucker_coords(params["pucker_angle"])
    clathrin_rigid_body = {
        "constituent_types": ["bs","bs","bs","n"],
        "positions": tri_coords[1:],
        "orientations": [(1,0,0,0)]*4
    }

    rigid = hoomd.md.constrain.Rigid()
    rigid.body["clat"] = clathrin_rigid_body

    # create integrator
    simulation.operations.integrator = hoomd.md.Integrator(dt=params["timestep"], integrate_rotational_dof=True)
    simulation.operations.integrator.rigid = rigid

    # create bond/angle/dihedral types
    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params["sigma"] = dict(k=params["kr"], r0=params["sigma"])

    angle = hoomd.md.angle.Harmonic()
    angle.params["theta1"] = dict(k=params["ktheta"], t0=params["theta1"])
    angle.params["theta2"] = dict(k=params["ktheta"], t0=params["theta2"])

    dihedral = hoomd.md.dihedral.Periodic()
    dihedral.params["omega"] = dict(k=params["komega"], d=-1, n=1, phi0=params["omega"])

    # add bond types to simulation
    simulation.operations.integrator.forces.append(harmonic)
    simulation.operations.integrator.forces.append(angle)
    simulation.operations.integrator.forces.append(dihedral)

    # finish setting up integrator; we only want to integrate rigid center dofs (and free, but we don't have any of those)
    rigid_centers_and_free = hoomd.filter.Rigid(("center", "free"))
    brownian = hoomd.md.methods.Brownian(
        filter=rigid_centers_and_free,
        kT=1.0,
        default_gamma=params["gamma"],
        default_gamma_r=[params["gamma_r"]]*3  #(gamma_r,gamma_r,gamma_r)
    )
    simulation.operations.integrator.methods.append(brownian)

# takes two vectors, returns scipy rotation that takes a to b
def rot_a_to_b(a,b):
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    rotvec = np.cross(a,b)/(norma*normb)
    sintheta = np.linalg.norm(rotvec)
    if np.dot(a,b) < 0.0:
        theta = np.pi - np.arcsin(sintheta)
    else:
        theta = np.arcsin(sintheta)
    return Rotation.from_rotvec(theta*rotvec/sintheta)

# routine to fold unwrapped coordinates back into the hoomd
# simulation box, which goes from [-L/2,L/2)
def folded_coords(p,box):
    return ((p+(box/2))%box)-(box/2)

class ProgressWriter(hoomd.custom.Action):
    def __init__(self,total_iters):
        self.total_iters = total_iters

    def act(self,timestep):
        percent = 100*timestep/self.total_iters
        print(f"Progress: {percent:.2f}%")

def parse_nerdss_mol_file(filename):
    mol = {}
    mol["name"] = filename[:-4]
    mol["sites"] = {}
    mol["D"] = 0.0
    mol["Dr"] = 0.0
    with open(filename,"r") as file:
        for line in file:
            if len(line.strip()) == 0 or line.strip()[0] == "#":
                continue # skip comment lines or empty lines
            
            if "=" in line:
                lr = line.strip().split("=") # split into left and right sides of equality
                if lr[0].strip() == "D":
                    arr = json.loads(lr[1]) # use json to parse array string
                    mol["D"] = float(arr[0]) # currently I only support one isotropic diffusion const
                elif lr[0].strip() == "Dr":
                    arr = json.loads(lr[1]) # use json to parse array string
                    mol["Dr"] = float(arr[0]) # currently I only support one isotropic diffusion const
            else:
                tokens = line.strip().split()
                if (len(tokens) == 4) and tokens[0].isalnum():
                    try:
                        name = tokens[0]
                        coords = np.array([float(t) for t in tokens[1:]])
                        mol["sites"][name] = coords
                    except:
                        # either this isn't actually a coordinate line or it's malformed
                        pass
    return mol

# Assuming a NERDSS frame given in the following format:
# nerdss_frame.json:
    # {
        # "names": ["clat","clat",...],
        # "coords": [
            # [0.0 ,0.0 ,0.0],
            # [..., ..., ...],
            # ...
        # ],
        # "rotations": [
            # [..., ..., ...],
            # ...
        # ],
        # "bond_types": {
            # "bs1-bs1": {"mol1":"clat", "mol2":"clat", "n1": [.., .., ..], "n2": [.., .., ..], "sigma":..., "theta1":..., "k_sigma":..., ...},
            # "bs1-bs2": {...},
            # ...
        # },
        # "bonds": [
            # {"molindex1":0, "molindex2":1, "site1": "bs1", "site2": "bs1", "type": "bs1-bs1"},
            # ...
        # ],
        # "membrane_bonds": {
            # "bind_l": 15.0,
            # "k_mem": 1000.0,
            # "bonds": [{"molindex":0, "site": "pm1"}, ...]
        # }
    # }
#
# Along with matching *.mol files
# NOTE: COM *must* be the first site listed in the .mol file
# NOTE: avg_gamma can be useful if the system becomes numerically
#       unstable due to large differences in molecule mobility
def setup_simulation_from_nerdss(simulation,params,avg_gamma=False):
    verbose = params["verbose"]
    if verbose: print(f"Reading nerdss frame data file {params['complex_file']}...")
    
    with open(params["complex_file"],"r") as file:
        data = json.load(file)[0]

    if verbose: print("File loaded.")

    brownian = hoomd.md.methods.Brownian(
        filter=hoomd.filter.Rigid(("center", "free")),
        kT=1.0
    )

    # create a gsd frame object which we will populate with the nerdss snapshot data
    frame = gsd.hoomd.Frame()
    particle_types = []
    
    mol_templates = {}
    gamma_vals = {}
    gamma_r_vals = {}
    for name in data["names"]:
        if name not in mol_templates:
            mol_fname = name + ".mol"
            mol_templates[name] = parse_nerdss_mol_file(mol_fname)

            # register the appropriate diffusion constants in the integrator
            # gamma = kT/D, kT=1 is our energy unit, and we rescale time from us to ns
            brownian.gamma[name] = (1.0e3)*1.0/mol_templates[name]["D"]
            brownian.gamma_r[name] = [(1.0e3)*1.0/mol_templates[name]["Dr"]]*3

            gamma_vals[name] = brownian.gamma[name]
            gamma_r_vals[name] = brownian.gamma_r[name][0]

            particle_types.append(name)
            for key in mol_templates[name]["sites"]:
                if (key != "COM") and (key not in particle_types):
                    particle_types.append(key)
    if avg_gamma:
        for key in mol_templates:
            brownian.gamma[key] = np.mean(list(gamma_vals.values()))
            brownian.gamma_r[key] = [np.mean(list(gamma_r_vals.values()))]*3
    
    harmonic = hoomd.md.bond.Harmonic()
    angle = hoomd.md.angle.Harmonic()
    dihedral = hoomd.md.dihedral.Periodic()
    
    # these lists help with reverse index lookup
    bond_types = [] # two-body sigma bonds
    angle_types = [] # three-body theta angle bonds
    dihedral_types = [] # four-body dihedral angle bonds
    
    for bond_name,bond in data["bond_types"].items():
        n1_name = bond_name + "-n1"
        n2_name = bond_name + "-n2"
        mol_templates[bond["mol1"]]["sites"][n1_name] = mol_templates[bond["mol1"]]["sites"]["COM"] + np.array(bond["n1"])
        mol_templates[bond["mol2"]]["sites"][n2_name] = mol_templates[bond["mol2"]]["sites"]["COM"] + np.array(bond["n2"])
        print(f"n1_name: {n1_name}, coords: {mol_templates[bond['mol1']]['sites'][n1_name]}")
        print(f"n2_name: {n2_name}, coords: {mol_templates[bond['mol2']]['sites'][n2_name]}")
        particle_types.append(n1_name)
        particle_types.append(n2_name)

        bond_types.append("sigma_" + bond_name)
        harmonic.params[bond_types[-1]] = dict(k=bond["k_sigma"], r0=bond["sigma"])

        if bond["theta1"] is not None:
            angle_types.append("theta1_" + bond_name)
            angle.params[angle_types[-1]] = dict(k=bond["k_theta1"], t0=bond["theta1"])

        if bond["theta2"] is not None:
            angle_types.append("theta2_" + bond_name)
            angle.params[angle_types[-1]] = dict(k=bond["k_theta2"], t0=bond["theta2"])

        if bond["phi1"] is not None:
            dihedral_types.append("phi1_" + bond_name)
            # NOTE: NERDSS phi angles are defined differnetly from standard dihedrals, pi shift
            dihedral.params[dihedral_types[-1]] = dict(k=bond["k_phi1"], d=-1, n=1, phi0=np.pi+bond["phi1"])

        if bond["phi2"] is not None:
            dihedral_types.append("phi2_" + bond_name)
            # NOTE: NERDSS phi angles are defined differnetly from standard dihedrals, pi shift
            dihedral.params[dihedral_types[-1]] = dict(k=bond["k_phi2"], d=-1, n=1, phi0=np.pi+bond["phi2"])

        if bond["omega"] is not None:
            dihedral_types.append("omega_" + bond_name)
            dihedral.params[dihedral_types[-1]] = dict(k=bond["k_omega"], d=-1, n=1, phi0=bond["omega"]%(2*np.pi))

    # print out complete mol templates and site keys for debugging
    print("MOLECULE TEMPLATES")
    for key in mol_templates:
        print(key+":")
        pprint(mol_templates[key])
    
    rigid = hoomd.md.constrain.Rigid()
    for molname,template in mol_templates.items():
        template["rigid"] = {
            "constituent_types": [key for key in template["sites"] if key != "COM"],
            "positions": [template["sites"][key] for key in template["sites"] if key != "COM"],
            "orientations": [(1,0,0,0) for key in template["sites"] if key != "COM"]
        }
        rigid.body[molname] = template["rigid"]

    # add particle types to current frame
    frame.particles.types = particle_types
    
    position = []
    typeid = []
    mass = [] # NOTE: not used for integration, but hoomd still calculates a velocity using this
    moment_inertia = [] # NOTE: similar to above, not used for integration but hoomd calculates an angular velocity using this
    orientation = []
    body = []

    com_index = [] # index this with our json index to get hoomd index
    
    for i in range(len(data["names"])):
        com = np.array(data["coords"][i])
        position.append(com)
        mass.append(1) # NOTE: revisit this, per comments above
        moment_inertia.append(np.ones(3)) # NOTE: revisit this, per comments above
        body_id = len(body)
        body.append(body_id)
        com_index.append(body_id)

        typeid.append(particle_types.index(data["names"][i]))

        r = Rotation.from_quat(data["rotations"][i],scalar_first=True)
        orientation.append(data["rotations"][i])
        
        for sitename in mol_templates[data["names"][i]]["sites"]:
            if sitename == "COM":
                continue
            v = np.array(mol_templates[data["names"][i]]["sites"][sitename]) # - com
            vp = r.apply(v)
            position.append(com + vp)
            mass.append(1) # NOTE: revisit this, per comments above
            moment_inertia.append(np.ones(3)) # NOTE: revisit this, per comments above
            orientation.append((1,0,0,0))
            body.append(body_id)
            typeid.append(particle_types.index(sitename))

    frame.particles.N = len(typeid)
    frame.particles.position = position
    frame.particles.typeid = typeid
    frame.particles.mass = mass # masses of constituent particles are not used
    frame.particles.moment_inertia = moment_inertia # same for moments of inertia
    frame.particles.orientation = orientation
    frame.particles.body = body
    frame.configuration.box = [*params["box_l"], 0, 0, 0]

    if verbose:
        print("Populated particle data.")
        print(f"Number of particles: {frame.particles.N}\n")
        print(f"Particle positions:\n{frame.particles.position}\n")
        print(f"Particle type ids: {frame.particles.typeid}\n")
        print(f"Particle masses: {frame.particles.mass}\n")
        print(f"Particle body indices:\n{frame.particles.body}\n")
        print(f"Particle inertia tensors: {frame.particles.moment_inertia}\n")
        print(f"Particle orientations:\n{frame.particles.orientation}\n")
        print(f"Simulation box geometry: {frame.configuration.box}")

    bond_groups = []
    bond_ids = []
    angle_groups = []
    angle_ids = []
    dihedral_groups = []
    dihedral_ids = []
    for b in data["bonds"]:
        btype = data["bond_types"][b["type"]]
        
        com1_ind = com_index[b["molindex1"]]
        mol1_name = data["names"][b["molindex1"]]

        com2_ind = com_index[b["molindex2"]]
        mol2_name = data["names"][b["molindex2"]]

        # this keys() shenanigans is because i need to know the order the particles were added
        # this also means that we MUST use python 3.7+, as before this keys do not guarantee insertion indexing
        site1_ind = com1_ind + list(mol_templates[mol1_name]["sites"].keys()).index(b["site1"])
        site2_ind = com2_ind + list(mol_templates[mol2_name]["sites"].keys()).index(b["site2"])

        n1_ind = com1_ind + list(mol_templates[mol1_name]["sites"].keys()).index(b["type"] + "-n1")
        n2_ind = com2_ind + list(mol_templates[mol2_name]["sites"].keys()).index(b["type"] + "-n2")

        # sigma
        bond_groups.append([site1_ind, site2_ind])
        bond_ids.append(bond_types.index("sigma_" + b["type"]))
        
        if btype["theta1"] is not None:
            angle_groups.append([com1_ind, site1_ind, site2_ind])
            angle_ids.append(angle_types.index("theta1_" + b["type"]))

        if btype["theta2"] is not None:
            angle_groups.append([com2_ind, site2_ind, site1_ind])
            angle_ids.append(angle_types.index("theta2_" + b["type"]))

        if btype["phi1"] is not None:
            dihedral_groups.append([n1_ind, com1_ind, site1_ind, site2_ind])
            dihedral_ids.append(dihedral_types.index("phi1_" + b["type"]))

        if btype["phi2"] is not None:
            dihedral_groups.append([n2_ind, com2_ind, site2_ind, site1_ind])
            dihedral_ids.append(dihedral_types.index("phi2_" + b["type"]))

        if btype["omega"] is not None:
            if (btype["phi1"] is not None) and (btype["phi2"] is not None):
                dihedral_groups.append([com1_ind, site1_ind, site2_ind, com2_ind])
                dihedral_ids.append(dihedral_types.index("omega_" + b["type"]))
            elif (btype["phi1"] is not None) and (btype["phi2"] is None):
                dihedral_groups.append([com1_ind, site1_ind, com2_ind, n2_ind])
                dihedral_ids.append(dihedral_types.index("omega_" + b["type"]))
            elif (btype["phi1"] is None) and (btype["phi2"] is not None):
                dihedral_groups.append([n1_ind, com1_ind, site2_ind, com2_ind])
                dihedral_ids.append(dihedral_types.index("omega_" + b["type"]))
            else:
                dihedral_groups.append([n1_ind, com1_ind, com2_ind, n2_ind])
                dihedral_ids.append(dihedral_types.index("omega_" + b["type"]))
    
    frame.bonds.N = len(bond_ids)
    frame.bonds.types = bond_types
    frame.bonds.group = bond_groups
    frame.bonds.typeid = bond_ids

    frame.angles.N = len(angle_ids)
    frame.angles.types = angle_types
    frame.angles.group = angle_groups
    frame.angles.typeid = angle_ids

    frame.dihedrals.N = len(dihedral_ids)
    frame.dihedrals.types = dihedral_types
    frame.dihedrals.group = dihedral_groups
    frame.dihedrals.typeid = dihedral_ids

    if verbose:
        print("\n")
        print(f"{frame.bonds.N} spring bonds")
        print(f"{frame.angles.N} angular bonds")
        print(f"{frame.dihedrals.N} dihedral bonds")

    # output initial frame for checking
    with gsd.hoomd.open(name=f"out_{params['jobname']}/init.gsd", mode="x") as f:
        f.append(frame)

    # set up the simulation state from our created gsd frame
    simulation.create_state_from_snapshot(frame)
    
    # create rigid body integrator and append the brownian dynamics method
    simulation.operations.integrator = hoomd.md.Integrator(dt=params["timestep"], integrate_rotational_dof=True)
    simulation.operations.integrator.rigid = rigid
    simulation.operations.integrator.methods.append(brownian)
    # add bond types to integrator
    simulation.operations.integrator.forces.append(harmonic)
    simulation.operations.integrator.forces.append(angle)
    simulation.operations.integrator.forces.append(dihedral)

    mem_bonded_inds = []
    for mb in data["membrane_bonds"]["bonds"]:
        mol_name = data["names"][mb["molindex"]]
        com_ind = com_index[mb["molindex"]]
        particle_ind = com_ind + list(mol_templates[mol_name]["sites"].keys()).index(mb["site"])
        mem_bonded_inds.append(particle_ind)

    params["bind_l"] = data["membrane_bonds"]["bind_l"]
    params["kbond_pm"] = data["membrane_bonds"]["k_mem"]
    
    return mem_bonded_inds

def setup_rigid_simulation_from_nerdss(simulation,params,avg_gamma=False):
    verbose = params["verbose"]
    if verbose: print(f"Reading nerdss frame data file {params['complex_file']}...")
    
    with open(params["complex_file"],"r") as file:
        data = json.load(file)[0]

    if verbose: print("File loaded.")

    brownian = hoomd.md.methods.Brownian(
        filter=hoomd.filter.Rigid(("center", "free")),
        kT=1.0
    )

    # create a gsd frame object which we will populate with the nerdss snapshot data
    frame = gsd.hoomd.Frame()
    particle_types = ["lattice"]
    
    mol_templates = {}
    gamma_vals = {}
    gamma_r_vals = {}
    for name in data["names"]:
        if name not in mol_templates:
            mol_fname = name + ".mol"
            mol_templates[name] = parse_nerdss_mol_file(mol_fname)

            particle_types.append(name)
            for key in mol_templates[name]["sites"]:
                if (key != "COM") and (key not in particle_types):
                    particle_types.append(key)

    position = [np.mean(data["coords"],axis=0)]
    typeid = [0]
    mass = [1] # NOTE: not used for integration, but hoomd still calculates a velocity using this
    moment_inertia = [np.ones(3)] # NOTE: similar to above, not used for integration but hoomd calculates an angular velocity using this
    orientation = [(1,0,0,0)]
    body = [0]

    com_index = [] # index this with our json index to get hoomd index

    D_vals = []
    D_r_vals = []
    
    for name,pos,rot in zip(data["names"],data["coords"],data["rotations"]):
        com_index.append(len(typeid))
        position.append(pos)
        typeid.append(particle_types.index(name))
        mass.append(1)
        moment_inertia.append(np.ones(3))
        orientation.append((1,0,0,0))
        body.append(0)

        D_vals.append(mol_templates[name]["D"])
        D_r_vals.append(mol_templates[name]["Dr"])

        r = Rotation.from_quat(rot, scalar_first=True)

        for sitename in mol_templates[name]["sites"]:
            if sitename == "COM":
                continue
            v = np.array(mol_templates[name]["sites"][sitename]) # - pos
            vp = r.apply(v)

            position.append(pos + vp)
            mass.append(1)
            moment_inertia.append(np.ones(3))
            orientation.append((1,0,0,0))
            body.append(0)
            typeid.append(particle_types.index(sitename))

    D_tot = 1.0/np.sum(1.0/np.array(D_vals))
    D_r_tot = np.power(np.sum(np.power(np.array(D_r_vals),-1.0/3.0)), -3.0)

    brownian.gamma["lattice"] = (1.0e3)*1.0/D_tot
    brownian.gamma_r["lattice"] = [(1.0e3)*1.0/D_r_tot]*3

    rigid = hoomd.md.constrain.Rigid()
    rigid.body["lattice"] = {
        "constituent_types": [particle_types[i] for i in typeid[1:]],
        "positions": position[1:],
        "orientations": orientation[1:]
    }

    frame = gsd.hoomd.Frame()
    frame.particles.types = particle_types
    frame.particles.N = len(typeid)
    frame.particles.position = position
    frame.particles.typeid = typeid
    frame.particles.mass = mass # masses of constituent particles are not used
    frame.particles.moment_inertia = moment_inertia # same for moments of inertia
    frame.particles.orientation = orientation
    frame.particles.body = body
    frame.configuration.box = [*params["box_l"], 0, 0, 0]

    # output initial frame for checking
    with gsd.hoomd.open(name=f"out_{params['jobname']}/init.gsd", mode="x") as f:
        f.append(frame)

    # set up the simulation state from our created gsd frame
    simulation.create_state_from_snapshot(frame)
    
    # create rigid body integrator and append the brownian dynamics method
    simulation.operations.integrator = hoomd.md.Integrator(dt=params["timestep"], integrate_rotational_dof=True)
    simulation.operations.integrator.rigid = rigid
    simulation.operations.integrator.methods.append(brownian)

    mem_bonded_inds = []
    for mb in data["membrane_bonds"]["bonds"]:
        mol_name = data["names"][mb["molindex"]]
        com_ind = com_index[mb["molindex"]]
        particle_ind = com_ind + list(mol_templates[mol_name]["sites"].keys()).index(mb["site"])
        mem_bonded_inds.append(particle_ind)

    params["bind_l"] = data["membrane_bonds"]["bind_l"]
    params["kbond_pm"] = data["membrane_bonds"]["k_mem"]

    return mem_bonded_inds
