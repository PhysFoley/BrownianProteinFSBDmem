import os
import sys
from datetime import datetime

import json

import numpy as np
from scipy.stats import uniform

import gsd.hoomd
import hoomd

from sys_setup import *
from membrane import Membrane,MembraneWriter,MembraneForce

# UNIT SYSTEM:
# length = nm
# time   = ns
# energy = kT (310K)
#
# All other units are derived, e.g.
# mass = T^2 * E / L^2 ~ 2.578 MDa

default_params = {
    "jobname": "test",
    "calc_energy": False, # whether or not to calculate potential energy each step in HOOMD
    "pucker_angle": 110.0,
    "bond_cutoff": 10.0, # nm, cutoff distance for creating bonds in initial config
    "kr": 1000.0, # spring bond strength (kT/nm^2)
    "ktheta": 1000.0, # angle bond strengths (kT/rad^2)
    "komega": 1000.0,  # dihedral bond strength (kT/rad^2)
    "simulate_membrane": True,
    "bind_l": 15.0, # nm
    "kbond_cm": 1000.0, # clathrin-membrane bond strength
    "box_l": [200.0,200.0,200.0], # nm
    "mem_N": 55, # membrane subdivision number (must be odd in current implementation)
    "kappa": 20.0, # kT, membrane bending modulus
    "mem_sigma": 0.0, # kT nm^-2, membrane surface tension
    "eta": 24.0, # cytosol viscosity
    "sigma": 5.0, # nm, clat-clat bond length
    "theta1": np.pi,
    "theta2": np.pi,
    "phi1:": np.nan,
    "phi2": np.nan,
    "omega": 0.0,
    "timestep": 0.01, # ns
    "gamma": 77.0, # kT ns nm^-2
    "gamma_r": 33_333.0, # kT ns rad^-2
    "total_sim_time": 1000.0, # ns, should be integer multiple of timestep (not checked)
    "clat_out_inter": 1000, # step interval (1000 => every 10 ns with default timestep)
    "mem_out_inter": 1000, # step interval
    "progress_out_incr": 5.0, # percent, how often to write out simulation progress
    "com_coords_file": "",
    "rot_vectors_file": ""
}

if len(sys.argv) > 1:
    paramfilename = sys.argv[1]

    with open(paramfilename,"r") as pfile:
        params = json.load(pfile)

    # want komega to default to ktheta if given, not a single default value (unless neither are given)
    if ("ktheta" in params) and ("komega" not in params):
        params["komega"] = params["ktheta"]
    
    for key in default_params:
        if key not in params:
            params[key] = default_params[key]
else:
    params = default_params

try:
    os.mkdir(f"out_{params["jobname"]}")
except:
    print(f"Directory out_{params["jobname"]} either already exists or cannot be created.")
    timecode = datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")
    params["jobname"] += timecode
    print(f"Creating directory out_{params["jobname"]} instead.")
    os.mkdir(f"out_{params["jobname"]}")

total_sim_iters = int(params["total_sim_time"]/params["timestep"])

if params["com_coords_file"] == "" or params["rot_vectors_file"] == "":
    print("Error: Must specify initial positions and orientations with com_coords_file and rot_vectors_file")
    quit()
else:
    with open(params["com_coords_file"],"r") as infile:
        com_coords = json.load(infile)
    with open(params["rot_vectors_file"],"r") as infile:
        rot_vectors = json.load(infile)

# create user-specified initial setup in gsd frame
frame = create_arbitrary_initial_frame(
    params["pucker_angle"],
    params["bond_cutoff"],
    com_coords,
    rot_vectors,
    np.array(params["box_l"])
)
N_clat = (frame.particles.N)//5
clat_inds = [i for i in range(N_clat)]

# create simulation, setup clathrin and integrator
simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=int(uniform.rvs(scale=65536)))
simulation.create_state_from_snapshot(frame)
setup_clathrin_integrator(simulation, params)

logger = hoomd.logging.Logger()

if params["calc_energy"]:
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    simulation.operations.computes.append(thermo)
    logger.add(thermo, quantities=['potential_energy'])

if params["simulate_membrane"]:
    membrane = Membrane(
        simulation,
        params["mem_N"],
        clat_inds,
        params["timestep"],
        params["kbond_cm"],
        params["bind_l"],
        params["kappa"],
        params["mem_sigma"],
        params["eta"]
    )
    mem_writer = hoomd.write.CustomWriter(
        action = MembraneWriter(membrane,f"out_{params["jobname"]}/mem_traj.json"),
        trigger = hoomd.trigger.Periodic(params["mem_out_inter"])
    )
    simulation.operations.writers.append(mem_writer)

gsd_writer = hoomd.write.GSD(
    filename = f"out_{params["jobname"]}/trajectory.gsd",
    trigger = hoomd.trigger.Periodic(params["clat_out_inter"]),
    mode = "xb",
    dynamic = ["property","attribute","topology","momentum"],
    logger=logger
)
prog_writer = hoomd.write.CustomWriter(
    action = ProgressWriter(total_sim_iters),
    trigger = hoomd.trigger.Periodic(int(params["progress_out_incr"]*total_sim_iters/100))
)

simulation.operations.writers.append(gsd_writer)
simulation.operations.writers.append(prog_writer)

start = datetime.now()

simulation.run(total_sim_iters)

end = datetime.now()

print(f"Elapsed time: {(end-start).seconds} seconds")

if params["simulate_membrane"]: mem_writer._action.finalize()

print("Done.")
