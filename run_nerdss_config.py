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
    "verbose": True,
    "calc_energy": False, # whether or not to calculate potential energy each step in HOOMD
    "simulate_membrane": True,
    "box_l": [200.0,200.0,200.0], # nm
    "mem_N": 55, # membrane subdivision number (must be odd in current implementation)
    "kappa": 20.0, # kT, membrane bending modulus
    "mem_sigma": 0.0, # kT nm^-2, membrane surface tension
    "eta": 24.0, # cytosol viscosity
    "timestep": 0.01, # ns
    "total_sim_time": 100.0, # ns, should be integer multiple of timestep (not checked)
    "part_out_inter": 1000, # step interval (1000 => every 10 ns with default timestep)
    "mem_out_inter": 1000, # step interval
    "progress_out_incr": 5.0, # percent, how often to write out simulation progress
    "complex_file": "", # json file containing prepared NERDSS bonded complex data
    "rigid_assembly": False # treat the nerdss assembly as a rigid body? (flexible membrane only)
}

if len(sys.argv) > 1:
    paramfilename = sys.argv[1]

    with open(paramfilename,"r") as pfile:
        params = json.load(pfile)
    
    for key in default_params:
        if key not in params:
            params[key] = default_params[key]
else:
    params = default_params

print("Parameter Dictionary:")
print(params)

try:
    os.mkdir(f"out_{params["jobname"]}")
except:
    print(f"Directory out_{params["jobname"]} either already exists or cannot be created.")
    timecode = datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")
    params["jobname"] += timecode
    print(f"Creating directory out_{params["jobname"]} instead.")
    os.mkdir(f"out_{params["jobname"]}")

total_sim_iters = int(params["total_sim_time"]/params["timestep"])

if params["complex_file"] == "":
    print("Error: Must provide pre-processed bonded complex data json file from NERDSS simulation.")
    quit()

# create simulation, setup clathrin and integrator
simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=int(uniform.rvs(scale=65536)))
if params["rigid_assembly"]:
    mem_bonded_inds = setup_rigid_simulation_from_nerdss(simulation,params)
else:
    mem_bonded_inds = setup_simulation_from_nerdss(simulation,params)

logger = hoomd.logging.Logger()

if params["calc_energy"]:
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    simulation.operations.computes.append(thermo)
    logger.add(thermo, quantities=['potential_energy'])

if params["simulate_membrane"]:
    membrane = Membrane(
        simulation,
        params["mem_N"],
        mem_bonded_inds,
        params["timestep"],
        params["kbond_pm"], # this is populated from the json file in the setup function above
        params["bind_l"], # this is populated from the json file in the setup function above
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
    trigger = hoomd.trigger.Periodic(params["part_out_inter"]),
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
