The archive in this directory contains an example of the simulation data written out by an "open cap" simulation with the following parameter file:

{
    "jobname": "data", 
    "timestep": 0.005, 
    "total_sim_time": 1000000.0, 
    "clat_out_inter": 100000, 
    "mem_out_inter": 100000, 
    "kr": 500.0, 
    "ktheta": 1080.0, 
    "komega": 500.0, 
    "pucker_angle": 107.5, 
    "kappa": 20.0, 
    "com_coords_file": "open_cap_coords.json", 
    "rot_vectors_file": "open_cap_rotvecs.json"
}

The files "open_cap_coords.json" and "open_cap_rotvecs.json" can be found in the "arbitrary_geom" example directory in this repository.
