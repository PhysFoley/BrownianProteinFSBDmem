import numpy as np
from scipy.spatial.transform import Rotation
from sys_setup import parse_nerdss_mol_file

debug = False

def residual(rotvec,points):
    R = Rotation.from_rotvec(rotvec)
    return np.sum(np.power(R.apply(points)[:,2],2.0))

def get_float(varname,default=500.0):
    while(True):
        userin = input(f"  {varname}: ")
        if userin.strip() == "":
            print(f"Using default value {default} for {varname}")
            return default
        else:
            try:
                value = float(userin)
                return value
            except:
                print("Invalid input value")

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from scipy.optimize import minimize
    import sys
    import json
    
    fname = sys.argv[1]
    comp_ind = int(sys.argv[2])
    site_names = sys.argv[3:]

    with open(fname,"r") as file:
        data = json.load(file)

    compl = data[comp_ind]

    mol_data = {}
    for name in list(set(compl["names"])):
        mol_data[name] = parse_nerdss_mol_file(name + ".mol")

    coms = compl['coords']
    rots = [Rotation.from_quat(q,scalar_first=True) for q in compl['rotations']]

    compl["membrane_bonds"] = {}
    compl["membrane_bonds"]["bonds"] = []
    site_points = []
    for i,name in enumerate(compl["names"]):
        for site in site_names:
            if site in mol_data[name]["sites"]:
                compl["membrane_bonds"]["bonds"].append({"molindex":i, "site":site})
                point = coms[i] + rots[i].apply(mol_data[name]["sites"][site])
                site_points.append(point)
    site_points = np.array(site_points)

    sites_com = np.mean(site_points,axis=0)
    rel_points = site_points - sites_com

    rotvec = minimize(residual,[0.0,0.0,0.0],args=(rel_points,)).x
    R = Rotation.from_rotvec(rotvec)

    rotated_sites = R.apply(rel_points)
    rotated_coms = R.apply(coms - sites_com)

    if np.mean(rotated_coms[:,2]) < np.mean(rotated_sites[:,2]):
        R2 = Rotation.from_rotvec([np.pi,0.0,0.0])
        R = R2*R

    rotated_sites = R.apply(rel_points)
    rotated_coms = R.apply(coms - sites_com)

    # shift so that all binding sites are at or above the initial membrane
    rotated_coms[:,2] -= np.min(rotated_sites[:,2])
    rotated_sites[:,2] -= np.min(rotated_sites[:,2]) # (just for visualization, we don't use these)

    ax = plt.axes(projection='3d')
    ax.scatter(rotated_sites[:,0],rotated_sites[:,1],rotated_sites[:,2])
    ax.scatter(rotated_coms[:,0],rotated_coms[:,1],rotated_coms[:,2])
    ax.set_aspect("equal")
    plt.show()

    ofname = "processed_" + fname.split("\\")[-1].split("/")[-1]
    compl['coords'] = rotated_coms.tolist()
    compl['rotations'] = [(R*r).as_quat(scalar_first=True).tolist() for r in rots]

    # done with coordinate edits, now we need bond parameters
    print("\nEnter bond parameters:\n")
    
    compl["membrane_bonds"]["bind_l"] = get_float("membrane-site bind_l", default=10.0)
    compl["membrane_bonds"]["k_mem"] = get_float("membrane tether stiffness")
    
    for name,bt in compl["bond_types"].items():
        print(f"\nBond Type: {name}")
        
        bt["k_sigma"] = get_float("k_sigma")

        optional = ["theta1","theta2","phi1","phi2","omega"]
        for var in optional:
            if bt[var] is not None:
                bt["k_"+var] = get_float("k_"+var)
            else:
                bt["k_"+var] = None

    # one last modification:
    # due to difficulties in nerdss, the order of the molecules in
    # the *bond* dictionary might not match the order of the
    # molecules in the *bond_type*. This matters, though, so we
    # have to sort it out using the name of the bond_type    
    for bond in compl["bonds"]:
        present_bond_type = compl["names"][bond["molindex1"]] + "(" + bond["site1"] + "!1)." 
        present_bond_type += compl["names"][bond["molindex2"]] + "(" + bond["site2"] + "!1)"
        
        if present_bond_type != bond["type"]:
            if debug:
                print(f"Swapping order of molecules for this bond:")
                print(bond)
            bond["molindex1"], bond["molindex2"] = bond["molindex2"], bond["molindex1"]
            bond["site1"], bond["site2"] = bond["site2"], bond["site1"]

        present_bond_type = compl["names"][bond["molindex1"]] + "(" + bond["site1"] + "!1)." 
        present_bond_type += compl["names"][bond["molindex2"]] + "(" + bond["site2"] + "!1)"
        
        if present_bond_type != bond["type"]:
            print(f"ERROR: neither order of molecules matches bond type {bond['type']}")
            quit()

        if "mol1" not in compl["bond_types"][bond["type"]]:
            compl["bond_types"][bond["type"]]["mol1"] = compl["names"][bond["molindex1"]]
            compl["bond_types"][bond["type"]]["mol2"] = compl["names"][bond["molindex2"]]
    
    with open(ofname,"w") as outfile:
        outfile.write(json.dumps([compl],indent=2))
