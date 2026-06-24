import numpy as np
from scipy.optimize import brentq
import warnings
import os
import time


try:
    from ansys.aedt.core import Hfss
except ImportError:
    raise ImportError("PyAEDT not found. Run:  pip install pyaedt")


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — ACD profile math
# ══════════════════════════════════════════════════════════════════════════════

Z_c             = 200.0   # Characteristic impedance [Ohm]
h               = 184   # Antenna height [mm]
alpha           = 0      # Point-charge ratio  (0 = pure line charge)
N_PTS           = 900     # Profile sample count (more = smoother)
PLATE_THICKNESS = 2.0     # [mm]
FEED_GAP        = 12.0     # Gap between plates at feed point [mm]
R_MIN_MM        = 0.5     # Minimum r kept in profile (keep away from Z-axis)
_EPS            = 1e-12   # Singularity guard for math

OFFSET          = 50
OFFSET2         = 75

FOCAL_LENGTH  = 184.0   # mm  — adjust to your reflector focal length
DISH_DIAMETER = 460.0   # mm  — adjust to your dish diameter
OFFSET_Z      = 0

focal_length = FOCAL_LENGTH
arm_half_width = 5
dish_diameter = DISH_DIAMETER
offset_Z = OFFSET_Z 
increment = 5

def theta0_from_Zc(Zc):
    return 2.0 * np.arctan(1.0 / np.exp(Zc / 120.0))

def eq8_residual(z0, alpha, h, Theta0):
    if z0 <= 0 or z0 >= h:
        return np.inf
    denom = h**2 - z0**2
    lhs   = np.log(Theta0**(-2))
    rhs   = np.log(h**2 / denom) + 2.0 * alpha * z0**2 / denom
    return lhs - rhs

def solve_z0(alpha, h, Theta0):
    zs = np.linspace(1e-6 * h, (1 - 1e-6) * h, 2000)
    fs = np.array([eq8_residual(z, alpha, h, Theta0) for z in zs])
    for i in range(len(fs) - 1):
        if np.isfinite(fs[i]) and np.isfinite(fs[i+1]) and fs[i] * fs[i+1] < 0:
            return brentq(eq8_residual, zs[i], zs[i+1], args=(alpha, h, Theta0))
    raise ValueError(f"Could not bracket z0 for alpha={alpha}")

def rhs_eq7(z, r, z0, alpha):
    r  = max(r, _EPS)
    A  = np.sqrt(z**2        + r**2)
    Bp = np.sqrt((z + z0)**2 + r**2)
    Bm = np.sqrt((z - z0)**2 + r**2)
    d  = max((z + z0 + Bp) * (z - z0 + Bm), _EPS)
    n  = max((z + A)**2, _EPS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_term = np.log(n / d)
    pt = alpha * z0 / max(Bm, _EPS) - alpha * z0 / max(Bp, _EPS)
    return log_term + pt

def compute_profile(h, z0, alpha, Theta0, n_pts=300, r_max=90.0):
    """Returns (z_mm, r_mm) arrays of the outer ACD contour."""
    z_arr = np.linspace(1e-4, h * 0.9999, n_pts)
    r_arr = np.full(n_pts, np.nan)
    lhs   = np.log(Theta0**(-2))

    for i, z in enumerate(z_arr):
        def f(r, _z=z):
            return rhs_eq7(_z, r, z0, alpha) - lhs

        r_test = np.linspace(1e-6, r_max, 3000)
        fv     = np.array([f(rv) for rv in r_test])

        for j in range(len(fv) - 1):
            if np.isfinite(fv[j]) and np.isfinite(fv[j+1]) and fv[j] * fv[j+1] < 0:
                try:
                    r_arr[i] = brentq(f, r_test[j], r_test[j+1], xtol=1e-8)
                except Exception:
                    pass
                break

    mask = ~np.isnan(r_arr)
    return z_arr[mask], r_arr[mask]


# ── Compute ────────────────────────────────────────────────────────────────────
Theta0 = theta0_from_Zc(Z_c)
print(f"Theta_0 = {np.degrees(Theta0):.4f} deg")

z0 = solve_z0(alpha, h, Theta0)
print(f"z0      = {z0:.4f} mm")

print("Computing ACD profile points ...")
z_full, r_full = compute_profile(h, z0, alpha, Theta0, n_pts=N_PTS)
print(f"  {len(z_full)} valid points  |  " f"r: {r_full.min():.3f}–{r_full.max():.3f} mm  |  " f"z: {z_full.min():.3f}–{z_full.max():.3f} mm")


feed_gap_half = FEED_GAP / 2.0

z_cont, r_cont = z_full, r_full
print(f"  Contour after trimming: {len(z_cont)} points  |  "  f"r: {r_cont.min():.3f}–{r_cont.max():.3f} mm  |  "  f"z: {z_cont.min():.3f}–{z_cont.max():.3f} mm")
contour_pts = [[float(r), 0.0, float(z)] for r, z in zip(r_cont, z_cont)]


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Launch HFSS
# ══════════════════════════════════════════════════════════════════════════════

print("\nLaunching HFSS 2024 R2 ...")

hfss = Hfss(
    project       = "ACD_Plate_50Ohm",
    design        = "ACD_alpha0",
    solution_type = "DrivenModal",
    version       = "2024.2",
    new_desktop   = True,
    non_graphical = False,
    close_on_exit = False,
)

modeler = hfss.modeler
modeler.model_units = "mm"
print("HFSS session ready.")

hfss["focal_length"]  = f"{FOCAL_LENGTH}mm"
hfss["Dish_diameter"] = f"{DISH_DIAMETER}mm"
hfss["offset_Z"] = f"{OFFSET_Z}mm"


def build_full_closed_profile(r_cont, z_cont):

    pts = []

    
    for r, z in zip(r_cont, z_cont):
        pts.append([float(r), 0.0, float(z)])

    
    for r, z in zip(r_cont[::-1], z_cont[::-1]):
        pts.append([float(-r), 0.0, float(z)])

    PT50 = pts[50][2]
    print(f"Point cut is: {pts[50]}")
    print(f"Point cut is: {PT50}")
    return pts, PT50


# ── Upper plate ───────────────────────────────────────────────────────────────
print("Building full closed upper plate profile (covered polyline) ...")

upper_pts, pt50 = build_full_closed_profile(r_cont, z_cont)

upper_cover = modeler.create_polyline(
    points         = upper_pts,
    close_surface  = True,         
    cover_surface  = True,          
    name           = "ACD_upper_plate",
    material       = "pec",
)

# print("Thickening upper plate sheet → solid plate (+Z direction) ...")
# modeler.thicken_sheet(
#     assignment  = upper_cover.name,
#     thickness   = PLATE_THICKNESS,  # +Z = grow upward (into positive z)
#     both_sides  = False,
# )
upper_cover.color = (255, 128, 0)   # orange

feed_subtract_upper = modeler.create_box(
    origin   = [-FEED_GAP/2, -PLATE_THICKNESS, 0],
    sizes    = [FEED_GAP,    PLATE_THICKNESS, pt50/2],
    name     = "feed_subtract_upper",
    material = "air",
)
feed_subtract_upper.transparency = 0.9

print("Subtracting small portion from upper plate ...")
modeler.subtract(
    blank_list      = upper_cover.name,   
    tool_list       = feed_subtract_upper.name, 
    keep_originals  = False,              
)


hfss.modeler.rotate(
    assignment=upper_cover.name,
    axis="X",
    angle=60,
    units = "deg"   
)

hfss.modeler.rotate(
    assignment=upper_cover.name,
    axis="Z",
    angle=60,
    units = "deg"   
)


print("Mirroring about XY plane (Z → -Z) to get lower plate ...")
modeler.mirror(
    assignment = upper_cover,
    origin     = [0, 0, 0],          
    vector     = [0, 0, 1],          
    duplicate  = True,              
    duplicate_assignment=True
)

lower_obj = modeler[upper_cover]
lower_obj.color = (0, 128, 255)      # blue

print("Both plates created successfully.")

cen = []

verts = modeler.get_object_vertices(upper_cover.name)

for v in verts:
    p = modeler.get_vertex_position(v)
    cen.append(p)
print(v, cen[len(cen)-1])

# check = modeler.create_polyline(
#     points         = cen,
#     close_surface  = False,         
#     cover_surface  = False,          
#     name           = "check_surface",
# )

x, y, z = cen[1]
x1, y1, z1 = cen[0]

P = [
    [x, y, z],
    [x1, y1, z1],
    [x1, y1, -z1],
    [x, y, -z]
]

print(P)


try_mid_1 = np.array(P[0], dtype="float")
try_mid_2 = np.array(P[1], dtype = "float")
try_mid_3 = np.array(P[2], dtype="float")
try_mid_4 = np.array(P[3], dtype = "float")

P_mid_1 = (try_mid_1 + try_mid_2)/2
P_mid_2 = (try_mid_3 + try_mid_4)/2

G = [
    [P_mid_1[0], P_mid_1[1], P_mid_1[2]],
    [P_mid_2[0], P_mid_2[1], P_mid_2[2]],
    [0-offset_Z, P_mid_2[1], P_mid_2[2]],
    [0-offset_Z, P_mid_1[1], P_mid_1[2]],
]

connector = modeler.create_polyline(
    points         = P,
    close_surface  = True,         
    cover_surface  = True,         
    name           = "connect",
    material       = "pec",
)

connector.color = (255,0,0)

feed_line = modeler.create_polyline(
    points         = G,
    close_surface  = True,          
    cover_surface  = True,         
    name           = "feed",
    material       = "air",
)

feed_line.color = (250,0,0)


# # Create parabolic dish using equation based curve
print("Creating equation-based parabolic curve ...")
 
parabola = modeler.create_equationbased_curve(
    x_t        = "0",                                   
    y_t        = "(1./(4*focal_length))*(_t)*(_t)",     
    z_t        = "_t",                                    
    t_start    = "0",                                     
    t_end      = "Dish_diameter/2",                      
    num_points = 0,                                     
    name       = "IRA_FEM_FD",           
)
 
print(f"  Curve created: {parabola.name}")
parabola.color = (255, 200, 0)   # orange

parabola.sweep_around_axis(axis=1, sweep_angle=180)

hfss.modeler.move(
    assignment=parabola,
    vector=[-OFFSET_Z,-FOCAL_LENGTH-increment,0.0],

)


# # Define gnd plane parameters
origin_point = [0-offset_Z, -(FOCAL_LENGTH)-OFFSET, -(FOCAL_LENGTH)-OFFSET2]        
dimensions = [1.2*(FOCAL_LENGTH+OFFSET),    2*(FOCAL_LENGTH+OFFSET2)]           

gnd = hfss.modeler.create_rectangle(
    orientation="YZ",
    origin=origin_point,
    sizes=dimensions,
    name="gnd_plane",
    material="pec"
)

gnd.transparency = 0.5
print(f"  Gnd plane created: {gnd.name}")


# ── Air / radiation box ───────────────────────────────────────────────────────
air_box = hfss.create_open_region(frequency="3.5GHz", boundary="PML")
print("All geometry created.")

# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — Lumped Resistor
# ══════════════════════════════════════════════════════════════════════════════

mid_idx = len(cen) // 2

pts_around_center = cen[mid_idx-30 : mid_idx+2]

t_max = dish_diameter / 2     # max radial parameter

pts_AB = [list(map(float, p)) for p in pts_around_center]
A = np.array(pts_AB[0],  dtype=float)
B = np.array(pts_AB[-1], dtype=float)

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Compute arm direction (from origin toward parabola)
# ══════════════════════════════════════════════════════════════════════════

AB_mid       = (A + B) / 2.0                        # midpoint of AB
arm_dir      = AB_mid                               # arm goes from origin → AB_mid
arm_dir_unit = arm_dir / np.linalg.norm(arm_dir)   # normalize

print(f"AB midpoint:    {AB_mid}")
print(f"Arm direction:  {arm_dir_unit}")

shift_x = -offset_Z    # mm — parabola shifted -1mm along X
shift_y =  0.0    # mm — no shift in Y
shift_z =  0.0    # mm — no shift in Z

def ray_parabola_intersection(start_pt, direction, focal_length):
   
    Px, Py, Pz = start_pt
    Dx, Dy, Dz = direction

    Px_s = Px - shift_x    # X component relative to shifted parabola center
    Py_s = Py - shift_y    # Y component relative to shifted parabola center
    Pz_s = Pz - shift_z    # Z component relative to shifted parabola center


    # Quadratic coefficients
    a = (Dx**2 + Dz**2) / (4 * focal_length)
    b = (2*Px_s*Dx + 2*Pz_s*Dz) / (4*focal_length) - Dy
    c = (Px_s**2 + Pz_s**2) / (4*focal_length) - focal_length - Py_s

    disc = b**2 - 4*a*c

    if disc < 0:
        raise ValueError(f"Ray does not intersect parabola (discriminant = {disc:.6f})")

    s1 = (-b + np.sqrt(disc)) / (2*a)
    s2 = (-b - np.sqrt(disc)) / (2*a)

    print(f"The roots of the equation: s1 = {s1:.04f} and s2 = {s2:.04f}")

    # Keep only positive s (forward along ray direction)
    s_vals = sorted([s for s in [s1, s2] if s > 1e-6])

    if not s_vals:
        raise ValueError("No forward intersection found")

    s = s_vals[-1]   

    hit_pt = np.array([Px + s*Dx,
                        Py + s*Dy,
                        Pz + s*Dz])
    return hit_pt, s


O, s_O = ray_parabola_intersection(A, arm_dir_unit, focal_length+increment)
C, s_C = ray_parabola_intersection(B, arm_dir_unit, focal_length+increment)

print(f"\nA = {np.round(A, 4)}  →  O = {np.round(O, 4)}  (s = {s_O:.4f} mm)")
print(f"B = {np.round(B, 4)}  →  C = {np.round(C, 4)}  (s = {s_C:.4f} mm)")


num_pts = len(pts_AB)
pts_OC = []
for i in range(num_pts):
    ab_interp = pts_AB[i]  

    try:
        hit_pt, s = ray_parabola_intersection(ab_interp, arm_dir_unit, focal_length+increment)
        pts_OC.append(hit_pt.tolist())
    except ValueError as e:
        pts_OC.append(ab_interp.tolist())

print(f"\nAB edge: {len(pts_AB)} points")
print(f"OC edge: {len(pts_OC)} points")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: Sanity check — verify OC points lie on parabola
# ══════════════════════════════════════════════════════════════════════════

print(f"\nSanity check — points on shifted parabola:")
for label, pt in [("O", O), ("C", C)]:
    x, y, z  = pt
    y_expect = ((x - shift_x)**2 + (z - shift_z)**2) / (4*focal_length) - focal_length + shift_y
    err      = abs(y - y_expect)
    status   = "✓ ON shifted parabola" if err < 0.01 else "✗ NOT on parabola"
    print(f"  {label} = {np.round(pt, 4)}")
    print(f"    Y_actual={y:.6f}  Y_expected={y_expect:.6f}  err={err:.2e}  {status}")

pts_AB.extend(pts_OC[::-1])
resistor = hfss.modeler.create_polyline(
    points        = pts_AB,
    name          = "lumped_resistor",
    close_surface = True,
    cover_surface = True
)
resistor.color = (0, 0, 255)
print("\nCreated the lumped RLC sheet")

print("Mirroring about XY plane (Z → -Z) to get lower resistor ...")

modeler.mirror(
    assignment = resistor,
    origin     = [0, 0, 0],          
    vector     = [0, 0, 1],          
    duplicate  = True,              
    duplicate_assignment=True
)

lower_resistor = modeler[resistor]
lower_resistor.color = (0, 120, 255)      

print("Both resistors created successfully.")


O_point = np.array(pts_OC[0], dtype = "float")
C_point = np.array(pts_OC[-1], dtype = "float")

integration_vector_1 = (O_point + C_point)/2                            
integration_vector_1.tolist()
integration_vector_2 = pts_AB[int(num_pts/2)+7]

rlc = hfss.assign_lumped_rlc_to_sheet(
    assignment   = "lumped_resistor",          
    start_direction   = [integration_vector_2, integration_vector_1],
    rlc_type="Serial",
    resistance   = Z_c/2,                    
    inductance   = 0,                       
    capacitance  = 0,                      
    name         = "LumpedRLC"
)

integration_vector_3 = [integration_vector_1[0], integration_vector_1[1], -integration_vector_1[2]]
integration_vector_4 = [integration_vector_2[0], integration_vector_2[1], -integration_vector_2[2]]

# # Assign lumped RLC to the sheet
rlc_1 = hfss.assign_lumped_rlc_to_sheet(
    assignment="lumped_resistor_1",
    start_direction = [integration_vector_4, integration_vector_3],  
    rlc_type="Serial",              
    resistance=Z_c/2,                      
    inductance=0,                   
    capacitance=0,                  
    name="LumpedRLC1"
)

# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — Boundaries & excitation
# ══════════════════════════════════════════════════════════════════════════════

print("Assigning PEC boundaries ...")
hfss.assign_perfect_e(
    assignment = ["ACD_upper_plate", "IRA_FEM_FD", "ACD_upper_plate_1", "gnd_plane", "connect"],
    name       = "PEC_plates",
)

print("Assigning lumped port ...")
port = hfss.lumped_port(
    assignment        = "feed",
    reference         = "gnd_plane",
    name              = "LumpedPort1",
    integration_line  = hfss.axis_directions.XPos,  
    impedance         = Z_c/4,
    renormalize       = False,
    deembed           = False,
)

print(f"Port created: {port.name}")


# ─────────────────────────────────────────────────
# COMPLETE MESH SETUP — All mesh operations
# ─────────────────────────────────────────────────

# ── 1. Apply Curvilinear Elements ──────────────────
curve_mesh = hfss.mesh.assign_curvilinear_elements(
    assignment=["IRA_FEM_FD"],   # your curved geometry objects
    name="ApplyCurvilinear"
)
curve_mesh.update()

# ── 2. Model Resolution ────────────────────────────
resolution = hfss.mesh.assign_model_resolution(
    assignment=["IRA_FEM_FD"],
    defeature_length=0.1,                  # None = use default
    name="ModelResolution"
)
resolution.update()

# ── 3. Length Based — "connect & Feed" 
connector = hfss.mesh.assign_length_mesh(
    assignment=["connect", "feed"],   
    maximum_length=0.1,    
    maximum_elements=None,                 
    name="connect_mesh"
)
connector.update()

# ── 4. Length Based — "Feed_arm" ───────────────────────
arm = hfss.mesh.assign_length_mesh(
    assignment=["ACD_upper_plate", "ACD_upper_plate_1"],            
    maximum_elements=None,                     
    name="arm_mesh"
)
arm.update()

# ── 5. Length Based — "Resistor" ───────────────────────
R = hfss.mesh.assign_length_mesh(
    assignment=["lumped_resistor", "lumped_resistor_1"],            
    maximum_length=4,
    maximum_elements=None,                     
    name="resistor_mesh"
)
R.update()

# ── 6. Length Based — "Gnd_plane" ─────────────
ground = hfss.mesh.assign_length_mesh(
    assignment=["gnd_plane"],    # TEM arm objects
    maximum_length=10,  
    maximum_elements=None,                   
    name="gnd_mesh"
)
ground.update()

# ── 7. Length Based — "Reflector" ─────────────────
reflector = hfss.mesh.assign_length_mesh(
    assignment=["IRA_FEM_FD"],              
    maximum_length=5, 
    maximum_elements=None,                    
    name="reflector_mesh"
)
reflector.update()

project_path = os.path.join(os.path.expanduser("~"), "ACD_Plate_rework_again_2_alpha0.aedt")
# project_path = r"D:\IITB\HFSS\ACD_Plate_rework_alpha0.aedt"
hfss.save_project(project_path)
print(f"\nProject saved: {project_path}")

print("Project saved — mesh operations committed.")


# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — Solution setup & frequency sweep
# ══════════════════════════════════════════════════════════════════════════════

print("Creating solution setup ...")
setup = hfss.create_setup(name="Setup1")
setup.props["Frequency"]     = "3.5GHz"
setup.props["MaximumPasses"] = 20
setup.props["MaxDeltaS"]     = 0.02
setup.props["BasisOrder"]    = 1
setup.update()
print("[DONE] Adaptive setup created — will refine from custom initial mesh.")


print("Adding frequency sweep 100MHz – 15 GHz ...")
sweep = hfss.create_linear_count_sweep(
    setup              = "Setup1",
    unit               = "GHz",
    start_frequency    = 0.1,
    stop_frequency     = 7.5,
    num_of_freq_points = 901,
    name               = "Sweep1",
    sweep_type         = "Interpolating",
    save_fields        = True,
    interpolation_max_solutions = 1000,
)

# Set Minimum Solutions (corresponds to "Minimum Solutions" field in the dialog)
sweep.props["MinSolutions"] = 750       # default is 0 (as shown in your screenshot)
sweep.update()
setup.update()

# hfss.analyze("Setup1")

# Step 1: Verify all mesh operations are registered
print("\nRegistered mesh operations:")
for op in hfss.mesh.meshoperations:
    print(f"  - {op.name} on {op.props.get('Objects', 'N/A')}")

# Step 2: Generate mesh using the setup name
print("\nGenerating mesh...")
hfss.save_project()   # save again just before meshing

mesh_success = hfss.mesh.generate_mesh("Setup1")

if mesh_success:
    print("[DONE] Mesh generated using assigned length operations.")
else:
    print("[WARN] Mesh generation returned False — check HFSS message window.")
# ============================================================================
# Collect model objects
# ============================================================================

object_names = []
# Add sheets
for obj in hfss.modeler.sheet_objects:
    object_names.append(obj.name)

print(f"Number of objects found = {len(object_names)}")
print(object_names)

mesh_plot = hfss.post.create_fieldplot_surface(
    assignment    = object_names,   # all objects in the model
    quantity = "Mesh",
    setup      = "Setup1",
    plot_name="Mesh1",                         # display the plot window
)
print(f"Full model mesh plot: {mesh_plot}")



# ============================================================================
# Check available sweeps
# ============================================================================

# print("\nAvailable analysis sweeps:")
# print(hfss.existing_analysis_sweeps)

# setup_name = hfss.existing_analysis_sweeps[0]

# print(f"\nUsing solution: {setup_name}")

# ============================================================================
# Create mesh plot
# ============================================================================

# try:

#     mesh_plot = hfss.post.create_fieldplot_surface(
#         assignment=object_names,
#         quantity="Mesh",
#         setup=setup_name,
#         plot_name="Mesh1"
#     )

#     print("[DONE] Mesh plot created successfully.")
#     print(mesh_plot)

# except Exception as e:

#     print("\n[ERROR] Failed to create mesh plot")
#     print(e)


# ── Uncomment to solve immediately ───────────────────────────────────────────
# hfss.analyze_setup("Setup1")

# ── Uncomment to plot S11 after solving ──────────────────────────────────────
# import matplotlib.pyplot as plt
# sol = hfss.post.get_solution_data(
#     expressions      = "dB(S(LumpedPort1,LumpedPort1))",
#     setup_sweep_name = "Setup1 : Sweep1",
# )
# plt.plot(sol.primary_sweep_values, sol.data_real())
# plt.xlabel("Frequency (GHz)"); plt.ylabel("S11 (dB)")
# plt.title("ACD Plate Return Loss — Zc=200Ω, α=0")
# plt.grid(True); plt.show()

# hfss.insert_infinite_sphere(
#     phi_start=-180,
#     phi_stop=180,
#     phi_step=5,
#     theta_start=-180,
#     theta_stop=180,
#     theta_step=5,
#     name="Infinite Sphere1"
# )
# antenna_data = hfss.get_antenna_data(
#     setup=hfss.nominal_adaptive,
#     sphere="Infinite Sphere1",
# )

# plot_data = hfss.get_traces_for_plot()
# report = hfss.post.create_report(plot_data)
# solution = report.get_solution_data()
# plt = solution.plot(solution.expressions)

print("\nDone. Review geometry in HFSS before solving.")