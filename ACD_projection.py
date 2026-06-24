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

Z_c             = 400.0   # Characteristic impedance [Ohm]
h               = 184.0   # Antenna height [mm]
alpha           = 1      # Point-charge ratio  (0 = pure line charge)
N_PTS           = 900     # Profile sample count (more = smoother)
PLATE_THICKNESS = 2.0     # [mm]
FEED_GAP        = 8.0     # Gap between plates at feed point [mm]
R_MIN_MM        = 0.5     # Minimum r kept in profile (keep away from Z-axis)
_EPS            = 1e-12   # Singularity guard for math

OFFSET          = 50
OFFSET2         = 75

FOCAL_LENGTH  = 180.0   # mm  — adjust to your reflector focal length
DISH_DIAMETER = 460.0   # mm  — adjust to your dish diameter

focal_length = 180.0
arm_half_width = 5
dish_diameter = 460

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
# z_cont, r_cont = prepare_open_contour(z_full, r_full, feed_gap_half, R_MIN_MM)
z_cont, r_cont = z_full, r_full
print(f"  Contour after trimming: {len(z_cont)} points  |  "  f"r: {r_cont.min():.3f}–{r_cont.max():.3f} mm  |  "  f"z: {z_cont.min():.3f}–{z_cont.max():.3f} mm")

# Build [x, y, z] point list (profile lies in XZ plane, y=0)
contour_pts = [[float(r), 0.0, float(z)] for r, z in zip(r_cont, z_cont)]


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Launch HFSS
# ══════════════════════════════════════════════════════════════════════════════

print("\nLaunching HFSS 2024 R2 ...")

hfss = Hfss(
    project       = "ACD_Plate_400Ohm",
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

def build_full_closed_profile(r_cont, z_cont):

    pts = []

    # Right half: bottom → tip  (forward along profile)
    for r, z in zip(r_cont, z_cont):
        pts.append([float(r), 0.0, float(z)])

    # Left half: tip → bottom  (reverse, negate r for X-mirror)
    for r, z in zip(r_cont[::-1], z_cont[::-1]):
        pts.append([float(-r), 0.0, float(z)])

    PT50 = pts[50][2]
    print(f"Point 20 is: {pts[50]}")
    print(f"Point 20 is: {PT50}")
    return pts, PT50


# ── Upper plate ───────────────────────────────────────────────────────────────
print("Building full closed upper plate profile (covered polyline) ...")

upper_pts, pt50 = build_full_closed_profile(r_cont, z_cont)

upper_cover = modeler.create_polyline(
    points         = upper_pts,
    close_surface  = True,          # closes the loop (adds segment back to pt[0])
    cover_surface  = True,          # fills the closed loop → flat sheet in XZ plane
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
    blank_list      = upper_cover.name,   # object to cut into
    tool_list       = feed_subtract_upper.name, # cutting tool
    keep_originals  = False,               # keep tool for reuse on lower plate
)


hfss.modeler.rotate(
    assignment=upper_cover.name,
    axis="X",
    angle=45,
    units = "deg"   
)

hfss.modeler.rotate(
    assignment=upper_cover.name,
    axis="Z",
    angle=45,
    units = "deg"   
)


print("Mirroring about XY plane (Z → -Z) to get lower plate ...")
modeler.mirror(
    assignment = upper_cover,
    origin     = [0, 0, 0],          # mirror plane passes through origin
    vector     = [0, 0, 1],          # normal [0,0,1] = XY plane → flips Z
    duplicate  = True,              # move in place (clone is already separate)
    duplicate_assignment=True
)

# Rename and colour the lower plate
lower_obj = modeler[upper_cover]
lower_obj.color = (0, 128, 255)      # blue

print("Both plates created successfully.")

cen = []

verts = modeler.get_object_vertices(upper_cover.name)

for v in verts:
    p = modeler.get_vertex_position(v)
    cen.append(p)
print(v, cen[len(cen)-1])

x, y, z = cen[len(cen)-1]

P = [
    [ x,  y,  z],
    [-y, -x,  z],
    [-y, -x, -z],
    [ x,  y, -z]
]

print(P)

G = [
    [-y, -x,  z],
    [0,  -x,  z],
    [0,  -x, -z],
    [-y, -x, -z],
]

connector = modeler.create_polyline(
    points         = P,
    close_surface  = True,          # closes the loop (adds segment back to pt[0])
    cover_surface  = True,          # fills the closed loop → flat sheet in XZ plane
    name           = "connect",
    material       = "pec",
)

connector.color = (255,0,0)

feed_line = modeler.create_polyline(
    points         = G,
    close_surface  = True,          # closes the loop (adds segment back to pt[0])
    cover_surface  = True,          # fills the closed loop → flat sheet in XZ plane
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
    t_start    = "0",                                     # apex
    t_end      = "Dish_diameter/2",                       # rim
    num_points = 0,                                       # 0 = smooth NURBS, not segmented
    name       = "IRA_FEM_FD",           # matches screenshot name
)
 
print(f"  Curve created: {parabola.name}")
parabola.color = (255, 200, 0)   # orange

parabola.sweep_around_axis(axis=1, sweep_angle=180)

hfss.modeler.move(
    assignment=parabola,
    vector=[0.0,-FOCAL_LENGTH,0.0],

)

# # Define gnd plane parameters
origin_point = [0, -(FOCAL_LENGTH)-OFFSET, -(FOCAL_LENGTH)-OFFSET2]        # [X, Y, Z] coordinates of the lower-left corner
dimensions = [1.2*(FOCAL_LENGTH+OFFSET),    2*(FOCAL_LENGTH+OFFSET2)]           # [Width, Height] inside the selected plane

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
air_box = hfss.create_open_region(frequency="7.5GHz", boundary="PML")
print("All geometry created.")

# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — Lumped Resistor
# ══════════════════════════════════════════════════════════════════════════════

mid_idx = len(cen) // 2

pts_around_center = cen[mid_idx-50 : mid_idx+51]

t_max = dish_diameter / 2     # max radial parameter

# ══════════════════════════════════════════════════════════════════════════
# CORRECTED parabola world-coordinate function
# After sweep around Y-axis (axis=1) and move by -focal_length in Y:
#
#   X_world(t, phi) = t * sin(phi)
#   Y_world(t, phi) = t²/(4f) - focal_length
#   Z_world(t, phi) = t * cos(phi)
#
# The feed arm lies in the XY plane → phi = 90° → sin=1, cos=0
#   X_world = t
#   Y_world = t²/(4f) - focal_length
#   Z_world = 0
# ══════════════════════════════════════════════════════════════════════════

def point_on_reflector(t, focal_length, phi_deg=90.0):
   
    phi = np.radians(phi_deg)
    x = t * np.sin(phi)
    y = (t**2) / (4 * focal_length) - focal_length   # shifted by move
    z = t * np.cos(phi)
    return np.array([x, y, z])


# Your known A and B from the code comment:
A = np.array(pts_around_center[0],  dtype=float)
B = np.array(pts_around_center[-1], dtype=float)


def project_feedpoint_to_reflector(feed_pt, focal_length):
    
    x, y, z = feed_pt
    t   = np.sqrt(x**2 + z**2)            # radial parameter
    phi = np.degrees(np.arctan2(x, z))    # sweep angle in degrees
    
    return point_on_reflector(t, focal_length, phi), t, phi

O, t_O, phi_O = project_feedpoint_to_reflector(A, focal_length)
C, t_C, phi_C = project_feedpoint_to_reflector(B, focal_length)

print(f"A (feed) = {A}")
print(f"O (on reflector) = {O}  [t={t_O:.3f}, phi={phi_O:.2f}°]")
print()
print(f"B (feed) = {B}")
print(f"C (on reflector) = {C}  [t={t_C:.3f}, phi={phi_C:.2f}°]")


num_pts = len(pts_around_center)

# Interpolate between A and B (feed points) to get intermediate t and phi
alphas = np.linspace(0, 1, num_pts)

OC_curve_pts = []
for alpha in alphas:
    # Interpolate feed point between A and B
    feed_interp = (1 - alpha) * A + alpha * B
    
    x, y, z     = feed_interp
    t   = np.sqrt(x**2 + z**2)
    phi = np.degrees(np.arctan2(x, z))
    
    pt_on_reflector = point_on_reflector(t, focal_length, phi)
    OC_curve_pts.append(pt_on_reflector.tolist())

print(f"\nOC curve: {len(OC_curve_pts)} points")
print(f"  Start O = {OC_curve_pts[0]}")
print(f"  End   C = {OC_curve_pts[-1]}")


pts_around_center = [list(map(float, p)) for p in cen[mid_idx-50 : mid_idx+51]]

# Verify A and B match endpoints of pts_around_center
print(f"\nAB edge start = {pts_around_center[0]}")
print(f"AB edge end   = {pts_around_center[-1]}")

pts_around_center.extend(OC_curve_pts[::-1])
resistor = hfss.modeler.create_polyline(
    points       = pts_around_center,
    name         = "lumped_resistor",
    close_surface= True,
    cover_surface= True
)
resistor.color = (255, 0, 0)
print("Created termination sheet on reflector surface")

print("Mirroring about XY plane (Z → -Z) to get lower resistor ...")

modeler.mirror(
    assignment = resistor,
    origin     = [0, 0, 0],          # mirror plane passes through origin
    vector     = [0, 0, 1],          # normal [0,0,1] = XY plane → flips Z
    duplicate  = True,              # move in place (clone is already separate)
    duplicate_assignment=True
)

# Rename and colour the lower plate
lower_resistor = modeler[resistor]
lower_resistor.color = (0, 120, 255)      # blue

print("Both resistors created successfully.")


rlc = hfss.assign_lumped_rlc_to_sheet(
    assignment   = "lumped_resistor",          # name of ABOC surface after connect()
    start_direction   = hfss.axis_directions.YPos,
    rlc_type="Parallel",
    resistance   = 200,                     # Ohms
    inductance   = 0,                       # Henry (0 = not used)
    capacitance  = 0,                       # Farad (0 = not used)
    name         = "LumpedRLC"
)


# # Assign lumped RLC to the sheet
rlc_1 = hfss.assign_lumped_rlc_to_sheet(
    assignment="lumped_resistor_1",
    start_direction=hfss.axis_directions.YPos,  # Integration line direction
    rlc_type="Parallel",                # "Parallel" or "Serial"
    resistance=200,                      # Ohms
    inductance=0,                    # Henry
    capacitance=0,                   # Farads
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
    integration_line  = hfss.axis_directions.XPos,  # Integration line direction
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

# ── 3. Length Based — "connect & Feed" (0.1 mm, on selection) ─
connector = hfss.mesh.assign_length_mesh(
    assignment=["connect", "feed"],   
    maximum_length=0.1,    
    maximum_elements=None,                 
    name="connect_mesh"
)
connector.update()

# ── 4. Length Based — "Feed_arm" ───────────────────────
arm = hfss.mesh.assign_length_mesh(
    assignment=["ACD_upper_plate", "ACD_upper_plate_1"],             # your feed geometry name
    maximum_length=4,
    maximum_elements=None,                     
    name="arm_mesh"
)
arm.update()

# ── 5. Length Based — "Resistor" ───────────────────────
R = hfss.mesh.assign_length_mesh(
    assignment=["lumped_resistor", "lumped_resistor_1"],             # your feed geometry name
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
    assignment=["IRA_FEM_FD"],               # ground/reflector plane object
    maximum_length=5, 
    maximum_elements=None,                    
    name="reflector_mesh"
)
reflector.update()

project_path = os.path.join(os.path.expanduser("~"), "ACD_Plate_200Ohm_try.aedt")
# project_path = r"D:\IITB\HFSS\ACD_Plate_200Ohm.aedt"
hfss.save_project(project_path)
print(f"\nProject saved: {project_path}")

print("Project saved — mesh operations committed.")


# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — Solution setup & frequency sweep
# ══════════════════════════════════════════════════════════════════════════════

print("Creating solution setup ...")
setup = hfss.create_setup(name="Setup1")
setup.props["Frequency"]     = "7.5GHz"
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
    stop_frequency     = 15.0,
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