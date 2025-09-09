# generate_point_cloud_sandbox.py
# Creates 30 distinct synthetic point cloud scenes at real scale
# Density target about 400 pts per square meter, clear spacing between scenes
# Exports individual LAS or LAZ plus a combined file, writes a legend.json
# Optional Open3D preview, if ADD_RGB is true it shows RGB, otherwise grayscale from intensity

import math
import json
import numpy as np
import laspy
from pathlib import Path

try:
    import open3d as o3d
    HAVE_O3D = True
except Exception:
    HAVE_O3D = False

# -----------------------------
# Global configuration
# -----------------------------
OUT_DIR = Path("pointcloud_sandbox_output")
WRITE_LAZ = True             # set to False to write LAS instead of LAZ
SHOW_PREVIEW = True         # set to True to open an Open3D window after export
ADD_RGB = True               # set to False for intensity only

RNG = np.random.default_rng(42)

# density 400 pts per m^2 implies grid spacing near 0.05 m
TARGET_PPM2 = 400.0
BASE_SPACING = (1.0 / TARGET_PPM2) ** 0.5

# jitter for natural look
XY_JITTER = 0.00
Z_JITTER = 0.000

# scene placement
GRID_COLUMNS = 5
GRID_ROWS = 6
CELL_SIZE = 12.0

# material intensity presets, 16 bit LAS intensity meaning 0 to 65535
MAT_INT = {
    "asphalt":       (8000, 1500),
    "concrete":      (18000, 3000),
    "paint_white":   (42000, 4000),
    "paint_yellow":  (36000, 3500),
    "metal":         (30000, 5000),
    "wood":          (20000, 4000),
    "vegetation":    (16000, 3000),
    "plastic":       (22000, 4000),
    "gravel":        (14000, 3000),
    "water":         (6000, 1500),
    "soil":          (12000, 2500),
    "glass":         (8000, 2000),
}

# material colors for RGB, values 0 to 255, converted to 16 bit when written
MAT_RGB_255 = {
    "asphalt":       (50, 50, 50),
    "concrete":      (170, 170, 170),
    "paint_white":   (245, 245, 245),
    "paint_yellow":  (240, 210, 0),
    "metal":         (180, 185, 190),
    "wood":          (160, 120, 70),
    "vegetation":    (40, 130, 40),
    "plastic":       (180, 200, 220),
    "gravel":        (120, 120, 120),
    "water":         (30, 60, 130),
    "soil":          (120, 85, 60),
    "glass":         (180, 220, 240),
}

# ASPRS like classes with a few custom codes
CLASS = {
    "unclassified": 1,
    "ground": 2,
    "low_veg": 3,
    "high_veg": 5,
    "building": 6,
    "wire_guard": 13,
    "wire_conductor": 14,
    "bridge_deck": 17,
    "road_surface": 23,
    "curb": 24,
    "barrier": 25,
    "street_marking": 26
}

# -----------------------------
# Helpers
# -----------------------------

def make_grid(nx, ny, spacing):
    xs = np.arange(nx) * spacing
    ys = np.arange(ny) * spacing
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    return gx.ravel(), gy.ravel()

def jitter(arr, scale):
    return arr + RNG.normal(0.0, scale, size=arr.shape)

def clipped_intensity(mean, std, size):
    vals = RNG.normal(mean, std, size=size)
    vals = np.clip(vals, 0, 65535)
    return vals.astype(np.uint16)

def color_arrays_for(mat_key, size):
    if not ADD_RGB:
        return None, None, None
    r8, g8, b8 = MAT_RGB_255.get(mat_key, (200, 200, 200))
    # expand to arrays and convert to 16 bit by multiplying by 257
    r = np.full(size, r8 * 257, dtype=np.uint16)
    g = np.full(size, g8 * 257, dtype=np.uint16)
    b = np.full(size, b8 * 257, dtype=np.uint16)
    return r, g, b

def stack_fields(parts):
    if not parts:
        return None
    xs = np.concatenate([p["x"] for p in parts])
    ys = np.concatenate([p["y"] for p in parts])
    zs = np.concatenate([p["z"] for p in parts])
    intensity = np.concatenate([p["intensity"] for p in parts]).astype(np.uint16)

    cls_list = []
    for p in parts:
        if "cls" in p:
            cls_list.append(p["cls"].astype(np.uint8))
        elif "class" in p:
            cls_list.append(p["class"].astype(np.uint8))
        else:
            raise KeyError("part missing classification key cls or class")
    classification = np.concatenate(cls_list).astype(np.uint8)

    out = dict(x=xs, y=ys, z=zs, intensity=intensity, cls=classification)

    if ADD_RGB and "red" in parts[0]:
        red = np.concatenate([p["red"] for p in parts]).astype(np.uint16)
        green = np.concatenate([p["green"] for p in parts]).astype(np.uint16)
        blue = np.concatenate([p["blue"] for p in parts]).astype(np.uint16)
        out["red"] = red
        out["green"] = green
        out["blue"] = blue

    return out

def translate(points, dx, dy, dz=0.0):
    points["x"] = points["x"] + dx
    points["y"] = points["y"] + dy
    points["z"] = points["z"] + dz
    return points

def ensure_cls(points):
    if "cls" not in points:
        if "class" in points:
            points["cls"] = points["class"].astype(np.uint8)
        else:
            raise KeyError("points missing classification key cls or class")
    return points

def attach_rgb(d, mat_key):
    if not ADD_RGB:
        return d
    r, g, b = color_arrays_for(mat_key, d["x"].size)
    d["red"] = r
    d["green"] = g
    d["blue"] = b
    return d

# -----------------------------
# Primitive makers
# -----------------------------

def make_plane(area_w, area_h, z0, mat_key, cls, spacing=BASE_SPACING, slope=(0.0, 0.0)):
    nx = max(1, int(area_w / spacing))
    ny = max(1, int(area_h / spacing))
    gx, gy = make_grid(nx, ny, spacing)
    
    # Center the plane at origin
    gx = gx - area_w / 2
    gy = gy - area_h / 2
    
    gx = jitter(gx, XY_JITTER)
    gy = jitter(gy, XY_JITTER)
    z = z0 + slope[0] * gx + slope[1] * gy
    z = jitter(z, Z_JITTER)
    mean, std = MAT_INT[mat_key]
    inten = clipped_intensity(mean, std, gx.size)
    d = {"x": gx, "y": gy, "z": z, "intensity": inten, "class": np.full(gx.size, CLASS[cls], dtype=np.uint8)}
    return attach_rgb(d, mat_key)

def make_box(l, w, h, base_z, mat_key, cls, spacing=BASE_SPACING):
    faces = []
    faces.append(make_plane(l, w, base_z, mat_key, cls, spacing))
    faces.append(make_plane(l, w, base_z + h, mat_key, cls, spacing))

    def side_x(x0):
        ny = max(1, int(h / spacing))
        nz = max(1, int(w / spacing))
        gy, gz = make_grid(ny, nz, spacing)
        gy = jitter(gy, XY_JITTER)
        gz = jitter(gz, XY_JITTER)
        x = np.full(gy.size, x0)
        y = gz - w/2  # Center the side face
        z = base_z + gy
        mean, std = MAT_INT[mat_key]
        inten = clipped_intensity(mean, std, gy.size)
        d = {"x": x, "y": y, "z": z, "intensity": inten, "class": np.full(x.size, CLASS[cls], dtype=np.uint8)}
        return attach_rgb(d, mat_key)

    def side_y(y0):
        nx = max(1, int(l / spacing))
        nz = max(1, int(h / spacing))
        gx, gz = make_grid(nx, nz, spacing)
        gx = jitter(gx, XY_JITTER)
        gz = jitter(gz, XY_JITTER)
        x = gx - l/2  # Center the side face
        y = np.full(gx.size, y0)
        z = base_z + gz
        mean, std = MAT_INT[mat_key]
        inten = clipped_intensity(mean, std, gx.size)
        d = {"x": x, "y": y, "z": z, "intensity": inten, "class": np.full(x.size, CLASS[cls], dtype=np.uint8)}
        return attach_rgb(d, mat_key)

    faces.append(side_x(-l/2))
    faces.append(side_x(l/2))
    faces.append(side_y(-w/2))
    faces.append(side_y(w/2))
    return stack_fields(faces)

def make_cylinder(radius, height, base_z, mat_key, cls, spacing=BASE_SPACING):
    circumference = 2.0 * math.pi * radius
    n_theta = max(12, int(circumference / spacing))
    n_h = max(2, int(height / spacing))
    theta = np.linspace(0, 2.0 * math.pi, n_theta, endpoint=False)
    z = np.linspace(base_z, base_z + height, n_h)
    theta, z = np.meshgrid(theta, z, indexing="xy")
    theta = theta.ravel()
    z = z.ravel()
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    x = jitter(x, XY_JITTER * 0.5)
    y = jitter(y, XY_JITTER * 0.5)
    z = jitter(z, Z_JITTER)
    mean, std = MAT_INT[mat_key]
    inten = clipped_intensity(mean, std, x.size)
    d = {"x": x, "y": y, "z": z, "intensity": inten, "class": np.full(x.size, CLASS[cls], dtype=np.uint8)}
    return attach_rgb(d, mat_key)

def make_wire(p0, p1, sag_ratio, mat_key, cls, spacing=BASE_SPACING):
    dx = np.linalg.norm(np.array(p1[:2]) - np.array(p0[:2]))
    length = max(dx, 0.1)
    n = max(20, int(length / spacing))
    ts = np.linspace(0.0, 1.0, n)
    x = p0[0] + ts * (p1[0] - p0[0])
    y = p0[1] + ts * (p1[1] - p0[1])
    base_z = p0[2] + ts * (p1[2] - p0[2])
    sag = sag_ratio * length
    z = base_z - 4.0 * sag * (ts - 0.5) ** 2 + sag
    x = jitter(x, XY_JITTER * 0.1)
    y = jitter(y, XY_JITTER * 0.1)
    z = jitter(z, Z_JITTER)
    mean, std = MAT_INT[mat_key]
    inten = clipped_intensity(mean, std, x.size)
    d = {"x": x, "y": y, "z": z, "intensity": inten, "class": np.full(x.size, CLASS[cls], dtype=np.uint8)}
    return attach_rgb(d, mat_key)

# -----------------------------
# Composite makers with shape fixes
# -----------------------------

def make_stairs(width, depth, step_h, n_steps, base_z, mat_key, cls, spacing=BASE_SPACING):
    parts = []
    for i in range(n_steps):
        z = base_z + i * step_h
        t = make_plane(width, depth, z, mat_key, cls, spacing)
        t["y"] = t["y"] + i * depth
        parts.append(t)
        riser = make_plane(width, step_h, z, mat_key, cls, spacing)
        y_orig = riser["y"]
        riser["z"] = z + y_orig
        riser["y"] = np.full_like(y_orig, i * depth)
        parts.append(riser)
    return stack_fields(parts)

def make_curb(run_len, curb_w, curb_h, base_z, road_slope, spacing=BASE_SPACING):
    road = make_plane(run_len, 3.5, base_z, "asphalt", "road_surface", spacing, slope=road_slope)
    curb_top = make_plane(run_len, curb_w, base_z + curb_h, "concrete", "curb", spacing)
    curb_top["y"] += 3.5
    curb_face = make_plane(run_len, curb_h, base_z, "concrete", "curb", spacing)
    y_face = curb_face["y"]
    curb_face["z"] = base_z + y_face
    curb_face["y"] = np.full_like(y_face, 3.5)
    return stack_fields([road, curb_top, curb_face])

def make_crosswalk(width, length, base_z, spacing=BASE_SPACING):
    road = make_plane(length, width, base_z, "asphalt", "road_surface", spacing)
    bars = []
    bar_w = 1.0
    gap = 1.0
    pos = 0.0
    while pos < length:
        bar = make_plane(min(bar_w, length - pos), width, base_z + 0.002, "paint_white", "street_marking", spacing * 0.9)
        bar["x"] += pos
        bars.append(bar)
        pos += bar_w + gap
    return stack_fields([road] + bars)

def make_barrier_jersey(run_len, base_z, spacing=BASE_SPACING):
    base_w = 0.6
    top_w = 0.2
    height = 0.8
    top = make_plane(run_len, top_w, base_z + height, "concrete", "barrier", spacing)
    top_shift = base_w - top_w
    top["y"] += top_shift
    face1 = make_plane(run_len, height, base_z, "concrete", "barrier", spacing)
    y1 = face1["y"]
    face1["z"] = base_z + y1 * (height / base_w)
    face1["y"] = np.zeros_like(y1)
    face2 = make_plane(run_len, height, base_z, "concrete", "barrier", spacing)
    y2 = face2["y"]
    face2["z"] = base_z + y2 * (height / top_w)
    face2["y"] = np.full_like(y2, top_shift)
    base = make_plane(run_len, base_w, base_z, "concrete", "barrier", spacing)
    return stack_fields([base, face1, face2, top])

def make_street_patch(width, length, base_z, slope=(0.0, 0.0), spacing=BASE_SPACING):
    return make_plane(width, length, base_z, "asphalt", "road_surface", spacing, slope=slope)

def make_manholes_and_grates(count, base_z, spacing=BASE_SPACING):
    parts = []
    for _ in range(count):
        r = 0.35
        h = 0.05
        cx = RNG.uniform(1.0, 7.0)
        cy = RNG.uniform(1.0, 7.0)
        ring = make_cylinder(r, h, base_z, "metal", "unclassified", spacing * 0.9)
        ring["x"] += cx
        ring["y"] += cy
        parts.append(ring)
    return stack_fields(parts)

def make_bench(base_z, spacing=BASE_SPACING):
    leg1 = make_box(0.1, 0.4, 0.45, base_z, "wood", "unclassified", spacing)
    leg2 = make_box(0.1, 0.4, 0.45, base_z, "wood", "unclassified", spacing)
    leg2 = translate(leg2, 1.2, 0.0, 0.0)
    seat = make_box(1.3, 0.45, 0.05, base_z + 0.45, "wood", "unclassified", spacing)
    return stack_fields([leg1, leg2, seat])

def make_tree(base_z, spacing=BASE_SPACING):
    trunk = make_cylinder(0.15, 3.5, base_z, "wood", "high_veg", spacing)
    crown_rad = 1.5
    n = max(800, int(4.0 * math.pi * crown_rad ** 2 * 20))

    # correct the above line, Python needs math.pi, keeping a comment to highlight
    n = max(800, int(4.0 * math.pi * crown_rad**2 * 20))
    phi = RNG.uniform(0, 2.0 * math.pi, n)
    costheta = RNG.uniform(-1.0, 1.0, n)
    u = RNG.uniform(0.7, 1.0, n)
    theta = np.arccos(costheta)
    r = crown_rad * u**(1.0/3.0)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = base_z + 3.5 + r * np.cos(theta)
    x = jitter(x, XY_JITTER)
    y = jitter(y, XY_JITTER)
    z = jitter(z, Z_JITTER)
    inten = clipped_intensity(*MAT_INT["vegetation"], x.size)
    crown = {"x": x, "y": y, "z": z, "intensity": inten, "class": np.full(x.size, CLASS["high_veg"], dtype=np.uint8)}
    crown = attach_rgb(crown, "vegetation")
    trunk = attach_rgb(trunk, "wood")
    return stack_fields([trunk, crown])

def make_sidewalk(width, length, base_z, spacing=BASE_SPACING):
    return make_plane(length, width, base_z, "concrete", "ground", spacing)

def make_power_pole_and_lines(base_z, spacing=BASE_SPACING):
    pole = make_cylinder(0.2, 9.0, base_z, "wood", "unclassified", spacing)
    arm = make_box(2.0, 0.15, 0.15, base_z + 7.5, "wood", "unclassified", spacing)
    arm = translate(arm, -1.0, 0.0, 0.0)
    p0 = (-1.0, -5.0, base_z + 7.3)
    p1 = (-1.0,  5.0, base_z + 7.2)
    w1 = make_wire(p0, p1, sag_ratio=0.12, mat_key="metal", cls="wire_conductor", spacing=spacing * 0.6)
    p0b = (1.0, -5.0, base_z + 7.4)
    p1b = (1.0,  5.0, base_z + 7.3)
    w2 = make_wire(p0b, p1b, sag_ratio=0.10, mat_key="metal", cls="wire_conductor", spacing=spacing * 0.6)
    return stack_fields([pole, arm, w1, w2])

def make_parking_bumpers(count, base_z, spacing=BASE_SPACING):
    parts = []
    for i in range(count):
        bumper = make_box(1.8, 0.2, 0.12, base_z, "concrete", "unclassified", spacing)
        bumper = translate(bumper, i * 2.5, 0.0, 0.0)
        parts.append(bumper)
    return stack_fields(parts)

def make_speed_hump(width, length, height, base_z, spacing=BASE_SPACING):
    nx = max(1, int(length / BASE_SPACING))
    ny = max(1, int(width / BASE_SPACING))
    gx, gy = make_grid(nx, ny, BASE_SPACING)
    gx = jitter(gx, XY_JITTER)
    gy = jitter(gy, XY_JITTER)
    zy = height * 0.5 * (1.0 - np.cos(np.clip(gy / width, 0, 1) * math.pi))
    z = base_z + zy
    z = jitter(z, Z_JITTER)
    inten = clipped_intensity(*MAT_INT["asphalt"], gx.size)
    d = {"x": gx, "y": gy, "z": z, "intensity": inten, "class": np.full(gx.size, CLASS["road_surface"], dtype=np.uint8)}
    return attach_rgb(d, "asphalt")

def make_guardrail(run_len, base_z, spacing=BASE_SPACING):
    x = np.arange(0.0, run_len, max(spacing, 0.05))
    y = np.zeros_like(x)
    z = np.full_like(x, base_z + 0.75)
    x = jitter(x, 0.01)
    z = jitter(z, 0.01)
    inten = clipped_intensity(*MAT_INT["metal"], x.size)
    rail = {"x": x, "y": y, "z": z, "intensity": inten, "class": np.full(x.size, CLASS["unclassified"], dtype=np.uint8)}
    rail = attach_rgb(rail, "metal")
    posts = []
    for px in np.arange(0.0, run_len, 2.0):
        p = make_cylinder(0.05, 1.0, base_z, "metal", "unclassified", spacing)
        p = translate(p, px, 0.0, 0.0)
        posts.append(p)
    return stack_fields([rail] + posts)

def make_utility_box(base_z, spacing=BASE_SPACING):
    return make_box(1.2, 0.6, 1.5, base_z, "metal", "unclassified", spacing)

def make_phone_cabinet(base_z, spacing=BASE_SPACING):
    return make_box(0.9, 0.5, 1.4, base_z, "metal", "unclassified", spacing)

def make_bollards(n, base_z, spacing=BASE_SPACING):
    parts = []
    for i in range(n):
        b = make_cylinder(0.1, 1.0, base_z, "metal", "unclassified", spacing)
        b = translate(b, i * 1.2, 0.0, 0.0)
        parts.append(b)
    return stack_fields(parts)

def make_driveway_crown(width, length, crown_h, base_z, spacing=BASE_SPACING):
    nx = max(1, int(length / BASE_SPACING))
    ny = max(1, int(width / BASE_SPACING))
    gx, gy = make_grid(nx, ny, BASE_SPACING)
    gx = jitter(gx, XY_JITTER)
    gy = jitter(gy, XY_JITTER)
    center = width * 0.5
    z = base_z + crown_h * np.exp(-((gy - center) ** 2) / (2 * (0.3 * width) ** 2))
    z = jitter(z, Z_JITTER)
    inten = clipped_intensity(*MAT_INT["asphalt"], gx.size)
    d = {"x": gx, "y": gy, "z": z, "intensity": inten, "class": np.full(gx.size, CLASS["road_surface"], dtype=np.uint8)}
    return attach_rgb(d, "asphalt")

# ten extra makers for variety

def make_stop_sign(base_z, spacing=BASE_SPACING):
    post = make_cylinder(0.06, 2.5, base_z, "metal", "unclassified", spacing)
    face = make_box(0.8, 0.02, 0.8, base_z + 1.8, "plastic", "unclassified", spacing)
    face = translate(face, -0.4, 0.0, 0.0)
    return stack_fields([post, face])

def make_streetlight(base_z, spacing=BASE_SPACING):
    pole = make_cylinder(0.12, 7.5, base_z, "metal", "unclassified", spacing)
    arm = make_box(1.8, 0.12, 0.12, base_z + 6.8, "metal", "unclassified", spacing)
    head = make_box(0.5, 0.25, 0.15, base_z + 6.9, "glass", "unclassified", spacing)
    head = translate(head, 1.4, 0.0, 0.0)
    return stack_fields([pole, arm, head])

def make_fire_hydrant(base_z, spacing=BASE_SPACING):
    body = make_cylinder(0.18, 0.9, base_z, "metal", "unclassified", spacing)
    cap = make_cylinder(0.22, 0.15, base_z + 0.9, "metal", "unclassified", spacing)
    side1 = make_cylinder(0.06, 0.25, base_z + 0.45, "metal", "unclassified", spacing)
    side1 = translate(side1, 0.0, 0.18, 0.0)
    side2 = make_cylinder(0.06, 0.25, base_z + 0.45, "metal", "unclassified", spacing)
    side2 = translate(side2, 0.0, -0.18, 0.0)
    return stack_fields([body, cap, side1, side2])

def make_mailbox_cluster(base_z, spacing=BASE_SPACING):
    post = make_cylinder(0.08, 1.4, base_z, "metal", "unclassified", spacing)
    box = make_box(0.9, 0.4, 0.6, base_z + 1.4, "metal", "unclassified", spacing)
    box = translate(box, -0.45, -0.2, 0.0)
    return stack_fields([post, box])

def make_bike_rack_u(base_z, spacing=BASE_SPACING):
    left = make_cylinder(0.04, 0.9, base_z, "metal", "unclassified", spacing)
    left = translate(left, -0.3, 0.0, 0.0)
    right = make_cylinder(0.04, 0.9, base_z, "metal", "unclassified", spacing)
    right = translate(right, 0.3, 0.0, 0.0)
    p0 = (-0.3, 0.0, base_z + 0.9)
    p1 = (0.3, 0.0, base_z + 0.9)
    arc = make_wire(p0, p1, sag_ratio=-0.25, mat_key="metal", cls="unclassified", spacing=spacing * 0.5)
    return stack_fields([left, right, arc])

def make_trash_can(base_z, spacing=BASE_SPACING):
    drum = make_cylinder(0.28, 0.9, base_z, "metal", "unclassified", spacing)
    lid = make_cylinder(0.30, 0.05, base_z + 0.9, "metal", "unclassified", spacing)
    return stack_fields([drum, lid])

def make_picnic_table(base_z, spacing=BASE_SPACING):
    top = make_box(1.8, 0.9, 0.05, base_z + 0.75, "wood", "unclassified", spacing)
    bench1 = make_box(1.8, 0.3, 0.05, base_z + 0.45, "wood", "unclassified", spacing)
    bench1 = translate(bench1, 0.0, -0.7, 0.0)
    bench2 = make_box(1.8, 0.3, 0.05, base_z + 0.45, "wood", "unclassified", spacing)
    bench2 = translate(bench2, 0.0, 0.7, 0.0)
    leg1 = make_box(0.08, 0.6, 0.45, base_z, "wood", "unclassified", spacing)
    leg1 = translate(leg1, -0.7, -0.3, 0.0)
    leg2 = make_box(0.08, 0.6, 0.45, base_z, "wood", "unclassified", spacing)
    leg2 = translate(leg2, 0.7, -0.3, 0.0)
    leg3 = {k: v.copy() for k, v in leg1.items()}
    leg3 = translate(leg3, 0.0, 0.6, 0.0)
    leg4 = {k: v.copy() for k, v in leg2.items()}
    leg4 = translate(leg4, 0.0, 0.6, 0.0)
    return stack_fields([top, bench1, bench2, leg1, leg2, leg3, leg4])

def make_median_island_with_curbs(length, width, curb_h, base_z, spacing=BASE_SPACING):
    pad = make_plane(length, width, base_z, "concrete", "ground", spacing)
    top = make_plane(length, 0.25, base_z + curb_h, "concrete", "curb", spacing)
    top2 = make_plane(length, 0.25, base_z + curb_h, "concrete", "curb", spacing)
    top2["y"] += width - 0.25
    end1 = make_plane(0.25, width, base_z + curb_h, "concrete", "curb", spacing)
    end2 = make_plane(0.25, width, base_z + curb_h, "concrete", "curb", spacing)
    end2["x"] += length - 0.25
    return stack_fields([pad, top, top2, end1, end2])

def make_ped_ramp_with_tactile(width, length, rise, base_z, spacing=BASE_SPACING):
    slope = rise / length
    ramp = make_plane(length, width, base_z, "concrete", "ground", spacing, slope=(slope, 0.0))
    pad = make_plane(0.9, width * 0.8, base_z + slope * (length - 0.9) + 0.005, "paint_yellow", "street_marking", spacing * 0.9)
    pad["x"] += length - 0.9
    pad["y"] += width * 0.1
    return stack_fields([ramp, pad])

def make_storm_inlet_grate(base_z, spacing=BASE_SPACING):
    frame = make_box(1.0, 0.6, 0.08, base_z, "metal", "unclassified", spacing)
    bars = []
    for y in np.linspace(0.05, 0.55, 6):
        bar = make_box(0.95, 0.02, 0.02, base_z + 0.02, "metal", "unclassified", spacing)
        bar = translate(bar, -0.475, y - 0.3, 0.0)
        bars.append(bar)
    return stack_fields([frame] + bars)

# -----------------------------
# Scene library
# -----------------------------

def build_scene_library():
    scenes = []
    scenes.append(("street_patch", lambda: make_street_patch(8.0, 10.0, 0.0, slope=(0.0, 0.01))))
    scenes.append(("crosswalk", lambda: make_crosswalk(6.0, 10.0, 0.0)))
    scenes.append(("curb_with_road", lambda: make_curb(10.0, 0.3, 0.15, 0.0, road_slope=(0.0, 0.002))))
    scenes.append(("jersey_barrier", lambda: make_barrier_jersey(10.0, 0.0)))
    scenes.append(("stairs", lambda: make_stairs(2.0, 0.3, 0.15, 12, 0.0, "concrete", "building")))
    scenes.append(("bench", lambda: make_bench(0.0)))
    scenes.append(("power_pole_wires", lambda: make_power_pole_and_lines(0.0)))
    scenes.append(("manholes", lambda: make_manholes_and_grates(4, 0.0)))
    scenes.append(("barrier_and_curb", lambda: stack_fields([
        translate(make_barrier_jersey(8.0, 0.0), 0.0, 0.0, 0.0),
        translate(make_curb(8.0, 0.3, 0.15, 0.0, road_slope=(0.0, 0.0)), 0.0, 2.0, 0.0)
    ])))
    scenes.append(("sidewalk", lambda: make_sidewalk(3.0, 10.0, 0.0)))
    scenes.append(("parking_bumpers", lambda: make_parking_bumpers(4, 0.0)))
    scenes.append(("speed_hump", lambda: make_speed_hump(6.0, 8.0, 0.12, 0.0)))
    scenes.append(("utility_cabinet", lambda: make_utility_box(0.0)))
    scenes.append(("phone_cabinet", lambda: make_phone_cabinet(0.0)))
    scenes.append(("tree", lambda: make_tree(0.0)))
    scenes.append(("furniture_boxes", lambda: stack_fields([
        translate(make_box(1.0, 0.6, 0.7, 0.0, "wood", "unclassified"), 0.0, 0.0, 0.0),
        translate(make_box(0.8, 0.8, 1.0, 0.0, "plastic", "unclassified"), 1.4, 0.2, 0.0)
    ])))
    scenes.append(("driveway_crown", lambda: make_driveway_crown(6.0, 8.0, 0.07, 0.0)))
    scenes.append(("guardrail", lambda: make_guardrail(10.0, 0.0)))
    scenes.append(("bollards", lambda: make_bollards(6, 0.0)))
    scenes.append(("street_banked", lambda: make_street_patch(8.0, 10.0, 0.0, slope=(0.02, 0.0))))

    # ten new scenes
    scenes.append(("stop_sign", lambda: make_stop_sign(0.0)))
    scenes.append(("streetlight", lambda: make_streetlight(0.0)))
    scenes.append(("fire_hydrant", lambda: make_fire_hydrant(0.0)))
    scenes.append(("mailbox_cluster", lambda: make_mailbox_cluster(0.0)))
    scenes.append(("bike_rack_u", lambda: make_bike_rack_u(0.0)))
    scenes.append(("trash_can", lambda: make_trash_can(0.0)))
    scenes.append(("picnic_table", lambda: make_picnic_table(0.0)))
    scenes.append(("median_island", lambda: make_median_island_with_curbs(4.0, 1.6, 0.12, 0.0)))
    scenes.append(("ped_ramp_tactile", lambda: make_ped_ramp_with_tactile(1.8, 2.0, 0.15, 0.0)))
    scenes.append(("storm_inlet_grate", lambda: make_storm_inlet_grate(0.0)))
    return scenes

# -----------------------------
# IO
# -----------------------------

def write_las(filepath, pts):
    # point format 3 supports color and time, we use it for flexibility
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [0.0, 0.0, 0.0]
    las = laspy.LasData(header)

    las.x = pts["x"]
    las.y = pts["y"]
    las.z = pts["z"]
    las.intensity = pts["intensity"]

    cls = pts.get("cls", pts.get("class"))
    if cls is None:
        raise KeyError("points missing classification key cls or class")
    las.classification = cls.astype(np.uint8)

    if ADD_RGB and all(k in pts for k in ["red", "green", "blue"]):
        las.red = pts["red"].astype(np.uint16)
        las.green = pts["green"].astype(np.uint16)
        las.blue = pts["blue"].astype(np.uint16)

    las.return_number = np.ones(pts["x"].size, dtype=np.uint8)
    las.number_of_returns = np.ones(pts["x"].size, dtype=np.uint8)
    las.write(filepath)

# -----------------------------
# Main
# -----------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenes = build_scene_library()

    combined_parts = []
    legend_rows = []

    for idx, (name, maker) in enumerate(scenes):
        pc = maker()
        pc = ensure_cls(pc)
        row = idx // GRID_COLUMNS
        col = idx % GRID_COLUMNS
        ox = col * CELL_SIZE
        oy = row * CELL_SIZE
        pc = translate(pc, ox, oy, 0.0)

        combined_parts.append(pc)
        legend_rows.append({
            "index": idx + 1,
            "name": name,
            "origin_x": ox,
            "origin_y": oy,
            "approx_points": int(pc["x"].size)
        })

        ext = "laz" if WRITE_LAZ else "las"
        out_path = OUT_DIR / f"{idx+1:02d}_{name}.{ext}"
        temp_las = OUT_DIR / f"{idx+1:02d}_{name}.las"
        write_las(temp_las, pc)
        if WRITE_LAZ:
            las = laspy.read(temp_las)
            las.write(out_path)
            temp_las.unlink()
        else:
            temp_las.rename(out_path)
        print(f"Wrote {out_path}")

    all_pts = stack_fields(combined_parts)
    ext = "laz" if WRITE_LAZ else "las"
    combined_tmp = OUT_DIR / "combined_sandbox.las"
    combined_out = OUT_DIR / f"combined_sandbox.{ext}"
    write_las(combined_tmp, all_pts)
    if WRITE_LAZ:
        las = laspy.read(combined_tmp)
        las.write(combined_out)
        combined_tmp.unlink()
    else:
        combined_tmp.rename(combined_out)
    print(f"Wrote {combined_out}")

    with open(OUT_DIR / "legend.json", "w") as f:
        json.dump(legend_rows, f, indent=2)
    print("Wrote legend.json")

    total_pts = all_pts["x"].size
    unique_classes, class_counts = np.unique(all_pts["cls"], return_counts=True)
    print(f"Total points, {total_pts}")
    print("Class histogram, {class, count}")
    for c, n in zip(unique_classes.tolist(), class_counts.tolist()):
        print(f"{c}, {n}")

    if SHOW_PREVIEW and HAVE_O3D:
        pts = np.vstack([all_pts["x"], all_pts["y"], all_pts["z"]]).T
        if ADD_RGB and all(k in all_pts for k in ["red", "green", "blue"]):
            colors = np.vstack([
                all_pts["red"],
                all_pts["green"],
                all_pts["blue"]
            ]).T.astype(np.float32) / 65535.0
        else:
            colors = (all_pts["intensity"].astype(np.float32) / 65535.0).reshape(-1, 1)
            colors = np.repeat(colors, 3, axis=1)

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
