import argparse
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
import scene_settings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere

EPS = 1e-13

def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array, output_path_and_file_name):
    image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))
    image.save(output_path_and_file_name)

def print_program_args(args):
    print(f'\n{"="*20} program arguments: {"="*20}\n {args}.\n')


def print_parsed_scene_file_data(camera, scene_settings, objects):
    print(f'{"="*20} parsed scene file data: {"="*20}')
    print("Camera:", camera.__dict__)
    print("\nScene Settings:", scene_settings.__dict__)
    print("\nObjects in the scene:")
    for obj in objects:
        # print a short summary for each object
        print("\t", type(obj).__name__, obj.__dict__)
    print()

###########################
## Ray tracing functions ##
###########################

def normalize(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def reflect(L, N):
    # L,N normalized
    return 2.0 * np.dot(N, L) * N - L

def intersect_sphere(ray_o, ray_d, sph):
    C = np.array(sph.position, float)
    r = float(sph.radius)

    oc = ray_o - C
    a = np.dot(ray_d, ray_d)  # =1 if ray_d normalized
    b = 2.0 * np.dot(oc, ray_d)
    c = np.dot(oc, oc) - r*r
    disc = b*b - 4*a*c
    if disc < 0:
        return None

    s = np.sqrt(disc)
    t1 = (-b - s) / (2*a)
    t2 = (-b + s) / (2*a)

    t = None
    if t1 > EPS:
        t = t1
    elif t2 > EPS:
        t = t2
    else:
        return None

    P = ray_o + t * ray_d
    N = normalize(P - C)
    return t, P, N, sph.material_index

def intersect_plane(ray_o, ray_d, pln):
    N = normalize(np.array(pln.normal, float))
    c = float(pln.offset)

    denom = np.dot(ray_d, N)
    if abs(denom) < EPS:
        return None

    t = (c - np.dot(ray_o, N)) / denom
    if t <= EPS:
        return None

    P = ray_o + t * ray_d
    # normal pointing against the ray
    if np.dot(N, -ray_d) < 0:
        N = -N
    return t, P, N, pln.material_index

def intersect_cube(ray_o, ray_d, box):
    C = np.array(box.position, float)
    s = float(box.scale)
    h = s / 2.0
    bmin = C - h
    bmax = C + h

    tmin = -float("inf")
    tmax = float("inf")

    for axis in range(3):
        if abs(ray_d[axis]) < EPS:
            #if ray is parallel to the slabs
            if ray_o[axis] < bmin[axis] or ray_o[axis] > bmax[axis]:
                return None
        else:
            inv = 1.0 / ray_d[axis]
            t1 = (bmin[axis] - ray_o[axis]) * inv
            t2 = (bmax[axis] - ray_o[axis]) * inv
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None

    t = tmin if tmin > EPS else tmax
    if t <= EPS:
        return None

    P = ray_o + t * ray_d
    # calculate normal at intersection point
    N = np.zeros(3, float)
    tol = EPS
    if abs(P[0] - bmin[0]) < tol: N = np.array([-1, 0, 0], float)
    elif abs(P[0] - bmax[0]) < tol: N = np.array([ 1, 0, 0], float)
    elif abs(P[1] - bmin[1]) < tol: N = np.array([0, -1, 0], float)
    elif abs(P[1] - bmax[1]) < tol: N = np.array([0,  1, 0], float)
    elif abs(P[2] - bmin[2]) < tol: N = np.array([0, 0, -1], float)
    else: N = np.array([0, 0,  1], float)

    if np.dot(N, -ray_d) < 0:
        N = -N

    return t, P, N, box.material_index

def closest_hit(ray_o, ray_d, surfaces):
    best = None
    best_t = float("inf")

    for obj in surfaces:
        if obj.__class__.__name__ == "Sphere":
            hit = intersect_sphere(ray_o, ray_d, obj)
        elif obj.__class__.__name__ == "InfinitePlane":
            hit = intersect_plane(ray_o, ray_d, obj)
        elif obj.__class__.__name__ == "Cube":
            hit = intersect_cube(ray_o, ray_d, obj)
        else:
            continue

        if hit is None:
            continue

        t, P, N, midx = hit
        if t < best_t:
            best_t = t
            best = (t, P, N, midx)

    return best  # or None

def is_occluded(ray_o, ray_d, max_dist, surfaces):
    # does any object block the ray
    for obj in surfaces:
        if obj.__class__.__name__ == "Sphere":
            hit = intersect_sphere(ray_o, ray_d, obj)
        elif obj.__class__.__name__ == "InfinitePlane":
            hit = intersect_plane(ray_o, ray_d, obj)
        elif obj.__class__.__name__ == "Cube":
            hit = intersect_cube(ray_o, ray_d, obj)
        else:
            continue

        if hit is None:
            continue
        t = hit[0]
        if EPS < t < max_dist:
            return True
    return False

def build_perp_basis(w):
    w = normalize(w)
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(w, a)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = normalize(np.cross(w, a))
    v = normalize(np.cross(w, u))
    return u, v

###########################
###### Shadow factor ######
###########################

def shadow_factor(P, N, light, scene_settings, surfaces):
    light_pos = np.array(light.position, float)
    si = float(light.shadow_intensity)

    # hard shadow
    if int(scene_settings.root_number_shadow_rays) <= 1 or float(light.radius) <= 0.0:
        Lvec = light_pos - P
        dist = np.linalg.norm(Lvec)
        if dist < EPS:
            return 1.0
        L = Lvec / dist
        o = P + N * EPS
        blocked = is_occluded(o, L, dist - EPS, surfaces)
        return 1.0 if not blocked else (1.0 - si)

    # soft shadow:
    Ngrid = int(scene_settings.root_number_shadow_rays)
    total = Ngrid * Ngrid
    hits = 0

    # plane perpendicular to light-to-point direction
    w = normalize(P - light_pos)
    u, v = build_perp_basis(w)

    # radius/width
    side = float(light.radius)
    cell = side / Ngrid
    half = side / 2.0

    for i in range(Ngrid):
        for j in range(Ngrid):
            rx = (i + np.random.rand()) * cell - half
            ry = (j + np.random.rand()) * cell - half
            sample = light_pos + u * rx + v * ry

            dir_to_sample = sample - P
            dist = np.linalg.norm(dir_to_sample)
            if dist < EPS:
                continue
            d = dir_to_sample / dist

            o = P + N * EPS
            if not is_occluded(o, d, dist - EPS, surfaces):
                hits += 1

    visible = hits / total
    # by shadow_intensity:
    return (1.0 - si) + si * visible



def phong_shade(P, N, ray_dir, material, lights, scene_settings, surfaces):
    kd = np.array(material.diffuse_color, float)
    ks = np.array(material.specular_color, float)
    shin = float(material.shininess)

    V = normalize(-ray_dir)
    out = np.zeros(3, float)

    for light in lights:
        light_pos = np.array(light.position, float)
        I = np.array(light.color, float)

        Lvec = light_pos - P
        dist = np.linalg.norm(Lvec)
        if dist < EPS:
            continue
        L = Lvec / dist

        nl = float(np.dot(N, L))
        if nl <= 0:
            continue

        sf = shadow_factor(P, N, light, scene_settings, surfaces)

        # diffuse
        diff = kd * I * max(0.0, nl)

        # specular (Phong)
        R = reflect(L, N)
        rv = max(0.0, float(np.dot(R, V)))
        spec = ks * (I * float(light.specular_intensity)) * (rv ** shin)

        out += sf * (diff + spec)

    return np.clip(out, 0.0, 1.0)


def trace_ray(ray_o, ray_d, surfaces, materials, lights, scene_settings, depth):
    if depth > int(scene_settings.max_recursions):
        return np.array(scene_settings.background_color, float)
    hit = closest_hit(ray_o, ray_d, surfaces)
    if hit is None:
        return np.array(scene_settings.background_color, float)
    _, P, N, midx = hit
    mat = materials[int(midx) - 1]
    local_color = phong_shade(P, N, ray_d, mat, lights, scene_settings, surfaces)
    refl = np.array(mat.reflection_color, float)
    transparency = float(mat.transparency)

    # Reflection
    reflection_color = np.zeros(3, float)
    if np.any(refl > EPS):
        reflect_dir = normalize(reflect(-ray_d, N))
        reflect_o = P + N * EPS
        reflection_color = trace_ray(reflect_o, reflect_dir, surfaces, materials, lights, scene_settings, depth + 1) * refl

    # Transparency
    if transparency > EPS:
        # Shoot the same ray just past the intersection point
        transmit_o = P + ray_d * EPS
        background_color = trace_ray(transmit_o, ray_d, surfaces, materials, lights, scene_settings, depth + 1)
        # Blend according to the assignment formula
        color = (
            background_color * transparency +
            local_color * (1 - transparency) +
            reflection_color
        )
        return np.clip(color, 0.0, 1.0)
    else:
        # Opaque: blend local and reflection only
        color = local_color * (1 - refl) + reflection_color
        return np.clip(color, 0.0, 1.0)

def compute_output_image(camera, scene_settings, objects, image_array):
    H, W, _ = image_array.shape
    materials = [o for o in objects if o.__class__.__name__ == "Material"]
    lights    = [o for o in objects if o.__class__.__name__ == "Light"]
    surfaces  = [o for o in objects if o.__class__.__name__ in ("Sphere", "InfinitePlane", "Cube")]
    pos = np.array(camera.position, float)
    look_at = np.array(camera.look_at, float)
    up_vec = np.array(camera.up_vector, float)
    forward = normalize(look_at - pos)
    right = -normalize(np.cross(forward, up_vec))
    up = -normalize(np.cross(right, forward))
    pc = pos + forward * float(camera.screen_distance)
    screen_w = float(camera.screen_width)
    screen_h = screen_w * (H / W)
    px = screen_w / W
    py = screen_h / H
    bg = np.array(scene_settings.background_color, float)
    for j in range(H):
        for i in range(W):
            x = ((i + 0.5) - W / 2.0) * px
            y = (H / 2.0 - (j + 0.5)) * py
            Pscreen = pc + right * x + up * y
            ray_o = pos
            ray_d = normalize(Pscreen - pos)
            color = trace_ray(ray_o, ray_d, surfaces, materials, lights, scene_settings, 0)
            image_array[j, i, :] = np.clip(color * 255.0, 0, 255)




def main():
    # handling program arguments:
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()
    
    # testing args parsing values:
    print_program_args(args)

    # Parse the scene file:
    camera, scene_settings, objects = parse_scene_file(args.scene_file)
    # printing parsed scene file data for debugging:
    print_parsed_scene_file_data(camera, scene_settings, objects)
    

    # Implementing the ray tracer:
    # creating 2d image array of the resulting image:
    image_array = np.zeros((args.height, args.width, 3), dtype=float)
    bg = np.array(scene_settings.background_color, float) * 255.0
    image_array[:] = bg
    # computing the image by ray tracing:
    compute_output_image(camera, scene_settings, objects, image_array)

    # Save the output into an image:
    save_image(image_array, args.output_image)


if __name__ == '__main__':
    main()







