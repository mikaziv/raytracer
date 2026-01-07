import argparse
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere


def parse_camera_and_one_sphere(scene_path: str):
    camera = None
    sphere = None

    with open(scene_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            obj_type = parts[0]
            params = [float(x) for x in parts[1:]]

            if obj_type == "cam":
                if len(params) != 11:
                    raise ValueError(f"cam expects 11 numbers, got {len(params)}: {line}")
                camera = Camera(
                    position=np.array(params[0:3], dtype=float),
                    look_at=np.array(params[3:6], dtype=float),
                    up_vector=np.array(params[6:9], dtype=float),
                    screen_distance=float(params[9]),
                    screen_width=float(params[10]),
                )

            elif obj_type == "sph" and sphere is None:
                if len(params) != 5:
                    raise ValueError(f"sph expects 5 numbers, got {len(params)}: {line}")
                sphere = Sphere(
                    position=np.array(params[0:3], dtype=float),
                    radius=float(params[3]),
                    material_index=int(params[4]),
                )

            if camera is not None and sphere is not None:
                break

    if camera is None:
        raise ValueError("No 'cam' line found in scene file.")
    if sphere is None:
        raise ValueError("No 'sph' line found in scene file.")

    return camera, sphere


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
    image = Image.fromarray(np.uint8(image_array))

    # saving the image to the specified path:
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


def compute_output_image(camera, scene_settings, objects, image_array):
    pass


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
    image_array = np.zeros((args.height, args.width, 3))
    image_array.fill(255)  # white background instead of black for visibility
    # computing the image by ray tracing:
    compute_output_image(camera, scene_settings, objects, image_array)

    # Save the output into an image:
    save_image(image_array, args.output_image)


if __name__ == '__main__':
    main()







