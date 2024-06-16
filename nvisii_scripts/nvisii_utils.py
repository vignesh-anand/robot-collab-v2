import math
import os

import numpy as np
import nvisii
import xml.etree.ElementTree as ET

def load_object(
    geom,
    geom_name,
    geom_type,
    geom_quat,
    geom_pos,
    geom_size,
    geom_scale,
    geom_rgba,
    geom_tex_name,
    geom_tex_file,
    geom_mesh,
    class_id,
    meshes,
):
    """
    Function that initializes the meshes in the memory.

    Args:
        geom (MJCF geom element): Geom Object from MJCF model to load

        geom_name (str): Name for the object.

        geom_type (str): Type of the object. Types include "box", "cylinder", or "mesh".

        geom_quat (array): Quaternion (wxyz) of the object.

        geom_pos (array): Position of the object.

        geom_size (array): Size of the object.

        geom_scale (array): Scale of the object.

        geom_rgba (array): Color of the object. This is only used if the geom type is not
                           a mesh and there is no specified material.

        geom_tex_name (str): Name of the texture for the object

        geom_tex_file (str): File of the texture for the object

        geom_mesh (str): Name of the mesh associated with the geom

        class_id (int) : Class id for the component

        meshes (dict): Meshes for the object
    """

    robotiq_assets = ["base_mount", "base", "coupler", "driver", "follower", "pad", "silicone_pad", "spring_link"]
    primitive_types = ["box", "cylinder"]
    component = None


    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Ensure parent_dir ends with a separator
    if not parent_dir.endswith(os.path.sep):
        parent_dir += os.path.sep
    path_dir = os.path.join(parent_dir, "rocobench", "envs", "assets/")

    if geom_type == "box":
        # print("Box Component")
        component = nvisii.entity.create(
            name=geom_name,
            mesh=nvisii.mesh.create_box(name=geom_name, size=nvisii.vec3(geom_size[0], geom_size[1], geom_size[2])),
            transform=nvisii.transform.create(geom_name),
            material=nvisii.material.create(geom_name),
        )

    elif geom_type == "cylinder":
        # print("Cyl Component")
        component = nvisii.entity.create(
            name=geom_name,
            mesh=nvisii.mesh.create_capped_cylinder(name=geom_name, radius=geom_size[0], size=geom_size[1]),
            transform=nvisii.transform.create(geom_name),
            material=nvisii.material.create(geom_name),
        )

    elif geom_type == "sphere":
        # print("Sph Component")
        component = nvisii.entity.create(
            name=geom_name,
            mesh=nvisii.mesh.create_sphere(name=geom_name, radius=geom_size[0]),
            transform=nvisii.transform.create(geom_name),
            material=nvisii.material.create(geom_name),
        )

    elif geom_type == "mesh":
        filename = meshes[geom_mesh]["file"]
        filename = filename.split('-')[0] + "." + filename.split('.')[-1]
        # print("Entered Mesh,", geom_name)
        dir = ""
        if "panda" in geom_name:
            dir = path_dir + "panda/assets/"
        if "ur5e" in geom_name:
            dir = path_dir + "ur5e_robotiq/"
            
            robotiq_flag = False
            for f in robotiq_assets:
                if f == filename.split('.')[0]:
                    dir = dir + "robotiq_assets/"
                    robotiq_flag = True

            if not robotiq_flag:
                dir = dir + "ur5e_assets/"    
    
            
        # elif "ur5e" in geom_name:
        #     for f in robotiq_assets:
        #         if f in filename:
        #             path_dir += "ur5e_robotiq/robotiq_assets/"
        #         else:
        #             path_dir += "ur5e_robotiq/ur5e_assets/"

        # print("Before:", filename)
        filename = dir + filename
        # print("After:", filename)
        # print("Position:", geom_pos)
        component = nvisii.import_scene(
            file_path=filename,
            position=nvisii.vec3(geom_pos[0], geom_pos[1], geom_pos[2]),
            scale=(geom_scale[0], geom_scale[1], geom_scale[2]),
            rotation=nvisii.quat(geom_quat[0], geom_quat[1], geom_quat[2], geom_quat[3]),
        )

    entity_ids = []
    if isinstance(component, nvisii.scene):
        for i in range(len(component.entities)):
            entity_ids.append(component.entities[i].get_id())
    else:
        entity_ids.append(component.get_id())

    if geom_type in primitive_types:
        component.get_transform().set_position(nvisii.vec3(float(geom_pos[0]), float(geom_pos[1]), float(geom_pos[2])))

    if geom_tex_file is not None and geom_tex_name is not None and geom_type != "mesh":

        texture = nvisii.texture.get(geom_tex_name)

        if texture is None:
            texture = nvisii.texture.create_from_file(name=geom_tex_name, path=geom_tex_file)

        component.get_material().set_base_color_texture(texture)
    else:
        if "gripper" in geom_name:
            if geom_rgba is not None:
                if isinstance(component, nvisii.scene):
                    for entity in component.entities:
                        entity.get_material().set_base_color(nvisii.vec3(geom_rgba[0], geom_rgba[1], geom_rgba[2]))
                else:
                    component.get_material().set_base_color(nvisii.vec3(geom_rgba[0], geom_rgba[1], geom_rgba[2]))
            elif "hand" in geom_name:
                for entity in component.entities:
                    entity.get_material().set_base_color(nvisii.vec3(0.05, 0.05, 0.05))

    return component, entity_ids