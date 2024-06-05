import xml.etree.ElementTree as ET
from collections import namedtuple

import numpy as np
import nvisii

from nvisii_base_parser import BaseParser
# from robosuite.renderers.nvisii.nvisii_utils import load_object
from nvisii_utils import load_object
from robosuite.utils.mjcf_utils import string_to_array

Components = namedtuple(
    "Components", ["obj", "geom_index", "element_id", "parent_body_name", "geom_pos", "geom_quat", "dynamic"]
)


class Parser(BaseParser):
    def __init__(self, renderer, env, segmentation_type):
        """
        Parse the mujoco xml and initialize NVISII renderer objects.
        Args:
            env (Mujoco env): Environment to parse
        """

        super().__init__(renderer, env)
        self.segmentation_type = segmentation_type
        # self.create_class_mapping()
        self.components = {}

        self.geom_typetoname = {0:"plane", 1: "heightfield", 2: "sphere", 3: "capsule", 
                                4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh", 8: "sdf"}

    # WORKS!
    def parse_textures(self):
        """
        Parse and load all textures and store them
        """

        self.texture_attributes = {}
        self.texture_id_mapping = {}

        for texture in self.xml_root.iter("texture"):
            texture_type = texture.get("type")
            texture_name = texture.get("name")
            texture_file = "/home/adityadutt/Desktop/robot-collab-v2/rocobench/envs/assets/objects/textures/white-wood.png" #texture.get("file") 
            texture_rgb = texture.get("rgb1")

            if texture_file is not None:
                # print(texture_file)
                texture.attrib["file"] = texture_file
                self.texture_attributes[texture_name] = texture.attrib
            else:
                color = np.array(string_to_array(texture_rgb))
                self.texture_id_mapping[texture_name] = (color, texture_type)

    # WORKS; Note: Many materials do not have textures.
    def parse_materials(self):
        """
        Parse all materials and use texture mapping to initialize materials
        """

        self.material_texture_mapping = {}
        for material in self.xml_root.iter("material"):
            material_name = material.get("name")
            texture_name = material.get("texture")
            self.material_texture_mapping[material_name] = texture_name

    # WORKS
    def parse_meshes(self):
        """
        Create mapping of meshes.
        """
        self.meshes = {}
        for mesh in self.xml_root.iter("mesh"):
            self.meshes[mesh.get("name")] = mesh.attrib

    # TBD
    def parse_geometries(self):
        """
        Iterate through each goemetry and load it in the NVISII renderer.
        """
        self.parse_meshes()
        element_id = 0
        repeated_names = {}
        block_rendering_objects = ["VisualBread_g0", "VisualCan_g0", "VisualCereal_g0", "VisualMilk_g0"]

        self.entity_id_class_mapping = {}

        for i in range(self.env.physics.model.ngeom):
            # print(i)


            geom = self.env.physics.model.geom(i)

            #Modify meshes to store files corresponding to geom names
            

            parent_body = self.env.physics.model.body(geom.bodyid.item())
            parent_body_name = parent_body.name

            geom_name = geom.name
            
            geom_type = self.geom_typetoname[geom.type.item()]
            # print("Geom Types stored: ", geom.type.item(), geom_type)

            geom_rgba = geom.rgba if geom.rgba is not None else None

            if geom_name is None:
                if parent_body_name in repeated_names:
                    geom_name = parent_body_name + str(repeated_names[parent_body_name])
                    repeated_names[parent_body_name] += 1
                else:
                    geom_name = parent_body_name + "0"
                    repeated_names[parent_body_name] = 1

            # if (geom.group.item() == 3) and geom_type != "plane") or ("collision" in geom_name):
            if (geom.group.item() == 3) or ("collision" in geom_name):
                continue

            if "floor" in geom_name or "wall" in geom_name: #or geom_name in block_rendering_objects:
                continue

            geom_quat = geom.quat

            # handling special case of bins arena
            # if "bin" in parent_body_name:
            #     geom_pos = geom.pos + parent_body.pos
            # else:
            if geom_type == "mesh":
                geom_pos = parent_body.pos
            else:
                geom_pos = geom.pos

            # print(geom.pos)
            
            # if geom_type == "mesh":
            #     geom_scale = string_to_array(self.meshes[geom.get("mesh")].get("scale", "1 1 1"))
            # else:
            #     geom_scale = [1, 1, 1]
            # geom_size = string_to_array(geom.get("size", "1 1 1"))
            geom_scale = [1, 1, 1]
            geom_size = geom.size

            # geom_mat = geom.get("material")

            # tags = ["bin"]
            # dynamic = True
            # if self.tag_in_name(geom_name, tags):
            #     dynamic = False

            # geom_tex_name = None
            # geom_tex_file = None
            geom_mesh = None
            for geom_index, g in enumerate(self.xml_root.iter("geom")):
                # print(g.get("name"), geom_name)
                if g.get("mesh") is not None and g.get("name") == geom_name:
                    geom_mesh = g.get("mesh")

            # if geom_mat is not None:
            #     geom_tex_name = self.material_texture_mapping[geom_mat]

            #     if geom_tex_name in self.texture_attributes:
            #         geom_tex_file = self.texture_attributes[geom_tex_name]["file"]
            geom_mat = geom.matid if geom.matid > -1 else None
            geom_mat = self.env.physics.model.material(geom_mat).name if geom_mat is not None else None
            # tags = ["bin"]
            dynamic = True
            # if self.tag_in_name(geom_name, tags):
            #     dynamic = False

            geom_tex_name = None
            geom_tex_file = None

            if geom_mat is not None:
                geom_tex_name = self.material_texture_mapping[geom_mat]

                if geom_tex_name in self.texture_attributes:
                    geom_tex_file = self.texture_attributes[geom_tex_name]["file"]

            # class_id = self.get_class_id(geom_index, element_id)

            # load obj into nvisii
            # print("Loading Object")
            obj, entity_ids = load_object(
                geom=geom,
                geom_name=geom_name,
                geom_type=geom_type,
                geom_quat=geom_quat,
                geom_pos=geom_pos,
                geom_size=geom_size,
                geom_scale=geom_scale,
                geom_rgba=geom_rgba,
                geom_tex_name=geom_tex_name,
                geom_tex_file=geom_tex_file,
                geom_mesh=geom_mesh,
                class_id=0,  # change
                meshes=self.meshes,
            )

            element_id += 1

            for entity_id in entity_ids:
                self.entity_id_class_mapping[entity_id] = 0 ##CHANGE

            self.components[geom_name] = Components(
                obj=obj,
                geom_index=i, #geom idx
                element_id=element_id,
                parent_body_name=parent_body_name,
                geom_pos=geom_pos,
                geom_quat=geom_quat,
                dynamic=dynamic,
            )

        self.max_elements = element_id

    def create_class_mapping(self):
        """
        Create class name to index mapping for both semantic and instance
        segmentation.
        """
        self.class2index = {}
        for i, c in enumerate(self.env.model._classes_to_ids.keys()):
            self.class2index[c] = i
        self.class2index[None] = i + 1
        self.max_classes = len(self.class2index)

        self.instance2index = {}
        for i, instance_class in enumerate(self.env.model._instances_to_ids.keys()):
            self.instance2index[instance_class] = i
        self.instance2index[None] = i + 1
        self.max_instances = len(self.instance2index)

    def get_class_id(self, geom_index, element_id):
        """
        Given index of the geom object get the class id based on
        self.segmentation type.
        """

        if self.segmentation_type[0] == None or self.segmentation_type[0][0] == "element":
            class_id = element_id
        elif self.segmentation_type[0][0] == "class":
            class_id = self.class2index[self.env.model._geom_ids_to_classes.get(geom_index)]
        elif self.segmentation_type[0][0] == "instance":
            class_id = self.instance2index[self.env.model._geom_ids_to_instances.get(geom_index)]

        return class_id

    def tag_in_name(self, name, tags):

        """
        Checks if one of the tags in body tags in the name

        Args:
            name (str): Name of geom element.

            tags (array): List of keywords to check from.
        """
        for tag in tags:
            if tag in name:
                return True
        return False
