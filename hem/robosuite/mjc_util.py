import robosuite
import hem
import os
import xml.etree.ElementTree as ET


def postprocess_model_xml(xml_str):
    """
    This function postprocesses the model.xml collected from a MuJoCo demonstration
    in order to make sure that the STL files can be found.
    """
    robosuite_path = os.path.split(robosuite.__file__)[0].split('/')
    hem_path =  os.path.split(hem.__file__)[0].split('/')

    # replace mesh and texture file paths
    tree = ET.fromstring(xml_str)
    root = tree
    asset = root.find("asset")
    meshes = asset.findall("mesh")
    textures = asset.findall("texture")
    all_elements = meshes + textures

    for elem in all_elements:
        old_path = elem.get("file")
        if old_path is None:
            continue
        old_path_split = old_path.split("/")

        if 'hem' in old_path:
            ind = [loc for loc, val in enumerate(old_path_split) if val == "hem"][0]
            path_split = hem_path
        else:
            ind = max(
                loc for loc, val in enumerate(old_path_split) if val == "robosuite"
            )  # last occurrence index
            path_split = robosuite_path
        new_path_split = path_split + old_path_split[ind + 1 :]
        new_path = "/".join(new_path_split)
        elem.set("file", new_path)

    return ET.tostring(root, encoding="utf8").decode("utf8")
