import xml.etree.ElementTree as ET
import glob
import os

def find(name, path):
    base = os.path.abspath(os.path.expanduser(path))
    full_pattern = os.path.join(base, "**", name)
    matches = glob.glob(full_pattern, recursive=True)
    if not matches:
        return None
    return os.path.relpath(matches[0], base)

def meshfinder(tree):
    root = tree.getroot()

    compiler = root.find('compiler')
    if compiler is not None:
        meshdir = compiler.get('meshdir', '')

    for asset in root.findall('.//asset'):
        for meshfinder in list(asset.findall('meshfinder')):
            directory = meshfinder.get('directory', '')
            directory = os.path.join(meshdir, directory)
            idx = list(asset).index(meshfinder)

            new_meshes = []
            for mesh in meshfinder.findall('mesh'):
                name = mesh.get('name')
                alias = mesh.get('alias')
                ext  = mesh.get('ext')

                pattern = f"{name}.{ext}" if ext else f"{name}.*"
                match = find(pattern, directory)

                mesh_attrib = {
                    'file': os.path.relpath(os.path.join(directory, match),meshdir),
                    'name': alias if alias else name
                }
                new_meshes.append(ET.Element('mesh', mesh_attrib))

            asset.remove(meshfinder)
            for offset, new in enumerate(new_meshes):
                asset.insert(idx + offset, new)

if __name__ == '__main__':
    # Example: read 'model_template.xml', process, and print result
    filepath = 'models/meshfinder_test.xml'
    tree = ET.parse(filepath)
    meshfinder(tree)

    tree.write(
        'models/parsed.xml',
        encoding='utf-8',
        xml_declaration=True,
        method='xml'
    )
