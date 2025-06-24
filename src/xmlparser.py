import xml.etree.ElementTree as ET
import glob
import os
import re

def find_all(name, path, ext=None, split=False):
    """
    If split=False: return a single relative match (first one) or None.
    If split=True: return all relative matches of the form name###.ext (or name###.*)
    """
    base = os.path.abspath(os.path.expanduser(path))
    if split:
        if ext:
            pattern = os.path.join(base, "**", f"{name}[0-9][0-9][0-9].{ext}")
        else:
            pattern = os.path.join(base, "**", f"{name}[0-9][0-9][0-9].*")
        matches = glob.glob(pattern, recursive=True)
        matches.sort()
        return [os.path.relpath(m, base) for m in matches]
    else:
        # single‐match mode
        if ext:
            pattern = os.path.join(base, "**", f"{name}.{ext}")
        else:
            pattern = os.path.join(base, "**", f"{name}.*")
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            return []
        return [os.path.relpath(matches[0], base)]

def meshfinder(tree):
    root = tree.getroot()
    compiler = root.find('compiler')
    meshdir = compiler.get('meshdir', '') if compiler is not None else ''

    for asset in root.findall('.//asset'):
        for mf in list(asset.findall('meshfinder')):
            directory = mf.get('directory', '')
            search_root = os.path.join(meshdir, directory)
            idx = list(asset).index(mf)

            new_meshes = []
            for mesh in mf.findall('mesh'):
                name  = mesh.get('name')
                alias = mesh.get('alias')
                ext   = mesh.get('ext')
                split = mesh.get('split', 'false').lower() in ('true','1','yes')

                rel_paths = find_all(name, search_root, ext=ext, split=split)
                if not rel_paths:
                    raise FileNotFoundError(f"No mesh files found for {name} (split={split}) in {search_root}")

                for rel in rel_paths:
                    suffix = ''
                    if split:
                        stem = os.path.splitext(os.path.basename(rel))[0]
                        m = re.match(re.escape(name) + r'(\d{3})$', stem)
                        suffix = m.group(1) if m else ''

                    if split and alias:
                        mesh_name = f"{alias}{suffix}"
                    elif split:
                        mesh_name = f"{name}{suffix}"
                    else:
                        mesh_name = alias if alias else name

                    mesh_attrib = {
                        'file': os.path.normpath(os.path.join(directory, rel)),
                        'name': mesh_name
                    }
                    new_meshes.append(ET.Element('mesh', mesh_attrib))

            asset.remove(mf)
            for offset, new in enumerate(new_meshes):
                asset.insert(idx + offset, new)

def indent(elem, level=0, indent_str="  "):
    """
    Recursively adds indentation to an ElementTree in-place.
    """
    i = "\n" + level * indent_str
    if len(elem):
        # if no text or only whitespace, set the indentation
        if not elem.text or not elem.text.strip():
            elem.text = i + indent_str
        # recurse on children
        for child in elem:
            indent(child, level+1, indent_str)
        # ensure the last child has a tail
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = i
    else:
        # for leaf nodes, ensure there's a newline after them
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

if __name__ == '__main__':
    # Example: read 'model_template.xml', process, and print result
    filepath = 'models/meshfinder_test.xml'
    tree = ET.parse(filepath)
    meshfinder(tree)

    root = tree.getroot()
    indent(root)

    tree.write(
        'models/parsed.xml',
        encoding='utf-8',
        xml_declaration=True,
        method='xml'
    )
