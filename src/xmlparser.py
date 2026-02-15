"""XML preprocessing helpers for mesh expansion and flex pin generation."""

import xml.etree.ElementTree as ET
import glob
import os
import re
import meshio
import trimesh
import numpy as np

def find_all(name, path, ext=None, split=False):
    """
    If split=False: return a list with the first relative match (or [] if none).
    If split=True: return all relative matches of the form name###.ext (or name###.*).
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
        if ext:
            pattern = os.path.join(base, "**", f"{name}.{ext}")
        else:
            pattern = os.path.join(base, "**", f"{name}.*")
        matches = glob.glob(pattern, recursive=True)
        return [os.path.relpath(matches[0], base)] if matches else []

def meshfinder(tree):
    """
    Expands all <meshfinder> blocks under <asset>, returning a dict `splits`
    where splits[base_or_alias] = [ "base000", "base001", ... ] for each split mesh.
    """
    root = tree.getroot()
    compiler = root.find('compiler')
    meshdir = compiler.get('meshdir', '') if compiler is not None else ''

    splits = {}

    for asset in root.findall('.//asset'):
        for mf in list(asset.findall('meshfinder')):
            directory   = mf.get('directory', '')
            search_root = os.path.join(meshdir, directory)
            idx         = list(asset).index(mf)

            new_meshes = []
            for mesh in mf.findall('mesh'):
                base  = mesh.get('name')
                alias = mesh.get('alias')
                ext   = mesh.get('ext')
                do_split = mesh.get('split', 'false').lower() in ('true','1','yes')

                rel_paths = find_all(base, search_root, ext=ext, split=do_split)
                if not rel_paths:
                    raise FileNotFoundError(f"No mesh files found for {base} (split={do_split}) in {search_root}")

                if do_split:
                    full_names = []
                    for rel in rel_paths:
                        stem = os.path.splitext(os.path.basename(rel))[0]
                        full_names.append(stem)
                    # record under alias if present, else base
                    key = alias or base
                    splits[key] = full_names

                for rel in rel_paths:
                    # extract suffix for split
                    suffix = ''
                    if do_split:
                        stem = os.path.splitext(os.path.basename(rel))[0]
                        m = re.match(re.escape(base) + r'(\d{3})$', stem)
                        suffix = m.group(1) if m else ''

                    # determine output mesh name
                    if do_split:
                        # alias + suffix if alias, else base + suffix
                        mesh_name = f"{(alias or base)}{suffix}"
                    else:
                        mesh_name = alias or base

                    mesh_attrib = {
                        'file': os.path.normpath(os.path.join(directory, rel)),
                        'name': mesh_name
                    }
                    new_meshes.append(ET.Element('mesh', mesh_attrib))

            # splice in and remove placeholder
            asset.remove(mf)
            for offset, new in enumerate(new_meshes):
                asset.insert(idx + offset, new)

    return splits

def expand_geoms(tree, splits):
    """
    For any <geom mesh="X" …> where X in splits, replace it by one <geom> per split:
    - mesh and name get alias+suffix (mesh_ref is alias or base)
    - mass is divided equally among them
    - all other attributes are preserved
    """
    root = tree.getroot()

    # collect worldbody and all bodies
    containers = []
    wb = root.find('worldbody')
    if wb is not None:
        containers.append(wb)
    containers.extend(root.findall('.//body'))

    for parent in containers:
        for geom in list(parent.findall('geom')):
            mesh_ref = geom.get('mesh')
            if mesh_ref in splits:
                mass = float(geom.get('mass', '0'))
                variants = splits[mesh_ref]
                count = len(variants)
                if count == 0:
                    continue
                new_mass = mass / count
                idx = list(parent).index(geom)

                new_geoms = []
                for stem in variants:
                    # extract the numeric suffix from the stored stem
                    m = re.search(r'(\d{3})$', stem)
                    suffix = m.group(1) if m else ''

                    new_name = f"{mesh_ref}{suffix}"
                    attrib = geom.attrib.copy()
                    attrib['mesh'] = new_name
                    attrib['name'] = new_name
                    attrib['mass'] = str(new_mass)
                    new_geoms.append(ET.Element('geom', attrib))

                parent.remove(geom)
                for offset, ng in enumerate(new_geoms):
                    parent.insert(idx + offset, ng)

def findpins(surfacemesh, flex):
    """Find flex node ids that lie on/near a surface mesh."""
    msh = meshio.read(flex)
    points = msh.points

    surface = trimesh.load(surfacemesh, force='mesh')
    distances = trimesh.proximity.signed_distance(surface, points)

    # Keep tolerance fixed for now; exposed configuration was deferred intentionally.
    tol = 1e-3
    mask = np.abs(distances) < tol

    ids = np.nonzero(mask)[0]

    return ids

# TODO: account for mesh-level translation offsets when computing pin IDs.
def flexpin(tree):
    """Replace <pinmesh> declarations with explicit <pin id="..."> entries."""
    root = tree.getroot()

    compiler = root.find('compiler')
    meshdir = compiler.get('meshdir', '')

    asset = root.find('.//asset')
    mesh_lookup = {}
    if asset is not None:
        for m in asset.findall('mesh'):
            name = m.get('name')
            file = m.get('file')
            if name and file:
                mesh_lookup[name] = file

    for flex in root.findall('.//flexcomp'):
        flex_file = flex.get('file', '')

        for pinmesh in list(flex.findall('pinmesh')):
            mesh_name = pinmesh.get('mesh')
            mesh_file = mesh_lookup.get(mesh_name)
            if mesh_file is None:
                raise KeyError(f"Mesh '{mesh_name}' not found in <asset>")

            pin_ids = findpins(os.path.join(meshdir, mesh_file), os.path.join(meshdir, flex_file))

            idx = list(flex).index(pinmesh)
            flex.remove(pinmesh)
            for offset, pid in enumerate(pin_ids):
                pin = ET.Element('pin', {'id': str(pid)})
                flex.insert(idx + offset, pin)

def parse(filepath):
    """Parse MuJoCo XML and apply meshfinder, split geom expansion, and flex pinning."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    splits = meshfinder(tree)
    expand_geoms(tree, splits)
    flexpin(tree)

    return ET.tostring(root, encoding='unicode',xml_declaration=True)

def indent(elem, level=0, indent_str="  "):
    i = "\n" + level * indent_str
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + indent_str
        for child in elem:
            indent(child, level+1, indent_str)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

if __name__ == '__main__':
    filepath = 'models/phantom_barrutia.xml'
    tree = ET.parse(filepath)
    splits = meshfinder(tree)
    expand_geoms(tree, splits)
    flexpin(tree)

    root = tree.getroot()
    indent(root)
    tree.write(
        'models/parsed.xml',
        encoding='utf-8',
        xml_declaration=True,
        method='xml'
    )
