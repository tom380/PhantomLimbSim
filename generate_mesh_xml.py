import argparse
from pathlib import Path

TEMPLATE = '        <mesh file="{file}" name="{name}"/>'


def sanitize(name: str) -> str:
    return name.replace(' ', '_').replace('-', '_').lower()


def generate_lines(mesh_dir: Path):
    for ext in ('*.stl', '*.STL', '*.obj', '*.OBJ'):
        for path in sorted(mesh_dir.glob(ext)):
            yield TEMPLATE.format(file=path.name, name=sanitize(path.stem))


def main():
    parser = argparse.ArgumentParser(description="Generate MJCF mesh assets")
    parser.add_argument('mesh_dir', type=Path, help='directory with mesh files')
    args = parser.parse_args()

    for line in generate_lines(args.mesh_dir):
        print(line)


if __name__ == '__main__':
    main()
