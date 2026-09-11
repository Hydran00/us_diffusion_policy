"""Extract only the two meshes needed for geometric liver localization."""

import hashlib
import json
import subprocess
import uuid
from pathlib import Path


def extract_meshes(output, image="i4h_sim_build:ultrasound-simulator"):
    """Copy from a stopped container. No GPU, network, or simulator process."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    identity = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    name = "us-dp-mesh-export-" + uuid.uuid4().hex[:12]
    subprocess.run(["docker", "create", "--name", name, image], capture_output=True, check=True)
    try:
        output.mkdir(parents=True)
        for filename in ("Skin.obj", "Liver.obj"):
            subprocess.run(
                ["docker", "cp", f"{name}:/opt/ultrasound-mesh/{filename}", str(output / filename)],
                check=True,
            )
        metadata = {"image": image, "image_id": identity, "mesh_units": "mm", "sha256": {}}
        for filename in ("Skin.obj", "Liver.obj"):
            metadata["sha256"][filename] = sha256(output / filename)
        (output / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        subprocess.run(["docker", "rm", name], capture_output=True, check=True)
    return output


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
