"""Generate the 57 mm Rubik-style pose cube used by the cube8 sim.

The upstream Gazebo asset this replaces (`osrf/gazebo_models/cube_20k`) is a
plain blue test cube: it carries the size and mass of a speed cube but not the
appearance a colour-based pose hypothesis needs. This generator rebuilds it
as a brand-neutral *stickerless* 3x3: one rigid body, a dark grooved core, and
**54 facelet plates** in the six standard colours.

Why one body and 54 plates, rather than 27 cubies:

* a single collision box gives one stable rigid body — no cubie-on-cubie
  contact jitter and no 27-body inertia to validate;
* each facelet is an independent visual with an exact position, so a face
  region can be projected, sampled and classified per sticker without relying
  on texture or UV layout;
* the plates sit proud of the core by half a thickness with a gap between them,
  which produces the dark separators a face sampler needs in order to avoid
  mixing two stickers into one mean colour.

Numbers are the same ones the reference design uses (0.057 m side, 0.0162 m
facelets with a 0.0012 m gap, 0.0006 m plate thickness, 0.08 kg). The model is
generated from constants: no mesh, no texture, no download, and the same input
always produces the same SDF.

Provenance: the design is adapted from `nekomario28/lime-interactive-transport`
(`pose_cube_asset.py`), which derives it from `osrf/gazebo_models/cube_20k`
(upstream author Mihai Dhola, CC BY 3.0). No upstream mesh is copied or
required; :func:`write_model` records the attribution alongside the generated
files.
"""

from __future__ import annotations

import math
from pathlib import Path

MODEL_NAME = 'lime_rubik_pose_cube'
SIDE_LENGTH_M = 0.057
MASS_KG = 0.08
FACELET_GRID_SIZE = 3
FACELET_COUNT = 6 * FACELET_GRID_SIZE * FACELET_GRID_SIZE
FACELET_SIZE_M = 0.0162
FACELET_GAP_M = 0.0012
FACELET_THICKNESS_M = 0.0006
CORE_RGB = (0.012, 0.012, 0.014)

UPSTREAM_REPOSITORY = 'https://github.com/osrf/gazebo_models.git'
UPSTREAM_COMMIT = '8163eb4b5e7e21985c6591d1c0bfb56468c0093f'
UPSTREAM_MODEL_PATH = 'cube_20k'
UPSTREAM_AUTHOR = 'Mihai Dhola'
UPSTREAM_LICENSE = 'Creative Commons Attribution 3.0 Unported'

#: Face order matches the cube8 PnP vertex labels: +x/-x, +y/-y, +z/-z.
FACE_SPECS = (
    ('pos_x', 'red', (0.91, 0.04, 0.035)),
    ('neg_x', 'orange', (1.00, 0.25, 0.015)),
    ('pos_y', 'blue', (0.02, 0.20, 0.92)),
    ('neg_y', 'green', (0.02, 0.72, 0.18)),
    ('pos_z', 'white', (0.96, 0.96, 0.94)),
    ('neg_z', 'yellow', (0.98, 0.84, 0.02)),
)


class RubikPoseCubeError(ValueError):
    """Raised when the generated model fails its own geometry checks."""


def _material(r, g, b, *, specular):
    ambient = f'{r * 0.82:.3f} {g * 0.82:.3f} {b * 0.82:.3f} 1'
    diffuse = f'{r:.3f} {g:.3f} {b:.3f} 1'
    highlight = f'{specular:.3f} {specular:.3f} {specular:.3f} 1'
    return (
        '<material>'
        f'<ambient>{ambient}</ambient>'
        f'<diffuse>{diffuse}</diffuse>'
        f'<specular>{highlight}</specular>'
        '</material>'
    )


def _visual(name, pose, size, color):
    pose_text = ' '.join(f'{value:.6f}' for value in (*pose, 0.0, 0.0, 0.0))
    size_text = ' '.join(f'{value:.6f}' for value in size)
    return (
        f'      <visual name="{name}">\n'
        f'        <pose>{pose_text}</pose>\n'
        f'        <geometry><box><size>{size_text}</size></box></geometry>\n'
        f'        {_material(*color, specular=0.18)}\n'
        '      </visual>'
    )


def facelet_pose_and_size(face):
    """Return the plate centre and size for every facelet of ``face``.

    The plates are laid out on a pitch of facelet + gap, so the visible gap
    between two neighbours is exactly ``FACELET_GAP_M``.
    """
    half = SIDE_LENGTH_M / 2.0
    normal = half + FACELET_THICKNESS_M / 2.0
    pitch = FACELET_SIZE_M + FACELET_GAP_M
    offsets = tuple(index * pitch for index in (-1, 0, 1))
    size = FACELET_SIZE_M
    thickness = FACELET_THICKNESS_M
    cells = []
    for row, first in enumerate(offsets):
        for column, second in enumerate(offsets):
            if face == 'pos_x':
                pose, box = (normal, second, -first), (thickness, size, size)
            elif face == 'neg_x':
                pose, box = (-normal, -second, -first), (thickness, size, size)
            elif face == 'pos_y':
                pose, box = (-second, normal, -first), (size, thickness, size)
            elif face == 'neg_y':
                pose, box = (second, -normal, -first), (size, thickness, size)
            elif face == 'pos_z':
                pose, box = (second, -first, normal), (size, size, thickness)
            elif face == 'neg_z':
                pose, box = (second, first, -normal), (size, size, thickness)
            else:
                raise RubikPoseCubeError(f'unknown face {face!r}')
            cells.append((row, column, pose, box))
    return tuple(cells)


def _facelet_visuals():
    visuals = []
    for face, color_name, color in FACE_SPECS:
        for row, column, pose, box in facelet_pose_and_size(face):
            name = f'face_{face}_{color_name}_r{row}_c{column}'
            visuals.append(_visual(name, pose, box, color))
    return tuple(visuals)


def build_model_sdf(*, static=False):
    """Return the full ``<sdf>`` document for the pose cube.

    ``static=True`` keeps the cube fixed in the world, which is what a
    camera-only colour probe needs: the pose is then exactly the spawn pose
    instead of something gravity decided.
    """
    inertia = MASS_KG * (SIDE_LENGTH_M ** 2) / 6.0
    facelets = '\n'.join(_facelet_visuals())
    side = SIDE_LENGTH_M
    contact = (
        '<contact><ode><kp>1000000</kp><kd>10</kd>'
        '<min_depth>0.0001</min_depth><max_vel>0.1</max_vel></ode></contact>'
    )
    return f'''<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{MODEL_NAME}">
    <static>{'true' if static else 'false'}</static>
    <allow_auto_disable>true</allow_auto_disable>
    <link name="body">
      <inertial>
        <mass>{MASS_KG}</mass>
        <inertia>
          <ixx>{inertia:.8f}</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>{inertia:.8f}</iyy><iyz>0</iyz><izz>{inertia:.8f}</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><box><size>{side} {side} {side}</size></box></geometry>
        <surface>
          <friction><ode><mu>0.9</mu><mu2>0.9</mu2></ode></friction>
          {contact}
        </surface>
      </collision>
      <visual name="grooved_core">
        <geometry><box><size>{side} {side} {side}</size></box></geometry>
        {_material(*CORE_RGB, specular=0.025)}
      </visual>
{facelets}
    </link>
  </model>
</sdf>
'''


def model_config_xml():
    """Return the accompanying ``model.config`` document."""
    return f'''<?xml version="1.0"?>
<model>
  <name>Lime Rubik-style 3x3 Pose Cube</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author>
    <name>{UPSTREAM_AUTHOR} (upstream cube_20k)</name>
    <email>noreply@example.invalid</email>
  </author>
  <description>
    A nominal 57 mm dynamic, brand-neutral stickerless 3x3 cube proxy: one
    rigid collision body and 54 facelet visuals in the six standard colours,
    for pose and face-colour experiments.
  </description>
</model>
'''


def attribution_text():
    """Return the provenance note stored next to the generated model."""
    return f'''# Upstream model attribution

This generated model is a design adaptation of `osrf/gazebo_models/cube_20k`,
the plain blue 57 mm test cube used for pose experiments.

- Repository: `{UPSTREAM_REPOSITORY}`
- Commit: `{UPSTREAM_COMMIT}`
- Path: `{UPSTREAM_MODEL_PATH}`
- Upstream author: {UPSTREAM_AUTHOR}
- Upstream license: {UPSTREAM_LICENSE}

The generated model replaces the high-poly test mesh with native SDFormat box
primitives, keeps the 0.057 m nominal side length, adds dynamic mass, inertia
and contact properties, and renders a brand-neutral stickerless 3x3 appearance
using 54 facelet visuals (3x3 per face, {FACELET_SIZE_M} m plates with a
{FACELET_GAP_M} m gap). It contains no logo, no internal mechanism and no
turning joints: the object is one rigid body, which is what pose and grasp
experiments need.

No upstream file is copied into this repository; the geometry is regenerated
from constants, so the model is reproducible from source alone.
'''


def write_model(directory, *, static=False):
    """Write ``model.sdf``, ``model.config`` and ``UPSTREAM.md``.

    Returns the three paths in that order. The directory is created when
    missing; existing files are overwritten so regeneration is deterministic.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    sdf_path = target / 'model.sdf'
    config_path = target / 'model.config'
    attribution_path = target / 'UPSTREAM.md'
    sdf_path.write_text(build_model_sdf(static=static), encoding='utf-8')
    config_path.write_text(model_config_xml(), encoding='utf-8')
    attribution_path.write_text(attribution_text(), encoding='utf-8')
    return sdf_path, config_path, attribution_path


def facelet_count_in(sdf_text):
    """Return the number of generated facelet visuals in ``sdf_text``."""
    return sdf_text.count('<visual name="face_')


def facelet_extent_m():
    """Return the occupied width of one facelet grid along a face."""
    return (FACELET_GRID_SIZE * FACELET_SIZE_M
            + (FACELET_GRID_SIZE - 1) * FACELET_GAP_M)


def check_geometry():
    """Validate the constants against each other; raise when inconsistent."""
    if SIDE_LENGTH_M <= 0.0:
        raise RubikPoseCubeError('side length must be positive')
    if FACELET_SIZE_M <= 0.0 or FACELET_THICKNESS_M <= 0.0:
        raise RubikPoseCubeError('facelet dimensions must be positive')
    if FACELET_GAP_M < 0.0:
        raise RubikPoseCubeError('facelet gap must not be negative')
    if facelet_extent_m() > SIDE_LENGTH_M:
        raise RubikPoseCubeError(
            'facelet grid does not fit on the cube face'
        )
    if len(FACE_SPECS) != 6:
        raise RubikPoseCubeError('exactly six faces are required')
    if FACELET_COUNT != 54:
        raise RubikPoseCubeError('a 3x3 cube has 54 facelets')
    if math.isclose(SIDE_LENGTH_M, 0.0):
        raise RubikPoseCubeError('side length degenerates to zero')
    return True
