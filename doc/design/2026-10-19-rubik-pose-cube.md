# 57 mm Rubik-style pose cube: reproduction design

Date: 2026-10-19
Scope: the simulated cube used by the Cube8 pose/colour lanes.

## Why

The upstream Gazebo asset this repository's sim inherited
(`osrf/gazebo_models/cube_20k`) is a **plain blue 57 mm cube**. It carries the
right size and mass for a speed cube, but a colour-based pose hypothesis needs
six distinguishable faces: a single-colour cube cannot separate the three
symmetry-equivalent poses measured in
[the ambiguity note](../research/2026-10-19-cube-pose-ambiguity.md).

So the blue cube is rebuilt as a brand-neutral **stickerless 3x3**: same 57 mm
envelope, same mass, but 54 facelets in the six standard colours.

## Asset facts

| Property | Value |
| --- | --- |
| Side length | 0.057 m |
| Mass | 0.08 kg |
| Inertia | solid cube, `m·L²/6` on every axis |
| Collision | one axis-aligned box, 0.057 m, µ = 0.9, ODE contact 1e6/10 |
| Faces | 6 × (3 × 3) facelets = **54** |
| Facelet plate | 0.0162 m square, 0.0006 m thick |
| Gap between plates | 0.0012 m |
| Plate stand-off | half the side + half the thickness |
| Core | dark (0.012, 0.012, 0.014), low specular — the groove colour |
| Body | **one rigid body** (not 27 cubies) |
| Dependencies | none: no mesh, no texture, no download |

## Why one body and 54 plates (not 27 cubies)

1. **Stable physics.** One collision box means no cubie-on-cubie contact
   jitter and one inertia tensor to validate, instead of 27 bodies and 162
   contact pairs.
2. **Exact sampling geometry.** Each facelet is a visual at a known local pose,
   so a face region can be projected, sampled and classified per sticker
   without relying on a texture or UV layout.
3. **Separators exist.** The plates sit proud of the core with a 0.0012 m gap,
   so a 3×3 sticker sample lands on sticker interiors and the dark grooves show
   up between them — which is what keeps a face mean from mixing two stickers.

The trade-off is explicit: this is a **pose and colour proxy**, not a turning
cube. There are no internal mechanisms and no layer joints.

## Face-to-colour assignment

| Face | Colour |
| --- | --- |
| `+x` | red |
| `-x` | orange |
| `+y` | blue |
| `-y` | green |
| `+z` | white |
| `-z` | yellow |

Opposite faces always differ, so the three faces visible from any corner are
always three distinct colours. That is the property the pose hypothesis
selection depends on.

**This table is a single invariant with the scorer.** `cube_pose_hypotheses`
scores hypotheses against the same face-to-colour table, and a test pins the two
modules together. That pin exists because the first version of this work had
them swapped (`+y` white / `+z` blue in the scorer, blue / white in the asset):
every identity hypothesis then scored as a mismatch, and the pose could never
resolve even with perfect colour observations. Two modules that must agree need
a test that says so.

## Generation and use

```python
from barcode_detector.rubik_pose_cube import write_model
write_model('models/lime_rubik_pose_cube', static=False)   # sim / grasp
write_model('models/lime_rubik_pose_cube_static', static=True)  # colour probe
```

`write_model` writes `model.sdf`, `model.config` and `UPSTREAM.md` (provenance),
and is deterministic: the same constants always produce byte-identical output.
`static=True` freezes the cube so a camera-only probe knows the exact pose
instead of reading whatever gravity produced.

## Measured (Gazebo, generated model)

Setup: the generated model spawned static at 0.35 m from the sim camera, one
corner pointed at the lens so three faces are visible; a fill light and a dark
background were added to the probe world so a shaded white face is separable
from the grey floor.

- visible slots: `-x`, `+y`, `+z`
- hypotheses enumerated: **3** (identity + two 120° rotations about the hidden
  corner axis), as the ambiguity measurement predicted
- observed: `-x` orange, `+y` blue, `+z` white — all three correct
- face evidence: `-x` L 16.5 / chroma 36.7, `+y` L 18.3 / chroma 68.9,
  `+z` L 98.8 / chroma 2.5 (9 samples each)
- selection: **resolved, agreement 3, margin 3**

Before the palette fix, the same frame produced `agreement 1, margin 1` and a
*wrong* non-identity hypothesis. That is the failure mode the invariant test now
prevents.

## Limits

- Lighting and material are simulated, not measured: the colour anchors are sRGB
  references, and the shading tolerance is a hue test with a chroma guard.
- A **shadowed white** face is close to grey; the classifier abstains rather
  than guessing, and the earlier probe world (grey floor, single sun) abstained
  exactly there. The probe world now has a fill light and a dark background.
- The measured run used the **Gazebo truth pose**; the detector path still needs
  an eight-keypoint checkpoint before this can run from `/cube_pose_result`.
- Nothing here grants motion authority. A resolved colour semantics is an
  observation.

## Provenance

Design adapted from `nekomario28/lime-interactive-transport`
(`pose_cube_asset.py`), which derives it from `osrf/gazebo_models/cube_20k`
(author Mihai Dhola, CC BY 3.0, pinned commit `8163eb4b…`). No upstream file —
in particular no mesh — is copied into this repository; the geometry is
regenerated from constants, and `UPSTREAM.md` records the attribution next to
every generated model.
