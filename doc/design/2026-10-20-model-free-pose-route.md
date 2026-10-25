# Model-free cube pose: geometry and colour, no learned checkpoint

Date: 2026-10-20
Status: **implemented and measured; opt-in via `perception_backend=color_frontend`**

## Why a second route

The shipped eight-keypoint checkpoint was measured to be unusable on the
simulated cube: its predicted vertices were inconsistent with *any* cube pose
even at a 30 px reprojection budget, and the same weakness showed on a real
photograph ([checkpoint check](../research/2026-10-19-eight-keypoint-checkpoint-check.md)).
The two-face colour frontend, by contrast, passed its geometry gate on the same
frames and only failed later on an index-convention bug
([frontend check](../research/2026-10-19-two-face-frontend-index-check.md)).

So the cube pose does not have to depend on a trained keypoint model at all. This
route uses a detector for a **bounding box only** and gets the pose from geometry
and colour.

## Pipeline

```
YOLO bbox (trained detector, box only)
  -> two visible face regions by hue / white thresholds      (cube8_two_face_color_frontend)
  -> face quads snapped onto the object silhouette           (measured: 12.4 px -> 2.4 px)
  -> view-local vertices from the two quads                  (cube8_two_face_view_local)
  -> PnP over every corner relabelling, accept only if one fits
                                                             (cube8_view_local_pnp)
  -> visible face colours sampled per sticker                (cube_face_colors)
  -> local slot -> physical face map from the colour labels  (cube_model_free_pose)
  -> canonical pose, colour-fixed
```

The last step is what makes this a *colour-fixed* pose rather than a view-local
one: each labelled face identifies an axis and a sign (`blue` = `+y`,
`orange` = `-x`, …), so the local frame is mapped onto the physical cube frame
directly. If the labelled faces imply a **mirror** (determinant `-1`) the pose is
refused rather than reflected into place.

## Measured

On the same Gazebo frame of the generated 57 mm cube (0.35 m, corner-on):

| Quantity | Value |
| --- | --- |
| View-local PnP residual | **1.31 px** |
| Pose translation vs truth | **(0.0, 0.0, 0.350) m** — sub-millimetre |
| Relabellings that fit the 4 px gate | **1** (two at 15 px, reported as ambiguous) |
| Face colours observed | orange / blue / white — all correct |
| Resolution | `color_fixed_semantics_resolved: true`, agreement 3 |
| Corner error (median) | 12.41 px → **2.37 px** after silhouette snapping |

The single-sun world is kept as a **negative** fixture: there the shaded white
face is not separable from the grey floor, the colour leg abstains, the pose is
returned unresolved with the reason recorded, and the geometry stays available as
evidence. Both frames are test fixtures (`test/fixtures/cube_model_free/`), so the
measured behaviour is pinned by `test_cube_model_free_pose.py` rather than
described in prose.

## Enabling it

`color_frontend` is the default, so the node needs no arguments. To use the
trained keypoint route instead:

```bash
ros2 run barcode_detector yolo_pose_node --ros-args \
  -p perception_backend:=yolov8-pose
```

With `color_frontend`, a pose is published
only when the colour semantics resolve; an unresolved observation is reported in
`pose_3d_debug` (status `unresolved`) with its face evidence, never as a pose.

## The single-face route, and which pose the consumer gets

The two-face frontend above needs two visible faces, which the grasp camera mounted low
on the base never has: it sees one face and the cube's top edge. `cube_single_view_pose`
covers that case — six faces times four in-plane rotations, solved with IPPE from the
face's own colour mask, back-facing hypotheses pruned, and the face identified by
comparing the sampled sticker colour against the standard palette.

Its position measures comparably to the depth centre — through the node at 0.4/0.6/0.8 m,
0.33/0.59/0.53 cm against 0.10/0.31/0.50 cm — and both are published, so a consumer reads
the position from whichever route and the orientation from the face. Which one occupies
`pose_3d` (the primary slot) is decided by those numbers: the depth centre does, because
it is the better position, and the face solve goes beside it as `pose_3d_face`. Each route
records its own `geometry_id`, so a consumer that cares picks a route by geometry rather
than by which field happens to be filled.

A frame that finds a box but no solve republishes the last accepted pose instead of
nothing, bounded by the node's memory-age parameter and tagged so a consumer can tell a
remembered pose from a measured one; past that bound the frame carries no pose at all
rather than a stale one.

Naming a field in a doc or a consumer is a claim about the node, so it is worth checking
against the node: a contract described from the intent rather than from the publisher
reads as implemented and is not.

One property of the single-face solve determines what the grasp can do: **its orientation
is quantised to the face grid**. The solve reports which face it is looking at, so a cube
turned 30 degrees still reports -90 — the angle needed to align the jaws with the cube is
not in the pose, and the grasp approaches at yaw 0 as a result
([measured](../research/2026-10-20-grasp-oss-survey.md)).

## Limits

- The two-face frontend needs **two or more visible faces**: a single face cannot
  separate the symmetry-equivalent poses, and the frontend refuses.
- The colour vocabulary is the six standard faces; a cube with a non-standard
  palette needs the palette changed in one place (`cube_face_colors`).
- Lighting matters for white: a heavily shaded white face is grey and the leg
  abstains by design.
- Boxes matter: the frontend's gates are relative to the bbox diagonal, and a
  much larger box (e.g. a truth-derived one) fails the gate. Feed it a detector
  box.
- The ported `cube8_*` modules keep the cube8 lane's formatting and are pinned by
  the lane's own smokes plus `test_cube_model_free_pose.py`; they are a port, not
  a second implementation, and the lane remains the source of truth.
