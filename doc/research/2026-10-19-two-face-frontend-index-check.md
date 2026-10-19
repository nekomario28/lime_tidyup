# Two-face colour frontend: index convention and quad accuracy on a real render

Date: 2026-10-19
Status: **measured — the colour-frontend route is blocked by two concrete defects**

## What was run

The model-free path on one Gazebo frame of the generated 57 mm cube
(`rubik_pose_cube`, static, 0.35 m, corner-on, bright world):

```
YOLO bbox (492, 223, 791, 512)
  -> cube8_two_face_color_frontend.recover_two_face_view_local_from_bgr
  -> gate-passed view-local geometry
  -> cube8_view_local_pnp.estimate_view_local_pose_from_observations
  -> (never reached) dev2 symmetry hypotheses + face colours
```

The frame was the one the checkpoint check rejected too, so the two routes are
being compared on identical evidence.

## What worked

The frontend recovered the two-face view cleanly:

- `visible_y_color: blue`, `visible_z_color: white` — both correct for this view
- support pixels: blue 13 942, white 23 369
- `gate_passed: true`, no gate failures
- shared-edge score 0.0446 (gate 0.08), bbox diagonal 415.8 px
- `observed_indices: [2,3,4,5,6,7]`, `derived_indices: [1]`,
  `color_fixed_semantics_resolved: false` (as designed)
- the six vertices are well separated (minimum pairwise distance 124 px)

So the colour leg, the quads and the geometry gate all work on this render.

## Defect 1: the local index labels are a permutation of the PnP contract

The view-local geometry was projected with the **Gazebo truth pose** and each
frontend point was matched to its nearest canonical vertex (`CANONICAL_VERTEX_LABELS`
order, the order `estimate_pose_from_observed_vertices` pairs against):

| frontend local index | nearest canonical vertex | distance px |
| --- | --- | --- |
| 2 | **3** | 15.1 |
| 3 | **2** | 3.9 |
| 4 | **5** | 5.4 |
| 5 | **4** | 15.8 |
| 6 | **1** | 6.8 |
| 7 | **6** | 8.6 |

Every observed index is permuted. PnP pairs image point *i* with canonical
vertex *i*, so the correspondence set is internally inconsistent and the solve
fails — which is exactly what was measured:
`estimate_view_local_pose_from_observations` →
`Cube8PnpError: solvePnPRansac failed`, with **both** an empty distortion model
and a five-zero one, and in a clean container at the runtime combination
(opencv-python-headless 4.10.0.84 + numpy 1.26.4). The lane's own
`cube8_two_face_view_local_smoke.py` passes in that same container with the same
observation indices, because its synthetic quads are generated with the
frontend's convention and therefore never exercise this mapping.

This is the second time in this work that two modules agreeing "by convention"
did not: the first was the face-to-colour table (fixed and pinned with a test).
The same treatment is needed here — either the frontend emits canonical indices,
or the PnP adapter applies an explicit, tested permutation.

## Defect 2: quad corners are 4-17 px from the true vertices

Even with the indexing fixed, the fitted face-quad corners sit 4–17 px from the
projected cube vertices (median about 7 px) while the runtime PnP gate is 4 px.
A colour-mask corner fit on a low-resolution render is not yet accurate enough
to pass that gate; either the corner fit needs refinement against the projected
geometry, or the gate needs a documented, measured rationale rather than being
loosened ad hoc.

## What this means for the un-withhold path

Both routes to a pose are now measured:

| route | measured outcome |
| --- | --- |
| 8-keypoint checkpoint (pinned ONNX) | keypoints inconsistent with any cube pose at a 30 px budget |
| two-face colour frontend | gate passes, then PnP fails on a permuted index convention |

Neither is usable as-is, and neither failure is in the dev2 evidence layers:
`cube_visibility`, `cube_pose_hypotheses`, `cube_face_colors` and
`cube_pose_evidence` all behave as designed (fail closed, report why). The work
needed is inside the two producer stages, and the cheapest next step is defect 1:
a mapping fix plus a test that ties the frontend's indices to
`CANONICAL_VERTEX_LABELS`, in the same style as the palette pin.

## What this does not claim

- Nothing here says the frontend is wrong for the two-face views it was built
  and smoke-tested for; the measurement is a three-face render plus the
  canonical-order pairing used by the runtime PnP.
- No threshold was loosened to make a pose appear.
