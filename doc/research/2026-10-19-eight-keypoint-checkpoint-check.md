# Eight-keypoint checkpoint check on the generated pose cube

Date: 2026-10-19
Status: **measured negative — the available checkpoint is not usable on this render**

## What was run

The full chain on a single Gazebo frame of the generated 57 mm cube
(`rubik_pose_cube`, static, 0.35 m, corner-on):

```
YOLO 2D bbox (best.pt, conf 0.92-0.97)
  -> Cube8 RTMPose ONNX, 8 ordered vertices
     (sha256 3c04a6939c736a1730b2b0c93c66a10cf60b29e48fc545b1a0c7515aa23dacac,
      the value pinned by the cube8 lane's model-export CI)
  -> estimate_cube_pose_3d
  -> visibility gate -> symmetry hypotheses -> face colours
```

The checkpoint was found on this host at
`~/Videos/lime/cube8-rtmpose/32595602886-3-recovery-32773097860/model.onnx`
and its sha256 matches the pinned value exactly.

## Measured

Two lighting variants (bright world as inherited; dark probe world with a fill
light) and three bbox expansions. Decoding was run twice: once with
`rtmlib==0.0.16` from PyPI and once with the **pinned rtmlib commit**
`03a1693e…` checked out and put on `PYTHONPATH`. The keypoints were identical,
so the decode is not the variable.

| bbox expansion | keypoints inside bbox | mean keypoint confidence | PnP (4 px budget) |
| --- | --- | --- | --- |
| 1.00 | 7 / 8 | 0.462 | rejected — reprojection error |
| 1.15 | 8 / 8 | 0.544 | rejected — insufficient ≥0.5 correspondences |
| 1.30 | 4 / 8 | 0.672 | rejected — `solvePnPRansac failed` |

Raising the acceptance budget does not rescue it:

| reprojection budget | result |
| --- | --- |
| 4 px | rejected |
| 8 px | rejected |
| 15 px | rejected — inlier RMS still above budget |
| 30 px | rejected — inlier RMS still above budget |

On the real-cube sample photo (`yolo_ws/sample.png`) the same checkpoint puts
6 of 8 keypoints inside the detected bbox, so the weakness is not specific to
the simulator.

The rendered keypoints (`rtmpose_keypoints.jpg`) show the failure directly: the
eight predictions are not the cube's corners. Several sit on one face edge,
two lie outside the silhouette, and only three clear the 0.5 confidence floor.

## Reading

- The wiring is not the problem: bbox, preprocessing, SimCC decode and the
  camera-frame PnP contract all run, and the decode agrees between the pinned
  source and the released package.
- The checkpoint's predicted vertices are **inconsistent with any cube pose** at
  a 30 px budget, so this is not a tuning question. Either the checkpoint's
  keypoint semantics differ from the eight canonical vertices the PnP expects,
  or its training appearance (the donor's own Gazebo capture) is far enough from
  this render that the vertices are simply not localised.
- One concrete semantic risk worth naming: the canonical contract includes the
  **occluded** eighth vertex. A model trained to emit it is predicting a point it
  cannot see, so its error lands behind the silhouette — which is what the
  annotation shows.

## Consequence for the un-withhold path

The colour leg is measured working (`doc/design/2026-10-19-rubik-pose-cube.md`:
resolved, agreement 3, margin 3), but only with a **pose** supplied. The running
node needs a pose first, and this checkpoint does not provide one. So:

1. **Do not enable the evidence stages expecting resolution from this
   checkpoint.** They fail closed exactly as designed — `pose_3d_debug` reports
   the rejection and `pose_evidence.color` reports `no accepted pose`.
2. The two live options are a **domain-matched eight-keypoint checkpoint**
   (needs capture + export; the dataset is not in either repository) or the
   **colour-frontend route that needs no keypoint model** (the cube8 lane's
   two-face colour frontend plus view-local PnP). The second is the cheaper
   experiment and does not depend on this artefact.

## What this does not claim

- It does not say the checkpoint is bad in its training domain; no in-domain
  image was available here (the donor's `datasets/cube8_pose/` is gitignored and
  empty).
- It does not propose loosening the reprojection gate to make the pose "work":
  at a 30 px budget the residual is still over budget, so a looser gate would
  accept a pose the observations do not support.
