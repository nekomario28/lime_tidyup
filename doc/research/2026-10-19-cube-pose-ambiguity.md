# Cube pose ambiguity: what geometry cannot decide

Date: 2026-10-19
Scope: the `Cube8` pose leg of `ccc` (`/cube_pose_result` → `GoFrontCube`).

## Question

`color_fixed_semantics_resolved` is never set, so the Cube8 lanes withhold grasp
candidates. Before adding landmarks (a 54-facelet model) to resolve it, the
question worth answering first is how much of the ambiguity is actually
landmark-poor versus structurally undecidable.

## Method

A colourless cube is invariant under its own rotation group. A pose estimated
from corner observations is therefore determined only up to the symmetries that
keep the observed/hidden vertex partition intact. For each view type the
measurement enumerates all 24 proper rotations of the cube, keeps the ones whose
relabelling also keeps hidden corners hidden, and checks how well the
alternative pose reproduces the *same* pixels. It also reports, for each
surviving hypothesis, which physical face each visible local slot would have to
be, and hence the colour each visible face would have to show.

Script: `facelet_decision.py` (kept with the cube8 sim scratch, not in this
repository). Cube 57 mm, K = (500, 500, 320, 240), camera at 0.5–0.9 m.

## Measured

| View | Visible vertices | Geometry-equivalent poses | Orientation spread | Max reprojection residual | Distinct colour assignments |
| --- | --- | --- | --- | --- | --- |
| 3-face corner (`ccc` approach view) | 7 | **3** | 0°, **120°**, 240° | **0.0 px** | **3** |
| 3-face, other corner | 7 | 3 | 0°, 120°, 240° | 0.0 px | 3 |
| 2-face edge | 6 | **2** | 0°, **180°** | 0.0 px | **2** |
| 1-face frontal | 4 | **4** | 0°, 90°, 180°, 270° | 0.0 px | **1** |

Two consequences follow directly from the 0.0 px residual:

1. The ambiguity is **structural**, not a measurement or solver problem. All
   three hypotheses explain the same seven pixels exactly, so a better model, a
   better PnP, RANSAC, or LM refinement cannot separate them. Solver work
   (inlier-only refinement, robust PnP) improves precision within one hypothesis
   and does nothing for this.
2. The wrong hypothesis is not a small error: the alternatives are **120° apart**
   (3-face) or **180° apart** (2-face). A grasp planned from the wrong one is
   wrong by that much.

## Decision

- **Do not move to a 54-facelet landmark model for pose.** The 3-face and 2-face
  views — the ones `ccc` actually approaches through — are separated by the
  *visible face colours* alone: three colours distinguish the three hypotheses
  exactly (two colours the two hypotheses). One colour per visible face is
  sufficient evidence; 54 landmarks are not required to break this symmetry.
- **Keep withholding single-face views.** A frontal view has four
  geometry-equivalent poses 90° apart and all of them predict the *same* face
  colour, so colour cannot break the tie. In practice the runtime never reaches
  that tie: four coplanar corners do not determine a pose, so
  `estimate_cube_pose_3d` already rejects the configuration (verified: the
  coplanar four-point sample raises instead of returning a pose). The four-fold
  count stays a geometric fact, not a runtime risk; the budget belongs to
  approaching until a second face is visible.
- **The existing two-hypothesis chirality selector is not enough for the 3-face
  case.** Enumerating `identity` and `swap_xy` covers a reflection; the measured
  ambiguity is the **three-element stabiliser of the hidden corner** (identity
  plus two 120° rotations). A selector for `ccc` must enumerate three
  hypotheses, score them by predicted-versus-observed visible face colour, and
  report the margin rather than a single answer.

## Implemented from this decision

- `barcode_detector/cube_visibility.py` — vertex visibility from the pose, with
  the `observed / unmeasured / occluded / contradicted` split, so a confident
  detection of a hidden corner is reported instead of averaged into the pose.
- `barcode_detector/cube_pose_hypotheses.py` — the measured hypothesis set,
  enumerating only the symmetries that keep hidden corners hidden, and the
  colour each visible slot would have to show under each of them.
- `barcode_detector/cube_face_colors.py` — CIELAB hue classification of a face
  sampled at its nine sticker centres, with an explicit abstain: shading is
  allowed to change lightness, and a patch between two anchors or with no
  chroma yields `None` rather than a nearest-neighbour guess.
- `barcode_detector/cube_pose_evidence.py` — the assembly, used by
  `yolo_pose_node` behind two parameters that are **off by default**
  (`enable_visibility_gate`, `enable_face_color_selection`), so the published
  `/cube_pose_result` contract is unchanged unless a caller opts in. `bin/ccc`
  exposes both through `CUBE_EVIDENCE_GATE=1`.
- 49 unittest cases cover the three modules and the assembly, including a guard
  that the vertex sign order matches the PnP contract's vertex order.

## What this does not claim

- No colour classifier accuracy, no real lighting/material measurement; the
  measurement is geometric and uses the standard six-colour assignment.
- No change to `yolo_pose_node` yet: this fixes the design decision, not the
  runtime. Wiring colour evidence into the node remains a separate slice.
- Nothing here authorises motion. A resolved pose is an observation; the
  existing acceptance gates and the fail-closed grasp path stay as they are.
