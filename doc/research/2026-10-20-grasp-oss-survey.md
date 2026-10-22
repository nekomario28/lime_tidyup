# 把持を成立させる OSS の調査（2026-10-20）

このラウンドで塞いだ欠陥（カメラのマウント、重複パブリッシャ、depth の近距離、
MoveIt の計画予算、actor スレッド内 IK）のあとに残った問題は 2 つだけだった。

1. 腕が測定した把持姿勢へ到達しない
2. 指が立方体を挟んでも保持できない（摩擦で持てない）

以下は、その 2 つを既存 OSS で解けるかの調査と、このイメージ上での検証結果。

## 結論（先に）

- **1 は新しい OSS を入れなくても解ける。** 既に依存に含まれる `pymoveit2` が
  `allowed_planning_time` の setter を持っている（`moveit2.py` 2234 行）。
  既定 0.5 秒のまま呼んでいたのが失敗の原因で、5 秒にすれば OMPL が goal 領域を
  サンプルできる。
- **2 は `gazebo_grasp_fix` で解ける。** `kongoncharuk/gazebo-pkgs` の `humble`
  ブランチが ROS 2 (ament_cmake) 版で、このイメージでビルドとロードの両方に成功した。
  リポジトリの URDF には既にこのプラグインの設定
  （`link7` / 左右の `gripper_*_link`）が書かれている。
- 予備として `ros-humble-pick-ik` が apt にある。KDL が床付近で `-31`
  (NO_IK_SOLUTION) を返す姿勢の保険になる。

## 検証した候補

| 候補 | 何を解くか | 検証結果 |
| --- | --- | --- |
| `kongoncharuk/gazebo-pkgs` @ `humble` → `libgazebo_grasp_fix.so` | 接触ベースの把持。指の接触を検出し、`disable_collisions_on_attach` で仮想関節にして保持する。`LinkAttacher`（今のテレポート接続）の置き換え | **この環境でビルド成功（2 パッケージ 19.7 s）**、かつ gzserver への**ロード成功**（下記ログ）。依存は apt で取得可（`ros-humble-gazebo-dev` / `ros-humble-gazebo-ros` / `libprotobuf-dev`） |
| `pymoveit2` の `allowed_planning_time` | pose goal の計画予算が 0.5 秒固定 → OMPL が「goal 領域をサンプルできない」で失敗 | 既存コピーに setter あり（2057 / 2234 行）。**1 行で変更可能** |
| `ros-humble-pick-ik` 1.1.2 | KDL の IK 失敗（床付近）を大域最適化 IK で解く。MoveIt 2 の IK プラグインとして差し替え | apt に candidate あり（未導入）。`kinematics.yaml` の `kinematics_solver` を変えるだけ |
| `ros-humble-gripper-controllers` 2.54.1 | グリッパ制御 | apt に candidate あり（導入済み） |
| `trac_ik` | 別系統の IK | apt には無い（bloom release のみ）。`pick_ik` で足りるので優先度低 |
| `IFRA_LinkAttacher` | 現在の接続方式 | 継続利用可（把持失敗時の保険） |

### `gazebo_grasp_fix` のロード確認（実ログ）

```
[grasp]: GazeboGraspFix: Using release_tolerance 0.001
[grasp]: GazeboGraspFix: Adding collision scoped name ...gripper_left_link::gripper_left_link_collision
[grasp]: GazeboGraspFix: Adding collision scoped name ...gripper_right_link::gripper_right_link_collision
[grasp]: Subscribing contact manager to topic ~/turtlebot3_lime_system/contacts
[grasp]: grasp plugin is loaded
```

ビルドは `/tmp/graspws`（`dependencies.repos` には未反映）、
`GAZEBO_PLUGIN_PATH` と `LD_LIBRARY_PATH` に
`libgazebo_grasp_plugin/gazebo_grasp_plugin` と `gazebo_version_helpers/lib` を足すと
ロードできる（`libgazebo_version_helpers.so` が無いとロードに失敗する）。

## 推奨する適用順

1. `cm1` の `arm` 生成直後に `arm.allowed_planning_time = 5.0` を設定する。
   これで `ArmToCube` の pose goal が通るか再確認する（現状は 0.5 秒で失敗）。
2. `dependencies.repos` のコメントアウトを活かして `kongoncharuk/gazebo-pkgs`
   (`humble`) を追加し、`lib_ws` でビルド。`GAZEBO_PLUGIN_PATH` /
   `LD_LIBRARY_PATH` を `sim/stack.sh` に足す。`LinkAttacher` は把持失敗時の
   フォールバックに降格する。
3. 1 でも届かない姿勢が残るなら `ros-humble-pick-ik` を入れ、`kinematics.yaml` の
   `kinematics_solver` を `pick_ik/PickIkPlugin` に差し替える。

## 参考（現状の数値）

- 指の関節は `effort=1 N` / `dynamics friction=1.0` / `damping=0.7` でほぼ閉じられない。
  `sim/rsp_sim.launch.py` の tuning で 10 N / 0.01 / 0.02、パッド `mu=2.0` に変更済み。
- 立方体は 0.08 kg、`mu 0.9`、1 m 先の床。
- depth は `<min_depth>0.05</min_depth>` を plugin に入れて 0.26 m から有効。
  1 m での中心推定は 0.841 m（真値 0.851 m）。

## 実走で分かった決定的な制約（追記）

`ccapproach`（探索・接近）+ `sim/grasp_cube.py`（IK→関節軌道→把持→持ち上げ）を
実走させ、指先位置を TF で実測した。

- **position-only IK**（shipped 設定）: IK は解けるが掌の向きが自由なので、指先は目標の
  **5.7 cm 上**に来る（実測 pads `(0.274, -0.022, 0.050)` / 立方体 `(0.311, -0.001, -0.007)`）。
  その誤差を補正すると end effector の z が **-0.019 m**（床下）になり IK は `-31`。
- **向きを効かせた IK**（`position_only_ik: False`）: 掌を下に向けた姿勢は、到達距離内
  （例 0.19 m）でも **全候補が `-31`**。

つまり **「掌を下にして床の立方体を 0.3 m 先で掴む」は、この 6 軸アームの可動域的・
姿勢的に成立しない**（チューニングの問題ではない）。実測にもとづく結論。

### 成立しうる代替（推奨順）

1. **前面把持**: 掌を +x に向け、指を立方体の左右面で挟む。手首が立方体の高さに留まるため
   可動域に余裕がある。`gazebo_grasp_fix` は「接触したら保持」なので、指が触れさえすれば
   把持が成立する。
2. **台を置く**: 立方体を数 cm 高い台に載せ、掌下向きの把持を可能にする（シナリオ変更）。
3. **接触アシスト前提で接近だけ最適化**: 指先が立方体に触れる位置まで近づけ、
   保持はプラグインに任せる。


## Addendum 2026-10-22: what the close-range correction can and cannot do

Measured while trying to make contact repeatable.

- **Driving the base cannot close a pads-vs-cube residual.** Both the pads and the
  cube ride on the base, so `pads - cube` is invariant under a base move. An earlier
  verify loop drove the base by the residual and watched the same 0.023 m for four
  checks; it was correcting the one thing that could not change.
- **The correction has to be an arm re-aim.** IK leaves the pads consistently short
  of the aim (0.022 m at every height tried), and re-solving at `aim + residual`
  moves them by exactly that amount (measured: pads 0.217 -> 0.240 for a 0.023 m
  correction).
- **Once the arm is in front of the camera, the cube measurement stops being a
  fixed reference.** With the arm extended toward the cube the re-measured centre
  walked forward about 2 cm per step (0.240 -> 0.261 -> 0.277) and the close-range
  aims then drifted backwards as each solve used the drifted point. The arm
  occludes the very face the pose comes from.
- **Distinguishing sign convention worth keeping:** `residual > 0` means the pads
  are *short* of the cube, because `pads_target = cube_x - FINGER_REACH_M`.
- Contact has been reached in two runs (dz -0.05 and -0.06, `stalled=True` at
  0.0155 and 0.0151) and missed in four. The misses share one shape: the visual
  residual was not credible by the time the jaws closed.

That was tried, and it is the answer. The loop now takes at most two visual checks,
corrects the aim on the arm, and then hands over to a touch sweep: close, and if the
jaws pass each other, open, push the aim 5 mm forward and close again, lowering the
aim a centimetre when a whole sweep finds nothing at that height.

Two runs in a row gripped on the very first close of the sweep (`stalled=True` at
0.0147 and -0.0074). In the second, the camera still reported the pads 0.021 m short
when the close stalled -- the occlusion error in the other direction, and proof that
the visual residual should not be an acceptance test at this range. The sweep costs
nothing when it works first time, and it is the only thing that worked when it did
not.

### Sweep envelope, measured 2026-10-22 (three runs with the sweep in place)

| run | where the grip happened | forward searched | height steps used |
|---|---|---|---|
| 1 | touch 0.0 | 0 mm | 0 |
| 2 | touch 0.0 | 0 mm | 0 |
| 3 | touch 0.5 | 25 mm (5 x 5 mm) | 0 |

So the sweep is not decoration: one run in three needed 25 mm of forward search, and
none needed a height step. The 5 mm creep resolved it without overshooting into the
cube, and the 5 mm height step has not yet been exercised -- it exists for the
centimetre of run-to-run height spread that earlier runs did show (-0.05 vs -0.06).

## Replacing the tuning with measurement (2026-10-22, later)

Every constant that had been tuned by re-running the grasp is now measured at run
time. Four quantities, four measurements:

| was | is |
|---|---|
| `GRASP_CANDIDATES` (pitch, dz) pairs, tuned over six runs | **where the pads actually go**: each round moves the command by the measured shortfall (round 1: +0.044/0/-0.077, round 2: 0/0/0) |
| `GRASP_PITCH_ORDER` (150, 120, 135, 105) by hand | the pitch that converges is the one used; the grid is walked until a pose is accepted |
| base distance 0.24 m fixed | **the arm's own reach**, bisected through IK (measured 0.307 m at pitch 150), minus a 3 cm margin for the sweep |
| creep 5 mm, lowering 5 mm to 10 mm fixed | the sweep doubles its step on an empty close (an empty close proves the cube is beyond the pads) and lowers by what was already tried |

Result of the first run with all four in place: reach 0.307 m measured, aim converged
to 0.000 m in two rounds, one empty close, one 5 mm lowering, then `stalled=True`.

Two mistakes made on the way are worth keeping, because both look like progress:

- Sizing the sweep step and the lowering from `target - aim` used the *wrist-to-pads*
  offset (0.022 m), not a measured error. It produced a 2.2 cm first step and a 4.7 cm
  drop, and the arm ran out of reach.
- Correcting the aim from one fixed first guess hangs when that guess is unreachable:
  at the pre-grasp distance the centre aim is refused at every pitch while 3 cm lower
  solves at 150. The grid of first guesses exists only to find one accepted pose.

### Tuning the last two numbers from their measured distributions, and one revert

`PAD_TOLERANCE_M` and `REACH_MARGIN_M` were the last hand-picked values. Both now come
from what the runs did:

| number | data | value |
|---|---|---|
| pad tolerance | one correction takes the error from up to 77 mm to under 0.5 mm, so the gate only absorbs TF rounding | 12 mm -> **3 mm** |
| reach reserve | the largest forward shortfall a sweep has needed is 25 mm | 30 mm -> **35 mm** (25 mm + one 8 mm step) |

Two mistakes found while measuring them, both worth keeping:

- The reach probe used the cube's centre height (0.307 m at pitch 150). The pads work
  4-8 cm lower, where the reach is shorter, so "there is room" was measured where the
  arm never goes. The probe now runs at the grasp height, and the reserve is checked
  against the corrected aim's own height.
- The reserve has to be bought with a **forward** drive: the pads sit on the base, so
  only the cube comes closer. Retreating buys nothing.

The adaptive *sweep* was reverted. Doubling the forward step on each empty close
(8/16/32 mm) gripped in none of three runs, while the fixed 5 mm sweep has gripped at
0.0, 0.0, 0.5 and 1.0 across four. The doubling was not understood well enough to
keep: it moves 3-7x further per attempt between two close measurements, and the
close is the only sensor that works there.

### The last height guess was unnecessary (2026-10-22, last)

`GRASP_BELOW_M = 0.045` was the final hand-picked number: "the pads work this far below
the cube centre", taken from a single measurement. It was used to ask the reach question
at the right height, and the recent logs show it was also wrong -- the first-round z
shortfall at the centre aim measured 0.077 m, not 0.045 m.

It is gone rather than corrected. Both sides of the reach comparison can be measured:
where the pads are now (`fingertip_midpoint`, TF) and how far forward they can be posed
at that height (IK bisection). room = reach(pads_z) - pads_x. A failed aim is answered
by driving forward, which is the only direction that brings the cube into the envelope.

Two runs with that in place: pads 0.240 m, reach 0.307 m at that height, 0.067 m of
search room; aim converged 0.044/0/-0.077 -> 0/0/0; grip on touch 1.0 and 0.1.

What is left hand-picked in the grasp path is only the search's own step sizes (5 mm),
which are bounded by what a close costs rather than by geometry.
