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
