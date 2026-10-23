# Lime の把持を MuJoCo で測る（2026-10-23）

## なぜ別エンジンか: Gazebo では把持を判定できない

`gazebo_grasp_fix` を切って同じ把持（`CCCSIM_GRASP_PLUGIN=0`）を走らせた実測:

| 量 | 実測値 |
|---|---|
| 指 | `touch 0.0: gripper at 0.0111 stalled=True`（接触は検出） |
| 立方体の世界姿勢 | `x=2.09 y=-0.25 z=0.0284` ＝ **床の上**（半辺 0.0284）。開始位置から約 1.1 m 押し出された |
| ロボットの世界姿勢 | `x=-35.26 y=13.98 z=0.561` ＝ **6 N の把持力で 35 m 飛ぶはずが無い** |
| 外部カメラ（終了時） | 視野が空（ロボットも立方体も画面外） |

プラグインを入れた実行では立方体が飛び（既知の Gazebo 接触発散）、切ると今度はロボットが
飛ぶ。**どちらでも「指が保持したか」を観測できない**。ここが MuJoCo / Genesis に移す理由。

## 移せるもの / 移せないもの

- **移せる**: 認識（`rclpy` 非依存を確認: `cube_single_view_pose` / `cube_bbox_source` /
  `cube_model_free_pose` / `cube_pose_tracking` / `cube_face_colors` / `cube_overlay`）と
  `sim/pose_math.py`
- **移せない（＝移植の本体）**: `sim/grasp_cube.py` の ROS 界面 14 か所（`/compute_ik`、
  軌道コントローラ、TF、`/cmd_vel`、グリッパ action）と Gazebo 固有のもの全部
- **そのまま移る考え方**: 触覚探索の判定（指令に対する関節の遅れ）は物理量なので、
  力制限つき位置サーボを持つエンジンなら同じ規則が使える

## MuJoCo への取り込み（実測で必要だったこと）

`sim/mj/convert_lime_urdf.py` が以下を全部やる。

1. `xacro use_sim:=false` で素の URDF を得る（gazebo/ros2_control タグが消える）
2. **`link6` に `<inertial>` が 2 つある**（単位球のプレースホルダ＋実値）→ MuJoCo は
   繰り返し要素を拒否するので実値だけ残す
3. メッシュは**実ファイル**が要る。install 空間は symlink install なので `docker cp` すると
   リンク切れになる → `turtlebot3_ws/turtlebot3_lime/...` のソース側から取る
4. 唯一の `.dae` は RealSense カメラ（腕に関係なし・MuJoCo 非対応）→ 削除
5. `package://` を相対パスにし、`scale="0.001"`（STL が mm）は MuJoCo が解釈する

## 確認できたこと（MuJoCo 3.17 / CPU / python3.12 venv）

```
scene: nq=17 nv=16 nu=8 nbody=12
settled: cube z=0.0284  max|qvel|=0.0000        ← 立方体は床に静止、発散なし
arm joints: [-0.0, 0.408, -0.185, 0.031, 0.001, -0.0]
end effector: [0.0585 0.     0.4686]
gripper_left_link  [0.0731 0.036  0.5379]
gripper_right_link [0.0752 -0.036 0.5374]
gripper joints: [0.015, 0.015]
cube: [0.24 -0. 0.0284]   max|qvel|=0.493
```

- 関節は 10（車輪 2 + 腕 6 + グリッパ 2）、アクチュエータは腕とグリッパに 8
- 重力下で指令に追従（上の `joint2` は 1.0 指令に対し 0.408 ＝ サーボ剛性が不足。
  移植時に軌道追従として作り直すべき箇所）
- 指先の body（`gripper_left_link` / `gripper_right_link`）と `link7` の姿勢が読める
- 立方体・床は `sim/spawn_cube.py` と同じ寸法（57 mm / 0.08 kg / mu 0.9）

## 残っている作業（問い = 「摩擦で保持できるか」に答えるまで）

1. **IK**: MuJoCo に IK は無い。6 関節に対する DLS（減衰最小二乗）を 80 行程度。
   姿勢拘束は Gazebo 側と同じ「パッドを狙い点に、掌を pitch で」
2. **方針の移植**: `aim_at_pads` / 余白判定 / 触覚探索を、`RobotInterface`（IK・関節指令・
   パッド姿勢・グリッパ開閉）の裏に出す。この抽出は**実機移植でも同じ作業**なので投資が
   無駄にならない
3. **サーボ剛性**: 上の sag を消す（位置サーボの kp を上げるか、軌道追従を実装）
4. **測る**: 持ち上げ後も立方体が指の間に留まるか（MuJoCo に attach プラグインは無いので、
   留まれば摩擦で保持、落ちれば保持できていない）

## 実機・別エンジンへの含意

- 認識は ROS 非依存なので、MuJoCo の offscreen 描画を食わせれば**別レンダラで同じ認識が
  成立するか**の独立検証にもなる（AMD なので EGL か OSMesa）
- Genesis は GPU 前提でこのホスト（AMD/ROCm）ではリスクが高い。まず MuJoCo CPU。
  並列 RL が必要になったらそこで再検討する
