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


## 統合（2026-10-23、実測で通した）

`bin/cccsim --mujoco` で、Gazebo の代わりに MuJoCo が物理を担当する。**同じコンテナ・同じ
コントローラ・同じトピック**で、物理エンジンだけが違う。

```
./bin/cccsim --mujoco ...        # Dockerfile.mujoco で作った lime-mujoco:local を使う
CCCSIM_MJ_CUBE_X=0.24            # シーンに立方体を置く（位置をシーン生成時に決める）
CCCSIM_MJ_CAMERA=1               # ベースカメラをシーンに足す（CameraPlugin 用）
```

### 構成

| 要素 | 実装 |
|---|---|
| シーン MJCF | `sim/mj/build_scene.py`（URDF→MuJoCo importer→床・アクチュエータ・立方体・カメラ）。`stack.sh scene` がコンテナ内で生成 |
| ros2_control の URDF | `sim/mj/description_mujoco.py` が Gazebo 版からプラグインとパラメータを差し替え。`rsp_sim.launch.py` が `CCCSIM_BACKEND=mujoco` で適用 |
| 物理 | `mujoco_ros2_control/MujocoSystemInterface`（Humble バイナリ、apt） |
| 制御ノード | `sim/mj/mujoco_control.launch.py`（Humble では `~/robot_description` → `/robot_description` の remap が必要） |
| コントローラ | `sim/mj/controllers_mujoco.yaml` — 稼働中の 3 本（joint_state_broadcaster / arm_controller / gripper_controller）をそのまま |
| `/cmd_vel` | `BaseVelocityPlugin`（`body: base_footprint`）。車輪にアクチュエータを置かない |

### 実測で必要だったこと（統合側）

1. **アクチュエータ名＝ジョイント名**。`joint1_pos` のように別名を付けると ros2_control の
   ジョイントと対応せず、`missing state/command interfaces` を並べてハードウェアの初期化が
   失敗する（シーンのロード自体は成功しているので紛らわしい）。
2. **Humble では remap が要る**。制御ノードは private な `~/robot_description` を読むため、
   `ros2 run ... -r ~/robot_description:=...` では効かない（`~/` が展開されない）。
   launch ファイルで remap を渡すのが確実。
3. **車輪の velocity コマンドは `position` アクチュエータでは満たせない**（インターフェースの
   仕様）。だから車輪は ros2_control から外し、`/cmd_vel` は BaseVelocityPlugin が受ける。
4. **腕のトルク上限**。URDF の `effort="1"` がそのまま MuJoCo の `actuatorfrcrange` になり、
   手首が指令に 0.3 rad 届かない。サーボ定格の 10 N·m に緩めた（`ARM_TORQUE`）。
5. `pkill -f ros2_control_node` は**自分自身のシェルにも一致**して殺すので、
   `pkill -f "[r]os2_control_node"` を使う（これで再起動が無言で失敗していた）。

### 確認できたこと

```
[rsp_sim] backend: mujoco, scene /root/cccsim_evidence/scene.mjcf
[MujocoSystemInterface]: Loading 'mujoco_model' from: '/root/cccsim_evidence/scene.mjcf'
[MujocoSimulation]: Sim ready / Running in HEADLESS mode
[MujocoSystemInterface]: Registering MuJoCo actuator 'joint1' for joint 'joint1' … joint6
[MujocoSystemInterface]: Successfully loaded and initialized plugin: base_velocity_plugin
[resource_manager]: Successful initialization / configure / activate of hardware
ros2 control list_controllers:
  joint_state_broadcaster  active
  arm_controller           active
  gripper_controller       active
```

腕を**既存の arm_controller** に指令して MuJoCo の状態が動くことまで確認:

```
before: [-0.0, 0.0003, 0.0003, 0.0, 0.0003, 0.0]
after:  [0.0001, 0.4054, -0.4398, 0.3, 0.0007, 0.0001]
commanded:      [0.0,    0.35,  -0.45,   0.3, 0.0,    0.0]   max error 0.055 rad
```

### 次（この統合の続き）

1. 立方体をシーンに置き（`CCCSIM_MJ_CUBE_X`）、`FreeJointStatePublisherPlugin` で立方体の
   姿勢を購読する（＝持ち上げ後に指へ残っているかを測る物差し）
2. `--tree ccapproach --grasp --record` を MuJoCo でそのまま走らせる（grasp_cube.py は
   `/compute_ik` と MoveIt を使うので、そのまま動く想定）
3. `CameraPlugin` を既存トピック名（`/camera/camera/color/image_raw` ほか）に向け、
   MuJoCo の描画で認識が成立するかを測る


## L2: MuJoCo の描画で認識を測る（実測）

同じ `sim/eval_cube_recognition.py` を MuJoCo のカメラに対して走らせた（カメラは
`sim/mj/build_scene.py --camera` が base に取り付け、`CameraPlugin` が既存のトピック名で
配信する。知覚ノードは無改造）。

```
d=0.40  face-quad-v1 err 0.0571 m   face -x /orange
d=0.60  face-quad-v1 err 0.0636 m   face -x /orange
d=0.80  face-quad-v1 err 0.0532 m   face -x /orange
d=1.00  face-quad-v1 err 0.0579 m   face -x /orange
```

**認識は成立している**: 3×3 のステッカー面を正しく捉え、面 `-x`・色 orange を正しく同定
（Gazebo と同じ結論）。ただし**距離が約 1 割近くずれる**（誤差が距離に比例せずほぼ一定）。

原因を実測で切り分けた結果:

| 測ったもの | 値 | 解釈 |
|---|---|---|
| 深度画像の立方体面 | 0.912-0.915 m | 幾何（面まで 0.9125 m）と一致 → **描画は正しい** |
| `colour_blob` が返す箱 | 54×54 px | 面の投影は 37.9 px → **箱が大きすぎる** |
| 面クワッドの実測幅 | 39 px | ステッカー面 33.9 px / 面全体 37.9 px → 面全体を掴んでいる |
| PnP の返す距離 | 0.83 m | 0.051/0.057 の仮定で 0.912 × 0.895 = 0.816 ✓ 説明がつく |

つまり **MuJoCo の描画では色ソースの箱が立方体の面より広く出る**ため、クワッドが
ステッカー面ではなく面全体を覆い、認識側が仮定する「ステッカー面＝0.051 m」との比だけ
距離が縮む。Gazebo では箱がステッカー面に密着していたのでこのバイアスが出なかった。

**これは MuJoCo のバグではなく、シーンの見え方に対する認識の頑健さの問題**（照明・
アンチエイリアスで黒い素地の縁が暖色に寄る、などが候補）。次にやることは、シーンの
照明/材質を Gazebo の見え方に寄せて箱が密着するようにし、再測定すること。
切り分け用のスクリプトは `/tmp/mj_solve.py`・`/tmp/mj_quad.py`・`/tmp/mj_height.py` の形で
使った（コンテナ内で実行）。

## 統合の確認（再掲・追記）

- 立方体の配置は `sim/spawn_cube.py` が両対応: MuJoCo では
  `set_free_joint_state`（「Gazebo の set_entity_state 相当」）でシーンの自由関節を動かす
- 立方体と台車の実姿勢は `FreeJointStatePublisherPlugin`（`/free_joint_states`）で読める
  — 実測: cube world (1.0000, 0, 0.0284) / base_footprint world (0, 0, -0.0002)
- カメラは **base のボディ内**に置く（`worldbody` 直下だと world 座標になり、base_link の
  +0.06 m を失って距離に比例する誤差が出る。実測 0.8→2.6 cm）
