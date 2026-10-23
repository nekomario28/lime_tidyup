# Lime を別シミュレータへ移す: 公開物の探索（2026-10-23）

対象の問い: 把持が**摩擦で保持できているか**（Gazebo は接触発散で観測不能、実測は
[`2026-10-23-mujoco-lime-import.md`](2026-10-23-mujoco-lime-import.md)）と、
**別レンダラでも認識が成立するか**。

探索は GitHub 公式 API（read-only）で行い、候補は必ず実ファイル／実 API で確認した。
星数・最終更新は取得時点の値。

## 結論（先に）

1. **Lime / OpenManipulator の MJCF は世の中に存在しない**（コード検索で `open_manipulator_sara` /
   `turtlebot3_lime` の MJCF は 0 件。Menagerie にも無い）。**シーンは自作が必要**。
2. ただし**「Gazebo を MuJoCo に差し替える」部分は公式実装がある**:
   `ros-controls/mujoco_ros2_control`。**Humble 対応の CI があり、プラグインに
   `camera` / `base_velocity` / `fts` / `lidar` が既にある**。
3. したがって最も安い道は「IK と方針を自作移植（当初の L1 案）」ではなく、
   **物理バックエンドだけ差し替える**こと。`bin/cccsim`・木・MoveIt・把持スクリプト・認識・
   録画ハーネスをそのまま使い、Gazebo ステージを MuJoCo に置き換える。
   L1（摩擦で保持するか）と L2（別レンダラで認識）は**同じ統合から同時に得られる**。
4. 変換は `urdf2mjcf` がほぼやってくれる（我々の URDF の重複 inertial も処理）。ただし
   MuJoCo 3.17 向けに**2 点の後処理**が要る（下記）。

## 候補と判定

| 候補 | 判定 | 根拠（実測） | ライセンス |
|---|---|---|---|
| **`ros-controls/mujoco_ros2_control`** | **ADOPT** | ros-controls 公式。★269、2026-10-18 更新。CI に **Humble(jammy)** あり。`SystemInterface` プラグイン、URDF→MJCF 生成ユーティリティ、`mujoco_ros2_control_demos` に `mobile_base` 等 | Apache-2.0 |
| `.../mujoco_ros2_control_plugins` の **`camera_plugin.cpp`** | **ADOPT（L2 の要）** | カメラ映像を publish する実装が存在 → 既存の認識ノードが無改造で食える | 同上 |
| 同 `base_velocity_plugin` | ADOPT | `/cmd_vel` 経路がそのまま使える | 同上 |
| 同 `fts_grav_comp_plugin` / `*_lidar_plugin` | REFERENCE | 力覚・LiDAR。今回は不要だが存在は把握 | 同上 |
| `moveit/mujoco_ros2_control_examples` | REFERENCE | ★40。MoveIt ＋ peg-in-hole の実例（MoveIt 結線の参考） | NOASSERTION |
| **`kscalelabs/urdf2mjcf`** | **ADAPT** | MIT ★142、PyPI v0.2.39。我々の URDF を変換でき、**重複 inertial を警告なしで処理**、off-diagonal 慣性の警告、根に freejoint を追加。ただし後処理 2 点が必要 | MIT |
| MuJoCo 内蔵 URDF importer（`sim/mj/convert_lime_urdf.py`） | ADAPT（予備） | 動作確認済み。**collision をプリミティブに落とす**ので、巨大な visual メッシュ問題を回避できる | MuJoCo (Apache-2.0) |
| Menagerie **`robotis_omy_3m`** | REFERENCE | 同じ Robotis の 6 軸腕。`<position ... inheritrange="1"/>`＋class でゲインを書く作法、`keyframe` の home 姿勢 — うちの腕のアクチュエータ設計の雛形 | Apache-2.0 |
| `momoiorg-repository/Turtlebot3_Lime_IsaacSim_Humble` | REFERENCE | ★9 MIT。**上流（同じ momoiorg）自身の Lime 別シミュレータ移植**。中身は `turtlebot3_lime_{bringup,description,moveit_config,navigation2}` の ROS パッケージで、**USD は Google Drive 配布**・NVIDIA GPU 前提。ROS 側の構成の参考にはなるが MJCF は無い | MIT |
| `AJAXa239/OpenMANIPULATOR-X` | **REFERENCE_ONLY** | OpenManipulator-X を MuJoCo＋実 STL で動かし ROS 2 に橋渡し（38 MB）。設計の参考には最良だが**ライセンス表記が無い**ので複製不可 | 不明 |
| Genesis（`Genesis-Embodied-AI/Genesis`） | **HOLD** | ★30k Apache-2.0、活発（2026-10-22）。`examples/manipulation` あり、URDF/MJCF 取り込み可。ただし **Lime/TurtleBot3 のアセットは無い**・GPU 前提（このホストは AMD） | Apache-2.0 |
| `youtalk/iknet` | REJECT | ★52。OMX の学習ベース IK。我々が必要とするのは DLS 程度で、依存を増やす理由が無い | Apache-2.0 |
| TurtleBot3 を MuJoCo に載せた小規模リポジトリ群（`mobrob` 等） | REFERENCE_ONLY | ★24 以下、ライセンス不明。台車モデルの参考程度 | 不明 |

## `urdf2mjcf` を使う場合に必要な後処理（実測）

1. **`scale` の位置**: urdf2mjcf は `<geom ... scale="0.001 …">` に書くが MuJoCo 3.17 は
   geom の `scale` を受け付けない（`unrecognized attribute: 'scale'`）。
   → mesh asset 側へ移す（12 個の geom で発生）。
2. **RealSense の `.dae`**: `package://realsense2_description/meshes/d435.dae` が残る。
   MuJoCo は `.dae` を読めない（腕に不要）→ 削除。
3. **面数の上限**: MuJoCo は 1 メッシュ 20 万面まで。`turtlebot3_bigwheel_orin/base.stl` は
   **584,632 面**（ヘッダとサイズは整合、バイナリ STL）で拒否される。
   エラーメッセージは "perhaps this is an ASCII file?" と出るが**実際の原因は面数**。
   → 台車・車輪・LiDAR は collision のプリミティブ（box/cylinder）を使い visual を落とす
   （MuJoCo 内蔵 importer はそうしている）。腕のメッシュは 2,740 面で問題なし。

## 改訂した進め方

**物理バックエンド差し替え（L1+L2 同時）**

1. `mujoco_ros2_control` を Humble でビルド（`mujoco_vendor` ごと、colcon）
2. Lime の MJCF シーンを作る（URDF→urdf2mjcf→後処理、床・立方体・カメラを含む）
3. `turtlebot3_lime_system.ros2_control.xacro` のプラグイン名を Gazebo から MuJoCo に差し替え
   （**コントローラ設定はそのまま流用**）
4. `camera_plugin` を既存のカメラトピック名・光学系に合わせる（`/camera/camera/color/image_raw`,
   `camera_info`）＋今の静的 TF（`0.059 0 -0.002`）はそのまま
5. `base_velocity_plugin` で `/cmd_vel` を接続
6. `bin/cccsim` の gazebo ステージを MuJoCo ステージに置き換え（`gazebo_grasp_fix` は使わない
   — そもそも「保持できるか」を測るのが目的）
7. そのまま `--tree ccapproach --grasp --record` を走らせ、**持ち上げ後に立方体が指の間に
   残るか**を測る

**残る自作部分**: シーン MJCF、`<mujoco>` 設定、`rsp_sim.launch.py`/`stack.sh` の
Gazebo 固有部分（RealSense plugin 除去・world ファイル・grasp plugin）の差し替え。

**`sim/mj/` の自作スクリプトの位置づけ**: ROS を介さず秒単位で回るので、
**摩擦係数や把持力の閾値を掃引する実験台**として残す（ROS 統合の代替ではなく補完）。

## 参照した一次情報（取得時点）

- `gh api repos/ros-controls/mujoco_ros2_control/readme`, `contents/mujoco_ros2_control_plugins/src`
- `gh api repos/kscalelabs/urdf2mjcf/readme`, PyPI v0.2.39 の実行
- `gh search code open_manipulator_sara` / `turtlebot3_lime mjcf`（MJCF 0 件）
- `gh api repos/google-deepmind/mujoco_menagerie/contents`（robotis: `dynamixel_2r`,
  `robotis_omy_3m`, `robotis_op3` のみ）
- `gh api repos/momoiorg-repository/Turtlebot3_Lime_IsaacSim_Humble/...`
- `gh api repos/Genesis-Embodied-AI/Genesis`
