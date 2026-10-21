# 立方体認識: 取り込める OSS の調査（2026-10-21）

把持の残課題は「指が立方体のどこに当たるか」だが、その前に**認識を外部から強化できるか**を
広く調べた。結論を先に書く。

- **いまのボトルネックは認識精度ではない**。色ブロブ + depth で立方体中心は 1 cm 程度で
  出ており（実測 0.841 m vs 真値 0.851 m @1 m）、把持の失敗は腕の可動域と指の当たり位置。
  認識に投資するなら「把持の改善に効く形」に限る。
- それでも取り込む価値があるのは **マーカー（ArUco）方式**。学習不要・CPU のみ・0.2–2 m で
  6DoF 姿勢が厳密に出る。`cv2.aruco` はこのイメージに既にあり（4.9.0）、apt にも
  `ros-humble-aruco-opencv` 2.4.2 / `ros-humble-aruco-ros` 5.0.6 がある。リポジトリには
  `aruco_box0..9`（マーカー付き箱モデル）まで同梱されているのに、**どのコードも使っていない**。
- マーカーを付けたくない場合の現実的な次善は **adaptive-vision の pose refine**。
  ただし adaptive-vision に**検出器は無い**（facelet の 2D 観測を与える前提）。

## 1. 既にこの手元にあるもの

### 1.1 公開側 `lime_tidyup/dev2`（いまの実装）

- `cube_bbox_source.py`: `detector` / `depth_blob` / `colour_blob`。色ブロブは 1 m で検出可。
- `cube_model_free_pose.py`: 幾何 + 色で 6DoF（1 m では画素不足で棄却、bbox のみ）。
- `cube8_two_face_color_frontend.py`: 2面カラーフロントエンド（取り込み済み）。
- `yolo_pose_node.py`: `/cube_pose_result` に `pose_3d`（depth 中央値の中心）を追加済み。

### 1.2 非公開 `nekomario28/lime-interactive-transport`（前回作ったもの）

`mirror/lime-tidyup-cube8-lane-20260826` に **Cube8 認識スタック一式**がある（386 ファイル）。

| ファイル | 役割 |
| --- | --- |
| `cube8_rtmpose.py` | RTMPose(ONNX) でキューブ 8 キーポイント推論（MMPose 互換） |
| `cube8_pnp.py` / `cube8_pnp_assist.py` | キーポイントからの PnP |
| `cube8_view_local_pnp.py` / `cube8_projective_hidden.py` | 視点ローカル PnP / 隠れ面の射影 |
| `cube8_two_face_color_frontend.py` / `cube8_two_face_view_local.py` | 2面の色で対称性を解く |
| `cube8_depth_assist.py` / `cube8_geometry.py` | 深度補助・幾何 |
| `cube8_perception_router.py` | 「どの認識部品を走らせるか」のコスト考慮ルーティング |
| `cube8_grasp.py` | 把持側 |
| `runs/pose/train-2/weights/best.pt` | 学習済みポーズ重み（6.2 MB） |
| ブランチ `handoff/cube8-rtmpose-checkpoint-20260825-2320` | RTMPose チェックポイントの引き継ぎ |

**実測の但し書き**（当時の結論）: この 8 キーポイント重みは、30 px の予算でも
どの立方体姿勢とも一致しない頂点を出した。だからモデルフリー（幾何＋色）に切り替えた。
ただし `view_local_pnp` / `projective_hidden` は **検出済み bbox の精緻化**には流用できる。

## 2. `adaptive-vision` から取り込めるもの

`~/adaptive-vision`（MIT/Apache、`publish = false`、`#![forbid(unsafe_code)]`）。

| クレート | 提供 | 本件での使い道 |
| --- | --- | --- |
| `av-cube` | 立方体ドメイン（facelet、`CubeObjectModel`、`project_face_grid`、色分類 Lab/HSV、`detect_move`） | 面ごとの色判定とラベル付け |
| `av-pose` | `CubePoseTracker`（DLT PnP refine）、`AdaptiveCubeTracker` | **bbox から得た観測を 6DoF に精緻化** |
| `av-geometry` | `pnp.rs` / `robust_pnp.rs` / `depth.rs` / `camera.rs` | 姿勢推定の数値部品 |
| `av-pipeline` | 注釈パイプライン CLI（`av-annotate` 等） | 姿勢は**与える前提**（自前検出なし） |

要点: **検出器は無い**。`av-pipeline` の doc にも「It never measures a pose: the caller's pose
provenance is retained」と明記されている。したがって取り込むなら
「自前の色ブロブ bbox → facelet 観測 → av-pose で refine」という形になる。
Rust なので Python からは CLI 化（`crates/av-pipeline/src/bin` に前例あり）か、
DLT refine 部分（100 行程度）を Python に移植するのが軽い。

## 3. 外部 OSS（実測で確認した事実）

| 候補 | 実績 | 方式 | 本件での評価 |
| --- | --- | --- | --- |
| `cv2.aruco`（既存 4.9.0） | OpenCV contrib | マーカー → `solvePnP` | **最有力**。CPU のみ、学習不要、0.2–2 m で厳密 |
| `ros-humble-aruco-opencv` 2.4.2（apt） | パッケージ提供あり | 同上の ROS2 ラッパ | すぐ入る |
| `ros-humble-aruco-ros` 5.0.6（apt） | パッケージ提供あり | 同上 | 使うならこちらが新しい |
| `JMU-ROBOTICS-VIVA/ros2_aruco` | 142★（2024-02） | 同上 | 参考実装として十分 |
| `AIRLab-POLIMI/ros2-aruco-pose-estimation` | 75★（2024-07） | RGB(+depth) で姿勢 | 同上 |
| `NVlabs/FoundationPose` | 3646★（CVPR24） | RGB(D) → 6DoF、モデル必要 | GPU/重依存。この環境では非現実的 |
| `ammar-n-abbas/FoundationPoseROS2` | 299★ | 上記の ROS2 統合 | 同上（GPU 前提） |
| `JiehongLin/SAM-6D` | 731★ | セグメンテーション + 6DoF | 同上 |
| `THU-DA-6D-Pose-Group/GDR-Net` | 350★ | 単一 RGB → 6DoF | 同上（学習済み前提） |
| `ROS2 Humble` 系 AprilTag ラッパ（数件、0–12★） | 小規模 | マーカー | ArUco と同系統、優劣なし |
| Rubik 専用の小規模リポジトリ（1–38★） | 数千行 | スマホ写真の色グリッド解析 | ROS 統合なし。色分類の参考程度 |

Hugging Face には取り込める Rubik 検出/姿勢モデルは無かった（データセット・無関係モデルのみ）。

## 4. 推奨（効く順）

1. **マーカー方式（ArUco）**: シミュの立方体にマーカーを貼り、`cv2.aruco` で厳密姿勢。
   把持の「どこを掴むか」を確定させる用途に直結する。リポジトリ同梱の `aruco_box*` は
   いま未使用なので、そこに接続するのが自然。
2. **adaptive-vision の refine**: マーカーを付けたくない場合。自前 bbox → facelet 観測 →
   `CubePoseTracker::refine`。`av-cube` の色分類は、いまの `cube_face_colors.py` と重複するので
   置換対象にもなる。
3. **非公開リポジトリの `cube8_*` の再評価**: 特に `view_local_pnp` / `projective_hidden` は
   精緻化部品として流用価値あり（重み自体は当時不整合と測定済み）。
4. FoundationPose 系は GPU が使える環境が前提。今回は見送り。

## 5. 注意

把持の失敗は認識ではなく**腕の可動域（掌を下に向けた姿勢が IK 不成立）と指の当たり位置**
に起因することが実測で分かっている。認識を強化しても、この 2 点を解かない限り
「掴めた」にはならない。立方体が吹き飛ぶ挙動は Gazebo 側の問題として切り分け済み。

## 6. 実装と実測（silhouette refine を試した結果）

前節の「2. adaptive-vision 型の refine」を実際に実装した（`cube_pose_tracking.py`）。

- 事前姿勢から 8 頂点を投影 → 可視頂点を `cube_visibility` で選別 → 観測シルエットの角と
  **重心まわりの角度で 1 対 1 対応** → `estimate_cube_pose_3d` で PnP。
- 合成画像のユニットテスト 4 本は通過（摂動させた事前姿勢から真値へ収束、破綻時は拒否）。
- **実画像では通用しなかった**。実測（0.4 / 0.6 / 0.8 / 1.0 m、真値を事前姿勢に与えても）:
  PnP が再投影ゲートで棄却、または約 **130° 反転した解**を返す。
- 原因は実測で特定: 立方体は床に軸合わせで置かれ、カメラは正面寄りなので、**シルエットの角は
  4 点（共面）**しか得られない。共面 4 点の PnP は鏡像曖昧を持つ。
  さらに色マスクは赤い前面しか覆わず（白い上面が抜ける）、それを床との色差マスクに替えても
  角数は 4 のままだった。
- したがって距離を伸ばす認識には、**非共面の頂点が 6 点以上**必要 = 上面の奥の角が見える
  視点/画素数が要る。dev2 の two-face ルートが「立方体 250 px 必要」としているのと同じ制約。

**結論**: refine の実装は正しいが、この配置（軸合わせ＋正面寄り＋40–100 px）では
情報量が足りない。実機で効かせるには (a) 立方体に近づく、(b) 見下ろし気味の視点にする、
(c) 上面の奥行き corner を検出するキーポイント検出器（前作 `cube8_rtmpose` 系）を併用する、
のいずれかが要る。マーカー無しで距離を伸ばす唯一の現実解は (c)。

## 7. 実機のマウントに合わせた再測定（2026-10-21 追記）

これまでの測定は**存在しないカメラ**（base_link +0.155 前・+0.18 上・26° 下向き）で
行われていた。実機の RealSense は「2 枚のサンドイッチ板のあいだ、低い位置、立方体と同程度の
高さ」。上流の world ファイル（`practice_ws/worlds/test_house.world`）にその姿勢がそのまま
残っていた: `<pose>0.05915 0 0.0583 0 -0 0</pose>` = base_link から 0.059 前・高さほぼ同じ・
**傾きなし**。`sim/camera_mount.py` をこの値に合わせた。

同じハーネスで再測定:

| 指標 | 誤ったマウント（傾き 26°） | 実機マウント（水平・低位置） |
| --- | --- | --- |
| depth 中心の位置誤差 0.4–1.0 m | 7.8 / 7.6 / 7.8 / 8.1 cm | **1.00 / 1.07 / 1.15 / 1.26 cm** |
| silhouette refine | 130° 反転 or 棄却 | **棄却**（可視対応点が 4 未満） |

理由も実測と一致する: カメラが立方体と同じ高さで水平なので、見えるのは**正面 1 面のみ**。
4 点は共面で、共面 PnP は鏡像曖昧を持つため refine は答えを出さない。

**実機での含意**: この視点では「単一フレームの素の立方体」から得られるのは
**(a) 位置 ~1 cm** と **(b) 正面の面姿勢（yaw は 90° 単位で曖昧）**。yaw の曖昧さは
面の色パターン（`cube_face_colors` / `rubik_pose_cube`）で解ける。完全な 6DoF が要るなら
もう 1 視点（回り込む）か、上面の奥 corner が見える視点が要る。
