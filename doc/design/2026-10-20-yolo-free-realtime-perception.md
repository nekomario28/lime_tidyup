# YOLO を捨てられるか — realtime camera だけで `ccc2` を回す設計

Date: 2026-10-20
Status: **姿勢モデルは不要（実測）。bbox の供給だけが残る要件**

## 問い

`ccc2` の知覚から、`yolo_ws`（ultralytics / torch / `best.pt` / RTMPose ONNX /
rtmlib）を全部落として、**realtime camera の画像だけ**でキューブ姿勢を出せるか。
BT（`SearchCube` → `TakeAim` → `GoFrontCube` → `ArmToCube`）は変えずに。

## 1. BT が実際に読むフィールド（実測）

`/cube_pose_result` の消費者を grep した結果、必要なのは **3 つだけ**:

| フィールド | 使う側 | 用途 |
| --- | --- | --- |
| `detections[].confidence` | `_read_best_cube_detection`, `SearchCube`, `TakeAim` | 候補の採否・しきい値 |
| `detections[].box_xyxy` | `TakeAim`, `GoFrontCube`, `SearchCube` | 画像中央合わせ・接近判定 |
| `detections[].pose_3d` | `cube_pose.py`, `GoFrontCube(use_pose_3d)` | 距離・姿勢の証拠 |

**`keypoints_xy` / `class_name` / `class_id` / `model_path` は誰も読んでいない。**
つまりキーポイントモデルを落としても BT 契約は壊れない。

## 2. 姿勢はモデル無しで出る（既に実測済み）

`doc/design/2026-10-20-model-free-pose-route.md`:

- 2 面の色領域 → シルエット精密化 → view-local PnP（ラベル探索）→ 色で物理面を確定
- **RMS 1.31 px / 並進 サブ mm / 色解決 true**（同一フレーム、学習モデル不使用）

必要なのは **bbox だけ**。そこが今回の測定対象。

## 3. bbox を色だけで作れるか（実測：無理）

全フレームから色シード（6 色＋白）を取り、最大 2 領域の union を bbox にする案を
実装して 2 つの実フレームで測った:

| フレーム | 最大領域 | 2 番目 | 結果 |
| --- | --- | --- | --- |
| fill-light（暗背景） | `white` 154,880 px（**床**） | `blue` 14,327 px | bbox 不成立（fail-closed） |
| single-sun（灰世界） | `white` 884,478 px（**床**） | `red` 14,280 px | bbox 不成立（fail-closed） |

理由は 2 つ:

1. **床が白ステッカーより明るい**。白シードは「低彩度＋高明度」なので床を拾う。
2. **赤と橙が同じ面を二重に拾う**（橙面の色相が両方のシードに入る）。

→ 色だけのブートストラップは、この照明では**成立しない**（設計としては
fail-closed で正しいが、実用にならない）。

## 4. 最適設計（BT 上）

```
camera (realtime) ─┐
                   ├─> bbox source ─┐
depth (RealSense) ─┘                │
                                    v
                        cube_color_perception  (1 node, cv2+rclpy)
                          two-face colour frontend + silhouette refine
                          + view-local PnP (label search)
                          + face-colour → physical face map
                                    │
                                    v
                     /cube_pose_result  (confidence, box_xyxy, pose_3d)
                                    │
                    existing BT: SearchCube → TakeAim → GoFrontCube → ArmToCube
```

### bbox source は差し替え可能な 1 点にする

| source | いつ使う | 実装 |
| --- | --- | --- |
| **`depth_blob`（推奨・実機）** | 実機 RealSense。作業距離帯（例 0.15–0.8 m）の**連結成分**で、面積・アスペクト・平面残差が妥当なもの | depth 画像のしきい値＋connectedComponents（学習不要・数 ms） |
| `detector`（sim/保険） | sim には現状 colour カメラしかない。既存 YOLO の box をそのまま使う | 現行 |
| `colour_pair`（実験） | 背景が単色と分かっている時だけ | 今回の試作。既定は無効 |

**sim の最適化**: `cube8_sim_setup.build_camera_sdf` に depth を足せば、sim でも
`depth_blob` で回せる（world は既に 57 mm キューブを置いている）。これで
`yolo` サービスごと不要になる。

### `confidence` の再定義（BT 契約を保つ）

従来は検出器の確率。モデル無しでは**幾何＋色の合格度**にする:

```
confidence = min(face_support_ok, geometry_gate_ok, 1 / (1 + rms_px), colour_resolved)
```

`SearchCube`/`TakeAim` のしきい値 0.30 は「候補の確からしさ」なので、この定義でも
意味が通る（幾何ゲート合格＋RMS<4px なら 0.5 以上になる）。

### 何が不要になり、何が残るか

| 不要 | 残る |
| --- | --- |
| `yolo` コンテナ / `yolo_ros2-image_yuu` | lime コンテナ（ROS, MoveIt, Nav2） |
| ultralytics, torch, torchvision | cv2, numpy, rclpy, tf2 |
| `best.pt`, RTMPose ONNX, rtmlib, onnxruntime | `pymoveit2`, `moveit`, `pytwb`/BT |
| `yolo_ws` の datasets/runs/log | `practice_ws`, `pytwb_ws` |

**削減効果（実測ベース）**: yolo イメージは torch+opencv で数 GB、モデル 6.5MB×n。
推論レイテンシも消える（色＋幾何は cv2 の数 ms オーダー。実測はしていない）。

## 5. 移行ステップ（各段の合格条件つき）

1. **bbox を外から注入できるようにする**（今は `estimate_model_free_pose` が
   bbox を受け取る形なので、`source='detector'` のまま 1 ノード化）
   → 合格: `/cube_pose_result` の JSON が現行と同形で、BT が無変更で動く。
2. **`depth_blob` を実装**（sim の depth カメラ or 実機 RealSense）
   → 合格: 2 フレーム以上で bbox が安定し、姿勢が RMS<4px。
3. **`ccc2` を 1 コンテナ化**（compose から `yolo` を外す）
   → 合格: `./bin/ccc2` が yolo 無しで `ccik` まで到達。
4. **colour_pair は実験専用**として残す（既定 off）
   → 合格: 単色背景ワールドでのみ使えることが doc に書かれている。

## 6. リスクと BT 側の受け方

| リスク | BT 側の受け方 |
| --- | --- |
| 1 面しか見えない | `SearchCube` が回り続ける（姿勢は出さない） |
| 色が未解決（影） | `ArmToCube` が動かない（fail-closed 実装済み） |
| 床が白い | `depth_blob` で回避。`colour_pair` は使わない |
| 実機の clutter | depth の作業距離帯＋平面残差＋アスペクトで棄却。棄却時は「未検出」= 探索継続 |

## 7. 未検証（この文書が主張しないこと）

- `depth_blob` は**未実装**。実装して測るまで「実機で YOLO 不要」とは言わない。
- 実機の照明・材質・clutter での色判定は未測定（sim の合成レンダのみ）。
- レイテンシ・FPS は測っていない（どちらの経路も）。
- 現状の sim カメラは colour のみなので、sim で `depth_blob` を測るには
  `build_camera_sdf` に depth sensor を足す必要がある。
