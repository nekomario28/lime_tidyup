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
   → 合格: yolo サービス無し（知覚ノードは `perception_backend:=color_frontend`）で、当時の測定用ツリー `ccik` まで到達。
4. **colour_pair は実験専用**として残す（既定 off）
   → 合格: 単色背景ワールドでのみ使えることが doc に書かれている。

## 6. リスクと BT 側の受け方

| リスク | BT 側の受け方 |
| --- | --- |
| 1 面しか見えない | `SearchCube` が回り続ける（姿勢は出さない） |
| 色が未解決（影） | `ArmToCube` が動かない（fail-closed 実装済み） |
| 床が白い | `depth_blob` で回避。`colour_pair` は使わない |
| 実機の clutter | depth の作業距離帯＋平面残差＋アスペクトで棄却。棄却時は「未検出」= 探索継続 |

## 7. 実装（step 2–3 完了）

| 実装 | 内容 | 検証 |
| --- | --- | --- |
| `cube_bbox_source.py` | `detector` / `depth_blob` を切り替える 1 点。`colour_pair` は**実験専用として明示的に拒否** | 17 unittest（近傍ブロブ受理、平面・斜面・小ブロブ・境界接触・範囲外を拒否、決定論、設定不正） |
| `cube_model_free_pose` | `bbox_source` / `depth_image` / `roi_xyxy` を受ける。結果に `bbox_source` / `bbox_quality` / `bbox_xyxy` を付与 | depth 経路のテスト 3 本（depth から姿勢、平面のみは拒否、bbox も depth も無い場合は拒否） |
| `yolo_pose_node` | `perception_backend=color_frontend` + `bbox_source=depth_blob` では**検出器を読み込まない**（`ultralytics` 不在でも import 可）。`depth_topic` / `depth_scale_m` を追加し、depth を購読して深度フレームから bbox を作る。`confidence` は幾何品質（色未解決でも接近は許可、フラグは `pose_3d` に載る） | `ultralytics` を欠落させた状態で import 成功を確認 |
| 知覚ノードを Lime コンテナ内で起動（当時: `docker-compose.ccc2.yaml` + `bin/ccc2 --no-yolo`）| `yolo` サービスを使わず、Lime コンテナ内で知覚ノードを build/起動（`yolo_ws` を mount）。depth topic/scale は CLI で指定 | dry-run で両モードのコマンド列を確認（--no-yolo: yolo なし、`bbox_source:=depth_blob`、`depth_topic` 指定） |

`confidence` の再定義について: 色未解決のときに confidence を 0 に落とすと
**接近そのものが止まり、2 面目が視野に入らず永久に未解決**になる。そのため
confidence は幾何品質のみとし、「色解決済みか」は `pose_3d` のフラグで運び、
把持可否は下流（`ArmToCube`）が判断する。

### 実測した `depth_blob` の位置づけ

**単位テスト（合成 depth）**: 0.45 m のブロブ＋0.95 m の平面で、平面のみ・斜面・
小ブロブ・境界接触・範囲外を棄却し、近傍ブロブを選ぶことを確認。

**sim の実 depth（Gazebo）**: `cube8_sim_setup.build_camera_sdf` に depth sensor
（`type='depth'`、topic は `/cube8_sim/depth/depth/image_raw`、32FC1、実測レンジ
0.301–1.993 m）を足して測定:

| 項目 | 値 |
| --- | --- |
| 経路 | depth blob → 2面色 → シルエット精密化 → PnP（ラベル探索）→ 物理面確定 |
| bbox source | `depth_blob`（**検出器なし・学習なし**）、quality 0.70 |
| 並進 | **(-0.0004, -0.0001, 0.351) m** vs 真値 (0, 0, 0.35) m ＝ **1.05 mm** |
| RMS | 1.03 px |
| 色解決 | **true**（white / blue / orange） |
| label swaps | (F, F, F)＝同一視点でもラベル探索が正しい割当を選んでいる |

→ **`--no-yolo`（1 コンテナ）が実データで成立する**ことを sim で確認。
**実機 RealSense の depth は依然未測定**。

## 8. 見える化（rqt / 内蔵ビューア）

知覚結果を**そのまま画面に出す**。`cube_overlay.py` が純関数で描き、ノードが
`debug_image_topic`（既定 `/cube_pose_debug_image`）へ publish する。描くのは
**測ったものだけ**:

| 描画 | 意味 |
| --- | --- |
| 緑の wireframe ＋ 3 軸 | 推定姿勢を投影。**ズレていれば一目で分かる** |
| 橙の矩形 ＋ `bbox: depth_blob q=0.70` | 使った bbox と**その供給元** |
| 黄の四角 ＋ `+x=white` 等 | 可視面の領域と**分類された色**（棄権は `?`） |
| 灰のポリゴン | 角をスナップしたシルエット |
| 上部の status 行 | `accepted` / `unresolved` / `rejected` ＋ **理由**、rms、bbox 元、色解決、`swaps=FFF` |
| 下部の色一覧 | 可視スロットの観測色 |

**拒否フレームも描く**（赤い status ＋ 理由）。「キューブは見えているのに知覚が
断った、なぜか」がこの表示の一番の用途で、空のウィンドウでは答えられない。

見る方法は 3 つ:

```bash
./bin/cubeview                        # OpenCV ウィンドウ（このホスト）
./bin/cubeview --rqt                  # rqt_image_view（無ければ案内を出す）
./bin/cubeview --snapshot out.jpg     # 1 枚保存して終了（ヘッドレス・CI 用）
./bin/cubeview --topic /other --scale 1.5
```

yolo 無しモードの実行中はログに `./bin/cubeview` の案内を出す。
実測（sim depth フレーム、検出器なし）: wireframe が 57 mm キューブに一致、
`+x=white +y=blue +z=orange`（すべて正解）、`rms=1.03px`、`colors=resolved`、
`swaps=FFF`。再現は `doc/research/probes/2026-10-20_render_overlay.py`。

## 9. sim での統合確認（実ノード、検出器なし）

Gazebo（生成した 57 mm キューブ、カメラ 0.45 m 高、depth sensor あり）に対して
**実ノード**を起動:

```bash
ros2 run barcode_detector yolo_pose_node --ros-args \
  -p perception_backend:=color_frontend -p bbox_source:=depth_blob \
  -p depth_topic:=/cube8_sim/depth/depth/image_raw -p depth_scale_m:=1.0 \
  -r /camera/camera/color/image_raw:=/cube8_sim/camera/image_raw \
  -r /camera/camera/color/camera_info:=/cube8_sim/camera/camera_info
```

| 項目 | 実測 |
| --- | --- |
| 起動ログ | `no detector loaded (model-free route, bbox from depth_blob)` |
| 周期 | 1 Hz で `published 1 model-free detections` |
| `/cube_pose_result` | `confidence 0.557`、`model_path: ""`、`model_sha256: ""` |
| `box_xyxy` | `[477.4, 197.4, 814.6, 534.6]`（depth blob 由来） |
| `pose_3d.position_m` | `(-0.00037, -0.00010, 0.35098)` vs 真値 (0,0,0.35) ＝ **0.98 mm** |
| `pose_3d` 他 | `rms 1.03 px`、`correspondence 6 / inlier 6`、`inlier_mask [0,0,1,1,1,1,1,1]`（隠れ頂点0 は観測に入らない） |
| 色 | `color_fixed_semantics_resolved: true`、`+x white / +y blue / +z orange`、`label_swaps [F,F,F]` |
| overlay | `/cube_pose_debug_image` に publish、`debug_image_path` へも保存（同一 77,249 bytes） |

overlay の実画像: `doc/research/evidence/2026-10-20-model-free-overlay-sim.jpg`
（wireframe がキューブに一致、bbox は橙枠で `bbox: depth_blob q=0.70`、面ラベルは
`+x=white +y=blue +z=orange`、上部に `accepted rms=1.03px … swaps=FFF`）。
再現手順: `doc/research/probes/2026-10-20_run_sim_nodector.sh`。

## 10. 未検証（この文書が主張しないこと）

- `depth_blob` は合成 depth でしか測っていない。**実機 RealSense の depth では未測定**。
- 実機の照明・材質・clutter での色判定は未測定（sim の合成レンダのみ）。
- レイテンシ・FPS は測っていない（どちらの経路も）。
- 現状の sim カメラは colour のみなので、sim で `depth_blob` を測るには
  `build_camera_sdf` に depth sensor を足す必要がある。
