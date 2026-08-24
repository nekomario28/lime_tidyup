# cc系Behavior Tree向け簡易Cube 3D Pose

## Scope / contract

- 対象: `practice_ws/trees/cc*.xml` の `GoFrontCube` 接近処理。
- 入力: YOLO Poseの固定順8頂点、各keypoint confidence、ROS 2
  `CameraInfo`。
- 出力: 既存 `/cube_pose_result` JSON内の任意 `pose_3d`。座標系は
  color camera frame、位置はm、姿勢はquaternion xyzw。
- `GoFrontCube` は鮮度、inlier数、reprojection errorを満たすposeのcamera
  Z距離のみを優先する。不成立時は既存Depth距離へ戻る。
- 非対象: map/base_linkへのTF変換、時系列平滑化、把持姿勢生成、MoveIt2計画、
  物理ロボットでの把持成功・安全性の証明。

## Live authority / revisions

- 実装先: `c0a25004a6/lime_tidyup`, branch `dev2`, base
  `18432b768d983940007c275a741c628c26b01dc4`。
- Portfolio policy: `nekomario28/project-incubator`
  `ed5f83869777cb1572a5afa28b25b5aebc03d9d7`。
- 主donor: `nekomario28/lime-interactive-transport` Draft PR #4 exact head
  `a7374141bc8484db798f858409033dc7fb266a02`。
- 補助境界: `nekomario28/decidelta`
  `4596d24ba6d1617dcce3d8a549b1107fd7b14862`,
  `docs/applications/motion-capture-routing.md`。
- 調査日: 2026-08-24 (Asia/Tokyo)。

`gh` はシステムPATHに存在しなかったため、同じ認証済みGitHub SSHのexact
repository/PR refsを読み取り取得した。GitHub SSH userは`nekomario28`。

## Candidates checked

| Candidate | Decision | Evidence / reason |
|---|---|---|
| 現行 `yolo_pose_node.py` + `system.py` | ADAPT | 2D bbox/keypoint JSONとDepth fallbackが既にあり、後方互換の追加面が最小。 |
| `lime-interactive-transport` Cube8 PnP | ADAPT | 8頂点の固定object geometry、CameraInfo、RANSAC PnP、inlier/reprojection gate、camera-frame poseとテストを実コードで確認。MIT。 |
| `decidelta` motion-capture routing | BORROW_CONCEPT | stale/low-confidence sourceを拒否しbounded fallbackを選ぶ境界のみ採用。連続fusion runtimeは持ち込まない。 |
| 新ROS interface package / world registry | HOLD | 簡易cc接近には過大。将来、複数consumerまたはmap-frame poseが必要になった時だけ再検討。 |
| bbox中心Depthのみ | SUPERSEDED_AS_PRIMARY | fallbackとして保持するが、accepted 3D poseがある場合の第一距離源にはしない。 |

## Decision / unique delta

既存JSONを壊さず、各detectionへ任意 `pose_3d` を足す。PnPが失敗しても2D
detectionは発行する。consumer側は、geometry ID、camera frame、3秒以内の鮮度、
4点以上のcorrespondence/inlier、XMLで指定した最大reprojection errorを満たす時だけ
Z距離を使う。それ以外はDepthを使う。

Reuseにより、独自PnP数学、独自頂点順、独自品質契約、新ROS message packageの設計を
避けた。実装する固有差分は既存String JSONと`GoFrontCube`への薄いadapterだけ。

## Risks / evidence boundary

- 8 keypointの学習ラベル順がdonorの固定頂点順と一致することは実データで未確認
  (`UNKNOWN`)。順が違えば姿勢は意味を持たない。
- cube side length既定値0.057mはdonor設定を採用。実物寸法との一致は
  `NOT RUN`。
- 静的テストはvalidation/fallback契約を検査するだけで、YOLO精度、camera calibration、
  Gazebo/実機のtranslation/rotation error、把持成功、安全性を証明しない。
- 知覚はcandidateを出すだけで、移動・把持許可のauthorityではない。

## Reusable-task capture

`NONE`。PnP coreの再利用候補は既に`lime-interactive-transport`がsemantic ownerであり、
この変更はLime固有JSON/BT adapterである。別consumerが必要になった場合はdonorからの
shared-component extractionを再検討する。

## Research stop reason / next safe action

強いportfolio donorの実コード・テスト・licenseとfallback境界を確認でき、残る不確実性は
追加検索より実データ検証で解消すべき段階になった。

## Verification receipt

- pure Python unit: 7 tests PASS（PnP acceptance/rejection 3、consumer
  freshness/quality/fallback 4）。
- Python compile: changed Python files PASS。一度、既存root所有`__pycache__`への
  書き込みが拒否されたため、等価な一時cache routeで再実行した。
- XML: `cc.xml`, `ccc.xml`, `ccab.xml`, `cccb.xml` parse PASS。
- ROS package build: `barcode_detector`と`cm1`を既存Docker imageの一時workspaceで
  `colcon build --packages-select`しPASS。新規moduleがinstall対象に含まれることを確認。
- OpenCV runtime: 既存`yolo_ros2-image_yuu`でsynthetic 8-point projectionを実
  `solvePnPRansac`へ入力し、truth `z=0.55m`に対して`z=0.55m`, RMS `0.0px`でPASS。
- ROS node dependency import: ROS 2、cv_bridge、Ultralytics込みでPASS。
- changed-file lint/docstring: `ament_flake8` / `ament_pep257` PASS。
- `barcode_detector` full `colcon test`: feature tests 3 PASS、package pep257 PASS、
  copyright 1 SKIP。flake8だけ既存未変更`roboflow_node.py:1`のunused `time`でFAIL。
  base `18432b7`にも同じ行があるため、本変更のgreen claimには含めない。
- `cm1/lib/actor/system.py` whole-file flake8: 77件FAIL。変更箇所以外を含む既存lint
  debtであり、今回の新規helper/wrapper/testはfile-scoped checkでPASS。
- 実カメラ、Gazebo scene、実機移動・把持: `NOT RUN`。

次の安全な作業は、実カメラで固定8頂点順と実物side lengthを確認し、CameraInfo frame、
PnP translation error、Depthとの差、rejection/fallbackログを記録すること。実機移動は、
この観測gateを通してから別途authorizeする。
