# cube8 レーンの retire 記録

Date: 2026-10-20

## 何をしたか

| 対象 | 操作 | 場所 |
| --- | --- | --- |
| `work/dev2-cube8-runtime-20260826`（tip `3347743`, 131 commits） | **private へミラー** | `nekomario28/lime-interactive-transport` の `mirror/lime-tidyup-cube8-lane-20260826`（同一 SHA `3347743` を検証） |
| 同上（public `lime_tidyup`） | **削除** | `origin/work/dev2-cube8-runtime-20260826` |
| レーン専用 smoke / CI 定義 | **private へ移動** | `nekomario28/lime-interactive-transport` の `test/cube8-lane-smokes` → `tests/lane/` |
| ローカル `cube8fix` ブランチ | **削除**（コードは dev2 に移植済み、テストは private に退避済み） | `lime_tidyup` |
| ローカル退避 | 残置 | `lime_tidyup` の `backup/cube8-lane-20260826` |

## なぜ

- `dev2` が**実行される唯一のコピー**（当時の `bin/ccc` / `bin/ccc2` は `./yolo_ws` をコンテナに
  mount して colcon build する）。レーン側の修正は原理的に走らない。
- レーンと `dev2` は同じ 6 モジュールを二重に持っており、`dev2` が上位互換
  （6 色語彙＋サポート順の 2 面選択、ラベル探索、シルエット精密化）。
- レーンの CI は未 push のまま一度も走っておらず、self-hosted runner 前提。
  失う実利がない。

## 参照

- レーン唯一のテスト主張は `dev2` の単体テストに移植済み
  （`test_cube_model_free_pose.py`: 曖昧性の拒否、シルエットスナップ）。
- レーンの設計判断は `doc/research/2026-10-19-two-face-frontend-index-check.md`
  と `doc/design/2026-10-20-model-free-pose-route.md` に残っている。

## 残したもの

`origin/work/dev2-sample-viz-20260826`（`a99871e`）は**残置**。これは可視化の
一点物証跡レーンで、今回の retire 対象（cube8 ランタイム）ではない。不要なら
別途指示があってから消す。
