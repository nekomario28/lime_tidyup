# LimeSimulDemo_Course の Gazebo 物理設定を使った把持試験

## 参照したもの

- リポジトリ: [tutai3-lec/LimeSimulDemo_Course](https://github.com/tutai3-lec/LimeSimulDemo_Course)
- 読み取り ref: `main` / `544d163cc385adb0ee5538183b3ec1c4eda58615`
- これは **Gazebo Classic 11 / ROS 2 Humble** の実装。新しい `gz sim` ではない。
- `pytwb_ws/resource/sim_house.world`: ODE, `max_step_size=0.001`, `real_time_update_rate=1000`; ground friction `mu=100`, `mu2=50`。
- `pytwb_ws/resource/turtlebot3_manipulation.urdf.xacro`: upstream OpenManipulator/TurtleBot3 記述に `gazebo_grasp_fix` プラグインを指定。
- README は Coke Can の質量を `0.39 kg` から約 `0.039 kg` に下げるよう案内。

## 実験

コースの ODE profile を Lime の簡易把持シーンへ一時適用した。元の `sim/world/cube_world.world`
にあった `quick` solver の 150 iterations / `sor=1.3` / 2 mm contact layer を除き、コースの
1 ms / 1000 Hz ODE defaults と床の `mu=100`, `mu2=50` を使った。**grasp plugin は無効**にし、
接触だけで cube を保持できるかを試した。コース由来の profile は試験中だけ使用し、作業ツリーへ
残していない。

実行（当時の sim ハーネス）: `./bin/cccsim --cube-distance 0.8 --tree ccapproach --grasp`、Gazebo Classic。
ローカル evidence: `yolo_ws/log/cccsim/course-physics-ab-distance08/`。

- Gazebo server、カメラ、3 controller、知覚、接近 tree が起動・完了。
- grasp 処理は target に到達し、指の stall を観測してから lift trajectory を送った。
- ただし、**lift 後も cube は床に残った**。最終 world pose の `z=0.02840 m` は床上の cube
  center（約 `0.0285 m`）であり、持ち上げ保持は成立していない。
- 初期距離 `1.0 m` ではこの approach tree の後でも IK が届かず、`cube at 0.336 m` で停止した。
  `0.8 m` は本試験で把持接触まで進むために使った初期配置であり、距離を選んだ結果はこの範囲で
  しか一般化できない。

同じ作業中に、grasp node が callback thread で spin しながら main thread からも
`rclpy.spin_until_future_complete()` を呼び、Humble の wait set で `IndexError: wait set index too big`
が出る問題を再現した。future を async に送り、callback thread が完了させた future を event で待つ
方式へ変更し、把持試験はその先の IK・接触・lift まで進むようになった。

## 結論の境界

この測定で確認したのは「コース由来 ODE profile で Lime のシミュレーションと接触動作が動く」こと。
**摩擦だけの把持保持は確認できなかった**。また、コースのデモが `gazebo_grasp_fix` を含み、
README が物体を 10 分の 1 の質量へ変更する手順を示しているため、そのデモが動くこと自体は
friction-only の L1 を証明しない。stall は接触の証拠にとどまり、保持の判定には lift 後の実 cube pose
を使う。
