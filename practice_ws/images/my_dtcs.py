import cv2  # openCVライブラリのインポート
import numpy as np  # numpyライブラリのインポート
from cv2 import aruco, imread, imwrite

##　↓↓↓↓↓↓↓inRangeWrap, calc_centroidは変更しないでください↓↓↓↓↓↓
# inRangeを色相が0付近や180付近の色へ対応する形へ修正
def inRangeWrap(hsv, lower, upper):
    if lower[0] <= upper[0]:
        return cv2.inRange(hsv, lower, upper)
    else:
        # 180をまたぐ場合
        lower1 = np.array([0, lower[1], lower[2]])
        upper1 = np.array([upper[0], upper[1], upper[2]])
        lower2 = lower
        upper2 = np.array([179, upper[1], upper[2]])
        return cv2.bitwise_or(
            cv2.inRange(hsv, lower1, upper1),
            cv2.inRange(hsv, lower2, upper2)
        )
    
def calc_centroid(mask):
    M = cv2.moments(mask)
    if M["m00"] != 0:
        # 重心座標を計算SS
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        s = np.count_nonzero(mask)/(mask.shape[0]*mask.shape[1])
        return cx, cy, s
    else:
        return None   
##　↑↑↑↑↑↑↑inRangeWrap, calc_centroidは変更しないでください↑↑↑↑↑↑↑

def d_ball(img):
    # 画像の読み込み
    draw_img = img.copy() # 元データを書き換えないようにコピーを作成
    # HSVに変換（色指定はRGBよりHSVの方が扱いやすい）
    hsv_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2HSV)

    # BGR空間での抽出範囲
    ## ボール
    lower = np.array([0, 220, 170]) # 色相, 彩度, 明度 の下限
    upper = np.array([10, 240, 255]) # 色相, 彩度, 明度 の上限

    # 指定範囲に入る画素を抽出（白が該当部分）
    mask = inRangeWrap(hsv_img, lower, upper)
    
    try:
        x, y, s = calc_centroid(mask)
        print(f"{s=}")
        return x, y
    except TypeError:
        return None

def d_coke(img):
    # 画像の読み込み
    draw_img = img.copy() # 元データを書き換えないようにコピーを作成
    # HSVに変換（色指定はRGBよりHSVの方が扱いやすい）
    hsv_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2HSV)

    # BGR空間での抽出範囲
    ## コーラ缶
    lower = np.array([170, 230, 0]) # 色相, 彩度, 明度 の下限
    upper = np.array([180, 250, 255]) # 色相, 彩度, 明度 の上限

    # 指定範囲に入る画素を抽出（白が該当部分）
    mask = inRangeWrap(hsv_img, lower, upper)
    
    try:
        x, y, s = calc_centroid(mask)
        print(f"{s=}")
        return x, y
    except TypeError:
        return None

def d_circle(img):
    # 画像読み込み
    draw_img = img.copy()

    # 前処理（グレースケール＋ぼかし）
    gray = cv2.cvtColor(draw_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)  # ノイズ低減

    # 円検出（HoughCircles）
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=30,
        param1=100, param2=50,   
        minRadius=10, maxRadius=40  
    )
    # マスク作成（検出円を塗りつぶし）
    mask = np.zeros(gray.shape, dtype=np.uint8)
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for x, y, r in circles:
            cv2.circle(mask, (x, y), r, 255, -1)     # マスク（白塗り）
            break

    x, y, s = calc_centroid(mask)
    print(f"{s=}")
    if x and y:
        return x, y
    else:
        return None

def d_cube(img):
    print("👉 [1] d_cube started")
    if img is None:
        return None

    draw_img = img.copy() # 元データを書き換えないようにコピーを作成
    hsv_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2HSV)

    # 1. 各色のHSV範囲を定義（白は誤検出防止のため除外しています）
    color_ranges = {
        "red":    (np.array([170, 120, 50]), np.array([10, 255, 255])),  # 赤（180をまたぐ）
        "blue":   (np.array([100, 120, 50]), np.array([130, 255, 255])), # 青
        "yellow": (np.array([22, 120, 50]),  np.array([35, 255, 255])),  # 黄
        "orange": (np.array([10, 120, 50]),  np.array([21, 255, 255])),  # 橙
        "green":  (np.array([38, 100, 50]),  np.array([85, 255, 255]))   # 黄緑〜緑
    }

    detected_masks = []
    pixel_threshold = 50 # 【調整可能】1色あたり最低限必要な画素数（面積）

    # 2. 各色ごとにマスク画像を作って、一定以上の面積があるか調べる
    for color_name, (lower, upper) in color_ranges.items():
        # 既存の inRangeWrap 関数を使用
        mask = inRangeWrap(hsv_img, lower, upper)
        
        # 小さなノイズ（ゴミ）を消す処理
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 画面内にその色が何ピクセルあるかカウント
        num_pixels = np.count_nonzero(mask)
        if num_pixels > pixel_threshold:
            # 条件を満たした色マスクをリストに保存
            detected_masks.append(mask)

    # 3. 鮮やかな色が 「2色以上」 見つかった場合のみ、ルービックキューブと判定
    if len(detected_masks) >= 2:
        # 見つかったすべての色のマスクを1つに合体（合成）させる
        combined_mask = detected_masks[0]
        for m in detected_masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, m)
            
        try:
            # 既存の calc_centroid 関数を使用して、合体したマスク全体の重心を計算
            x, y, s = calc_centroid(combined_mask)
            print(f"🟢 [2] d_cube success! Found colors: {len(detected_masks)}")
            return x, y
        except TypeError:
            return None
        
    # 1色以下しか見つからない場合は、キューブではないと判断して無視する
    print(f"🔴 [2] d_cube failed: found only {len(detected_masks)} colors")
    return None