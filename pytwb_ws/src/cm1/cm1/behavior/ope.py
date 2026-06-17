# import cv2

# # カメラデバイス取得
# cap = cv2.VideoCapture(0)
# # QRCodeDetectorを生成
# detector = cv2.QRCodeDetector()

# while True:
#     # カメラから1フレーム読み取り
#     ret, frame = cap.read()

#     # QRコードを認識
#     data = detector.detectAndDecode(frame)

#     # 読み取れたらデコードした内容をprint
#     if data[0] != "":
#         print(data[0])

#     # ウィンドウ表示
#     cv2.imshow('frame', frame)

#     # Qキー押すと終了
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # 終了処理
# cap.release()
# cv2.destroyAllWindows()
