from ultralytics import YOLO

# 1. 載入你剛剛熱騰騰訓練好的專屬模型
model = YOLO('best.pt')

print("模型載入成功！準備開啟 WebCam...")

# 2. 開啟魔法！呼叫電腦的預設視訊鏡頭 (0)，並即時顯示偵測結果
# 參數說明：
# source="0" 代表預設鏡頭。如果你想辨識影片，可以改成 source="測試影片.mp4"
# show=True  代表直接跳出一個視窗秀出畫好框框的畫面
results = model.predict(source="0", show=True, conf=0.55)