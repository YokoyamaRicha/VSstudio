import shutil
import glob
import os
import cv2
import numpy as np

max = 200 # １個の動画から生成する画像数
input_path = "./dataset/mp4"
output_path = "./dataset/output_png"

# 矩形検出
def detect_rectangle(img):
    # グレースケール
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 背景の多少のノイズは削除する
    _, gray_img  = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)      
    # 輪郭検出
    contours, _ = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rect = None
    for contour in contours:
        # ある程度の面積が有るものだけを対象にする
        area = cv2.contourArea(contour, False);
        if area < 1000:
            continue
        # 輪郭を直線近似する
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour,  epsilon, True)
        # 最大サイズの矩形を取得する
        x, y, w, h = cv2.boundingRect(contour)
        if(rect != None):
            if(w * h < rect[2] * rect[3]):
                continue
        rect = [x, y, w, h]
    return rect

# 透過イメージ保存
def create_transparent_image(img):
        # RGBを分離
        ch_b, ch_g, ch_r = cv2.split(img[:,:,:3])

        # アルファチャンネル生成
        h, w, _ = img.shape
        ch_a = np.zeros((h, w) ,dtype = 'uint8')
        ch_a += 255
        # 各チャンネルを結合
        rgba_img = cv2.merge((ch_b, ch_g, ch_r, ch_a))
        # マスク
        color_lower = np.array([0, 0, 0, 255]) 
        # color_upper = np.array([80, 80, 80, 255]) 
        color_upper = np.array([40, 40, 40, 255]) 
        mask = cv2.inRange(rgba_img, color_lower, color_upper)
        return cv2.bitwise_not(rgba_img, rgba_img, mask=mask)

def save_image(class_name, img):
    path = "{}/{}".format(output_path, class_name)
    if os.path.exists(path) == False:
        os.makedirs(path)
    for i in range(1000):
        filename = "{}/{}.png".format(path, i)
        if os.path.exists(filename) == False:
            cv2.imwrite(filename, img)
            print(filename)
            return

def main():

    os.makedirs(output_path, exist_ok=True)

    if(os.path.exists(output_path)==False):
        os.makedirs(output_path)

    moves = glob.glob("{}/*.mp4".format(input_path))
    for move in moves:
        basename = os.path.basename(move) 
        class_name = basename.split('_')[0]


        cap = cv2.VideoCapture(move)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print("width:{}　height:{} frames:{}".format(width, height, frame_count))
        interval = int(frame_count / max)

        counter = 0
        while True:
            counter += 1    
            # カメラ画像取得
            _, frame = cap.read()
            if(frame is None):
                break
            if(counter%interval != 0):
                continue
            # 縮小
            frame = cv2.resize(frame, (int(width/2), int(height/2)))

            # 矩形検出
            rect = detect_rectangle(frame)
            if(rect != None):
                x, y, w, h = rect
                # 切り取り
                save_img = frame[y: y+h, x: x+w]
                # 透過保存
                img = create_transparent_image(save_img)
                save_image(class_name, img)
                # 表示
                frame = cv2.rectangle(frame, (x, y), (x+w, y+h),(0,255,0),2)

            # 画像表示
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()