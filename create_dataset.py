import json
import glob
import random
import os
import shutil
import math
import numpy as np
import cv2
from PIL import Image

MAX = 100 

CLASS_NAME=["BOX"]
COLORS = [(0,0,175)]

BACKGROUND_IMAGE_PATH = "./dataset/background_images"
TARGET_IMAGE_PATH = "./dataset/output_png"
OUTPUT_PATH = "./dataset/output_ground_truth"

S3Bucket = "s3://ground_truth_dataset"
manifestFile = "output.manifest"


BASE_WIDTH = 200  
BACK_WIDTH = 640  
BACK_HEIGHT = 480 

class Background:
    def __init__(self, backPath):
        self.__backPath = backPath

    def get(self):
        imagePath = random.choice(glob.glob(self.__backPath + '/*.jpg'))
        return cv2.imread(imagePath, cv2.IMREAD_UNCHANGED) 

class Target:
    def __init__(self, target_path, base_width, class_name):
        self.__target_path = target_path
        self.__base_width = base_width
        self.__class_name = class_name

    def get(self, class_id):
        
        class_name = self.__class_name[class_id]
        image_path = random.choice(glob.glob(self.__target_path + '/' + class_name + '/*.png'))
        target_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) 

        
        h, w, _ = target_image.shape
        aspect = h/w
        target_image = cv2.resize(target_image, (int(self.__base_width * aspect), self.__base_width))

        
        mode = random.randint(0, 3)
        if(mode == 0):
            target_image = cv2.rotate(target_image, cv2.ROTATE_90_CLOCKWISE)
        elif(mode == 1):
            target_image = cv2.rotate(target_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif(mode == 2):
            target_image = cv2.rotate(target_image, cv2.ROTATE_180)
        return target_image


class Transformer():
    def __init__(self, width, height):
        self.__width = width
        self.__height = height
        self.__min_scale = 0.3
        self.__max_scale = 1

    def warp(self, target_image):
        
        target_image = self.__resize(target_image)

        
        mode = random.randint(0, 3)
        if(mode == 0):
            target_image = self.__rote(target_image, random.uniform(0, 30))
        elif(mode == 1):
            target_image = self.__rote(target_image, random.uniform(320, 360))

        
        h, w, _ = target_image.shape
        left =  random.randint(0, self.__width - w)
        top = random.randint(0, self.__height - h)
        rect = ((left, top), (left + w, top + h))

        new_image = self.__synthesize(target_image, left, top)
        return (new_image, rect)

    def __resize(self, img):
        scale = random.uniform(self.__min_scale, self.__max_scale)
        w, h, _ = img.shape
        return cv2.resize(img, (int(w * scale), int(h * scale)))

    def __rote(self, target_image, angle):
        h, w, _ = target_image.shape
        rate = h/w
        scale = 1
        if( rate < 0.9 or 1.1 < rate):
            scale = 0.9
        elif( rate < 0.8 or 1.2 < rate):
            scale = 0.6
        center = (int(w/2), int(h/2))
        trans = cv2.getRotationMatrix2D(center, angle , scale)
        return cv2.warpAffine(target_image, trans, (w,h))

    def __synthesize(self, target_image, left, top):
        background_image = np.zeros((self.__height, self.__width, 4), np.uint8)
        back_pil = Image.fromarray(background_image)
        front_pil = Image.fromarray(target_image)
        back_pil.paste(front_pil, (left, top), front_pil)
        return np.array(back_pil)

class Effecter():

    
    def gauss(self, img, level):
        return cv2.blur(img, (level * 2 + 1, level * 2 + 1))

    
    def noise(self, img):
        img = img.astype('float64')
        img[:,:,0] = self.__single_channel_noise(img[:,:,0])
        img[:,:,1] = self.__single_channel_noise(img[:,:,1])
        img[:,:,2] = self.__single_channel_noise(img[:,:,2])
        return img.astype('uint8')

    def __single_channel_noise(self, single):
        diff = 255 - single.max()
        noise = np.random.normal(0, random.randint(1, 100), single.shape)
        noise = (noise - noise.min())/(noise.max()-noise.min())
        noise= diff*noise
        noise= noise.astype(np.uint8)
        dst = single + noise
        return dst


def box(frame, rect, class_id):
    ((x1,y1),(x2,y2)) = rect
    label = "{}".format(CLASS_NAME[class_id])
    img = cv2.rectangle(frame,(x1, y1), (x2, y2), COLORS[class_id],2)
    img = cv2.rectangle(img,(x1, y1), (x1 + 150,y1-20), COLORS[class_id], -1)
    cv2.putText(img,label,(x1+2, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return img


def marge_image(background_image, front_image):
    back_pil = Image.fromarray(background_image)
    front_pil = Image.fromarray(front_image)
    back_pil.paste(front_pil, (0, 0), front_pil)
    return np.array(back_pil)


class Manifest:
    def __init__(self, class_name):
        self.__lines = ''
        self.__class_map={}
        for i in range(len(class_name)):
            self.__class_map[str(i)] = class_name[i]

    def appned(self, fileName, data, height, width):

        date = "0000-00-00T00:00:00.000000"
        line = {
            "source-ref": "{}/{}".format(S3Bucket, fileName),
            "boxlabel": {
                "image_size": [
                    {
                        "width": width,
                        "height": height,
                        "depth": 3
                    }
                ],
                "annotations": []
            },
            "boxlabel-metadata": {
                "job-name": "xxxxxxx",
                "class-map": self.__class_map,
                "human-annotated": "yes",
                "objects": {
                    "confidence": 1
                },
                "creation-date": date,
                "type": "groundtruth/object-detection"
            }
        }
        for i in range(data.max()):
            (_, rect, class_id) = data.get(i)
            ((x1,y1),(x2,y2)) = rect
            line["boxlabel"]["annotations"].append({
                "class_id": class_id,
                "width": x2 - x1,
                "top": y1,
                "height": y2 - y1,
                "left": x1
            })
        self.__lines += json.dumps(line) + '\n'

    def get(self):
        return self.__lines


class Data:
    def __init__(self, rate):
        self.__rects = []
        self.__images = []
        self.__class_ids = []
        self.__rate = rate

    def get_class_ids(self):
        return self.__class_ids

    def max(self):
        return len(self.__rects)

    def get(self, i):
        return (self.__images[i], self.__rects[i], self.__class_ids[i])

    def append(self, target_image, rect, class_id):
        conflict = False
        for i in range(len(self.__rects)):
            iou = self.__multiplicity(self.__rects[i], rect)
            if(iou > self.__rate):
                conflict = True
                break
        if(conflict == False):  
            self.__rects.append(rect)
            self.__images.append(target_image)
            self.__class_ids.append(class_id)
            return True
        return False


    def __multiplicity(self, a, b):
        (ax_mn, ay_mn) = a[0]
        (ax_mx, ay_mx) = a[1]
        (bx_mn, by_mn) = b[0]
        (bx_mx, by_mx) = b[1]
        a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
        b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)
        abx_mn = max(ax_mn, bx_mn)
        aby_mn = max(ay_mn, by_mn)
        abx_mx = min(ax_mx, bx_mx)
        aby_mx = min(ay_mx, by_mx)
        w = max(0, abx_mx - abx_mn + 1)
        h = max(0, aby_mx - aby_mn + 1)
        intersect = w*h
        return intersect / (a_area + b_area - intersect)


class Counter():
    def __init__(self, max):
        self.__counter = np.zeros(max)

    def get(self):
        n = np.argmin(self.__counter)
        return int(n)

    def inc(self, index):
        self.__counter[index]+= 1

    def print(self):
        print(self.__counter)

def main():


    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.mkdir(OUTPUT_PATH)

    target = Target(TARGET_IMAGE_PATH, BASE_WIDTH, CLASS_NAME)
    background = Background(BACKGROUND_IMAGE_PATH)

    transformer = Transformer(BACK_WIDTH, BACK_HEIGHT)
    manifest = Manifest(CLASS_NAME)
    counter = Counter(len(CLASS_NAME))
    effecter = Effecter()

    no = 0

    while(True):
        background_image = background.get()

      
        data = Data(0.1)
        for _ in range(20):
        
            class_id = counter.get()
   
            target_image = target.get(class_id)
      
            (transform_image, rect) = transformer.warp(target_image)
            frame = marge_image(background_image, transform_image)
   
            ret = data.append(transform_image, rect, class_id)
            if(ret):
                counter.inc(class_id)

        print("max:{}".format(data.max()))
        frame = background_image
        for index in range(data.max()):
            (target_image, _, _) = data.get(index)
           
            frame = marge_image(frame, target_image)

        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        
        frame = effecter.gauss(frame, random.randint(0, 2))
        frame = effecter.noise(frame)

        
        fileName = "{:05d}.png".format(no)
        no+=1

                    
        cv2.imwrite("{}/{}".format(OUTPUT_PATH, fileName), frame)
        
        manifest.appned(fileName, data, frame.shape[0], frame.shape[1])

        for i in range(data.max()):
            (_, rect, class_id) = data.get(i)
        
            frame = box(frame, rect, class_id)

        counter.print()
        print("no:{}".format(no))
        if(MAX <= no):
            break

    
        cv2.imshow("frame", frame)
        cv2.waitKey(1)

    
    with open('{}/{}'.format(OUTPUT_PATH, manifestFile), 'w') as f:
        f.write(manifest.get())

main()