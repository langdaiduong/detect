import time
from numpy.core.fromnumeric import shape
import tensorflow as tf
# from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from core.config import cfg

input_size = 416
video_path = "./data/video/haaa.mp4"

saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-classroom-tiny-416', tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']
#chay camera
# try:
#     vid = cv2.VideoCapture(int(0))
# except:
#     vid = cv2.VideoCapture(0)
#chay video
try:
    vid = cv2.VideoCapture(int(video_path))
except:
    vid = cv2.VideoCapture(video_path)

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names
# 0: person, 1: hand
def image_best(number_frame, minfame, classes = read_class_names(cfg.YOLO.CLASSES), allowed_classes=list(read_class_names(cfg.YOLO.CLASSES).values())):
    num_classes = len(classes)
    #print(num_classes)
    #lưu số frame detect của môi class
    count_frame_class = [0]*num_classes
    #lưu độ chính xác detect của môi class
    max_score_class = [0]*num_classes
    #số frame hiện tại
    count_frame = 0
    result_image = []
    #print(result_image)
    class_arr = [[]]*num_classes
    #lưu độ chính xác cao nhất của một class trong tất cả các frame
    while count_frame < number_frame:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        if count_frame % 2 == 0:
            # width:480 height: 854 cua video
            # frame_size = frame.shape[:2]
            # print(frame_size)
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            batch_data = tf.constant(image_data)
            #pred_bbox : chua toa do box du doan
            #array([[[ 0.16512309, -0.01713137,  0.66294336,  0.20054576, 0.00593987,  0.44626427]]]
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.2,
                score_threshold=0.5
            )
            #print(boxes)
            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            #print(pred_bbox)
            image = utils.draw_bbox(frame, pred_bbox)
            #print(pred_bbox)
            #######################################################################################################################################
            # xử lý đếm khung hình nhận là hand up, person và gán vào list thông tin khung hình có độ chính xác cao nhất
            for i in range(num_classes):
                if pred_bbox[2][0][i] == i:
                    if pred_bbox[1][0][i] > 0.5:
                        count_frame_class[i]+=1
                    if pred_bbox[1][0][i] > max_score_class[i] and pred_bbox[2][0][i] == i:
                        max_score_class[i] = pred_bbox[1][0][i]
                        class_arr[i].append([pred_bbox,image])
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)
            result = np.asarray(image)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        count_frame +=1
        print("frame:" + str(count_frame))
        classes = read_class_names(cfg.YOLO.CLASSES)
        for i in range(num_classes):
            print(classes[i] + ":" + str(count_frame_class[i]))
    cv2.destroyAllWindows()
    ###############################################################################
    #print(max_score_class)
    for z in range(num_classes):
        if count_frame_class[z] > minfame[z]:
            for i in range(len(class_arr[z])):
                if class_arr[z][i][0][1][0][z] == max_score_class[z] and class_arr[z][i][0][2][0][z] == z:
                    print("check",class_arr[z][i])
                    result_image.append(class_arr[z][i])
                    print(z, "->", result_image[z], "->",class_arr[z][i][0][2][0][z])

    classes = read_class_names(cfg.YOLO.CLASSES)
    max_score = [0]*num_classes
    result = []
    for i in range(num_classes):
        if count_frame_class[i] >= minfame[i]:
            print(i, "->", result_image[i][0][1][0][i], result_image[i][0][2][0][i])
            if result_image[i][0][1][0][i] > max_score[i] and result_image[i][0][2][0][i]==i:
                max_score[i] = result_image[i][0][1][0][i]
    print(max_score)
    for i in range(num_classes):    
        if count_frame_class[i] >= minfame[i]:
            if result_image[i][0][1][0][i] == max_score[i]:
                #print(result_image[i][0][0][1][0][i], "->",i)
                if result_image[i][0][2][0][i] == i:
                    arr_result = [[classes[result_image[i][0][2][0][i]]],result_image[i][0][0][0][i],result_image[i][0][1][0][i]]
                    result.append(arr_result)
                    #print("result", result)
                    if result_image[i][1] is not None:
                        image = Image.fromarray(result_image[i][1].astype(np.uint8))
                        image = cv2.cvtColor(np.array(result_image[i][1]), cv2.COLOR_BGR2RGB)
                        cv2.imwrite('./detections/detection'+ str(i) +'.png', image)
    return result
#image_best(number frame,[min frame class 0, min frame class 1])               
result = image_best(150, [10, 10, 1])
if result != []:
    print(result)
else: 
    print("no result")