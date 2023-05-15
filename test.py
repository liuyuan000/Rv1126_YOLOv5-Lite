import numpy as np
import cv2
import os
import urllib.request
from matplotlib import gridspec
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.python.platform import gfile
from rknn.api import RKNN


GRID0 = 80
GRID1 = 40
GRID2 = 20
LISTSIZE = 6
SPAN = 3
NUM_CLS = 1
MAX_BOXES = 500
OBJ_THRESH = 0.5
NMS_THRESH = 0.6
IMG_SIZE=640

CLASSES = ["mask"]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov3_post_process(input_data):
    # yolov3
    masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    stride = [8,16,32]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
              [59, 119], [116, 90], [156, 198], [373, 326]]
    # yolov3-tiny
    # masks = [[3, 4, 5], [0, 1, 2]]
    # anchors = [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]]
    generate_boxes = []
    pred_index = 0
    ratiow = 1
    ratioh = 1
    print(input_data.shape)
    for n in range(3):
        num_grid_x = int(IMG_SIZE / stride[n])
        num_grid_y = int(IMG_SIZE / stride[n])
        for q in range(3):
            anchor_w = anchors[n*3+q][0]
            anchor_h = anchors[n*3+q][1]
            for i in range(num_grid_y):
                for j in range(num_grid_x):
                    preds = input_data[pred_index]
                    box_score = preds[4]
                    if(box_score>OBJ_THRESH):
                        class_score = 0
                        class_ind = 0
                        for k in range(NUM_CLS):
                            if preds[k+5]>class_score:
                                class_score = preds[k+5]
                                class_ind = k
                        cx = (preds[0]*2-0.5+j)*stride[n]
                        cy = (preds[1]*2-0.5+i)*stride[n]
                        w = np.power(preds[2]*2,2)*anchor_w
                        h = np.power(preds[3]*2,2)*anchor_h

                        xmin = (cx - 0.5*w)*ratiow
                        xmax = (cx + 0.5*w)*ratiow
                        ymin = (cy - 0.5*h)*ratioh
                        ymax = (cy + 0.5*h)*ratioh

                        generate_boxes.append([xmin, ymin, w, h, class_score, class_ind])
                    pred_index += 1



    print(generate_boxes)

    generate_boxes = np.array(generate_boxes)
    classes = generate_boxes[:,5]

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        # b = boxes[inds]
        # c = classes[inds]
        # s = scores[inds]
        b = generate_boxes[inds][:,0:4]
        s = generate_boxes[inds][:,4]
        c = generate_boxes[inds][:,5]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        cl = int(cl)
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(x, y, x+w, y+h))

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
    cv2.imwrite("result.jpg",image)



if __name__ == '__main__':

    ONNX_MODEL = './simplify.onnx'
    RKNN_MODEL_PATH = './yolov5.rknn'
    im_file = './mask.jpg'
    DATASET = './dataset.txt'


    # Create RKNN object
    rknn = RKNN()

    NEED_BUILD_MODEL = True

    if NEED_BUILD_MODEL:
        # Load caffe model
        print('--> Loading model')
        ret = rknn.load_onnx(model=ONNX_MODEL)
        if ret != 0:
            print('load caffe model failed!')
            exit(ret)
        print('done')

        rknn.config(reorder_channel='0 1 2', mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],target_platform='rv1126')

        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
        if ret != 0:
            print('build model failed.')
            exit(ret)
        print('done')

        # Export rknn model
        print('--> Export RKNN model')
        ret = rknn.export_rknn(RKNN_MODEL_PATH)
        if ret != 0:
            print('Export rknn model failed.')
            exit(ret)
        print('done')
    else:
        # Direct load rknn model
        print('Loading RKNN model')
        ret = rknn.load_rknn(RKNN_MODEL_PATH)
        if ret != 0:
            print('load rknn model failed.')
            exit(ret)
        print('done')

    print('--> init runtime')
    # ret = rknn.init_runtime()
    ret = rknn.init_runtime(target='rv1126')
    if ret != 0:
        print('init runtime failed.')
        exit(ret)
    print('done')

    img = cv2.imread(im_file)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # inference
    print('--> inference')
    outputs = rknn.inference(inputs=[img])
    print('done')
    



    boxes, classes, scores = yolov3_post_process(outputs[0][0])

    image = cv2.imread(im_file)
    image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
    if boxes is not None:
        draw(image, boxes, scores, classes)

    cv2.imwrite("result.jpg",image)
    # cv2.imshow("results", image)
    # cv2.waitKey(5000)

    rknn.release()

