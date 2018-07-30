import sys
import time
from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
import cv2

namesfile = None


def detect(cfgfile, weightfile):
    """Method that opens the webcam feed and pushes these images through YOLO"""
    m = Darknet(cfgfile)
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    use_cuda = True
    if use_cuda:
        m.cuda()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()  # returns (True, image) if everything went well
        sized = frame[:m.width, :m.height]
        start = time.time()

        # nms thresholding etc happens here
        boxes = do_detect(m, sized, 0.3, 0.4, use_cuda)
        finish = time.time()
        class_names = load_class_names(namesfile)

        curr_frame = plot_boxes_for_webcam(sized, boxes, class_names, finish - start)
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 1000, 1000)
        cv2.imshow('frame', curr_frame)  # ,gray
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    if len(sys.argv) == 4:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]

        globals()["namesfile"] = sys.argv[3]
        detect(cfgfile, weightfile)
    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile imgfile names')

