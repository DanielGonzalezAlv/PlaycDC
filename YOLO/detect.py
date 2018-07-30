import sys
import time
from PIL import Image, ImageDraw
#from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet

namesfile=None
def detect(cfgfile, weightfile, imgfile):
    """Method that performs prediction given an image and a weightfile"""
    m = Darknet(cfgfile)
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    use_cuda = True
    if use_cuda:
        m.cuda()
    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((m.width, m.height))
    start = time.time()
    boxes = do_detect(m, sized, 0.1, 0.1, use_cuda)
    finish = time.time()
    print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))
    class_names = load_class_names(namesfile)
    plot_boxes(img, boxes, 'predictions.jpg', class_names)

if __name__ == '__main__':
    if len(sys.argv) == 5:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        globals()["namesfile"] = sys.argv[4]
        detect(cfgfile, weightfile, imgfile)
    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile imgfile names')
