import cv2
import os
import sys
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
sys.path.append('/home/lab/Documents/python/ClumsyChickens') 
import cockheads
from cockheads import ChickenConfig
import colorsys
import random
from skimage import io
import itertools
import colorsys
from skimage.measure import find_contours
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mping



try:
    import mrcnn
except ImportError as err:
    MRCNN_ROOT = os.path.join(os.environ['HOME'], 'Documents/python/Mask_RCNN')
    # add Mask_RCNN to path
    sys.path.append(MRCNN_ROOT)
finally:
    from mrcnn.config import Config
    from mrcnn import utils
    from mrcnn import model as modellib
    from mrcnn import visualize

     
videoWriter = cv2.VideoWriter('/home/lab/Documents/chicken.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, (1080,1920))
MODEL_DIR ='/home/lab/Documents/python/ClumsyChickens/logs'
weights_path = '/home/lab/Documents/python/ClumsyChickens/logs/chicken20190925T2023/mask_rcnn_chicken_0040.h5'

weights_path = '/home/lab/Documents/20191210/mask_rcnn_chicken_0702.h5'

class ChickenInferenceConfig(ChickenConfig):
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = ChickenInferenceConfig()
config.display() 
model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)

model.load_weights(weights_path, by_name=True)

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors



def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image 
 
def display_instances(image, boxes, masks, class_ids,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
 
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    colors =  random_colors(N)
    masked_image = image.astype(np.uint8).copy()
    for i in range(N):
        color = colors[i]
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            image =  cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
               
        # Mask
        mask = masks[:, :, i]
        if show_mask:
            image = apply_mask(masked_image, mask, color)
            
        # Label
        if not captions:
            class_id = class_ids[i]
            
            score = scores[i] if scores is not None else None
            label = 'head'
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
    
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        image = cv2.putText(
            image, caption, (x1, y1),cv2.FONT_HERSHEY_SIMPLEX , 0.7, (255, 255,255 ), 1)
      return image        
 
vc = cv2.VideoCapture('/home/lab/Documents/test_2.avi')

APs = []
c=1
a=1
count = 0 
if vc.isOpened(): 
    rval , frame = vc.read()
else:
    rval = False
 
timeF = 25


while rval:   
    rval, frame = vc.read()
    if(c%timeF == 0):
       results = model.detect([frame], verbose=0)
        r = results[0]
        # frame=frame[:,:,::-1]
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], r['scores'],
            #show_bbox=False, show_mask=False,
            #title="Predictions"
        )
       if len(r['scores']) == 13: 
            count=count+1

        cv2.imshow('123',frame)
        cv2.imwrite('/home/lab/Documents/12345.jpg',frame)
         videoWriter.write(frame)
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        print(count)    
  
    cv2.waitKey(1)
vc.release()

  





