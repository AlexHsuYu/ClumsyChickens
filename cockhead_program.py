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

     
videoWriter = cv2.VideoWriter('/home/lab/Documents/chicken.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (1284,726))
MODEL_DIR ='/home/lab/Documents/python/ClumsyChickens/logs'
weights_path = '/home/lab/Documents/python/ClumsyChickens/logs/chicken20200820T1955/mask_rcnn_chicken_0098.h5'

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
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image    
def display_instances(image, boxes, masks, class_ids,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    ax.imshow(frame.astype(np.uint8))        
    plt.savefig('/home/lab/Documents/video/'+str(a)+'_1'+'.jpg',bbox_inches='tight',dpi=100,pad_inches=0.0) 
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = [class_id]
            x = random.randint(x1, (x1 + x2) // 2)
            # caption = "{} {:.3f}".format(label, score) if score else label

            caption = "chicken {:.3f}".format(score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")
        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
             # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)           
    ax.imshow(masked_image.astype(np.uint8))

vc = cv2.VideoCapture('/home/lab/Documents/16week.avi')

c=1
a=1 
if vc.isOpened(): 
    rval , frame = vc.read()
else:
    rval = False
 
timeF = 10
 
while rval:   
    rval, frame = vc.read()
    if(c%timeF == 0):
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame=frame[:,:,::-1]
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], r['scores'],
            # show_bbox=True, show_mask=True,
            #title="Predictions"
        )
        plt.savefig('/home/lab/Documents/video/'+str(a)+'.jpg',bbox_inches='tight',dpi=100,pad_inches=0.0)
        a = a + 1

    c = c + 1
    cv2.waitKey(1)
vc.release() 

  





