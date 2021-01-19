#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import sys
import cv2
import numpy as np
sys.path.append('/home/lab/Documents/python/ClumsyChickens')
import matplotlib
matplotlib.use('Agg')
from cockheads import ChickenConfig
import colorsys

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
warnings.filterwarnings('ignore')
import copy
import imutils
from imutils import perspective

from mrcnn import model as modellib

MODEL_DIR ='/home/lab/Documents/python/ClumsyChickens/logs'
weights_path = '/home/lab/Documents/python/ClumsyChickens/logs/resnext101_240/mask_rcnn_chicken_0239.h5'

class ChickenInferenceConfig(ChickenConfig):
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = ChickenInferenceConfig()
config.display() 
model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)

model.load_weights(weights_path, by_name=True)

class ReIDunit():
     #現在ID,顯示ID,bbox,存在,消失
     replyID = 0
     showID = 0
     bbox = []
     nowInFrame = True
     disappearTime = 0
     
     def __init__(self, track_id,bbox):
         self.replyID = self.showID = track_id
         #print(self.replyID)
         bbox = bbox
    
class ReIDcontroler():
    
    Have_maskrcnn_bbox = True#False
    Have_deep_sort_bbox = False
    Have_fit_bbox = False
    Have_mask = False
    Have_original_ID = False
    ID_reply_frame = 3
    ID_delete_frame = 5
    
    
    
    showtracker = []
    reid_array = []
    lossarray = []
    reid_array_ctr = 0
    
    file = None
    
    def __init__(self):
         self.file = open('data.txt', 'w')
         
    def RomoveClose(self, tracks, frame):
        for indexi, trackbboxi in enumerate(tracks):
            if not  trackbboxi.is_confirmed() or  trackbboxi.time_since_update > 1:
                continue 
            bboxi = trackbboxi.to_tlbr()
            pointi = ((int)((bboxi[2] + bboxi[0])/2), (int)((bboxi[3] + bboxi[1])/2))
            cv2.circle(frame, pointi, 2,  (0, 0, 255), 4)
            if pointi[0] < 260 or pointi[0] > 1725:
                del(self.showtracker[indexi])
                continue
            for indexj, trackbboxj in enumerate(tracks):
                if indexj <= indexi:
                    continue
                #print(len(tracks))
                
                if not  trackbboxj.is_confirmed() or  trackbboxj.time_since_update > 1:
                    continue 
                bboxj = trackbboxj.to_tlbr()
                
                iw = min(bboxi[2], bboxj[2]) - max(bboxi[0], bboxj[0])
                A_area =  B_area = IoU = 0
                if iw > 0:
                   ih = min(bboxi[3], bboxj[3]) - max(bboxi[1], bboxj[1])  
                   if ih > 0:
                        A_area = (bboxi[2] - bboxi[0]) * (bboxi[3] - bboxi[1])
                        B_area = (bboxj[2] - bboxj[0]) * (bboxj[3] - bboxj[1])
                        Aiou = iw * ih / float(A_area)
                        Biou = iw * ih / float(B_area)
                        IoU = max(Aiou, Biou) 
                
                if IoU >0.6:
                    if (A_area > B_area):
                        trackbboxj = trackbboxi
                    #print(Aiou,Biou, IoU, indexi)
                    del(self.showtracker[indexi])
     
    def DeleteOldID(self):
        for i in self.reid_array:
            if i.nowInFrame:
                i.nowInFrame = False
            else:
                i.disappearTime += 1
                if i.disappearTime > 100:
                    del(i)
        for i in self.lossarray:
             i['onfame'] = False
        #self.lossarray = []
    
    def CompareID(self):
       
        self.reid_array_ctr = 0
        
        for track in self.showtracker:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            find = False
            for i in self.reid_array:
                if i.replyID == track.track_id:
                    i.bbox = bbox
                    i.nowInFrame = True
                    i.disappearTime = 0
                    find = True
                    for j in self.lossarray:
                        if track.track_id == j['ID']:
                            del(j)
                    break
                
            if not(find):
                find = False
                for i in self.lossarray:
                    if track.track_id == i['ID']:
                        i['safeTime'] += 1
                        i['onfame'] = True
                        find = True
                        break
                if not find:
                    self.lossarray.append({'ID' : track.track_id, 'bbox' : bbox, 'safeTime' : 0, 'onfame' : True})
                
            self.reid_array_ctr += 1
        for i in self.lossarray:
            if not i['onfame']:
               i['safeTime'] -= 1 
        self.AddID()
        self.CompareLoseID()
    
    def AddID(self):
        while self.reid_array_ctr > len(self.reid_array) and self.lossarray != []:
            self.reid_array.append(ReIDunit(self.lossarray[0]['ID'], self.lossarray[0]['bbox']))
            del(self.lossarray[0])
    
    def CompareLoseID(self):
        for loss in self.lossarray:
            if loss['safeTime'] < self.ID_reply_frame: ###############
                if loss['safeTime'] < -self.ID_delete_frame:
                    self.lossarray.remove(loss)
                continue

            n = float("inf")
            n = 200
            s = -1
            for reid in self.reid_array:
                if reid.nowInFrame == False:
                    
                    #print([loss['bbox']])
                    pointi = self.CenterOfbbox(loss['bbox'])
                    if reid.bbox == []:
                        continue
                    pointj = self.CenterOfbbox(reid.bbox)
                    d = self.Distance(pointi, pointj)# + reid.disappearTime * 1000
                    if d < n :
                        n = d;
                        s = reid;
            if s != -1:
                #print(11111,loss['ID'], loss['bbox'])
                s.replyID = loss['ID']
                s.bbox = loss['bbox']
                s.nowInFrame = True
                s.disappearTime = 0
                del(loss)
            else:
                self.AddLossID(loss)
                

    def CenterOfbbox(self, bbox):
        return [(bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2]
    
    def Distance(self, point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
    
    def AddLossID(self, track):
        self.reid_array.append(ReIDunit(track['ID'], track['bbox']));
        return track['ID']
    
    def Draw(self, frame, results, Number_of_frames):
        #for reid in self.reid_array :
            #print(reid.replyID, reid.showID, reid.bbox)
        for track in self.showtracker:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            
            ID = 0
            for reid in self.reid_array :
                
                if reid.replyID == track.track_id:
                    ID = reid.showID

            if self.Have_deep_sort_bbox:
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)#deep sort 
            if self.Have_original_ID:
                cv2.putText(frame, "{} {}".format(str(ID),str(track.track_id)),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
            else:
                cv2.putText(frame, "{}".format(str(ID)),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
            
            if(ID == 0):
                print(self.lossarray)
                continue
                #ID = self.AddLossID(track)
            
            #Draw mask
            boxes = results['rois']
            score = results['scores'] #
            N = boxes.shape[0]
            for i in range(N):
    
            # Bounding box
                if not np.any(boxes[i]):
                    # Skip this instance. Has no bbox. Likely lost in image cropping.
                    continue
                y1, x1, y2, x2 = boxes[i]
                
                
                iw = min(x2, bbox[2]) - max(x1, bbox[0])
                A_area =  B_area = IoU = 0
                if iw > 0:
                   ih = min(y2, bbox[3]) - max(y1, bbox[1])  
                   if ih > 0:
                        A_area = (y2 - y1) * (x2 - x1)
                        B_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        Aiou = iw * ih / float(A_area)
                        Biou = iw * ih / float(B_area)
                        IoU = max(Aiou, Biou) 
                if IoU > 0.8:
                    if self.Have_maskrcnn_bbox:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),(255,0,255), 2)#mask rcnn
                    #cv2.putText(frame, str(score[i]),(int(x1), int(y1)),0, 5e-3 * 200, (255,0,255),2) #
                    
                    mask = results['masks'][:, :, i]
                    
                    if self.Have_mask:
                        brightness = 1.0
                        hsv = [(int(ID) / 20, 1, brightness)]
                        color = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))[0]
                        #color = (int((ID * 50) % 255),int((ID * 80) % 255),int((ID * 70) % 255))
                        
                        alpha = 0.5
                        for c in range(3):
                            frame[:, :, c] = np.where(mask == 1,
                                                      frame[:, :, c] *
                                                      (1 - alpha) + alpha * color[c] * 255,
                                                      frame[:, :, c]) 
                        
                    pixelsum = np.sum(mask == 1)
                    mask = np.where(mask == True,255,0)
                    mask = np.array(mask, dtype=np.uint8)
                    ret, thresh = cv2.threshold(mask, 127, 255, 0)
                    cnts = contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    
                    cnts = imutils.grab_contours(cnts)
                    
                    if len(cnts) == 1:
                        c=cnts[0]
                               
                        box = cv2.minAreaRect(c)
                        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                        box = np.array(box, dtype="int")
                        box = perspective.order_points(box)
                        if self.Have_fit_bbox:
                            cv2.drawContours(frame, [box.astype("int")], -1, color, 2)
                        (tl, tr, br, bl) = box 
                        
                        #長軸
                        short = ((tr[0] - tl[0])**2+(tr[1] - tl[1])**2)**0.5
                        long = ((tr[0] - br[0])**2+(tr[1] - br[1])**2)**0.5
                        
                        Aspect_ratio = (((tr[0] - tl[0])**2+(tr[1] - tl[1])**2)**0.5)/ \
                        (((tr[0] - br[0])**2+(tr[1] - br[1])**2)**0.5)
                        Aspect_ratio = Aspect_ratio if Aspect_ratio < 1 else 1 / Aspect_ratio
                        
                        point = ((int)((bbox[2] + bbox[0])/2), (int)((bbox[3] + bbox[1])/2))
                        
                         
                        
                        self.file.write("{} {} {} {} {} {} {} {} {}\n".format(str(Number_of_frames), str(ID), str(Aspect_ratio), str(pixelsum), str(point[0]), str(point[1]), str(score[i]), str(short), str(long)))
                    
                    break

from imutils.perspective import four_point_transform

def Correction(src):
    width  = src.shape[1]
    height = src.shape[0]
    
    four_points = [[60,0],[width - 40, 40],[0 , height],[width, height - 40]]
    src = four_point_transform(src, np.array(four_points))
    
    distCoeff = np.zeros((4,1),np.float64)
    
      # TODO: add your coefficients here!
    k1 = -2.8E-9; # negative to remove barrel distortion
    k2 = -9.8E-10;
    p1 = -7.4E-6;
    p2 = -8.4E-12;
    
    distCoeff[0,0] = k1;
    distCoeff[1,0] = k2;
    distCoeff[2,0] = p1;
    distCoeff[3,0] = p2;
    # assume unit matrix for camera
    cam = np.eye(3,dtype=np.float32)
    
    cam[0,2] = width/2.0  + 40 # define center x
    cam[1,2] = height/2.0 + 50 # define center y
    cam[0,0] = 10.        # define focal length x
    cam[1,1] = 10.        # define focal length y
    
      # here the undistortion will be computed
    src = cv2.undistort(src,cam,distCoeff)
    
    return src


def main():
    # Definition of the parameters
    video_capture = cv2.VideoCapture('/home/lab/Documents/20200412.avi')
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        #list_file = open('detection.txt', 'w')
        frame_index = -1
      
    reID = ReIDcontroler()
    
    Number_of_frames = 0
    
    fps = 0.0
    while True:
        ret, frame = video_capture.read()  # frame shape 1920*1080*3
        if ret != True:
            break
        frame = Correction(frame)
        frame = frame[...,::-1]
        t1 = time.time()
        results = model.detect([frame], verbose=0)
        
        r = copy.deepcopy(results[0])
        frame = frame[...,::-1]
       
        """
        length = int(frame.shape[0])
        width = int(frame.shape[1])

        dw = 1./width
        dh = 1./length
        """
        x = r['rois'][:,1]/1.0
        y = r['rois'][:,0]/1.0
        w = r['rois'][:,3] - r['rois'][:,1]
        h = r['rois'][:,2] - r['rois'][:,0]
        
        r['rois'][:,0] = x
        r['rois'][:,2] = w
        r['rois'][:,1] = y
        r['rois'][:,3] = h

        # print(x,y,w,h)
        

        features = encoder(frame,r['rois'])
        # score to 1.0 here).
        detections = [Detection(r['rois'], 1.0, feature) for r['rois'], feature in zip(r['rois'], features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        print(frame.shape)
        
        
        """track_results = []
        tttt = tracker
        dddd = detections
        for track in tttt.tracks:
            bbox = tttt.to_tlbr()
            x = int(bbox[0])
            y = int(bbox[3])
           """ 
        #id回復
        reID.showtracker = tracker.tracks
        
        reID.RomoveClose(tracker.tracks, frame)
                    
        reID.DeleteOldID()
        
        reID.CompareID()
        
        reID.Draw(frame, results[0], Number_of_frames)
        cv2.rectangle(frame, (0,0), (200,40),(0,0,0), -1) 
        cv2.putText(frame, "{}".format(str(Number_of_frames)),(0, 30),0, 5e-3 * 200, (255,255,255),2)
        
        Number_of_frames += 1
        
        cv2.imshow('', frame)
        cv2.waitKey(10)
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
        print("fps= %f"%(fps))
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
       # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
    cv2.destroyAllWindows()        

if __name__ == '__main__':
    main()
