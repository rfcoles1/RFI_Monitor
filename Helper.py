import numpy as np
import tensorflow as tf

class EDNN_helper(object):
    """author: Kyle Mills"""
    def __init__(self,L,f,c):
        assert f <= L/2, "Focus must be less that half the image size to use this implementation."
        assert (f + 2*c) <= L, "Total tile size (f+2c) is larger than input image."
        self.l = L
        self.f = f
        self.c = c
        
    def __roll(self, in_, num, axis):
        D = tf.transpose(in_, perm=[axis,1-axis]) #if axis=1, transpose first
        D = tf.concat([D[num:,:], D[0:num, :]], axis=0)
        return tf.transpose(D, perm=[axis, 1-axis]) #if axis=1, transpose back

    def __slice(self, in_, x1, y1, w, h):
        return in_[x1:x1+w, y1:y1+h]

    def ednn_split(self,in_):
        tiles = []
        for iTile in range(int(self.l/self.f)):
            for jTile in range(int(self.l/self.f)):
                #calculate the indices of the centre of this tile (i.e. the centre of the focus region)
                cot = (iTile*self.f + self.f//2, jTile*self.f + self.f//2)
                foc_centered = in_
                #shift picture, wrapping the image around, 
                #so that focus is centered in the middle of the image
                foc_centered = self.__roll(foc_centered, int(self.l//2-cot[0]),0)
                foc_centered = self.__roll(foc_centered, int(self.l//2-cot[1]),1)
                #Finally slice away the excess image that we don't want to appear in this tile
                final = self.__slice(foc_centered, int(self.l//2-self.f//2-self.c),\
                    int(self.l//2-self.f//2-self.c), 2*self.c+self.f, 2*self.c+self.f)
                tiles.append(final)
        return tf.expand_dims(tiles, axis=3)
        
###Not Currently in use        
class YOLO_helper(object):
    """author: Pulkit Sharma"""
    def _yolo_filter_boxes(self, box_confidence, boxes, threshold = .6):
        filtering_mask = box_confidence>threshold
        boxes = tf.boolean_mask(boxes, filtering_mask)
        scores = tf.boolean_mask(box_confidence, filtering_mask)
        return scores, boxes
    
    
    def _yolo_corners(self, boxes):
        box_xy = boxes[:,0:2]
        box_wh = boxes[:,3:4]
        box_mins = box_xy - box_wh/2.0
        box_maxes = box_xy + box_wh/2.0
        return tf.concat([box_mins[:, 1:2], box_mins[:, 0:1],\
                    box_maxes[:, 1:2], box_maxes[:, 0:1]], axis = -1)


    def _yolo_NMS(self, scores, boxes, iou_threshold = 0.5):
        max_box_t = tf.Variable(5, dtype = 'int32')
        nms_ind = tf.image.non_max_suppression(boxes, scores, max_box_t, 0.5)
        scores = tf.gather(scores, nms_ind)
        boxes = tf.gather(boxes, nms_ind)
        return scores, boxes

    def yolo_eval(self, outputs, score_threshold = .6, iou_threshold = .5):
        box_confidence = outputs[:,:,0]
        boxes = outputs[:,:, 1:5]
        print('before filter')
        print(np.shape(box_confidence))
        print(np.shape(boxes))
        boxes1 = boxes
        scores, boxes = self._yolo_filter_boxes(box_confidence, boxes)
        print('after ftiler')
        print(np.shape(scores))
        print(np.shape(boxes))
        boxes = self._yolo_corners(boxes)
        scores, boxes = self._yolo_NMS(scores, boxes, iou_threshold)
        return scores, boxes