import os

class Config:
    L = 32#one dimensional length of image 
    f = 8 #focus of one tile 
    gridN = int(L/f) #number of grids to split the input image into     
    c = 4 #context 
    
    
    """More than 1 box is not implemented - #TODO""" 
    boxN = 1 #number of boxes for each grid square
    
    
    #image generation
    min_objects = 0
    max_objects = 5
    filter = 0 #applies a gaussian filter with this number as sigma 
    noise = 0.2 #background noise of the generated image
    #source generation
    min_l = 4
    max_l = 10
    min_w = 1
    max_w = 4
    
    """
    default: d_low = 1, d_high = 6, , r_flip = 0.5
    to get straight vertical lines, set d_low = max_l, d_high = max_l + 1, r_flip = 1
    """
    #d represents the diagonal ration of the sources. i.e d=1 means a 1 to 1 diagonal source
    d_low = 1 
    d_high = 6
    r_flip = 0.5 #percentage to flip 
    
    #learning parameters 
    dataN = 2500 #size of data set to generate
    test_percent = 0.8 #split the data into train and test sets
    lr = 0.001
    batch_size = 100 
    keep_prob = 0.75 #dropout probability 
    
    chkpoint = 10
    
    #yolo parameters 
    filter_threshold = 0.5
    mixing_iou = 0.0 
    
    save_path = 'saved_sources'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = save_path + '/'