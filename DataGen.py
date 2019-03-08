import numpy as np
from scipy.ndimage.filters import gaussian_filter

from DataHandle import *

from Config import Config
config = Config()

def make_data(N): #Makes N images to be used to training the network
    image = np.zeros((N, config.L, config.L)) #raw data
    labels = np.zeros((N, config.max_objects, 4)) #true value associated with the data
    
    
    for i in range(N):  
        image[i] = image[i] + np.random.normal(0, config.noise, image[i].shape)#create some background noise        
        image[i][image[i] < 0] = 0
        
        for o in range(np.random.randint(config.min_objects,config.max_objects+1)): #add the sources
            tmp = np.zeros((config.L, config.L)) 
            
            l = np.random.randint(config.min_l, config.max_l) #length of the source
            w = np.random.randint(config.min_w, config.max_w) #width  ' ' ' 
            
            diag_ratio = np.random.randint(config.d_low, config.d_high) ###i.e a one to one ratio means perfectly diagonal
            
            #generate coordinates of the lowest left corner 
            x = np.random.randint(0,config.L-l) 
            y = np.random.randint(0,config.L-w-(l//diag_ratio))

            #add the rest of the source
            for j in range(l):
                for k in range(w):
                    tmp[x+j, y+k+(j//diag_ratio)] = 1

            #create the label 
            labels[i,o] = [x,y,l,w+(l//diag_ratio)-np.equal(l%diag_ratio,0)]
        
            
            r = np.random.rand()
            if r < config.r_flip: #flip half of the sources
                im = image[i]
                tmp = np.rot90(tmp)
                labels[i,o] = [config.L-y , x, -(w+(l//diag_ratio)-np.equal(l%diag_ratio,0)), l]
            
            #TO-DO - not sure what this does tbh
            if labels[i,o][2] < 0: 
                labels[i,o][0] += labels[i,o][2]
                labels[i,o][2] *= -1
                
            image[i] += tmp
        image[i] = gaussian_filter(image[i],config.filter)
    return image, labels 

#need to separate the labels such that sources are split between the grid squares 
def get_True(labels):
    true = np.zeros([4,4,5])
    for i in range(np.shape(labels)[0]):
        if(labels[i][2] != 0): 
            cx = labels[i][0] + labels[i][2]/2.0
            cy = labels[i][1] + labels[i][3]/2.0
            ind1 = int(cx//config.f)
            ind2 = int(cy//config.f)
            
            if true[ind1, ind2, 0] == 0: #if the selected tile is empty, add the source
                true[ind1,ind2,0] = 1
                true[ind1,ind2,1] = (cx - config.f*(cx//config.f))/(config.f)
                true[ind1,ind2,2] = (cy - config.f*(cy//config.f))/(config.f)
                true[ind1,ind2,3] = labels[i][2]/config.L
                true[ind1,ind2,4] = labels[i][3]/config.L
            else: #if there is already a source in the tile, we merge the bounding box to include both 
                currx, curry, currw, currh = true[ind1, ind2, 1:5]
                currx = currx*config.f + ind1*config.f 
                curry = curry*config.f + ind2*config.f
                currw, currh = currw*config.L, currh*config.L
                
                newbox = merge_boxes([currx, curry, currw, currh],[cx,cy, labels[i][2], labels[i][3]])
                
                true[ind1,ind2,1] = (newbox[0] - config.f*(newbox[0]//config.f))/(config.f)
                true[ind1,ind2,2] = (newbox[1] - config.f*(newbox[1]//config.f))/(config.f)
                true[ind1,ind2,3] = newbox[2]/config.L
                true[ind1,ind2,4] = newbox[3]/config.L

    return np.reshape(true, [-1, config.gridN**2, config.boxN*5])   
       

def gen_TestTrain(N = config.dataN):
    data, labels = make_data(N)
    true_labels = []
    for i in labels:
        true_labels.append(get_True(i))
    true_labels = np.reshape(np.array(true_labels), [-1, config.gridN**2, 5*config.boxN])
    
    n = int(N*config.test_percent)
    train_data = data[:n]
    test_data = data[n:]
    train_labels = true_labels[:n]
    test_labels = true_labels[n:]
    
    return train_data, test_data, train_labels, test_labels

def gen_VideoTest(l = 10): #generates one long image to use for video
    imgs, _ = make_data(l)
    return np.concatenate(imgs) 