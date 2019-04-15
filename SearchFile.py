import sys 
import numpy as np

from Config import Config
from Net import Network
from DataHandle import *
config = Config()

def main(file_name):
    
    try:
        data = np.load('Data/' + str(file_name) + '.npy')
        data = np.clip(data,0,5)/5
    except:
        print("Invalid File Name")
        return
    
    net = Network()
    net.load()
 
    f = open('Detections_' + str(file_name) + '.txt', 'a')
    
    for x in np.arange(0, np.shape(data)[1]-config.L,config.L):
        c = data[:, x:x+config.L]

        buffer = ImgBuffer()
        for n in range(np.shape(c)[0]-config.L):
    
            currIm = np.reshape(c[n:n+config.L,:], [1, config.L, config.L])
            pred = net.predict(currIm) 

            boxes = process_pred(pred)

            buffer.update_timer()
            buffer.process_new(boxes, n, x)
            buffer.process_existing()

        buffer.clear_buffer()
        np.savetxt(f, buffer.final_array, delimiter=',', newline='\n')
    f.close()    
        
if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Need one argument for file name")