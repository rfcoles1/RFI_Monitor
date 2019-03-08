import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.gridspec as gs

from DataGen import *
from DataHandle import *

def plot_25_ims(outlines = False): #plots a 5x5 grid of images as examples
    data, labels = make_data(25)
    fig, axs = plt.subplots(5,5, figsize=(10,10))
    for i in range(5):
        for j in range(5):
            axs[i][j].imshow(data[5*i+j].T, origin = 'lower', vmax = 1, vmin = 0)
            axs[i][j].axis("off")
            if outlines == True:
                o = 0
                while(labels[5*i+j,o][2] != 0):
                    axs[i][j].add_patch(patch.Rectangle((labels[5*i+j,o][0], labels[5*i+j,o][1]),\
                        labels[5*i+j,o][2], labels[5*i+j,o][3], ec='w', fc='none'))
                    o += 1

                    
def plot_True_Example(): #plots one image with the true sources highlighted
    img, label = make_data(1)
    true = get_True(label[0])
    print(true)
    fig = plt.imshow(img[0], vmax = 1, vmin = 0)
    ax = plt.gca()
    plt.xticks(np.arange(config.f, config.L, config.f))
    plt.yticks(np.arange(config.f, config.L, config.f))
    plt.axis("on")
    plt.grid(True)
    
    boxes = process_pred(true)
    for z in range(len(boxes)):
        cx, cy, w, h = boxes[z]
        ax.add_patch(patch.Circle((cy,cx), 0.5, ec = 'r', fc = 'r'))
        ax.add_patch(patch.Rectangle((cy-h/2, cx-w/2),\
            h, w, ec='r', fc='none'))  

    
def plot_Pred(img, label, pred, showTrue = True):
    if showTrue == True:
        true = get_True(label[0])
    fig = plt.imshow(img[0], vmax = 1, vmin = 0)
    ax = plt.gca()
    plt.xticks(np.arange(config.f, config.L, config.f))
    plt.yticks(np.arange(config.f, config.L, config.f))
    plt.axis("on")
    plt.grid(True)

    for z in range(config.gridN**2):
        i = z//config.gridN
        j = z%config.gridN
        if showTrue == True:
            if true[0][z][0] == 1:
                truex = true[0][z][1]*config.f + i*config.f
                truey = true[0][z][2]*config.f + j*config.f
                truew = true[0][z][3]*config.L
                trueh = true[0][z][4]*config.L
                ax.add_patch(patch.Circle((truey,truex), 0.5, ec = 'r', fc = 'r'))
                ax.add_patch(patch.Rectangle((truey-trueh/2, truex-truew/2),\
                            trueh, truew, ec='r', fc='none'))    
                      
        if pred[0][z][0] > config.filter_threshold:
            w = pred[0][z][3]*config.L
            h = pred[0][z][4]*config.L
            cx = pred[0][z][1]*config.f + i*config.f
            cy = pred[0][z][2]*config.f + j*config.f
            ax.add_patch(patch.Circle((cy,cx), 0.5, ec = 'w', fc = 'w'))
            ax.add_patch(patch.Rectangle((cy-h/2, cx-w/2),\
                            h, w, ec='w', fc='none'))             

def plot_Boxes(img, boxes):
    plt.cla()
    fig = plt.imshow(img[0], vmax = 1, vmin = 0)
    ax = plt.gca()
    plt.xticks(np.arange(config.f, config.L, config.f))
    plt.yticks(np.arange(config.f, config.L, config.f))
    plt.axis("on")
    plt.grid(True)  

    for z in range(len(boxes)):
        cx, cy, w, h = boxes[z]
        ax.add_patch(patch.Circle((cy,cx), 0.5, ec = 'w', fc = 'w'))
        ax.add_patch(patch.Rectangle((cy-h/2, cx-w/2),\
            h, w, ec='r', fc='none'))  
    
def plot_Extracted(boxes, imgs):
    for z in range(len(imgs)):
        fig, ax1 = plt.subplots()
        ax1.imshow(imgs[z], vmin = 0, vmax = 1)
        plt.title("Center = (" + str(np.round((boxes[z][0]+boxes[z][2])/2, 2)) \
            + ", " + str(np.round((boxes[z][1]+boxes[z][3])/2,2)) + ")")
        plt.show()
       
