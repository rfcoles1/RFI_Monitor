import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from Helper import *
from Config import Config 


class Network():
    def __init__(self):
        self.config = Config()
        
        """EDNN implementation author: Kyle Mills""" 
        helper = EDNN_helper(L=self.config.L,f=self.config.f,c=self.config.c)

        self.x = tf.placeholder(tf.float32, [None, self.config.L, self.config.L])
        self.y = tf.placeholder(tf.float32, [None, self.config.gridN**2, 5*self.config.boxN])
        self.keep_prob = tf.placeholder(tf.float32)
        
        tiles = tf.map_fn(helper.ednn_split, self.x, back_prop=False)
        tilesT = tf.transpose(tiles, perm=[1,0,2,3,4])
        tilesF = tf.reverse(tilesT, axis = [0])
        output = tf.map_fn(self.NN, tilesF, back_prop =True)
        self.predicted = tf.transpose(output, perm=[1,0,2])
        
        #predicted = tf.reshape(output, [-1, 16, 5])
        #print(predicted.get_shape().as_list())

        #loss defined by position
        tmp1 = tf.square(self.predicted[:,:,1] - self.y[:,:,1])
        tmp2 = tf.square(self.predicted[:,:,2] - self.y[:,:,2])
        self.loss = tf.reduce_sum(tf.math.multiply(self.y[:,:,0], tmp1+tmp2))

        #loss defined by dimensions
        tmp1 = tf.square(self.predicted[:,:,3] - self.y[:,:,3])
        tmp2 = tf.square(self.predicted[:,:,4] - self.y[:,:,4])
        self.loss += tf.reduce_sum(tf.math.multiply(self.y[:,:,0], tmp1+tmp2))
        
        """
        tmp1 = tf.square(tf.where(tf.is_nan(tf.sqrt(self.predicted[:,:,3])), tf.zeros_like(self.predicted[:,:,3]), self.predicted[:,:,3]) - tf.sqrt(self.y[:,:,3]))
        tmp2 = tf.square(tf.where(tf.is_nan(tf.sqrt(self.predicted[:,:,4])), tf.zeros_like(self.predicted[:,:,4]), self.predicted[:,:,4]) - tf.sqrt(self.y[:,:,4]))
        self.loss += tf.reduce_sum(tf.math.multiply(self.y[:,:,0], tmp1+tmp2))
        """
        self.loss += tf.reduce_sum(tf.square(self.y[:,:,0] - self.predicted[:,:,0]))
        
        self.op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(self.loss)
        
        #scores,boxes,boxes2,mask = yolo_eval(predicted)
        self.sess = tf.Session() 
        self.reset()
    
    def NN(self, _in): #predicts the middle of the source and the dimensions of the box
        inp = tf.reshape(_in, (-1, (self.config.f + 2*self.config.c)**2))
        lay1 = tf.contrib.layers.fully_connected(inp, 128)
        drop1 = tf.nn.dropout(lay1, self.keep_prob)
        lay2= tf.contrib.layers.fully_connected(drop1, 128)
        drop2 = tf.nn.dropout(lay2, self.keep_prob)
        lay3 = tf.contrib.layers.fully_connected(drop2, 128, tf.nn.relu)
        drop3 = tf.nn.dropout(lay3, self.keep_prob)
        out = tf.contrib.layers.fully_connected(drop3, 5, \
                            activation_fn = tf.nn.relu)
        return out
    
    def CNN(self, _in): #predicts the middle of the source and the dimensions of the box
        inp = tf.reshape(_in, (-1,self.config.f + 2*self.config.c,self.config.f + 2*self.config.c,1))
        conv1 = tf.contrib.layers.conv2d(inp,128, 
                                kernel_size = [3,3],
                                padding = 'same',
                                activation_fn = tf.nn.relu)
        pool1 = tf.contrib.layers.max_pool2d(conv1,
                                kernel_size = [2,2],
                                stride = 4)

        conv2 = tf.contrib.layers.conv2d(pool1,256,
                                kernel_size = [3,3],
                                padding = 'same',
                                activation_fn = tf.nn.relu)
        pool2 = tf.contrib.layers.max_pool2d(conv2,
                                kernel_size = [2,2],
                                stride = 4)
        pool2_flat = tf.reshape(pool2, [-1, 256])
    
        fc1 = tf.contrib.layers.fully_connected(pool2_flat, 1024,
                                activation_fn = tf.nn.relu)
        fc2 = tf.contrib.layers.fully_connected(fc1, 128,
                                activation_fn = tf.nn.relu)
        fc3 = tf.contrib.layers.fully_connected(fc2, 5,
                                activation_fn = tf.nn.relu)
        return fc3
    
    def reset(self):
        self.curr_epoch = 0
        self.loss_history = []
        self.sess.run(tf.global_variables_initializer()) 
        
    def train(self, inp, out, num_epoch): #net trains on input data, using output for true values
        num_batches = int(inp.shape[0]/self.config.batch_size)
        inp, out = shuffle(inp,out)#ensures that each batch is representative of the data 
        
        for epoch in range(num_epoch): #how many times the net sees the entire data set 
            avg_loss = 0
            for batch in range(num_batches): #batch the data
                l, _ = self.sess.run([self.loss, self.op], feed_dict = {
                    self.x: inp[batch*self.config.batch_size:(batch+1)*self.config.batch_size],
                    self.y: out[batch*self.config.batch_size:(batch+1)*self.config.batch_size],
                    self.keep_prob: self.config.keep_prob})
                avg_loss += l
            if epoch%self.config.chkpoint == 0:
                print("Step " + str(self.curr_epoch) + ", Loss = " + "{:.4f}".format(avg_loss/num_batches))
            self.loss_history.append(avg_loss)
            self.curr_epoch += 1
    
    def test(self, inp, out): #obtain loss for a test dataset 
        l = self.sess.run([self.loss], feed_dict = {
            self.x: inp, self.y: out, self.keep_prob: 1})      
        return l 

    def predict(self, inp): #gain prediction with no training 
        preds = self.sess.run([self.predicted], feed_dict = {
            self.x: inp, self.keep_prob: 1})
        return preds[0]
    
    def save(self): 
        saver = tf.train.Saver()
        saver.save(self.sess, save_path='./checkpts/test_model')
        
    def load(self):
        self.reset()
        saver = tf.train.Saver()
        saver.restore(self.sess, './checkpts/test_model')
    

        
                
