import tensorflow as tf

class DCGAN():
    
    def __init__(self):
        
        #======================
        # Setup Hyperparameters
        #======================
        
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = (100,)
        # Generator:
        self.gen_learning_rate = 0.0002
        self.gen_clipvalue     = 0.1
        self.gen_keep_prob     = 0.8
        self.gen_momentum      = 0.95
        gen_opt = Adam(self.gen_learning_rate, self.gen_clipvalue)
        # Discriminator:
        self.dis_learning_rate = 0.000005
        self.dis_clipvalue     = 0.1
        self.dis_keep_prob     = 0.8
        self.dis_momentum      = 0.95
        dis_opt = Adam(self.dis_learning_rate, self.dis_clipvalue)
    
    def build_generator(self, reuse=False, training=True):
        
        def gen_layer(input_, filters, kernel_size, strides):
            '''Layer used to build the generator'''
            x = tf.nn.conv2d_transpose(value=input_, output_shape=filters,
                                       strides=strides, padding='SAME')
            x = tf.nn.relu(x)
            x = tf.nn.dropout(x, keep_prob=self.gen_keep_prob)
            x = tf.nn.batch_normalization(x)
        
        
        with tf.variable_scope('generator', reuse=reuse):
            # First fully connected layer
            x1 = tf.layers.dense(z, 4*4*512)
            # Reshape it to start the convolutional stack
            x1 = tf.reshape(x1, (-1, 4, 4, 512))
        
        
            
            

            
        
