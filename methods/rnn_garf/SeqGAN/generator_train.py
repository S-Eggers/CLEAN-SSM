from methods.rnn_garf.SeqGAN.models import GeneratorPretraining, Generator
from methods.rnn_garf.SeqGAN.utils import GeneratorPretrainingGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import os
import numpy as np
import tensorflow as tf
sess = tf.compat.v1.Session()
import keras.backend as K
K.set_session(sess)


class GeneratorTrainer(object):
    '''
    Manage training
    '''
    def __init__(self, order, B, T, g_E, g_H, d_E, d_H, d_dropout, generate_samples,path_pos, path_neg, path_rules, g_lr=1e-3, d_lr=1e-3, n_sample=16,  init_eps=0.1, remove_amount_of_error_tuples=0):
        self.B, self.T = B, T               #batch size，max_length
        self.order=order
        self.g_E, self.g_H = g_E, g_H
        self.d_E, self.d_H = d_E, d_H
        self.d_dropout = d_dropout
        self.generate_samples = generate_samples
        self.g_lr, self.d_lr = g_lr, d_lr
        self.eps = init_eps
        self.init_eps = init_eps        #Exploration rate ϵ. i.e., the strategy is to select the current maximum value action with probability 1-ϵ and to select the new action at random with probability ϵ
        self.top = os.getcwd()          #The os.getcwd() method is used to return the current working directory
        self.path_pos = path_pos        #Address where the original data is located
        self.path_neg = path_neg        #Address where data is generated
        self.path_rules = path_rules        
        self.g_data = GeneratorPretrainingGenerator(self.path_pos, order=order, B=B, T=T, min_count=1, remove_amount_of_error_tuples=remove_amount_of_error_tuples) # next method produces x, y_true data; both are the same data, e.g. [BOS, 8, 10, 6, 3, EOS], [8, 10, 6, 3, EOS]
        self.V = self.g_data.V          #Total vocabulary in the data

        self.generator_pre = GeneratorPretraining(self.V, g_E, g_H)         #A 4-layer neural network input-embedding-lstm-dense
        self.generator = Generator(sess, B, self.V, g_E, g_H, g_lr)
        self.rule={}

    def pre_train(self, g_epochs=3, d_epochs=1, g_pre_path=None ,d_pre_path=None, g_lr=1e-3, d_lr=1e-3):        #The actual parameters are given by the config
        self.pre_train_generator(g_epochs=g_epochs, g_pre_path=g_pre_path, lr=g_lr)

    def pre_train_generator(self, g_epochs=3, g_pre_path=None, lr=1e-3):
        print("Pre-training generator")
        if g_pre_path is None:
            self.g_pre_path = os.path.join(self.top, 'data', 'save', 'generator_pre.hdf5')  #D:\PycharmProjects\Garf-master\data\save\generator_pre.hdf5
        else:
            self.g_pre_path = g_pre_path

        g_adam = Adam(lr)
        self.generator_pre.compile(g_adam, 'categorical_crossentropy')  #Training is performed, the optimizer is Adam, and the loss function is a categorical cross-entropy function for multi-classification
        print('Generator training')
        self.generator_pre.summary()            #model.summary() in keras is used to output the status of the parameters of each layer of the model
        # print("++++++++++++++++++++")
        self.generator_pre.fit_generator(       #The return value is a History object. Its History.history property is a record of the training loss and evaluation values for successive epochs, as well as the validation set loss and evaluation values
            self.g_data,                        #This should be an instance of a generator or Sequence (keras.utils.Sequence) object
            steps_per_epoch=None,
            epochs=g_epochs,
            callbacks=[
                EarlyStopping(monitor="loss", patience=3, min_delta=0.01)
            ])
        self.generator_pre.save_weights(self.g_pre_path)    #Save the weights to generator_pre.hdf5
        self.reflect_pre_train()                #Mapping layer layer weights to agent

    def load_pre_train(self, g_pre_path, d_pre_path):
        self.generator_pre.load_weights(g_pre_path)
        self.reflect_pre_train()

    def load_pre_train_g(self, g_pre_path):
        self.generator_pre.load_weights(g_pre_path)
        self.reflect_pre_train()

    def load_pre_train_d(self, d_pre_path):
        self.discriminator.load_weights(d_pre_path)

    def reflect_pre_train(self):                        #Mapping layer layer weights to agent
        i = 0
        for layer in self.generator_pre.layers:
            if len(layer.get_weights()) != 0:           #If the weight of this layer is not 0
                w = layer.get_weights()
                self.generator.layers[i].set_weights(w)       #then the weight of the corresponding layer in agent is set to w
                i += 1

    def save(self, g_path, d_path):
        self.generator.save(g_path)

    def load(self, g_path, d_path):
        self.generator.load(g_path)

    def generate_rules(self, file_name, generate_samples):
        #self.B=1
        path_rules = os.path.join(self.top, 'data', 'save', file_name)
        print(path_rules)

        self.generator.generate_rules(
            8, self.g_data, generate_samples, path_rules)

    # def predict_rules(self):
    #     # self.agent.generator.predict_rules()
    #     result=self.agent.generator.multipredict_rules_argmax(reason=['10005','BOAZ','AL','36251'])
    #     result_ = self.agent.generator.multipredict_rules_argmax(reason=['10005', 'BOAZ', 'AL', '36251'])


    def train_rules(self,rule_len,path):
        path_rules = os.path.join(self.top, 'data', 'save', path)
        self.generator.train_rules(rule_len,path_rules)

    def filter(self,path):
        self.generator.filter(path)

    def repair(self,path):
        self.generator.repair(1,path,self.order)#3

    # def repair_SeqGAN(self):
    #     self.agent.generator.repair_SeqGAN()