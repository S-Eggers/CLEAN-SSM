import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.utils import to_categorical
import tensorflow as tf
import pickle
import linecache
import sqlite3
import math
import tqdm


def GeneratorPretraining(V, E, H):
    '''
    Model for Generator pretraining. This model's weights should be shared with
        Generator.
    # Arguments:
        V: int, Vocabrary size
        E: int, Embedding size
        H: int, LSTM hidden size
    # Returns:
        generator_pretraining: keras Model
            input: word ids, shape = (B, T)
            output: word probability, shape = (B, T, V)
    '''
    # in comment, B means batch size, T means lengths of time steps.
    input = Input(shape=(None,), dtype='int32', name='Input') # (B, T)
    out = Embedding(V, E, mask_zero=True, name='Embedding')(input)  # (B, T, E)
                                                                    # Build a chain of functions between Layers, Embedding layer to change the output shape
    out = Bidirectional(LSTM(H, return_sequences=True, name='LSTM'))(out)  # (B, T, H)
                                                            # Constructing a chain of functions between Layers
    out = TimeDistributed(                                               # The TimeDistributed layer performs a Dense operation on each vector
        Dense(V, activation='softmax', name='DenseSoftmax'),             # Define a neural layer with V nodes, using softmax activation function
        name='TimeDenseSoftmax')(out)    # (B, T, V)
    generator_pretraining = Model(input, out)
    return generator_pretraining

class Generator():
    'Create Generator, which generate a next word.'

    # def rule_dict(self):


    def __init__(self, sess, B, V, E, H, lr=1e-3):
        '''
        # Arguments:
            B: int, Batch size
            V: int, Vocabrary size
            E: int, Embedding size
            H: int, LSTM hidden size
        # Optional Arguments:
            lr: float, learning rate, default is 0.001
        '''
        self.sess = sess
        self.B = B
        self.V = V
        self.E = E
        self.H = H
        self.lr = lr
        self._build_gragh()
        self.reset_rnn_state()


    def _build_gragh(self):
        state_in = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))      # Pass in the data, where None refers to the size of the batch size, which can be any number, and 1 refers to the size of the data
        h_in_forward = tf.compat.v1.placeholder(tf.float32, shape=(None, self.H))    
        c_in_forward = tf.compat.v1.placeholder(tf.float32, shape=(None, self.H))
        h_in_backward = tf.compat.v1.placeholder(tf.float32, shape=(None, self.H))    
        c_in_backward = tf.compat.v1.placeholder(tf.float32, shape=(None, self.H))     
        action = tf.compat.v1.placeholder(tf.float32, shape=(None, self.V))   # onehot (B, V)
        reward  = tf.compat.v1.placeholder(tf.float32, shape=(None, ))         # (B, ) the rewards of each batch

        self.layers = []

        embedding = Embedding(self.V, self.E, mask_zero=True, name='Embedding')     # The first layer is the Embedding layer
        out = embedding(state_in)                                                   # Input is (B,V), output (B,1,E)
        self.layers.append(embedding)

        lstm = Bidirectional(LSTM(self.H, return_state=True, name='LSTM'))                         
        out, next_h_forward, next_c_forward, next_h_backward, next_c_backward = lstm(out, initial_state=[h_in_forward, c_in_forward, h_in_backward, c_in_backward])
        self.layers.append(lstm)

        dense = Dense(self.V, activation='softmax', name='DenseSoftmax')            # Fully connected layer
        prob = dense(out)                                                           # Input is (B,H), output is (B,V)
        self.layers.append(dense)

        log_prob = tf.math.log(tf.reduce_mean(input_tensor=prob * action, axis=-1)) # (B, )         Take each row of data and multiply it with onehot's action, then take the logarithm of the average
        loss = - log_prob * reward
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
        minimize = optimizer.minimize(loss)
        #The following operations are for the overall training of lstm, while this paper requires a single step, with each update guided by rl, so the following operations are not performed
        #model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])  # Compilation Model
        #print(model.summary())                                                                   # Show model structure
        #model.fit(data, labels)  # starts training                                               # Fitting the network
        #loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)                         # Testing
        #classes = model.predict_classes(X_test, batch_size=32)
        #proba = model.predict_proba(X_test, batch_size=32)                                       # Use

        self.state_in = state_in
        self.h_in_forward = h_in_forward
        self.c_in_forward = c_in_forward
        self.h_in_backward = h_in_backward
        self.c_in_backward = c_in_backward
        self.action = action
        self.reward = reward
        self.prob = prob
        self.minimize = minimize
        self.next_h_forward = next_h_forward
        self.next_c_forward = next_c_forward
        self.next_h_backward = next_h_backward
        self.next_c_backward = next_c_backward
        self.loss = loss

        self.init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(self.init_op)

    def reset_rnn_state(self):
        self.h_fwd = np.zeros([self.B, self.H])
        self.c_fwd = np.zeros([self.B, self.H])
        self.h_bwd = np.zeros([self.B, self.H])
        self.c_bwd = np.zeros([self.B, self.H])

    def set_rnn_state(self, h_fwd, c_fwd, h_bwd, c_bwd):             #h、c: np.array, shape = (B,H)，ex.（32,64）
        self.h_fwd = h_fwd
        self.c_fwd = c_fwd
        self.h_bwd = h_bwd
        self.c_bwd = c_bwd

    def get_rnn_state(self):
        return self.h_fwd, self.c_fwd, self.h_bwd, self.c_bwd

    def predict(self, state, stateful=True):
        '''
        Predict next action(word) probability
        # Arguments:
            state: np.array, previous word ids, shape = (B, 1)
        # Optional Arguments:
            stateful: bool, default is True
                if True, update rnn_state(h, c) to Generator.h, Generator.c
                    and return prob.
                else, return prob, next_h, next_c without updating states.
        # Returns:
            prob: np.array, shape=(B, V)
        '''
        # state = state.reshape(-1, 1)
        feed_dict = {
            self.state_in: state,
            self.h_in_forward: self.h_fwd,
            self.c_in_forward: self.c_fwd,
            self.h_in_backward: self.h_bwd,
            self.c_in_backward: self.c_bwd
        }
        prob, next_h_fwd, next_c_fwd, next_h_bwd, next_c_bwd = self.sess.run(
            [self.prob, self.next_h_forward, self.next_c_forward, self.next_h_backward, self.next_c_backward],
            feed_dict)
        # print(prob.shape)
        # print(next_c.shape)
        # print(next_c)

        if stateful:
            self.h_fwd = next_h_fwd
            self.c_fwd = next_c_fwd
            self.h_bwd = next_h_bwd
            self.c_bwd = next_c_bwd
        return prob if stateful else (prob, next_h_fwd, next_c_fwd, next_h_bwd, next_c_bwd)

    def update(self, state, action, reward, h_fwd=None, c_fwd=None, h_bwd=None, c_bwd=None, stateful=True):
        if h_fwd is None:
            h_fwd = self.h_fwd
        if c_fwd is None:
            c_fwd = self.c_fwd
        if h_bwd is None:
            h_bwd = self.h_bwd
        if c_bwd is None:
            c_bwd = self.c_bwd
        state = state[:, -1].reshape(-1, 1)
        reward = reward.reshape(-1)
        feed_dict = {
            self.state_in: state,
            self.h_in_forward: h_fwd,
            self.c_in_forward: c_fwd,
            self.h_in_backward: h_bwd,
            self.c_in_backward: c_bwd,
            self.action: to_categorical(action, self.V),
            self.reward: reward
        }
        _, loss, next_h_fwd, next_c_fwd, next_h_bwd, next_c_bwd = self.sess.run(
            [self.minimize, self.loss, self.next_h_forward, self.next_c_forward, self.next_h_backward, self.next_c_backward],
            feed_dict)

        if stateful:
            self.h_fwd = next_h_fwd
            self.c_fwd = next_c_fwd
            self.h_bwd = next_h_bwd
            self.c_bwd = next_c_bwd
        return loss if stateful else (loss, next_h_fwd, next_c_fwd, next_h_bwd, next_c_bwd)

    def sampling_word(self, prob):

        action = np.zeros((self.B,), dtype=np.int32)
        for i in range(self.B):
            p = prob[i]                                        # p is an array of 1-dimensional V columns
            # print("p:",p)
            p /=p.sum()                                        # Total probability normalization
            action[i] = np.random.choice(self.V, p=p)          # Probabilistically choose a number between 0 and V according to the probability provided by p
        #     print("action[i]",action[i])
        # print("action",action)
        return action

    def sampling_sentence(self, T, BOS=1):                     # Generate sentences based on rnn

        self.reset_rnn_state()                                  # Parameters h,c state reset
        action = np.zeros([self.B, 1], dtype=np.int32)          # Generate a vertical array of size B, initially all 0
        action[:, 0] = BOS                                      # First full placement BOS
        actions = action
        # print(T)
        for _ in range(T):                                      # T is the maximum length of a sentence
            prob = self.predict(action)                         # Predicts the subsequent sequence based on the current sequence and returns the parameters
            # print("prob:",prob)
            action = self.sampling_word(prob).reshape(-1, 1)    # Sample words according to parameters and convert to 1 column
            # print("action:",action)
            # print(len(action))
            for i in range(len(action)):
                if (action[i][0] == 4):
                    action[i][0] = 3
                    # print("Convert none to unknown")
            # print("Post-transformation action:",action)
            actions = np.concatenate([actions, action], axis=-1)#The set of B sentences composed by the id of the word
            # print("actions:",actions)
        # Remove BOS
        actions = actions[:, 1:]        # Take all the data from the first column to the right side of the data, i.e. all the data on the right side except column 0
        # print(actions)
        self.reset_rnn_state()

        return actions

    def generate_samples(self, T, g_data, num, output_file):
        print("Generate_samples was executed")
        sentences=[]

        for _ in range(num // self.B + 1):
            actions = self.sampling_sentence(T)                 # Based on the neural network, the sentences are generated and returned as an array of the ids of the words

            actions_list = actions.tolist()                     # Change from array to list

            for sentence_id in actions_list:
                # print("sentence_id",sentence_id)
                sentence = [g_data.id2word[action] for action in sentence_id if action != 0 and action != 2]     # Reverse the id to word, this is a concise version of the above
                sentences.append(sentence)                      # Generators generate sentences
                # print(sentences)

        output_str = ''

        for i in range(num):
            # print(sentences[i])
            # if (sentences[i] is None):
            #     sentences[i]=""
            #sentence_str = ""
            #for word in sentences[i]:
            #    sentence_str += f"{word} "
            #output_str += sentence_str + '\n'
            output_str += ' '.join(sentences[i]) + '\n'
            # print(output_str)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_str)
        print("The generated sequence is written to",output_file)

    def sampling_rule(self,   T, BOS=1):                     # Generate rules based on rnn
        # print("The sampling_rule is executed")

        f = open('data/save/id2word.txt', 'r')
        id2word = eval(f.read())
        f.close()
        # print(id2word[684])
        f = open('data/save/word2id.txt', 'r')
        word2id = eval(f.read())
        # print(word2id)
        f.close()

        self.reset_rnn_state()
        action = np.zeros([self.B, 1], dtype=np.int32)          # into a vertical array of size B, initially all 0
        actions = np.empty([self.B, 0], dtype=np.int32)
        # print("actions",actions)
        action[:, 0] = BOS                                      # First full placement BOS
        for _ in range(T):                                      # T is the maximum length of a sentence

            prob = self.predict(action)                         # Predict the subsequent sequence based on the current sequence and return the parameters, which are input to the network in the shape of (B,1)

            action = self.sampling_word(prob).reshape(-1, 1)    # Sample words according to parameters and convert to 1 column
            # print("Predicted output at training time:", action,id2word[action[0][0]])
            action_=np.argmax(prob, axis=-1).reshape([-1, 1])
            # print("Predicted output at training_:", action_,id2word[action_[0][0]])
            # print("_________________")
            show2 = []
            for id in action:
                # print(id)
                word = id2word[id[0]]
                show2.append(word)
            # print("Prediction selection:",show2)

            show3 = np.array(show2).reshape(-1, 1)
            # print(show3)
            actions = np.concatenate([actions, show3], axis=-1) # Rule Growth
            # print("Current Rules:",actions)
        self.reset_rnn_state()               # Reset the state of lstm, it is important to delete the equivalent of the above words to continue to predict, rather than based on the current state prediction


        return actions

    def generate_rules(self, T, g_data, num, output_file):
        print("Generate_rules was executed")
        # print(output_file)
        rules=[]
        for _ in range(num // self.B + 1):
            actions = self.sampling_rule(T)                 # Generate sentences based on neural networks
            # print(actions.shape)                            # array, size (B,T)
            print(actions)
            actions_list = actions.tolist()                 # change from array to list, dimension remains the same, ex.(32,7)

            for rule_word in actions_list:                    # Cycle B times
                rule = rule_word
                # print(rule)
                rules.append(rule)                      # Generators generate sentences
                # print(rules)

        output_str = ''

        for i in range(num):
            # print(rules[i])
            for n in range(len(rules[i])):
                if (rules[i][n] == None):
                    rules[i][n] = '<UNK>'
            # print(rules[i])
            output_str += ','.join(rules[i]) + '\n'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_str)
        print("The number of samples generated is",num)
        print("Written", output_file)

    def predict_rules(self):                    # Rules for generating one-to-one orders
        print("Executed predict_rules")
        f = open('data/save/id2word.txt', 'r')
        id2word = eval(f.read())
        f.close()
        f = open('data/save/word2id.txt', 'r')
        word2id = eval(f.read())
        f.close()

        self.reset_rnn_state()
        reason = np.zeros([self.B, 1], dtype=np.int32)  # Generate a vertical array of size B, initially all 0
        word = '10001' # 10001,DOTHAN,KINGMAN,POCAHONTAS,21303;BOAZ is wrong
        reason[0][0] = word2id[word]
        # print("Reason for input section:", reason, id2word[reason[0][0]])  # ,action.shape,type(action)
        prob = self.predict(reason)
        result = np.argmax(prob, axis=-1).reshape([-1, 1])
        # print("Prediction of the maximum probable outcome of the output:", result, id2word[result[0][0]])
        result_ = np.random.choice(self.V, p=prob[0])
        # print("Predicted output with probability distribution_:", result_, id2word[result_])



    def multipredict_rules_argmax(self,reason):    # Rules for generating many-to-one orders
        print("Executed multipredict_rules")
        f = open('data/save/id2word.txt', 'r')
        id2word = eval(f.read())
        f.close()
        f = open('data/save/word2id.txt', 'r')
        word2id = eval(f.read())
        f.close()
        self.reset_rnn_state()
        action = np.zeros([self.B, 1], dtype=np.int32)
        # print(action.shape)
        # print(action)
        # reason=['10005','BOAZ','AL','36251']#['10005','BOAZ']
        # print(type(reason))
        for i in range(len(reason)):
            word = reason[i]
            action[0][0] = word2id[word]
            prob = self.predict(action)
            result = np.argmax(prob, axis=-1).reshape([-1, 1])
            result = id2word[result[0][0]]
            # Temporarily comment out the following
            #result_ = np.random.choice(self.V, p=prob[0])
            #result_ = id2word[result_]
            # This section shows recoverable
            # if (i==len(reason)-1):
            #     print("Reason for input section:", reason)  # ,action.shape,type(action)
            #     print("Prediction of the maximum probable outcome of the output:", result, )
            #     print("Predicted output with probability distribution_:", result_, )
        return result

    def multipredict_rules_probability(self,reason):    # Rules for generating many-to-one orders
        print("Executed multipredict_rules")
        f = open('data/save/id2word.txt', 'r')
        id2word = eval(f.read())
        f.close()
        f = open('data/save/word2id.txt', 'r')
        word2id = eval(f.read())
        f.close()

        self.reset_rnn_state()
        action = np.zeros([self.B, 1], dtype=np.int32)
        # print(action.shape)
        # print(action)
        # reason=['10005','BOAZ','AL','36251']#['10005','BOAZ']
        # print(type(reason))
        for i in range(len(reason)):
            word = reason[i]
            try:
                action[0][0] = word2id[word]
            except:
                action[0][0] = '3'

            prob = self.predict(action)
            result = np.argmax(prob, axis=-1).reshape([-1, 1])
            result = id2word[result[0][0]]
            result_ = np.random.choice(self.V, p=prob[0])
            result_ = id2word[result_]
            # This section shows recoverable
            # if (i==len(reason)-1):
            #     print("Reason for input section:", reason)  # ,action.shape,type(action)
            #     print("Prediction of the maximum probable outcome of the output:", result, )
            #     print("Predicted output with probability distribution_:", result_, )
        return result_

    def train_rules(self,rule_len,path):
        max_outer_iter = 100
        max_inner_iter = 100
        print("Executed train_rules")
        rules_idx = rule_len
        print("Sequence sample size is",rules_idx)
        f = open('data/save/id2word.txt', 'r')
        id2word = eval(f.read())
        f.close()
        f = open('data/save/word2id.txt', 'r')
        word2id = eval(f.read())
        f.close()
        f = open('data/save/rules_final.txt', 'r')
        try:
            rules_final = {} # If you want to improve on the previous version each time, it is eval(f.read())
        except:
            rules_final = {}
        f.close()
        
        # print(rules_idx, rules_final, path)
        progress_bar = tqdm.tqdm(range(rules_idx))
        for idx in progress_bar:
            # print(f"{idx}/{rules_idx}")
            sentence = linecache.getline(path, idx)  # Read the idx row of the original data
            words = sentence.strip().split(",")  # list type
            # print(words)
            # print()
            # print("sentence",sentence)          # Add explainable time display
            # print("words:",words)               #words: ['31301', 'BENSON', 'AZ', '85602', 'COCHISE']
            # print("words[0]:",words[0])
            f = open('data/save/att_name.txt', 'r')
            label2att = eval(f.read())
            f.close()


            # LHS multi-attribute function dependency, here if you want to restore the filtering at the time of rule generation, check this module of the E drive backup
            reason = words                                 #words: ['31301', 'BENSON', 'AZ', '85602', 'COCHISE']
            self.reset_rnn_state()                         # Resetting the LSTM state
            action = np.zeros([self.B, 1], dtype=np.int32)
            # print("self.B",self.B)
            for i in range(len(reason)-2):  # len(reason)
                # print("___________________")
                flag=i      # i is the last one of the current reason part, flag is the mark, is of course the first element in the reason location, when flag < 0, on behalf of the front has no use elements, that is, the reason part can not launch the result, give up the rule
                flag_=flag  # Add an index of information response, starting from the last position of REASON and moving forward

                outer_iteration_count = 0
                while(flag>=0):
                    if outer_iteration_count > max_outer_iter:
                        print("Too many outer iterations")
                        break
                    # print("flag",flag)
                    sqlex = ""
                    dic_name=[]
                    left_=[]        # Used to store the reason part of the construction dictionary
                    word_=[]        # is used to store the name of the constructed dictionary, which is actually the same as dic_name
                    inner_iteration_count = 0
                    while (flag_!=i):
                        if inner_iteration_count > max_inner_iter:
                            print("Too many inner iterations")
                            break
                        # print("i=",i,"flag=",flag,"flag_=",flag_)
                        word = reason[flag_]
                        left = label2att[flag_]  # New Information
                        try:
                            action[0][0] = word2id[word]
                        except:
                            print("Not in the dictionary", word, type(word))
                            action[0][0] = '3'

                        # print("Add information",action,"即",left,":",word)    # Add explainable time display
                        prob = self.predict(action)
                        dic_name.append(word)
                        left_.append(left)
                        word_.append(word)
                        flag_=flag_+1
                        inner_iteration_count += 1
            
                    word = reason[flag_]  # word: 31301;word: BENSON;word: AZ;word: 85602;word: COCHISE，One at a time
                    dic_name.append(word)
                    # print(word)
                    left = label2att[flag_]  # Get the attribute name of the corresponding position in the dictionary
                    right = label2att[i + 1]

                    try:
                        action[0][0] = word2id[word]
                    except:
                        action[0][0] = '3'
                        print("No such word in the dictionary")

                    prob = self.predict(action)
                    result = np.argmax(prob, axis=-1).reshape([-1, 1])  # Dictionary Index
                    result = id2word[result[0][0]]  # Actual content
                    # result_ = np.random.choice(self.V, p=prob[0])
                    # result_ = id2word[result_]
                    # print("Reason for input section:", word)  # ,action.shape,type(action)
                    # print("Prediction of the maximum probable outcome of the output:", result)
                    # print("Predicted output with probability distribution_:", result_)
                    # print("___________________")
                    self.reset_rnn_state()
                    # Create dictionary for subsequent addition of partial rules, but sorting needs to be in front
                    for n in range(len(left_)):
                        # print("Supplementary dictionaries")
                        # print(rules_final)
                        if (n==0):
                            addtwodimdict(rules_final, str(dic_name), 'reason', {str(left_[n]): str(word_[n])})
                        else:
                            addtwodimdict(rules_final[str(dic_name)], 'reason', str(left_[n]), str(word_[n]))


                    if (i==flag):
                        # print("At this point i=flag")
                        addtwodimdict(rules_final,str(dic_name), 'reason',{str(left): str(word)})
                        addtwodimdict(rules_final,str(dic_name), 'result', {str(right): str(result)})
                        # print(rules_final)
                        # rules_final.update({str(dic_name): {'reason': {str(left): str(word)},
                        #                                     'result': {str(right): str(result)}}})
                    else:
                        # print("Now i!=flag")
                        addtwodimdict(rules_final[str(dic_name)], 'reason',str(left), str(word))
                        addtwodimdict(rules_final[str(dic_name)], 'result', str(right), str(result))
                        # print(rules_final)

                    # print("At this point the cause section is", rules_final[str(dic_name)]['reason'],"Predicted value：", rules_final[str(dic_name)]['result'],"The actual value is",reason[i+1])    #  Add explainable time display
                    # Save the rules you think are correct to the dictionary, delete the wrong rules
                    if (result==reason[i+1]):
                        # print("The predicted value is the same as the actual value, save rule")       #Add explainable time display
                        # If it comes out, jump out of the loop, otherwise the flag is moved forward by one bit and additional information is added to continue the test
                        break
                    else:
                        # print("The predicted value does not match the actual value, increase the reason part")    # Add explainable time display
                        del rules_final[str(dic_name)]

                    flag=flag-1
                    flag_=flag
                    outer_iteration_count += 1
                    # if flag<0:
                        # print("There is no more information available, the reason part can not launch the result, abandon the rule")    # Add explainable time display

            f = open('data/save/rules_final.txt', 'w')
            # print(str(rules_final))
            f.write(str(rules_final))
            f.close()

            f = open('data/save/rules_read.txt', 'w')
            # print(str(rules_final))
            for item in rules_final.items():
                f.write(str(item))
                f.write('\r\n')
            f.write(str(rules_final))
            f.close()


        print("The rule generation is complete and the number of",len(rules_final))
        # print(str(rules_final))

    def filter(self,path):

        print("The filter is executed")
        f = open('data/save/att_name.txt', 'r')
        label2att = eval(f.read())
        f.close()
        att2label = {v: k for k, v in label2att.items()}  # Dictionary Reverse Transfer
        f = open('data/save/rules_final.txt', 'r')
        rules_final = eval(f.read())
        f.close()
        l1=len(rules_final)
        # print(rules_final)
        num = 0
        # conn = cx_Oracle.connect('system', 'Pjfpjf11', '127.0.0.1:1521/orcl')  # Connecting to the database
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        for rulename, ruleinfo in list(rules_final.items()):
            num += 1
            # print("Filter page", num, "Rules and corresponding data")
            # print("ruleinfo:", ruleinfo)

            left = list(ruleinfo['reason'].keys())
            # print(left)
            word = list(ruleinfo['reason'].values())
            # print(word)
            k = list(ruleinfo['result'].keys())
            right = k[0]
            v = list(ruleinfo['result'].values())
            result = v[0]



            sqlex = left[0] + "\"='" + word[0] + "'"
            i = 1
            while (i < len(left)):
                sqlex = sqlex + " and \"" + left[i] + "\"='" + word[i] + "'"
                i += 1

            sql1 = "select \"" + right + "\" from \"" + path + "\" where \"" + sqlex
            # print(sql1)         #select "MINIT" from "UIS_copy" where "CUID"='9078' and "RUID"='15896' and "SSN"='463210223' and "FNAME"='Monken'
            cursor.execute(sql1)  # "City","State" ,where rownum<=10
            rows = cursor.fetchall()
            num1=len(rows)
            if num1<3:
                # print("The data that satisfy the rule are",num1,"Article, the source is presumed to be wrong data, no sense of repair, delete the rule "",rules_final[str(rulename)])
                del rules_final[str(rulename)]
                continue
            else:
                t_rule=1
                for row in rows:
                    if (str(row[-1]) == str(result)):  # In this case, the rule matches the data, and the confidence of the rule increases
                        t_rule = t_rule + 1
                        print("-->", t_rule, end='')
                    else:  # In this case, the rule is contrary to the data, and the confidence of the rule is reduced
                        t_rule = t_rule - 2
                        print("-->", t_rule, end='')
                        flag = 0  # Mark the rule as conflicting with the data
                rules_final[str(rulename)].update({'confidence': t_rule})  # Rule confidence initialization
                # rules_final[str(rulename)].update({'confidence': 1})
            # sql2 = "select \"" + right + "\" from \"" + path + "\" where \"" + sqlex + " and \"" + right + "\"='" + result + "'"
            # # print(sql2)
            # cursor.execute(sql2)
            # rows = cursor.fetchall()
            # num2 = len(rows)
            # # print(num2)
            # ratio = num2 / num1
            # if ratio < 0.51:
            #     # print("Measure the source is wrong data, no sense to fix, delete the rule")
            #     del rules_final[str(rulename)]
            #     continue

        cursor.close()
        conn.close()


        f = open('data/save/rules_final.txt', 'w')
        f.write(str(rules_final))
        f.close()
        l2=len(rules_final)
        print("Rule filtering is complete and the remaining number of", )
        print(str(l2))
        # print(str(rules_final))           # Post-filtering rules
        with open('data/save/log_filter.txt', 'w') as f:
            f.write("The number of original rules is")
            f.write(str(l1))
            f.write("After rule filtering, the remaining number of")
            f.write(str(l2))
            f.write("__________")
        f.close()

    def detect(self,rows,result,rulename,LHS,RHS,att2label,label2att):
        dert = 0
        t0=1
        t_rule=t0
        t_tuple=t0
        t_max=t_tuple   # The maximum value of confidence in different tuples satisfying the RULE condition
        flag=1         # Flag whether the rule conflicts with the data
        flag_trust = 0  # 0 for believe data, 1 for believe rule
        for row in rows:
            if (str(row[RHS]) == str(result)):
                continue
            else:
                dert += 1
                flag = 0   # Mark the rule as conflicting with the data
        if (flag==1):           # If the rule does not conflict with the data, a great confidence level is given directly
            t_rule=t_rule+100
            flag_trust = 3  # 3 means the rule is correct and there is no conflict
            return flag_trust
        else:                   # The rule conflicts with the data, then the confidence of each tuple is calculated to adjust t_rule
            print("The rule conflicts with the data")
            print("Estimated changes for this restoration", dert)
            error_row=[]
            rule_other=[]
            t_rule=t0
            for row in rows:    # Each tuple that satisfies the rule condition
                AV_p=[]
                t_tp = 999   # The confidence of the current tuple is calculated as the minimum value of the confidence of different AV_i in a tuple, and a large value is set first to avoid the interference of the initial value.
                t_tc = t0
                # flag_p=0     # Used to record the position of the attribute corresponding to the lowest confidence level in AV_p
                # rule_p_name=[] # Record the rule with the highest confidence that can repair the attribute with the lowest confidence in the above AV_p
                # print("The tuples that match the current rule are：", row)
                for i in LHS:       # Calculate the minimum value of confidence in different AV_i in a tuple
                    AV_p.append(row[i])
                    t_AV_i = t0
                    # rulename_p_max = []
                    # t_rmax = 0
                    attribute_p=label2att[i]
                    for rulename_p, ruleinfo_p in list(self.rule.items()):      # Traversing the dictionary
                        if rulename == rulename_p:
                            continue
                        if t_AV_i>100 or t_AV_i<-100:
                            break
                        v = list(ruleinfo_p['result'].values())
                        left = list(ruleinfo_p['reason'].keys())
                        word = list(ruleinfo_p['reason'].values())
                        k = list(ruleinfo_p['result'].keys())
                        t_r = ruleinfo_p['confidence']
                        if t_r<0:
                            continue
                        right = k[0]
                        if attribute_p == right:
                            flag_equal = 0  # Can the rule determine the token of row[i]
                            for k in range(len(left)):
                                if row[att2label[left[k]]] == word[k]:  # If the tuple where row[i] is located satisfies all AV_p of a rule, mark it as 1
                                    flag_equal = 1
                                else:
                                    flag_equal = 0
                                    break
                            if flag_equal == 1:  # If the row[i] in the tuple can be determined by other rules, check whether it satisfies the rule
                                # print(row, "Medium")
                                # print(right, "Can be determined by other rules：", ruleinfo)
                                result2 = v[0]
                                # if t_rmax < t_r:  # Record the maximum rule confidence in these rules
                                #     t_rmax = t_rmax
                                #     rulename_p_max = rulename_p  # Record the identification of the most trusted rule in the dictionary
                                if str(row[i]) == str(result2):    # Retrieve other rules to determine the confidence level of each token in the tuple, increase if it is satisfied, and decrease if it is not.
                                    t_AV_i = t_AV_i + t_r
                                else:
                                    t_AV_i = t_AV_i - t_r
                                    print("The tuples that match the current rule are：", row)
                                    print("In AV_p",str(row[i]), "with", str(result2), "does not match, the corresponding rule is", ruleinfo_p, "Its confidence level is", t_r)

                    if t_tp > t_AV_i:
                        t_tp = t_AV_i
                        # flag_p=i
                        # rule_p_name=rulename_p_max


                for rulename_c, ruleinfo_c in list(self.rule.items()):  # Iterate through the dictionary, calculate t_c
                    if rulename==rulename_c:
                        continue
                    v = list(ruleinfo_c['result'].values())
                    left = list(ruleinfo_c['reason'].keys())
                    word = list(ruleinfo_c['reason'].values())
                    k = list(ruleinfo_c['result'].keys())
                    t_r = ruleinfo_c['confidence']
                    if t_r < 0:
                        continue
                    right = k[0]
                    attribute_c = label2att[RHS]
                    if attribute_c == right:
                        flag_equal = 0  # Can the rule determine the token of row[i]
                        for k in range(len(left)):
                            if row[att2label[left[k]]] == word[k]:  # If the tuple in which AV_c is located satisfies all AV_p of a rule, mark it as 1
                                flag_equal = 1
                            else:
                                flag_equal = 0
                                break
                        if flag_equal == 1:  # If the AV_c in the tuple can be determined by other rules, check whether it satisfies the rules
                            result2 = v[0]
                            if str(row[RHS]) == str(result2):
                                t_tc = t_tc + t_r
                            else:
                                t_tc = t_tc - t_r
                                print("The tuples that match the current rule are：", row)
                                print("In AV_c",str(row[RHS]), "with", str(result2), "does not match, the corresponding rule is", ruleinfo_c, "Its confidence level is", t_r)

                if t_tp==999:        # means that all cells in it cannot be determined by other rules, reset its value to t0
                    t_tp=t0
                if t_tc < t_tp:
                    t_tuple = t_tc
                else:
                    t_tuple = t_tp


                # print("The part that matches the rule is", AV_p, "-->",row[RHS],"Its confidence level is",t_tuple)
                if (str(row[RHS]) == str(result)):  # The tuple data is consistent with the rule, and the confidence level increases
                    # print("At this time t_rule=",t_rule,"t_tuple=",t_tuple,"math.ceil(math.log(1+t_tuple))=",math.ceil(math.log(1+t_tuple)))
                    # print("The rule determines the value of",result,";The actual value is",row[RHS],"match,rule confidence increase",t_rule, end='')
                    if t_tuple>0:
                        t_rule = t_rule + math.ceil(math.log(1+t_tuple))
                    else:
                        t_rule = t_rule + t_tuple
                    t_max = t_max
                    print("-->", t_rule, end='')
                else:  # If the tuple data violates the rule, calculate the confidence of the corresponding tuple
                    # print("At this time t_rule=", t_rule, "t_tuple=", t_tuple, "int(math.log(abs(t_tuple)))=",
                    #       int(math.log(abs(t_tuple))))
                    # print("The rule determines the value of", result, ";The actual value is", row[RHS], "Violation, rule confidence reduction", t_rule, end='')
                    if t_tuple>0:
                        t_rule = t_rule - 2*t_tuple
                    else:
                        t_rule = t_rule + math.ceil(math.log(1+abs(t_tuple)))
                        print("-->", t_rule, end='')

                    if (t_rule < -100):
                        flag_trust = 0
                        return flag_trust  # In this case, the confidence level of the rule is too small, so the loop is directly jumped and marked as error.

                if t_max < t_tuple:
                    t_max = t_tuple
                # if t_rule < t_max:
                #     flag_trust = 0
                #     return flag_trust

                # elif t_rule > t_max:
                #     error_row.append(row)
                #     rule_other.append(rule_p_name)


            print("The final rule confidence level is",t_rule,"The tuple with which it conflicts has the highest confidence level of",t_max)
        if (t_rule > t_max ):
             flag_trust = 1  # At this point the rule is considered correct and the data is modified
        elif (t_rule < t_max ):
             flag_trust = 0
             # print("The final rule confidence level is", t_rule, "The tuple with which it conflicts has the highest confidence level of", t_max)
             return flag_trust  # At this point the data is considered correct and the rule is modified
        self.rule[str(rulename)].update({'confidence': t_rule}) # Rule confidence initialization can be considered to be taken out separately
        print()
        return flag_trust


    def repair(self,iteration_num,path,order):

        print("Performed a REPAIR")
        f = open('data/save/att_name.txt', 'r')
        label2att = eval(f.read())
        f.close()
        # print(label2att)
        att2label = {v: k for k, v in label2att.items()}  # Dictionary Reverse Transfer
        f = open('data/save/rules_final.txt', 'r')
        self.rule = eval(f.read())
        f.close()
        # print(self.rule)
        num = 0
        error_rule=0
        error_data=0
        # conn = cx_Oracle.connect('system', 'Pjfpjf11', '127.0.0.1:1521/orcl')  # Connecting to the database
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        for rulename, ruleinfo in list(self.rule.items()):
            num += 1
            print("Fix the first", num, "Rules and corresponding data")
            # print("rulename:" + rulename)
            print("ruleinfo:", ruleinfo)

            left = list(ruleinfo['reason'].keys())
            # print(left)
            word = list(ruleinfo['reason'].values())
            # print(word)
            k = list(ruleinfo['result'].keys())
            right = k[0]
            v = list(ruleinfo['result'].values())
            result = v[0]

            LHS = []
            LHS.append(att2label[left[0]])
            RHS=att2label[right]
            sqlex = left[0] + "\"='" + word[0] + "'"
            i = 1
            # AV_p = "\""+left[0]+"\""+","     # Convert the data in left to string form
            while (i < len(left)):
                sqlex = sqlex + " and \"" + left[i] + "\"='" + word[i] + "'"
                # AV_p = AV_p +"\""+ left[i]+"\""+","
                LHS.append(att2label[left[i]])
                i += 1
                # print(sqlex)
            # print("AV_p Index：",LHS,"AV_c Index：",RHS)
            # AV_c = "\"" + right + "\""
            # print("AV_p:",AV_p,"AV_c:",AV_c)

            sql1 = "select * from \"" + path + "\" where \"" + sqlex
            #sql1 = "select " +AV_p+ AV_c + " from \"" + path + "\" where \"" + sqlex
            # sql1 = "select \"" + right + "\" from \"" + path + "\" where \"" + sqlex
            print(sql1)
            cursor.execute(sql1)  # "City","State" ,where rownum<=10
            rows = cursor.fetchall()
            # print("rows:")
            flag_trust=self.detect(rows,result,rulename,LHS,RHS,att2label,label2att)

            if (flag_trust == 3):       # 3 means the rule is correct, and there is no conflict, proceed directly to the next rule
                continue

            if (flag_trust == 0):
                error_rule += 1

            s1=0
            while (flag_trust == 0 and s1 < 3):
                print("Rules can't be trusted, fix the rules")
                print("Repair rules right")
                s1 += 1
                result=self.multipredict_rules_probability(word)
                print("Right side changed to",result)
                flag_trust=self.detect(rows,result,rulename,LHS,RHS,att2label,label2att)

                    # print("trust=",trust)
                if (flag_trust==1):
                    print("Rule repair successful")
                    addtwodimdict(self.rule,str(rulename), 'result', {str(right): str(result)})
                    print("The modified rule is",self.rule[str(rulename)])
                elif (flag_trust==0 and s1==5):
                    print("No replacement fix on the right side of the rule")


            s2 = 0
            while (flag_trust == 0 and s2 < 3):
                result = v[0]
                print("Repair rules left")
                s2 += 1
                min=10
                flag = int(att2label[left[0]])
                # print(flag)
                if (min > flag):
                    min = flag  # The index corresponding to the leftmost part of the current REASON section
                # print(min)
                if(min==0):
                    print("No fixes can be added to the left side of the rule, delete the rule")
                    del self.rule[str(rulename)]
                    break
                left_new=label2att[min-1]
                print("Add",left_new,"Information")
                sqladd= "select \"" + left_new + "\" from \"" + path + "\" where \"" + sqlex+"and \"" + right + "\"='" + result + "'"
                print("sqladd:",sqladd)
                cursor.execute(sqladd)
                rows_left = cursor.fetchall()
                # print(rows[0][0])
                # print(word)
                # print(self.rule[str(rulename)])


                #Reconstructing the dictionary
                if(rows_left ==[]):
                    # print("There is no condition modified on the left side of the rule, delete the rule")
                    del self.rule[str(rulename)]
                    break
                # print(rows_left)
                addtwodimdict(self.rule[str(rulename)], 'reason', str(left_new), str(rows_left[0][0]))
                for n in range(len(word)):
                    del self.rule[str(rulename)]['reason'][left[n]]
                    addtwodimdict(self.rule[str(rulename)], 'reason', str(left[n]), str(word[n]))
                # left = list(ruleinfo['reason'].keys())
                # Otherwise, the addition to the dictionary should be at the top, but now it's at the end
                # tex=[]
                # tex.append(rows_left[0][0])
                # for t in range(len(word)):
                #     tex.append(word[t])
                # # print(tex)
                # word = tex
                left = list(ruleinfo['reason'].keys())
                word = list(ruleinfo['reason'].values())
                # print(word)
                # print(self.rule[str(rulename)])
                sqlex = left[0] + "\"='" + word[0] + "'"
                i = 1
                while (i < len(left)):
                    sqlex = sqlex + " and \"" + left[i] + "\"='" + word[i] + "'"
                    i += 1
                sql1 = "select * from \"" + path + "\" where \"" + sqlex
                # print(sql1)
                cursor.execute(sql1)  # "City","State" ,where rownum<=10
                rows = cursor.fetchall()

                if (len(rows)<3):
                    continue

                result = self.multipredict_rules_argmax(word)
                # print(result)
                flag_trust=self.detect(rows,result,rulename,LHS,RHS,att2label,label2att)
                if (flag_trust == 1):
                    print("Rule repair successful")
                    print("The modified rule is", self.rule[str(rulename)])
                elif (flag_trust == 1 and min!=0) :
                    # print("There is no condition modified on the left side of the rule, delete the rule")
                    del self.rule[str(rulename)]
                    break
            if (flag_trust == 0):
                print("Rule is not available to fix, delete the rule")

            if (flag_trust == 1):
                t0=1
                for row in rows:
                    if (str(row[RHS]) == str(result)):
                        continue
                    else:
                        AV_p = []
                        t_tp = 999  # The confidence of the current tuple is calculated as the minimum value of the confidence of different AV_i in a tuple, and a large value is set first to avoid the interference of the initial value.
                        t_tc = t0
                        flag_p=0     # Used to record the position of the attribute corresponding to the lowest confidence level in AV_p
                        rule_p_name=[] # Record the rule with the highest confidence that can repair the attribute with the lowest confidence in the above AV_p
                        print("The tuples that match the current rule are：", row)
                        for i in LHS:  # Calculate the minimum value of confidence in different AV_i in a tuple
                            AV_p.append(row[i])
                            t_AV_i = t0
                            attribute_p = label2att[i]
                            rulename_p_max = []
                            t_rmax = -999       # The maximum confidence level of the rules that correct AV_i in the following iterative dictionary, initially set to a minimal value
                            for rulename_p, ruleinfo_p in list(self.rule.items()):  # Traversing the dictionary
                                if rulename == rulename_p:
                                    continue
                                if t_AV_i > 100 or t_AV_i < -100:
                                    break
                                v = list(ruleinfo_p['result'].values())
                                left = list(ruleinfo_p['reason'].keys())
                                word = list(ruleinfo_p['reason'].values())
                                k = list(ruleinfo_p['result'].keys())
                                t_r = ruleinfo_p['confidence']
                                if t_r < 0:
                                    continue
                                right = k[0]
                                if attribute_p == right:
                                    flag_equal = 0  # Can the rule determine the token of row[i]
                                    for k in range(len(left)):
                                        if row[att2label[left[k]]] == word[k]:  # If the tuple where row[i] is located satisfies all AV_p of a rule, mark it as 1
                                            flag_equal = 1
                                        else:
                                            flag_equal = 0
                                            break
                                    if flag_equal == 1:  # If the row[i] in the tuple can be determined by other rules, check whether it satisfies the rule
                                        # print(row, "Medium")
                                        # print(right, "Can be determined by other rules：", ruleinfo)
                                        result2 = v[0]
                                        if t_rmax < t_r:  # 记Record the maximum rule confidence in these rules
                                            t_rmax = t_rmax
                                            rulename_p_max = rulename_p  # Record the identification of the most trusted rule in the dictionary
                                        if str(row[i]) == str(result2):
                                            t_AV_i = t_AV_i + t_r
                                        else:
                                            t_AV_i = t_AV_i - t_r
                                            print("In AV_p", str(row[i]), "with", str(result2), "does not match, the corresponding rule is", ruleinfo_p,
                                                  "Its confidence level is", t_r)

                            if t_tp > t_AV_i:
                                t_tp = t_AV_i
                                flag_p=i                    # Record the index of AV_i with the lowest confidence
                                rule_p_name=rulename_p_max  # Record the name of the rule that corrects this AV_i with the highest confidence

                        for rulename_c, ruleinfo_c in list(self.rule.items()):  # Iterate through the dictionary, calculate t_c
                            if rulename == rulename_c:
                                continue
                            v = list(ruleinfo_c['result'].values())
                            left = list(ruleinfo_c['reason'].keys())
                            word = list(ruleinfo_c['reason'].values())
                            k = list(ruleinfo_c['result'].keys())
                            t_r = ruleinfo_c['confidence']
                            if t_r < 0:
                                continue
                            right = k[0]
                            attribute_c = label2att[RHS]
                            if attribute_c == right:
                                flag_equal = 0  # Can the rule determine the token of row[i]
                                for k in range(len(left)):
                                    if row[att2label[left[k]]] == word[k]:  # If the tuple in which AV_c is located satisfies all AV_p of a rule, mark it as 1
                                        flag_equal = 1
                                    else:
                                        flag_equal = 0
                                        break
                                if flag_equal == 1:  # If the AV_c in the tuple can be determined by other rules, check whether it satisfies the rules
                                    result2 = v[0]
                                    if str(row[RHS]) == str(result2):
                                        t_tc = t_tc + t_r
                                    else:
                                        t_tc = t_tc - t_r
                                        print("In AV_c", str(row[RHS]), "with", str(result2), "does not match, the corresponding rule is", ruleinfo_c, "Its confidence level is",
                                              t_r)

                        if t_tp == 999:  # means that all cells in it cannot be determined by other rules, reset its value to t0
                            t_tp = t0
                        if t_tc < t_tp or t_tc == t_tp:
                            print("In this case, the data result is considered partially wrong, and the data is repaired according to the rule, the current rule is",rulename,"-->",result,"t_p is",t_tp,"t_c is",t_tc)
                            for x in range(len(row)-1):  # t2
                                if x == 0:
                                    sql_info = f"\"{label2att[x]}\"='{row[x]}'"
                                else:
                                    sql_info += f" and \"{label2att[x]}\"='{row[x]}'"
                            sql_update = "update \"" + path + "\" set \"Label\"='2' , \"" + label2att[RHS] + "\"='" + result + "' where  " + sql_info + ""
                            print("Original: ", sql_info)
                            print("Update Information: ", sql_update)
                            cursor.execute(sql_update)
                            conn.commit()
                        else:
                            print(rule_p_name)
                            if rule_p_name==[]:
                                print("There may be errors")
                                continue
                            rname=self.rule[str(rule_p_name)]
                            v2 = list(rname['result'].values())
                            result2 = v2[0]
                            print("At this point, the data inference is considered partially wrong, and the data is "
                                  "repaired according to the rule, the current rule is", rule_p_name, "-->", result2,
                                  "t_p is", t_tp, "t_c is", t_tc)
                            for x in range(len(row)-1):  # t2
                                if x == 0:
                                    sql_info = "\"" + label2att[x] + "\"='" + row[x] + "'"
                                else:
                                    sql_info = sql_info + " and \"" + label2att[x] + "\"='" + row[x] + "'"
                            sql_update = "update \"" + path + "\" set \"Label\"='2' , \"" + label2att[flag_p] + "\"='" + result2 + "' where  " + sql_info + ""
                            print("Original: ", sql_info)
                            print("Update Information: ", sql_update)
                            cursor.execute(sql_update)
                            conn.commit()
                            continue


            # if (flag_trust == 1):
            #     # Fix only the wrong data marked as '1' and marked as 2
            #     # sql_update = "update \"Hosp2_rule_copy\" set \"" + right + "\"='" + result + "' , \"Label\"='2'   where \"Label\"='1' or \"Label\"='2' and \"" + sqlex
            #     # print(sql_update)
            #     # sql_check = "select * from \"" + path + "\"   where  \"" + sqlex#\"" + right + "\"  #(\"Label\"='1' or \"Label\"='2') and
            #     sql_check = "select * from \"" + path + "\"   where  (\"Label\"='1' or \"Label\"='2') and \"" + sqlex  # \"" + right + "\"
            #     # print(sql_check)
            #     cursor.execute(sql_check)
            #     row_check = cursor.fetchall()
            #     row_check = [x[:-1] for x in row_check]
            #     if order == 0:
            #         row_check = [x[::-1] for x in row_check]
            #     # print(row_check)
            #     flag_check=att2label[right]
            #     for row in row_check:
            #         t2 = len(row)
            #         att = list(label2att.values())
            #         # print(right,row[flag_check],result)
            #         if (row[flag_check]!=result):
            #             error_data+=1
            #             for i in range(t2):  # t2
            #                 if i == 0:
            #                     sql_info = "\"" + att[i] + "\"='" + row[i] + "'"
            #                 else:
            #                     sql_info = sql_info + " and \"" + att[i] + "\"='" + row[i] + "'"
            #             # sql_info = sql_info + " and (\"Label\"='1' or \"Label\"='2')"
            #             print("原始：",sql_info)
            #             # row = list(row)
            #             # row[flag_check] = result
            #             # print("row[flag_check]:",row[flag_check])
            #             # print("result",result)
            #             sql_update="update \"" + path + "\" set \"Label\"='2' , \"" + att[flag_check] + "\"='" + result + "' where  " + sql_info + ""
            #             print("Update Information：", sql_update)
            #             cursor.execute(sql_update)
            #             conn.commit()

        cursor.close()
        conn.close()
            # if num>200:
            #     break

        print("Repair completed")
        print("Save repair rules")
        print("Rule dictionary size", len(self.rule))
        # print(str(self.rule))
        f = open('data/save/rules_final.txt', 'w')
        f.write(str(self.rule))
        f.close()
        with open('data/save/log.txt', 'a') as f:
            f.write("Total number of rules used this time")
            f.write(str(num))
            f.write("Number of rule errors")
            f.write(str(error_rule))
            f.write("Number of data errors")
            f.write(str(error_data))
            f.write("__________")
            f.close()


        # This can be used to loop through the fixes until there is no new error data
        # if (iteration_num>0):
        #     print(iteration_num)
        #     if (error_rule != 0):
        #         self.repair(iteration_num-1,path,order)


    def save(self, path):
        weights = []
        for layer in self.layers:
            w = layer.get_weights()
            weights.append(w)
        with open(path, 'wb') as f:
            pickle.dump(weights, f)

    def load(self, path):
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        for layer, w in zip(self.layers, weights):
            layer.set_weights(w)

def addtwodimdict(thedict, key_a, key_b, val):
  if key_a in thedict:
    thedict[key_a].update({key_b: val})
  else:
    thedict.update({key_a:{key_b: val}})
