import keras
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.utils import to_categorical
from keras.layers import Dense, Embedding, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.losses import categorical_crossentropy
from methods.distilled_garf.SeqGAN.utils import GeneratorPretrainingGenerator, sentence_to_ids, pad_seq
from methods.distilled_garf.SeqGAN.models import GeneratorPretraining


class GeneratorDistilledGenerator(GeneratorPretrainingGenerator):              #Take the data directly from the original data and make x and y_true as x_train and y_train i.e. training data and labels
    def __init__(self, temperature, path, order, B, T, min_count=1, shuffle=True, remove_amount_of_error_tuples=0):
        super().__init__(path, order, B, T, min_count, shuffle, remove_amount_of_error_tuples)
        self.temperature = temperature
        self.teacher_model = None
        self.predict_counter = 0
        
    def set_teacher_model(self, teacher_model_path: str, V, E, H):
        if self.teacher_model is not None:
            raise Exception('Teacher model is already set')
        
        self.teacher_model_path = teacher_model_path
        self.V = V
        self.E = E
        self.H = H
        
    def on_epoch_end(self):
        super().on_epoch_end()
        print(f"Predict counter: {self.predict_counter}")
        self.predict_counter = 0

    def __getitem__(self, idx):
        if self.teacher_model_path is None:
            raise Exception('Teacher model is not set')

        self.teacher_model = GeneratorPretraining(self.V, self.E, self.H)
        self.teacher_model.load_weights(self.teacher_model_path)
        
        if not hasattr(self, 'soft_label_cache'):
            self.soft_label_cache = {}
        
        x, y_true_hard = [], []
        start = (idx-1) * self.B + 1
        end = idx * self.B + 1
        max_length = 0
        for i in range(start, end):
            if self.shuffle:
                idx = self.shuffled_indices[i]
            else:
                idx = i

            sentence = self.rows[idx]                         
            words = []
            for i in sentence:
                words.append(i)
            ids = sentence_to_ids(self.vocab, words)        

            ids_x, ids_y_true_hard = [], []                      

            ids_x.append(self.BOS)                          
            ids_x.extend(ids)                               
            ids_x.append(self.EOS) 
            x.append(ids_x)                                 

            ids_y_true_hard.extend(ids)
            ids_y_true_hard.append(self.EOS) 
            y_true_hard.append(ids_y_true_hard)                       

            max_length = max(max_length, len(ids_x))

        if self.T is not None:
            max_length = self.T

        for i, ids in enumerate(x):
            x[i] = x[i][:max_length]                

        for i, ids in enumerate(y_true_hard):
            y_true_hard[i] = y_true_hard[i][:max_length]

        x = [pad_seq(sen, max_length) for sen in x]     
        x = np.array(x, dtype=np.int32)

        y_true_hard = [pad_seq(sen, max_length) for sen in y_true_hard]
        y_true_hard = np.array(y_true_hard, dtype=np.int32)

        y_true_hard = to_categorical(y_true_hard, num_classes=self.V)

        y_true_soft = []
        for input_vector in x:
            input_tuple = tuple(input_vector)  # Convert the input vector to a tuple so it can be used as a dictionary key
            if input_tuple in self.soft_label_cache:
                soft_labels = self.soft_label_cache[input_tuple]
            else:
                soft_labels = self.teacher_model.predict(np.array([input_vector]))[0]  # Predict only accepts arrays, not single vectors
                self.soft_label_cache[input_tuple] = soft_labels
                self.predict_counter += 1
            y_true_soft.append(soft_labels)

        y_true_soft = np.array(y_true_soft)  # Convert the list of soft labels to a numpy array
        y_true_soft = y_true_soft / self.temperature # Apply temperature

        return (x, [y_true_hard, y_true_soft])

def DistilledGenerator(V, E, H):
    '''
    Create a simpler student model.
    V: int, Vocabulary size
    E: int, Embedding size
    H: int, LSTM hidden size
    '''
    input = Input(shape=(None,), dtype='int32', name='Input')
    out = Embedding(V, E, mask_zero=True, name='Embedding')(input)
    out = LSTM(H//2, return_sequences=True, name='LSTM')(out)  # Less LSTM units
    hard_out = TimeDistributed(
        Dense(V, activation='softmax', name='Hard_DenseSoftmax'),
        name='Hard_TimeDenseSoftmax'
    )(out)
    
    soft_out = TimeDistributed(
        Dense(V, activation='softmax', name='Soft_DenseSoftmax'),
        name='Soft_TimeDenseSoftmax'
    )(out)

    student_model = Model(input, [hard_out, soft_out])
    return student_model