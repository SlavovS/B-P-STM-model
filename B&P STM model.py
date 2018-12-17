# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 15:59:54 2018

@author: Slavi Slavov
"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout




#  Auto-encoding model

def ran_num_gen():
    result=[]
    numbers = np.random.randint(0,9,3)
    result.append(numbers)
    
    return result
  

input_numbers = []
for num in range(0,99):
    input_numbers.append(ran_num_gen())

input_numbers = np.array(input_numbers, dtype=float)

output_numbers = input_numbers
  

input_test = []
for num in range(0,99):
    input_test.append(ran_num_gen())

input_test = np.array(input_test, dtype=float)

output_test = input_test


recall_model = Sequential()
recall_model.add(LSTM(3, input_shape=(1,3), return_sequences=True))
    
recall_model.add(Dense(3))
recall_model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])
#recall_model.fit(input_numbers, output_numbers, batch_size=1, nb_epoch=500, verbose=2, validation_data=(input_test,output_test))
#recall_model.save('mini_auto_encoder_model.h5')

from keras.models import load_model
recall_model = load_model('mini_auto_encoder_model.h5')

input_validation = []
for num in range(0,99):
    input_validation.append(ran_num_gen())

input_validation = np.array(input_validation, dtype=float)

check = np.array([[7,2,8]], dtype=float).reshape(1,1,3)
predict=recall_model.predict(check)
#########################################################################################
#  STM for serial recall model   (Botvinick & Plaut, 2006)
  
    
def ran_letter_gen():
    numbers = np.random.permutation([0, 0, 1])   
    return np.append(numbers, [0])

def recall_cue():
    return np.array([0, 0, 0, 1])



input_numbers = []
for letter in range(0, 8):
    for sequence in range(0,3):
        input_numbers.append(ran_letter_gen())
    input_numbers.append(recall_cue())
    
    
    


input_numbers = np.array(input_numbers, dtype=float)
#input_final = 

cues = input_numbers[3::4]


#output_numbers =             input_numbers

end_recall = recall_cue()




stm_model = Sequential()
stm_model.add(LSTM(4, input_shape=(1,4), return_sequences=True))
    
stm_model.add(Dense(4))
stm_model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])
#stm_model.fit(input_final, output_numbers, batch_size=1, nb_epoch=500, verbose=2, validation_data=(input_test,output_test))
#stm_model.save('stm_model.h5')