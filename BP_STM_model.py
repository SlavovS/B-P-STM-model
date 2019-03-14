
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import TimeDistributed


#  STM model for serial recall   (Botvinick & Plaut, 2006)
  
    
from BP_STM_parameters import *



# Generating a letter (localist representation), and an "off" recall cue.
def ran_letter_gen():
    letters = np.random.permutation(letters_3)   
    return np.array(np.append(letters, [0])).astype("float64")


# Generates an "on" recall cue attached to the letters
def recall_cue():
    return np.array(np.append(np.zeros(len(letters_3)), [1]))#np.array([0, 0, 0, 1])



# Generating the input and output letters

input_letters = []
output_letters = []
def input_output():

    global input_letters
    global output_letters
  
    for sequence in range(0, length_of_list):
        # These are the number of letters in a list; 
        input_letters.append(ran_letter_gen())
    
    output_letters = input_letters 
    return input_letters, output_letters    

   
# Generating the input and output for the model

final_input_letters = []
final_output_letters = []    
def final_input():
    global input_letters
    global output_letters
    global final_input_letters
    global final_output_letters
    input_letters = []
    output_letters = []
    input_output()
    global final_input_letters
    
    for i in range(len(input_letters)):
        input_letters.append(recall_cue())  # appends the exact number of recall
                                            # cues as there are letters
    
    final_input_letters = np.array(np.append(input_letters, recall_cue()))
    final_input_letters = np.reshape(final_input_letters, (2*length_of_list+1, len(letters_3) + 1))
    
    final_output_letters = np.array(output_letters[0:-length_of_list]+output_letters[0:-length_of_list+1])
    
    return final_input_letters, final_output_letters


# Generating the input and output for training

X = []
y = []
for letter in range(0, training_iterations):
    X.append(final_input()[0])
    y.append(final_input()[1])
    
X = np.array(X, dtype=float)
y = np.array(y, dtype=float)



# Validation

X_test = []
y_test = []
for letter in range(0, test_iterations):
    X_test.append(final_input()[0])
    y_test.append(final_input()[1])
    
X_test = np.array(X, dtype=float)
y_test = np.array(y, dtype=float)




# Fitting the model


BP_model = Sequential()
BP_model.add(
        LSTM(
                units = 50, 
                input_shape=(None, len(letters_3) + 1), 
                return_sequences=True
        )
    )
    
BP_model.add(
        TimeDistributed(
                Dense(
                        units = len(letters_3) + 1,
                        activation="softmax"
                )
            )
        )
BP_model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
BP_model.fit(        
        X, y,
        batch_size=1, 
        nb_epoch=500, 
        verbose=2, 
        validation_data=(X_test, y_test)
    )
BP_model.save('B&P_model.h5')
