
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
    letters = np.random.permutation(letters_26)   
    return np.array(np.append(letters, [0])).astype("float64")


# Generates an "on" recall cue attached to the letters
def recall_cue():
    return np.array(np.append(np.zeros(len(letters_26)), [1]))#np.array([0, 0, 0, 1])



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
    final_input_letters = np.reshape(final_input_letters, (2*length_of_list+1, len(letters_26) + 1))
    
    final_output_letters = np.array(output_letters[0:-length_of_list]+output_letters[0:-length_of_list+1])
    
    return final_input_letters, final_output_letters


# Generating the input and output for training

X = []
y = []
for letter in range(0, training_examples):
    training_array = final_input()
    X.append(training_array[0])
    y.append(training_array[1])
    
X = np.array(X, dtype=float)
y = np.array(y, dtype=float)



# Validation

X_test = []
y_test = []
for letter in range(0, test_examples):
    test_array = final_input()
    X_test.append(test_array[0])
    y_test.append(test_array[1])
    
X_test = np.array(X_test, dtype=float)
y_test = np.array(y_test, dtype=float)




# Fitting the model


BP_model = Sequential()
BP_model.add(
        LSTM(
                units = 50, 
                input_shape=(None, len(letters_26) + 1), 
                return_sequences=True
        )
    )
    
BP_model.add(
        TimeDistributed(
                Dense(
                        units = len(letters_26) + 1,
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
BP_model.save('Chernova_B&P_model.h5')


  #Reloading and testing the model:
      #Epoch 500/500  70 training examples, 30 test examples:
      # loss: 5.9517e-05 - acc: 1.0000 - val_loss: 1.9525 - val_acc: 0.7286

from keras.models import load_model
ISR_model = load_model('Chernova_B&P_model.h5')


#Epoch 500/500   with 10 training examples
#0s - loss: 0.0462 - acc: 0.9714 - val_loss: 0.0459 - val_acc: 0.9714
check = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]], 
            dtype=float).reshape(1, 2*length_of_list+1, len(letters_26) + 1)

#check = np.array([[0,0,1,0],
#                  [0,1,0,0],
#                  [1,0,0,0],
#                  [0,0,0,1],
#                  [0,0,0,1],
#                  [0,0,0,1],
#                  [0,0,0,1]],
#            dtype=float).reshape(1, 2*length_of_list+1, len(letters_3) + 1)


prediction = ISR_model.predict(check)
print(prediction[0][3])
print(prediction[0][4])
print(prediction[0][5])

# Epoch 500/500  with 100 training examples
# 1s - loss: 0.2319 - acc: 0.8671 - val_loss: 0.2168 - val_acc: 0.8900


# Checking the cosine similarity
from scipy import spatial

dataSetI = check[0][0]
dataSetII = prediction[0][3]
result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
print("The cosine similarity between a letter in the input and the model's output is: {} ".format(result))
