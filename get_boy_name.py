import keras
import tensorflow as tf
import numpy as np

char_to_index = dict( (chr(i+96), i) for i in range(1,27))
char_to_index[' '] = 0
char_to_index['-'] = 27
char_to_index['.'] = 28

index_to_char = dict( (i, chr(i+96)) for i in range(1,27))
index_to_char[0] = ' '
index_to_char[27] = '-'
index_to_char[28] = '.'
max_char = 19
char_dim = 29

x = np.zeros((1, max_char, char_dim))

model = tf.keras.models.load_model('boy_name.h5')

def make_name():  
    name = []
    x = np.zeros((1, max_char, char_dim))
    end = False
    i = 0
    while end==False:
        probs = list(model.predict(x)[0,i])
        probs = probs / np.sum(probs)
        index = np.random.choice(range(char_dim), p=probs)
        if i == max_char-2:
            character = '.'
            end = True
        else:
            character = index_to_char[index]
        name.append(character)
        x[0, i+1, index] = 1
        i += 1
        if character == '.':
            end = True
    
    print(''.join(name))


def make_name2(start_ch):  
    name = []
    x = np.zeros((1, max_char, char_dim))
    ii = char_to_index[start_ch]
    x[0,0,ii] = 1
    end = False
    i = 1
    name.append(start_ch)
    while end==False:
        probs = list(model.predict(x)[0,i])
        probs = probs / np.sum(probs)
        index = np.random.choice(range(char_dim), p=probs)
        if i == max_char-2:
            character = '.'
            end = True
        else:
            character = index_to_char[index]
        name.append(character)
        x[0, i+1, index] = 1
        i += 1
        if character == '.':
            end = True
    
    print(''.join(name))
   
a = int(input('Do you want to enter first character of name: 0 for No / 1 for Yes'))

if a == 0:
  for i in range(10):
    make_name()
else:
  b = input('enter first character')
  b = b.lower()
  for i in range(10):  
    make_name2(b)
input()    
