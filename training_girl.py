import pandas as pd
import numpy as np

data = pd.read_csv('girl_names.csv', index_col = False,  names = ['Name'], engine='python')
data_boy = []
for i, j in data.iterrows():
  data_boy.append(j['Name'])
  
lis = []
for i in range(len(data_boy)):
  j = data_boy[i]
  j = j.split(' ')
  j = j[0].split("'")
  j = j[0].split('�')
  lis.extend(j)

data_boy = [x.lower() + '.' for x in lis]

lis = []
for j in data_boy:
  chars = list(j)
  lis.extend(chars)
lis = set(lis)

char_to_index = dict( (chr(i+96), i) for i in range(1,27))
char_to_index[' '] = 0
char_to_index['-'] = 27
char_to_index['.'] = 28

index_to_char = dict( (i, chr(i+96)) for i in range(1,27))
index_to_char[0] = ' '
index_to_char[27] = '-'
index_to_char[28] = '.'

max_char = len(max(data_boy, key=len))
m = len(data_boy)
char_dim = len(char_to_index)

X = np.zeros((m, max_char, char_dim))
Y = np.zeros((m, max_char, char_dim))

for i in range(m):
    name = list(data_boy[i])
    for j in range(len(name)):
        X[i, j, char_to_index[name[j]]] = 1
        if j < len(name)-1:
            Y[i, j, char_to_index[name[j+1]]] = 1
            
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.callbacks import LambdaCallback

model = Sequential()
model.add(LSTM(128, input_shape=(max_char, char_dim), return_sequences=True))
model.add(Dense(char_dim, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

name_generator = LambdaCallback(on_epoch_end = generate_name_loop)

model.fit(X, Y, batch_size=64, epochs=500, callbacks=[name_generator], verbose=1)

model.save('girl_name.h5')
