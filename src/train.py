import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


bank_notes = pd.read_csv('bank_note_data.csv')
#Output: (1372, 5)
bank_notes_without_class = bank_notes.drop('Class', axis=1)
scaler = StandardScaler()
scaler.fit(bank_notes_without_class)
scaled_features = pd.DataFrame(data=scaler.transform(bank_notes_without_class), columns=bank_notes_without_class.columns)

bank_notes = bank_notes.rename(columns={'Class': 'Authentic'})

bank_notes.loc[bank_notes['Authentic'] == 0, 'Forged'] = 1
bank_notes.loc[bank_notes['Authentic'] == 1, 'Forged'] = 0

x = scaled_features
y = bank_notes[['Authentic', 'Forged']]

x = x.as_matrix()
y = y.as_matrix()
(x_train, x_valid)=x[100:],x[100:]
(y_train, y_valid)=y[100:],y[100:]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = Sequential()
model.add(Dense(4, input_dim=4, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(4, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=50,
          batch_size=100,validation_data=(x_valid, y_valid))
score = model.evaluate(x_test, y_test, batch_size=100)



model.save('model.h5')