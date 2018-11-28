import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
bank_notes = pd.read_csv('bank_note_data.csv')

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
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(x_train, y_train)
preds_rfc = rfc.predict(x_test)
# Get only the 'Forged' column values from y_test and preds_rfc
y_test_forged = [item[1] for item in y_test]
preds_rfc_forged = [item[1] for item in preds_rfc]
# Print confusion matrix
print(confusion_matrix(y_test_forged, preds_rfc_forged))