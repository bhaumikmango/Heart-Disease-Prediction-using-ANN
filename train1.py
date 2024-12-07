import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load data
data = pd.read_csv("heart.csv")

# Data preprocessing
X = data.iloc[:, :13].values
y = data["target"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Save the scaler model
joblib.dump(sc, 'scaler.pkl')

# Build the ANN model
classifier = Sequential()
classifier.add(Dense(units=8, activation='relu', kernel_initializer='uniform', input_dim=13))
classifier.add(Dense(units=14, activation='relu', kernel_initializer='uniform'))
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size=8, epochs=100, verbose=1)

# Save the ANN model
classifier.save('heart_disease_model1.h5')

print("Model training complete. Model and scaler saved.")