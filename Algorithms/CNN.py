import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Lambda, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.metrics import r2_score

# 1. Data Generation and Preparation
np.random.seed(42)
tf.random.set_seed(42)

num_samples = 2000   
time_steps = 5       
feature_dim = 50   

X = np.random.rand(num_samples, time_steps, feature_dim).astype(np.float32)

y = 5.0 * np.sum(X, axis=(1, 2))  
y = y.reshape(-1, 1)              
y += 0.1 * np.random.randn(num_samples, 1) 

print("X shape:", X.shape)  
print("y shape:", y.shape)  

split_idx = int(0.8 * num_samples)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print("X_train shape:", X_train.shape)  
print("y_train shape:", y_train.shape)   
print("X_test shape:", X_test.shape)      
print("y_test shape:", y_test.shape)     

# 2. Define an Improved CNN Model with Sum Pooling
inputs = Input(shape=(time_steps, feature_dim), name="input_sequence")
x = Conv1D(filters=64, kernel_size=2, activation='relu', padding='same')(inputs)
x = Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)  # shape: (batch, 64)
outputs = Dense(1, activation='linear')(x)

model = Model(inputs=inputs, outputs=outputs, name="Improved_CNN_Yield_Model")
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

# 3. Train the Model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
# 4. Evaluate the Model and Compute the R² Score
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {test_loss:.4f}")

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Test R²: {r2:.4f}")
print("Predictions shape:", y_pred.shape)

# 5. Predict on New Data
X_new = np.random.rand(1, time_steps, feature_dim).astype(np.float32)
new_pred = model.predict(X_new)
print("Prediction for new sample (CNN):", new_pred[0])
