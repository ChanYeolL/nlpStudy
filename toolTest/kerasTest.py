#-----------------------------------------------------
# Editor:Chanyeol Liu    Date:2019/12/06
# Code:CL191206
# Purpose:Keras Test
# https://study.163.com/course/courseLearn.htm?courseId=1003340023#/learn/video?lessonId=1003802569&courseId=1003340023
#-----------------------------------------------------
import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

X = np.linspace(-1, 1, 200)
np.random.shuffle(X)
y = 0.5*X + 2 + np.random.normal(0, 0.05, (200, ))

plt.scatter(X, y)
plt.show()

X_train, y_train = X[:160], y[:160]
X_test, y_test = X[160:], y[160:]

model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))

model.compile(loss='mse', optimizer='sgd')

print("Training ----------")
for step in range(301):
    cost = model.train_on_batch(X_train, y_train)
    if step%100 == 0:
        print('train cost:', cost)

print("Test ----------")
cost = model.evaluate(X_test, y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('W: ', W, 'b: ', b)

y_pred = model.predict(X_test)
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.show()
