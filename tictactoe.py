import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#1
def load_ttt(shuffle=False):   
    label={'x':0,'o':1,'b':2,'false':0,'true':1}
    data = np.loadtxt("./Data/tic-tac-toe.csv", skiprows=1, delimiter=',',
                      converters={0:lambda name: label[name.decode()],
                                  1:lambda name: label[name.decode()],
                                  2:lambda name: label[name.decode()],
                                  3:lambda name: label[name.decode()],
                                  4:lambda name: label[name.decode()],
                                  5:lambda name: label[name.decode()],
                                  6:lambda name: label[name.decode()],
                                  7:lambda name: label[name.decode()],
                                  8:lambda name: label[name.decode()],
                                  9:lambda name: label[name.decode()]})
    if shuffle:
        np.random.shuffle(data)
    return data
  
#데이터 로드
#학습데이터와 테스트 데이터 나누기
def train_test_data_set(ttt_data, test_rate=0.4): 
    n = int(ttt_data.shape[0]*(1-test_rate)) 
    x_train = ttt_data[:n,:-1]
    y_train = ttt_data[:n, -1] 
    
    x_test = ttt_data[n:,:-1]
    y_test = ttt_data[n:,-1]
    return (x_train, y_train), (x_test, y_test)

    
ttt_data = load_ttt(shuffle=True)
(x_train, y_train), (x_test, y_test) = train_test_data_set(ttt_data, test_rate=0.4)
print("x_train.shape:", x_train.shape) #(574x9)
print("y_train.shape:", y_train.shape) #(574x1)
print("x_test.shape:",  x_test.shape) #(384x9)
print("y_test.shape:",  y_test.shape) #(384x1)
   
y_train = tf.keras.utils.to_categorical(y_train)

y_test= tf.keras.utils.to_categorical(y_test)
print("y_train=", y_train)
print("y_test=", y_test)

#2
n = 10  
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=n, input_dim=9, activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=n, activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
model.summary() 

#3
def MSE(y, t):
    return tf.reduce_mean(tf.square(y - t))

CCE = tf.keras.losses.CategoricalCrossentropy()
#opt = tf.keras.optimizers.RMSprop(learning_rate=0.01)
#opt = tf.keras.optimizers.SGD(learning_rate=0.01)
opt = tf.keras.optimizers.Adam(learning_rate=0.01) 
#model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
#model.compile(optimizer=opt, loss= MSE, metrics=['accuracy'])
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=opt, loss= CCE, metrics=['accuracy'])


ret = model.fit(x_train, y_train, epochs=400, verbose=2) 
print("len(model.layers):", len(model.layers)) 
loss = ret.history['loss'] 
plt.plot(loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
y_pred = model.predict(x_train) 
y_label = np.argmax(y_pred, axis = 1) 
#y_label이랑 y_true랑 비교해서 같으면 맞는거
C = tf.math.confusion_matrix(np.argmax(y_train, axis = 1), y_label)
print("confusion_matrix(C):", C)

