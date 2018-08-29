#chuong trinh keras cai tien mo hinh huan luyen bang cach them vao dropdout layer de tong quat hon voi nhung du lieu unseen
#thu nghiem dung cac optimizer khac nhau

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop,Adam
from keras.utils import np_utils
np.random.seed(1671)

#huan luyen mang
NB_EPOCH=20
BATCH_SIZE=128
VERBOSE=1
NB_CLASSES=10 #so lop ngo ra, trong truong hop nay la 10
OPTIMIZER=RMSprop() #optimizer
N_HIDDEN=128
VALIDATION_SPLIT=0.2 #dung 20% cua train lam validation
DROPOUT=0.3
#lay du lieu de huan luyen
(X_train,y_train),(X_test,y_test)=mnist.load_data()
#X_train la 60000 dong 28x28, can reshape ve 60000x784
RESHAPED=784
X_train=X_train.reshape(60000,RESHAPED)
X_test=X_test.reshape(10000,RESHAPED)
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
#chuan hoa du lieu
X_train /=255
X_test /=255
print(X_train.shape[0],'mau train')
print(X_test.shape[0],'Mau test')
Y_train=np_utils.to_categorical(y_train,NB_CLASSES)
Y_test=np_utils.to_categorical(y_test,NB_CLASSES)

#dung model voi 10 ngo ra 
#soft max cho lop cuoi cung
model=Sequential()
model.add(Dense(N_HIDDEN,input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=OPTIMIZER,metrics=['accuracy'])
history=model.fit(X_train,Y_train,batch_size=BATCH_SIZE,epochs=NB_EPOCH,verbose=VERBOSE,validation_split=VALIDATION_SPLIT)

score=model.evaluate(X_test,Y_test,verbose=VERBOSE)
print('Test score:',score[0])
print('Test accuracy:',score[1])

