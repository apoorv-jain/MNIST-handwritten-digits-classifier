import scipy.special
from activations import *
from network import *
import keras
from keras.models import Sequential
from keras.layers import Dense
BATCH_SIZE=10
LEARNING_RATE=0.01
epochs=300
layers=[784,20,10]
# Below function is built just to make a loading bar used
# for making the progress of parsing visual
def loading(actual,upper):
    a=[]
    limit=actual/upper*20
    for i in range(20):
        if i<limit:
            a.append("=")
        else:
            a.append(" ")
    print(a)
# Used to parse the .csv file to use it
# in the training and testing
def parse(file,size):
    print("Parsing the data")
    tic=time.time()
    file=open(file)
    file.readline()
    data_x=[]
    data_y=[]
    for i in range(size):
        if (i % (size/20)) == 0:
            print("PROGRESS")
            loading(i,size)
        new=file.readline()
        line=new.split(",")
        label=(int)(line[0])
        actual_labels=[0,0,0,0,0,0,0,0,0,0]
        actual_labels[label]=1
        pixel=[((int)(i)/255) for i in line[1:]]
        data_y.append(actual_labels)
        data_x.append(pixel)
    data_x=np.array(data_x)
    data_y=np.array(data_y)
    toc=time.time()
    print("Data parsed in "+(str)(toc-tic)+"  Seconds")
    return data_x,data_y
print("LOADING THE TRAINING SET")
x_train,y_train=parse("E:/MNIST/mnist-in-csv/mnist_train.csv",60000)
print("LOADING THE TEST SET")
x_test,y_test=parse("E:/MNIST/mnist-in-csv/mnist_train.csv",10000)
model=Sequential()
model.add(Dense(units=layers[1],activation='sigmoid', input_dim=layers[0]))
model.add(Dense(units=layers[2],activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=LEARNING_RATE,
              momentum=0.9, nesterov=True),metrics=['accuracy'])
print("Training Begins")
tic=time.time()
model.fit(x=x_train,y=y_train,epochs=epochs,batch_size=BATCH_SIZE)
toc=time.time()
print("Training complete in {0} seconds".format(toc-tic))
print(model.evaluate(x_test,y_test))
model.save("model.h5") #saves the model for future use 
