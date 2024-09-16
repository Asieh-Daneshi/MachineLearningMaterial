import tensorflow as tf

# here I define a function that trains my network
def train_model(Xtrain, Ytrain, num_nodes, dropout_prob, lr, batch_size, epochs):       # lr is the learning rate
    nn_model=tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes, activation='relu',input_shape=(10,)),     # "Dense" means that all neurons are connected, number of units=32, "input_shape" is the number of input nodes
        tf.keras.layers.Dropout(dropout_prob),      # "Dropout" randomly chooses some nodes with "dropout_prob" probability and doesn't include them in the training.
        tf.keras.layers.Dense(num_nodes, activation='relu',input_shape=(10,)),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation='sigmoid')      # "output layer" is just one node and the activation function is sigoid
        ])
    
    # in tensorflow, after building the network, we have to compile it!
    nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy',metrics=['accuracy'])   
    # default learning rate is 0.001, which we are also using
    # here, we use "Adam" optimizer. However, there are lots of optimizers.
    # for metrics, the NN by deafult computes the "loss". When we add "accuracy", later we can plot both of them to see how our network works.
    
    
    
    # here, we train our model. "history" is all the stuff that our network went through during training!
    history=nn_model.fit(
        Xtrain, Ytrain, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0
    )       # "validation_split" determines the fraction of the training data that will be used for validation
            # when we put "verbose=0" it means that we don't want to see whole training process, we just want to see the output!
    return nn_model, history

# *****************************************************************************
# this function is available on internet for free
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
    # loss is the loss on the training data, and val_loss is the loss on the validation data
    ax1.plot(history.history['loss'], label='loss')     # on the left side, we plot loss during training phase. Remember that loss is computed be default
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary crossentropy')       # the loss function in line 14 was binary crossentropy
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    plt.show()
# ***********************************************************************   
# plots
# plot_history(history)


# ***********************************************************************
# in this block, we model our network for different number of nodes, dropout probability, learning rate, and batch size
least_val_loss=float('inf')
least_loss_model=None
epochs=100
for num_nodes in [16,32,32]:
    for dropout_prob in [0, 0.1]:
        for lr in [0.1, 0.005, 0.001]:
            for batch_size in [20, 50, 100]:
                # For each set of parameters, we want to see number of nodes, droupout probability, learning rate, and batch size
                print(f" {num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}")
                # When we put f  a string, it means that whatever comes in front of it in {} must be replaced by its value. 
                # For example, if I write f"result", the output will be "result", but if I write f"{result}", the output will be numerical values of the result!
                model, history = train_model(Xtrain, Ytrain, num_nodes, dropout_prob, lr, batch_size, epochs)
                # plot_loss(history)
                # plot_loss(accuracy)
                plot_history(history)
                # here, we do whatever we did for training set, now for validation set
                val_loss=model.evaluate(Xvalid, Yvalid)
                # in this part, I am just finding the least validation loss that my model can achieve
                model_loss=val_loss[0]
                if model_loss < least_val_loss:
                    least_val_loss =model_loss
                    least_loss_model=model
                
                
Ypred=least_loss_model.predict(Xtest)       # since we used sigmoid function in the output layer, "Ypred" values are between 0 and 1
Ypred=(Ypred > 0.5).astype(int)         # here, I just binarize the output values. Meaning that I pick Ypred values larger than 0.5 and convert them into integers. Then, the output will be 0s and 1s.
Ypred=Ypred.reshape(-1,)            # here we convert our Ypred into a 1-dimensional array

# here we can see the classification report of our NN like other classification methods that we discussed, and compare its accuracy with previous models
from sklearn.metrics import classification_report
print(classification_report(Ytest, Ypred))