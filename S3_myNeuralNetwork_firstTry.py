import tensorflow as tf

nn_model=tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu',input_shape=(10,)),     
    tf.keras.layers.Dense(16, activation='relu',input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')     
    ])
nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy',metrics=['accuracy'])     
   

history=nn_model.fit(
    Xtrain, Ytrain, epochs=100, batch_size=50, validation_split=0.2, verbose=0
    )       


# These plot functions are available on internet, at tensorflow pages!
# A good thing about tensorflow is that it keeps track of everything (history). It gives us the opportunity to go over it later and plot it!
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Binary crossentropy')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
# ***********************************************************************   
# plots
plot_loss(history)
plot_accuracy(history)