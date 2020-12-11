# Convolutional Neural Network

# Installing Keras
# pip install --upgrade keras

#Building the CNN

# Importing the Keras libraries and packages
"""from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense"""
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email.encoders import encode_base64
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
im = cv2.imread('17.jpeg')
bbox, label, conf = cv.detect_common_objects(im)
if len(label) == 0:
    print ("It is not a Car")
else:  

    # Initialising the CNN
    #classifier = Sequential()
    
    #we can change the input shape and max pooling shape for better accuracy but the training time will increase.
    
    # Step 1 - Convolution
    #classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
    
    # Step 2 - Pooling
    #classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Adding a second convolutional layer
    #classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    #classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Step 3 - Flattening
    #classifier.add(Flatten())
    
    # Step 4 - Full connection
    #classifier.add(Dense(output_dim = 128, activation = 'relu'))
    #classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
    
    # Compiling the CNN
    #classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #Fitting the CNN to the images
    
    #The below code is from keras documentation
    
    #from keras.preprocessing.image import ImageDataGenerator
    #train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
    
    #test_datagen = ImageDataGenerator(rescale = 1./255)
    
    #training_set = train_datagen.flow_from_directory('dataset/training_set',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
    
    #test_set = test_datagen.flow_from_directory('dataset/test_set',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
    
    #classifier.fit_generator(training_set,samples_per_epoch = 3033,nb_epoch = 3,validation_data = test_set,nb_val_samples = 768)
    
    #classifier.summary()
    
    # Saving weights
    #fname = (r"C:\Users\DADDY'S HOME\Downloads\Major\Damage-car-detection-master\Damage-car-detection-master\dmg_car_detection\dmg_car-weights-CNN.h5")
    #classifier.save(fname)
    
    from keras.models import load_model
    # Loading weights
    fname = (r"C:\Users\DADDY'S HOME\Downloads\Major\Damage-car-detection-master\Damage-car-detection-master\dmg_car_detection\dmg_car-weights-CNN.h5")
    new1 = load_model(fname)
    #classifier.summary()
    
    # Predicting a new image using our CNN model
    import numpy as np
    from keras.preprocessing import image
    
    #Target size should be same as the input shape of the cnn
    test_image = image.load_img(r"C:\Users\DADDY'S HOME\Downloads\Major\Damage-car-detection-master\Damage-car-detection-master\dmg_car_detection\images\17.jpeg", target_size=(64,64,3))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = new1.predict(test_image)
    
    if result[0][0] >= 0.7:
        prediction = 'Car condition is Good'
        print (prediction)
    else:
        prediction = 'Car is Damaged'
    #print (prediction)
    
    #import pyttsx3
    
        def nothing(x):
            pass
        
        image_x, image_y = 64,64
        
        from keras.models import load_model
        classifier = load_model('Trained_model.h5')
        
        def predictor():
               import numpy as np
               from keras.preprocessing import image
               test_image = image.load_img('17.jpeg', target_size=(64, 64))
               test_image = image.img_to_array(test_image)
               test_image = np.expand_dims(test_image, axis = 0)
               result = classifier.predict(test_image)
               
               if result[0][0] == 1:
                      return 'It has DENT'
               elif result[0][1] == 1:
                      return 'It has SCRATCH'
               elif result[0][2] == 1:
                      return 'It has SHATTERED GLASS'
        
               
    
    # creates SMTP session 
    
    
           
        fromaddr = "nishankparasher@gmail.com"
        toaddr = "yashm99999@gmail.com"
           
        # instance of MIMEMultipart 
        msg = MIMEMultipart() 
          
        # storing the senders email address   
        msg['From'] = fromaddr 
          
        # storing the receivers email address  
        msg['To'] = toaddr 
          
        # storing the subject  
        msg['Subject'] = "New Entry"
          
        # string to store the body of the mail 
        body = prediction +". "+ predictor()
         
        # attach the body with the msg instance 
        msg.attach(MIMEText(body, 'plain')) 
          
        # open the file to be sent  
        filenames = ["17.jpeg"]
        for file in filenames:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(open(file, 'rb').read())
            encode_base64(part)
            part.add_header('Content-Disposition', 'attachment; filename="%s"'% os.path.basename(file))
            msg.attach(part)
          
        # attach the instance 'p' to instance 'msg' 
        msg.attach(part)  
        # creates SMTP session 
        s = smtplib.SMTP('smtp.gmail.com', 587)   
        # start TLS for security 
        s.starttls()  
        # Authentication 
        s.login(fromaddr, "nishank1411")  
        # Converts the Multipart msg into a string 
        text = msg.as_string()  
        # sending the mail 
        s.sendmail(fromaddr, toaddr, text) 
        # terminating the session 
        s.quit() 
