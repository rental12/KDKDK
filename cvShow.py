from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
import cv2
import time

def readnum(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #img = cv2.resize(img,(28,28))
    #img = cv2.threshold(img, 0, 255,
    #cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cv2.imwrite("processed_iamge.png", img)
    img = np.array(img)
    img = img.reshape(-1, 28,28,1)
    img = np.pad(img,[(0,0),(2,2),(2,2),(0,0)],'constant')
    # img = (img-np.mean(img,axis=(1,2),keepdims=True))
    return img


#cap = cv2.VideoCapture(0)
test='mn/9.png'
test_image = cv2.imread(test)

    

  
    
    
   
interpreter = tf.lite.Interpreter(model_path="mnistmodel.tflite")
test=readnum(test_image)
s = input_data = np.array(test, dtype=np.float32)
start_time = time.clock()
interpreter.allocate_tensors()
interpreter.set_tensor(interpreter.get_input_details()[0]["index"], s)
interpreter.invoke()
output = interpreter.tensor(interpreter.get_output_details()[0]["index"])()[0]
print(time.clock() - start_time, "seconds")
digit = np.argmax(output)
print('Predicted Digit: %d\nConfidence: %f' % (digit, output[digit]))
        
        

# Print the model's classification result
digit = np.argmax(output)
print('Predicted Digit: %d\nConfidence: %f' % (digit, output[digit]))
cv2.imshow("cap",test_image)
    
cv2.destroyAllWindows()
    
    

