from flask import Flask, render_template, request
import cv2
from keras.models import load_model
import numpy as np
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing import image, sequence
import cv2
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm
##########
# importing libraries

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage.io
import io

# from scipy.misc import impip save
from imageio import imwrite

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage import data, img_as_float
from skimage import exposure
from PIL import Image, ImageEnhance
from skimage.util import img_as_bool
from skimage import exposure


vocab = np.load('mine_vocab.npy', allow_pickle=True)

vocab = vocab.item()

inv_vocab = {v:k for k,v in vocab.items()}


print("+"*50)
print("vocabulary loaded")


embedding_size = 128
vocab_size = len(vocab)
max_len = 40


image_model = Sequential()

image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))


language_model = Sequential()

language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))


conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, return_sequences=True)(conca)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs = out)

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model.load_weights('mine_model_weights.h5')

print("="*150)
print("MODEL LOADED")

resnet = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')

# resnet = load_model('resnet.h5')

print("="*150)
print("RESNET MODEL LOADED")




app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    
    global model, resnet, vocab, inv_vocab

    img = request.files['file1']

    img.save('static/file.jpg')

    print("="*50)
    print("IMAGE WAVED")

    # initializing MTCNN and InceptionResnetV1 

    mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40) # keep_all=False
    mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # keep_all=True
    # mtcnn = MTCNN(keep_all=True, device=device)
    resnet2 = InceptionResnetV1(pretrained='vggface2').eval()

    # loading data.pt file
    # load_data = torch.load('static/data.pt') 
    load_data = torch.load('static/data.pt') 
    embedding_list = load_data[0] 
    name_list = load_data[1] 

    cam = cv2.VideoCapture('static/file.jpg')
    k=0
    while True:
        ret, frame = cam.read()
        print(ret)
        if frame is not None:
            frame = cv2.resize(frame, (frame.shape[1] * 2,frame.shape[0] *2))

        if not ret:
            print("fail to grab frame, try again")
            break
        
        #!python inference_gfpgan.py -i frame -o frame -v 1.3 -s 2 --bg_upsampler realesrgan
        # cv2_imshow(frame)
        # #Enhance Image Pixels
        # sr = cv2.dnn_superres.DnnSuperResImpl_create()
        # path = "/content/gdrive/MyDrive/FaceRecog/facenet-pytorch/resolutionEnhancerModel/EDSR_x4.pb"
        # sr.readModel(path)
        # sr.setModel("edsr",4)
        # result = sr.upsample(frame)
        # # Resized image
        # resized = cv2.resize(result,dsize=None,fx=4,fy=4)
        # cv2_imshow(resized)
        # dst = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
        # cv2_imshow(dst)
        # contrastInput(dst)
        img = Image.fromarray(frame)
        
        img_cropped_list, prob_list = mtcnn(img, return_prob=True) 
        print(prob_list)
        
        boxes, _ = mtcnn.detect(img)
        print(boxes)
        
        if img_cropped_list is not None:
            boxes, _ = mtcnn.detect(img)
            print(prob_list)        
            for i, prob in enumerate(prob_list):
                if prob>0.90:
                    emb = resnet2(img_cropped_list[i].unsqueeze(0)).detach() 
                    
                    dist_list = [] # list of matched distances, minimum distance is used to identify the person
                    
                    for idx, emb_db in enumerate(embedding_list):
                        dist = torch.dist(emb, emb_db).item()
                        dist_list.append(dist)

                    min_dist = min(dist_list) # get minumum dist value
                    min_dist_idx = dist_list.index(min_dist) # get minumum dist index
                    name = name_list[min_dist_idx] # get name corrosponding to minimum dist
                    
                    box = boxes[i] 
                    
                    original_frame = frame.copy() # storing copy of frame before drawing on it
                    
                    if min_dist<0.85:
                        print(box[0],box[1])
                        if(name == 'unknown'):
                            frame = cv2.rectangle(frame, (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])), (255,0,0), 2)
                            continue
                        frame = cv2.putText(frame, name+' '+str(min_dist), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA, False)
                        
                        crop = frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
                        print((box[0],box[1]) , (box[2],box[3]))
                        print(name)
                        print(int(box[0]),int(box[1]) ,int(box[2]),int(box[3]))
                        cv2.imwrite('static/crop.jpg', crop)   
                        frame = cv2.rectangle(frame, (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])), (255,0,0), 2)
                    
        k = k+1                    
        # cv2.imwrite(f'static/file_{k}.jpg',frame)

        cv2.imwrite('static/file.jpg', frame)

        # frame.save('static/file.jpg')
        # cv2.imshow(frame)
        # cv2_imshow(frame)

        #function not working
        # dir = f'static/file_{i}.jpg'
        # cap = caption(dir)
        # print('c'*11)

        image = cv2.imread('static/file.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (224,224))

        image = np.reshape(image, (1,224,224,3))

        
        
        incept = resnet.predict(image).reshape(1,2048)
        # resnet.config.run_functions_eagerly(True)
        # incept = resnet(image).reshape(1,2048)
        print("="*50)
        print("Predict Features")

        text_in = ['startofseq']

        final = ''

        print("="*50)
        print("GETING Captions")

        count = 0
        while tqdm(count < 20):

            count += 1

            encoded = []
            for i in text_in:
                encoded.append(vocab[i])

            padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1,max_len)

            sampled_index = np.argmax(model.predict([incept, padded]))

            sampled_word = inv_vocab[sampled_index]

            if sampled_word != 'endofseq':
                final = final + ' ' + sampled_word

            text_in.append(sampled_word)
        print(final)
        height, width, layers = frame.shape  
        frame = cv2.putText(frame, final, (51, 51), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0),3, cv2.LINE_AA, False)                
        cv2.imwrite(f'static/file__cap_{k}.jpg',frame)
                

        # print(cap)
        k = cv2.waitKey(1)
        if k%256==27: # ESC
            print('Esc pressed, closing...')
            break
            
        elif k%256==32: # space to save image
            print('Enter your name :')
            name = input()
            
            # create directory if not exists
            if not os.path.exists('photos/'+name):
                os.mkdir('photos/'+name)
                
            img_name = "photos/{}/{}.jpg".format(name, int(time.time()))
            cv2.imwrite(img_name, original_frame)
            print(" saved: {}".format(img_name))
            
            
    cam.release()
    cv2.destroyAllWindows()
    return render_template('detected.html')
    
@app.route('/after', methods=['GET', 'POST'])
def after():

    global model, resnet, vocab, inv_vocab

    image = cv2.imread('static/file.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224,224))

    image = np.reshape(image, (1,224,224,3))

    
    
    incept = resnet.predict(image).reshape(1,2048)
    # resnet.config.run_functions_eagerly(True)
    # incept = resnet(image).reshape(1,2048)
    print("="*50)
    print("Predict Features")


    text_in = ['startofseq']

    final = ''

    print("="*50)
    print("GETING Captions")

    count = 0
    while tqdm(count < 20):

        count += 1

        encoded = []
        for i in text_in:
            encoded.append(vocab[i])

        padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1,max_len)

        sampled_index = np.argmax(model.predict([incept, padded]))

        sampled_word = inv_vocab[sampled_index]

        if sampled_word != 'endofseq':
            final = final + ' ' + sampled_word

        text_in.append(sampled_word)

    return render_template('after.html', data=final)

def caption(dir):
    global model, resnet, vocab, inv_vocab
    
    image = cv2.imread(dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224,224))

    image = np.reshape(image, (1,224,224,3))

    
    
    incept = resnet.predict(image).reshape(1,2048)
    # resnet.config.run_functions_eagerly(True)
    # incept = resnet(image).reshape(1,2048)
    print("="*50)
    print("Predict Features")


    text_in = ['startofseq']

    final = ''

    print("="*50)
    print("GETING Captions")

    count = 0
    while tqdm(count < 20):

        count += 1

        encoded = []
        for i in text_in:
            encoded.append(vocab[i])

        padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1,max_len)

        sampled_index = np.argmax(model.predict([incept, padded]))

        sampled_word = inv_vocab[sampled_index]

        if sampled_word != 'endofseq':
            final = final + ' ' + sampled_word

        text_in.append(sampled_word)
        return final
if __name__ == "__main__":
    app.run(debug=True)

