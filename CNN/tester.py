import tensorflow as tf
import numpy as np
import pandas as pd
CATEGORIES =["أ","ب","ت","ث","ج","ح","خ","د","ذ","ر","ز","س","ش","ص","ض","ط","ظ","ع","غ","ف","ق","ك","ل","م","ن","و"
              ,"ؤ","ي"]
from googletrans import Translator
import goslate
translator=Translator()
gs=goslate.Goslate()
model = tf.keras.models.load_model("arabic_classifier_model.h5")
x_test = pd.read_csv("../CNN-2/testing.csv",header=None)
x_test = x_test.values.astype('float32')
x_test = x_test.reshape(-1, 32, 32, 1)
x_test = x_test.astype("float32")/255.
prediction = model.predict(x_test)
pred=np.array(prediction)
final=[]
text=""
translator = Translator()
for i in pred:
    final=[]
    k=0
    p=0
    for j in i:
      k= int(j+0.5)
      if k==1:
        ans=CATEGORIES[p]
        print(p)
        #translatedText = gs.translate(ans, 'eng')
        #print(translatedText)
        text=text+ans
        print(ans)
      final.append(k)
      p+=1
print(text)
translatedText = gs.translate(text,'eng')
print(translatedText)
#translator.translate(text, dest='en')