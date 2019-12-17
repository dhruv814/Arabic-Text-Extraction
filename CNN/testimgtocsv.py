from PIL import Image
import numpy
import os
import PIL.ImageOps
b_new=[]
for i in range(29,43):
 img = Image.open('./dataset/Shiin/Middle/'+str(i)+'.png').convert('L')
 img = img.resize((32,32),Image.ANTIALIAS)
 imgarray = numpy.array(img).T
 b=imgarray.ravel()
 c=numpy.reshape(imgarray,1024)
 bnew = list(b)
 b_new.append(bnew)
#c = numpy.reshape(b, (numpy.product(b.shape),))
 print(imgarray)
print(b_new)
numpy.savetxt('testing.csv',b_new,fmt = '%.18g', delimiter = ' ', newline = os.linesep)

#pd.read_csv('testing.csv').T.to_csv('output.csv',header=False)