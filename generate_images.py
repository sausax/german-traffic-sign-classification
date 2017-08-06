import pickle
import matplotlib.pyplot as plt


f = open('data/valid.p', mode='rb')
img_arr = pickle.load(f)

for idx, img in enumerate(img_arr['features']):
	#print(img)
	plt.imshow(img)
	plt.savefig('imgs/'+str(idx)+'.jpg')

