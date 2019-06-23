from model import SiameseNetwork
import data_preprocessing as dp
import os
import pickle
import numpy as np

s = SiameseNetwork()
embs = pickle.load(open("tmp/embeddings.pkl", 'rb'))

Arnold_Schwarzenegger = []
for img in os.listdir("demo/Arnold Schwarzenegger"):
	Arnold_Schwarzenegger.append(s.get_embeddings([dp.load_image("demo/Arnold Schwarzenegger/" + img)])[0])

Barack_Obama = []
for img in os.listdir("demo/Barack Obama"):
	Barack_Obama.append(s.get_embeddings([dp.load_image("demo/Barack Obama/" + img)])[0])

Bradley_Cooper = []
for img in os.listdir("demo/Bradley Cooper"):
	Bradley_Cooper.append(s.get_embeddings([dp.load_image("demo/Bradley Cooper/" + img)])[0])

threshold = 6e-7

Arnold_Schwarzenegger_embeddings = embs[48]
Barack_Obama_embeddings = embs[55]
Bradley_Cooper_embeddings = embs[74]


print("Barack Obama says he is Barack Obama")
for emb in Barack_Obama:
	similarity = np.max(s.predict_with_embeddings([emb]*10, Barack_Obama_embeddings))
	if similarity >= threshold:
		print("Yes, you are Barack Obama")
	else:
		print("YOU SHALL NOT PASS!") 

print("Bradley Cooper says he is Bradley Cooper")
for emb in Bradley_Cooper:
	similarity = np.max(s.predict_with_embeddings([emb]*10, Bradley_Cooper_embeddings))
	if similarity >= threshold:
		print("Yes, you are Bradley Cooper")
	else:
		print("YOU SHALL NOT PASS!") 

print("Arnold_Schwarzenegger says he is Bradley Cooper")
for emb in Arnold_Schwarzenegger:
	similarity = np.max(s.predict_with_embeddings([emb]*10, Bradley_Cooper_embeddings))
	print(similarity)
	if similarity >= threshold:
		print("Yes, you are Barack Obama")
	else:
		print("YOU SHALL NOT PASS!") 
