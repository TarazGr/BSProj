from model import SiameseNetwork
import data_preprocessing as dp
import os

s = SiameseNetwork()

imgs = []
for img in os.listdir('demo'):
	imgs.append(dp.load_image(os.path.join('demo', img)))

threshold = 6e-7

for i, img in enumerate(imgs):
	matched = False
	scores = s.predict([img * (len(imgs))], imgs)
	for j, score in enumerate(scores):
		if i != j:
			if score >= threshold:
				print("There is a match between", i, "and", j)
				matched = True
	if not matched:
		print("There was no matching identity for", i)
