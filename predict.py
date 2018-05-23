
import numpy as np
from KerasHelper import KerasHelper
import data

KerasHelper.log_level_decrease()
# KerasHelper.numpy_show_entire_array(28)


X_pred, X_id, label = data.load_data_predict()

X_pred = X_pred.reshape(X_pred.shape[0], 28, 28, 1)

X_pred = X_pred.astype('float32')
X_pred /= 255


model = KerasHelper.load_model("model")

######################### Predict #########################
predictions = model.predict(X_pred)

# print("OUTPUT")
AllPrediction = []
for i in range(predictions.shape[0]):
	indexMax = np.argmax(predictions[i])
	AllPrediction.append(indexMax)

with open("output", "w+") as file:
	for line in label[:-1]:
		file.write(line + ",")
	else:
		file.write(str(label[-1]))
		file.write("\n")

	for line, id in zip(AllPrediction, X_id):
		file.write(str(id) + "," + str(line) + "\n")
