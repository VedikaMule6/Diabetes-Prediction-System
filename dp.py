from tkinter import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

#load the data
data = pd.read_csv("dbsep23.csv")

#handle null data
data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NAN)

data.fillna({"Glucose": data["Glucose"].mean(),
              "BloodPressure": data["BloodPressure"].mean(),
              "SkinThickness": data["SkinThickness"].mean(),
              "Insulin": data["Insulin"].mean(),
              "BMI": data["BMI"].mean()}, inplace=True)

#divide into features and target
features = data.drop("Outcome", axis="columns")
target = data["Outcome"]

mms = MinMaxScaler()
nfeatures = mms.fit_transform(features)

k = int(len(data) ** 0.5)
if k % 2 == 0:
    k = k + 1
model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
model.fit(nfeatures, target)

#function to predict diabetes
def predict_diabetes():
    try:
        pregnancies = float(ent_pregnancies.get())
        glucose = float(ent_glucose.get())
        bp = float(ent_bp.get())
        st = float(ent_skin_thickness.get())
        insulin = float(ent_insulin.get())
        bmi = float(ent_bmi.get())
        dpf = float(ent_dpf.get())
        age = float(ent_age.get())
    except ValueError:
        result_label.configure(text="Incorrect input")
        return

    input_data = [[pregnancies, glucose, bp, st, insulin, bmi, dpf, age]]
    input_data_scaled = mms.transform(input_data)
    prediction = model.predict(input_data_scaled)

    if prediction[0] == 0:
        result_label.configure(text="No (Not Diabetic)")
    else:
        result_label.configure(text="Yes (Diabetic)")

#tkinter app
root = Tk()
root.title("Diabetes Prediction App")
root.geometry("500x600")

font_large = ("Impact", 13)

lab_header = Label(root, text="Diabetes Predictor in Women", font=font_large)
lab_header.pack(pady=30)

Label(root, text="Pregnancies:", font=font_large).pack()
ent_pregnancies = Entry(root, font=font_large)
ent_pregnancies.pack()

Label(root, text="Glucose:", font=font_large).pack()
ent_glucose = Entry(root, font=font_large)
ent_glucose.pack()

Label(root, text="Blood Pressure:", font=font_large).pack()
ent_bp = Entry(root, font=font_large)
ent_bp.pack()

Label(root, text="Skin Thickness:", font=font_large).pack()
ent_skin_thickness = Entry(root, font=font_large)
ent_skin_thickness.pack()

Label(root, text="Insulin:", font=font_large).pack()
ent_insulin = Entry(root, font=font_large)
ent_insulin.pack()

Label(root, text="BMI:", font=font_large).pack()
ent_bmi = Entry(root, font=font_large)
ent_bmi.pack()

Label(root, text="Diabetes Pedigree Function:", font=font_large).pack()
ent_dpf = Entry(root, font=font_large)
ent_dpf.pack()

Label(root, text="Age:", font=font_large).pack()
ent_age = Entry(root, font=font_large)
ent_age.pack()

predict_button = Button(root, text="Predict", font=font_large, command=predict_diabetes)
predict_button.pack()

result_label = Label(root, text="", font=font_large)
result_label.pack()

root.mainloop()
