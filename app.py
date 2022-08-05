import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
from keras.models import load_model

from matplotlib.animation import FuncAnimation


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

st.title('Electricity Theft Detector')

uploaded_file = st.file_uploader("Choose a file (File Type Should be CSV)")


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    dataset = df
    dataset["Month"] = pd.to_datetime(df["Datetime"]).dt.month
    dataset["Year"] = pd.to_datetime(df["Datetime"]).dt.year
    dataset["Date"] = pd.to_datetime(df["Datetime"]).dt.date
    dataset["Time"] = pd.to_datetime(df["Datetime"]).dt.time
    dataset["Week"] = pd.to_datetime(df["Datetime"]).dt.week
    dataset["Day"] = pd.to_datetime(df["Datetime"]).dt.day_name()
    dataset = df.set_index("Datetime")
    dataset.index = pd.to_datetime(dataset.index)
    NewDataSet = dataset.resample('D').mean()
    #NewDataSet.drop(['Month','Year','Week'],axis=1,inplace=True)
    TestDataSet = NewDataSet.tail(200)
    Df_Total = pd.concat((NewDataSet[df.columns[1]], TestDataSet[df.columns[1]]), axis=0)
    inputs = Df_Total[len(Df_Total) - len(TestDataSet) - 60:].values
    inputs = inputs.reshape(-1, 1)
    sc = MinMaxScaler(feature_range=(0, 1))
    sc.fit(inputs)
    inputs = sc.transform(inputs)





    X_test = []
    for i in range(60, 260):
        X_test.append(inputs[i - 60:i])

    # Convert into Numpy Array
    X_test = np.array(X_test)

    # Reshape before Passing to Network
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    model=load_model('ETD_MODEL.h5')
    y_predict=model.predict(X_test)
    y_predict = sc.inverse_transform(y_predict)

    True_MegaWatt = TestDataSet[df.columns[1]].to_list()
    Predicted_MegaWatt = y_predict
    dates = TestDataSet.index.to_list()

    Machine_Df = pd.DataFrame(data={
        "Date": dates,
        "TrueMegaWatt": True_MegaWatt,
        "PredictedMegaWatt": [x[0] for x in Predicted_MegaWatt]
    })

    st.dataframe(Machine_Df)
    True_MegaWatt = TestDataSet[df.columns[1]].to_list()
    Predicted_MegaWatt = [x[0] for x in Predicted_MegaWatt]
    dates = TestDataSet.index.to_list()

    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    x = dates
    y = True_MegaWatt

    y1 = Predicted_MegaWatt

    plt.plot(x, y, color="green",label="True MegaWatt")
    plt.plot(x, y1, color="red",label="Predicted MegaWatt")
    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    plt.xlabel('Dates')
    plt.ylabel("Power in MW")
    plt.title("Machine Learned the Pattern Predicting Future Values ")
    plt.legend()

    st.subheader("Machine Learned the Pattern Predicting Future Values:")

    st.pyplot(fig)

