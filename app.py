# Import necessary libraries
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
import pandas as pd


# Load the classifier from the Pickle file
model_file = open("model.pkl", "rb")
classifier = pickle.load(model_file)

cat_enc_file = open("cat_encoder.pkl", "rb")
cat_encoder = pickle.load(cat_enc_file)

lab_enc_file = open("label_enc.pkl", "rb")
lab_encoder = pickle.load(lab_enc_file)

scaler_file = open("scaler.pkl", "rb")
scaler = pickle.load(scaler_file)


def predict_rating(Age, Gender, State, Genre, Format, Device, Viewing_Frequency, Total_Viewing_Time, Plan, Duration):

    # data = {
    # 	'CustomerID': 1, 'Name': "", 'Age': Age, 'Gender': Gender, 'State': State,
    # 	'Genre': Genre, 'Format': Format, 'Device': Device,
    # 	'Viewing_Frequency': Viewing_Frequency, 'Total_Viewing_Time': Total_Viewing_Time,
    # 	'Plan': Plan, 'Monthly_Cost': "", 'Duration': Duration, 'Rating':Rating
    # }

    data = {
        'Age': [Age], 'Gender': [Gender], 'State': [State], 'Genre': [Genre], 'Format': [Format],
        'Device': [Device], 'Viewing_Frequency': [Viewing_Frequency],
        'Total_Viewing_Time': [Total_Viewing_Time], 'Plan': [Plan], 'Duration': [Duration]
    }

    df = pd.DataFrame(data=data)

    # Categorical features
    cat_features = ['Gender', 'Genre', 'State', 'Format', 'Device', 'Plan']

    #  Numerical features in dataset
    num_features = ['Duration', 'Total_Viewing_Time',
                    'Viewing_Frequency', 'Age']
    # num_features = ['Duration', 'Total_Viewing_Time',
    #                 'Viewing_Frequency']

    # Convert categorical features in dataset to 'category' type
    df[cat_features] = df[cat_features].astype("category")

    # Convert numerical features in dataset to 'int' type
    # df[num_features] = df[num_features].astype("int64")

    # Generate age groups from the age feature
    bins = [18, 31, 49, 60, 125]
    labels = ["Young Adult", "Middle Aged", "Old Adult", "Elderly"]
    df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

    X = df.drop(columns=['Age'])

    # Categorical features
    categorical = X.select_dtypes(include="category").columns

    # Encode the categorical features(columns) and generate appropriate column names
    cat_df = pd.DataFrame(cat_encoder.transform(X[categorical]), index=X.index)
    cat_df.columns = cat_encoder.get_feature_names(categorical)

    # Join the new dataframe to our original data
    X = X.join(cat_df)

    # Drop the categorical columns
    X = X.drop(columns=categorical)

    # Numerical features(columns) in the dataset
    numerical = X.select_dtypes(include="number").columns

    # Tranform numerical data
    # X[numerical] = scaler.transform(X[numerical])
    # X[num_features] = scaler.transform(X[num_features])

    prediction = classifier.predict(X)
    y_class = lab_encoder.inverse_transform(prediction)[0]
    probability = classifier.predict_proba(X)[0]

    print("Prediction: ", prediction)
    print("Class: ", y_class)
    print("Probability: ", probability)

    # return prediction
    return y_class


def main():
    st.title("Mav TV")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Predict Customer Satisfaction </h2>
    </div>
    """
    # Format	Device	Viewing_Frequency	Total_Viewing_Time		Monthly_Cost
    st.markdown(html_temp, unsafe_allow_html=True)
    # Name = st.text_input("Name", "Type Here")
    # Age = st.text_input("Age", "Type Here")
    Age = st.slider("Age", min_value=18, max_value=130, step=1)
    Gender = st.radio("Gender", ('Male', 'Female'))

    # State = st.radio("State", ('Comedy', 'Drama', 'Documentary'))
    State = st.text_input("State", "Type Here")

    Device = st.radio(
        "Device", ('Decoder', 'Tv-App', 'Browser', 'Mobile-App'))
    Format = st.radio("Format", ('TV-Shows', 'Live-Events'))

    Genre = st.radio("Recommended movie genre", (
        'Religion', 'Comedy', 'Sports', 'Music', 'Documentaries', 'News',
        'Cartoons', 'Action', 'Talk-Shows', 'Reality', 'Drama'
    ))

    # Plan = st.radio("Plan", ('Basic', 'Premium', 'Elite'))
    Plan = st.selectbox("Select a plan", ['Basic', 'Premium', 'Elite'])
    Duration = st.text_input("Duration", "Type Here")

    Viewing_Frequency = st.text_input("Viewing Frequency", "Type Here")
    Total_Viewing_Time = st.text_input("Total Viewing Time", "Type Here")

    result = ""
    if st.button("Predict"):
        result = predict_rating(
            Age, Gender, State, Genre, Format, Device, Viewing_Frequency, Total_Viewing_Time, Plan, Duration
        )
    st.success('Subscriber satisfaction is predicted to be: {}'.format(result))


if __name__ == '__main__':
    main()
