import pickle
# import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

rfmodel = pickle.load(open('models/rfmodel.pkl', 'rb'))
dtmodel = pickle.load(open('models/dtmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # test = pd.read_csv("test.csv",index_col=0)
    # print(model.predict([[26,4788,48,1,0,1,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0]]))
    base=[0,0,0,0]
    job=[0,0,0]
    housing=[0,0]
    savings=[0,0,0,0]
    checking=[0,0,0]
    purpose=[0,0,0,0,0,0,0]

    base[0]=int(request.form['Age'])
    base[1]=int(request.form['Credit amount'])
    base[2]=int(request.form['Duration'])
    if request.form['Sex_male']=="1":
        base[3]=1
    else:
        base[3]=0
    if int(request.form['Job'])<3:
        job[int(request.form['Job'])]=1
    if int(request.form['Housing'])<2:
        housing[int(request.form['Housing'])]=1
    savings[int(request.form['Saving'])]=1
    checking[int(request.form['Checking'])]=1
    purpose[int(request.form['Purpose'])]=1

    int_features=base+job+housing+savings+checking+purpose

    features = [np.array(int_features)]  
    
    prediction = rfmodel.predict(features) if request.form['Model']=='1' else dtmodel.predict(features)
    
    model="Decision Tree " if request.form['Model']=='0' else "Random Forest "
    output="Low Risk" if prediction[0]==1 else "High Risk"
    return render_template('index.html',model=model,features='['+','.join([str(x) for x in int_features])+']'   , prediction=output)

if __name__ == "__main__":
    app.run(debug=True)


