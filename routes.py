import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np

from utils import onehotCategorical

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':

        entered_li = []

        # ========== Part 2.3 ==========
        # YOUR CODE START HERE
        data_month = request.form['month']
        data_promo = request.form['promo']
        data_stateHoliday = request.form['state_holiday']
        data_assortment = request.form['assortment']
        data_store = request.form['store']
        data_dayOfWeek = request.form['day_of_the_week']
        data_promo2 = request.form['promo2']
        data_schoolHoliday = request.form['school_holiday']
        data_storeType = request.form['store_type']

        # get request values


        # one-hot encode categorical variables (only for categorical variables)
        data1 = onehotCategorical(int(data_store), 1112, 0)
        data2 = onehotCategorical(int(data_storeType), 4, 1)
        data3 = onehotCategorical(int(data_assortment), 3, 1)
        data4 = onehotCategorical(int(data_stateHoliday), 4, 1)

        
        # manually specify competition distance
        comp_dist = 5458.1


        # build 1 observation for prediction
        temp1 = np.append(data1,data2)
        temp1 = np.append(temp1,data3)
        temp1 = np.append(temp1,data4)
        temp2 = np.array([comp_dist,int(data_promo2),int(data_promo),int(data_dayOfWeek),int(data_month),int(data_schoolHoliday)])
        entered_li = np.append(temp1,temp2)

        # ========== End of Part 2.3 ==========

        # make prediction
        prediction = model.predict(np.array(entered_li).reshape(1, -1))
        label = str(np.squeeze(prediction.round(2)))

        return render_template('index.html', label=label)

if __name__ == '__main__':
    # load ML model
    # ========== Part 2.2 ==========
    # YOUR CODE START HERE
    model = joblib.load('rm.pkl')
    # ========== End of Part 2.2 ==========
    # start API
    app.run(host='0.0.0.0', port=8000, debug=True)
    #app.run(host='127.0.0.1', port=5000, debug=True)
    #app.run(debug=True)