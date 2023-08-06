from flask import Flask, redirect, render_template, request
from flask_mysqldb import MySQL
from collections import Counter
import csv
import numpy as np
import pandas as pd
data = pd.read_csv("E:\cit assignments\project\instagram_reach (3).csv")
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS 
stopwords = set(STOPWORDS) 
stopwords.add('will')
import re
import seaborn as sns
sns.set()
plt.style.use('seaborn-whitegrid')
def WordCloudPlotter(dfColumn):
    colData = data[dfColumn]
    textCloud = ''
    for mem in colData:
        textCloud = textCloud + str(mem)
    
    # plotting word cloud
    wordcloud = WordCloud(width = 800, height = 800,background_color ='white', 
                          stopwords = stopwords,  min_font_size = 10).generate(textCloud)
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.style.use('seaborn-whitegrid')
    plt.imshow(wordcloud) 
    plt.rcParams.update({'font.size': 25})
    plt.axis("off") 
    plt.title('trending: ' + str(dfColumn))
    plt.tight_layout(pad = 0) 
    plt.show()
def PlotData(features):
    plt.figure(figsize= (10, 10))    
    pltNum = 1
    for mem in features:
        plt.subplot(1, 2 , pltNum)
        plt.style.use('seaborn-whitegrid')
        plt.grid(True)
        plt.title('likes based on'+ str(mem))
        sns.regplot(data = data, x = mem, y = 'Likes' , color = 'green')
        pltNum += 1

    plt.show()
import numpy as np
features = np.array(data[['Followers', 'Time since posted']], dtype = 'float32')
targets = np.array(data['Likes'], dtype = 'float32')
maxValLikes = max(targets)


targets = targets/maxValLikes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(features, targets, test_size = 0.1, random_state = 42)

stdSc = StandardScaler()
xTrain = stdSc.fit_transform(xTrain)
xTest = stdSc.transform(xTest)

from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(xTrain, yTrain)


def PredictionsWithConstantFollowers(model, followerCount, scaller, maxVal):
    followers = followerCount * np.ones(24)
    hours = np.arange(1, 25)
    
    # defining vector 
    featureVector = np.zeros((24, 2))
    featureVector[:, 0] = followers
    featureVector [:, 1] = hours
    
    # doing scalling
    featureVector = scaller.transform(featureVector)
    predictions = model.predict(featureVector)
    predictions = (maxValLikes * predictions).astype('int')
    
    plt.figure(figsize= (10, 10))
    plt.plot(hours, predictions)
    plt.style.use('seaborn-whitegrid')
    plt.scatter(hours, predictions, color = 'g')
    plt.grid(True)
    plt.xlabel('hours since posted')
    plt.ylabel('Likes')
    plt.title('your next post likes may be like this ')
    plt.show()
app = Flask(__name__,static_folder="static")

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'sriram123'
app.config['MYSQL_DB'] = 'project'
mysql = MySQL()
mysql.init_app(app)
@app.route('/')
def main():
    return render_template('login.html')
@app.route('/',methods=['POST'])
def check():
    username=request.form['u']
    password=request.form['p']
    con=mysql.connection.cursor()
    cursor=mysql.connect.cursor()
    cursor.execute("select USERNAME,PASSWORD from instagrampro where USERNAME='"+ username +"'and PASSWORD='"+ password +"'")
    data=cursor.fetchone()
    if data is None:
        return "username or password is wrong"
    else:
        con.execute('insert into temp (username,password) value(%s,%s)',(username,password))
        mysql.connection.commit()
        return redirect('/sign')
@app.route('/sign',methods=['POST','GET'])
def contact():
    value = request.form
    print(value)
    if "TOP10" in request.form:
        return redirect('/TOP10')
    if "TRENDING" in request.form:
        return redirect('/hash')
    if "LOGOUT" in request.form:
        return redirect('/logout')
    if "VIEW" in request.form:
        return redirect('/view')
    if "PREDICT" in request.form:
        return redirect('/predict')
    if request.method == "POST":
        first_name = request.form.get("firstname")
        with open("E:\cit assignments\project\hashtags.csv", 'r') as file:
            reader = csv.reader(file)
            peoples = []
            for row in reader:
                peoples.append(row)
        a=[]
        for i in range(1,len(peoples)):
            a.append(peoples[i])
        b=[]
        for i in a:
            b=b+i
        split_sentences = [sentence.split("#") for sentence in b]
        c=[]
        for t in split_sentences:
            c+=t
        frequency = Counter(c)
        sorted_frequency = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        h=[]
        for e in range(2,12):
            h.append(sorted_frequency[e])
        d=[]
        for i in h:
            d.append(i[0])
        if first_name in d:
            return "this tag is in trending if you post this you will get more likes"
        else:
            return "this tag is not in trending if you post this you will not get much likes"
    return render_template('webpage.html')
@app.route('/TOP10')
def top():
    with open("E:\cit assignments\project\Book1.csv", 'r') as file:
        reader = csv.reader(file)
        peoples = []
        for row in reader:
            peoples.append(row)
    people=[]
    u=[]
    for i in peoples:
        u.append(i[0])
        u.append(int(i[1]))
        people.append(u)
        u=[]
    for i in range(1, len(people)):
        j = i
        while j > 0 and people[j-1][1] > people[j][1]:
            people[j-1], people[j] = people[j], people[j-1]
            j -= 1
    a=[]
    for i in range(2,12):
        a.append(people[-i])
    return render_template('top.html',a=a)

@app.route('/view')
def view():
    PlotData(['Followers', 'Time since posted'])
    return render_template('predict.html')
@app.route('/predict')
def predict():
    cur = mysql.connection.cursor()
    cur.execute("SELECT Followers FROM instagrampro where USERNAME=any (select username from temp)")
    info = cur.fetchone()
    my_result = int(''.join(map(str, info)))

    if my_result<500:
        PredictionsWithConstantFollowers(gbr, 500, stdSc, maxValLikes)
    else:
        PredictionsWithConstantFollowers(gbr, 1000, stdSc, maxValLikes)
    return render_template('about.html')
@app.route('/logout')
def logout():
    con=mysql.connection.cursor()
    con.execute('truncate table temp')
    mysql.connection.commit()
    return redirect('/')
@app.route('/hash',methods=['post','get'])
def log():
    WordCloudPlotter('Hashtags')
    with open("E:\cit assignments\project\hashtags.csv", 'r') as file:
        reader = csv.reader(file)
        peoples = []
        for row in reader:
            peoples.append(row)
    a=[]
    for i in range(1,len(peoples)):
        a.append(peoples[i])
    b=[]
    for i in a:
        b=b+i
    split_sentences = [sentence.split("#") for sentence in b]
    c=[]
    for t in split_sentences:
        c+=t
    frequency = Counter(c)
    sorted_frequency = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
    h=[]
    for e in range(2,12):
        h.append(sorted_frequency[e])
    d=[]
    for i in h:
        d.append(i[0])
    return render_template('top10.html',d=d)

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.run(debug=True)