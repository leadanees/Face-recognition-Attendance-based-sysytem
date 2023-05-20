import cv2
import os
from flask import Flask,request,render_template,redirect, url_for, session,flash
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import shutil
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
from apscheduler.schedulers.background import BackgroundScheduler
import matplotlib.pyplot as plt
import csv
import mysql.connector
import geocoder

#getting the current location
global saboloc
g = geocoder.ip('me')
saboloc=g.latlng  ## this give latitude and longitude of current location
print(saboloc)

#making the database
mydb = mysql.connector.connect(host="localhost",user="root",password="")
mycursor = mydb.cursor()
mycursor.execute("CREATE DATABASE IF NOT EXISTS pythonlogin")

#all global running functions here
def sensor():
    print("Scheduler is alive!")
    #keeping the location in check
    saboloc=g.latlng  ## this give latitude and longitude of current location
    print(saboloc)
    #making new  csvfile function
    checktime()
    
sched = BackgroundScheduler(daemon=True)
sched.add_job(sensor,'interval',seconds=10)
sched.start()

#### Defining Flask App
app = Flask(__name__)

b=0
#### Saving Date today in 2 different formats
min=datetime.now().strftime("%M")
datetoday = date.today().strftime("%m_%d_%y")+"_"+str(min) #04_03_23_<built-in function min>
datetoday2 = date.today().strftime("%d-%B-%Y") #03-April-2023
datetoday3 = date.today().strftime("_%Y_%m_%d") # _2023_04_23
main_time=int(min)

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')
# with open(f'Attendance.csv','w') as g:
#         g.write('Roll,Name')

#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

#### extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')

#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l

##adding new file
def checktime():
    global datetoday
    global main_time
    global b
    b=0
    curr_time=int(datetime.now().strftime("%M"))
    if(curr_time>=main_time):
        main_time=int(main_time)+1
        if(main_time>=60):
            main_time=60-main_time
        datetoday=date.today().strftime("%m_%d_%y")+"_"+str(main_time)
        if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
            #beforermdir()
            rmdir()
            with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
                f.write('Name,Roll,Time')

#### Add Attendance of a specific user
def add_attendance(name):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
    account = cursor.fetchone()
    # passing the user prn here
    userprn1=str(account["prn"])

    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    if(userprn1==userid):
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        if int(userid) not in list(df['Roll']):
            with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
                f.write(f'\n{username},{userid},{current_time}')

################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def index():
    names,rolls,times,l = extract_attendance()    
    return render_template('index.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 

#### This function will run when we click on Take Attendance Button
#after loggedin
@app.route('/start',methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html',totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder. Please add a new face to continue.') 
    #matching the location of the student
    if(saboloc[0]==19.0931 and saboloc[1]==72.9049):
    #if(saboloc[0]==19.0728 and saboloc[1]==72.8826):#my home location
        print("its in saboo")
        cap = cv2.VideoCapture(0)
        ret = True
        while ret:
            ret,frame = cap.read()
            if extract_faces(frame)!=():
                (x,y,w,h) = extract_faces(frame)[0]
                cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
                face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
                identified_person = identify_face(face.reshape(1,-1))[0]
                add_attendance(identified_person)
                cv2.putText(frame,f'{"face detected"}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            cv2.imshow('Attendance',frame)
            if cv2.waitKey(1)==27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("its not in saboo")
        return redirect(url_for('home'))
    # names,rolls,times,l = extract_attendance()   
    # return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 
    return redirect(url_for('home')) 

# before login
@app.route('/start1',methods=['GET'])
def start1():
       return redirect(url_for('login')) 

#### This function will run when we add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names,rolls,times,l = extract_attendance()
    return render_template('login.html')    
    #return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 

##remove dir
def rmdir():
    source_folder = r"Attendance\\"
    destination_folder = r"newattendence\\"
    # fetch all files transfer and delete
    for file_name in os.listdir(source_folder):
        # construct full file path
        source = source_folder + file_name
        destination = destination_folder + file_name
        mail(source)
        insertdata(source)
        # copy only files
        if os.path.isfile(source):
            # send the data to sql before dropping the file
            shutil.copy(source, destination)
            print('copied', file_name)
            os.remove(source)

# def beforermdir():
#     source_folder = r"Attendance\\"
#     for file_name in os.listdir(source_folder):
#         # construct full file path
#         source = source_folder + file_name
#     # clearing the attendance.csv
#     filename = "Attendance.csv"
#     # opening the file with w+ mode truncates the file
#     f = open(filename, "w+")
#     f.write('Roll,Name')
#     f.close()
#     # read specific columns of csv file using Pandas
#     df = pd.read_csv(source,usecols =['Roll','Name'])
#     df1 = pd.read_csv('Attendance.csv')
#     #concatting the files
#     df1 = pd.concat([df1,df],ignore_index = True)
#     #putting the concated file into attendance.csv for attendance
#     df1.to_csv('Attendance.csv',index=False) 

#mail function
def mail(g):
    fromaddr = "saadvadanagara786@gmail.com"
    toaddr = "saadvadanagara@gmail.com"
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "SUBJECT OF THE EMAIL"
    body = "TEXT YOU WANT TO SEND"
    msg.attach(MIMEText(body, 'plain'))
    filename = g
    # filename = "Attendance\Attendance-02_23_23_43.csv"
    attachment = open(filename, "rb")
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    msg.attach(part)
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "qvxkmkrexkobhflh")
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()
    print("email sended")

#Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = '1a2b3c4d5e'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'pythonlogin'

# Intialize MySQL
mysql = MySQL(app)

#making the table for accounts
data_accounts = MySQLdb.connect (host="localhost" , user="root" , passwd="" ,db="pythonlogin")
cursor = data_accounts.cursor()
Student_accounts= ("CREATE TABLE IF NOT EXISTS accounts(id int NOT NULL AUTO_INCREMENT,username varchar(50) Not Null,password varchar(255) Not Null,email varchar(100) Not Null,PRIMARY KEY (id),prn int)")
cursor.execute(Student_accounts)

# http://localhost:5000/ - the following will be our login page, which will use both GET and POST requests
@app.route('/login', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password,))
        # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    # Show the login form with message (if any)
    return render_template('login.html', msg=msg)

# http://localhost:5000/logout - this will be the logout page
@app.route('/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('index'))

# http://localhost:5000/register - this will be the registration page, we need to use both GET and POST requests
@app.route('/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'prn' in request.form and'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        prn = request.form["prn"]
        password = request.form['password']
        email = request.form['email']
                # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email or not prn:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s,%s)', (username, password, email,prn))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
            return render_template('newuser.html',totalreg=totalreg(),datetoday2=datetoday2)
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)

# # http://localhost:5000/home - this will be the home page, only accessible for loggedin users
@app.route('/home')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        names,rolls,times,l = extract_attendance()    
        return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2,username=session['username']) 
        #return render_template('home.html', username=session['username'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

# http://localhost:5000/profile - this will be the profile page, only accessible for loggedin users
@app.route('/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

#all the backend schema
#use the attendance.csv to put data to sql if you use beforermdir
# send data to sql file
# brcount=0 # keep 0 if u want to make the table or keep 1


# making initial student database
database = MySQLdb.connect (host="localhost" , user="root" , passwd="" ,db="pythonlogin")
cursor = database.cursor()
cursor.execute("SHOW TABLES LIKE '%student%';")#check if there is any table named student
a=cursor.fetchall()
#if there is no table then lenth is 0


if len(a)==0:
    rollno=[]
    c=0
    for i in range(76):
        if(c==0):
            c=1
            continue
        if i<10:
            i='0'+str(i)
        b='61190'+str(i)
        rollno.append(b)

    database = MySQLdb.connect (host="localhost" , user="root" , passwd="" ,db="pythonlogin")
    cursor = database.cursor()
    #making the table for students
    Student_table= ("CREATE TABLE IF NOT EXISTS Student(PRN int,PRIMARY KEY (PRN),CC INT,BC INT,PM INT,EM INT,BDA INT)")
    cursor.execute(Student_table)
    default_insert="insert into Student values(%s,%s,%s,%s,%s,%s)"
    
    Student_table_cc=("CREATE TABLE if not EXISTS cc(RollNo int,PRIMARY KEY(RollNo),"+datetoday3+" int)")
    cursor.execute(Student_table_cc)
    Student_table_bc=("CREATE TABLE if not EXISTS bc(RollNo int,PRIMARY KEY(RollNo),"+datetoday3+" int)")
    cursor.execute(Student_table_bc)
    Student_table_em=("CREATE TABLE if not EXISTS em(RollNo int,PRIMARY KEY(RollNo),"+datetoday3+" int)")
    cursor.execute(Student_table_em)
    Student_table_pm=("CREATE TABLE if not EXISTS pm(RollNo int,PRIMARY KEY(RollNo),"+datetoday3+" int)")
    cursor.execute(Student_table_pm)
    Student_table_bda=("CREATE TABLE if not EXISTS bda(RollNo int,PRIMARY KEY(RollNo),"+datetoday3+" int)")
    cursor.execute(Student_table_bda)
    
    for i in rollno:
        cursor.execute(default_insert,(i,0,0,0,0,0))
        database.commit()
        cursor.execute("insert into cc values("+i+",0)")
        database.commit()
        cursor.execute("insert into bc values("+i+",0)")
        database.commit()
        cursor.execute("insert into em values("+i+",0)")
        database.commit()
        cursor.execute("insert into pm values("+i+",0)")
        database.commit()
        cursor.execute("insert into bda values("+i+",0)")
        database.commit()
    print("data inserted")


# time system for subjects
now=datetime.now()
Stimecc=now.replace(hour=00,minute=1,second=0,microsecond=0);
Etimecc=now.replace(hour=10,minute=59,second=0,microsecond=0);
Stimebc=now.replace(hour=11,minute=0,second=0,microsecond=0);
Etimebc=now.replace(hour=11,minute=59,second=0,microsecond=0);
Stimeem=now.replace(hour=12,minute=0,second=0,microsecond=0);
Etimeem=now.replace(hour=12,minute=59,second=0,microsecond=0);
Stimepm=now.replace(hour=14,minute=0,second=0,microsecond=0);
Etimepm=now.replace(hour=14,minute=59,second=0,microsecond=0);
Stimebda=now.replace(hour=15,minute=0,second=0,microsecond=0);
Etimebda=now.replace(hour=22,minute=59,second=0,microsecond=0);

cctotal=0
bctotal=0
bdatotal=0
emtotal=0
pmtotal=0

#adding data to data bases
def insertdata(datafile):
    chknewday()
    global cctotal,bdatotal,bctotal,emtotal,pmtotal,se
  
    # opening the CSV file
    with open(datafile, mode ='r')as file:
        database = MySQLdb.connect (host="localhost" , user="root" , passwd="" ,db="pythonlogin")
        cursor = database.cursor()
        # insert_query="UPDATE student SET "+se+"="+c+" WHERE PRN="+i[1]
        # retrive_query="select "+se+" from Student where prn="+i[1]
        csvFile = csv.reader(file)

        #for count
        if(now>Stimecc and now<Etimecc):
            se="cc"
            # cctotal+=1 error
            cctotal = cctotal+1
        if(now>Stimebc and now<Etimebc):
            se="bc"
            bctotal=bctotal+1
        if(now>Stimepm and now<Etimepm):
            se="pm"
            pmtotal=pmtotal+1
        if(now>Stimeem and now<Etimeem):
            se="em"
            emtotal=emtotal+1
        if(now>Stimebda and now<Etimebda):
            se="bda"
            bdatotal=bdatotal+1
        ch=0
        for i in csvFile:
            if(ch==0):
                ch=1
                continue
            #for other tables
            if(now>Stimecc and now<Etimecc):
                cursor.execute("UPDATE cc SET "+datetoday3+"= 1 WHERE RollNo="+i[1])
            if(now>Stimebc and now<Etimebc):
                cursor.execute("UPDATE bc SET "+datetoday3+"= 1 WHERE RollNo="+i[1])
            if(now>Stimepm and now<Etimepm):
                cursor.execute("UPDATE pm SET "+datetoday3+"= 1 WHERE RollNo="+i[1])
            if(now>Stimeem and now<Etimeem):
                cursor.execute("UPDATE em SET "+datetoday3+"= 1 WHERE RollNo="+i[1])
            if(now>Stimebda and now<Etimebda):
                cursor.execute("UPDATE bda SET "+datetoday3+"= 1 WHERE RollNo="+i[1])
            #for student table
            retrive_query="select "+se+" from Student where prn="+i[1]
            cursor.execute(retrive_query)
            a=cursor.fetchall()
            print(a)  #((0,),())
            a=(int(a[0][0]))
            c=a+1
            insert_query="UPDATE student SET "+se+"="+str(c)+" WHERE PRN="+i[1]
            cursor.execute(insert_query)
            database.commit()
        print(" student data inserted")
        
def chknewday():
    global datetoday3
    #for testing new day uncomment below line
    # dat="_02_04_2923"
    dat=date.today().strftime("_%Y_%m_%d")
    if(str(datetoday3)!=str(dat)):
        database = MySQLdb.connect (host="localhost" , user="root" , passwd="" ,db="pythonlogin")
        cursor = database.cursor()
        datetoday3=dat
        cursor.execute("ALTER TABLE CC ADD "+dat+" int DEFAULT 0;")
        cursor.execute("ALTER TABLE bc ADD "+dat+" int DEFAULT 0;")
        cursor.execute("ALTER TABLE em ADD "+dat+" int DEFAULT 0;")
        cursor.execute("ALTER TABLE pm ADD "+dat+" int DEFAULT 0;")
        cursor.execute("ALTER TABLE bda ADD "+dat+" int DEFAULT 0;")
        print("new data inserted")

# making the backend for analytics data

#most important page core of the project

@app.route('/analysis')
def analysis():
    global userprn
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        # passing the user id here
        adminuser=str(account["username"])
        userprn=str(account["prn"])
        # getting all the user attendance from the sql data base
        if(adminuser == "admin"):
            rollno=[]
            c=0
            for i in range(76):
                if(c==0):
                    c=1
                    continue
                if i<10:
                    i='0'+str(i)
                b='61190'+str(i)
                rollno.append(b)
            acc=0
            abda=0
            abc=0
            apm=0
            aem=0
            att=0
            for i in rollno:
                a=totalattend(i)
                acc=acc+a[0]
                abda=abda+a[1]
                abc=abc+a[2]
                apm=apm+a[3]
                aem=aem+a[4]
                att=att+a[5]

            data = {'Task' : 'Hours per Day', 'attended' : acc, 'Not attended' : cctotal*75}
            data1 = {'Task' : 'Hours per Day', 'attended' : abda, 'Not attended' : bdatotal*75}
            data2= {'Task' : 'Hours per Day', 'attended' : abc, 'Not attended' : bctotal*75}
            data3 = {'Task' : 'Hours per Day', 'attended' : apm, 'Not attended' : pmtotal*75}
            data4 = {'Task' : 'Hours per Day', 'attended' : aem, 'Not attended' : emtotal*75}
            mtt=cctotal+bctotal+bdatotal+emtotal+pmtotal
            data5 = {'Task' : 'Hours per Day', 'attended' : att, 'Not attended' : mtt*75}
            return render_template('analysisadmin.html', account=account, data=data,data1=data1,data2=data2,data3=data3,data4=data4,data5=data5)
        
        else:
            a=getattend()
            # making data for every piechart
            data = {'Task' : 'Hours per Day', 'attended' : a[0], 'Not attended' : cctotal}
            data1 = {'Task' : 'Hours per Day', 'attended' : a[1], 'Not attended' : bdatotal}
            data2= {'Task' : 'Hours per Day', 'attended' : a[2], 'Not attended' : bctotal}
            data3 = {'Task' : 'Hours per Day', 'attended' : a[3], 'Not attended' : pmtotal}
            data4 = {'Task' : 'Hours per Day', 'attended' : a[4], 'Not attended' : emtotal}
            mtt=cctotal+bctotal+bdatotal+emtotal+pmtotal
            data5 = {'Task' : 'Hours per Day', 'attended' : a[5], 'Not attended' : mtt}
            return render_template('analysis.html', account=account, data=data,data1=data1,data2=data2,data3=data3,data4=data4,data5=data5)
    
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

processed_text = "61190"
@app.route('/adminanalysis')
def adminanalysis():
    if 'loggedin' in session:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        # passing the  here
        adminuser=str(account["username"])
        global userprn
        userprn=str(account["prn"])
        if(adminuser == "admin"):
            return render_template('adminanalysis.html')
        else:
            return render_template('studentsee.html')
             
    return redirect(url_for('home'))

@app.route('/adminanalysis', methods =["GET", "POST"])
def adminanalysis_form():
    global processed_text

    if request.form['form'] == 'form1':
        text = request.form['text']
        processed_text = text.upper()
        print(processed_text)
        a=admingetattend()
        # making data for every piechart
        data = {'Task' : 'Hours per Day', 'attended' : a[0], 'Not attended' : cctotal}
        data1 = {'Task' : 'Hours per Day', 'attended' : a[1], 'Not attended' : bdatotal}
        data2= {'Task' : 'Hours per Day', 'attended' : a[2], 'Not attended' : bctotal}
        data3 = {'Task' : 'Hours per Day', 'attended' : a[3], 'Not attended' : pmtotal}
        data4 = {'Task' : 'Hours per Day', 'attended' : a[4], 'Not attended' : emtotal}
        mtt=cctotal+bctotal+bdatotal+emtotal+pmtotal
        data5 = {'Task' : 'Hours per Day', 'attended' : a[5], 'Not attended' : mtt}
        return render_template('adminsee.html',data=data,data1=data1,data2=data2,data3=data3,data4=data4,data5=data5)

    if request.form['form'] == 'form2':
        text = request.form['text']
        processed_text = text.upper()
        startdate = request.form['date1']
        lastdate = request.form['date2']
        a=tuseranalytic(startdate,lastdate)
        data = {'Task' : 'Hours per Day', 'attended' : a[0], 'Not attended' : a[5]}
        data1 = {'Task' : 'Hours per Day', 'attended' : a[1], 'Not attended' : a[5]}
        data2= {'Task' : 'Hours per Day', 'attended' : a[2], 'Not attended' : a[5]}
        data3 = {'Task' : 'Hours per Day', 'attended' : a[3], 'Not attended' : a[5]}
        data4 = {'Task' : 'Hours per Day', 'attended' : a[4], 'Not attended' : a[5]}
        data5={'Task' : 'total attended', 'attended' : a[6], 'Not attended' : (a[5]*5)}
        return render_template('adminsee.html',data=data,data1=data1,data2=data2,data3=data3,data4=data4,data5=data5)

    if request.form['form'] == 'form3':
        startdate = request.form['date1']
        lastdate = request.form['date2']
        acc=0
        abda=0
        abc=0
        apm=0
        aem=0
        att=0
        tta=0
        rollno=[]
        c=0
        for i in range(76):
            if(c==0):
                c=1
                continue
            if i<10:
                i='0'+str(i)
            b='61190'+str(i)
            rollno.append(b)
        for i in rollno:
            k=i
            a=totaltuseranalytic(k,startdate,lastdate)
            acc=acc+a[0]
            abda=abda+a[1]
            abc=abc+a[2]
            apm=apm+a[3]
            aem=aem+a[4]
            att=att+a[5]
            tta=tta+a[6]
        data = {'Task' : 'Hours per Day', 'attended' : acc, 'Not attended' : att}
        data1 = {'Task' : 'Hours per Day', 'attended' : abda, 'Not attended' : att}
        data2= {'Task' : 'Hours per Day', 'attended' : abc, 'Not attended' : att}
        data3 = {'Task' : 'Hours per Day', 'attended' : apm, 'Not attended' : att}
        data4 = {'Task' : 'Hours per Day', 'attended' : aem, 'Not attended' : att}
        data5={'Task' : 'total attended', 'attended' : tta, 'Not attended' : att}
        return render_template('adminsee.html',data=data,data1=data1,data2=data2,data3=data3,data4=data4,data5=data5)
    
    if request.form['form'] == 'form4':
        startdate = request.form['date1']
        lastdate = request.form['date2']
        a=totaltuseranalytic(userprn,startdate,lastdate)
        data = {'Task' : 'Hours per Day', 'attended' : a[0], 'Not attended' : a[5]}
        data1 = {'Task' : 'Hours per Day', 'attended' : a[1], 'Not attended' : a[5]}
        data2= {'Task' : 'Hours per Day', 'attended' : a[2], 'Not attended' : a[5]}
        data3 = {'Task' : 'Hours per Day', 'attended' : a[3], 'Not attended' : a[5]}
        data4 = {'Task' : 'Hours per Day', 'attended' : a[4], 'Not attended' : a[5]}
        data5={'Task' : 'total attended', 'attended' : a[6], 'Not attended' : (a[5]*5)}
        return render_template('adminsee.html',data=data,data1=data1,data2=data2,data3=data3,data4=data4,data5=data5)
    
#main functions
def totaltuseranalytic(k,startdate,lastdate):
    sinp=startdate
    einp=lastdate
    # start date division
    sday=sinp[8:10]
    smonth=sinp[5:7]
    syear=sinp[0:4]
    # end date division
    eday=einp[8:10]
    emonth=einp[5:7]
    eyear=einp[0:4]
    #making dates
    dates=[]
    if(int(smonth)==int(emonth)):
        print("if executed")
        for i in range(int(sday),(int(eday)+1)):
            if i<10:
                a="0"+str(i)
                dates.append("_"+syear+"_"+smonth+"_"+a)
            else:
                dates.append("_"+syear+"_"+smonth+"_"+str(i))
    else:
        print("else executeed")
        for i in range(int(smonth),int(emonth)+1):
            if(int(smonth)==i):
                print("same start month")
                st = pd.Timestamp(int(syear), int(i), int(sday))
                for j in range(int(sday),(st.daysinmonth+1)):
                    if j<10:
                        a="0"+str(j)
                        if i<10:
                            dates.append("_"+syear+"_"+aa+"_"+a)
                        else:
                            dates.append("_"+syear+"_"+str(i)+"_"+a)
                    else:
                        if i<10:
                            aa="0"+str(i)
                            dates.append("_"+syear+"_"+aa+"_"+str(j))
                        else:
                            dates.append("_"+syear+"_"+str(i)+"_"+str(j))
            elif(i==int(emonth)):
                print("same end month")
                for k in range(1,(int(eday)+1)):
                    if k<10:
                        a="0"+str(k)
                        if i<10:
                            dates.append("_"+syear+"_"+aa+"_"+a)
                        else:
                            dates.append("_"+syear+"_"+str(i)+"_"+a)
                    else:
                        if i<10:
                            aa="0"+str(i)
                            dates.append("_"+syear+"_"+aa+"_"+str(k))
                        else:
                            dates.append("_"+syear+"_"+str(i)+"_"+str(k))
            else:
                print("full month")
                st = pd.Timestamp(int(syear), int(i), int(sday))
                for l in range(1,(st.daysinmonth+1)):
                    if l<10:
                        a="0"+str(l)
                        if i<10:
                            dates.append("_"+syear+"_"+aa+"_"+a)
                        else:
                            dates.append("_"+syear+"_"+str(i)+"_"+a)
                    else:
                        if i<10:
                            aa="0"+str(i)
                            dates.append("_"+syear+"_"+aa+"_"+str(l))
                        else:
                            dates.append("_"+syear+"_"+str(i)+"_"+str(l))

    database = MySQLdb.connect (host="localhost" , user="root" , passwd="" ,db="pythonlogin")
    cursor = database.cursor()

    rollno=k
    print(rollno)
    global cccounter, bdacounter,bccounter,emcounter,pmcounter
    cccounter=0
    bdacounter=0
    bccounter=0
    emcounter=0
    pmcounter=0

    for i in dates:
        database = MySQLdb.connect (host="localhost" , user="root" , passwd="" ,db="pythonlogin")
        cursor = database.cursor()
        cursor.execute("SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE COLUMN_NAME LIKE '"+i+"';")
        a=cursor.fetchall()
        if(len(a)):
            cursor.execute("select "+str(i)+" from cc where ROLLNO ="+str(rollno))
            a=cursor.fetchall()
            cccounter=cccounter+int(a[0][0])
            cursor.execute("select "+str(i)+" from bc where ROLLNO ="+str(rollno))
            a=cursor.fetchall()
            bccounter=bccounter+int(a[0][0])
            cursor.execute("select "+str(i)+" from em where ROLLNO ="+str(rollno))
            a=cursor.fetchall()
            emcounter=emcounter+int(a[0][0])
            cursor.execute("select "+str(i)+" from pm where ROLLNO ="+str(rollno))
            a=cursor.fetchall()
            pmcounter=pmcounter+int(a[0][0])
            cursor.execute("select "+str(i)+" from bda where ROLLNO ="+str(rollno))
            a=cursor.fetchall()
            bdacounter=bdacounter+int(a[0][0])

    tt=cccounter+bdacounter+bccounter+pmcounter+emcounter
    totaltstudentanaytics=[cccounter,bdacounter,bccounter,pmcounter,emcounter,len(dates),tt]
    return totaltstudentanaytics

def tuseranalytic(startdate,lastdate):
    sinp=startdate
    einp=lastdate
    # start date division
    sday=sinp[8:10]
    smonth=sinp[5:7]
    syear=sinp[0:4]
    # end date division
    eday=einp[8:10]
    emonth=einp[5:7]
    eyear=einp[0:4]
    #making dates
    dates=[]
    if(int(smonth)==int(emonth)):
        print("if executed")
        for i in range(int(sday),(int(eday)+1)):
            if i<10:
                a="0"+str(i)
                dates.append("_"+syear+"_"+smonth+"_"+a)
            else:
                dates.append("_"+syear+"_"+smonth+"_"+str(i))
    else:
        print("else executeed")
        for i in range(int(smonth),int(emonth)+1):
            if(int(smonth)==i):
                print("same start month")
                st = pd.Timestamp(int(syear), int(i), int(sday))
                for j in range(int(sday),(st.daysinmonth+1)):
                    if j<10:
                        a="0"+str(j)
                        if i<10:
                            dates.append("_"+syear+"_"+aa+"_"+a)
                        else:
                            dates.append("_"+syear+"_"+str(i)+"_"+a)
                    else:
                        if i<10:
                            aa="0"+str(i)
                            dates.append("_"+syear+"_"+aa+"_"+str(j))
                        else:
                            dates.append("_"+syear+"_"+str(i)+"_"+str(j))
            elif(i==int(emonth)):
                print("same end month")
                for k in range(1,(int(eday)+1)):
                    if k<10:
                        a="0"+str(k)
                        if i<10:
                            dates.append("_"+syear+"_"+aa+"_"+a)
                        else:
                            dates.append("_"+syear+"_"+str(i)+"_"+a)
                    else:
                        if i<10:
                            aa="0"+str(i)
                            dates.append("_"+syear+"_"+aa+"_"+str(k))
                        else:
                            dates.append("_"+syear+"_"+str(i)+"_"+str(k))
            else:
                print("full month")
                st = pd.Timestamp(int(syear), int(i), int(sday))
                for l in range(1,(st.daysinmonth+1)):
                    if l<10:
                        a="0"+str(l)
                        if i<10:
                            dates.append("_"+syear+"_"+aa+"_"+a)
                        else:
                            dates.append("_"+syear+"_"+str(i)+"_"+a)
                    else:
                        if i<10:
                            aa="0"+str(i)
                            dates.append("_"+syear+"_"+aa+"_"+str(l))
                        else:
                            dates.append("_"+syear+"_"+str(i)+"_"+str(l))

    database = MySQLdb.connect (host="localhost" , user="root" , passwd="" ,db="pythonlogin")
    cursor = database.cursor()

    rollno=processed_text
    print(rollno)
    global cccounter, bdacounter,bccounter,emcounter,pmcounter
    cccounter=0
    bdacounter=0
    bccounter=0
    emcounter=0
    pmcounter=0
    
    for i in dates:
        database = MySQLdb.connect (host="localhost" , user="root" , passwd="" ,db="pythonlogin")
        cursor = database.cursor()
        #checking if date exists
        cursor.execute("SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE COLUMN_NAME LIKE '"+i+"';")
        a=cursor.fetchall()
        if(len(a)):
            cursor.execute("select "+str(i)+" from cc where ROLLNO ="+str(rollno))
            a=cursor.fetchall()
            cccounter=cccounter+int(a[0][0])
            cursor.execute("select "+str(i)+" from bc where ROLLNO ="+str(rollno))
            a=cursor.fetchall()
            bccounter=bccounter+int(a[0][0])
            cursor.execute("select "+str(i)+" from em where ROLLNO ="+str(rollno))
            a=cursor.fetchall()
            emcounter=emcounter+int(a[0][0])
            cursor.execute("select "+str(i)+" from pm where ROLLNO ="+str(rollno))
            a=cursor.fetchall()
            pmcounter=pmcounter+int(a[0][0])
            cursor.execute("select "+str(i)+" from bda where ROLLNO ="+str(rollno))
            a=cursor.fetchall()
            bdacounter=bdacounter+int(a[0][0])

    tt=cccounter+bdacounter+bccounter+pmcounter+emcounter
    tstudentanaytics=[cccounter,bdacounter,bccounter,pmcounter,emcounter,len(dates),tt]
    return tstudentanaytics

def totalattend(i):
    database = MySQLdb.connect (host="localhost" , user="root" , passwd="" ,db="pythonlogin")
    cursor = database.cursor()
    sub=["cc","bda","em","pm","bc"]
    retrive_querycc="select "+sub[0]+" from Student where prn="+str(i)
    retrive_querybda="select "+sub[1]+" from Student where prn="+str(i)
    retrive_queryem="select "+sub[2]+" from Student where prn="+str(i)
    retrive_querypm="select "+sub[3]+" from Student where prn="+str(i)
    retrive_querybc="select "+sub[4]+" from Student where prn="+str(i)

    cursor.execute(retrive_querycc)
    a=cursor.fetchall()
    anlayticscc=int(a[0][0])

    cursor.execute(retrive_querybda)
    a=cursor.fetchall()
    anlayticsbda=int(a[0][0])

    cursor.execute(retrive_queryem)
    a=cursor.fetchall()
    anlayticsem=int(a[0][0])

    cursor.execute(retrive_querypm)
    a=cursor.fetchall()
    anlayticspm=int(a[0][0])

    cursor.execute(retrive_querybc)
    a=cursor.fetchall()
    anlayticsbc=int(a[0][0])

    tt=anlayticscc+anlayticsbda+anlayticsbc+anlayticsem+anlayticspm
    totalanalytics=[anlayticscc,anlayticsbda,anlayticsbc,anlayticsem,anlayticspm,tt]
    return totalanalytics

def admingetattend():
    database = MySQLdb.connect (host="localhost" , user="root" , passwd="" ,db="pythonlogin")
    cursor = database.cursor()
    sub=["cc","bda","em","pm","bc"]
    retrive_querycc="select "+sub[0]+" from Student where prn="+processed_text
    retrive_querybda="select "+sub[1]+" from Student where prn="+processed_text
    retrive_queryem="select "+sub[2]+" from Student where prn="+processed_text
    retrive_querypm="select "+sub[3]+" from Student where prn="+processed_text
    retrive_querybc="select "+sub[4]+" from Student where prn="+processed_text

    cursor.execute(retrive_querycc)
    a=cursor.fetchall()
    anlayticscc=int(a[0][0])

    cursor.execute(retrive_querybda)
    a=cursor.fetchall()
    anlayticsbda=int(a[0][0])

    cursor.execute(retrive_queryem)
    a=cursor.fetchall()
    anlayticsem=int(a[0][0])

    cursor.execute(retrive_querypm)
    a=cursor.fetchall()
    anlayticspm=int(a[0][0])

    cursor.execute(retrive_querybc)
    a=cursor.fetchall()
    anlayticsbc=int(a[0][0])

    tt=anlayticscc+anlayticsbda+anlayticsbc+anlayticsem+anlayticspm
    totalanalytics=[anlayticscc,anlayticsbda,anlayticsbc,anlayticsem,anlayticspm,tt]
    return totalanalytics

def getattend():
    database = MySQLdb.connect (host="localhost" , user="root" , passwd="" ,db="pythonlogin")
    cursor = database.cursor()
    sub=["cc","bda","em","pm","bc"]
    retrive_querycc="select "+sub[0]+" from Student where prn="+userprn
    retrive_querybda="select "+sub[1]+" from Student where prn="+userprn
    retrive_queryem="select "+sub[2]+" from Student where prn="+userprn
    retrive_querypm="select "+sub[3]+" from Student where prn="+userprn
    retrive_querybc="select "+sub[4]+" from Student where prn="+userprn

    cursor.execute(retrive_querycc)
    a=cursor.fetchall()
    anlayticscc=int(a[0][0])

    cursor.execute(retrive_querybda)
    a=cursor.fetchall()
    anlayticsbda=int(a[0][0])

    cursor.execute(retrive_queryem)
    a=cursor.fetchall()
    anlayticsem=int(a[0][0])

    cursor.execute(retrive_querypm)
    a=cursor.fetchall()
    anlayticspm=int(a[0][0])

    cursor.execute(retrive_querybc)
    a=cursor.fetchall()
    anlayticsbc=int(a[0][0])

    tt=anlayticscc+anlayticsbda+anlayticsbc+anlayticsem+anlayticspm
    totalanalytics=[anlayticscc,anlayticsbda,anlayticsbc,anlayticsem,anlayticspm,tt]
    return totalanalytics

#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)