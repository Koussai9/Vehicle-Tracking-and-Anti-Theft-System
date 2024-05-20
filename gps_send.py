import pyrebase
import serial
import pynmea2

firebaseConfig={
    "apiKey": "AIzaSyDy1e0fgGt6gesvHAqwiXlTFQJfuAzXyHA",
    "authDomain": "gps-tracker-b8f9e.firebaseapp.com",
    "projectId": "gps-tracker-b8f9e",
    "storageBucket": "gps-tracker-b8f9e.appspot.com",
    "messagingSenderId": "198852122326",
    "appId": "1:198852122326:web:26decbc59cb1197936db77",
    "databaseURL": "https://gps-tracker-b8f9e-default-rtdb.firebaseio.com"
    }

firebase=pyrebase.initialize_app(firebaseConfig)
db=firebase.database()

while True:
        port="/dev/ttyAMA0"
        ser=serial.Serial(port, baudrate=9600, timeout=0.5)
        dataout = pynmea2.NMEAStreamReader()
        newdata=ser.readline()
        n_data = newdata.decode('latin-1')
        if n_data[0:6] == '$GPRMC':
                newmsg=pynmea2.parse(n_data)
                lat=newmsg.latitude
                lng=newmsg.longitude
                gps = "Latitude=" + str(lat) + " and Longitude=" + str(lng)
                print(gps)
                data = {"LAT": lat, "LNG": lng}
                db.update(data)
                print("Data sent")
