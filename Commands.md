For UI = cd then enter Tab until you get .\ui\ then enter 
Backend = just copy paste this in new terminal - python anti_drone_system/web_server.py

whenever i will make changes you have to take pull with this command - 

git pull origin main    (also you can restart the backend if changes are not visible)

make sure you do not push any of your code in my main branch!!


[for wired usb connection, change mavlink -> connection type = serial and serial port to your comport

for wireless connection, change mavlink -> connection type = udp and baudrate to 115200 (also update baudrate of your drone)] 

These changes will take place in settings.yaml file when you save and restart the pipeline.