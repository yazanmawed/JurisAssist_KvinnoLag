To run the project please install the libraries on your system based on your OS:
=================================================================================
==================================== Windows ====================================
=================================================================================
python -m venv venv
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\activate
pip install flask
pip install flask flask-cors
python.exe -m pip install --upgrade pip


=================================================================================
================================= MacOS | Linux =================================
=================================================================================
python3 -m venv venv
source venv/bin/activate
sudo -s
pip3 install flask
pip3 install flask flask-cors
python.exe -m pip install --upgrade pip