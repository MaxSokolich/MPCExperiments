This is a repo containing just tracking information from a FLIR Blackfly camera




4)  need to install qt5
    - sudo apt install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools  
    - sudo apt install qt5-default

5) need to install Spinnaker FLIR camera SDK and python API: 
    - https://flir.app.boxcn.net/v/SpinnakerSDK/file/1093743440079
    - may need: sudo apt-mark manual qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools for spinview 

7) need to add "self.cam.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)" above self.cam.BeginAcquistion() line in $ .local/lib/python3.8/site-packages/EasyPySpin.videocapture.py




/opt/homebrew/bin/python3.9 -m PyQt5.uic.pyuic uis/MPCGUI.ui -o gui_widgets.py

![UI](https://github.com/user-attachments/assets/979f23b6-1e24-40b6-b29a-52032e553023)

![Training1](https://github.com/user-attachments/assets/e0969555-ee57-4302-857e-66830ccb79b5)
