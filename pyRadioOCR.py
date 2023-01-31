# -------------------------------------------------------------
# pyRadioOCR v0.8.1a
#
# https://youtu.be/FJd5-t7Id1A
#
# (C) 2023 Pablo Sabbag, VA3HDL
# Released under GNU Public License (GPL)
# 
# To-do:
# option to save settings to config.ini from app
# enumerate video capture devices
# enumerate video capture devices capabilities
#
# Changelog:
# 2023-01-30 Added ability to save the settings on the
#            confi.ini file
# 2023-01-29 Added Voice support, GUI, and user settings 
#            on the config.ini file
# -------------------------------------------------------------

"""
00. CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
01. CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
02. CAP_PROP_POS_AVI_RATIO Relative position of the video file
03. CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
04. CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
05. CAP_PROP_FPS Frame rate.
06. CAP_PROP_FOURCC 4-character code of codec.
07. CAP_PROP_FRAME_COUNT Number of frames in the video file.
08. CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
09. CAP_PROP_MODE Backend-specific value indicating the current capture mode.
10. CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
11. CAP_PROP_CONTRAST Contrast of the image (only for cameras).
12. CAP_PROP_SATURATION Saturation of the image (only for cameras).
13. CAP_PROP_HUE Hue of the image (only for cameras).
14. CAP_PROP_GAIN Gain of the image (only for cameras).
15. CAP_PROP_EXPOSURE Exposure (only for cameras).
16. CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
17. CAP_PROP_WHITE_BALANCE Currently unsupported
18. CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
"""
from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import socket
import uuid
import pyttsx3
import configparser

# Networking ------------------------------------------------------------------------------------------

def send2commander(command1="",command2="",value=""): # command1=CmdSetFreq command2=xcvrfreq value=21230.55
    # Example freq only <command:10>CmdSetFreq<parameters:20><xcvrfreq:8>21230.55
    # Freq and mode combo <command:14>CmdSetFreqMode<parameters:56><xcvrfreq:5>14080<xcvrmode:4>RTTY
    parm = "<" + command2 + ":" + str(len(value)) + ">" + value
    payload = "<command:" + str(len(command1)) + ">" + command1 + "<parameters:" + str(len(parm)) + ">" + parm
    print("Commander: ", payload)
    host = "127.0.0.1" # socket.gethostname()
    port = 52002
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.connect( ( host, port) )
    try: 
        s.connect((host, port)) 
    except socket.gaierror as e: 
        print ("Address-related error connecting to server: %s" % e)         
    except socket.error as e: 
        print ("Connection error: %s" % e)     
    try: 
        s.sendall( bytes( payload + "\n", "utf-8"))
    except socket.error as e: 
        print ("Error sending data: %s" % e) 
    s.close()    

#send2commander( "CmdSetFreq", "xcvrfreq", "14347.12")
#send2commander( "CmdSetMode", "1", "USB")

def send2log4om(command1="",value=""): 
    # <RemoteControlRequest> 
    #  <MessageId>C0FC027F-D09E-49F5-9CA6-33A11E05A053</MessageId> 
    #  <RemoteControlMessage>SetTxFrequency</RemoteControlMessage> 
    #  <Frequency>14075000</Frequency > 
    # </RemoteControlRequest>     
    # <?xml version="1.0" encoding="utf-8"?><RadioInfo>  <app>LOG4OM2</app>  <StationName>OPTIPLEX</StationName>  <OpCall>VA3HDL</OpCall>  <RadioNr>1</RadioNr>  <Freq>0</Freq>  <TXFreq>0</TXFreq>  <Mode>LSB</Mode>  <ActiveRadioNr>1</ActiveRadioNr>  <IsSplit>false</IsSplit>  <IsTransmitting>false</IsTransmitting></RadioInfo>
    payload = """<?xml version="1.0" encoding="utf-8"?>\n<RemoteControlRequest>\n<MessageId>""" + str(uuid.uuid4()).upper() + "</MessageId>\n<RemoteControlMessage>" + command1 + \
              "</RemoteControlMessage>\n" + value + "\n</RemoteControlRequest>"

    print("Log4OM: ", payload)
    host = "127.0.0.1" # socket.gethostname()
    port = 2241
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)    
    s.sendto( bytes( payload, "utf-8"), ( host, port))

#send2log4om( "SetTxFrequency", "<Frequency>14333222</Frequency>")
#send2log4om( "SetMode", "<Mode>USB</Mode>")

# /Networking  ------------------------------------------------------------------------------------------

def checkFreq(Frequency=0.0):
    #print(Frequency)
    if Frequency > 1799.99 and Frequency < 2000:
        return True, "160m", "LSB"
    elif Frequency > 3499.99 and Frequency < 4000:
        return True, "80m", "LSB"
    elif Frequency > 5331.99 and Frequency < 5405.1:   # 60m channels 5,332 kHz, 5,348 kHz, 5,358.5 kHz, 5,373 kHz and 5,405 kHz.
        return True, "60m", "USB"
    elif Frequency > 6999.99 and Frequency < 7300:
        return True, "40m", "LSB"
    elif Frequency > 9999.99 and Frequency < 10300:
        return True, "30m", "CW"
    elif Frequency > 13999.99 and Frequency < 14350:
        return True, "20m", "USB"
    elif Frequency > 17999.99 and Frequency < 18600:
        return True, "17m", "USB"
    elif Frequency > 20999.99 and Frequency < 21450:
        return True, "15m", "USB"
    elif Frequency > 24889.99 and Frequency < 24990:
        return True, "12m", "USB"
    elif Frequency > 27999.99 and Frequency < 29600:
        return True, "10m", "USB"
    elif Frequency > 29599.99 and Frequency < 29700:
        return True, "10m", "FM"
    elif Frequency > 49999.99 and Frequency < 54000:
        return True, "6m", "FM"
    elif Frequency > 143999.99 and Frequency < 148000:
        return True, "2m", "FM"
    elif Frequency > 429999.99 and Frequency < 460000:
        return True, "70cm", "FM"
    else:         
        return False, "", ""

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

# Read the configuration
config = configparser.ConfigParser()
config.read("./config.ini")

# Initialize the Voice engine
synthesizer = pyttsx3.init()
synthesizer.setProperty("volume", config.getfloat("VOICE", "volume"))
synthesizer.setProperty("rate", config.getint("VOICE", "rate"))
if config.getboolean("VOICE", "enabled"):
    synthesizer.say(config.get("VOICE","welcome"))
    synthesizer.runAndWait() 
    synthesizer.stop()

# Initialize the GUI ------------------------------------------------------------------------------------------
root = Tk()
root.geometry("850x600")
root.title("pyRadioOCR v0.8.1a by VA3HDL")

right_frame = Frame(root,  width=200,  height=480,  bg='grey')
right_frame.pack(side='right',  fill='both',  padx=5,  pady=5,  expand=True, anchor=E)

# Create a canvas to display the video feed
canvas = Canvas(root, width=640, height=480, bg='black')
canvas.pack(side='top', expand=True, anchor=NW)

left_frame = Frame(root,  width=200,  height=300,  bg='grey')
left_frame.pack(side='left',  fill='both',  padx=5,  pady=5,  expand=True)

lft_col1 = Frame(left_frame,  width=100,  height=300,  bg='grey')
lft_col1.pack(side='left',  fill='both',  padx=5,  pady=5,  expand=True)

lft_col2 = Frame(left_frame,  width=100,  height=300,  bg='grey')
lft_col2.pack(side='right',  fill='both',  padx=5,  pady=5,  expand=True)

def on_select(v):    
    print(v)
    if not debugMode.get():
        cv2.destroyWindow("Webcam")
        cv2.destroyWindow("Gray")
        cv2.destroyWindow("Thresh")
        cv2.destroyWindow('ROI')
        cv2.destroyWindow("Resized")

# Checkboxes
snd2com = BooleanVar()
snd2com.set(config.getboolean("INTEGRATION", "commander"))
c1 = Checkbutton(lft_col1, text="Commander", variable=snd2com, command=lambda: on_select(snd2com.get()))
#c1.pack(anchor=NW)
c1.pack(side=TOP, fill=NONE, expand=FALSE, anchor=NW)

snd2log = BooleanVar()
snd2log.set(config.getboolean("INTEGRATION", "log4om"))
c2 = Checkbutton(lft_col1, text="Log4OM", variable=snd2log, command=lambda: on_select(snd2log.get()))
#c2.pack(anchor=NW)
c2.pack(side=TOP, fill=NONE, expand=FALSE, anchor=NW)

voice = BooleanVar()
voice.set(config.getboolean("VOICE", "enabled"))
c3 = Checkbutton(lft_col1, text="Voice", variable=voice, command=lambda: on_select(voice.get()))
#c3.pack(anchor=NW)
c3.pack(side=TOP, fill=NONE, expand=FALSE, anchor=NW)

autoROI = BooleanVar()
autoROI.set(config.getboolean("PREPROCESS", "autoROI"))
c4 = Checkbutton(lft_col1, text="Auto ROI", variable=autoROI, command=lambda: on_select(autoROI.get()))
#c4.pack(anchor=NW)
c4.pack(side=TOP, fill=NONE, expand=FALSE, anchor=NW)

invert = BooleanVar()
invert.set(config.getboolean("PREPROCESS", "invert"))
c5 = Checkbutton(lft_col2, text="Invert", variable=invert, command=lambda: on_select(invert.get()))
#c5.pack(anchor=NE)
c5.pack(side=TOP, fill=NONE, expand=FALSE, anchor=NW)

noDecimal = BooleanVar()
noDecimal.set(config.getboolean("POSTPROCESS", "noDecimal"))
c6 = Checkbutton(lft_col2, text="No decimals", variable=noDecimal, command=lambda: on_select(noDecimal.get()))
#c6.pack(anchor=NE)
c6.pack(side=TOP, fill=NONE, expand=FALSE, anchor=NW)

debugMode = BooleanVar()
debugMode.set(config.getboolean("OCR", "debug"))
c7 = Checkbutton(lft_col2, text="Debug mode", variable=debugMode, command=lambda: on_select(debugMode.get()))
#c7.pack(anchor=NE)
c7.pack(side=TOP, fill=NONE, expand=FALSE, anchor=NW)

rotate180 = BooleanVar()
rotate180.set(config.getboolean("PREPROCESS", "rotate180"))
c8 = Checkbutton(lft_col2, text="Rotate 180 deg", variable=rotate180, command=lambda: on_select(rotate180.get()))
#c8.pack(anchor=NE)
c8.pack(side=TOP, fill=NONE, expand=FALSE, anchor=NW)

# Sliders
brightCam = Scale(right_frame, from_=0, to=255, length=200, orient='horizontal', label="Cam Brightness")
brightCam.pack(anchor=NE)
brightCam.set(config.get("CAMERA", "brightness"))

exposureSet = Scale(right_frame, from_=0, to=-14, length=200, orient='horizontal', label="Exposure")
exposureSet.pack(anchor=NE)
exposureSet.set(config.get("CAMERA", "exposure"))

slantSet = Scale(right_frame, from_= 0.0, to= 1.0, digits = 3, resolution = 0.01, length=200, orient='horizontal', label="Slant %")
slantSet.pack(anchor=NE)
slantSet.set(config.get("PREPROCESS", "slant"))

threshSet = Scale(right_frame, from_=0, to=255, length=200, orient='horizontal', label="Threshold")
threshSet.pack(anchor=NE)
threshSet.set(config.get("PREPROCESS", "threshold"))

brightSet = Scale(right_frame, from_=0, to=255, length=200, orient='horizontal', label="Preproc Brightness")
brightSet.pack(anchor=NE)
brightSet.set(config.get("PREPROCESS", "brightness"))

scaleSet = Scale(right_frame, from_=0, to=500, length=200, orient='horizontal', label="Scale %")
scaleSet.pack(anchor=NE)
scaleSet.set(config.get("PREPROCESS", "scale"))

# /GUI ------------------------------------------------------------------------------------------

# Main code loop---------------------------------------------------------------------------------
def update():
    # Global variables
    global counter
    global numbers
    global new_numbers
    global new_band
    global new_mode
    global y1
    global maxX
    global threshold
    global oldthresh
    global percentage
    global autoROI
    global ROIset
    global ready2set
    global whitebkg    
    global custom_oem
    global minX
    global maxX
    global minY
    global maxY
    global y1
    global y2
    global x1
    global x2

    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightCam.get())    
    cap.set(cv2.CAP_PROP_EXPOSURE, exposureSet.get())

    frame = cap.read()[1]

    # OCR code  ------------------------------------------------------------------------------------------
    img = frame
    height, width, _ = img.shape

    #img = img[200:height-200, 250:width-200]

    if debugMode.get():
        cv2.imshow("Webcam", img)

    if rotate180.get():
        img = cv2.rotate(img, cv2.ROTATE_180)

    # Affine top to left
    rows, cols, ch = img.shape
    src_points = np.float32([[0,0], [cols-1,0], [0,rows-1]])    
    dst_points = np.float32([[0,0], [int(slantSet.get()*(cols-1)),0], [int((1-slantSet.get())*(cols-1)),rows-1]])
    matrix = cv2.getAffineTransform(src_points, dst_points)
    dst = cv2.warpAffine(img, matrix, (cols,rows))

    # Erode
    #kernel = np.ones((1, 1), np.uint8)
    img2 = dst # cv2.erode(dst, kernel, iterations=2)
    
    # Blur
    # img2 = cv2.GaussianBlur(img, (1,1), 5)
    
    # resize image
    width = int(img2.shape[1] * scaleSet.get() / 100)
    height = int(img2.shape[0] * scaleSet.get() / 100)
    dim = (width, height)    
    resized = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)

    if not autoROI.get() and not ROIset and ready2set:
        height, width, _ = resized.shape       
    
        # Select ROI
        r = cv2.selectROI("Select the ROI area", resized)
             
        y1 = int( r[1])
        y2 = int(r[1]+r[3])
        x1 = int(r[0])
        x2 = int(r[0]+r[2])                

        # Display cropped image
        cv2.imshow("Cropped image", resized[y1:y2,x1:x2])
        cv2.destroyWindow("Select the ROI area")

        ready2set = False
        ROIset = True

    # brightness
    resized = increase_brightness(resized, value=brightSet.get())

    # Convert the image to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (2,2))    

    # threshold grayscale image to extract glare
    if invert.get():
        thresh = cv2.threshold(gray, threshSet.get(), 255, cv2.THRESH_BINARY_INV)[1]
    else:
        thresh = cv2.threshold(gray, threshSet.get(), 255, cv2.THRESH_BINARY)[1]
        
    #--- take the absolute difference of the images ---
    if not (oldthresh is None):
        res = cv2.absdiff( thresh, oldthresh)
         #--- convert the result to integer type ---
        res = res.astype(np.uint8)
        #--- find percentage difference based on number of pixels that are not zero ---
        percentage = (np.count_nonzero(res) * 100)/ res.size
        if debugMode.get():
            print( "Diff in frame: ", percentage)

    oldthresh = thresh   

    # Erode again
    # kernel = np.ones((1, 1), np.uint8)
    # thresh = cv2.erode(thresh, kernel, iterations=2)
  
    canny_output = cv2.Canny(gray, threshold, threshold * 2)       
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
    
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])        
    
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)  
    
    while counter < 1000 and autoROI.get():   # take 1000 samples to define the ROI automatically
        counter += 1
        minX = 0
        minY = 0    
        maxY = 0
        height, width = thresh.shape
        # roi = thresh
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours_poly[i])
            # if h > 40 and w > 10:  # 320x160
            if h > 22 and w > 10 and w < 20:
                #counter += 1
                cv2.drawContours(drawing, contours_poly, i, (0, 255, 0))
                cv2.rectangle(drawing, (x, y), (x+w, y+h), (0, 0, 255), 2)
                minX = min(minX, x)
                maxX = max(maxX, x+w)
                minY = min(minY, y)
                maxY = max(maxY, y+h)
                y1 = y - 5
                y2 = maxY + 5
                x1 = minX
                x2 = maxX + 5
                if debugMode.get():
                    print(y, x, h, w, y1, y2, x1, x2)
                cv2.imshow('Contours', drawing)

    if y1==None:
        roi = thresh
    else:
        roi = thresh[y1:y2, x1:x2]
        # paste the ROI over a white canvas
        whitebkg.fill(255)
        bkgH, bkgW = whitebkg.shape
        roiH, roiW = roi.shape
        yoff = round((bkgH-roiH)/2)
        xoff = round((bkgW-roiW)/2)
        whitebkg[yoff:yoff+roiH, xoff:xoff+roiW] = roi        
        roi = whitebkg
        resized = cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

    numbers = pytesseract.image_to_string(roi, config=custom_oem)
    numbers = numbers.replace("\n", "")

    for char in config.get("POSTPROCESS", "strip"):
        numbers = numbers.replace(char, "")
    
    d = pytesseract.image_to_data(roi, config=custom_oem, output_type=Output.DICT)
    
    if debugMode.get():
        print(numbers)
        print(d['conf'])
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 40 and numbers.isnumeric():
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            roi = cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if noDecimal.get():
                fnumbers = numbers
                log4omDec = "000"
            else:
                fnumbers = numbers[:len(numbers)-1] + "." + numbers[-1:]
                log4omDec = "00"
            if debugMode.get():
                print(fnumbers)
            if float(fnumbers) > 460000.0 or float(fnumbers) < 520.0:
                maxX = 0    # Out of valid range ignore and reset the ROI
            else:
                valid, band, mode = checkFreq(float(fnumbers))
                if new_numbers != fnumbers and valid and percentage > 0.33:
                    new_numbers = fnumbers
                    new_band = band
                    new_mode = mode
                    tcp_numbers = fnumbers + "0"
                    if debugMode.get():
                        print( d['conf'], new_numbers, tcp_numbers)
                    if snd2com.get():
                        send2commander("CmdSetFreq", "xcvrfreq", tcp_numbers)
                        send2commander("CmdSetMode", "1", mode)
                    if snd2log.get():    
                        send2log4om( "SetTxFrequency", "<Frequency>" + numbers + log4omDec + "</Frequency>")
                        send2log4om( "SetMode", "<Mode>" + mode + "</Mode>")
                    if voice.get():
                        synthesizer.say(new_numbers + " kilohertz")
                        synthesizer.runAndWait()                    
                        synthesizer.stop()
    # end For loop
                 
    if debugMode.get():
        # Display the recognized numbers on the screen
        cv2.putText(resized, new_numbers  + " kHz " + new_band + " " + new_mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)    
        # cv2.imshow("Eroded", eroded)
        cv2.imshow("Gray", gray)
        cv2.imshow("Thresh", thresh)
        cv2.imshow('ROI', roi)
        cv2.imshow("Resized", resized)

    lastKey = cv2.waitKey(1) & 0xFF
    if lastKey == ord('r'):
        if debugMode.get():
            print("Reset...")
        maxX = 0
        counter = 0
        ROIset = False
    if lastKey == ord('s'):
        if debugMode.get():
            print("Set ROI...")
        ROIset = False        
        ready2set = True
    elif lastKey == ord('q'):
        # Stop the program if the 'q' key is pressed    
        quit()

    # /OCR code  ------------------------------------------------------------------------------------------
    if rotate180.get():
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    cv2.putText(frame, new_numbers  + " kHz " + new_band + " " + new_mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    canvas.imgtk = photo    
    
    root.after(20,update)

# End of Main code loop---------------------------------------------------------------------------------

# Start the video capture
cap = cv2.VideoCapture(config.getint("CAMERA", "device"))

# set the camera values
cap.set( 3, config.getint("CAMERA", "frame_width"))
cap.set( 4, config.getint("CAMERA", "frame_height"))
#cap.set(cv2.CAP_PROP_FPS, 1)

# initialize the global variables
counter = 0
numbers = ""
new_numbers = ""

new_band = ""
new_mode = ""

minX = None
maxX = None
minY = None
maxY = None
y1 = None
y2 = None
x1 = None
x2 = None

threshold = 255 # initial threshold        

oldthresh = None
percentage = 1
ROIset = False
ready2set = False

# create white canvass
scaleInt = int(config.getint("PREPROCESS", "scale")/100)
whitebkg = np.zeros([config.getint("CAMERA", "frame_height")*scaleInt,
                     config.getint("CAMERA", "frame_width")*scaleInt],
                     dtype=np.uint8)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

custom_oem = r"--oem 3 --psm 7 -l " + config.get("OCR", "language") + " -c tessedit_char_whitelist=" + config.get("OCR", "whitelist")

def key_pressed(event):
    global ROIset
    global ready2set
    if debugMode.get():
        print("Key Pressed: " + event.char)

    if event.char == 's':
        if debugMode.get():
            print("Event - Set ROI...")
        ROIset = False
        ready2set = True
        autoROI.set(False)
    elif event.char == 'q' or ord(event.char) == 27:
        # Stop the program if the 'q' key is pressed
        config.set("VOICE", "enabled", str(voice.get()))
        config.set("INTEGRATION", "commander", str(snd2com.get()))
        config.set("INTEGRATION", "log4om", str(snd2log.get()))
        config.set("PREPROCESS", "autoROI", str(autoROI.get()))
        config.set("PREPROCESS", "invert", str(invert.get()))
        config.set("POSTPROCESS", "noDecimal", str(noDecimal.get()))
        config.set("OCR", "debug", str(debugMode.get()))
        config.set("CAMERA", "brightness", str(brightCam.get()))
        config.set("CAMERA", "exposure", str(exposureSet.get()))
        config.set("PREPROCESS", "slant", str(slantSet.get()))
        config.set("PREPROCESS", "threshold", str(threshSet.get()))
        config.set("PREPROCESS", "brightness", str(brightSet.get()))
        config.set("PREPROCESS", "scale", str(scaleSet.get()))
        config.set("PREPROCESS", "rotate180", str(rotate180.get()))

        with open('./config.ini', 'w') as configfile:
            config.write(configfile)
        if debugMode.get():            
            print("Event - Quit...")
        quit()

root.bind("<Key>", key_pressed)

update()
root.mainloop()