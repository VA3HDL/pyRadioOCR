# 00. CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
# 01. CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
# 02. CAP_PROP_POS_AVI_RATIO Relative position of the video file
# 03. CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
# 04. CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
# 05. CAP_PROP_FPS Frame rate.
# 06. CAP_PROP_FOURCC 4-character code of codec.
# 07. CAP_PROP_FRAME_COUNT Number of frames in the video file.
# 08. CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
# 09. CAP_PROP_MODE Backend-specific value indicating the current capture mode.
# 10. CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
# 11. CAP_PROP_CONTRAST Contrast of the image (only for cameras).
# 12. CAP_PROP_SATURATION Saturation of the image (only for cameras).
# 13. CAP_PROP_HUE Hue of the image (only for cameras).
# 14. CAP_PROP_GAIN Gain of the image (only for cameras).
# 15. CAP_PROP_EXPOSURE Exposure (only for cameras).
# 16. CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
# 17. CAP_PROP_WHITE_BALANCE Currently unsupported
# 18. CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)

import numpy as np
import cv2
import pytesseract
from pytesseract import Output
import socket
import uuid

def send2commander(command1="",command2="",value=""): # command1=CmdSetFreq command2=xcvrfreq value=21230.55
    # Example freq only <command:10>CmdSetFreq<parameters:20><xcvrfreq:8>21230.55
    # Freq and mode combo <command:14>CmdSetFreqMode<parameters:56><xcvrfreq:5>14080<xcvrmode:4>RTTY
    parm = "<" + command2 + ":" + str(len(value)) + ">" + value
    payload = "<command:" + str(len(command1)) + ">" + command1 + "<parameters:" + str(len(parm)) + ">" + parm
    print("Commander: ", payload)
    host = "127.0.0.1" # socket.gethostname()
    port = 52002
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect( ( host, port) )
    s.sendall( bytes( payload + "\n", "utf-8"))    
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

def checkFreq(Frequency=0.0):
    #print(Frequency)
    if Frequency > 1799.99 and Frequency < 2000:
        return True, "160", "LSB"
    elif Frequency > 3499.99 and Frequency < 4000:
        return True, "80", "LSB"
    elif Frequency > 5331.99 and Frequency < 5405.1:   # 60m channels 5,332 kHz, 5,348 kHz, 5,358.5 kHz, 5,373 kHz and 5,405 kHz.
        return True, "60", "USB"
    elif Frequency > 6999.99 and Frequency < 7300:
        return True, "40", "LSB"
    elif Frequency > 9999.99 and Frequency < 10300:
        return True, "30", "CW"
    elif Frequency > 13999.99 and Frequency < 14350:
        return True, "20", "USB"
    elif Frequency > 17999.99 and Frequency < 18600:
        return True, "17", "USB"
    elif Frequency > 20999.99 and Frequency < 21450:
        return True, "15", "USB"
    elif Frequency > 24889.99 and Frequency < 24990:
        return True, "12", "USB"
    elif Frequency > 27999.99 and Frequency < 29600:
        return True, "10", "USB"
    elif Frequency > 29599.99 and Frequency < 29700:
        return True, "10", "FM"
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

# Initialize the webcam
cap = cv2.VideoCapture(2)    # 0=Logitech // 2=Wyze

# set the camera values
cap.set( 3, 640)
cap.set( 4, 480)
cap.set( 5, 1)
# cap.set(10, 120)                       # Works only with Logitech
# cap.set(cv2.CAP_PROP_EXPOSURE, -3.0)   # Works only with Logitech

counter = 0
numbers = ""
new_numbers = ""
new_band = ""
new_mode = ""
maxX = 0
oldthresh = None
percentage = 1
whitebkg = np.zeros([200,300],dtype=np.uint8)
#cv2.imshow('Single Channel Window', whitebkg)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#pytesseract.pytesseract.environ = ".\\"

# this loop is only needed to tune the camera
while False:
    # Capture an image from the webcam
    ret, img = cap.read()
    height, width, _ = img.shape
    # img = img[125:height-125, 200:width-300]    # 640 x 480 settings Wyze mounted from the arm beside the radio

    # Select ROI
    r = cv2.selectROI("select the area", img)
    
    # Crop image
    cropped_image = img[int( r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    
    # Display cropped image
    cv2.imshow("Cropped image", cropped_image)
    cv2.waitKey(0)
    break

    # Show the webcam feed
    cv2.imshow("Webcam", img)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# this is the real thing
while True:
    # Capture an image from the webcam
    ret, img = cap.read()
    height, width, _ = img.shape

    #img = img[200:height-200, 250:width-200]    # 640 x 480 settings Logitech for FT-757GX    
    #img = img[300:height-350, 550:width-650]    # 1280 x 720 settings
    #img = img[150:height-150, 300:width-150]    # 640 x 480 settings Wyze
    img = img[125:height-125, 200:width-300]    # 640 x 480 settings Wyze mounted from the arm beside the radio    
    #img = img[160:height-100, 200:width-100]    # 320 x 200 settings Wyze
    cv2.imshow("Webcam", img)
    
    #src = img
    #src[:,:,2] = np.zeros([img.shape[0], img.shape[1]])    

    # Affine top to left
    rows, cols, ch = img.shape
    src_points = np.float32([[0,0], [cols-1,0], [0,rows-1]])
    #dst_points = np.float32([[0,0], [int(0.97*(cols-1)),0], [int(0.03*(cols-1)),rows-1]])     # FT-757GX
    dst_points = np.float32([[0,0], [int(0.92*(cols-1)),0], [int(0.08*(cols-1)),rows-1]])     # TS-930S
    matrix = cv2.getAffineTransform(src_points, dst_points)
    dst = cv2.warpAffine(img, matrix, (cols,rows))

    # Erode
    kernel = np.ones((1, 1), np.uint8)
    img2 = dst # cv2.erode(dst, kernel, iterations=2)

    # img2 = cv2.GaussianBlur(img, (1,1), 5)

    # resize image
    # scale_percent = 200 # percent of original size    FT-757GX setting
    scale_percent = 200 # percent of original size      TS-930S setting with Wyze cam
    width = int(img2.shape[1] * scale_percent / 100)
    height = int(img2.shape[0] * scale_percent / 100)
    dim = (width, height)    
    resized = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
    
    # brightness
    resized = increase_brightness(resized, value=35)

    # Convert the image to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (2,2))    

    # threshold grayscale image to extract glare
    # thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]         # Setting for FT-757GX
    
    thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)[1]         # Setting for TS-930S and Wyze 

    #--- take the absolute difference of the images ---
    if not (oldthresh is None):
        res = cv2.absdiff( thresh, oldthresh)
         #--- convert the result to integer type ---
        res = res.astype(np.uint8)

        #--- find percentage difference based on number of pixels that are not zero ---
        percentage = (np.count_nonzero(res) * 100)/ res.size
        print( "Diff in frame: ", percentage)

    oldthresh = thresh   

    # Erode again
    # kernel = np.ones((1, 1), np.uint8)
    # thresh = cv2.erode(thresh, kernel, iterations=2)

    # Convert image to gray and blur it
    #src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    #src_gray = cv2.blur(src_gray, (3,3))
    
    threshold = 255 # initial threshold        
    canny_output = cv2.Canny(gray, threshold, threshold * 2)       
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
    
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])        
    
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)  
    
    while counter < 1000:   #take 1000 samples to fix the ROI
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
                #print(y, x, h, w, y1, y2, x1, x2)
                
    roi = thresh[y1:y2, x1:x2]

    # paste the numbers over a large white canvass
    whitebkg.fill(255)
    x_offset = 0
    y_offset = 65
    whitebkg[y_offset:y_offset+roi.shape[0], x_offset:x_offset+roi.shape[1]] = roi
    roi = whitebkg

    # Use pytesseract to recognize the numbers in the image - Default config --psm 11
#        config="--oem 3 --psm 8 -c tessedit_char_whitelist=.0123456789 textord_old_xheight=1 textord_min_xheight=40 textord_max_noise_size=18")
#        config="--oem 3 --psm 8 -c tessedit_char_whitelist=.0123456789")    
#        config="--oem 1 --psm 13 -c tessedit_char_whitelist=.0123456789") +-,.0123456789E
# numbers = pytesseract.image_to_string(thresh, lang="letsgodigital", \

    numbers = pytesseract.image_to_string(roi, lang="train", config="--dpi 300 --oem 3 --psm 13 -c tessedit_char_whitelist=.0123456789")
    numbers = numbers.replace(".", "")
    numbers = numbers.replace(" ", "")
    numbers = numbers.replace("\n", "")
    d = pytesseract.image_to_data(roi, lang="train", config="--dpi 300 --oem 3 --psm 13 -c tessedit_char_whitelist=.0123456789", output_type=Output.DICT)
    print(d['conf'])
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 40:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            roi = cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
            fnumbers = numbers[:len(numbers)-1] + "." + numbers[-1:]
            if float(fnumbers) > 30000.0 or float(fnumbers) < 520.0:
                maxX = 0    # reset the ROI
            else:
                valid, band, mode = checkFreq(float(fnumbers))
                if new_numbers != fnumbers and valid and percentage > 0.33:
                    new_numbers = fnumbers
                    new_band = band
                    new_mode = mode
                    tcp_numbers = fnumbers + "0"
                    print( d['conf'], new_numbers, tcp_numbers)                    
                    send2commander("CmdSetFreq", "xcvrfreq", tcp_numbers)
                    send2commander("CmdSetMode", "1", mode)
                    #send2log4om( "SetTxFrequency", "<Frequency>" + numbers + "00" + "</Frequency>")
                    #send2log4om( "SetMode", "<Mode>" + mode + "</Mode>")

    # Display the recognized numbers on the screen
    cv2.putText(resized, new_numbers  + " KHz " + new_band + "m " + new_mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # cv2.imshow("Eroded", eroded)
    cv2.imshow("Gray", gray)

    cv2.imshow("Thresh", thresh)

    cv2.imshow('Contours', drawing)

    cv2.imshow('ROI', roi)

    # Show the webcam feed
    resized = cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Resized", resized)

    lastKey = cv2.waitKey(1) & 0xFF
    if lastKey == ord('r'):
        print("Reset...")
        maxX = 0
        counter = 0
    elif lastKey == ord('q'):
        # Break the loop if the 'q' key is pressed    
        break

# Release the webcam
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()