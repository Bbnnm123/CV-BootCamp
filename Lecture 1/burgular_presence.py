import cv2

cap = cv2.VideoCapture(0)   #0 is the default for webcam, replace the 0 with 'videoname.mp4' for video file input

frames = []
gap = 5
count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(gray)

    if len(frames) > gap + 1:
        frames.pop(0)   #removes the lowest frame value to maintain the same game size if it exceeds gap+1

    cv2.putText(frame, f'Frames captured: {len(frames)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if len(frames) >  gap:
        diff = cv2.absdiff(frames[0],frames[-1])
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY) #picking readings based of certain brigtness values, greater than 30, lower than 255 is the range we aiming for.

        contours, = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   

        for c in contours:      #creates bounding boxes around the contours(areas of pixels that have changed)
            if cv2.contourArea(c) < 1000:  #start playing around with this value to adjust sensitivity to detect only certain size objects (smaller the value the more sensitive)
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        motion = any(cv2.contourArea(c) > 1000 for c in contours)   #change sensitivity here as well

        if motion:
            cv2.putText(frame, 'Motion Detected', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(f'motion_frame_{count}.jpg', frame)  #saves frames with motion detected
        print(f"Saved motion_frame_{count}.jpg")

        cv2.imshow('Motion Detection', frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == 27:
            break

        cap.release()
cv2.destroyAllWindows()

