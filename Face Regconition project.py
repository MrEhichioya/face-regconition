#!/usr/bin/env python
# coding: utf-8

# In[22]:


import cv2
import numpy as np

# Load the model
model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'model.caffemodel')

# Create a video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    
    # Resize the frame to (300, 300) for faster processing
    frame = cv2.resize(frame, (300, 300))
    
    # Convert the frame from BGR to RGB for compatibility with the model
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)

    # Pass the blob through the model and make predictions
    model.setInput(blob)
    detections = model.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Get the bounding box coordinates
            x1 = int(detections[0, 0, i, 3] * frame.shape[1])
            y1 = int(detections[0, 0, i, 4] * frame.shape[0])
            x2 = int(detections[0, 0, i, 5] * frame.shape[1])
            y2 = int(detections[0, 0, i, 6] * frame.shape[0])

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('Object Detection', frame)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




