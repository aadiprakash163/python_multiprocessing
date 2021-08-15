import cv2
import time
import face_recognition


def face_detection(output_file):
    """This function takes the video feed and detect faces frame by frame."""
    
    # Read the video stream
    cap = cv2.VideoCapture('test_vid.mp4')
    # Set output video codec
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Define output frame size. Frames size is set to (512, 256) and a frame rate of 20.0 is used.
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (512, 256))
    
    font = cv2.FONT_HERSHEY_SIMPLEX # Define font for numbering on every frame
    frame_num = 0  # Initialize frame number counter


    try: 
        while cap.isOpened() and frame_num < NUM_FRAMES:
            ret, frame = cap.read()
            
            # If return values is false, terminate
            if not ret:
                break

            frame_num += 1
            # The frame is resized to increase FPS; can be skipped as well
            frame = cv2.resize(frame, dsize=(512, 256), interpolation=cv2.INTER_LINEAR)
            
            # Get faces in the frame and draw bounding boxes around them
            boxes = face_recognition.face_locations(frame)
            for box in boxes:
                top,right, bottom, left = box
                cv2.rectangle(frame, (right, top), (left, bottom), (0, 255, 0), 2)

            # Mark the frame with the frame number
            cv2.putText(frame,str(frame_num),(400, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
            # Display each frame
            cv2.imshow('img', frame)
            # Save the frame
            out.write(frame)

            # Stop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except:
        cap.release()
        out.release()
        

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def single_process(output_file):
    print("Finding faces in the camera stream using Single Process..")
    start = time.time()
    face_detection(output_file)
    end = time.time()
    processing_time = end - start
    print("Total time taken using single single process: ", processing_time)
    print("FPS: ", NUM_FRAMES/processing_time)

    # FPS is 13.24


NUM_FRAMES = 500 # Maximum number of frames is fixed
output_file = 'output_single_process.avi'
single_process(output_file)


