import cv2
import multiprocessing as mp
import time
import face_recognition

def create_op(output_buffer):
    """This function will create a single output video using frames stored in the output_buffer"""    
    frame_num = 1    # Initialize frame number to order frames    
    # Define output video properties
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter()
    out.open("OUTPUT_mp.avi", fourcc, 20, (512, 256))

    try:
        while True:
            if len(output_buffer) == 0:
                continue
            else:
                # print("length of output buffer: ", len(output_buffer))                  
                output_buffer.sort()   # Sort the common list such that the frame with least frame number is at the first position (like in priority queue)             
                if output_buffer[0][0] == frame_num:   # If the frame number at first position corresponds to current frame number, take it otherwise skip             
                    frame = output_buffer.pop(0)[1]                    
                    out.write(frame)
                    frame_num += 1  
                    print("Time elapsed so far is: ", time.time() - start)
                    
    except:
        # Release resources
        out.release()
        
        

def process_images_in_input_buffer(index, input_buffer, output_buffer):
    """This function calls the draw_faces function and store result in the output buffer"""    
    while True:
        if len(input_buffer) == 0:
            continue
        else:
            try:
                # print("length of input buffer: ", len(input_buffer))
                frame_index, frame = input_buffer.pop(0)
                bb_frame = draw_faces(frame_index, frame)
                output_buffer.append([frame_index, bb_frame])
                
            except:
                continue




def draw_faces(num_frame, frame):
    """The function finds faces in the image given and draws bounding boxes around them"""
    
    print("processing frame number : ", num_frame)
    # Resize is done to increase FPS. Can be skipped
    frame = cv2.resize(frame, dsize=(512, 256), interpolation=cv2.INTER_LINEAR)
    # Get faces in the frame and draw bounding boxes around them
    boxes = face_recognition.face_locations(frame)
    for box in boxes:
        top,right, bottom, left = box
        cv2.rectangle(frame, (right, top), (left, bottom), (0, 255, 0), 2)

    # Mark the frame with the frame number
    cv2.putText(frame,str(num_frame),(400, 50), FONT, 1, (0, 255, 255), 2, cv2.LINE_4)
     
    return frame
    
def get_captures(src, input_buffer):
    """This function captures frames from and camera and put them in the input buffer"""
    cap = cv2.VideoCapture(src)
    num_frame = 1 # maintain number of frames read
    
    try:
        while cap.isOpened() and num_frame < MAX_FRAME:  # The MAX_FRAME condition is kept to measure FPS. This condition can be removed for indefinite webcam processing
            ret, frame = cap.read()
            if not ret:
                break
            input_buffer.append([num_frame, frame])
            num_frame += 1
            
            cv2.imshow('img', frame) # Unprocessed camera feed is shown to let the user know that camera is recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break 

        cap.release()

    except:
        cap.release()


def multi_process():
    """ This function starts parallel processes that process separate frames of video simultaneously."""
    # Set the number of parallel processes
    num_processes = mp.cpu_count()
    # num_processes = 5
    
    processes = []
    
    #Define the video source. 0 for webcam feed and file name for processing video file
    src = 'test_vid.mp4'

    # mp.Manager is used to define input and output buffers that each process access
    with mp.Manager() as manager:        
        input_buffer = manager.list()
        output_buffer = manager.list()

        # Process that only read camera for capturing frames
        read_cam = mp.Process(target = get_captures, args = (src, input_buffer))
        read_cam.start()
        processes.append(read_cam)
    
        # Define processes that will process images from input_buffer
        for i in range(num_processes):
            p = mp.Process(target = process_images_in_input_buffer, args = (i, input_buffer, output_buffer, ))
            p.start()
            processes.append(p)

        # Define a separate process to create output video from frames aggregated in the output buffer
        op = mp.Process(target = create_op, args = (output_buffer,))
        op.start()
        processes.append(op)

        for process in processes:
            process.join()

start = time.time()
MAX_FRAME = 200  # Upto a max frame are taken to calculate FPS. The stream could otherwise go indefinitely
FONT = cv2.FONT_HERSHEY_SIMPLEX  # Font used for numbering frames

#FPS obtained is 13.57
if __name__ == "__main__":
    
    """Program to detect faces in video stream using multiprocessing"""    
    start = time.time()    
    multi_process()
    end = time.time()
    print("Total time taken: ", end-start, "(sec)")
    print("FPS: ", MAX_FRAME/end-start)