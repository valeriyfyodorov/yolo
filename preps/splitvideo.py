import cv2
import time
import os


def video_to_frames(input_loc, output_loc, every_nth_frame, reduce_ratio=0.5):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        print("OSError, skip creating folder - already exist")
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("Number of frames: ", video_length)
    count = 0
    print("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        # Write the results back to output location.
        if count % every_nth_frame == 0:
            h, w, _ = frame.shape
            h, w = int(h * reduce_ratio), int(w * reduce_ratio)
            resized = cv2.resize(frame, (w, h))
            cv2.imwrite(output_loc + "/" + os.path.basename(input_loc).split('.')
                        [0] + "%#05d.jpg" % (count+1), resized)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print("Done extracting frames.\n%d frames extracted" % count)
            print("It took %d seconds forconversion." % (time_end-time_start))
            break


if __name__ == "__main__":
    print("running main file")
    input_loc = '07.TS'
    output_loc = 'frames'
    every_nth_frame = 4
    reduce_ratio = 0.5
    video_to_frames(input_loc, output_loc, every_nth_frame)
