import sys
import queue

from constants import *
from queue import Queue
import threading

from chessboard import Chessboard
from video_processing import VideoProcessing
from solver import Solver

class Program:
    """Class that manage the program : creates workers, reads video frames, send frames to workers"""
    def __init__(self):
        print(TITLE)
        self.finished = False
        # Load video
        self.frame_counter = FRAME_START
        self.capture = cv2.VideoCapture(VIDEO_PATH)
        if not self.capture.isOpened():
            sys.exit(f"Unable to open {VIDEO_PATH}")
        self.nb_frames_to_process = int(int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) / FRAME_INT) + 1
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_counter)
        # Resize the video if possible
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[WIDTH_INDEX])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[HEIGHT_INDEX])
        self.must_resize = False
        if int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) != FRAME_SIZE[WIDTH_INDEX] or int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) != FRAME_SIZE[HEIGHT_INDEX]:
            print(f"🪧 Read of {VIDEO_PATH} in {FRAME_SIZE[0]}x{FRAME_SIZE[1]} not available. Will resize frames separately.")
            self.must_resize = True
        else:
            print(f"✅ Read of {VIDEO_PATH} in {FRAME_SIZE[0]}x{FRAME_SIZE[1]} succeded.")

        # Results of the the video processing workers
        self.chessboards_queue = queue.PriorityQueue()

        # Workers
        self.frames_queue = Queue()
        self.video_processing_workers = []
        for i in range(NB_VIDEO_PROCESSING_WORKERS):
            thread = threading.Thread(target=VideoProcessing.worker, args=(self.frames_queue, self.chessboards_queue, self.must_resize))
            thread.start()
            self.video_processing_workers.append(thread)
        print(f"🧵 {NB_VIDEO_PROCESSING_WORKERS} video processing workers started... ")
        self.solver_worker = threading.Thread(target=Solver.worker, args=(self.chessboards_queue, self.nb_frames_to_process))
        self.solver_worker.start()
        print(f"🧵 Solver worker started... ")

    def run(self):
        """
        Read the frames of the video and send them to the workers.
        """
        int_counter = 0
        ret, current_frame = self.capture.read()
        while ret:
            if self.frame_counter % FRAME_INT == 0:
                self.frames_queue.put((current_frame, int_counter))
                int_counter += 1
            self.frame_counter += 1
            ret, current_frame = self.capture.read()

    def close(self):
        """Join the workers, releases the capture and destroys windows."""
        print("🎉 Frames read finished, waiting for end of video processing...")
        self.frames_queue.join()
        for i in range(NB_VIDEO_PROCESSING_WORKERS):
            self.frames_queue.put(None)
        for i in range(NB_VIDEO_PROCESSING_WORKERS):
            self.video_processing_workers[i].join()
        print("🎉 Video processing finished, waiting for end of game processing...")
        self.solver_worker.join()
        print("🎉 Game analysis finished...")
