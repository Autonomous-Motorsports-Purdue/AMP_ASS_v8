import socket
import json
import threading
import queue
import time

class ThreadedTelemetryStreamer:
    def __init__(self, host='localhost', port=5005):
        self.host = host
        self.port = port
        self.data_queue = queue.Queue(maxsize=1) # Only keep the latest data point
        self.running = True
        
        # Start the background networking thread immediately
        self.thread = threading.Thread(target=self._network_loop, daemon=True)
        self.thread.start()

    def _network_loop(self):
        """ This runs in the background and does the 'blocking' work. """
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(1)
        
        while self.running:
            print(f"Streamer: Waiting for host on port {self.port}...")
            # .settimeout allows us to check self.running occasionally
            server_socket.settimeout(1.0) 
            try:
                conn, addr = server_socket.accept()
                with conn:
                    print(f"Streamer: Host connected from {addr}")
                    while self.running:
                        # Get the latest data from the queue (blocks until data exists)
                        data = self.data_queue.get() 
                        message = json.dumps(data) + '\n'
                        try:
                            conn.sendall(message.encode('utf-8'))
                        except (BrokenPipeError, ConnectionResetError):
                            print("Streamer: Host disconnected.")
                            break
            except socket.timeout:
                continue # Just loop back and check if we should still be running
        
        server_socket.close()

    def run(self, lat, lon, heading, steer, imu_heading):
        """
        Non-blocking: Call this from your main loop. 
        It puts data in a queue and returns immediately.
        """
        data = {"lat": lat, "lon": lon, "heading": heading, "steer":steer, "imu_heading":imu_heading}
        try:
            # If queue is full, remove old data to keep it fresh
            if self.data_queue.full():
                self.data_queue.get_nowait()
            self.data_queue.put_nowait(data)
        except queue.Full:
            pass 

# ==========================================
# MAIN ROBOT LOOP (Now completely unblocked!)
# ==========================================
if __name__ == "__main__":
    streamer = ThreadedTelemetryStreamer()

    try:
        while True:
            # YOUR ROBOT LOGIC HERE
            # This code will now run even if the host isn't connected!
            print("Robot is thinking/driving...") 
            
            # Simulated GPS update
            streamer.run(lat=40.4277, lon=-86.9187, heading=90.0)
            
            time.sleep(0.1) # 10Hz loop
    except KeyboardInterrupt:
        streamer.running = False