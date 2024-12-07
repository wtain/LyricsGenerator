import time
import ctypes

# Prevent sleep by simulating user activity
def prevent_sleep():
    while True:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)
        time.sleep(60)

if __name__ == "__main__":
    prevent_sleep()