"""
def parse(msg):
	# WORKER THREAD — no op()/TD access. Decode the model's message into plain,
	# thread-safe data (numpy array, or list/dict of primitives). Return None to skip.
	return msg.getTensor("pose")        # or keypoints, clusters, segmentation, ...
"""
import numpy as np

def fill(scriptOp, data):                 # data: list[PoseEntry]
	# MAIN THREAD — full TD access, runs inside the operator's cook.
	scriptOp.clear()
	if not data:
		scriptOp.copyNumpyArray(np.zeros((2, 17), dtype=np.float32))  # keep 2 ch, 0 samples
		return
	# Stack every person's keypoints as samples -> (P*17, 2)
	kp = np.concatenate(
		[np.asarray(p['keypoints'], dtype=np.float32) for p in data],
		axis=0,
		)
	arr = np.ascontiguousarray(kp.T)        # (2, P*17): ch0 = x, ch1 = y
	arr[1] = 1.0 - arr[1]                   # flip Y to match the flipud'd rgb display
	scriptOp.copyNumpyArray(arr)            # -> channels chan1 (x), chan2 (y)