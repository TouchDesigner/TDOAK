"""
def parse(msg):
	# WORKER THREAD — no op()/TD access. Decode the model's message into plain,
	# thread-safe data (numpy array, or list/dict of primitives). Return None to skip.
	return msg.getTensor("pose")        # or keypoints, clusters, segmentation, ...
"""
import numpy as np

def fill(scriptOp, data):                 # data: list[PoseEntry]
	# MAIN THREAD — full TD access, runs inside the operator's cook.
	stagingData = scriptOp.createGeometryStagingData()

	if not data:
		scriptOp.buildGeometry(stagingData)
		return
	
	stagingData.createPointAttrib(name='P')
	# Stack every person's keypoints as samples -> (P*17, 2)

	kp = np.concatenate(
		[np.asarray(p['keypoints'], dtype=np.float32) for p in data],
		axis=0,
		)
	
	# this is of shape (n*17,2), but we need 3 coordinates per point
	kp = np.pad(kp, ((0, 0), (0, 1)), mode='constant')
	# flip y
	kp[:, 1] = 1 - kp[:, 1]
	stagingData.appendPointAttribValues('P', kp.tolist())

	# now build the edges between the keypoints
	# be careful, as we have to add 18 to every new detected array point indexes to account for more than 1 detection.
	
	edge_chunks = []
	for i, p in enumerate(data):
	    edges = np.asarray(p['edges'], dtype=np.float32)
	    edges += i * 18
	    edge_chunks.append(edges)
	
	edges = np.concatenate(edge_chunks, axis=0)

	for e in edges:
		stagingData.appendPrimLine(int(e[0]),int(e[1]))

	scriptOp.buildGeometry(stagingData)
	