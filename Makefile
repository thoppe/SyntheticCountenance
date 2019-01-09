.PHONY: samples features

P2 = python P2_face_vectors.py
P3 = python P3_keypoints.py

all:
	echo "Use samples or features"

samples:
	python P0_generate_samples.py

features:
	python P1_compute_bbox.py
	$(P2) & $(P2) & $(P2) & $(P2) & $(P2) & $(P2) & $(P2) & $(P2)
	$(P3) & $(P3)
	python P4_image_attribs.py
