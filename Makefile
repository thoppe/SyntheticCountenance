.PHONY: samples features

P2 = python P2_face_vectors.py
P3a = python P3_keypoints_5.py
P3b = python P3_keypoints_68.py

all:
	echo "Use samples or features"

samples:
	python P0_generate_samples.py

features:
	python P1_compute_bbox.py
	$(P2) & $(P2) & $(P2) & $(P2) & $(P2) & $(P2) & $(P2) & $(P2)
	$(P3a) & $(P3a)
	$(P3b) & $(P3b)	
	python P4_image_attribs.py
	python C0_collect_data.py
