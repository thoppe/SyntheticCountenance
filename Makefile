all:
	echo "Use samples or features"
samples:
	python P0_generate_samples.py
features:
	python P1_compute_bbox.py
	python P2_face_vectors.py & python P2_face_vectors.py & python P2_face_vectors.py ; fg
	python P3_keypoints.py
	python P4_image_attribs.py
