# radiologylab
Tensorflow project for detecting fractures in radiology images

# Requirements
python 3.6

# Train model command history
pip install pandas

pip install tensorflow==1.14.0

pip install pillow

pip install matplotlib

python setup.py install

python setup.py build

python xml_a_csv.py --inputs=img_test --output=test

python xml_a_csv.py --inputs=img_training --output=training

python csv_a_tf.py --csv_input=CSV/test.csv --output_path=records/test.record --images=img

python csv_a_tf.py --csv_input=CSV/training.csv --output_path=records/training.record --images=img

# Copy deployment and nets folder from slim folder to [USER FOLDER]\AppData\Local\Programs\Python\Python36\Lib\site-package

python object_detection/train.py --logtostderr --train_dir=train --pipeline_config_path=model/faster_rcnn_resnet101_coco.config

python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path model/faster_rcnn_resnet101_coco.config --trained_checkpoint_prefix train/model.ckpt-58 --output_directory model_freezed


# jpeg images
python object_detection/object_detection_runner.py --labels configuration/label_map.pbtxt --images img --model model_freezed
