ML Fashion Item Detector

Base Model Used: YOLO11s

Hardware Used: AMD Ryzen 5600H Processor with 6 Cores and 12 Threads, Nvidia GeForce RTX 3050 Ti GPU (4GB Memory), 16GB RAM

Python Version: 3.12.1 on Windows 11

AI Assistance: Google Antigravity IDE (Gemini 3.8 flash)

Training Dataset Used:

- Hugging Face Fashionpedia: 10000 images for Clothing, Shoes and Accessories (https://huggingface.co/datasets/detection-datasets/fashionpedia)

- Open Images V7 (via FiftyOne's downloader): 1000 images for Jewellery (https://docs.ultralytics.com/datasets/detect/open-images-v7)

Total Dataset size: 11000 images (10000 Fashionpedia images for Clothes, Shoes and Accessories + 1000 Open Images V7 images for Jewellery)

Testing Dataset Used:

22 Random Fashion Images downloaded from Google Search

Fine-tuning Epochs and Time:

The model is fine-tuned on 30 Epochs (10 Epochs on 10000 Fashionpedia images + 20 Epochs on full 11000 images including the Open Images V7 Jewellery images). The fine-tuning took ~6 hours total across all 30 Epochs.

Setup Instructions:

Step 1: Clone the git repo to a local machine.

Step 2: Install all Python Dependencies using: pip install -r requirements.txt

Step 3: Run the model on an image to get the annotated image, the crop images and the JSON Detection Data using the UI. In the UI you can select the input image and change the model to use ('fashion_best.pt' by default), the output folder, Confidence and the max IoU. To run use: python gui.py

Step 4: Additionally the output of the inference is also saved and can be checked in the 'outputs' folder present in the project's root directory by default.

Approach:
1. Downloaded the Fashionpedia training dataset from Hugging Face (which is saved in 'fashion_yolo' folder in the project's root's directory by default) using: python train.py --prepare-only
2. Downloaded Jewellery dataset and integrate it with the existing Fashionpedia dataset using: python download_jewellery.py
3. Fine-tuned the YOLO11s model on the training dataset using: python train.py
4. The fine-tuned model is saved as 'fashion_best.pt' by default.
5. The model trains for 10 epochs by default. This can be adjusted by adding the '--epochs' arguement to the train.py command followed by a +ve integer signifying the number of epochs.
6. Additionally during training the first 10 layers of the model are frozen and are not updated for the first 3 epochs by default. This can be adjusted by adding the '--freeze-layers' arguement to the train.py command followed by a +ve integer signifying the number of epochs for which the first 10 layers of the model are to be frozen. This has been done for less overfitting on a small dataset and for faster epochs with less memory usage
7. By default the batch size hyperparameter is set to 16. This is can be adjusted by adding the '--batch' arguement  to the train.py command followed by a +ve integer or -1 for AutoBatch.
8. The image size (imgsz) hyperparameter is set to 640 pixels by default. This can be adjusted by adding the '--imgsz' arguement to the train.py command followed by a +ve multiple of 32 integer like 416, 1280, etc.
9. After fine-tuning the model is saved as 'fashion_best.pt' by default in the project's root directory. This model can then be used to inference or evauluation purposes.
10. To perform inference on an image using the model execute Step 3 of the Setup Instructions.
11. To perform evaluation on a testing dataset first put the testing images inside the 'test_dataset/images' folder of the project's root directory.
12. Then run annotate_ground_truth.py to annotate the ground truth images which uses the fine-tuned model with a low confidence of 0.15 to maximise recall. The annoated review images, labels and the annotation summary JSON file will be saved in the 'test_dataset' folder of the project's root directory by default. 
13. Manually review the annotated images and edit them if needed using the UI by running: python ground_truth_editor.py.
14. Perform the evaluation on the annotated and reviewed test dataset by running: python evaluate.py
15. By default the confidence is set to 0.30. This is can be adjusted by adding the '--conf' arguement  to the evaluate.py command followed by a number between 0 and 1.
16. By default the NMS IoU is set to 0.55 (implying overlapping predictions with IoU greater than 0.55 are considered duplicates and may be suppressed). This is can be adjusted by adding the '--iou-nms' arguement to the evaluate.py command followed by a number between 0 and 1.
17. By default the IoU Threshold which controls matching predictions to ground-truth boxes after inference (a prediction must have at least this IoU with an unmatched ground-truth box to count as a match) is set to 0.50. This can be adjusted by adding the '--iou-thresh' arguement to the evaluate.py command followed by a number between 0 and 1.
18. After a successfull evaluation of the testing dataset the evaluation report which contains the comparison images, confusion matrices, evaluation report JSON file and the per class metrics containing the Precision, Recall and F1 Scores of all the categories are saved to 'eval_results' folder in the project's root directory by default.

Evaluation Results:

The testing dataset has been populated and annotated. The evaluation report of the model on this testing dataset is present inside the 'eval_results' folder in the project's root directory. Additionally the testing dataset can be used on other images by deleting the existing images, labels, reviews and annotation summary present inside the 'test_dataset' folder and then adding new testing images to the 'test_dataset/images' folder. After this follow Approach points 12 and 13 to annotate the testing images. Then the evaluation can be run on this test dataset.

Future Improvements:

The following improvements could be implemented to improve the model's overall performance which were presently not implemented due to hardware constraints and limited time:

1. Using a more powerful base model like YOLO11m or YOLO11l.
2. Increasing the training dataset size, especially the jewellery images where the model currently struggles due to lack of adequate images in the training dataset espcially of rings and bracelets.
3. Increasing the image size (imgsz) from 640 to 960 or 1280.
4. Increasing the number of epochs from 30 to 50 - 80.
