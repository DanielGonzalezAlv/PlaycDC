![picture](poster/figures/playcdc.png)

Playing card detection and classification toolkit.

## Main Objectives

- **Create a general dataset** of a standard, 52-card deck of playing cards  in different poses, brightness situations and blurring levels annotated with bounding boxes around the ranks and suits and corresponding class information.
- **Train an object detection algorithm** on these synthesized data that performs bounding box localization and regression for classification. In particular, we train  the latest iteration of the YOLO object detection algorithm end-to-end.
- **Evaluate the algorithm on a hold-out validation dataset** covering all classes. As a performance metric, mean Average Precision (mAP) is used.
- **Deploy the model on a smartphone camera** as a proof of concept.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
For more information about the usage and results, see report & poster.

### Prerequisites

What things you need to install the software and how to install them

## Dataset Creation

Create your own card dataset.
- Use paste_canvas.py to perform blurring, sharpening, change of lighting to
  the cards and paste on convas.
- Use generate_data to perform rotations, translations and cropping of the
  image.

![picture](poster/figures/data_creation.jpg)


