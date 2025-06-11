## Model Training Process for Wildfire Smoke and Fire Detection

In the early stages of this project, I conducted a search for existing datasets containing images of smoke and fire that considered suitable for training a wildfire detection model. However, I was inititally unable to find any publically available datasets that met the specific criteria required for this application. My goal was to gather images depicting smoke and fire in wilderness settings, avoiding overly urban enviornments, with an emphasis on aerial views. This was essential to ensuring alighnent between the training data and the intended deployment of the final model.

Thus, I decided to create a custom dataset by sourcing and curating images from multiple repositories and open sources.

If you're interested in viewing the original Kaggle notebooks used throughout this project, they are available in the `model_creation_kaggle` folder.

#### Source 1: [Smoke-Fire-Detection-YOLO](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo)

I initially explored the Smoke and Fire Detection dataset by Sayed Gamal, which contains approximately 21,000 images. However, after visualizing the dataset, I found that it did not align with the specific requirements of my project. My goal was to detect smoke and fire from a distance in rural and natural environments. In contrast, this dataset included a mix of close-up shots, urban scenes, and varied perspectives that were not suitable for my intended use case.

To refine the dataset, I used CLIP with the prompt: 'an aerial drone view looking down at forests, mountains, or rural landscapes, where the ground and trees are visible but individual people and vehicles are small or absent.' I then filtered the results to include only images with a confidence score above 75%. This process reduced the dataset to approximately 5,800 images that better matched the visual criteria of my project.

Next, I used Roboflow to annotate and further filter the images. I manually annotated the first 1,000 images, then trained a preliminary model to assist with annotating the remaining data. Despite automation, I still had to review and re-annotate most of the images to ensure accuracy. Additionally, I removed many images that did not contribute meaningfully to the project. This process resulted in a dataset of 2,037 high-quality, labeled images.

Using this dataset, I trained my first model by applying transfer learning from TommyNgx’s YOLOv10 Fire and Smoke Detection model. I fine-tuned the model on my dataset for 75 epochs. While the evaluation metrics appeared promising, the model performed poorly when tested on truly unseen data. This revealed the need for a more diverse and representative dataset, prompting me to gather additional data.

#### Source 2. [wildfire-smoke-detection](https://www.kaggle.com/datasets/gloryvu/wildfire-smoke-detection)

After identifying the limitations of the Smoke and Fire Detection dataset by Sayed Gamal, I turned to the Wildfire Smoke Detection (WSDataset) available on Kaggle. The WSDataset contains over 11,000 annotated smoke images and 13,000 nonsmoke frames. However, since the dataset focuses exclusively on smoke and does not include fire, using all the smoke frames would have introduced a significant class imbalance in my project. To address this, I randomly selected 3,000 smoke images from the WSDataset to supplement my training data while maintaining a balanced distribution of smoke and fire examples.

Furthermore, upon reviewing the WSDataset annotations, I noticed that they did not align with my labeling approach. The annotations tended to focus only on the densest part of the smoke plume, whereas my methodology involved drawing bounding boxes around the entire visible smoke area. I chose this broader labeling strategy to maximize the model’s ability to detect smoke in a variety of conditions and visibility levels. Thus I had to annotate the images again. Following this, I trained two new models using the updated annotations to evaluate whether this more inclusive labeling approach would lead to improved detection performance.

#### Source 3. [Wildfire Smoke Dataset](https://public.roboflow.com/object-detection/wildfire-smoke)

To further improve dataset diversity, I incorporated additional images from the Wildfire Smoke dataset available on Roboflow. This dataset was particularly useful for including black-and-white and low-contrast images, which helped expand the model's ability to generalize across different visual conditions. After integrating this data and filtering for relevance and quality, my final dataset consisted of 5,767 annotated images. Using this complete dataset, I trained two final models—YOLOv11s and YOLOv11m—from scratch. 

While both models were trained and evaluated, YOLOv11s was selected as the primary model for production due to its favorable balance of performance and efficiency. YOLOv11m was used as a validation tool. The YOLOv11s model achieved strong performance, with an overall precision of 0.775, recall of 0.759, mAP50 of 0.79, and mAP50-95 of 0.446. It performed especially well on the smoke class, achieving a mAP50 of 0.84 and mAP50-95 of 0.52. Fire detection was more challenging, with a mAP50 of 0.74 and mAP50-95 of 0.372, likely due to the smaller number of fire-labeled instances. However, throughout the course of this project, I came to realize that smoke is often a more critical and earlier indicator of wildfires making it the more valuable target for detection. As a result, I was comfortable with the trade off in fire performance, as the model’s strength in detecting smoke made it highly effective for real-world wildfire monitoring scenarios.

## View the process
All of the Kaggle notebooks I used for this project, including dataset exploration, filtering, annotation strategies, and early experiments, have been saved in a separate folder named `model_creation_kaggle` within the project directory. These notebooks were exported directly from Kaggle to preserve the original workflow and analysis steps.