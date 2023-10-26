# GAN_anime_faces Creator: Project Overview

- Created a tool that creates anime faces. This tool should help me and other developers to develop how GAN works.
- Downloaded existing Dataset from Kaggle.
- Optimized Conditional GAN to get the best model possible
- build a Tensorboard to see results

## Code and Resource Used

- **Python Version:** 3.10
- **Packages:** pytorch, torchvision, cv2, tqdm, unittest
- **Dataset:** [https://www.kaggle.com/datasets/splcher/animefacedataset](https://www.kaggle.com/datasets/splcher/animefacedataset)
- **Model Theorie:** [https://www.youtube.com/watch?v=Hp-jWm2SzR8&list=PLhhyoLH6IjfwIp8bZnzX8QR30TRcHO8Va&index=5](https://www.youtube.com/watch?v=Hp-jWm2SzR8&list=PLhhyoLH6IjfwIp8bZnzX8QR30TRcHO8Va&index=5)

## Dataset Building
In this phase I took the data and selected it by index. Afterwards I transformed it to PIL image then I resized it and after I transformed it to a tensor. Afterward I read it with cv2 and I returned it.

## Model Building

In this stage I built first the discriminator. The discriminator consists of 5 convoluntional layers and in between a InstanceNormalization and LeakyRelu.
After I built the generator which consists of 5 convtranspose. Between every convtranspose layer there is a BatchNorm and a Relu layer.






