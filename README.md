## End-to-End Image Classification Model Development

In this repository, a lightweight deep learning model was trained on the CIFAR10 dataset to classify images into 10 objects. Final model achieved 90% accuracy on 10,000 testing images. Model architecture was inspired by [Karen Simonyan, Andrew Zisserman
 - Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556). A simple REST API built in Python with Flask was used for model deployment.

<!-- TABLE OF CONTENTS -->
<summary><h2 style="display: inline-block">Table of Contents</h2>
  <ol>
    <li>
      <a href="#model-development">Model Development</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## Model Development

The model was developed on the CIFAR-10 dataset consists of 60,000 32x32 pixels colored images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. For more information, visit: https://www.cs.toronto.edu/~kriz/cifar.html. Centering each pixel was the only preprocessing step taken before model training.

Also, This tech report (Chapter 3) describes the dataset and the methodology followed when collecting it in much greater detail. [Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

Models were trained using AWS EC2, with a p3.x2large instance.

The model architecture were inspired by [Karen Simonyan, Andrew Zisserman, 2014](https://arxiv.org/abs/1409.1556). This paper examined very deep ConvNet models with very small (3x3) convolution filters in the large-scale image recognition setting. Their results are the famous VGG16/VGG19 models.

A VGG layer is defined as:

```python
model.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(2, 2))
```

Since our problem here is much more simpler than the one in the paper. Shallower models with 1 to 3 VGG layers followed by 2 dense layers were examined as baseline models.

<img src="images/base3_plot.png" alt="Baseline Model" width="600"/>

Each model was able to achieved almost 100% in training but only got around 70% in testing, which signals over-fitting. However, high accuracy in training is good news, it means our baseline models are able to picking up structures in the dataset and we are not likely to need more VGG layers.

3 VGG layers model were selected to proceed with because it was able to get the highest testing accuracy out of the three baseline models, also because it takes neglectable extra computation time(under 2 mins difference for training 100 epochs). After some research, it seems like ConvNet-Relu-BatchNorm is a common triplet widely used in image recognition. BatchNorm layers can also act as a regularizer in our case. Other than BatchNorm, L2 regularization, DropOut layers, and data augmentation were also examined. I used Kera's ImageDataGenerator to perform data augmentation on the fly. Configured as below, because our images are low resolution, milder data augmentation were used.

```python
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)
```

<img src="images/augmented_base3_plot.png.png" alt="Augmented Model" width="600"/>

Out of four methods tried, data augmentation raised testing accuracy by the most, from 70% to 84%. The other methods, BatchNorm, L2 regularization and DropOut, were also able to increase the accuracy by 12%, 5%, and 9%, separately. However, only data augmentation was able to let the model increase testing accuracy continuously by observable amount after 20~30 epochs.

<img src="images/augmented_VGG3_centered_75_plot.png" alt="Regularize Model" width="600"/>

Combining every regularize approaches above, and using a 0.001 learning rate, the model achieved 86.25% testing accuracy. Normalization, as an alternative to the centering preprocessing step we took earlier, was also evaluted with the same model architecture, and turned out normalization was able to achieve a slightly better result of 87.33%. I decided to switch to normalization. Also, with a few tries, I opted to use 0.0003 as the learning rate for a 250 epochs model.

<img src="images/augmented_VGG3_normalized_3_250.png" alt="Regularize Model" width="600"/>

The model ended up at 88.9% testing accuracy, but achieved 90.14% at the 249th epochs, the accuracy wiggled around 89% at the last 30 epochs, so I decided to lower the learning rate to 0.0001 and let the model run for 25 extra epochs.

<img src="images/augmented_VGG3_normalized_3_250_1_25_plot.png" alt="Regularize Model" width="600"/>

The model hit 90% consistently in the 25 extra epochs. This is our final model and we will use this model to build our REST API.



Here's a blank template to get started:
**To avoid retyping too much info. Do a search and replace with your text editor for the following:**
`github_username`, `repo_name`, `twitter_handle`, `email`, `project_title`, `project_description`

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
2. Install NPM packages
   ```sh
   npm install
   ```



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/github_username/repo_name/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* []()
* []()
* []()





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username
