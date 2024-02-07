# https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/rise_imagenet.ipynb#scrollTo=ab3bd199

# libraries ----
import warnings

warnings.filterwarnings('ignore')  # disable warnings relateds to versions of tf
import numpy as np
from pathlib import Path

# keras model and preprocessing tools
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import load_model  # added to load resnet Raphael model
from keras import backend as K
from keras import utils
import joblib
# dianna library for explanation
import dianna
from dianna import visualization

# for plotting
from matplotlib import pyplot as plt


def main(
        image_path: Path = Path('0_Edinburgh_Nat_Gallery.jpg'),
        p_keep: float = 0.1,
        n_masks: int = 1000,
        feature_res: int = 6,
        file_name_appendix:str=None,
         ):
    # 1 - Loading the pretrained Raphael model

    # loading the resnet model used in the Raphael paper ( see here to get it):
    # https://www.dropbox.com/scl/fo/0e927qf6sjldoiaiatjey/h?rlkey=ohj36i4bnf7nou2zyfua6pim8&dl=0

    class Model():
        def __init__(self):
            ResNet_Path = Path("resnet50_model.h5")
            Model_Path = Path("28_09_2023_svm_final_model.pkl")
            K.set_learning_phase(0)
            self.resnet_model = load_model(ResNet_Path)

            # Load the final model
            self.svm_final = joblib.load(Model_Path)

            self.input_size = (224, 224)

        def load_and_preprocess_image(self, img: np.ndarray):
            return

        def extract_features(self, img, model):
            features = self.resnet_model.predict(preprocess_input(img))
            return features

        def run_on_batch(self, x):
            test_image_features = self.extract_features(x, model)

            # Use the loaded model to predict the category of the test image
            predictions = self.svm_final.predict(test_image_features)

            return np.array([[prediction, 1 - prediction] for prediction in predictions])

    model = Model()

    # Load and preprocess image to be explained
    def load_img(path):
        img = utils.load_img(path)
        x = utils.img_to_array(img)
        x = preprocess_input(x)
        return img, x

    # Call the function to load an image of a single instance in the test data from the `img` folder.
    img, x = load_img(image_path)
    # plt.imshow(img)

    print(f"{model.run_on_batch(x[None, ...]).shape=}")

    # 2 - Compute and visualize the relevance scores
    # Compute the pixel relevance scores using RISE and visualize them on the input image.

    # RISE masks random portions of the input image and passes the masked image through
    # the model — the masked portion that decreases accuracy the most is the most
    # “important” portion.#To call the explainer and generate relevance scores map,
    # the user need to specifiy the number of masks being randomly generated
    # (`n_masks`), the resolution of features in masks (`feature_res`)
    # and for each mask and each feature in the image, the probability of being kept
    # unmasked (`p_keep`).

    # takes about 35-40 minutes to run
    labels = [0, 1]
    relevances = dianna.explain_image(model.run_on_batch, x, method="RISE",
                                      labels=labels,
                                      n_masks=n_masks, feature_res=feature_res, p_keep=p_keep,
                                      axis_labels={2: 'channels'})

    # Make predictions and select the top prediction.
    def class_name(idx):
        if idx == 0:
            name = 'Raphael'
        elif idx == 1:
            name = 'Non-Raphael'
        else:
            name = f'class_idx={idx}'
        return name

    # print the name of predicted class, taking care of adding a batch axis to the model input
    class_name(np.argmax(model.run_on_batch(x[None, ...])))

    # Visualize the relevance scores for the predicted class on top of the input image.
    predictions = model.run_on_batch(x[None, ...])
    prediction_ids = np.argsort(predictions)[0][-1:-6:-1]
    prediction_ids

    file_name_base = '_'.join(file_name_elements)
    for class_idx in labels:
        relevance_map = relevances[class_idx]
        print(f'Explanation for `{class_name(class_idx)}` ({predictions[0][class_idx]}), '
              f'relevances: min={np.min(relevance_map)}, max={np.max(relevance_map)}, mean={np.mean(relevance_map)}')

        file_name_elements = [image_path, nmasks, n_masks, pkeep, p_keep, res, feature_res]
        if file_name_appendix:
            file_name_elements.append(file_name_appendix)
        visualization.plot_image(relevance_map, utils.img_to_array(img) / 255., heatmap_cmap='jet',
                                 output_filename=file_name_base + f'_{class_name(class_idx)}.png', show_plot=False)
    np.savez_compressed(file_name_base + '.npz', relevances=relevances)

    # Conclusions ----
    # The relevance scores are generated by passing multiple randomly masked inputs to the black-box model
    # and averaging their scores. The idea behind this is that whenever a mask preserves important parts
    # of the image it gets higher score.

    # The example here shows that the RISE method evaluates the relevance of each pixel/super pixel to
    # the classification. Pixels characterizing the bee are highlighted by the XAI approach, which gives
    # an intuition on how the model classifies the image. The results are reasonable, based on the human
    # visual preception of the image.


if __name__ == "__main__":
    main()
