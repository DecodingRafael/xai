# https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/rise_imagenet.ipynb#scrollTo=ab3bd199

import warnings
from model import Model
warnings.filterwarnings('ignore')  # disable warnings relateds to versions of tf
import numpy as np
from pathlib import Path
from keras import utils
import dianna
from dianna import visualization

# for plotting
def main(
        image_path: Path = Path('../data/0_Edinburgh_Nat_Gallery.jpg'),
        p_keep: float = 0.1,
        n_masks: int = 10,
        feature_res: int = 6,
        file_name_appendix:str=None,
         ):

    model = Model()
    img, x = load_img(image_path)

    print(f"{model.run_on_batch(x[None, ...]).shape=}")
    labels = [0, 1]

    file_name_elements = [image_path.name,
                          'nmasks', str(n_masks),
                          'pkeep', str(p_keep),
                          'res', str(feature_res)
                          ]

    file_name_base = '_'.join(file_name_elements)
    relevances = dianna.explain_image(model.run_on_batch, x, method="RISE",
                                      labels=labels,
                                      n_masks=n_masks, feature_res=feature_res, p_keep=p_keep,
                                      axis_labels={2: 'channels'})

    class_name(np.argmax(model.run_on_batch(x[None, ...])))

    # Visualize the relevance scores for the predicted class on top of the input image.
    predictions = model.run_on_batch(x[None, ...])

    for class_idx in labels:
        relevance_map = relevances[class_idx]
        print(f'Explanation for `{class_name(class_idx)}` ({predictions[0][class_idx]}), '
              f'relevances: min={np.min(relevance_map)}, max={np.max(relevance_map)}, mean={np.mean(relevance_map)}')

        if file_name_appendix:
            file_name_elements.append(file_name_appendix)
        visualization.plot_image(relevance_map, utils.img_to_array(img) / 255., heatmap_cmap='jet',
                                 output_filename=file_name_base + f'_{class_name(class_idx)}.png', show_plot=False)
    np.savez_compressed(file_name_base + '.npz', relevances=relevances)


def load_img(path):
    img = utils.load_img(path)
    x = utils.img_to_array(img)
    return img, x

def class_name(idx):
    if idx == 0:
        name = 'Raphael'
    elif idx == 1:
        name = 'Non-Raphael'
    else:
        name = f'class_idx={idx}'
    return name

if __name__ == "__main__":
    is_classification_run = False
    if is_classification_run:
        model = Model()
        img, x = load_img(Path('../data/0_Edinburgh_Nat_Gallery.jpg'))
        result = model.run_on_batch(x[None,...])
        print(result)

    else:
        for n_masks in [100, 300, 1000, 3000]:
            for p_keep in [0.5]:
                for feature_res in [6]:
                    main(n_masks=n_masks, p_keep=p_keep, feature_res=feature_res)
