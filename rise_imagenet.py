# https://colab.research.google.com/github/dianna-ai/dianna/blob/main/tutorials/rise_imagenet.ipynb#scrollTo=ab3bd199

import warnings
warnings.filterwarnings('ignore')  # disable warnings relateds to versions of tf
from typing import Optional
import pandas as pd
from model import Model
import numpy as np
from pathlib import Path
from keras import utils
import dianna
from dianna import visualization
import cv2


# for plotting
def explain_painting(
        image_path: Path = Path('data/0_Edinburgh_Nat_Gallery.jpg'),
        p_keep: float = 0.1,
        n_masks: int = 10,
        feature_res: int = 6,
        file_name_appendix: Optional[str] = None,
):
    model = Model()
    labels = [0, 1]
    
    file_name_base = create_file_name_base(feature_res, file_name_appendix, image_path, n_masks, p_keep)
        
    # The image will be read as a numpy array (in BGR format by default)
    x = cv2.imread(str(image_path))  # Convert Path to string

    if x is None:
        raise ValueError(f"Image not found at {image_path}")

    # Convert the image from BGR to RGB if needed
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

    print("----------------")
    print(f"Image type: {type(x)}, Image shape: {x.shape}")
        
    relevances = dianna.explain_image(model.run_on_batch, 
                                      x, # input_image : np.ndarray (Image data to be explained)                                      
                                      method = "RISE",
                                      labels = labels,
                                      n_masks = n_masks, 
                                      feature_res = feature_res, 
                                      p_keep = p_keep,
                                      axis_labels = {2: 'channels'})

    class_name(np.argmax(model.run_on_batch(x[None, ...])))

    # Visualize the relevance scores for the predicted class on top of the input image.
    predictions = model.run_on_batch(x[None, ...])

    for class_idx in labels:
        relevance_map = relevances[class_idx]
        print(f'Explanation for `{class_name(class_idx)}` ({predictions[0][class_idx]}), '
              f'relevances: min={np.min(relevance_map)}, max={np.max(relevance_map)}, mean={np.mean(relevance_map)}')

        # Use OpenCV to normalize and save the image (replace utils.img_to_array)
        # Assuming img is the original image already in np.ndarray
        img_normalized = x / 255.0  # OpenCV loads in range [0, 255], normalize to [0, 1] 
        
        visualization.plot_image(relevance_map, img_normalized, heatmap_cmap='jet',
                                 output_filename=file_name_base + f'_{class_name(class_idx)}.png', show_plot=False)
    np.savez_compressed(file_name_base + '.npz', relevances=relevances)


def create_file_name_base(feature_res, file_name_appendix, image_path, n_masks, p_keep):
    file_name_elements = [image_path.name,
                          'nmasks', str(n_masks),
                          'pkeep', str(p_keep),
                          'res', str(feature_res)
                          ]
    if file_name_appendix:
        file_name_elements.append(file_name_appendix)
    return '_'.join(file_name_elements)

#TODO: this function is not used
def load_img(path):
    img = utils.load_img(path) # Loads an image into PIL format.
    x = utils.img_to_array(img) # Converts a PIL Image instance to a Numpy array.
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
    
    # set to True to test. If correct, set to false asnd run real analysis
    is_classification_run = True
    
    if is_classification_run:
        paths = [Path(p) for p in ['data/0_Edinburgh_Nat_Gallery.jpg',
                                   'data/Madrid_Prado.jpg',
                                   #'data/0_Edinburgh_Nat_Gallery_100x100.jpg',
                                   'data/Warsaw_National Museum.jpg',
                                   #'data/Italian_Holy_Family_with_the_lamb_replica_100x100.jpg',
                                   "data/Not Rapheal/Lely #3 - Mary Framington - Christie's sale- edited copy.jpg",
                                   ]]
        results = []
        for path in paths:
            model = Model()

            result = model.run_on_batch(path)
            results.append(result)

        for path, result in zip(paths, results):
            print(f'{result=}')
            print(f'{path=}')
            print(pd.DataFrame([result], columns=[class_name(idx) for idx in [0, 1]]))

    else:
        painting_paths = [Path(p) for p in ['data/0_Edinburgh_Nat_Gallery.jpg', 'data/Madrid_Prado.jpg']]
        for painting_path in painting_paths:
            for n_masks in [10]:  #5000 ipv 10 in case code is correct; results are then more stable. start with 500, when the image doens not change, 500 would be enough 
                # alleen 0.7; 1 waarde kiezen voor bij final run
                for p_keep in [0.7, 0.9, 0.95]: # verhouding mask vs non mask pixels
                    #alleen 3 ; 1 waarde kiezen voor bij final run
                    for feature_res in [3, 6, 12]: # als je maskeert, wil je groepen maskeren die naast gelegen zijn
                        for run in range(3):
                            # heatmaps for the painting indicating the relevance of each pixel for the prediction
                            explain_painting(image_path         = painting_path,
                                             p_keep             = p_keep,
                                             n_masks            = n_masks,                                             
                                             feature_res        = feature_res,
                                             file_name_appendix = str(run),
                                             )
