## CASCADE: Contextual Sarcasm Detection in Online Discussion Forums

Code for the paper [CASCADE: Contextual Sarcasm Detection in Online Discussion Forums](https://arxiv.org/abs/1805.06413) (COLING 2018).

### Requirements
Code is written in Python (2.7 or 3) and requires Tensorflow (1.4.0) 

### Description

In this paper, we propose a ContextuAl SarCasm DEtector (CASCADE), which adopts a hybrid approach of both content- and context-driven modeling for sarcasm detection in online social media discussions (Reddit).
 
### Running CASCADE

Download the dataset file: `comments.json` from this [link](https://drive.google.com/file/d/1ew-85sh2z3fv1yGgIwBoeIHUvP8fMnxU/view?usp=sharing) and save it inside folder: `./CASCADE/data/`

#### User Embeddings: Stylometric features

The file `comments.json` has users and their corresponding tweets. Per user, there might be multiple number of tweets. Hence we concatenate all the tweets corresponding to a user with the '<END>' tag:

```
1. cd users
2. python create_per_user_paragraph.py
```

The ParagraphVector algorithm is used to generate the stylometric features. First, train the model:

```
3. python train_stylometric.py
```
generate `user_stylometric.csv` (user stlyometric features) using the trained model: 
```
4. python generate_stylometric.py
```

#### User Embeddings: Personality features

Pre-train a cnn-based model to detect personality features from text:
```
5. python process_data.py
6. python train_personality.py
```
To use the pre-trained model from our experiments, download the model weights: [personality_model_weights.zip](https://drive.google.com/file/d/1KK0p6tStgaEXLtAni1u3_W2jGlq8g1Nq/view?usp=sharing)  

and unzip inside folder: `./CASCADE/user/`

generate `user_personality.csv` (user personality features) using this model:

```
7. python generate_user_personality.py
```

#### User Embeddings: Multi-view fusion

Merge the `user_stylometric.csv` and `user_personality.csv` into a single merged `user_view_vectors.csv` file:
```
8.	python merge_user_views.py
```
Multi-view fusion of the user views (stylometric and personality) is performed using GCCA (~ CCA for two views). Generate fused user embeddings `user_gcca_embeddings.npz` using the following command:

```
9. python user_wgcca.py --input ./user_embeddings/user_view_vectors.csv --output ./user_embeddings/user_gcca_embeddings.npz --k 100 --no_of_views 2
```
This implementation of gcca has been adapted from <https://github.com/abenton/wgcca> .

***











