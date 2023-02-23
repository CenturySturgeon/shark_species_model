import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('sharks.pkl')

labels = learn.dls.vocab
def predict(img):
    #img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

import gradio as gr
title = "Shark Breed Classifier"
description = "A shark species classifier trained on Lautar's shark species dataset on kaggle with fastai. Created as a demo for Gradio and HuggingFace Spaces."
article="<p style='text-align: center'><a href='https://www.kaggle.com/datasets/larusso94/shark-species' target='_blank'>Blog post</a></p>"
enable_queue=True
gr.Interface(fn=predict,inputs=gr.inputs.Image(shape=(512, 512)),outputs=gr.outputs.Label(num_top_classes=3),title=title,description=description,article=article,enable_queue=enable_queue).launch(share=True)