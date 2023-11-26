
import json

city_anno_path = '/home/wong/Documents/DataSet/CityPersonFull/annotations/'
city_img_path = '/home/wong/Documents/DataSet/CityPersonFull/'
# train_file = 'custom_train.json'
train_file = 'custom_train_full_binary_debug.json'
# val_file = 'custom_val.json'
val_file = 'custom_val_full_binary_debug.json'

with open(city_anno_path + train_file) as f:
    coco_city_train = json.load(f)
coco_city_train


# In[31]:


import torchvision
import torch
import os

torch.set_float32_matmul_precision('medium')

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_path, anno_path, processor, train=True):
        ann_file = os.path.join(anno_path, train_file) if train else os.path.join(anno_path, val_file)
        super(CocoDetection, self).__init__(root=img_path, annFile = ann_file)
        self.processor = processor

    def __getitem__(self, idx): 
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target


# Based on the class defined above, we create training and validation datasets.

# In[32]:


from transformers import DetrImageProcessor, DeformableDetrImageProcessor
from transformers import DeformableDetrImageProcessor

# processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
processor = DeformableDetrImageProcessor.from_pretrained("facebook/deformable-detr-detic")

train_dataset = CocoDetection(img_path=city_img_path + 'train/', anno_path=city_anno_path, processor=processor, train=True)
val_dataset = CocoDetection(img_path=city_img_path + 'val/', anno_path=city_anno_path, processor=processor, train=False)


# As you can see, this dataset is tiny:

# In[33]:


print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))


# Let's verify an example by visualizing it. We can access the COCO API of the dataset by typing `train_dataset.coco`. 

# In[34]:


import numpy as np
import os
from PIL import Image, ImageDraw

# based on https://github.com/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb
image_ids = train_dataset.coco.getImgIds()
# let's pick a random image
image_id = image_ids[np.random.randint(0, len(image_ids))]
print('Image n°{}'.format(image_id))
image = train_dataset.coco.loadImgs([image_id])[0]
# print(image.keys())
print(image)
image = Image.open(os.path.join(city_img_path, 'train', image['file_name']))

annotations = train_dataset.coco.imgToAnns[image_id]
draw = ImageDraw.Draw(image, "RGBA")

cats = train_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}

for annotation in annotations:
  box = annotation['bbox']
  class_idx = annotation['category_id']
  x,y,w,h = tuple(box)
  draw.rectangle((x,y,x+w,y+h), outline='red', width=4)
  draw.text((x, y), id2label[class_idx], fill='white')
  # print(f"find {id2label[class_idx]} with category_id = {class_idx}")

image


# Next, let's create corresponding PyTorch dataloaders, which allow us to get batches of data. We define a custom `collate_fn` to batch images together. As DETR resizes images to have a min size of 800 and a max size of 1333, images can have different sizes. We pad images (`pixel_values`) to the largest image in a batch, and create a corresponding `pixel_mask` to indicate which pixels are real (1)/which are padding (0).  

# In[35]:


from torch.utils.data import DataLoader

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = processor.pad(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

# set batch_size = 4 in original DETR
train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2)
batch = next(iter(train_dataloader))


# Let's verify the keys of a single batch:

# In[36]:


batch.keys()


# Let's verify the shape of the `pixel_values`, and check the `target`::

# In[37]:


pixel_values, target = train_dataset[0]


# In[38]:


pixel_values.shape


# In[39]:


print(target)


# ## Train the model using PyTorch Lightning
# 
# Here we define a `LightningModule`, which is an `nn.Module` with some extra functionality.
# 
# For more information regarding PyTorch Lightning, I recommend the [docs](https://pytorch-lightning.readthedocs.io/en/latest/?_ga=2.35105442.2002381006.1623231889-1738348008.1615553774) as well as the [tutorial notebooks](https://github.com/PyTorchLightning/lightning-tutorials/tree/aeae8085b48339e9bd9ab61d81cc0dc8b0d48f9c/.notebooks/starters). 
# 
# You can of course just train the model in native PyTorch as an alternative.

# In[40]:


import pytorch_lightning as pl
from transformers import DetrConfig, DetrForObjectDetection
from transformers import DetrConfig, DeformableDetrForObjectDetection

class DeformableDetr(pl.LightningModule):
     def __init__(self, lr, lr_backbone, weight_decay):
         super().__init__()
         # replace COCO classification head with custom head
         # we specify the "no_timm" variant here to not rely on the timm library
         # for the convolutional backbone
         self.model = DeformableDetrForObjectDetection.from_pretrained("facebook/deformable-detr-detic",
                                                             num_labels=len(id2label),
                                                             ignore_mismatched_sizes=True)
         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
         self.lr = lr
         self.lr_backbone = lr_backbone
         self.weight_decay = weight_decay

     def forward(self, pixel_values, pixel_mask):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

       return outputs
     
     def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       pixel_mask = batch["pixel_mask"]
       labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

     def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())

        return loss

     def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item())

        return loss

     def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)
        
        return optimizer

     def train_dataloader(self):
        return train_dataloader

     def val_dataloader(self):
        return val_dataloader


# As PyTorch Lightning by default logs to Tensorboard, let's start it:

# In[41]:


# Start tensorboard.
# %load_ext tensorboard
# %tensorboard --logdir C:\Users\16415\WorkSpace\Jupyter\DETR-Notebook\lightning_logs\version_15


# Here we define the model, and verify the outputs.

# In[42]:


model = DeformableDetr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])


# The logits are of shape `(batch_size, num_queries, number of classes + 1)`. We model internally adds an additional "no object class", which explains why we have one additional output for the class dimension. 

# In[43]:


outputs.logits.shape


# Next, let's train! We train for a maximum of 300 training steps, and also use gradient clipping. You can refresh Tensorboard above to check the various losses.

# In[44]:


from pytorch_lightning import Trainer

trainer = Trainer(accelerator="auto", max_steps=100, gradient_clip_val=0.1)
trainer.fit(model)


# ## Push to the hub
# 
# We can simply call `push_to_hub` on our model and image processor after training to upload them to the 🤗 hub. Note that you can pass `private=True` if you don't want to share the model with the world (keep the model private).
# 
#  Alternatively, you could also define a custom callback in PyTorch Lightning to automatically push the model to the hub every epoch for instance, or at the end of training. See my [Donut fine-tuning notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Donut/CORD/Fine_tune_Donut_on_a_custom_dataset_(CORD)_with_PyTorch_Lightning.ipynb) regarding this.
# 
# Here we'll do it manually.

# In[ ]:


# !huggingface-cli login
# push_path_for_model = "WongChiang/detr_cityperson"
push_path_for_model = "WongChiang/deformable_detr_cityperson"
# push_path_for_processor = "WongChiang/detr-cityperson"
push_path_for_processor = "WongChiang/deformable-detr-cityperson"


# In[ ]:


# model.model.push_to_hub(push_path_for_model)
# processor.push_to_hub(push_path_for_processor)


# We can easily reload the model, and move it to the GPU as follows:

# In[ ]:


from transformers import DetrImageProcessor, DetrForObjectDetection

model = DeformableDetrForObjectDetection.from_pretrained("WongChiang/deformable_detr_cityperson", id2label=id2label)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
processor = DeformableDetrImageProcessor.from_pretrained("WongChiang/deformable-detr-cityperson")


# ## Evaluate the model
# 
# Finally, we evaluate the model on the validation set. For this we make use of the `CocoEvaluator` class available in a [tiny PyPi package](https://github.com/NielsRogge/coco-eval) I made. This class is entirely based on the original evaluator class used by the DETR authors.

# In[ ]:


# !pip install -q coco-eval


# To run the evaluation, we must make sure that the outputs of the model are in the format that the metric expects. For that we need to turn the boxes which are in (x1, y1, x2, y2) format into (x, y, width, height), and turn the predictions into a list of dictionaries:

# In[ ]:


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


#  Let's run the evaluation:

# In[ ]:


from coco_eval import CocoEvaluator
from tqdm.notebook import tqdm

# initialize evaluator with ground truth (gt)
evaluator = CocoEvaluator(coco_gt=val_dataset.coco, iou_types=["bbox"])

print("Running evaluation...")
for idx, batch in enumerate(tqdm(val_dataloader)):
    # get the inputs
    pixel_values = batch["pixel_values"].to(device)
    pixel_mask = batch["pixel_mask"].to(device)
    labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

    # forward pass
    with torch.no_grad():
      outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    
    # turn into a list of dictionaries (one item for each example in the batch)
    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    # default threshold = 0.5, modify it after converged
    results = processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=0.1)
    
    # provide to metric
    # metric expects a list of dictionaries, each item 
    # containing image_id, category_id, bbox and score keys 
    predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
    predictions = prepare_for_coco_detection(predictions)
    # print(predictions)
    evaluator.update(predictions)

evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()


# ## Inference (+ visualization)
# 
# Let's visualize the predictions of DETR on the first image of the validation set.

# In[ ]:


#We can use the image_id in target to know which image it is
pixel_values, target = val_dataset[10]


# In[ ]:


pixel_values = pixel_values.unsqueeze(0).to(device)
print(pixel_values.shape)


# In[ ]:


with torch.no_grad():
  # forward pass to get class logits and bounding boxes
  outputs = model(pixel_values=pixel_values, pixel_mask=None)
print("Outputs:", outputs.keys())
print(outputs)


# In[ ]:


import matplotlib.pyplot as plt

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


# In[ ]:


# load image based on ID
image_id = target['image_id'].item()
image = val_dataset.coco.loadImgs(image_id)[0]
image = Image.open(os.path.join(city_img_path+'train/', image['file_name']))

# postprocess model outputs
width, height = image.size
# threshold should be 0.9, modify it after converged
postprocessed_outputs = processor.post_process_object_detection(outputs,
                                                                target_sizes=[(height, width)],
                                                                threshold=0.2)
results = postprocessed_outputs[0]
print(results)
print(outputs)
plot_results(image, results['scores'], results['labels'], results['boxes'])


# In[ ]:




