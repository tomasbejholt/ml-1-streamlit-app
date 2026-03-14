# L12: Fine-Tuning, Object Detection & Deployment

The practical "real world" image lesson. Fine-tune pretrained models, introduce object detection with YOLO, deploy via FastAPI. Closes the image section with a shipped, working image classifier API.

## Topics

### Part 1 — Fine-Tuning Pretrained Models

**Why train from scratch when someone already did?** Pretrained models learned general visual features on millions of images. We borrow that knowledge and adapt it.

**Pretrained models in torchvision:**
- ResNet, EfficientNet — load with one line
- These models learned edges → textures → parts → objects on ImageNet

**Feature extraction:**
- Freeze the backbone (all pretrained layers)
- Replace the final classification head with your own
- Train only the new head on your data
- Fast, works well with small datasets

**Fine-tuning:**
- Unfreeze some or all pretrained layers
- Use a lower learning rate (don't destroy what was learned)
- Fine-tune on custom dataset
- Compare accuracy vs from-scratch CNN — dramatic difference

### Part 2 — Object Detection with YOLO

**Classification vs detection:**
- Classification: "what is this image?" → one label
- Detection: "where is everything?" → multiple bounding boxes + labels

**YOLO as a practical tool** — focus on usage, not internals:
- Load pretrained YOLOv8
- Run inference on images — see bounding boxes
- Fine-tune on a custom dataset (defect detection, document elements, etc.)
- Most common commercial CV task: quality control, warehouse, retail, documents

### Part 3 — Deployment

**Save the model:**
- `torch.save()` and `state_dict()`
- What to save and what not to save

**FastAPI endpoint:**
- Accept image upload → preprocess → predict → return JSON
- FastAPI is familiar territory from the web dev stack

**Containerize and deploy:**
- Docker container with model + API
- Deploy to AWS EC2 instance
- Production considerations: input validation, error handling, model versioning

## Outcome

Fine-tune a pretrained classifier AND a YOLO detector, then deploy one as a working API.
