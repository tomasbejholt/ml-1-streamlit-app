# L01: Lay of the Land

Show what ML can do, introduce core terminology. Goal: a mental map of the field and confidence that this is approachable.

## Topics

**Tour of ML domains** — working demos across data types to show the breadth before zooming in:
- Tabular: sklearn on Titanic (predict survival from passenger features)
- Images: torchvision ResNet classifying photos
- Text: HuggingFace pipeline doing sentiment analysis or summarization

**Supervised learning** — the core pattern everything else builds on:
- Input → Model → Output
- The model learns a mapping from examples (features → labels)
- Training vs inference: learning vs using what you learned

**Core terminology** — introduce these naturally through the demos:
- Features: the input data (columns, pixels, tokens)
- Labels: what we're predicting (survived/died, cat/dog, positive/negative)
- Model: the learned function
- Training: adjusting the model to fit the data
- Inference: using the trained model on new data

**Brief landscape** — just enough to orient, not overwhelm:
- Trees, neural networks, transformers exist
- Different tools for different problems
- "By week 6, you'll build and deploy your own classifier"

**The overwhelm talk** — normalize the learning curve early. ML has a brutal amount of moving parts (terminology, math, code patterns, conceptual layers). Repetition is how it clicks. Concepts that feel impossible in week 2 become obvious by week 6. "Släpp sargen" — let go of the need to master every detail before moving forward.

## Terminology Introduced

features, labels, model, training, inference, supervised learning

## Resources

Decent tutorial on pytorch with some good videos:
https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
