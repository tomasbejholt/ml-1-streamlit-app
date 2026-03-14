# L01: Lay of the Land

Show what ML can do, introduce core terminology. Goal: a mental map of the field and confidence that this is approachable.

## Notebook Structure

The notebook opens with "What is ML?" (traditional programming vs ML), then goes straight into three live demos before any taxonomy. The demos make the terminology concrete before defining it.

1. **What is ML?** — Traditional rules vs learned rules (spam filter example)
2. **Demo 1: Titanic** — Full tabular workflow: explore → train → evaluate → predict
3. **Demo 2: Images** — ResNet classifying pet photos, RGB channel visualization
4. **Demo 3: Text** — Sentiment analysis + text generation with HuggingFace
5. **The overwhelm talk** — Normalize the learning curve, "släpp sargen"
6. **ML Landscape** — Supervised vs unsupervised, data types, model families (compact)
7. **Workflow** — The universal loop: Understand → Prepare → Train → Evaluate → Iterate → Deploy
8. **Vocabulary tables** — Reference definitions for features, labels, loss, etc.
9. **Course roadmap** — What's ahead

## Topics

**Tour of ML domains** — working demos across data types to show the breadth before zooming in:
- Tabular: Random Forest on Titanic (predict survival from passenger features)
- Images: torchvision ResNet classifying Oxford Pets photos
- Text: HuggingFace pipeline doing sentiment analysis + text generation

**Supervised learning** — the core pattern everything else builds on:
- Input → Model → Output
- The model learns a mapping from examples (features → labels)
- Training vs inference: learning vs using what you learned

**Core terminology** — introduce naturally through the demos:
- Features: the input data (columns, pixels, tokens)
- Labels: what we're predicting (survived/died, cat/dog, positive/negative)
- Model: the learned function
- Training: adjusting the model to fit the data
- Inference: using the trained model on new data

**Brief landscape** — one compact section after the demos, not three:
- Supervised vs unsupervised vs self-supervised
- Data types → best tools (tabular → trees, images → CNNs, text → transformers)
- Model families: linear, trees, neural networks

**The overwhelm talk** — normalize the learning curve early. ML has a brutal amount of moving parts. Repetition is how it clicks. "Släpp sargen."

## Terminology Introduced

features, labels, model, training, inference, supervised learning, classification, regression, overfitting, underfitting

## Lecture Notes

- The Titanic demo is the longest section and the best teaching moment — walk through it interactively, let students make predictions before the model does
- The image demo downloads Oxford Pets (~800MB) on first run — consider running this cell before the lecture starts
- The text demos download distilbert and distilgpt2 (~250MB) — same applies
- The "overwhelm talk" lands better mid-lesson (after the demos tire people out) than at the very start

## Resources

**"What is ML" overviews:**
- 3Blue1Brown — But what is a neural network? (visual, 19 min): https://www.youtube.com/watch?v=aircAruvnKk
- StatQuest — Machine Learning Fundamentals (clear, beginner-friendly): https://www.youtube.com/watch?v=Gv9_4yMHFhI
- Google ML Crash Course — Framing ML problems: https://developers.google.com/machine-learning/crash-course/framing/ml-terminology

**Deeper dives (for curious students, not required):**
- PyTorch tutorial series (Patrick Loeber): https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
- fast.ai Practical Deep Learning — Lesson 1 (top-down intro, similar philosophy): https://course.fast.ai/Lessons/lesson1.html
- Kaggle Learn — Intro to Machine Learning (hands-on, short): https://www.kaggle.com/learn/intro-to-machine-learning
