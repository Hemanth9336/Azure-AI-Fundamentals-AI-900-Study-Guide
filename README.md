# Azure-AI-Fundamentals-AI-900-Study-Guide
Comprehensive study guide for Azure AI Fundamentals (AI-900) certification

Welcome to the Azure AI Fundamentals (AI-900) Study Guide! This repository contains comprehensive notes, practice tests, and resources to help you prepare for the AI-900 certification exam.

## Table of Contents
1. Introduction
2. Study Notes
3. Practice Tests
4. Additional Resources

## Introduction
This guide is designed to help you understand the key concepts and prepare effectively for the AI-900 exam. It covers the following topics:
- Describe Artificial Intelligence workloads and considerations
- Describe fundamental principles of machine learning on Azure
- Describe features of computer vision workloads on Azure
- Describe features of Natural Language Processing (NLP) workloads on Azure
- Describe features of generative AI workloads on Azure

### Describe Artificial Intelligence Workloads and Considerations

**Artificial Intelligence (AI)** refers to the simulation of human intelligence in machines that are programmed to think and learn. AI workloads involve tasks that require human-like cognitive functions such as learning, reasoning, problem-solving, perception, and language understanding. Key considerations for AI workloads include:
- **Data Quality and Quantity**: Ensuring the availability of high-quality and sufficient data for training AI models.
- **Ethics and Bias**: Addressing ethical concerns and mitigating biases in AI models to ensure fairness and inclusiveness.
- **Performance and Scalability**: Ensuring that AI solutions can handle large-scale data and perform efficiently.
- **Security and Privacy**: Protecting sensitive data and ensuring compliance with privacy regulations.

### Describe Fundamental Principles of Machine Learning on Azure

**Machine Learning (ML)** is a subset of AI that involves the use of algorithms and statistical models to enable computers to learn from and make predictions based on data. On Azure, the fundamental principles of ML include:
- **Supervised Learning**: Training models on labeled data, where the input and output are known. Common techniques include regression and classification.
- **Unsupervised Learning**: Training models on unlabeled data to identify patterns and relationships. Common techniques include clustering and dimensionality reduction.
- **Model Training and Evaluation**: Using training data to build models and validation data to evaluate their performance.
- **Deployment and Monitoring**: Deploying trained models to production environments and monitoring their performance over time.

### Describe Features of Computer Vision Workloads on Azure

**Computer Vision** is a field of AI that enables machines to interpret and make decisions based on visual data. On Azure, computer vision workloads include:
- **Image Classification**: Categorizing images based on their content using deep learning models.
- **Object Detection**: Identifying and locating objects within an image using bounding boxes.
- **Optical Character Recognition (OCR)**: Extracting text from images and documents.
- **Facial Recognition**: Detecting and analyzing human faces in images, including identifying attributes such as age and emotion.
- **Semantic Segmentation**: Classifying each pixel in an image to understand the objects and their boundaries.

### Describe Features of Natural Language Processing (NLP) Workloads on Azure

**Natural Language Processing (NLP)** is a field of AI that focuses on the interaction between computers and human language. On Azure, NLP workloads include:
- **Text Analysis**: Extracting insights from text data, such as sentiment analysis, key phrase extraction, and entity recognition.
- **Language Understanding**: Using models like Language Understanding Service (LUIS) to interpret and understand user intents and entities in natural language.
- **Machine Translation**: Translating text from one language to another using services like Azure Translator.
- **Speech Recognition and Synthesis**: Converting spoken language into text (speech-to-text) and generating spoken output from text (text-to-speech).

### Describe Features of Generative AI Workloads on Azure

**Generative AI** involves creating new content, such as images, text, or music, using AI models. On Azure, generative AI workloads include:
- **Image Generation**: Using models like DALL-E to generate images from textual descriptions.
- **Text Generation**: Using models like GPT-4 to generate coherent and contextually relevant text based on input prompts.
- **Content Creation**: Creating new and original content, such as articles, stories, or code, using generative models.
- **Ethical Considerations**: Ensuring that generative AI models are used responsibly, addressing potential biases, and mitigating harmful content.

## Study Notes
The study notes are organized by topic for easy reference:
- Artificial Intelligence Workloads and Considerations
- Machine Learning on Azure
- Computer Vision Workloads on Azure
- Natural Language Processing (NLP) Workloads on Azure
- Generative AI Workloads on Azure

# Artificial Intelligence Workloads and Considerations

- **Artificial Intelligence (AI)**: Computer programs that respond in ways normally associated with human reasoning, learning, and thought.
- **Azure AI Services**: A portfolio of AI services that can be incorporated into applications quickly and easily without specialist knowledge. It includes a multi-service resource created in the Azure portal that provides access to several different Azure AI services with a single key and endpoint.
- **API (Application Programming Interfaces)**: Enable software components to communicate, allowing one side to be updated without stopping the other from working.
- **Endpoint**: The location of a resource, such as an Azure AI service.
- **Key**: A private string used to authenticate a request.
- **Multi-service Resource**: The AI service resource created in the Azure portal that provides access to a bundle of AI services.
- **Single-service Resource**: A resource created in the Azure portal that provides access to a single Azure AI service, such as Speech, Vision, Language, etc. Each Azure AI service has a unique key and endpoint.
- **RESTful API**: A scalable web application programming interface used to access Azure AI services.
- **Fairness**: Involves evaluating and mitigating the bias introduced by the features of a model.
- **Privacy**: Ensures that privacy provisions are included in AI solutions.
- **Transparency**: Provides clarity regarding the purpose of AI solutions, the way they work, and their limitations.
- **Accountability**: Ensures that AI solutions meet ethical and legal standards that are clearly defined.
- **Inclusiveness**: AI systems should empower everyone and engage people.

# Machine Learning on Azure

- **Machine Learning (ML)**: The study of computer algorithms that improve automatically through experience. It involves the ability for computer programs to learn from large amounts of data, in a process known as “training”.
- **Supervised Learning**: Humans label the data and provide general guidance. It includes regression and classification algorithms.
  - **Classification Algorithms**: Used to predict the category to which an input value belongs.
  - **Regression Algorithms**: Used to predict numeric values.
- **Unsupervised Learning**: The ability to find patterns in data without human help. It includes clustering algorithms.
  - **Clustering Algorithms**: Group data points that have similar characteristics.
- **Model**: A program that can recognize patterns in data, predict future behaviors, categorize items, recognize people, objects, and landmarks using unseen images, and understand the context of natural human text or speech.
- **Validation Dataset**: A sample of data held back from a training dataset, used to evaluate the performance of the trained model.
- **Cleaning Missing Data**: Detecting missing values and performing operations to fix the data or create new values.
- **Summarizing Data**: Providing summary statistics, such as the mean or count of distinct values in a column.
- **Evaluate Model**: A component used to measure the accuracy of trained models.
- **Pipeline in Machine Learning Designer**: Before training a machine learning model, you must create a pipeline, add a dataset, add training modules, and eventually deploy a service.
- **Automated Machine Learning (Automated ML)**: Requires a dataset to create a run.
- **Workspace**: Must be created before accessing Machine Learning Studio.
- **Deployment Targets**: An Azure container instance and an AKS cluster can be created after training a model.
- **Endpoint Deployment**: Deploy the best performing model for client applications to use over the internet by using an endpoint.
- **Compute Clusters**: Used to train the model and are created directly after creating a Machine Learning workspace.
- **Compute Resources in Azure ML Studio**: Include Compute clusters, Attached compute, Kubernetes clusters, and Compute Instances.
- **Azure ML Studio Authoring Tools**: Include Automated ML, Designer, and Notebooks.

# Computer Vision Workloads on Azure

- **Categorizing**: Associating the contents of an image with a limited set of categories.
- **The Read API**: Takes an image and extracts the words, organizing the results by page and line.
- **Bounding Box Coordinates**: Returned by the Azure AI Vision services for image processing.
- **Image Description Task**: Each phrase includes a confidence score.
- **Object Detection**: Generates bounding boxes identifying the locations of different types of vehicles in an image.
- **Image Classification**: Classifies images based on their contents using deep learning techniques.
- **OCR (Optical Character Recognition)**: Detects and reads text in images.
- **Spatial Analysis**: Part of the Azure AI Vision service.
- **Semantic Segmentation**: Classifies individual pixels in an image depending on the object they represent.
- **Facial Detection**: Detects and analyzes human faces in an image, including identifying a person’s age based on a photograph.
- **Pre-trained Models**: Provided by the computer vision service, eliminating the need for choosing, training, and evaluating a model.
- **Azure Resource for Computer Vision**: Must be created to use computer vision, involving inferencing. To create an inferencing cluster, use Machine Learning studio.
- **Indexer**: Converts documents into JSON and forwards them to a search engine for indexing.

# Natural Language Processing (NLP) Workloads on Azure

- **Natural Language Processing (NLP)**: Allows a machine to read and understand human language (e.g., machine translation, question answering, sentiment analysis).
- **GPT-4 and GPT-3.5**: Can understand and generate natural language and code but not images.
- **Embeddings**: An Azure OpenAI model that converts text into numerical vectors for analysis. Used to search, classify, and compare sources of text for similarity.
- **Whisper**: Can transcribe and translate speech to text.
- **Transcribing**: Part of speech recognition, converting speech into a text representation.
- **Language Service**: Returns values like Language Name, ISO 639-1 Code, and Score.
- **Removing Stop Words**: The first step in the statistical analysis of terms used in a text.
- **Counting Occurrences**: Takes place after stop words are removed.
- **Creating a Vectorized Model**: Used to capture the semantic relationship between words.
- **Encoding Words as Numeric Features**: Frequently used in sentiment analysis.
- **Sentiment Analysis**: Evaluates text and returns sentiment scores and labels for each sentence.
- **Tokenization**: Part of speech synthesis, breaking text into individual words for phonetic assignment.
- **Key Phrase Extraction & Lemmatization**: Extracts key phrases to identify main concepts in a text. Lemmatization, also known as stemming, is part of language processing.
- **Entity Recognition**: Includes entity linking functionality that returns links to external websites to disambiguate terms identified in a text.
- **Named Entity Recognition**: Identifies and categorizes entities in unstructured text, such as people, places, organizations, and quantities.
- **Entity Linking, PII Detection, and Sentiment Analysis**: Elements of the Azure AI Service for Azure AI Language.
- **Language Identification, Speaker Recognition, and Voice Assistants**: Elements of the Azure AI Speech service.
- **Text Translation and Document Translation**: Part of the Translator service.
- **Azure AI Speech Service**: Can generate spoken audio from a text source for text-to-speech translation.
- **Azure AI Translator Service**: Supports text-to-text translation in more than 60 languages.
- **Azure AI Content Moderator**: Checks text, image, and video content for potentially offensive material.
- **Speech Recognition**: The ability to detect and interpret speech.
- **Speech Synthesis**: The ability to generate spoken output (speech-to-text and text-to-speech).
- **Language Understanding Service (LUIS)**: Understands natural language with three core concepts: Utterances, Entities, and Intents.
- **Azure Bots**: Can operate over the web, email, social media, and voice.

# Generative AI Workloads on Azure

- **DALL-E**: A model that can generate images from natural language.
- **NLP**: Deals with identifying the meaning of written or spoken language, but not detecting or reading text in images.
- **Generative AI Models**: Offer the capability of generating images based on a prompt using DALL-E models, such as generating images from natural language.
- **Image Generation Models**: Can take a prompt, a base image, or both, and create something new. They can create both realistic and artistic images, change the layout or style of an image, and create variations of a provided image.
- **System Messages**: Should be used to set the context for the model by describing expectations. Based on system messages, the model knows how to respond to prompts.
- **OpenAI**: A research company that developed ChatGPT, a chatbot that uses generative AI models. Azure OpenAI provides access to many of OpenAI’s AI models.
- **AI Impact Assessment Guide**: Documents the expected use of the system and helps identify potential harms.
- **Content Filters**: Enable you to suppress harmful content at the Safety System layer.
- **Initial Release to a Restricted User Base**: Enables you to minimize harm by gathering feedback and identifying issues before broad release.
- **Identifying Potential Harms**: The first stage when planning a responsible generative AI solution.

# Model Evaluation Metrics

- **Confusion Matrix (Error Matrix)**: Provides a tabulated view of predicted and actual values for each class. Used as a performance assessment for classification models.
  - **True Positive (TP)**: The number of positive cases that the model predicted correctly.
  - **True Negative (TN)**: The number of negative cases that the model predicted correctly.
  - **False Positive (FP)**: The number of positive cases that the model falsely predicted.
  - **False Negative (FN)**: The number of negative cases that the model falsely predicted.
- **Recall Metric**: Defines how many positive cases that the model predicted are actually correct. Calculated using the formula: $$\text{Recall} = \frac{TP}{TP + FN}$$.
- **Accuracy**: Measures the overall correctness of the model. Calculated using the formula: $$\text{Accuracy} = \frac{TP + TN}{\text{Total number of cases}}$$.
- **Precision**: Measures the accuracy of the positive predictions. Calculated using the formula: $$\text{Precision} = \frac{TP}{TP + FP}$$.
- **F1 Score**: The harmonic mean of precision and recall. Calculated using the formula: $$\text{F1 Score} = \frac{2TP}{2TP + FP + FN}$$.
- **Selectivity (True Negative Rate)**: Measures the proportion of actual negatives that are correctly identified. Calculated using the formula: $$\text{Selectivity} = \frac{TN}{TN + FP}$$.

# Regression Model Evaluation Metrics

- **Root Mean Squared Error (RMSE)**: Represents the square root of the mean of the squared errors between predicted and actual values.
- **Mean Absolute Error (MAE)**: Measures how close the model’s predictions are to the actual values. A lower score indicates better model performance.
- **Coefficient of Determination (R²)**: Reflects the model’s performance. The closer R² is to 1, the better the model fits the data.
- **Relative Absolute Error (RAE)**: Measures the total absolute error relative to the total absolute error of a simple predictor.
- **Relative Squared Error (RSE)**: Measures the total squared error relative to the total squared error of a simple predictor.

# Clustering Model Evaluation Metrics

- **Number of Points**: Evaluates the number of data points in each cluster.
- **Combined Evaluation**: Assesses the overall quality of the clustering model.

## Practice Tests
Test your knowledge with these practice tests:
- Practice Test 1 - https://learn.microsoft.com/en-us/credentials/certifications/azure-ai-fundamentals/practice/results?assessmentId=26&practice-assessment-type=certification&snapshotId=62c4c8c3-7c91-40f5-afc0-8dc0dda87a6d
- Practice Test 2 - https://learn.microsoft.com/en-us/credentials/certifications/azure-ai-fundamentals/practice/results?assessmentId=26&practice-assessment-type=certification&snapshotId=6ad25a8f-bf4c-4225-911a-9efc477bab7c

## Additional Resources
Here are some additional resources to aid your study:
- Microsoft Learn: AI-900 Learning Path - https://learn.microsoft.com/en-us/credentials/certifications/azure-ai-fundamentals/?practice-assessment-type=certification
- YouTube Playlist: Azure AI Fundamentals - https://youtu.be/edEfRpQSjXs?si=Z2Km_nLBX9NhOY9k
- Github: https://github.com/olafwrieden/Azure-AI-900-Practice-Questions
- Blog: https://www.whizlabs.com/blog/microsoft-azure-ai-fundamentals-questions/


