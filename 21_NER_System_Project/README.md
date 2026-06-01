# Transformer-Based Named Entity Recognition System

A production-ready Named Entity Recognition (NER) system powered by Hugging Face Transformers. The system automatically identifies and classifies entities such as Persons, Organizations, Locations, Dates, and Time expressions from unstructured text. It supports multiple Transformer models, confidence-based filtering, entity visualization, batch processing, and an interactive web dashboard for real-world NLP applications.

---

## Overview

Named Entity Recognition (NER) is one of the most important tasks in Natural Language Processing (NLP). It enables machines to identify and categorize key information within text, such as names of people, organizations, locations, dates, and time expressions.

This project leverages state-of-the-art Transformer models from Hugging Face to build a robust and scalable NER system capable of handling both individual text inputs and large datasets through batch processing.

---

## Key Features

### Core Functionality

- Transformer-based Named Entity Recognition
- Detection of Person, Organization, and Location entities
- Date and Time extraction using enhanced regex patterns
- Support for multiple entities within a sentence
- Confidence score display for every prediction
- Confidence threshold filtering
- Interactive text-based entity extraction
- Structured output generation

### Advanced Features

- Small Model Support (`dslim/bert-base-NER`)
- Large Model Support (`dbmdz/bert-large-cased-finetuned-conll03-english`)
- GPU and CPU auto-detection
- Batch CSV processing
- CSV export functionality
- Downloadable results
- Entity highlighting with custom colors
- Duplicate entity prevention
- Professional Streamlit dashboard
- spaCy vs Transformer comparison
- Clean modular architecture
- Production-ready code structure

---

## Bonus Features Implemented

| Feature | Status |
|----------|----------|
| Large Transformer Model Integration | ✅ |
| Small Model Support | ✅ |
| Confidence Score Filtering | ✅ |
| Entity Highlighting | ✅ |
| Streamlit Dashboard | ✅ |
| CSV Export | ✅ |
| Batch CSV Processing | ✅ |
| Download Results | ✅ |
| spaCy Comparison | ✅ |
| GPU Acceleration Support | ✅ |
| Duplicate Time Detection Fix | ✅ |
| Professional User Interface | ✅ |

---

## Technologies Used

### Machine Learning & NLP

- Hugging Face Transformers
- PyTorch
- spaCy
- Regular Expressions (Regex)

### Data Processing

- Pandas
- NumPy

### Visualization & Deployment

- Streamlit
- IPython Display

---

## Models Used

### Small Model

```text
dslim/bert-base-NER
```

Recommended for:

- Faster inference
- Limited memory environments
- CPU execution
- Development and testing

---

### Large Model

```text
dbmdz/bert-large-cased-finetuned-conll03-english
```

Recommended for:

- Higher prediction quality
- Production environments
- GPU-enabled systems
- Complex text analysis

---

## Supported Entity Types

| Entity Type | Description |
|------------|------------|
| PERSON | Names of individuals |
| ORGANIZATION | Companies, institutions, agencies |
| LOCATION | Countries, cities, regions, places |
| DATE | Calendar dates and date expressions |
| TIME | Time-related expressions |
| MISCELLANEOUS | Other recognized entities |

---

## Project Structure

```text
named_entity_recognition/
│
├── app.py
├── ner_system.ipynb
├── README.md
├── requirements.txt
│
├── outputs/
│   └── extracted_entities.csv
│
├── models/
│
└── assets/
```

---

## Installation

Install the required dependencies:

```bash
pip install transformers torch pandas streamlit spacy ipython
```

Download the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

### Compatibility Fix (If Required)

If you encounter a NumPy and spaCy compatibility issue:

```bash
pip uninstall -y spacy thinc numpy

pip install numpy==1.26.4
pip install spacy==3.7.5

python -m spacy download en_core_web_sm
```

---

## Running the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
ner_system.ipynb
```

Execute all cells sequentially.

---

## Running the Web Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

---

## Streamlit Dashboard

The web application includes:

- Text Entity Extraction Interface
- Batch CSV Processing Interface
- Small and Large Model Selection
- Confidence Threshold Control
- Real-Time Entity Highlighting
- Entity Statistics Dashboard
- CSV Export Functionality
- Downloadable Results
- Modern Dark-Themed User Interface

---

## Batch Processing Format

Upload a CSV file containing a column named:

```text
text
```

Example:

```csv
text
Elon Musk founded Tesla in California on 12 March 2024.
Google opened a new office in London at 10:30 AM.
Barack Obama visited Microsoft headquarters in Seattle.
```

---

## Example Input

```text
Elon Musk founded Tesla in California on 12 March 2024 at 10:30 AM.
```

---

## Example Output

| Entity | Label | Confidence |
|----------|----------|----------|
| Elon Musk | PERSON | 0.99 |
| Tesla | ORGANIZATION | 0.98 |
| California | LOCATION | 0.99 |
| 12 March 2024 | DATE | 1.00 |
| 10:30 AM | TIME | 1.00 |

---

## System Workflow

```text
Input Text
      │
      ▼
Transformer Model
      │
      ▼
Entity Detection
      │
      ▼
Date & Time Extraction
      │
      ▼
Confidence Filtering
      │
      ▼
Entity Highlighting
      │
      ▼
Structured Results
      │
      ▼
CSV Export & Visualization
```

---

## Confidence Threshold

The confidence threshold allows users to control the strictness of predictions.

```text
Lower Threshold  → Higher Recall
Higher Threshold → Higher Precision
```

Recommended value:

```text
0.70
```

---

## Output Files

All extracted entities are automatically stored in:

```text
outputs/extracted_entities.csv
```

This file contains:

- Extracted Entity
- Entity Type
- Confidence Score
- Source
- Original Text
- Timestamp

---

## Enhancements Included

- Dual-model architecture
- Enhanced date and time extraction
- Overlap-aware duplicate removal
- Improved entity visualization
- Batch processing pipeline
- Downloadable outputs
- Professional user interface
- Modular and maintainable codebase

---

## Requirements

```txt
numpy==1.26.4
spacy==3.7.5
torch
transformers
pandas
streamlit
ipython
```

---

## Project Status

| Component | Status |
|------------|----------|
| Transformer NER Pipeline | ✅ Complete |
| Small Model Integration | ✅ Complete |
| Large Model Integration | ✅ Complete |
| Date & Time Extraction | ✅ Complete |
| Confidence Filtering | ✅ Complete |
| Entity Highlighting | ✅ Complete |
| CSV Export | ✅ Complete |
| Batch Processing | ✅ Complete |
| Streamlit Dashboard | ✅ Complete |
| spaCy Comparison | ✅ Complete |
| Production-Level Structure | ✅ Complete |
| Bonus Features | ✅ Complete |

---

## Author

**Saif Ullah**

---

### Conclusion

This project demonstrates the practical application of Transformer-based Natural Language Processing for Named Entity Recognition. By combining advanced Transformer architectures, intelligent date and time extraction, interactive visualization, and scalable batch processing, the system provides a complete end-to-end solution suitable for educational, research, and real-world NLP workflows.