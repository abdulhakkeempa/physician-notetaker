# **Physician Notetaker - Emitrr Assignment**

## **How to Run This Application?**

Follow these steps to set up and run the application:

### **1. Clone the Repository**
```
git clone https://github.com/abdulhakkeempa/physician-notetaker.git
```
2. Navigate to the Project Directory  
```
cd app
```
3. Create a Virtual Environment
```
python -m venv
```
4. Activate the Virtual Environment
  - Linux/Mac  
  ```
  source venv/bin/activate (Linux/Mac)
  ```
  - Windows
  ```
venv/Scripts/Activate (Windows)
```
5. Install Dependencies
 ```
pip install -r requirements.txt
```
6. Run the Application
```
python app.py
```

## **Screenshots**
![image](https://github.com/user-attachments/assets/8a2f6fcd-3427-462c-b701-8f1573ade95a)
![image](https://github.com/user-attachments/assets/34810849-5be2-44df-9839-ade1af040173)

## **My Approach to the Problem**
### Named Entity Recognition
#### **Overview**  
The objective was to identify a pre-trained model suitable for the given task. Several models were tested, and their performance is documented below.

#### **Experiments & Findings**  

**1. GLiNER (Zero-Shot NER Model)**  
- Performed reasonably well as a zero-shot model.  
- Had limitations in accurately identifying domain-specific entities.  

**2. `d4data/biomedical-ner-all` (Fine-Tuned on Biomedical Data)**  
- Specifically fine-tuned for biomedical data.  
- Did not fit well for the given use case.  

**3. `bert-base-uncased_clinical-ner`**  
- Contained entity classes closely aligned with our problem statement.  
- Lacked the **Prognosis** class, which was essential for the task.  
- Successfully handled all other required entity types.  

#### **Conclusion**  
Fine-tuning would have been a good option; however, due to the lack of annotated data for the specific classes, the focus remained on selecting a suitable pre-trained model.

### **Sentiment & Intent Classification**
#### **Overview**  
The goal was to identify a suitable model for sentiment and intent classification. Initially, specific classification models for these tasks were explored, but none were found to be a good fit. This led to testing zero-shot models as an alternative approach.

#### **Experiments & Findings**  

#### **1. Searching for Task-Specific Classification Models**  
- Looked for pre-trained models specifically designed for sentiment and intent classification.  
- Could not find a suitable model that aligned well with the task requirements.  

#### **2. Exploring Zero-Shot Models**  
- Tested **`facebook/bart-large-mnli`**, a zero-shot classification model.  
- Performed well on the given examples and sample conversation.  
- Chose this model for sentiment and intent classification as it effectively handled the task.  









