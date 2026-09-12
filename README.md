# 🤖 AI-Powered Job Market Intelligence

An end-to-end NLP and Machine Learning application that analyzes job descriptions, predicts job categories, matches resumes with suitable job opportunities, identifies missing skills, and provides personalized skill and learning recommendations.

The system combines **DistilBERT, Sentence-BERT (SBERT), Natural Language Processing, Semantic Similarity, and Streamlit** to build an intelligent job recommendation and skill-gap analysis platform.

---

## 📌 Project Overview

The job market is continuously changing, with new technologies and skills becoming increasingly important.

Job seekers often face difficulties in:

- Understanding which job roles are suitable for their profile
- Identifying skills required by employers
- Comparing their resume with job requirements
- Finding missing or frequently demanded skills
- Deciding which skills they should learn next

This project addresses these challenges by building an **AI-powered Job Market Intelligence and Skill Gap Analysis System**.

The application takes a user's **resume** and a **job description** as input and provides:

- Job category prediction
- Resume–job matching
- Suitable job recommendations
- Skill gap analysis
- Missing skill identification
- Skill recommendations
- Learning area recommendations

---

# 🎯 Business Problem

Job seekers often apply for jobs without knowing whether their skills match the requirements of the role.

Recruiters and candidates also need to analyze large amounts of job descriptions to understand:

- What skills are currently in demand?
- Which job categories are growing?
- What skills are missing from a candidate's resume?
- Which jobs best match a candidate's profile?

Manually performing this analysis across thousands of job postings is time-consuming and inefficient.

Therefore, an automated AI-based system is required to analyze job postings and candidate resumes and provide actionable recommendations.

---

# 📝 Problem Statement

Develop an intelligent NLP-based system that can analyze large-scale job postings and candidate resumes to:

1. Classify job descriptions into relevant job categories.
2. Match candidate resumes with suitable jobs using semantic similarity.
3. Identify skills present in job postings.
4. Compare candidate skills with job requirements.
5. Identify missing skills.
6. Recommend important skills and learning areas.
7. Rank suitable job opportunities for the candidate.

---

# 🚀 Key Features

## 1. 📊 Job Market Intelligence

Analyzes job postings to understand the demand for different skills and job categories.

### Provides:

- Skill demand analysis
- Skill frequency
- Skill categories
- Job category distribution
- Demand percentage

---

## 2. 🧹 Job Description Preprocessing

Job descriptions are cleaned before being used for NLP processing.

### Preprocessing includes:

- Removing HTML tags
- Removing URLs
- Removing unnecessary spaces
- Handling missing values
- Removing duplicate job postings
- Creating a combined job text
 

Machine Learning & NLP
## 3. 🤗 DistilBERT Job Classification

The project uses DistilBERT from Hugging Face Transformers for job category classification.

Why DistilBERT?

DistilBERT is a smaller and faster version of BERT while maintaining strong language understanding capabilities.

It is useful for this project because job descriptions contain contextual information where the meaning of words depends on their surrounding text.

Example

Input:

We are looking for a Machine Learning Engineer
with experience in Python, TensorFlow, Deep Learning
and Artificial Intelligence.

Output:

AI & Machine Learning
4. 🏷️ Job Categories

The project classifies jobs into categories such as:

AI & Machine Learning
Data & Analytics
Software Development
Cloud & IT
Business & Management
Sales & Marketing
Finance & Accounting
Healthcare
Engineering
Other

Job categories are created using a rule-based labeling approach based primarily on job titles and are then used as labels for DistilBERT classification.

🔤 Text Tokenization

Job descriptions are converted into numerical representations using the DistilBERT tokenizer.

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

The text is:

Tokenized
Truncated
Padded
Converted into input IDs
Converted into attention masks

Maximum sequence length:

128 tokens
🧠 Sentence-BERT Resume Matching

The project uses Sentence-BERT (SBERT) for semantic similarity between resumes and jobs.

Model:

all-MiniLM-L6-v2

SBERT converts text into numerical embeddings.

Each resume and job is represented as a vector.

Example:

Resume
   ↓
SBERT
   ↓
384-dimensional embedding

The similarity between resume and job embeddings is calculated using cosine similarity.

🔎 Resume–Job Matching

The matching process follows:

Upload Resume
      ↓
Enter Job Description
      ↓
DistilBERT Job Category Prediction
      ↓
Filter Relevant Job Category
      ↓
Resume → SBERT Embedding
      ↓
Compare Resume with Jobs
      ↓
Cosine Similarity
      ↓
Rank Suitable Jobs

The application displays:

Job title
Company
Location
Job category
Match percentage
Job ranking
📄 Resume Processing

The application allows users to upload a resume in PDF format.

The system extracts text from the PDF using:

PyPDF

The extracted resume text is then cleaned before semantic analysis.

Resume preprocessing includes:
Removing URLs
Removing email addresses
Removing unnecessary characters
Removing extra spaces
Preparing text for SBERT
🧩 Skill Extraction

The project uses the structured skills available in the LinkedIn job posting dataset.

The job skills data is combined with the skills mapping table.

Files used:
job_skills.csv
skills.csv

The skill information is used to analyze:

Skills required by jobs
Skill demand
Skill categories
Candidate skill gaps
📉 Skill Gap Analysis

The system compares candidate skills with the skills associated with suitable jobs.

Workflow:
Resume Skills
      ↓
Candidate Skill Set

Job Requirements
      ↓
Required Skill Set

Candidate Skills
      +
Required Skills
      ↓
Skill Comparison
      ↓
Missing Skills

Example:

Required Skills:
Python
SQL
Excel
Power BI
Machine Learning

Candidate Skills:
Python
SQL
Excel

Missing Skills:
Power BI
Machine Learning
💡 Skill Recommendations

The system ranks missing skills based on their demand among suitable jobs.

Priority levels are assigned:

Demand	Priority
≥ 50%	High
25–49%	Medium
< 25%	Low

This helps candidates focus on skills that are more frequently required.

📚 Learning Recommendations

Missing skills are mapped to broader learning areas.

Example:

Information Technology
        ↓
IT Fundamentals & Technology

Engineering
        ↓
Engineering Fundamentals

Research
        ↓
Research & Analytical Skills

Marketing
        ↓
Marketing & Digital Marketing

The goal is to provide candidates with practical learning directions based on job market requirements.

💼 Suitable Job Ranking

Jobs are ranked according to semantic similarity.

Example:

Rank	Job Title	Match
1	Data Analyst	92.45%
2	Business Analyst	89.31%
3	Data Scientist	86.72%

This allows users to focus on jobs that are most relevant to their profile.

📂 Dataset

The project uses the LinkedIn Job Postings dataset from Kaggle.

Dataset:

LinkedIn Job Postings

Source:

https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

Main dataset files used:
postings.csv
job_skills.csv
skills.csv

Additional dataset files can be used for extended job market analysis.

📊 Dataset Information

The main postings.csv dataset contains information such as:

Job ID
Company
Job title
Job description
Location
Work type
Experience level
Remote availability
Skills description

The dataset contains more than 100,000 job postings, making it suitable for large-scale job market analysis.

🏗️ Project Architecture
                 ┌─────────────────────┐
                 │ LinkedIn Job Data   │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Data Preprocessing  │
                 └──────────┬──────────┘
                            ↓
              ┌─────────────┴─────────────┐
              ↓                           ↓
      Skill Intelligence            Job Classification
              ↓                           ↓
      Skill Demand Analysis          DistilBERT
              ↓                           ↓
              └─────────────┬─────────────┘
                            ↓
                    Job Market Model
                            ↓
                 ┌─────────────────────┐
                 │   Resume Upload     │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Resume Text Extract │
                 └──────────┬──────────┘
                            ↓
                       SBERT
                            ↓
                 Resume Embeddings
                            ↓
                 Resume–Job Matching
                            ↓
                    Skill Gap Analysis
                            ↓
                ┌───────────┴───────────┐
                ↓                       ↓
        Skill Recommendations     Learning Areas
                ↓                       ↓
                └───────────┬───────────┘
                            ↓
                   Streamlit Dashboard
🔄 Complete Project Workflow
1. Understand Business Problem
        ↓
2. Load Dataset
        ↓
3. Data Understanding
        ↓
4. Select Required Columns
        ↓
5. Handle Missing Values
        ↓
6. Remove Duplicates
        ↓
7. Clean Job Descriptions
        ↓
8. Create Job Text
        ↓
9. Explore Text Data
        ↓
10. Extract Technical/Structured Skills
        ↓
11. Normalize Skills
        ↓
12. Categorize Skills
        ↓
13. Analyze Skill Demand
        ↓
14. Create Job Categories
        ↓
15. Validate Categories
        ↓
16. Train/Validation/Test Split
        ↓
17. DistilBERT Tokenization
        ↓
18. Train DistilBERT
        ↓
19. Evaluate Classification
        ↓
20. Save Model
        ↓
21. Predict New Job Category
        ↓
22. Upload Resume
        ↓
23. Resume Text Cleaning
        ↓
24. Generate SBERT Embeddings
        ↓
25. Resume–Job Similarity
        ↓
26. Extract Resume Skills
        ↓
27. Skill Gap Analysis
        ↓
28. Missing Skill Identification
        ↓
29. Skill Recommendations
        ↓
30. Learning Recommendations
        ↓
31. Rank Suitable Jobs
        ↓
32. Streamlit Deployment
🛠️ Technologies Used
Programming Language
Python
Data Processing
Pandas
NumPy
Machine Learning
Scikit-learn
Natural Language Processing
Hugging Face Transformers
DistilBERT
Sentence-BERT
Deep Learning
TensorFlow
Keras
Resume Processing
PyPDF
Similarity
Cosine Similarity
Deployment
Streamlit
Model Hosting
Hugging Face
📦 Project Structure
AI-Powered-Job-Market-Intelligence/
│
├── app.py
├── requirements.txt
├── README.md
│
├── job_distilbert_model/
│   ├── config.json
│   ├── tf_model.h5
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── vocab.txt
│   └── label_encoder.pkl
│
├── job_embeddings.npy
│
└── job_embedding_metadata.csv
🤗 Hugging Face Model

The trained DistilBERT model and supporting files are hosted on Hugging Face.

Model Repository

https://huggingface.co/Srichetan/job-market-distilbert

The repository contains:

config.json
tf_model.h5
tokenizer.json
tokenizer_config.json
special_tokens_map.json
vocab.txt
label_encoder.pkl
job_embeddings.npy
job_embedding_metadata.csv
⚙️ Installation

Clone the repository:

git clone <YOUR_GITHUB_REPOSITORY_URL>

Navigate to the project:

cd AI-Powered-Job-Market-Intelligence

Create a virtual environment:

python -m venv myvenv

Activate the environment on Windows:

myvenv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
📋 Requirements
streamlit
tensorflow
tf-keras==2.20.0
transformers==4.46.3
huggingface-hub
sentence-transformers
scikit-learn
pandas
numpy
pypdf
▶️ Run the Application

Run:

streamlit run app.py

The application will open in your browser.

🖥️ Application Modules
🏠 Home

Provides an overview of the system and its capabilities.

📄 Resume Upload

Users can upload their resume in PDF format.

The system extracts and processes the resume text.

🧠 Job Prediction

Users can enter a job description.

The DistilBERT model predicts the corresponding job category.

Example:

Input:
Machine Learning Engineer with Python,
TensorFlow and Deep Learning experience.

Output:
AI & Machine Learning
🔎 Job Matching

Users provide:

Resume
Job Description

The system predicts the job category and identifies suitable jobs from the dataset.

The results are ranked using semantic similarity.

📉 Skill Gap Analysis

The system compares the candidate's skills with the requirements of suitable jobs.

It identifies missing skills and their demand.

💡 Skill Recommendations

The application recommends important missing skills based on job demand.

📚 Learning Recommendations

The application maps recommended skills to broader learning areas.

📈 Example Use Case
Candidate

A candidate uploads a resume containing:

Python
Pandas
NumPy
Machine Learning
SQL
Job Description
We are looking for a Data Analyst with experience in
SQL, Python, Excel, Power BI and data visualization.
System Output
Predicted Category:
Data & Analytics

Suitable Jobs:
Data Analyst
Business Analyst
Data Analytics Specialist

Missing Skills:
Power BI
Excel
Data Visualization

Recommended Learning Areas:
Business Intelligence
Data Visualization
🔐 Model & Data Considerations

The job category labels used for model training are generated using rule-based logic from job titles.

Therefore, the classification model learns from heuristically generated labels rather than manually annotated ground truth labels.

This is an important limitation and should be considered when interpreting classification performance.

⚠️ Limitations
Job categories are based on heuristic rules.
Structured dataset skills are broad LinkedIn skill categories rather than detailed technology-level skills.
Resume skill extraction uses available dataset skill names.
Matching quality depends on the quality of resume text and job descriptions.
Precomputed job embeddings limit matching to the available indexed jobs.
The system is designed as a decision-support tool and does not guarantee job suitability or employment.
🔮 Future Scope

The project can be extended with:

1. Advanced Skill Extraction

Use NER or transformer-based models to extract individual technical skills such as:

Python
SQL
TensorFlow
AWS
Docker
Power BI
2. Multi-Label Skill Prediction

Instead of predicting only one job category, the model could predict multiple relevant categories and skills.

3. Better Job Matching

Use direct semantic similarity between:

Resume ↔ Job Description

instead of relying mainly on category filtering.

4. Real-Time Job Data

Integrate job APIs or job platforms to retrieve current job postings.

5. Personalized Learning

Recommend courses and certifications based on identified skill gaps.

6. Advanced Recommendation System

Use hybrid recommendation techniques combining:

Resume similarity
Job description similarity
Skill overlap
Experience level
Location
Work type
Salary
7. LLM Integration

An LLM can be added to generate:

Resume improvement suggestions
Personalized career advice
Skill-gap explanations
Job application recommendations
👨‍💻 Author

Srichetan Badugu

B.Tech – Computer Science & Engineering (AI & ML)

Skills
Python
SQL
Machine Learning
Deep Learning
NLP
TensorFlow
Hugging Face Transformers
Data Analysis
Power BI
Streamlit
GitHub

https://github.com/BADUGU-SRICHETAN

Hugging Face

https://huggingface.co/Srichetan

⭐ Conclusion

The AI-Powered Job Market Intelligence & Skill Gap Analysis System provides an end-to-end solution for understanding job market requirements and helping candidates identify suitable career opportunities.

By combining:

NLP + DistilBERT + Sentence-BERT + Semantic Similarity + Skill Gap Analysis + Recommendation + Streamlit

the system transforms large-scale job posting data into actionable career insights.


### One important change before you put this on GitHub

https://github.com/BADUGU-SRICHETAN/AI-Powered-Job-Market-Intelligence-Using-NLP/

with your actual GitHub repository URL.

Also, if your Streamlit app is deployed, add this section near the top:

# 🚀 Live Demo

👉 **Streamlit App:** https://ai-powered-job-market-intelligence-using-nlp.streamlit.app/
