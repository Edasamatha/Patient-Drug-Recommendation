Patient Condition-Aware Drug Recommendation & Sentiment Analysis
📌 Overview
This project implements an advanced end-to-end healthcare analytics workflow that combines deterministic, condition-aware drug recommendation logic with NLP-driven sentiment analysis. The system is designed to provide personalized medication suggestions by analyzing structured clinical data alongside real-world patient feedback from clinical reviews. 
📂 Dataset
The system processes two primary types of data to ensure a holistic recommendation:
•	Structured Clinical Data: Includes patient parameters such as age, diagnosis, symptoms, and medical history used to determine clinical suitability. 
•	Unstructured Clinical Reviews: Real-world patient feedback sourced from medical forums to evaluate drug effectiveness and side effects. 
•	Tasks:
o	Data cleaning, normalization, and missing value handling using Pandas. 
o	Standardization of textual data through an NLP preprocessing pipeline involving tokenization and lemmatization. 
🛠️ Tools & Technologies
•	Python: Core programming language for data manipulation and system logic. 
•	Machine Learning: Scikit-learn for implementing classification models like Random Forest and Logistic Regression. 
•	Natural Language Processing (NLP): NLTK and spaCy for tokenization, stopword removal, and text processing. 
•	Web Frameworks: Flask for backend API services and Streamlit for the interactive user interface. 
•	API Integration: Hugging Face Inference API for sentiment classification using pre-trained transformer models. 
🔄 Project Steps
•	Step 1: Data Loading & EDA: Used Pandas to load clinical datasets and conducted exploratory analysis to identify patterns and correlations. 
•	Step 2: NLP Preprocessing: Standardized drug reviews by converting text to lowercase, removing punctuation, and applying TF-IDF vectorization. 
•	Step 3: Machine Learning Modeling: Trained supervised classifiers to map patient feature vectors to appropriate medication labels based on historical data. 
•	Step 4: Sentiment Analysis: Processed clinical feedback to categorize patient perception as positive, negative, or neutral. 
•	Step 5: Weighted Ranking Integration: Combined clinical prediction scores with aggregated sentiment scores using a weighted formula to rank final results. 
•	Step 6: Dashboard Development: Built a two-column Streamlit interface to handle structured inputs and dynamic text analysis. 
•	Step 7: Testing & Validation: Performed rigorous unit, integration, and performance testing to ensure system reliability. 
📊 Dashboard
The interactive dashboard includes:
•	Structured Input Section: Fields for medical condition selection, age, pregnancy status, and clinical factors like renal or liver impairment. 
•	Dynamic Intake Module: A text area for describing patient history and a file uploader supporting formats like .txt, .pdf, and .csv. 
•	Sentiment Aggregate: Real-time display of aggregated sentiment distribution for specific drugs based on patient reviews. 
•	Recommendation Display: Ranked lists of top-suggested drugs along with confidence scores and clinical review summaries. 
📈 Results & Insights
•	Context-Aware Ranking: The system successfully adjusts the rank of drugs based on overwhelmingly negative patient feedback despite clinical suitability. 
•	Transparency: Provided healthcare professionals with clear visibility into both clinical prediction probabilities and real-world sentiment trends. 
•	Robustness: Validated through K-fold cross-validation, maintaining consistent accuracy and reliability across medical categories. 
▶️ How to Run the Project
1.	Clone the Repository: git clone <repository-link>
2.	Install Required Libraries: pip install -r requirements.txt
3.	Setup Environment: Copy .env.example to .env and add your HF_API_KEY
4.	Run Application: streamlit run app.py
⭐ Project Value
This project showcases essential end-to-end skills for Data Analyst and Data Science roles, including:
•	Integration of structured clinical data and unstructured textual insights. 
•	Development of modular, scalable backend and frontend architectures using Flask and Streamlit. 
•	Practical application of machine learning and NLP to solve real-world healthcare challenges. 
📬 Contact
•	Author: Samatha Eda 
•	Email: edasamatha2005@gmail.com 
•	LinkedIn: linkedin.com/in/Edasamatha 
