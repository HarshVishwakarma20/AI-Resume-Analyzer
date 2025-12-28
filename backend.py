import pdfplumber as pdfplumber
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer,util
from skills_db import SKILLS_DB

model = SentenceTransformer('all-MiniLM-L6-v2') # Global Model Initialization

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

def extract_text_from_pdf(file_path):
    # Open the PDF file
    with pdfplumber.open(file_path) as pdf:
        text = ""
        # Loop through every page in the PDF (in case it's 2+ pages)
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def contact_info(text):
    contacts = {
        "Phone_no" : None,
        "email_id" : None,
        "Links" : []
    }
    # phone_pattern = r'\+?d{1,3}[-.\s]?\d{10}' rigid and strict structure
    phone_pattern = r'\+?\d[\d\s-]{8,}\d'   #flexible
    phone_match = re.search(phone_pattern,text)
    if phone_match:
        raw = phone_match.group(0)
        cleaned_PhoneNo = re.sub(r'[\D]','',raw)
        contacts["Phone_no"] = cleaned_PhoneNo

    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    email_match = re.search(email_pattern,text)
    if email_match:
         contacts["email_id"] = email_match.group(0)

    url_pattern = r'https?://\S+|www\.\S+'
    links = re.findall(url_pattern,text)
    if links:
        contacts["Links"] = links
    return contacts

def text_processing(resume_text):
    resume_text = re.sub(r'\(Tip:.*?\)', '', resume_text, flags=re.DOTALL)
    resume_text = re.sub(r'Page \d+', '', resume_text) #removes page number
    
    resume_text = resume_text.lower()
    resume_text = re.sub(r'[^a-z\s]','',resume_text)
    words = resume_text.split() #Tokenization

    stop_word = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_word] #removing stop words

    lemmatizer = WordNetLemmatizer()
    words =[lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

def match_score(resume_text,job_description_text):
    clean_resume = text_processing(resume_text)
    clean_JD = text_processing(job_description_text)

    document = [clean_resume,clean_JD]

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1) )
    tfidf_matrix = tfidf_vectorizer.fit_transform(document)

    #cosine similarity
    similarity_score = cosine_similarity(tfidf_matrix[0:1],tfidf_matrix[1:2])[0][0]

    return round(similarity_score*100,2)

def semantic_matching_Score(resume_text,job_description_text):
    embedding1 = model.encode(resume_text,convert_to_tensor=True)
    embedding2 = model.encode(job_description_text,convert_to_tensor=True)
    cosine_scores = util.cos_sim(embedding1,embedding2)
    return round(cosine_scores.item()*100,2)

def extract_skills(text):
    text = text.lower()
    skills_found = set()
    processed_text = text

    for skill_name,variation in SKILLS_DB.items():
        for synonym in variation:
            pattern = r'\b' + re.escape(synonym) + r'\b'
            if re.search(pattern,processed_text):
                skills_found.add(skill_name)
                break
    return skills_found

def keyword_matched(resume_text,job_description_text):
    resume_skills = extract_skills(resume_text)
    job_description_skills = extract_skills(job_description_text)

    matched_skills = resume_skills.intersection(job_description_skills)
    missing_skills = job_description_skills - resume_skills

    if len(job_description_skills)>0:
        skill_match_percentage = (len(matched_skills)/len(job_description_skills))*100
    else:
        skill_match_percentage = 100
    return round(skill_match_percentage,2),missing_skills,matched_skills

def generate_analysis_report(sbert_score,tfidf_score,keyword_score,matched_skills,missing_skills):
    # used weights as sbert : 60% , tf idf score : 10% , keyword matched score : 30%
    final_score = (sbert_score*0.6) + (tfidf_score*0.10) + (keyword_score*0.30)
    feedback = []
    if keyword_score < 40:
        feedback.append(f" !! Critical Skill Gap: You are missing "
                        f"{len(missing_skills)} key technical skills required for this role."
                        f"Consider explicitly adding these skills if you have experience with them.")
    elif keyword_score >= 75:
        feedback.append(
            "Excellent Skill Coverage !!! : Your resume includes nearly all "
            "the required hard skills mentioned in the job description."
        )
    if sbert_score < 50:
        feedback.append(
            "Contextual Misalignment : While you may have relevant experience, "
            "the way it is described does not strongly align with the role's "
            "expectations, responsibilities, or seniority level."
        )
    elif 50 <= sbert_score <= 60:
        feedback.append(
            "Moderate Context Match: Your experience is somewhat relevant, "
            "but the role expectations or seniority may differ."
        )
    else:
        feedback.append(
            "Good Contextual Match: Your experience aligns well with the role's "
            "responsibilities and intent, even if wording differs."
        )
    if tfidf_score < 20:
        feedback.append(
            "Vocabulary Improvement Needed: Your resume uses fewer of the "
            "industry-specific terms found in the job description. "
            "This may affect ATS ranking and recruiter matching despite having relevant experience."
        )
    elif tfidf_score < 35:
        feedback.append(
            "Vocabulary Optimization Opportunity: Consider mirroring job-specific "
            "phrasing to improve automated screening performance."
        )
    if sbert_score >= 60 and tfidf_score < 30:
        feedback.append(
            "Insights: Although keyword overlap is limited, strong semantic similarity "
            "suggests transferable experience and genuine role relevance."
        )
    if keyword_score >= 70 and sbert_score < 55:
        feedback.append(
        "Insight: While many required skills are listed, the overall role alignment "
        "appears weaker. Clarifying responsibilities and impact may improve relevance."
    )
    return round(final_score, 2), feedback

def interpret_score(score):
    if score >= 60:
        return "You have a Strong Match"
    elif score >= 50:
        return "You have a Potential Match"
    else:
        return "You have a Low Match"   
