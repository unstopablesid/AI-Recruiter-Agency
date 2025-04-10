from typing import Dict, Any, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class ResumeMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def calculate_match_score(self, resume_data: Dict[str, Any], job_requirements: Dict[str, Any]) -> Dict[str, float]:
        """Calculate match scores between resume and job requirements"""
        # Ensure all required keys exist
        job_requirements = self._ensure_required_keys(job_requirements)
        
        scores = {
            "skills_match": self._calculate_skills_match(
                resume_data.get("skills", []),
                job_requirements.get("required_skills", [])
            ),
            "experience_match": self._calculate_experience_match(
                resume_data.get("experience", []),
                job_requirements.get("required_experience", {})
            ),
            "education_match": self._calculate_education_match(
                resume_data.get("education", []),
                job_requirements.get("required_education", {})
            )
        }
        
        # Calculate overall score with weights
        weights = {
            "skills_match": 0.5,
            "experience_match": 0.3,
            "education_match": 0.2
        }
        
        scores["overall_match"] = sum(
            scores[category] * weights[category]
            for category in scores
        )
        
        return scores
        
    def _ensure_required_keys(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all required keys exist in the dictionary"""
        required_keys = {
            "required_skills": [],
            "required_experience": {"description": ""},
            "required_education": {"description": ""},
            "nice_to_have_skills": []
        }
        
        for key, default_value in required_keys.items():
            if key not in data:
                data[key] = default_value
                
        return data
        
    def _calculate_skills_match(self, resume_skills: List[str], required_skills: List[str]) -> float:
        """Calculate skills match percentage"""
        if not required_skills:
            return 1.0
            
        resume_skills_text = " ".join(resume_skills)
        required_skills_text = " ".join(required_skills)
        
        # Vectorize skills
        vectors = self.vectorizer.fit_transform([resume_skills_text, required_skills_text])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
        return float(similarity)
        
    def _calculate_experience_match(self, resume_experience: List[str], 
                                  required_experience: Dict[str, str]) -> float:
        """Calculate experience match percentage"""
        if not required_experience.get("description"):
            return 1.0
            
        # Extract experience text
        experience_text = " ".join(resume_experience)
        required_experience_text = required_experience.get("description", "")
        
        # Vectorize experience
        vectors = self.vectorizer.fit_transform([experience_text, required_experience_text])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
        return float(similarity)
        
    def _calculate_education_match(self, resume_education: List[str], 
                                 required_education: Dict[str, str]) -> float:
        """Calculate education match percentage"""
        if not required_education.get("description"):
            return 1.0
            
        # Extract education text
        education_text = " ".join(resume_education)
        required_education_text = required_education.get("description", "")
        
        # Vectorize education
        vectors = self.vectorizer.fit_transform([education_text, required_education_text])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
        return float(similarity) 