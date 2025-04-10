import PyPDF2
from docx import Document
from typing import Dict, Any, List
import re

class ResumeParser:
    def __init__(self):
        self.skills_pattern = re.compile(r'(?i)(?:skills|technical skills|expertise):\s*(.*?)(?=\n\n|\Z)')
        self.experience_pattern = re.compile(r'(?i)(?:experience|work history):\s*(.*?)(?=\n\n|\Z)')
        self.education_pattern = re.compile(r'(?i)(?:education|academic background):\s*(.*?)(?=\n\n|\Z)')
        
    def parse_resume(self, file_path: str) -> Dict[str, Any]:
        """Parse resume from file and extract information"""
        if file_path.endswith('.pdf'):
            return self._parse_pdf(file_path)
        elif file_path.endswith('.docx'):
            return self._parse_docx(file_path)
        else:
            raise ValueError("Unsupported file format")
            
    def _parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """Parse PDF resume"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return self._extract_information(text)
        
    def _parse_docx(self, file_path: str) -> Dict[str, Any]:
        """Parse DOCX resume"""
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return self._extract_information(text)
        
    def _extract_information(self, text: str) -> Dict[str, Any]:
        """Extract information from resume text"""
        return {
            "skills": self._extract_skills(text),
            "experience": self._extract_experience(text),
            "education": self._extract_education(text),
            "raw_text": text
        }
        
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        match = self.skills_pattern.search(text)
        if match:
            skills_text = match.group(1)
            return [skill.strip() for skill in skills_text.split(',')]
        return []
        
    def _extract_experience(self, text: str) -> List[Dict[str, str]]:
        """Extract work experience from resume text"""
        match = self.experience_pattern.search(text)
        if match:
            experience_text = match.group(1)
            # Basic implementation - can be enhanced with more sophisticated parsing
            experiences = []
            for exp in experience_text.split('\n'):
                if exp.strip():
                    experiences.append({"text": exp.strip()})
            return experiences
        return []
        
    def _extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education information from resume text"""
        match = self.education_pattern.search(text)
        if match:
            education_text = match.group(1)
            # Basic implementation - can be enhanced with more sophisticated parsing
            education = []
            for edu in education_text.split('\n'):
                if edu.strip():
                    education.append({"text": edu.strip()})
            return education
        return [] 