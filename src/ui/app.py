import streamlit as st
import os
import sys
from pathlib import Path
from typing import Dict, Any
import json

# Add the src directory to the Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import using the correct path
from parsers.resume_parser import ResumeParser
from models.llama_model import LlamaModel
from matching.matcher import ResumeMatcher

class RecruiterApp:
    def __init__(self):
        self.resume_parser = ResumeParser()
        self.llama_model = LlamaModel()
        self.matcher = ResumeMatcher()
        
    def run(self):
        st.title("AI Recruiter Agency")
        st.write("Upload resumes and job descriptions to get matching scores")
        
        # File uploaders
        resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
        job_description = st.text_area("Enter Job Description")
        
        if st.button("Analyze"):
            if resume_file and job_description:
                # Save uploaded file
                resume_path = self._save_uploaded_file(resume_file)
                
                # Parse resume
                resume_data = self.resume_parser.parse_resume(resume_path)
                
                # Analyze job requirements
                job_requirements = self.llama_model.analyze_job_requirements(job_description)
                
                # Calculate match scores
                scores = self.matcher.calculate_match_score(resume_data, job_requirements)
                
                # Display results
                self._display_results(scores, resume_data, job_requirements)
                
                # Clean up
                os.remove(resume_path)
            else:
                st.error("Please upload a resume and enter a job description")
                
    def _save_uploaded_file(self, uploaded_file) -> str:
        """Save uploaded file to temporary location"""
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        return str(file_path)
        
    def _display_results(self, scores: Dict[str, float], 
                        resume_data: Dict[str, Any], 
                        job_requirements: Dict[str, Any]):
        """Display analysis results"""
        st.header("Matching Results")
        
        # Overall match score
        st.subheader(f"Overall Match: {scores['overall_match']*100:.1f}%")
        
        # Detailed scores
        st.subheader("Detailed Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Skills Match", f"{scores['skills_match']*100:.1f}%")
        with col2:
            st.metric("Experience Match", f"{scores['experience_match']*100:.1f}%")
        with col3:
            st.metric("Education Match", f"{scores['education_match']*100:.1f}%")
        
        # Extracted information
        st.subheader("Extracted Information")
        
        with st.expander("Resume Analysis"):
            st.json(resume_data)
            
        with st.expander("Job Requirements Analysis"):
            st.json(job_requirements)

if __name__ == "__main__":
    app = RecruiterApp()
    app.run() 