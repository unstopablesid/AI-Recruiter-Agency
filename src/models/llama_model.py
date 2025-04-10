from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, List, Any
import os
from dotenv import load_dotenv

load_dotenv()

class LlamaModel:
    def __init__(self):
        self.model_name = "meta-llama/Llama-2-7b-chat-hf"  # Using Llama 2 as Llama 3.2 requires special access
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
        
    def load_model(self):
        """Load the Llama model and tokenizer"""
        try:
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Configure tokenizer
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Configure model
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to a simpler model for testing
            self.model_name = "gpt2"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
    def analyze_resume(self, resume_text: str) -> Dict[str, Any]:
        """Analyze resume text and extract key information"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        prompt = f"""
        You are an expert resume analyzer. Analyze the following resume and extract the following information in a structured format:

        1. Skills: List all technical and soft skills mentioned
        2. Work Experience: List each job with duration, role, and key responsibilities
        3. Education: List all educational qualifications with details
        4. Certifications: List all professional certifications
        5. Projects: List all significant projects with descriptions

        Format the output as a JSON-like structure with clear sections.

        Resume:
        {resume_text}
        """
        
        try:
            # Tokenize with proper padding
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate with proper parameters
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            analysis = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._parse_analysis(analysis)
            
        except Exception as e:
            print(f"Error in resume analysis: {e}")
            return self._get_default_resume_data()
    
    def analyze_job_requirements(self, job_description: str) -> Dict[str, Any]:
        """Analyze job description and extract requirements"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        prompt = f"""
        You are an expert job requirements analyzer. Analyze the following job description and extract:

        1. Required Skills: List all mandatory technical and soft skills
        2. Required Experience: Extract years of experience and specific experience requirements
        3. Required Education: List all educational requirements
        4. Nice-to-have Skills: List any additional preferred skills

        Format the output as a JSON-like structure with clear sections.

        Job Description:
        {job_description}
        """
        
        try:
            # Tokenize with proper padding
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate with proper parameters
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            analysis = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._parse_analysis(analysis)
            
        except Exception as e:
            print(f"Error in job requirements analysis: {e}")
            return self._get_default_job_requirements()
    
    def _get_default_resume_data(self) -> Dict[str, Any]:
        """Return default resume data structure"""
        return {
            "skills": [],
            "experience": [],
            "education": [],
            "certifications": [],
            "projects": []
        }
    
    def _get_default_job_requirements(self) -> Dict[str, Any]:
        """Return default job requirements structure"""
        return {
            "required_skills": [],
            "required_experience": {"description": ""},
            "required_education": {"description": ""},
            "nice_to_have_skills": []
        }
    
    def _parse_analysis(self, text: str) -> Dict[str, Any]:
        """Parse the model's output into structured data"""
        # For job requirements analysis
        if "Required Skills" in text:
            return {
                "required_skills": self._extract_list(text, "Required Skills"),
                "required_experience": self._extract_dict(text, "Required Experience"),
                "required_education": self._extract_dict(text, "Required Education"),
                "nice_to_have_skills": self._extract_list(text, "Nice-to-have Skills")
            }
        # For resume analysis
        else:
            return {
                "skills": self._extract_list(text, "Skills"),
                "experience": self._extract_list(text, "Work Experience"),
                "education": self._extract_list(text, "Education"),
                "certifications": self._extract_list(text, "Certifications"),
                "projects": self._extract_list(text, "Projects")
            }
    
    def _extract_list(self, text: str, section: str) -> List[str]:
        """Extract a list of items from a section"""
        try:
            start = text.find(section)
            if start == -1:
                return []
            start = text.find("\n", start) + 1
            end = text.find("\n\n", start)
            if end == -1:
                end = len(text)
            section_text = text[start:end].strip()
            return [item.strip() for item in section_text.split("\n") if item.strip()]
        except Exception:
            return []
    
    def _extract_dict(self, text: str, section: str) -> Dict[str, str]:
        """Extract a dictionary from a section"""
        try:
            start = text.find(section)
            if start == -1:
                return {"description": ""}
            start = text.find("\n", start) + 1
            end = text.find("\n\n", start)
            if end == -1:
                end = len(text)
            section_text = text[start:end].strip()
            return {"description": section_text}
        except Exception:
            return {"description": ""} 