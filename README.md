# AI Recruiter Agency

An intelligent system that analyzes resumes and matches them with job requirements using Llama 3.2 model.

## Project Structure

```
AI-Recruiter-Agency/
├── src/                    # Source code
│   ├── models/            # Llama model integration
│   ├── utils/             # Utility functions
│   ├── parsers/           # Resume and job description parsers
│   ├── matching/          # Matching algorithms
│   └── ui/                # User interface components
├── data/                  # Data storage
│   ├── resumes/          # Sample resumes
│   └── job_descriptions/ # Sample job descriptions
├── tests/                # Test cases
├── config/               # Configuration files
└── docs/                 # Documentation
```

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  
```

2. Install dependencies:s
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with your configuration.

## Usage

1. Run the application:
```bash
streamlit run src/ui/app.py
```

2. Upload resumes and job descriptions through the web interface.

3. View matching results and analysis.

## Features

- Resume parsing and analysis
- Job requirement extraction
- Skill matching algorithm
- Experience matching
- Education matching
- Overall compatibility score
- Detailed match breakdown

## Development

- Use `black` for code formatting
- Run `flake8` for linting
- Write tests using `pytest` #   A I - R e c r u i t e r - A g e n c y 
 
 #   A I - R e c r u i t e r - A g e n c y  
 