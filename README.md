# AI Recruiter Agency

An intelligent system that analyzes resumes and matches them with job requirements using Llama 3.2 model.

## Project Structure

```
AI-Recruiter-Agency/
├── src/                    # Source code
│   ├── models/            # AI model integration
│   │   ├── llama/        # Llama model implementation
│   │   └── embeddings/   # Text embedding models
│   ├── utils/            # Utility functions
│   │   ├── file_utils.py # File handling utilities
│   │   └── text_utils.py # Text processing utilities
│   ├── parsers/          # Document parsers
│   │   ├── resume_parser.py    # Resume parsing logic
│   │   └── job_parser.py       # Job description parsing
│   ├── matching/         # Matching algorithms
│   │   ├── skill_matcher.py    # Skill matching logic
│   │   ├── experience_matcher.py # Experience matching
│   │   └── education_matcher.py  # Education matching
│   └── ui/               # User interface components
│       ├── components/   # Reusable UI components
│       ├── pages/        # Streamlit pages
│       └── app.py        # Main application file
├── data/                 # Data storage
│   ├── resumes/         # Sample resumes
│   └── job_descriptions/ # Sample job descriptions
├── tests/               # Test cases
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── config/             # Configuration files
│   ├── model_config.yaml  # Model configuration
│   └── app_config.yaml    # Application configuration
├── docs/              # Documentation
│   ├── api/          # API documentation
│   └── user_guide/   # User guide
├── .env.example      # Example environment variables
├── .gitignore        # Git ignore file
├── requirements.txt  # Python dependencies
├── README.md        # Project documentation
└── LICENSE          # License file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/unstopablesid/AI-Recruiter-Agency.git
cd AI-Recruiter-Agency
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/MacOS
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration
# Add your API keys and other sensitive information
```

## Project Essentials

### Required Files

- `requirements.txt`: Lists all Python dependencies
- `.env`: Contains environment variables and API keys
- `src/ui/app.py`: Main Streamlit application
- `config/`: Configuration files for the application

### Development Tools

- Code formatting: `black`
- Linting: `flake8`
- Testing: `pytest`

### Running the Application

1. Start the Streamlit app:
```bash
streamlit run src/ui/app.py
```

2. Access the application at `http://localhost:8501`

## Features

- Resume parsing and analysis
- Job requirement extraction
- Skill matching algorithm
- Experience matching
- Education matching
- Overall compatibility score
- Detailed match breakdown

## Development Guidelines

1. Code Formatting:
```bash
black .
```

2. Linting:
```bash
flake8 .
```

3. Testing:
```bash
pytest tests/
```

## GitHub Repository

- Repository: [AI-Recruiter-Agency](https://github.com/unstopablesid/AI-Recruiter-Agency)
- Main Branch: `main`
- License: MIT

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.