import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

app = FastAPI()

# Set up static files and templates directory
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class ArticleRequest(BaseModel):
    article: str

def generate_summary(text: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
        ],
        max_tokens=150
    )
    summary = response.choices[0].message['content'].strip()
    return summary

def generate_questions(text: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates questions."},
            {"role": "user", "content": f"Create a question from the following text:\n\n{text}"}
        ],
        max_tokens=50
    )
    question = response.choices[0].message['content'].strip()
    return question

def generate_answer_choices(question: str, context: str) -> list:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates multiple-choice answers."},
            {
                "role": "user",
                "content": f"Create four multiple-choice answers for the question: '{question}'. "
                           f"Include one correct answer and three plausible but incorrect options based on the following context:\n\n{context}. "
                           f"Label the options with lowercase letters a, b, c, d."
            }
        ],
        max_tokens=200
    )
    choices_text = response.choices[0].message['content'].strip()
    choices_list = choices_text.split('\n')
    formatted_choices = [choice.strip() for choice in choices_list if choice.strip()]
    return formatted_choices[:4]

def extract_correct_answer(question: str, context: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides concise answers."},
            {"role": "user", "content": f"Provide a brief answer (10 words or fewer) for the following question based on this context:\n\nContext: {context}\n\nQuestion: {question}"}
        ],
        max_tokens=20
    )
    correct_answer = response.choices[0].message['content'].strip()

    # Truncate the answer if it exceeds 50 characters as a fallback
    if len(correct_answer) > 50:
        correct_answer = correct_answer[:50] + "..."

    return correct_answer

@app.get("/", response_class=HTMLResponse)
async def read_quiz(request: Request):
    return templates.TemplateResponse("quiz.html", {"request": request, "title": "Quiz"})

@app.get("/flashcards", response_class=HTMLResponse)
async def read_flashcards(request: Request):
    return templates.TemplateResponse("flashcards.html", {"request": request, "title": "Flashcards"})

@app.post("/generate-questions", response_class=JSONResponse)
async def generate_quiz_questions(article_request: ArticleRequest):
    article = article_request.article

    if not article:
        raise HTTPException(status_code=400, detail="Article text is required.")

    summary = generate_summary(article)
    question = generate_questions(summary)
    correct_answer = extract_correct_answer(question, article).strip().lower()
    choices = generate_answer_choices(question, article)

    correct_choice_label = None
    max_similarity = 0
    for choice in choices:
        choice_text = choice[2:].strip().lower()
        similarity = sum(word in choice_text for word in correct_answer.split())
        if similarity > max_similarity:
            max_similarity = similarity
            correct_choice_label = choice[0]

    if not correct_choice_label:
        correct_choice_label = "a"

    return {
        "question": question,
        "choices": choices,
        "correct_answer": correct_choice_label
    }

@app.post("/generate-flashcards", response_class=JSONResponse)
async def generate_flashcards(article_request: ArticleRequest):
    article = article_request.article

    if not article:
        raise HTTPException(status_code=400, detail="Article text is required.")

    summary = generate_summary(article)
    question = generate_questions(summary)
    answer = extract_correct_answer(question, article)

    return {
        "question": question,
        "answer": answer
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
