from fastapi import FastAPI, Depends, HTTPException, Security
from decouple import config
from app.model.fightModel import FightModel
from groq import Groq
import json
import time
from app.utils.database import Database
from openai import OpenAI
from bson import ObjectId
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
db = Database().db
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def openai_response(model, prompt):
    client = OpenAI(api_key=config("OPENAI_API_KEY"))

    completion = client.chat.completions.create(model=model, messages=prompt)

    return json.loads(completion.choices[0].message.content)


async def evalute_battle(id):
    client = Groq(api_key=config("GROQ_API_KEY"))
    session = await db.session.find_one({"_id": id})
    validation_question = await db.validation.find().to_list(length=1000)
    validation_question = [
        f"Q{index}: {x['questions']}" for index, x in enumerate(validation_question)
    ]
    models = ["gpt-4-turbo"]
    context1_Agent1 = "\n".join(
        x["result"]
        for x in session["battle"]["model1_aff"]
        if x["type"].startswith("affirmative")
    )
    context1_Agent2 = "\n".join(
        x["result"]
        for x in session["battle"]["model1_aff"]
        if x["type"].startswith("negative")
    )
    context2_Agent1 = "\n".join(
        x["result"]
        for x in session["battle"]["model2_aff"]
        if x["type"].startswith("affirmative")
    )
    context2_Agent2 = "\n".join(
        x["result"]
        for x in session["battle"]["model2_aff"]
        if x["type"].startswith("negative")
    )
    results = []
    for model in models:
        if model.startswith("gpt"):
            messages = [
                {
                    "role": "system",
                    "content": 'Evaluate the context and decide is the context satisfies the question. Format json, sample: [{"question" : "Q1", "result" : true}]. strictly follow the sample json format.',
                },
                {
                    "role": "user",
                    "content": f"Questions\n{'\n'.join(validation_question)}\nContext\n{context1_Agent1}",
                },
            ]
            results.append(openai_response(model, messages))
            messages = [
                {
                    "role": "system",
                    "content": 'Evaluate the context and decide is the context satisfies the question. Format json, sample: [{"question" : "Q1", "result" : true}]. strictly follow the sample json format. and dont include anything else in your response',
                },
                {
                    "role": "user",
                    "content": f"Questions\n{'\n'.join(validation_question)}\nContext\n{context1_Agent2}",
                },
            ]
            results.append(openai_response(model, messages))
            messages = [
                {
                    "role": "system",
                    "content": 'Evaluate the context and decide is the context satisfies the question. Format json, sample: [{"question" : "Q1", "result" : true}]. strictly follow the sample json format. and dont include anything else in your response',
                },
                {
                    "role": "user",
                    "content": f"Questions\n{'\n'.join(validation_question)}\nContext\n{context2_Agent1}",
                },
            ]
            results.append(openai_response(model, messages))
            messages = [
                {
                    "role": "system",
                    "content": 'Evaluate the context and decide is the context satisfies the question. Format json, sample: [{"question" : "Q1", "result" : true}]. strictly follow the sample json format. and dont include anything else in your response',
                },
                {
                    "role": "user",
                    "content": f"Questions\n{'\n'.join(validation_question)}\nContext\n{context2_Agent2}",
                },
            ]
            results.append(openai_response(model, messages))
            scores = {"agent1": 0, "agent2": 0}
            scores["agent1"] = sum([1 for i in results[0] if i["result"] == True])
            scores["agent2"] = sum([1 for i in results[1] if i["result"] == True])
            scores["agent1"] += sum([1 for i in results[2] if i["result"] == True])
            scores["agent2"] += sum([1 for i in results[3] if i["result"] == True])
            await db.session.update_one(
                {"_id": id},
                {
                    "$set": {"status": "completed"},
                    "$push": {
                        "scores": {"$each": [{"model": model, "results": scores}]}
                    },
                },
            )


async def battle(id, question, model1, model2):
    client = Groq(api_key=config("GROQ_API_KEY"))
    context = []

    def generate_completion(role, stage, content_function, model):
        content = content_function()
        messages = [
            {
                "role": "system",
                "content": f"Debate format: Karl Popper you're at a debating event this is your following role and stage of debate.\nType: {role}\nStage: {stage}",
            },
            {"role": "user", "content": content},
        ]
        completion = client.chat.completions.create(messages=messages, model=model)
        return {
            "type": f"{role}_{stage.lower()}",
            "result": completion.choices[0].message.content,
            "messages": messages,
            "model": model,
        }

    # Function to access context results dynamically
    def get_result(index):
        return context[index]["result"] if index < len(context) else ""

    # Generate each part of the debate
    parts = [
        ("affirmative", "Opening Speech", lambda: f"Topic: {question}", model1),
        (
            "negative",
            "Cross Examination",
            lambda: f"Opening speech of affirmative_opening: {get_result(0)}",
            model2,
        ),
        (
            "affirmative",
            "Answering questions",
            lambda: f"question: {get_result(1)}",
            model1,
        ),
        ("negative", "Opening Speech", lambda: f"Topic: {question}", model2),
        (
            "affirmative",
            "Cross Examination",
            lambda: f"Opening speech of negative_opening: {get_result(3)}",
            model1,
        ),
        (
            "negative",
            "Answering questions",
            lambda: f"question: {get_result(4)}",
            model2,
        ),
        (
            "affirmative",
            "Rebuttal, Previous speech of negative",
            lambda: f"{get_result(5)}",
            model1,
        ),
        (
            "negative",
            "Cross Examination",
            lambda: f"speech of affirmative_rebuttal: {get_result(6)}",
            model2,
        ),
        (
            "affirmative",
            "Answering questions",
            lambda: f"question: {get_result(7)}",
            model1,
        ),
        (
            "negative",
            "Rebuttal, Previous speech of affirmative",
            lambda: f"{get_result(2)}",
            model2,
        ),
        (
            "affirmative",
            "Cross Examination",
            lambda: f"speech of negative_rebuttal: {get_result(9)}",
            model1,
        ),
        (
            "negative",
            "Answering questions",
            lambda: f"question: {get_result(10)}",
            model2,
        ),
        (
            "affirmative",
            "Conclusion",
            lambda: " ".join(
                f"{item['type']}: {item['result']}"
                for item in context
                if item["type"].startswith("affirmative")
            ),
            model1,
        ),
        (
            "negative",
            "Conclusion",
            lambda: " ".join(
                f"{item['type']}: {item['result']}"
                for item in context[:-1]
                if item["type"].startswith("negative")
            ),
            model2,
        ),
    ]

    for role, stage, content_function, model in parts:
        context.append(generate_completion(role, stage, content_function, model))
    temp = model1
    model1 = model2
    model2 = temp
    model1_aff = context
    context = []
    for role, stage, content_function, model in parts:
        context.append(generate_completion(role, stage, content_function, model))

    await db.session.update_one(
        {"_id": id},
        {"$set": {"battle": {"model1_aff": model1_aff, "model2_aff": context}}},
    )

    # # Save as JSON
    # with open("output1.json", "w") as outfile:
    #     json.dump({"model1_aff": model1_aff, "model2_aff": context}, outfile)


@app.get("/")
async def read_root():
    return {"message": "Hello World"}


@app.get("/fight")
async def get_status(id):
    return id


@app.get("/results")
async def get_results(session_id: str):
    results = await db.session.find_one({"_id": ObjectId(session_id)})

    if results["type"] == "custom":
        response = {
            "type": results["type"],
            "results": results["scores"],
        }
        return response


@app.get("/analysis")
async def get_results(session_id: str):
    results = await db.session.find_one({"_id": ObjectId(session_id)})
    if results["type"] == "custom":
        for item in results["battle"]["model1_aff"]:
            item.pop("messages", None)
        for item in results["battle"]["model2_aff"]:
            item.pop("messages", None)
        response = {
            "type": results["type"],
            "results": results["battle"],
        }
        return response


async def fetch_questions():
    categories = await db.questions.distinct("category")  # Get all unique categories
    limited_results = []

    # We limit to 10 categories for the example; remove slicing if you want all categories
    for category in categories[:10]:
        # For each category, fetch 10 questions
        cursor = db.questions.find({"category": category}).limit(10)
        questions = await cursor.to_list(length=10)
        limited_results.extend(questions)

    return limited_results


@app.post("/fight")
async def fight(input: FightModel):
    if not input.defaultQuestion:
        # create session
        session = await db.session.insert_one(
            {
                "question": input.question,
                "model1": input.Model1,
                "model2": input.Model2,
                "status": "pending",
                "type": "custom",
            }
        )
        if input.question is None:
            raise HTTPException(status_code=400, detail="Question not provided")
        else:
            await battle(
                session.inserted_id, input.question, input.Model1, input.Model2
            )
            await evalute_battle(session.inserted_id)
            return {"status": "success", "session_id": str(session.inserted_id)}
    else:
        # Run full list of questions
        session = await db.session.insert_one(
            {
                "question": input.question,
                "model1": input.Model1,
                "model2": input.Model2,
                "status": "pending",
                "type": "default",
            }
        )
        questions = await db.questions.find().to_list(length=10000)
        for question in questions:
            # await battle(
            #     session.inserted_id, question["questions"], input.Model1, input.Model2
            # )
            print(question)
    # return grok(input.Model1, input.Model2)
