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
from collections import defaultdict


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
    session = await db.session.find_one({"_id": id})
    validation_question = await db.validation.find().to_list(length=1000)
    validation_question = [
        f"Q{index}: {x['questions']}" for index, x in enumerate(validation_question)
    ]
    models = ["gpt-4-turbo"]
    for question in session["battles"]:
        print(question)
        context1_Agent1 = "\n".join(
            x["result"]
            for x in question["model1_aff"]
            if x["type"].startswith("affirmative")
        )
        context1_Agent2 = "\n".join(
            x["result"]
            for x in question["model1_aff"]
            if x["type"].startswith("negative")
        )
        context2_Agent1 = "\n".join(
            x["result"]
            for x in question["model2_aff"]
            if x["type"].startswith("affirmative")
        )
        context2_Agent2 = "\n".join(
            x["result"]
            for x in question["model2_aff"]
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
                        "$push": {
                            "scores": {
                                "$each": [
                                    {
                                        "model": model,
                                        "results": scores,
                                        "question": question["question"],
                                        "category": question["category"],
                                    }
                                ]
                            }
                        },
                    },
                )
        await db.session.update_one(
            {"_id": id},
            {
                "$set": {
                    "status": "completed",
                },
            },
        )


async def battle(id, question, model1, model2, category=None):
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

    new_battle = {
        "question": question,
        "model1_aff": model1_aff,
        "model2_aff": context,
        "category": category,
    }

    await db.session.update_one(
        {"_id": id},
        {"$push": {"battles": new_battle}},
    )

    # # Save as JSON
    # with open("output1.json", "w") as outfile:
    #     json.dump({"model1_aff": model1_aff, "model2_aff": context}, outfile)


# @app.get("/leaderboard")
# async def get_leaderboard(id):
#     session = await db.session.find_one({"_id": ObjectId(id)})
#     if session["type"] == "custom":
#         return HTTPException(status_code=400, detail="Invalid Request")
#     else:
#         if session["status"] == "pending":
#             return {"status": "pending"}
#         else:
#             print(session["scores"])


@app.get("/")
async def read_root():
    return {"message": "Hello World"}


@app.get("/fight")
async def get_status(id):
    return id


@app.get("/results")
async def get_results(session_id: str, category: str = None, question: str = None):
    results = await db.session.find_one({"_id": ObjectId(session_id)})
    if results["type"] == "custom":
        response = {
            "type": results["type"],
            "results": results["scores"],
        }
        return response
    else:
        if category != None and question != None:
            for item in results["scores"]:
                if item["category"] == category and item["question"] == question:
                    response = {
                        "type": results["type"],
                        "results": item,
                    }
                    return response
        categories = defaultdict(
            lambda: {"questions": [], "agent1_scores": [], "agent2_scores": []}
        )

        # Organize data by category and collect scores
        for item in results["scores"]:
            category = item["category"]
            categories[category]["questions"].append(item["question"])
            categories[category]["agent1_scores"].append(item["results"]["agent1"])
            categories[category]["agent2_scores"].append(item["results"]["agent2"])

        # Prepare the output
        output = []
        for category, details in categories.items():
            mean_score_agent1 = sum(details["agent1_scores"]) / len(
                details["agent1_scores"]
            )
            mean_score_agent2 = sum(details["agent2_scores"]) / len(
                details["agent2_scores"]
            )
            output.append(
                {
                    "category": category,
                    "questions": details["questions"],
                    "score of model1": mean_score_agent1,
                    "score of model2": mean_score_agent2,
                }
            )
        response = {
            "type": results["type"],
            "results": output,
        }
        return response


@app.get("/analysis")
async def get_results(session_id: str, category: str = None, question: str = None):
    results = await db.session.find_one({"_id": ObjectId(session_id)})
    if results["type"] == "custom":
        for item in results["battles"][0]["model1_aff"]:
            item.pop("messages", None)
        for item in results["battles"][0]["model2_aff"]:
            item.pop("messages", None)
        response = {
            "type": results["type"],
            "results": results["battles"][0],
        }
        return response
    else:
        if category != None and question != None:
            for item in results["battles"]:
                if item["category"] == category and item["question"] == question:
                    for items in item["model1_aff"]:
                        items.pop("messages", None)
                    for items in item["model2_aff"]:
                        items.pop("messages", None)
                    response = {
                        "type": results["type"],
                        "results": item,
                    }
                    return response


async def fetch_questions():
    # Perform an aggregation to fetch 10 questions per category
    pipeline = [
        {
            "$group": {
                "_id": "$category",
                "questions": {"$push": {"question": "$questions", "_id": "$_id"}},
            }
        },
        {"$project": {"questions": {"$slice": ["$questions", 10]}}},
    ]

    cursor = db.questions.aggregate(pipeline)
    results = await cursor.to_list(None)  # Fetch all results from the cursor
    return results


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
        session = await db.session.insert_one(
            {
                "model1": input.Model1,
                "model2": input.Model2,
                "status": "pending",
                "type": "default",
            }
        )
        # Run full list of questions
        questions = await fetch_questions()
        for category in questions:
            for question in category["questions"]:
                await battle(
                    session.inserted_id,
                    question["question"],
                    input.Model1,
                    input.Model2,
                    category["_id"],
                )
        await evalute_battle(ObjectId("6640967518cc26a183adceb0"))
        return {"status": "success", "session_id": str(session.inserted_id)}
        # break
        # await evalute_battle(session.inserted_id)
    # return grok(input.Model1, input.Model2)
