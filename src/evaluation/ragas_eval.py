import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory, OpenAIEmbeddings
from src.retrieval.retriever import retrieve_and_rerank
from src.api.main import generate_answer
from openai import OpenAI

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Explicitly configure RAGAS to use OpenAI
ragas_llm = llm_factory("gpt-4o-mine", client = openai_client)
ragas_embeddings = OpenAIEmbeddings("text-embedding-3-small", client=openai_client)

eval_data = [
    {
        "question": "How do I set up the development environment for kubeflow pipelines?",
        "ground_truth": "You need to install the Kubeflow Pipelines SDK using pip install kfp and have a Kubernetes cluster running."
    },
    {
        "question": "How do I contribute to kubeflow pipelines?",
        "ground_truth": "Fork the repository, make changes, write tests, and submit a pull request following the contribution guidelines."
    },
    {
        "question": "What is kubeflow pipelines?",
        "ground_truth": "Kubeflow Pipelines is a platform for building and deploying portable and scalable machine learning workflows using Docker containers."
    },
]

collection_name = "kubeflow__pipelines"

questions = []
answers = []
contexts = []
ground_truths = []

for item in eval_data:
    print(f"Evaluating: {item['question']}")

    results = retrieve_and_rerank(item["question"], collection_name)
    
    # Truncate each chunk to 500 chars to avoid token limits
    context_texts = [r["text"][:500] for r in results]

    context_str = "\n\n".join([
        f"[Source: {r['metadata']['source']}]\n{r['text'][:500]}"
        for r in results
    ])

    answer = generate_answer(context_str, item["question"])

    questions.append(item["question"])
    answers.append(answer)
    contexts.append(context_texts)
    ground_truths.append(item["ground_truth"])

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

print("\nRunning RAGAS evaluation...")
results = evaluate(
    dataset,
    metrics=[Faithfulness(llm=ragas_llm), 
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
            ContextPrecision(llm=ragas_llm),
            ContextRecall(llm=ragas_llm)]
)

print("\n=== RAGAS Evaluation Results ===")
print(results)