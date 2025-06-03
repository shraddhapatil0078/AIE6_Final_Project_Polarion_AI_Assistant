import os
import chainlit as cl
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain.callbacks.tracers.evaluation import EvaluatorCallbackHandler
from langsmith.schemas import Run
from uuid import uuid4
from datetime import datetime, timezone
# Import your custom evaluator
from eval import PolarionEvaluator

# Load environment variables
load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_CLUSTER_URL = os.getenv("QDRANT_CLUSTER_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure OpenAI and Qdrant
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant_client = QdrantClient(url=QDRANT_CLUSTER_URL, api_key=QDRANT_API_KEY)

# Initialize vector store
qdrant_vectorstore = Qdrant(
    client=qdrant_client,
    embeddings=embedding_model,
    #collection_name="polarion_admin_guide_chunks_1"
    collection_name="polarion_admin_guide_chunks_jun2"
)

retriever = qdrant_vectorstore.as_retriever(search_kwargs={"k": 10})

# Initialize OpenAI chat model
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.2,
    streaming=True,
    openai_api_key=OPENAI_API_KEY,
)

# Define the RAG prompt template
RAG_PROMPT = """
You are **Polarion AI Assistant**, a specialist in Polarion ALM
configuration, customization, and administration.

**Ground rules**

1. Use **ONLY** the information in the CONTEXT.  
2. If the CONTEXT does **not** contain the answer, say  
   "I'm not sure from the provided documentation."  
3. Never invent URLs, file names, or steps that are not in the CONTEXT.  
4. Cite the **page_number** (from metadata) or any other supplied locator 
   when you reference a fact - e.g. "(p. 23)".

---

**CONTEXT**
{context}

---

**USER QUESTION**
{question}

---

**Respond in this format**

**Answer**  
A direct, concise answer to the question.

**Supporting details**  
• Bullet-point evidence or step-by-step instructions, each followed by a citation.

Example:
• Go to *Administration ▶ Home Page* (p. 42)  
• Switch the *Content type* field from "Classic Wiki" to "Rich Text" (p. 43)

If no answer is possible:
"*I'm not sure from the provided documentation.*"
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

# Create the RAG chain
# rag_chain = (
#     {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
#     | RunnablePassthrough.assign(context=itemgetter("context"))
#     | {"response": rag_prompt | llm, "context": itemgetter("context")}
# )

from langchain.chains import RetrievalQA

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": rag_prompt}
)

@cl.on_chat_start
async def start():
    await cl.Message(
        content="Hello! I'm your Polarion AI Assistant. How can I help you today?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    user_question = message.content

    # Placeholder response
    msg = cl.Message(content="")
    await msg.send()

    polarion_evaluator = PolarionEvaluator()
    # eval_callback = EvaluatorCallbackHandler(
    #     evaluators=[polarion_evaluator]
    # )

    # Run RAG chain
    #result = await rag_chain.ainvoke({"query": user_question}, callbacks=[eval_callback])
    result = await rag_chain.ainvoke({"query": user_question})
    # Extract and prepare context from source_documents
    source_docs = result.get("source_documents", [])
    context_text = "\n\n".join(doc.page_content for doc in source_docs)

    print("📚 Extracted Context for Eval:", context_text[:300])  # For debugging

    # Manually evaluate
    run_data = Run(
        id=str(uuid4()),
        name="polarion_eval_run",
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        run_type="chain",
        trace_id=str(uuid4()),
        inputs={"query": user_question, "context": context_text},
        outputs={"result": result["result"]}
    )
    eval_result = polarion_evaluator.evaluate_run(run_data)

    await msg.stream_token(result["result"])

    #print(f"Eval Result: {eval_result}")
    # Display evaluation score
    if eval_result and eval_result.score is not None:
        score = round(eval_result.score * 100, 1)
        await cl.Message(
            content=f"📊 **Evaluation Score (Faithfulness + Relevance):** {score}%"
        ).send()
    else:
        await cl.Message(content="⚠️ Evaluation failed or returned no score.").send()


# @cl.on_message
# async def main(message: cl.Message):
#     user_question = message.content

#     # Step 1: Create streaming placeholder
#     msg = cl.Message(content="")
#     await msg.send()

#     # Step 2: Stream answer using RAG prompt directly
#     retrieved_docs = await retriever.ainvoke(user_question)
#     context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

#     # Prepare the full prompt
#     prompt_input = rag_prompt.format(context=context_text, question=user_question)

#     # Step 3: Stream token by token
#     full_answer = ""
#     async for chunk in llm.astream(prompt_input):
#         token = chunk.content
#         if token:
#             full_answer += token
#             await msg.stream_token(token)

#     # Step 4: Evaluate after streaming completes
#     polarion_evaluator = PolarionEvaluator()

#     run_data = Run(
#         id=str(uuid4()),
#         name="polarion_eval_run",
#         start_time=datetime.now(timezone.utc),
#         end_time=datetime.now(timezone.utc),
#         run_type="chain",
#         trace_id=str(uuid4()),
#         inputs={"query": user_question, "context": context_text},
#         outputs={"result": full_answer}
#     )

#     eval_result = polarion_evaluator.evaluate_run(run_data)

#     # Step 5: Show evaluation result
#     if eval_result and eval_result.score is not None:
#         score = round(eval_result.score * 100, 1)
#         await cl.Message(
#             content=f"📊 **Evaluation Score (Faithfulness + Relevance):** {score}%"
#         ).send()
#     else:
#         await cl.Message(content="⚠️ Evaluation failed or returned no score.").send()