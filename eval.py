# eval.py

import re
import logging
from typing import Optional, Any
from langchain.evaluation import StringEvaluator
from langsmith.evaluation import RunEvaluator, EvaluationResult
from langsmith.schemas import Run, Example
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolarionEvaluator(RunEvaluator):
    """An LLM-based evaluator to assess faithfulness and relevance of Polarion answers."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
        self.latest_result = None  # Stores most recent result

        self.prompt = PromptTemplate(
            input_variables=["question", "answer", "context"],
            template="""
You are an expert evaluator for technical AI assistants.

Given a QUESTION, its AI-generated ANSWER, and the official CONTEXT documentation, score the response critically on:

- **Faithfulness:** Does the answer **strictly match** what’s in the CONTEXT? Penalize if it adds anything unsupported.
- **Relevance:** Does the answer **directly** address the QUESTION?
- **Clarity:** Is it **clear, concise, and well-structured**?

### SCORING RULES:
- 90-100: Fully accurate, context-supported, precise, clear.
- 70-89: Mostly correct but missing minor details or slightly vague.
- 50-69: Contains partial support, lacks specificity, or mildly hallucinated.
- 30-49: Vague, poorly supported, or partially incorrect.
- 0-29: Incorrect, hallucinated, or off-topic.

---

QUESTION:
{question}

ANSWER:
{answer}

CONTEXT:
{context}

### FORMAT YOUR OUTPUT LIKE THIS:

Step-by-step reasoning...

Score: <number from 0 to 100>
"""
        )

    def evaluate_run(self, run: Run, example: Optional[Example] = None) -> EvaluationResult:
        print("🔴 PRINT from evaluate_run reached") 
        try:
            question = run.inputs.get("query")  # RetrievalQA uses "query"
            answer = run.outputs.get("result", "")  # RetrievalQA uses "result"
            # Safely extract context if it's a list of documents
            context = run.inputs.get("context", "")  # Get context from inputs instead of outputs
            if not context:  # If context is empty, try to get it from source_documents
                context = run.outputs.get("source_documents", "")
                if isinstance(context, list):
                    context = "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in context)

            logger.info("🟡 Inside evaluator: question=%s", question)
            logger.info("🧪 Question: %s", question)
            logger.info("📦 Answer: %s", answer)
            logger.info("📚 Context: %s", context[:500])  # Avoid flooding logs

            prompt_input = self.prompt.format(question=question, answer=answer, context=context)
            result = self.llm.invoke(prompt_input).content

            reasoning, score_text = result.rsplit("Score:", maxsplit=1)
            score = float(re.search(r"\d+", score_text).group()) / 100.0

            self.latest_result = EvaluationResult(
                key="polarion_eval_score_hf",
                score=score,
                comment=reasoning.strip()
            )

            logger.info(f"✅ Eval Score: {self.latest_result.score}")
            logger.info(f"🧠 Eval Reasoning: {self.latest_result.comment}")

            return self.latest_result
        
        except Exception as e:
            logger.error("❌ Evaluation error: %s", str(e))
            return EvaluationResult(key="polarion_eval_score_hf", score=None, comment=str(e))
