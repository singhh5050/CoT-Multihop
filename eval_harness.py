"""
Multi-hop Question Answering Evaluation Harness

This module provides a comprehensive evaluation framework for comparing different 
reasoning approaches on multi-hop question answering tasks using the MuSiQue dataset.
"""

import os
import time
import re
import sys
import argparse
import asyncio
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import tiktoken
import evaluate
from datasets import load_dataset
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from autogen_agentchat.agents import BaseChatAgent
from autogen_core.models import UserMessage, AssistantMessage

# ===== CONFIG =====
# LangChain & LLM setup
BASE_LLM = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Token counter for rate limiting
ENC = tiktoken.encoding_for_model("gpt-4")

# ===== UTILITY FUNCTIONS =====
def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison by removing articles, punctuation, and extra whitespace.
    
    Args:
        text: The text to normalize
        
    Returns:
        Normalized text string
    """
    if not text:
        return ""
    # Remove articles and punctuation
    text = re.sub(r'\b(a|an|the)\b', ' ', text.lower())
    text = re.sub(r'[^\w\s]', ' ', text)
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_f1(pred: str, gold: str) -> float:
    """
    Calculate token-level F1 score following the HotpotQA/MuSiQue evaluation.
    
    Args:
        pred: Predicted answer
        gold: Gold reference answer
        
    Returns:
        F1 score between 0 and 1
    """
    norm_pred = normalize_answer(pred)
    norm_gold = normalize_answer(gold)
    
    if not norm_pred or not norm_gold:
        return int(norm_pred == norm_gold)  # Return 1 if both empty, 0 otherwise
    
    # Tokenize by whitespace
    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()

    # Count token occurrences
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    # Edge case: empty predictions/references
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return int(pred_tokens == gold_tokens)  # 1 if both empty, 0 otherwise
    
    # Calculate F1
    precision = num_same / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = num_same / len(gold_tokens) if len(gold_tokens) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def extract_formatted_answer(text: str) -> str:
    """Extract the answer from between [ANSWER] tags."""
    match = re.search(r'\[ANSWER\](.*?)\[/ANSWER\]', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

async def ensure_formatted_answer(text: str, question: str, llm=BASE_LLM) -> str:
    """Re-prompt the model if answer formatting is missing."""
    if "[ANSWER]" not in text or "[/ANSWER]" not in text:
        prompt = f"""You provided this answer to the question "{question}":

{text}

But you forgot to include the required format. Please restate ONLY your final answer in this format:
[ANSWER] your final answer [/ANSWER]"""
        
        response = await llm.agenerate([[SystemMessage(content="You are a helpful assistant."), 
                                         HumanMessage(content=prompt)]])
        new_text = response.generations[0][0].message.content
        return new_text
    return text

def confidence_interval(success: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate the confidence interval for a proportion.
    
    Args:
        success: Number of successful trials
        total: Total number of trials
        confidence: Confidence level (default: 0.95 for 95% confidence)
        
    Returns:
        Tuple containing (lower bound, upper bound)
    """
    if total == 0:
        return 0.0, 0.0
    
    p = success / total
    z = 1.96  # 95% confidence
    ci = z * np.sqrt((p * (1 - p)) / total)
    return p - ci, p + ci

# ===== PROMPTING TEMPLATES =====
# Optimized Direct Answer Prompting
direct_template = PromptTemplate(
    input_variables=["question"],
    template="""
Here are some examples of how to answer questions directly and concisely:

Q: Who wrote the novel '1984'?
A: [ANSWER] George Orwell [/ANSWER]

Q: What is the capital of France?
A: [ANSWER] Paris [/ANSWER]

Q: Is the speed of light faster than the speed of sound?
A: [ANSWER] Yes [/ANSWER]

Answer the following question with just the answer, no explanation.
Format your answer exactly as: [ANSWER] your_answer_here [/ANSWER]

Q: {question}
A: 
"""
)

# Optimized Chain of Thought Prompting
cot_template = PromptTemplate(
    input_variables=["question"],
    template="""
Here's how to solve complex questions step-by-step:

Q: Who directed the movie that won Best Picture in 2020?
A: Let me think through this step-by-step:
1. First, I need to identify which movie won Best Picture in 2020.
2. The movie "Parasite" directed by Bong Joon-ho won Best Picture at the 92nd Academy Awards in 2020.
3. Therefore, Bong Joon-ho directed the movie that won Best Picture in 2020.
[ANSWER] Bong Joon-ho [/ANSWER]

Q: {question}
A: Let me think through this step-by-step:
"""
)

# Create runnable chains
direct_chain = direct_template | BASE_LLM
cot_chain = cot_template | BASE_LLM

# ===== MULTI-AGENT IMPLEMENTATION =====
class CustomAgent(BaseChatAgent):
    """Custom AutoGen agent implementation for multi-agent CoT."""
    
    def __init__(self, name: str, system_message: str = "You are a helpful assistant."):
        super().__init__(name=name, description=f"Agent named {name} with system message: {system_message}")
        self.system_message = system_message

    async def on_messages(self, messages: List[UserMessage]) -> AssistantMessage:
        # Combine messages into one prompt
        prompt = "\n".join([msg.content for msg in messages])
        # Call the LLM via LangChain
        response = await BASE_LLM.agenerate([
            [SystemMessage(content=self.system_message), HumanMessage(content=prompt)]
        ])
        # Extract the assistant's reply
        reply = response.generations[0][0].message.content
        return AssistantMessage(content=reply, source=self.name)
    
    async def on_reset(self) -> None:
        # Nothing to reset for this simple agent
        pass
    
    @property
    def produced_message_types(self) -> List[type]:
        return [AssistantMessage]

# Instantiate custom agents for multi-agent CoT
reasoner = CustomAgent(
    name="Reasoner",
    system_message="""You are a reasoning agent that solves problems step-by-step. 
    After explaining your reasoning, always conclude with your final answer in this format: [ANSWER] your_answer_here [/ANSWER]"""
)
verifier = CustomAgent(
    name="Verifier",
    system_message="""You are a verifier agent that checks reasoning for errors and accuracy.
    If the reasoning is completely valid and leads to a correct conclusion, respond with:
    [VALID] The reasoning is correct.
    
    If there are any errors, mistakes, or issues with the reasoning, respond with:
    [INVALID] Brief explanation of the error."""
)

# ===== EVALUATION METHODS =====
async def agent_cot(question: str, question_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Multi-agent CoT: agent-driven refinement approach.
    
    Args:
        question: The question to answer
        question_id: Optional question identifier
        
    Returns:
        Dictionary containing results and metadata
    """
    start_time = time.time()
    # Reasoner produces initial CoT
    user_msg = UserMessage(content=f"Solve this step-by-step: {question}", source="user")
    reasoning = await reasoner.on_messages([user_msg])
    reasoning_output = reasoning.content
    
    # Check if the answer tag is included, re-prompt if needed
    if "[ANSWER]" not in reasoning_output:
        reasoning_output = await ensure_formatted_answer(reasoning_output, question)

    # Verifier checks reasoning
    verdict_msg = UserMessage(content=reasoning_output, source="user")
    verdict = await verifier.on_messages([verdict_msg])
    verdict_output = verdict.content
    
    is_valid = verdict_output.strip().lower().startswith("[valid]")
    
    # If verifier flags issues, ask for correction
    final_output = reasoning_output
    if not is_valid:
        correction_prompt = (
            f"Please revise your reasoning, addressing the verifier's feedback:\n{verdict_output}\n"
            f"Make sure to include your final answer in [ANSWER] [/ANSWER] tags."
        )
        correction = await reasoner.on_messages([UserMessage(content=correction_prompt, source="user")])
        final_output = correction.content
        
        # Ensure corrected answer has proper formatting
        if "[ANSWER]" not in final_output:
            final_output = await ensure_formatted_answer(final_output, question)
    
    end_time = time.time()
    
    return {
        "raw_output": final_output,
        "extracted_answer": extract_formatted_answer(final_output),
        "is_verified_valid": is_valid,
        "verification_message": verdict_output,
        "processing_time": end_time - start_time,
        "question_id": question_id
    }

# Metric evaluators
em_direct = evaluate.load('exact_match')
em_cot_single = evaluate.load('exact_match')
em_cot_multi = evaluate.load('exact_match')

# ===== MAIN EVALUATION HARNESS =====
def run_evaluation(
    split: str = 'validation', 
    num_examples: int = 5, 
    seed: int = 42, 
    checkpoint_interval: int = 20, 
    resume_from: str = None
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """
    Run the evaluation with checkpointing to save results incrementally.
    
    Args:
        split: Dataset split to use ('validation' recommended)
        num_examples: Number of examples to evaluate
        seed: Random seed for reproducibility
        checkpoint_interval: Save checkpoint after this many examples
        resume_from: Path to previous results file to resume from
        
    Returns:
        Tuple containing (results DataFrame, metrics dict, hop metrics file path)
    """
    # Use the resumed file name if provided, otherwise create new timestamped files
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if resume_from:
        results_file = resume_from
        # Create a metrics file with matching name pattern but preserving the timestamp from results
        base_name = os.path.basename(resume_from)
        if base_name.startswith("musique_results_") and base_name.endswith(".csv"):
            timestamp_part = base_name[len("musique_results_"):-4]  # Extract the timestamp
            metrics_file = f'musique_metrics_{timestamp_part}.csv'
            hop_metrics_file = f'metrics_by_hop_{timestamp_part}.csv'
        else:
            # If naming pattern doesn't match, create a new metrics file with current timestamp
            metrics_file = f'musique_metrics_{timestamp}.csv'
            hop_metrics_file = f'metrics_by_hop_{timestamp}.csv'
    else:
        results_file = f'musique_results_{timestamp}.csv'
        metrics_file = f'musique_metrics_{timestamp}.csv'
        hop_metrics_file = f'metrics_by_hop_{timestamp}.csv'
        
    print(f"Results will be saved to: {results_file}")
    print(f"Metrics will be saved to: {metrics_file}")
    print(f"Hop metrics will be saved to: {hop_metrics_file}")
    
    # Load validation set and sample examples
    val_stream = load_dataset('fladhak/musique', split=split, 
                             streaming=True, trust_remote_code=True)
    dataset_stream = val_stream.shuffle(seed=seed)
    
    # Initialize records and counters
    records = []
    start_idx = 0
    
    # Track hop counts for balanced sampling
    # MuSiQue has examples with 2, 3, and 4 hops
    hop_counts = {2: 0, 3: 0, 4: 0}  # Track how many of each hop count we've sampled
    target_hop_sequence = [2, 3, 4, 3, 2, 3, 4, 2]  # Cycle through hop counts for balanced sampling
    
    # Resume from previous run if specified
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from {resume_from}")
        prev_results = pd.read_csv(resume_from)
        records = prev_results.to_dict('records')
        start_idx = len(records)
        processed_ids = set(prev_results['question_id'])
        
        # Update hop counts from previous results
        for record in records:
            hop = min(4, int(record['hop_count']))  # Cap at 4 hops
            hop_counts[hop] = hop_counts.get(hop, 0) + 1
            
        print(f"Already processed {start_idx} examples")
        print(f"Current hop distribution: 2-hop: {hop_counts.get(2, 0)}, " 
              f"3-hop: {hop_counts.get(3, 0)}, " 
              f"4-hop: {hop_counts.get(4, 0)}")
    else:
        processed_ids = set()
    
    # Limit to remaining examples
    remaining_examples = num_examples - start_idx
    print(f"Will process {remaining_examples} more examples")

    async def process():
        """Internal async function to process examples"""
        direct_correct = 0
        cot_single_correct = 0
        cot_multi_correct = 0
        direct_f1_sum = 0
        cot_single_f1_sum = 0
        cot_multi_f1_sum = 0
        total = len(records)  # Count already processed examples from resumed file
        checkpoint_count = 0
        
        # Iterator to process examples one by one
        dataset_iter = iter(dataset_stream)
        
        for i in range(remaining_examples):
            # Determine which hop count to target next
            total_processed = sum(hop_counts.values())
            target_hop = target_hop_sequence[total_processed % len(target_hop_sequence)]
            
            # Get next example matching the target hop count
            example = None
            search_count = 0
            max_search = 1000  # Limit search to avoid infinite loop
            
            while search_count < max_search:
                try:
                    candidate = next(dataset_iter)
                    example_id = candidate.get('id', f"example_{i+start_idx}")
                    # Skip if already processed
                    if example_id in processed_ids:
                        search_count += 1
                        continue
                        
                    # Check hop count for MuSiQue dataset (length of question_decomposition)
                    candidate_hop = len(candidate['question_decomposition'])
                    
                    # Accept if it matches our target or if we've searched too many
                    if candidate_hop == target_hop or search_count > 500:
                        example = candidate
                        # Update the actual hop we found
                        target_hop = candidate_hop
                        break
                    
                    search_count += 1
                except StopIteration:
                    # If we run out of examples, restart the iterator
                    dataset_iter = iter(val_stream.shuffle(seed=seed+search_count))
                    search_count += 1
            
            if not example:
                print(f"Could not find example with target hop count {target_hop} after {max_search} attempts")
                break
                
            # Update hop count tracking
            hop_counts[target_hop] = hop_counts.get(target_hop, 0) + 1
            
            q = example['question']
            ref = example['answer']
            hop_count = len(example['question_decomposition'])
            total += 1
            processed_ids.add(example_id)
            checkpoint_count += 1
            
            print(f"Processing example {i+1}/{remaining_examples} (overall: {total}/{num_examples}, "
                  f"id: {example_id}, hops: {hop_count}, target: {target_hop})")
            
            start_time = time.time()

            # Direct answer prediction with optimized prompt
            direct_pred_raw = direct_chain.invoke({"question": q}).content.strip()
            # Ensure direct prediction has answer tags
            if "[ANSWER]" not in direct_pred_raw:
                direct_pred_raw = await ensure_formatted_answer(direct_pred_raw, q)
            direct_pred = extract_formatted_answer(direct_pred_raw)
            direct_time = time.time() - start_time
            
            # Single-agent CoT prediction
            start_time = time.time() 
            cot_pred_raw = cot_chain.invoke({"question": q}).content.strip()
            # Ensure CoT prediction has answer tags
            if "[ANSWER]" not in cot_pred_raw:
                cot_pred_raw = await ensure_formatted_answer(cot_pred_raw, q)
            cot_single_pred = extract_formatted_answer(cot_pred_raw)
            cot_single_time = time.time() - start_time
            
            # Multi-agent CoT prediction
            start_time = time.time()
            cot_multi_result = await agent_cot(q, example_id)
            cot_multi_pred = cot_multi_result["extracted_answer"]
            cot_multi_raw = cot_multi_result["raw_output"]
            cot_multi_time = time.time() - start_time

            # Calculate F1 scores
            direct_f1 = compute_f1(direct_pred, ref)
            cot_single_f1 = compute_f1(cot_single_pred, ref)
            cot_multi_f1 = compute_f1(cot_multi_pred, ref)
            
            # Track cumulative metrics
            direct_f1_sum += direct_f1
            cot_single_f1_sum += cot_single_f1
            cot_multi_f1_sum += cot_multi_f1
            
            # Count exact matches for confidence intervals
            if normalize_answer(direct_pred) == normalize_answer(ref):
                direct_correct += 1
            if normalize_answer(cot_single_pred) == normalize_answer(ref):
                cot_single_correct += 1
            if normalize_answer(cot_multi_pred) == normalize_answer(ref):
                cot_multi_correct += 1

            # Add to records
            records.append({
                'question_id': example_id,
                'question': q,
                'reference': ref,
                'direct_answer': direct_pred,
                'cot_single_answer': cot_single_pred,
                'cot_multi_answer': cot_multi_pred,
                'direct_raw': direct_pred_raw,
                'cot_single_raw': cot_pred_raw,
                'cot_multi_raw': cot_multi_raw,
                'hop_count': hop_count,
                'direct_time': direct_time,
                'cot_single_time': cot_single_time,
                'cot_multi_time': cot_multi_time,
                'direct_f1': direct_f1,
                'cot_single_f1': cot_single_f1,
                'cot_multi_f1': cot_multi_f1,
                'direct_correct': normalize_answer(direct_pred) == normalize_answer(ref),
                'cot_single_correct': normalize_answer(cot_single_pred) == normalize_answer(ref),
                'cot_multi_correct': normalize_answer(cot_multi_pred) == normalize_answer(ref),
                'is_multi_verified': cot_multi_result.get("is_verified_valid", False)
            })
            
            # Save checkpoint if needed
            if checkpoint_count >= checkpoint_interval:
                checkpoint_count = 0
                checkpoint_df = pd.DataFrame(records)
                checkpoint_df.to_csv(results_file, index=False)
                print(f"✓ Saved checkpoint after {total} examples")
                
                # Also save current metrics
                em_direct = evaluate.load('exact_match')
                em_cot_single = evaluate.load('exact_match')
                em_cot_multi = evaluate.load('exact_match')
                
                # Add current examples to metrics
                for record in records:
                    em_direct.add_batch(
                        predictions=[record['direct_answer']], 
                        references=[record['reference']]
                    )
                    em_cot_single.add_batch(
                        predictions=[record['cot_single_answer']], 
                        references=[record['reference']]
                    )
                    em_cot_multi.add_batch(
                        predictions=[record['cot_multi_answer']], 
                        references=[record['reference']]
                    )
                
                # Calculate intermediate metrics
                direct_ci_low, direct_ci_high = confidence_interval(direct_correct, total)
                cot_single_ci_low, cot_single_ci_high = confidence_interval(cot_single_correct, total)
                cot_multi_ci_low, cot_multi_ci_high = confidence_interval(cot_multi_correct, total)
                
                interim_metrics = {
                    'direct_exact_match': em_direct.compute()['exact_match'],
                    'direct_f1': direct_f1_sum / total if total > 0 else 0,
                    'cot_single_exact_match': em_cot_single.compute()['exact_match'],
                    'cot_single_f1': cot_single_f1_sum / total if total > 0 else 0,
                    'cot_multi_exact_match': em_cot_multi.compute()['exact_match'],
                    'cot_multi_f1': cot_multi_f1_sum / total if total > 0 else 0,
                    'sample_size': total,
                    'direct_ci': [direct_ci_low, direct_ci_high],
                    'cot_single_ci': [cot_single_ci_low, cot_single_ci_high],
                    'cot_multi_ci': [cot_multi_ci_low, cot_multi_ci_high]
                }
                
                # Print interim metrics
                print(f"\nInterim results after {total} examples:")
                print(f"Direct Answer:    EM={interim_metrics['direct_exact_match']:.4f}, "
                      f"F1={interim_metrics['direct_f1']:.4f}")
                print(f"Single-agent CoT: EM={interim_metrics['cot_single_exact_match']:.4f}, "
                      f"F1={interim_metrics['cot_single_f1']:.4f}")
                print(f"Multi-agent CoT:  EM={interim_metrics['cot_multi_exact_match']:.4f}, "
                      f"F1={interim_metrics['cot_multi_f1']:.4f}\n")
                
                # Save interim metrics
                pd.DataFrame([interim_metrics]).to_csv(metrics_file, index=False)
            
            # Sleep proportional to token count rather than fixed 1s
            token_count = len(ENC.encode(direct_pred_raw + cot_pred_raw + cot_multi_raw))
            await asyncio.sleep(max(0.1, token_count / 10000))  # Adjust rate based on token count

        # Compute final overall metrics
        direct_ci_low, direct_ci_high = confidence_interval(direct_correct, total)
        cot_single_ci_low, cot_single_ci_high = confidence_interval(cot_single_correct, total)
        cot_multi_ci_low, cot_multi_ci_high = confidence_interval(cot_multi_correct, total)
        
        avg_direct_f1 = direct_f1_sum / total if total > 0 else 0
        avg_cot_single_f1 = cot_single_f1_sum / total if total > 0 else 0
        avg_cot_multi_f1 = cot_multi_f1_sum / total if total > 0 else 0
        
        # Get final exact match scores
        em_direct = evaluate.load('exact_match')
        em_cot_single = evaluate.load('exact_match')
        em_cot_multi = evaluate.load('exact_match')
        
        for record in records:
            em_direct.add_batch(
                predictions=[record['direct_answer']], 
                references=[record['reference']]
            )
            em_cot_single.add_batch(
                predictions=[record['cot_single_answer']], 
                references=[record['reference']]
            )
            em_cot_multi.add_batch(
                predictions=[record['cot_multi_answer']], 
                references=[record['reference']]
            )
        
        results = {
            'direct_exact_match': em_direct.compute()['exact_match'],
            'direct_f1': avg_direct_f1,
            'cot_single_exact_match': em_cot_single.compute()['exact_match'],
            'cot_single_f1': avg_cot_single_f1,
            'cot_multi_exact_match': em_cot_multi.compute()['exact_match'],
            'cot_multi_f1': avg_cot_multi_f1,
            'sample_size': total,
            'direct_ci': [direct_ci_low, direct_ci_high],
            'cot_single_ci': [cot_single_ci_low, cot_single_ci_high],
            'cot_multi_ci': [cot_multi_ci_low, cot_multi_ci_high]
        }

        # Save final results and metrics
        final_df = pd.DataFrame(records)
        final_df.to_csv(results_file, index=False)
        
        # Save final metrics
        metrics_df = pd.DataFrame([results])
        metrics_df.to_csv(metrics_file, index=False)
        
        # Print final hop distribution
        print("\nFinal hop distribution:")
        for hop, count in sorted(hop_counts.items()):
            print(f"  {hop}-hop examples: {count} ({count/total*100:.1f}%)")
        
        return final_df, results, hop_metrics_file

    result = asyncio.run(process())
    return result  # This will now be a tuple (final_df, results, hop_metrics_file)

def analyze_by_hop(df: pd.DataFrame, hop_metrics_file: Optional[str] = None) -> pd.DataFrame:
    """
    Analyze results grouped by hop count.
    
    Args:
        df: DataFrame with evaluation results
        hop_metrics_file: Path to save hop metrics (if None, generates a timestamped filename)
        
    Returns:
        DataFrame with hop-specific metrics
    """
    if hop_metrics_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        hop_metrics_file = f'metrics_by_hop_{timestamp}.csv'
        
    grouped = []
    for hop in sorted(df['hop_count'].unique()):
        subset = df[df['hop_count'] == hop]
        em_d = evaluate.load('exact_match')
        em_cs = evaluate.load('exact_match')
        em_cm = evaluate.load('exact_match')
        
        direct_preds = []
        cot_single_preds = []
        cot_multi_preds = []
        refs = []
        direct_correct = 0
        cot_single_correct = 0
        cot_multi_correct = 0
        direct_f1_sum = 0
        cot_single_f1_sum = 0
        cot_multi_f1_sum = 0
        total = len(subset)
        
        for _, row in subset.iterrows():
            direct_preds.append(row['direct_answer'])
            cot_single_preds.append(row['cot_single_answer'])
            cot_multi_preds.append(row['cot_multi_answer'])
            refs.append(row['reference'])
            
            # Sum up F1 scores
            direct_f1_sum += row['direct_f1']
            cot_single_f1_sum += row['cot_single_f1']
            cot_multi_f1_sum += row['cot_multi_f1']
            
            # Count exact matches
            if normalize_answer(row['direct_answer']) == normalize_answer(row['reference']):
                direct_correct += 1
            if normalize_answer(row['cot_single_answer']) == normalize_answer(row['reference']):
                cot_single_correct += 1
            if normalize_answer(row['cot_multi_answer']) == normalize_answer(row['reference']):
                cot_multi_correct += 1
        
        em_d.add_batch(predictions=direct_preds, references=refs)
        em_cs.add_batch(predictions=cot_single_preds, references=refs)
        em_cm.add_batch(predictions=cot_multi_preds, references=refs)
        
        # Calculate confidence intervals
        direct_ci_low, direct_ci_high = confidence_interval(direct_correct, total)
        cot_single_ci_low, cot_single_ci_high = confidence_interval(cot_single_correct, total)
        cot_multi_ci_low, cot_multi_ci_high = confidence_interval(cot_multi_correct, total)
        
        direct_f1_avg = direct_f1_sum / total if total > 0 else 0
        cot_single_f1_avg = cot_single_f1_sum / total if total > 0 else 0
        cot_multi_f1_avg = cot_multi_f1_sum / total if total > 0 else 0
        
        grouped.append({
            'hop_count': hop,
            'sample_size': total,
            'direct_em': em_d.compute()['exact_match'],
            'direct_f1': direct_f1_avg,
            'direct_ci_low': direct_ci_low,
            'direct_ci_high': direct_ci_high,
            'cot_single_em': em_cs.compute()['exact_match'],
            'cot_single_f1': cot_single_f1_avg,
            'cot_single_ci_low': cot_single_ci_low,
            'cot_single_ci_high': cot_single_ci_high,
            'cot_multi_em': em_cm.compute()['exact_match'],
            'cot_multi_f1': cot_multi_f1_avg,
            'cot_multi_ci_low': cot_multi_ci_low,
            'cot_multi_ci_high': cot_multi_ci_high,
            'avg_direct_time': subset['direct_time'].mean(),
            'avg_cot_single_time': subset['cot_single_time'].mean(),
            'avg_cot_multi_time': subset['cot_multi_time'].mean()
        })
    
    hop_df = pd.DataFrame(grouped)
    hop_df.to_csv(hop_metrics_file, index=False)
    
    print("\nMetrics by hop count:")
    for _, row in hop_df.iterrows():
        print(f"\n== Hop Count: {row['hop_count']} (n={row['sample_size']}) ==")
        print(f"Direct:      EM={row['direct_em']:.4f}, F1={row['direct_f1']:.4f}, "
              f"95% CI=[{row['direct_ci_low']:.4f}, {row['direct_ci_high']:.4f}]")
        print(f"Single CoT:  EM={row['cot_single_em']:.4f}, F1={row['cot_single_f1']:.4f}, "
              f"95% CI=[{row['cot_single_ci_low']:.4f}, {row['cot_single_ci_high']:.4f}]")
        print(f"Multi CoT:   EM={row['cot_multi_em']:.4f}, F1={row['cot_multi_f1']:.4f}, "
              f"95% CI=[{row['cot_multi_ci_low']:.4f}, {row['cot_multi_ci_high']:.4f}]")
        print(f"Latency: Direct={row['avg_direct_time']:.2f}s, "
              f"Single CoT={row['avg_cot_single_time']:.2f}s, "
              f"Multi CoT={row['avg_cot_multi_time']:.2f}s")
    
    return hop_df

def analyze_dataset_hop_distribution(splits: List[str] = ['validation'], sample_size: int = 100, seed: int = 42) -> Dict[str, Dict[int, int]]:
    """
    Analyze the distribution of hop counts in the dataset.
    
    Args:
        splits: Dataset splits to analyze
        sample_size: Sample size (for streaming datasets)
        seed: Random seed
        
    Returns:
        Dictionary mapping split names to hop count distributions
    """
    results = {}
    
    for split in splits:
        dataset = load_dataset('fladhak/musique', split=split)
        
        # Count hops for all examples
        hop_counts = {}
        for example in dataset:
            hop_count = len(example['question_decomposition'])
            hop_counts[hop_count] = hop_counts.get(hop_count, 0) + 1
        
        total = len(dataset)
        print(f"\nHop distribution in {split} set (total of {total} examples):")
        for hop, count in sorted(hop_counts.items()):
            print(f"  {hop}-hop examples: {count} ({count/total*100:.1f}%)")
        
        results[split] = hop_counts
    
    return results

if __name__ == '__main__':
    """Command-line interface for the evaluation harness."""
    # Default configuration
    num_examples = 5
    resume_from = None
    checkpoint_interval = 20
    analyze_hops = False
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate models on MuSiQue dataset')
    parser.add_argument('--num_examples', '-n', type=int, default=5,
                        help='Number of examples to evaluate')
    parser.add_argument('--resume', '-r', type=str, default=None,
                        help='Path to previous results file to resume from')
    parser.add_argument('--checkpoint', '-c', type=int, default=20,
                        help='Save checkpoint after this many examples')
    parser.add_argument('--analyze-hops', '-a', action='store_true',
                        help='Analyze hop distribution in the dataset')
    
    args = parser.parse_args()
    num_examples = args.num_examples
    resume_from = args.resume
    checkpoint_interval = args.checkpoint
    analyze_hops = args.analyze_hops
    
    if analyze_hops:
        analyze_dataset_hop_distribution()
        sys.exit(0)
        
    if resume_from:
        print(f"Will resume from {resume_from}")
    
    print(f"Starting evaluation on {num_examples} examples from MuSiQue validation set...")
    print(f"Saving checkpoints every {checkpoint_interval} examples")
    
    result = run_evaluation(
        num_examples=num_examples,
        resume_from=resume_from,
        checkpoint_interval=checkpoint_interval
    )
    df_results, metrics, hop_metrics_file = result
    
    # Calculate and print overall results
    overall = {
        'direct_exact_match': df_results['direct_correct'].mean(),
        'direct_f1': df_results['direct_f1'].mean(),
        'cot_single_exact_match': df_results['cot_single_correct'].mean(),
        'cot_single_f1': df_results['cot_single_f1'].mean(),
        'cot_multi_exact_match': df_results['cot_multi_correct'].mean(),
        'cot_multi_f1': df_results['cot_multi_f1'].mean(),
        'sample_size': len(df_results),
    }
    
    # Calculate confidence intervals
    direct_correct = df_results['direct_correct'].sum()
    cot_single_correct = df_results['cot_single_correct'].sum()
    cot_multi_correct = df_results['cot_multi_correct'].sum()
    total = len(df_results)
    
    direct_ci_low, direct_ci_high = confidence_interval(direct_correct, total)
    cot_single_ci_low, cot_single_ci_high = confidence_interval(cot_single_correct, total)
    cot_multi_ci_low, cot_multi_ci_high = confidence_interval(cot_multi_correct, total)
    
    overall['direct_ci'] = [direct_ci_low, direct_ci_high]
    overall['cot_single_ci'] = [cot_single_ci_low, cot_single_ci_high]
    overall['cot_multi_ci'] = [cot_multi_ci_low, cot_multi_ci_high]
    
    # Print summary
    print("\nOverall results:")
    print(f"Sample size: {overall['sample_size']}")
    print(f"Direct Answer:    EM={overall['direct_exact_match']:.4f}, F1={overall['direct_f1']:.4f}, "
          f"95% CI=[{overall['direct_ci'][0]:.4f}, {overall['direct_ci'][1]:.4f}]")
    print(f"Single-agent CoT: EM={overall['cot_single_exact_match']:.4f}, F1={overall['cot_single_f1']:.4f}, "
          f"95% CI=[{overall['cot_single_ci'][0]:.4f}, {overall['cot_single_ci'][1]:.4f}]")
    print(f"Multi-agent CoT:  EM={overall['cot_multi_exact_match']:.4f}, F1={overall['cot_multi_f1']:.4f}, "
          f"95% CI=[{overall['cot_multi_ci'][0]:.4f}, {overall['cot_multi_ci'][1]:.4f}]")
    
    # Print hop distribution summary
    hop_counts = df_results['hop_count'].value_counts().sort_index()
    print("\nHop distribution summary:")
    for hop, count in hop_counts.items():
        print(f"  {hop}-hop examples: {count} ({count/len(df_results)*100:.1f}%)")
    
    # Run hop-specific analysis
    analyze_by_hop(df_results, hop_metrics_file)