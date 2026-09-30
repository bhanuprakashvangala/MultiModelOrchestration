"""
Pick-and-Spin: Multi-Objective Routing for Privacy-Preserving LLM Orchestration
Core routing implementation for DAI Workshop experiments
"""

import os
import time
import random
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import yaml
from openai import OpenAI


@dataclass
class RoutingDecision:
    """Represents a routing decision with full transparency"""
    selected_model: str
    strategy: str
    query_complexity: str
    is_privacy_sensitive: bool
    model_tier: str
    quality_score: float
    speed_score: float
    cost_score: float
    combined_score: float
    alternatives_considered: List[Dict[str, Any]]
    decision_rationale: str
    timestamp: str


@dataclass
class QueryResult:
    """Complete result from a query execution"""
    query_id: str
    query: str
    routing_decision: RoutingDecision
    response: str
    latency_ms: float
    tokens_used: int
    cost_score: float
    success: bool
    error: Optional[str] = None
    timestamp: str = None


class PickAndSpinRouter:
    """
    Pick-and-Spin routing system implementing multi-objective optimization
    for privacy-preserving multi-LLM orchestration
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize router with configuration"""

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # OpenAI-compatible client (endpoint and key from the environment)
        self.client = OpenAI(
            api_key=os.environ.get("LLM_API_KEY", "none"),
            base_url=os.environ.get("LLM_API_BASE", self.config['api']['base_url'])
        )

        self.models = self.config['models']
        self.strategies = self.config['routing_strategies']

        # Initialize tracking
        self.query_history = []
        self.model_usage_counts = {model: 0 for model in self.models.keys()}

        print(f"> Pick-and-Spin Router initialized")
        print(f"> Available models: {list(self.models.keys())}")
        print(f"> Available strategies: {list(self.strategies.keys())}")

    def detect_query_complexity_llm(self, query: str) -> Tuple[str, int]:
        """
        Detect query complexity using LLM-based analysis

        Uses a transformer model (gemma3) to intelligently classify complexity
        instead of keyword-based heuristics. Falls back to keyword method on error.

        Returns:
            Tuple of (complexity_level, suggested_max_tokens)
        """
        complexity_config = self.config['complexity_detection']
        llm_model = complexity_config.get('llm_model', 'gemma3')

        # Construct prompt for complexity detection
        prompt = f"""Analyze the complexity of this question and classify it as SIMPLE, MEDIUM, or COMPLEX.

- SIMPLE: Basic facts, definitions, arithmetic, short answers (e.g., "What is 2+2?", "Define photosynthesis")
- MEDIUM: Explanations, comparisons, moderate reasoning (e.g., "Explain how photosynthesis works", "Compare X and Y")
- COMPLEX: Deep analysis, multi-step reasoning, synthesis, technical depth (e.g., "Analyze the implications of...", "Evaluate the relationship between...")

Question: {query}

Respond with ONLY ONE WORD: SIMPLE, MEDIUM, or COMPLEX"""

        try:
            # Make API call to LLM for complexity detection
            completion = self.client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic for consistency
                max_tokens=10     # Only need one word response
            )

            response = completion.choices[0].message.content.strip().upper()

            # Parse response and map to complexity level
            if "SIMPLE" in response:
                return "simple", complexity_config['simple_max_tokens']
            elif "MEDIUM" in response:
                return "medium", complexity_config['medium_max_tokens']
            elif "COMPLEX" in response:
                return "complex", complexity_config['complex_max_tokens']
            else:
                # Unparseable response - fall back to keyword method
                print(f"  [Warning] LLM returned unparseable complexity: '{response}', falling back to keywords")
                return self._detect_query_complexity_keywords(query)

        except Exception as e:
            # API error - fall back to keyword method
            print(f"  [Warning] LLM complexity detection failed: {e}, falling back to keywords")
            return self._detect_query_complexity_keywords(query)

    def _detect_query_complexity_keywords(self, query: str) -> Tuple[str, int]:
        """
        Detect query complexity based on keyword heuristics (fallback method)

        Returns:
            Tuple of (complexity_level, suggested_max_tokens)
        """
        query_lower = query.lower()
        complexity_config = self.config['complexity_detection']

        # Check for complexity indicators
        if any(indicator in query_lower for indicator in complexity_config['complex_indicators']):
            return "complex", complexity_config['complex_max_tokens']
        elif any(indicator in query_lower for indicator in complexity_config['medium_indicators']):
            return "medium", complexity_config['medium_max_tokens']
        else:
            return "simple", complexity_config['simple_max_tokens']

    def detect_query_complexity(self, query: str) -> Tuple[str, int]:
        """
        Detect query complexity - uses LLM or keyword-based method

        Checks config to determine which method to use:
        - If use_llm=true: Uses transformer model for intelligent classification
        - If use_llm=false: Uses keyword-based heuristics

        Returns:
            Tuple of (complexity_level, suggested_max_tokens)
        """
        complexity_config = self.config['complexity_detection']
        use_llm = complexity_config.get('use_llm', False)

        if use_llm:
            return self.detect_query_complexity_llm(query)
        else:
            return self._detect_query_complexity_keywords(query)

    def detect_privacy_sensitive(self, query: str) -> bool:
        """Detect if query contains privacy-sensitive content"""
        query_lower = query.lower()
        sensitive_keywords = self.config['privacy']['sensitive_keywords']

        return any(keyword in query_lower for keyword in sensitive_keywords)

    def calculate_model_scores(
        self,
        complexity: str,
        strategy: str
    ) -> List[Dict[str, Any]]:
        """
        Calculate scores for each model based on strategy and complexity

        Returns:
            List of models with their scores, sorted by combined score
        """
        strategy_config = self.strategies[strategy]
        weights = strategy_config['weights']

        scored_models = []

        for model_name, model_config in self.models.items():
            # Base scores from configuration
            quality_score = model_config['quality_weight']
            speed_score = model_config['speed_weight']
            cost_score = 1.0 - model_config['cost_weight']  # Invert: lower cost = higher score

            # Complexity adjustment
            if complexity == "simple" and model_config['tier'] in ['xlarge', 'large']:
                # Penalize oversized models for simple queries
                cost_score *= 0.5
            elif complexity == "complex" and model_config['tier'] in ['small']:
                # Penalize undersized models for complex queries
                quality_score *= 0.6

            # Calculate combined score based on strategy weights
            combined_score = (
                weights['quality'] * quality_score +
                weights['speed'] * speed_score +
                weights['cost'] * cost_score
            )

            scored_models.append({
                'model': model_name,
                'tier': model_config['tier'],
                'quality_score': round(quality_score, 3),
                'speed_score': round(speed_score, 3),
                'cost_score': round(cost_score, 3),
                'combined_score': round(combined_score, 3)
            })

        # Sort by combined score (descending)
        scored_models.sort(key=lambda x: x['combined_score'], reverse=True)

        return scored_models

    def select_model(
        self,
        query: str,
        strategy: str = "balanced",
        explain: bool = True
    ) -> RoutingDecision:
        """
        Select optimal model for a query using Pick-and-Spin routing

        Args:
            query: User query
            strategy: Routing strategy (quality/speed/cost/balanced/baseline)
            explain: Whether to generate detailed explanation

        Returns:
            RoutingDecision with full transparency
        """

        # Analyze query
        complexity, suggested_tokens = self.detect_query_complexity(query)
        is_privacy_sensitive = self.detect_privacy_sensitive(query)

        # Handle baseline (random) strategy
        if strategy == "baseline":
            selected_model = random.choice(list(self.models.keys()))
            model_config = self.models[selected_model]

            return RoutingDecision(
                selected_model=selected_model,
                strategy=strategy,
                query_complexity=complexity,
                is_privacy_sensitive=is_privacy_sensitive,
                model_tier=model_config['tier'],
                quality_score=0.0,
                speed_score=0.0,
                cost_score=0.0,
                combined_score=0.0,
                alternatives_considered=[],
                decision_rationale="Random selection (baseline strategy)",
                timestamp=datetime.now().isoformat()
            )

        # Calculate scores for all models
        scored_models = self.calculate_model_scores(complexity, strategy)

        # Select top model
        top_choice = scored_models[0]
        selected_model = top_choice['model']

        # Generate explanation
        if explain:
            rationale = self._generate_rationale(
                strategy=strategy,
                complexity=complexity,
                selected_model=selected_model,
                top_choice=top_choice,
                alternatives=scored_models[1:3]
            )
        else:
            rationale = f"Selected {selected_model} using {strategy} strategy"

        decision = RoutingDecision(
            selected_model=selected_model,
            strategy=strategy,
            query_complexity=complexity,
            is_privacy_sensitive=is_privacy_sensitive,
            model_tier=top_choice['tier'],
            quality_score=top_choice['quality_score'],
            speed_score=top_choice['speed_score'],
            cost_score=top_choice['cost_score'],
            combined_score=top_choice['combined_score'],
            alternatives_considered=scored_models[1:4],  # Top 3 alternatives
            decision_rationale=rationale,
            timestamp=datetime.now().isoformat()
        )

        return decision

    def _generate_rationale(
        self,
        strategy: str,
        complexity: str,
        selected_model: str,
        top_choice: Dict,
        alternatives: List[Dict]
    ) -> str:
        """Generate human-readable explanation for routing decision"""

        model_params = self.models[selected_model]['params']
        model_tier = top_choice['tier']

        rationale = f"Selected {selected_model} ({model_params}, {model_tier}-tier) for {strategy} strategy. "

        # Strategy-specific explanation
        if strategy == "quality":
            rationale += f"This model offers highest quality (score: {top_choice['quality_score']:.2f}). "
        elif strategy == "speed":
            rationale += f"This model offers fastest response (speed score: {top_choice['speed_score']:.2f}). "
        elif strategy == "cost":
            rationale += f"This model is most cost-effective (cost score: {top_choice['cost_score']:.2f}). "
        elif strategy == "balanced":
            rationale += f"This model balances quality, speed, and cost (combined: {top_choice['combined_score']:.2f}). "

        # Complexity consideration
        if complexity == "simple":
            rationale += "Query complexity is low, small model sufficient. "
        elif complexity == "complex":
            rationale += "Query complexity is high, larger model recommended. "

        # Alternatives
        if alternatives:
            alt_names = [a['model'] for a in alternatives[:2]]
            rationale += f"Alternatives considered: {', '.join(alt_names)}."

        return rationale

    def execute_query(
        self,
        query_id: str,
        query: str,
        strategy: str = "balanced",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7
    ) -> QueryResult:
        """
        Execute a query with Pick-and-Spin routing

        Returns:
            QueryResult with full execution details
        """

        # Make routing decision
        routing_decision = self.select_model(query, strategy, explain=True)

        # Determine max tokens
        if max_tokens is None:
            _, max_tokens = self.detect_query_complexity(query)

        # Execute query
        start_time = time.time()

        try:
            completion = self.client.chat.completions.create(
                model=routing_decision.selected_model,
                messages=[{"role": "user", "content": query}],
                temperature=temperature,
                max_tokens=max_tokens
            )

            latency_ms = (time.time() - start_time) * 1000
            response = completion.choices[0].message.content
            tokens_used = completion.usage.total_tokens if completion.usage else 0

            # Calculate cost score
            model_cost_weight = self.models[routing_decision.selected_model]['cost_weight']
            cost_score = tokens_used * model_cost_weight

            # Update tracking
            self.model_usage_counts[routing_decision.selected_model] += 1

            result = QueryResult(
                query_id=query_id,
                query=query if not routing_decision.is_privacy_sensitive else "[REDACTED]",
                routing_decision=routing_decision,
                response=response[:200] + "..." if len(response) > 200 else response,
                latency_ms=round(latency_ms, 2),
                tokens_used=tokens_used,
                cost_score=round(cost_score, 2),
                success=True,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            result = QueryResult(
                query_id=query_id,
                query=query if not routing_decision.is_privacy_sensitive else "[REDACTED]",
                routing_decision=routing_decision,
                response="",
                latency_ms=round(latency_ms, 2),
                tokens_used=0,
                cost_score=0,
                success=False,
                error=str(e),
                timestamp=datetime.now().isoformat()
            )

        # Store in history
        self.query_history.append(result)

        return result

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get statistics about model usage"""
        total_queries = sum(self.model_usage_counts.values())

        return {
            'total_queries': total_queries,
            'model_usage': self.model_usage_counts,
            'model_distribution': {
                model: count / total_queries if total_queries > 0 else 0
                for model, count in self.model_usage_counts.items()
            }
        }


def main():
    """Test the routing system"""
    print("\n" + "="*80)
    print("PICK-AND-SPIN ROUTING SYSTEM TEST")
    print("="*80 + "\n")

    # Initialize router
    router = PickAndSpinRouter(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))

    # Test queries
    test_queries = [
        ("What is the capital of France?", "speed"),
        ("Explain quantum computing in detail", "quality"),
        ("What are common treatments for diabetes?", "balanced")
    ]

    for i, (query, strategy) in enumerate(test_queries, 1):
        print(f"\n[Test {i}] Strategy: {strategy}")
        print(f"Query: {query}\n")

        result = router.execute_query(
            query_id=f"test_{i}",
            query=query,
            strategy=strategy
        )

        if result.success:
            print(f"> Model: {result.routing_decision.selected_model}")
            print(f"> Complexity: {result.routing_decision.query_complexity}")
            print(f"> Privacy: {result.routing_decision.is_privacy_sensitive}")
            print(f"> Latency: {result.latency_ms:.0f}ms")
            print(f"> Tokens: {result.tokens_used}")
            print(f"> Rationale: {result.routing_decision.decision_rationale}")
        else:
            print(f"X Error: {result.error}")

    # Print statistics
    print("\n" + "="*80)
    stats = router.get_usage_statistics()
    print(f"Total queries: {stats['total_queries']}")
    print("Model usage:", stats['model_usage'])


if __name__ == "__main__":
    main()
