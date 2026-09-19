"""Public-state controls for the controlled dependency workload."""
from __future__ import annotations

from conductor.coordination.specialists import artifacts
from conductor.coordination.workloads import parse_request
from conductor.schema import ExecutionState, RoutingDecision


class WorkflowRulePolicy:
    """Deterministic capability scheduler with no private task metadata."""

    name = "public_state_rules"
    last_tokens = 0
    last_cost_usd = 0.0

    @staticmethod
    def _decision(agents: list[str], k: int, mode: str = "sequential") -> RoutingDecision:
        return RoutingDecision(agents[:k], mode).validate(k)

    def route(self, state: ExecutionState, k: int) -> RoutingDecision:
        request = parse_request(state)
        records = artifacts(state)
        kinds = {record["kind"] for record in records}
        result = next((record for record in reversed(records)
                       if record["kind"] == "candidate" and record.get("purpose") == "result"), None)
        if result is not None:
            if request["verification"] and not any(record["kind"] in {"verification", "execution"}
                                                     and record.get("verified") for record in records):
                return self._decision(["verifier"], k)
            return RoutingDecision([], terminate=True)
        operation, parameters = request["operation"], request["parameters"]
        if operation == "calculate":
            if "facts" not in kinds:
                chain = ["retriever", "math"]
                if parameters.get("post_transform"):
                    chain.append("coder")
                elif request["verification"]:
                    chain.append("verifier")
                return self._decision(chain, k)
            if parameters.get("post_transform"):
                arithmetic = any(record.get("purpose") == "arithmetic" for record in records)
                return self._decision(["coder"] if arithmetic else ["math", "coder"], k)
            return self._decision(["math"], k)
        if operation == "transform":
            if "facts" not in kinds:
                chain = ["retriever"]
                if parameters.get("pre_scale"):
                    chain.append("math")
                chain.append("coder")
                return self._decision(chain, k)
            if parameters.get("pre_scale") and not any(record.get("purpose") == "arithmetic" for record in records):
                return self._decision(["math", "coder"], k)
            return self._decision(["coder"], k)
        if operation == "research":
            if "facts" not in kinds:
                chain = ["retriever"]
                if parameters.get("threshold_offset"):
                    chain.extend(["math", "researcher"])
                else:
                    chain.append("researcher")
                return self._decision(chain, k)
            if parameters.get("threshold_offset") and not any(record.get("purpose") == "threshold" for record in records):
                return self._decision(["math", "researcher"], k)
            return self._decision(["researcher"], k)
        if operation == "combine":
            required = []
            if "facts" not in kinds:
                required.append("retriever")
            if parameters.get("rate_selector_from_tool") and "tool_result" not in kinds:
                required.append("tool_executor")
            if not parameters.get("rate_selector_from_tool") and "research" not in kinds:
                required.append("researcher")
            if required:
                return self._decision(required, k, "parallel" if len(required) > 1 else "sequential")
            if "research" not in kinds:
                return self._decision(["researcher"], k)
            return self._decision(["math", "verifier"] if request["verification"] else ["math"], k)
        if operation == "resolve_conflict":
            if "facts" not in kinds:
                return self._decision(["retriever", "critic", "researcher"], k)
            if "critique" not in kinds:
                return self._decision(["critic", "researcher"], k)
            if "research" not in kinds:
                return self._decision(["researcher"], k)
            if parameters.get("post_transform"):
                return self._decision(["coder"], k)
            return self._decision(["researcher"], k)
        if operation == "read_tool":
            failure = next((record for record in reversed(records) if record["kind"] == "tool_failure"), None)
            if failure is None:
                return self._decision(["tool_executor"], k)
            if failure["value"].get("fallback_fact_ref"):
                return self._decision(["retriever", "verifier"] if request["verification"] else ["retriever"], k)
            if "recovery" not in kinds:
                return self._decision(["researcher", "tool_executor", "verifier"] if request["verification"]
                                      else ["researcher", "tool_executor"], k)
            return self._decision(["tool_executor", "verifier"] if request["verification"] else ["tool_executor"], k)
        return RoutingDecision([], terminate=True)
