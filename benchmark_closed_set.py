#!/usr/bin/env python3
"""Closed-set API-Bank benchmark for compact language models.

This script refactors the core logic from the original experimental notebook into a
clean command-line workflow suitable for GitHub release and reproducibility.

Main capabilities
-----------------
1. Run a closed-set, single-call benchmark on API-Bank samples.
2. Export detailed and summary CSV files for each run.
3. Rebuild paper-ready tables from previously saved run folders.
4. Export an operational reliability / deployability table.

The implementation intentionally keeps the benchmark lightweight:
- OpenAI-compatible inference API wrapper
- minimal API-Bank executor
- BM25 tool retrieval for closed-set schema exposure
- deterministic prompting for function-calling evaluation
"""

from __future__ import annotations

import argparse
import inspect
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yaml
from openai import OpenAI
from rank_bm25 import BM25Okapi


EXCEPT_FILES = {"__init__.py", "api.py", "tool_search.py"}
SYSTEM_PROMPT_JSON = """You are a tool-use assistant.
Choose exactly one tool from the available tools and return ONLY one compact JSON object.
Do not add markdown, code fences, XML, explanation, or extra text.
The output format must be exactly:
{"name":"TOOL_NAME","arguments":{"arg1":"value1"}}
Rules:
- Use an exact tool name from the available tools.
- Include every required argument.
- If the request is ambiguous, make the single best tool choice.
- If the task would normally require multiple steps, choose the best first tool only.
"""

SYSTEM_PROMPT_XML = """You are a tool-use assistant.
Choose exactly one tool from the available tools and return ONLY one tool call wrapped in XML tags.
The output format must be exactly:
<tool_call>
{"name":"TOOL_NAME","arguments":{"arg1":"value1"}}
</tool_call>
Do not add explanation or extra text.
"""


@dataclass
class BenchmarkPaths:
    deepagent_root: Path
    api_bank_root: Path
    apis_dir: Path
    db_dir: Path
    level1_dir: Path
    runs_root: Path


@dataclass
class BenchmarkSettings:
    seed: int = 42
    max_samples: int = 50
    require_single_call: bool = True
    require_level1_filename: bool = True
    balanced_by_api: bool = True
    min_examples_per_api_first: bool = True
    exclude_unsupported_tools: bool = True
    manual_exclude_tools: Tuple[str, ...] = ("Translate",)
    tool_top_k: int = 12
    bm25_query_lowercase: bool = True
    temperature: float = 0.0
    global_max_tokens: int = 256
    global_timeout_seconds: int = 45


def resolve_env_placeholders(value: Any) -> Any:
    if isinstance(value, str):
        m = re.fullmatch(r"\$\{([A-Z0-9_]+)\}", value.strip())
        if m:
            return os.environ.get(m.group(1), "")
    if isinstance(value, list):
        return [resolve_env_placeholders(v) for v in value]
    if isinstance(value, dict):
        return {k: resolve_env_placeholders(v) for k, v in value.items()}
    return value


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return resolve_env_placeholders(yaml.safe_load(f))


def ensure_paths(deepagent_root: Path, runs_root: Path) -> BenchmarkPaths:
    deepagent_root = deepagent_root.resolve()
    api_bank_root = deepagent_root / "data" / "API-Bank"
    apis_dir = api_bank_root / "apis"
    db_dir = api_bank_root / "init_database"
    level1_dir = api_bank_root / "lv1-lv2-samples" / "level-1-given-desc-e2e"

    for p in [api_bank_root, apis_dir, db_dir, level1_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Required path not found: {p}")

    runs_root.mkdir(parents=True, exist_ok=True)
    return BenchmarkPaths(
        deepagent_root=deepagent_root,
        api_bank_root=api_bank_root,
        apis_dir=apis_dir,
        db_dir=db_dir,
        level1_dir=level1_dir,
        runs_root=runs_root.resolve(),
    )


def load_init_databases(database_dir: Path) -> Dict[str, Any]:
    dbs: Dict[str, Any] = {}
    if database_dir.exists():
        for fp in database_dir.glob("*.json"):
            try:
                dbs[fp.stem] = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                pass
    return dbs


def import_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MinimalAPIBankExecutor:
    def __init__(self, apis_dir: Path, database_dir: Path):
        self.apis_dir = Path(apis_dir)
        self.database_dir = Path(database_dir)
        self.init_databases = load_init_databases(self.database_dir)
        self.tool_classes: Dict[str, Any] = {}
        self.tool_infos: Dict[str, Dict[str, Any]] = {}
        self.token_checker = None
        self.failed_modules: Dict[str, str] = {}
        self._load_all_tools()

    def _load_all_tools(self) -> None:
        apis_dir_str = str(self.apis_dir.resolve())
        if apis_dir_str not in sys.path:
            sys.path.append(apis_dir_str)

        for file in sorted(self.apis_dir.glob("*.py")):
            if file.name in EXCEPT_FILES:
                continue
            try:
                module = import_module_from_path(file.stem, file)
            except Exception as e:
                self.failed_modules[file.name] = repr(e)
                continue

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if not isinstance(attr, type):
                    continue
                if not (
                    hasattr(attr, "description")
                    and hasattr(attr, "input_parameters")
                    and hasattr(attr, "output_parameters")
                ):
                    continue
                self.tool_classes[attr_name] = attr
                self.tool_infos[attr_name] = {
                    "name": attr_name,
                    "description": getattr(attr, "description", ""),
                    "input_parameters": getattr(attr, "input_parameters", {}),
                    "output_parameters": getattr(attr, "output_parameters", {}),
                }
                if attr_name == "CheckToken":
                    try:
                        kwargs = {}
                        if hasattr(attr, "database_name") and attr.database_name in self.init_databases:
                            kwargs["init_database"] = self.init_databases[attr.database_name]
                        self.token_checker = attr(**kwargs) if kwargs else attr()
                    except Exception:
                        self.token_checker = None

    def get_tool_openai_schema(self, tool_name: str) -> Dict[str, Any]:
        info = self.tool_infos[tool_name]
        props: Dict[str, Dict[str, Any]] = {}
        required: List[str] = []
        type_map = {
            "str": "string",
            "string": "string",
            "int": "integer",
            "integer": "integer",
            "float": "number",
            "number": "number",
            "bool": "boolean",
            "boolean": "boolean",
            "list": "array",
            "dict": "object",
        }
        for key, spec in info["input_parameters"].items():
            t = str(spec.get("type", "string")).lower()
            props[key] = {
                "type": type_map.get(t, "string"),
                "description": spec.get("description", ""),
            }
            required.append(key)
        return {
            "name": tool_name,
            "description": info["description"],
            "parameters": {"type": "object", "properties": props, "required": required},
        }

    def build_tool_instance(self, tool_name: str):
        tool_class = self.tool_classes[tool_name]
        init_kwargs = {}
        if hasattr(tool_class, "database_name") and getattr(tool_class, "database_name") in self.init_databases:
            if "init_database" in inspect.signature(tool_class.__init__).parameters:
                init_kwargs["init_database"] = self.init_databases[getattr(tool_class, "database_name")]
        try:
            needs_token = "token" in getattr(tool_class, "input_parameters", {})
        except Exception:
            needs_token = False
        if needs_token and self.token_checker is not None:
            if "token_checker" in inspect.signature(tool_class.__init__).parameters:
                init_kwargs["token_checker"] = self.token_checker
        return tool_class(**init_kwargs) if init_kwargs else tool_class()

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self.tool_classes:
            return {"success": False, "error": f"Tool {tool_name} not found", "result": None}
        try:
            instance = self.build_tool_instance(tool_name)
            result = instance.call(**arguments)
            return {"success": True, "error": None, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e), "result": None}

    def check_correctness(
        self,
        tool_name: str,
        predicted_result: Dict[str, Any],
        groundtruth_result: Dict[str, Any],
    ) -> Optional[bool]:
        if tool_name not in self.tool_classes:
            return None
        try:
            instance = self.build_tool_instance(tool_name)
            if hasattr(instance, "check_api_call_correctness"):
                return bool(instance.check_api_call_correctness(predicted_result, groundtruth_result))
        except Exception:
            return None
        return None


class ToolCatalog:
    def __init__(self, executor: MinimalAPIBankExecutor, settings: BenchmarkSettings):
        self.executor = executor
        self.settings = settings
        self.tool_names = sorted(executor.tool_infos.keys())
        self.tool_schemas = {name: executor.get_tool_openai_schema(name) for name in self.tool_names}
        self.tool_texts = {name: self._build_tool_text(name) for name in self.tool_names}
        self.tokenized_corpus = [self._tokenize(self.tool_texts[name]) for name in self.tool_names]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _build_tool_text(self, tool_name: str) -> str:
        info = self.executor.tool_infos[tool_name]
        parts = [tool_name, info.get("description", "")]
        for k, spec in info.get("input_parameters", {}).items():
            parts.extend([k, spec.get("description", ""), str(spec.get("type", ""))])
        return " ".join(str(x) for x in parts if x)

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower() if self.settings.bm25_query_lowercase else text
        return re.findall(r"[a-zA-Z0-9_]+", text)

    def shortlist(self, query: str, top_k: Optional[int] = None) -> List[str]:
        top_k = top_k or self.settings.tool_top_k
        q = query.lower() if self.settings.bm25_query_lowercase else query
        toks = self._tokenize(q)
        scores = self.bm25.get_scores(toks)
        ranked = sorted(zip(self.tool_names, scores), key=lambda x: x[1], reverse=True)
        return [name for name, _ in ranked[:top_k]]

    def schemas_for_query(self, query: str, schema_mode: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        if schema_mode == "full":
            names = list(self.tool_names)
        elif schema_mode == "bm25_topk":
            names = self.shortlist(query, top_k=self.settings.tool_top_k)
        else:
            raise ValueError(f"Unknown schema_mode: {schema_mode}")
        return names, [self.tool_schemas[n] for n in names]


def load_api_bank_samples(level1_dir: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for fp in sorted(level1_dir.glob("*.jsonl")):
        chat_history: List[Dict[str, Any]] = []
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chat_history.append(json.loads(line))
        user_query = None
        api_calls = []
        for item in chat_history:
            if item.get("role") == "User" and user_query is None:
                user_query = item.get("text", "")
            elif item.get("role") == "API":
                api_calls.append(
                    {
                        "api_name": item.get("api_name"),
                        "param_dict": item.get("param_dict", {}),
                        "result": item.get("result"),
                    }
                )
        if user_query and api_calls:
            samples.append(
                {
                    "file": fp.name,
                    "query": user_query,
                    "api_calls": api_calls,
                    "chat_history": chat_history,
                    "num_api_calls": len(api_calls),
                    "first_api_name": api_calls[0]["api_name"],
                    "is_level1_name": ("level-1" in fp.name),
                }
            )
    return samples


def is_supported_sample(
    sample: Dict[str, Any],
    executor: MinimalAPIBankExecutor,
    settings: BenchmarkSettings,
) -> Tuple[bool, str]:
    if settings.require_level1_filename and not sample["is_level1_name"]:
        return False, "not_level1_filename"
    if settings.require_single_call and sample["num_api_calls"] != 1:
        return False, "not_single_call"
    gold_api = sample["first_api_name"]
    if gold_api in settings.manual_exclude_tools:
        return False, "manually_excluded_tool"
    if settings.exclude_unsupported_tools and gold_api not in executor.tool_infos:
        return False, "unsupported_tool"
    return True, "ok"


def build_balanced_subset(
    api_buckets: Dict[str, List[Dict[str, Any]]],
    max_samples: int,
    seed: int,
    min_examples_per_api_first: bool,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    api_names = sorted(api_buckets.keys())
    local_buckets = {k: list(v) for k, v in api_buckets.items()}
    for k in api_names:
        rng.shuffle(local_buckets[k])

    chosen: List[Dict[str, Any]] = []
    used_ids = set()

    if min_examples_per_api_first:
        for api_name in api_names:
            if local_buckets[api_name] and len(chosen) < max_samples:
                s = local_buckets[api_name][0]
                sid = (s["file"], s["query"])
                if sid not in used_ids:
                    chosen.append(s)
                    used_ids.add(sid)

    round_idx = 1
    while len(chosen) < max_samples:
        added = False
        for api_name in api_names:
            bucket = local_buckets[api_name]
            if round_idx < len(bucket):
                s = bucket[round_idx]
                sid = (s["file"], s["query"])
                if sid not in used_ids:
                    chosen.append(s)
                    used_ids.add(sid)
                    added = True
                    if len(chosen) >= max_samples:
                        break
        if not added:
            break
        round_idx += 1
    return chosen[:max_samples]


def build_clean_subset(
    raw_samples: List[Dict[str, Any]],
    executor: MinimalAPIBankExecutor,
    settings: BenchmarkSettings,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, List[Dict[str, Any]]]]:
    filtered: List[Dict[str, Any]] = []
    reject_reasons: Counter = Counter()
    for sample in raw_samples:
        ok, reason = is_supported_sample(sample, executor, settings)
        if ok:
            filtered.append(sample)
        else:
            reject_reasons[reason] += 1

    api_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in filtered:
        api_buckets[sample["first_api_name"]].append(sample)

    subset = build_balanced_subset(
        api_buckets=api_buckets,
        max_samples=settings.max_samples,
        seed=settings.seed,
        min_examples_per_api_first=settings.min_examples_per_api_first,
    ) if settings.balanced_by_api else filtered[: settings.max_samples]

    return subset, dict(reject_reasons), api_buckets


def build_prompt(sample: Dict[str, Any], candidate_schemas: List[Dict[str, Any]], variant: str) -> Tuple[str, str]:
    user_block = (
        "User query:\n"
        + sample["query"]
        + "\n\nAvailable tools:\n"
        + json.dumps(candidate_schemas, ensure_ascii=False, indent=2)
        + "\n\nReturn exactly one tool call in the required format."
    )
    if variant == "json_only":
        return SYSTEM_PROMPT_JSON, user_block
    if variant == "xml_json":
        return SYSTEM_PROMPT_XML, user_block
    raise ValueError(f"Unknown variant: {variant}")


def try_extract_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def parse_tool_call(text: str) -> Dict[str, Any]:
    parsed = {
        "raw_output": text,
        "has_tool_call_tag": bool(re.search(r"<tool_call>.*?</tool_call>", text or "", re.DOTALL)),
        "json_candidate": None,
        "valid_json": False,
        "tool_name": None,
        "arguments": None,
        "parse_error": None,
        "no_call": False,
    }
    if text is None or text.strip() == "":
        parsed["no_call"] = True
        parsed["parse_error"] = "empty_output"
        return parsed
    cand = try_extract_json_object(text)
    parsed["json_candidate"] = cand
    if not cand:
        parsed["no_call"] = True
        parsed["parse_error"] = "no_json_object_found"
        return parsed
    try:
        obj = json.loads(cand)
        parsed["valid_json"] = True
        parsed["tool_name"] = obj.get("name")
        parsed["arguments"] = obj.get("arguments", {})
    except Exception as e:
        parsed["parse_error"] = f"invalid_json: {e}"
    return parsed


def normalize_scalar(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, str):
        return " ".join(x.strip().lower().split())
    if isinstance(x, float):
        if math.isnan(x):
            return None
        return round(x, 8)
    return x


def normalize_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: normalize_obj(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [normalize_obj(v) for v in obj]
    return normalize_scalar(obj)


def argument_exact_match(pred_args: Dict[str, Any], gold_args: Dict[str, Any]) -> bool:
    return normalize_obj(pred_args or {}) == normalize_obj(gold_args or {})


def api_name_exact_match(pred_name: str, gold_name: str) -> bool:
    return normalize_scalar(pred_name) == normalize_scalar(gold_name)


def estimate_cost_usd(usage: Dict[str, Any], model_spec: Dict[str, Any]) -> float:
    if not usage:
        return 0.0
    in_tok = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
    out_tok = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0
    return (in_tok / 1_000_000.0) * model_spec.get("input_cost_per_1m", 0.0) + (
        out_tok / 1_000_000.0
    ) * model_spec.get("output_cost_per_1m", 0.0)


def call_model(
    model_spec: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    settings: BenchmarkSettings,
) -> Dict[str, Any]:
    client = OpenAI(api_key=str(model_spec["api_key"]).strip(), base_url=str(model_spec["base_url"]).strip())
    started = time.time()
    timeout_seconds = model_spec.get("timeout_seconds", settings.global_timeout_seconds)
    max_tokens = model_spec.get("max_tokens", settings.global_max_tokens)

    try:
        if model_spec.get("api_mode", "chat") == "chat":
            resp = client.chat.completions.create(
                model=model_spec["model_name"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=settings.temperature,
                max_tokens=max_tokens,
                timeout=timeout_seconds,
            )
            text = resp.choices[0].message.content if resp.choices and resp.choices[0].message else ""
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) if getattr(resp, "usage", None) else 0,
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0) if getattr(resp, "usage", None) else 0,
                "total_tokens": getattr(resp.usage, "total_tokens", 0) if getattr(resp, "usage", None) else 0,
            }
        else:
            prompt = system_prompt + "\n\n" + user_prompt
            resp = client.completions.create(
                model=model_spec["model_name"],
                prompt=prompt,
                temperature=settings.temperature,
                max_tokens=max_tokens,
                timeout=timeout_seconds,
            )
            text = resp.choices[0].text if resp.choices else ""
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) if getattr(resp, "usage", None) else 0,
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0) if getattr(resp, "usage", None) else 0,
                "total_tokens": getattr(resp.usage, "total_tokens", 0) if getattr(resp, "usage", None) else 0,
            }
        latency = time.time() - started
        return {
            "success": True,
            "raw_text": (text or "").strip(),
            "usage": usage,
            "latency_sec": latency,
            "error": None,
            "error_type": None,
            "estimated_cost_usd": estimate_cost_usd(usage, model_spec),
        }
    except Exception as e:
        latency = time.time() - started
        err = repr(e)
        err_lower = err.lower()
        if "timeout" in err_lower:
            error_type = "timeout"
        elif "authentication" in err_lower or "401" in err_lower:
            error_type = "auth"
        elif "model does not exist" in err_lower or "400" in err_lower:
            error_type = "bad_model_or_request"
        else:
            error_type = "other"
        return {
            "success": False,
            "raw_text": "",
            "usage": {},
            "latency_sec": latency,
            "error": err,
            "error_type": error_type,
            "estimated_cost_usd": 0.0,
        }


def evaluate_one_sample(
    sample: Dict[str, Any],
    model_spec: Dict[str, Any],
    variant: str,
    schema_mode: str,
    tool_catalog: ToolCatalog,
    executor: MinimalAPIBankExecutor,
    settings: BenchmarkSettings,
) -> Dict[str, Any]:
    gold_call = sample["api_calls"][0]
    gold_name = gold_call["api_name"]
    gold_args = gold_call["param_dict"]
    gold_result = gold_call["result"]

    candidate_names, candidate_schemas = tool_catalog.schemas_for_query(sample["query"], schema_mode=schema_mode)
    gold_in_candidates = gold_name in candidate_names

    system_prompt, user_prompt = build_prompt(sample, candidate_schemas, variant=variant)
    model_out = call_model(model_spec, system_prompt, user_prompt, settings)
    parsed = parse_tool_call(model_out["raw_text"])

    api_match = api_name_exact_match(parsed["tool_name"], gold_name) if parsed["tool_name"] else False
    arg_match = argument_exact_match(parsed["arguments"], gold_args) if parsed["arguments"] is not None else False

    exec_success = False
    correctness: Any = False
    pred_exec = None
    if parsed["valid_json"] and parsed["tool_name"] and isinstance(parsed["arguments"], dict):
        pred_exec = executor.execute(parsed["tool_name"], parsed["arguments"])
        exec_success = bool(pred_exec.get("success"))
        if exec_success and api_match:
            correctness = executor.check_correctness(parsed["tool_name"], pred_exec["result"], gold_result)
            if correctness is None:
                correctness = api_match and arg_match
        else:
            correctness = False

    return {
        "file": sample["file"],
        "query": sample["query"],
        "gold_api_name": gold_name,
        "gold_arguments": gold_args,
        "gold_result": gold_result,
        "model_id": model_spec["model_id"],
        "model_label": model_spec["label"],
        "provider": model_spec["provider"],
        "prompt_variant": variant,
        "schema_mode": schema_mode,
        "candidate_tool_count": len(candidate_names),
        "candidate_tool_names": candidate_names,
        "gold_in_candidates": gold_in_candidates,
        "raw_output": model_out["raw_text"],
        "model_call_success": model_out["success"],
        "model_error": model_out["error"],
        "model_error_type": model_out["error_type"],
        "latency_sec": model_out["latency_sec"],
        "usage": model_out["usage"],
        "estimated_cost_usd": model_out["estimated_cost_usd"],
        "valid_json": parsed["valid_json"],
        "no_call": parsed["no_call"],
        "parse_error": parsed["parse_error"],
        "pred_api_name": parsed["tool_name"],
        "pred_arguments": parsed["arguments"],
        "api_name_exact_match": api_match,
        "argument_exact_match": arg_match,
        "execution_success": exec_success,
        "api_result_correct": bool(correctness),
        "has_tool_call_tag": parsed["has_tool_call_tag"],
        "json_candidate": parsed["json_candidate"],
        "pred_execution": pred_exec,
    }


def run_experiment(
    samples: List[Dict[str, Any]],
    model_specs: List[Dict[str, Any]],
    prompt_variants: Sequence[str],
    schema_modes: Sequence[str],
    run_name: str,
    paths: BenchmarkPaths,
    tool_catalog: ToolCatalog,
    executor: MinimalAPIBankExecutor,
    settings: BenchmarkSettings,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    active_models = [m for m in model_specs if m.get("enabled", True)]

    for model_spec in active_models:
        for schema_mode in schema_modes:
            for variant in prompt_variants:
                for sample in samples:
                    rows.append(
                        evaluate_one_sample(
                            sample=sample,
                            model_spec=model_spec,
                            variant=variant,
                            schema_mode=schema_mode,
                            tool_catalog=tool_catalog,
                            executor=executor,
                            settings=settings,
                        )
                    )

    df = pd.DataFrame(rows)
    group_cols = ["model_label", "provider", "schema_mode", "prompt_variant"]
    summary = (
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            n_samples=("model_label", "size"),
            gold_in_candidates_rate=("gold_in_candidates", "mean"),
            api_name_exact_match_rate=("api_name_exact_match", "mean"),
            argument_exact_match_rate=("argument_exact_match", "mean"),
            valid_json_rate=("valid_json", "mean"),
            no_call_rate=("no_call", "mean"),
            execution_success_rate=("execution_success", "mean"),
            api_result_correct_rate=("api_result_correct", "mean"),
            mean_latency_sec=("latency_sec", "mean"),
            median_latency_sec=("latency_sec", "median"),
            estimated_cost_per_sample_usd=("estimated_cost_usd", "mean"),
            model_call_failure_rate=("model_call_success", lambda s: (~s.fillna(False).astype(bool)).mean()),
            timeout_rate=("model_error_type", lambda s: s.fillna("").eq("timeout").mean()),
            mean_candidate_tool_count=("candidate_tool_count", "mean"),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )

    run_dir = paths.runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "settings": asdict(settings),
        "prompt_variants": list(prompt_variants),
        "schema_modes": list(schema_modes),
        "models": [{k: v for k, v in m.items() if k != "api_key"} for m in active_models],
    }

    df.to_csv(run_dir / "detailed_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(run_dir / "summary_results.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary_results.md").write_text(summary.to_markdown(index=False), encoding="utf-8")
    (run_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return df, summary


def sample_api_name(sample: Dict[str, Any]) -> Optional[str]:
    return sample.get("first_api_name") or sample.get("gold_api_name") or sample.get("api_name")


def sample_query(sample: Dict[str, Any]) -> Optional[str]:
    return sample.get("query") or sample.get("user_query")


def sample_file(sample: Dict[str, Any]) -> Optional[str]:
    return sample.get("file") or sample.get("source_file") or sample.get("file_name") or sample.get("path")


def save_subset_inventory(run_dir: Path, samples: Sequence[Dict[str, Any]]) -> None:
    inventory = pd.DataFrame(
        [
            {
                "gold_api_name": sample_api_name(s),
                "query": sample_query(s),
                "file": sample_file(s),
                "source_file": sample_file(s),
            }
            for s in samples
        ]
    )
    inventory.to_csv(run_dir / "paper_subset_inventory.csv", index=False, encoding="utf-8-sig")


def build_model_role_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    data = load_yaml(path)
    return {str(k): str(v) for k, v in (data or {}).items()}


def concat_run_summaries(runs_root: Path, run_names: Sequence[str]) -> pd.DataFrame:
    frames = []
    for run_name in run_names:
        path = runs_root / run_name / "summary_results.csv"
        if not path.exists():
            print(f"[MISS] {path}")
            continue
        df = pd.read_csv(path)
        df["run_name"] = run_name
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No summary_results.csv found for the requested runs.")
    return pd.concat(frames, ignore_index=True)


def apply_model_roles(df: pd.DataFrame, model_role_map: Dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    if "model_label" in df.columns:
        df["model_role"] = df["model_label"].map(model_role_map).fillna("other")
    return df


def aggregate_main_multiseed(
    runs_root: Path,
    run_names: Sequence[str],
    output_csv: Path,
    model_role_map: Dict[str, str],
) -> pd.DataFrame:
    summary = apply_model_roles(concat_run_summaries(runs_root, run_names), model_role_map)
    metric_cols = [
        c
        for c in [
            "gold_in_candidates_rate",
            "api_name_exact_match_rate",
            "argument_exact_match_rate",
            "valid_json_rate",
            "no_call_rate",
            "execution_success_rate",
            "api_result_correct_rate",
            "model_call_failure_rate",
            "timeout_rate",
            "mean_latency_sec",
        ]
        if c in summary.columns
    ]
    paper_main = summary.groupby(["model_label", "model_role"], dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    paper_main.columns = [
        "_".join([x for x in col if x]).rstrip("_") if isinstance(col, tuple) else col
        for col in paper_main.columns
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    paper_main.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return paper_main


def aggregate_ablation(
    runs_root: Path,
    run_names: Sequence[str],
    output_csv: Path,
    model_role_map: Dict[str, str],
) -> pd.DataFrame:
    summary = apply_model_roles(concat_run_summaries(runs_root, run_names), model_role_map)
    if "prompt_variant" not in summary.columns:
        summary["prompt_variant"] = summary["run_name"].apply(lambda x: "xml_json" if "xml_json" in x else "json_only")
    if "schema_mode" not in summary.columns:
        summary["schema_mode"] = summary["run_name"].apply(lambda x: "full" if x.endswith("__full") else "bm25_topk")

    keep_cols = [
        c
        for c in [
            "model_label",
            "model_role",
            "prompt_variant",
            "schema_mode",
            "gold_in_candidates_rate",
            "api_name_exact_match_rate",
            "argument_exact_match_rate",
            "valid_json_rate",
            "execution_success_rate",
            "api_result_correct_rate",
            "timeout_rate",
            "mean_latency_sec",
        ]
        if c in summary.columns
    ]
    paper_ablation = summary[keep_cols].sort_values(["model_label", "prompt_variant", "schema_mode"]).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    paper_ablation.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return paper_ablation


def aggregate_extended_main(
    runs_root: Path,
    run_names: Sequence[str],
    output_csv: Path,
    model_role_map: Dict[str, str],
) -> pd.DataFrame:
    summary = apply_model_roles(concat_run_summaries(runs_root, run_names), model_role_map)
    metric_cols = [
        c
        for c in [
            "gold_in_candidates_rate",
            "api_name_exact_match_rate",
            "argument_exact_match_rate",
            "valid_json_rate",
            "no_call_rate",
            "execution_success_rate",
            "api_result_correct_rate",
            "model_call_failure_rate",
            "timeout_rate",
            "mean_latency_sec",
        ]
        if c in summary.columns
    ]
    paper_main = summary.groupby(["model_label", "model_role"], dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    paper_main.columns = [
        "_".join([x for x in col if x]).rstrip("_") if isinstance(col, tuple) else col
        for col in paper_main.columns
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    paper_main.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return paper_main


def export_deployability_table(paper_main_extended_csv: Path, output_csv: Path) -> pd.DataFrame:
    paper_main_extended = pd.read_csv(paper_main_extended_csv)
    deployability_cols = [
        c
        for c in [
            "model_label",
            "model_role",
            "gold_in_candidates_rate_mean",
            "api_name_exact_match_rate_mean",
            "argument_exact_match_rate_mean",
            "valid_json_rate_mean",
            "execution_success_rate_mean",
            "api_result_correct_rate_mean",
            "timeout_rate_mean",
            "mean_latency_sec_mean",
        ]
        if c in paper_main_extended.columns
    ]
    deployability = paper_main_extended[deployability_cols].copy()
    sort_cols = [c for c in ["timeout_rate_mean", "execution_success_rate_mean", "valid_json_rate_mean", "mean_latency_sec_mean"] if c in deployability.columns]
    if sort_cols:
        ascending = [True, False, False, True][: len(sort_cols)]
        deployability = deployability.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    deployability.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return deployability


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def command_run(args: argparse.Namespace) -> None:
    settings = BenchmarkSettings(
        seed=args.seed,
        max_samples=args.max_samples,
        require_single_call=True,
        require_level1_filename=True,
        balanced_by_api=not args.no_balanced_by_api,
        min_examples_per_api_first=True,
        exclude_unsupported_tools=True,
        manual_exclude_tools=tuple(parse_csv_list(args.manual_exclude_tools)) if args.manual_exclude_tools else ("Translate",),
        tool_top_k=args.tool_top_k,
        bm25_query_lowercase=True,
        temperature=args.temperature,
        global_max_tokens=args.max_tokens,
        global_timeout_seconds=args.timeout_seconds,
    )

    paths = ensure_paths(Path(args.deepagent_root), Path(args.runs_root))
    models = load_yaml(Path(args.models_yaml))
    if not isinstance(models, list):
        raise ValueError("models_yaml must contain a list of model specs")

    executor = MinimalAPIBankExecutor(paths.apis_dir, paths.db_dir)
    tool_catalog = ToolCatalog(executor, settings)
    raw_samples = load_api_bank_samples(paths.level1_dir)
    subset, reject_reasons, api_buckets = build_clean_subset(raw_samples, executor, settings)

    print(f"Loaded raw samples: {len(raw_samples)}")
    print(f"Filtered clean subset: {len(subset)}")
    print(f"Reject reasons: {reject_reasons}")
    print(f"Unique supported APIs: {len(api_buckets)}")
    print(f"Loaded tools: {len(executor.tool_infos)}")
    if executor.failed_modules:
        print(f"Skipped tool modules: {len(executor.failed_modules)}")

    prompt_variants = parse_csv_list(args.prompt_variants)
    schema_modes = parse_csv_list(args.schema_modes)
    if not prompt_variants or not schema_modes:
        raise ValueError("At least one prompt variant and one schema mode are required")

    run_dir = paths.runs_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_subset_inventory(run_dir, subset)
    _, summary = run_experiment(
        samples=subset,
        model_specs=models,
        prompt_variants=prompt_variants,
        schema_modes=schema_modes,
        run_name=args.run_name,
        paths=paths,
        tool_catalog=tool_catalog,
        executor=executor,
        settings=settings,
    )
    print(summary.to_markdown(index=False))


def command_aggregate_main(args: argparse.Namespace) -> None:
    model_role_map = build_model_role_map(Path(args.model_roles_yaml) if args.model_roles_yaml else None)
    df = aggregate_main_multiseed(
        runs_root=Path(args.runs_root),
        run_names=parse_csv_list(args.run_names),
        output_csv=Path(args.output_csv),
        model_role_map=model_role_map,
    )
    print(df.to_markdown(index=False))


def command_aggregate_ablation(args: argparse.Namespace) -> None:
    model_role_map = build_model_role_map(Path(args.model_roles_yaml) if args.model_roles_yaml else None)
    df = aggregate_ablation(
        runs_root=Path(args.runs_root),
        run_names=parse_csv_list(args.run_names),
        output_csv=Path(args.output_csv),
        model_role_map=model_role_map,
    )
    print(df.to_markdown(index=False))


def command_aggregate_extended(args: argparse.Namespace) -> None:
    model_role_map = build_model_role_map(Path(args.model_roles_yaml) if args.model_roles_yaml else None)
    df = aggregate_extended_main(
        runs_root=Path(args.runs_root),
        run_names=parse_csv_list(args.run_names),
        output_csv=Path(args.output_csv),
        model_role_map=model_role_map,
    )
    print(df.to_markdown(index=False))


def command_export_deployability(args: argparse.Namespace) -> None:
    df = export_deployability_table(Path(args.paper_main_extended_csv), Path(args.output_csv))
    print(df.to_markdown(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Closed-set API-Bank benchmark for compact LLMs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_run = subparsers.add_parser("run", help="Run one benchmark configuration")
    p_run.add_argument("--deepagent-root", required=True, help="Path to DeepAgent root containing data/API-Bank")
    p_run.add_argument("--runs-root", default="./runs", help="Directory for benchmark outputs")
    p_run.add_argument("--models-yaml", required=True, help="YAML file containing model specifications")
    p_run.add_argument("--run-name", required=True, help="Name of the run output folder")
    p_run.add_argument("--prompt-variants", default="json_only", help="Comma-separated prompt variants")
    p_run.add_argument("--schema-modes", default="bm25_topk", help="Comma-separated schema exposure modes")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--max-samples", type=int, default=50)
    p_run.add_argument("--tool-top-k", type=int, default=12)
    p_run.add_argument("--temperature", type=float, default=0.0)
    p_run.add_argument("--max-tokens", type=int, default=256)
    p_run.add_argument("--timeout-seconds", type=int, default=45)
    p_run.add_argument("--manual-exclude-tools", default="Translate")
    p_run.add_argument("--no-balanced-by-api", action="store_true")
    p_run.set_defaults(func=command_run)

    p_main = subparsers.add_parser("aggregate-main", help="Aggregate multi-seed main runs")
    p_main.add_argument("--runs-root", required=True)
    p_main.add_argument("--run-names", required=True, help="Comma-separated run names")
    p_main.add_argument("--output-csv", required=True)
    p_main.add_argument("--model-roles-yaml", default=None)
    p_main.set_defaults(func=command_aggregate_main)

    p_ablation = subparsers.add_parser("aggregate-ablation", help="Aggregate ablation runs")
    p_ablation.add_argument("--runs-root", required=True)
    p_ablation.add_argument("--run-names", required=True, help="Comma-separated run names")
    p_ablation.add_argument("--output-csv", required=True)
    p_ablation.add_argument("--model-roles-yaml", default=None)
    p_ablation.set_defaults(func=command_aggregate_ablation)

    p_extended = subparsers.add_parser("aggregate-extended", help="Aggregate extended multi-model runs")
    p_extended.add_argument("--runs-root", required=True)
    p_extended.add_argument("--run-names", required=True, help="Comma-separated run names")
    p_extended.add_argument("--output-csv", required=True)
    p_extended.add_argument("--model-roles-yaml", default=None)
    p_extended.set_defaults(func=command_aggregate_extended)

    p_dep = subparsers.add_parser("export-deployability", help="Export deployability table from extended summary")
    p_dep.add_argument("--paper-main-extended-csv", required=True)
    p_dep.add_argument("--output-csv", required=True)
    p_dep.set_defaults(func=command_export_deployability)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
