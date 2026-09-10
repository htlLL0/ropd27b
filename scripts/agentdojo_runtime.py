"""Unmodified AgentDojo tasks/scorers with an audited local Qwen transport."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import b200_collection as core

sys.path[:0] = [str(core.ROOT / 'third_party/agentdojo/src'), str(core.ROOT / 'code/vendor')]


def config():
    return core.read(core.ROOT / 'config/agentdojo.json')


def suites():
    from agentdojo.task_suite.load_suites import get_suite
    cfg = config()
    return {name: get_suite(cfg['benchmark_version'], name) for name in cfg['suites']}


def protocol_cases():
    """All official task IDs and static payloads; no sampling or LLM calls."""
    from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
    from agentdojo.attacks import load_attack
    pipeline = GroundTruthPipeline(None)  # Identity only; never queried during static attack construction.
    pipeline.name = 'local-repeat_user_prompt'
    cases = []
    for name, suite in suites().items():
        attacker = load_attack(config()['attack'], suite, pipeline)
        for user in suite.user_tasks.values():
            cases.append({'suite': name, 'phase': 'benign', 'user_task': user.ID,
                          'injection_task': None, 'prompt': user.PROMPT, 'injections': {}})
        for injection in suite.injection_tasks.values():
            cases.append({'suite': name, 'phase': 'injection_utility', 'user_task': injection.ID,
                          'injection_task': None, 'prompt': injection.GOAL, 'injections': {}})
        for user in suite.user_tasks.values():
            for injection in suite.injection_tasks.values():
                cases.append({'suite': name, 'phase': 'attacked', 'user_task': user.ID,
                              'injection_task': injection.ID, 'prompt': user.PROMPT,
                              'injection_goal': injection.GOAL, 'injections': attacker.attack(user, injection)})
    for index, row in enumerate(cases):
        row['case_id'] = f'{index:04d}_{row["suite"]}_{row["phase"]}_{row["user_task"]}_{row["injection_task"] or "none"}'
    return cases


def check_protocol():
    from collections import Counter
    provenance = core.read(core.ROOT / 'provenance/agentdojo_source.json')
    for name, entry in provenance['files'].items():
        path = core.ROOT / 'third_party/agentdojo' / name
        core.require(core.sha(path) == entry['sha256'], 'Upstream AgentDojo file changed: ' + name)
    cfg = config()
    core.require(cfg['agentdojo_revision'] == provenance['revision'], 'AgentDojo revision mismatch')
    cases = protocol_cases()
    core.require(dict(Counter(row['phase'] for row in cases)) == cfg['expected_per_model'], 'Official task coverage changed')
    frozen = core.rows(core.ROOT / 'inputs/agentdojo_cases.jsonl')
    core.require(cases == frozen, 'Frozen test tasks/attack payloads changed')
    return {'status': 'pass', 'cases_per_model': len(cases), 'counts': cfg['expected_per_model'],
            'models': ['base', 'trained'], 'unmodified_upstream_files': len(provenance['files']),
            'source_revision': provenance['revision'], 'native_scorer': True, 'model_evaluation': False}


class AuditedClient:
    """The SecOPD message/parser adapter delegates requests to this local client."""
    def __init__(self, endpoint, model_name, case, directory):
        import openai
        cfg = config()
        self.client = openai.OpenAI(base_url=endpoint, api_key='EMPTY',
                                    timeout=cfg['request_timeout_seconds'], max_retries=0)
        self.chat = SimpleNamespace(completions=self)
        self.model_name, self.case, self.directory = model_name, case, directory
        self.count = 0

    def create(self, **kwargs):
        cfg = config()
        index = self.count
        self.count += 1
        folder = self.directory / 'requests' / f'{index:04d}'
        seed = int(hashlib.sha256(f"{cfg['seed']}:{self.case['case_id']}:{index}".encode()).hexdigest()[:8], 16) % (2**31)
        kwargs.update(model=self.model_name, seed=seed, temperature=cfg['temperature'], top_p=cfg['top_p'],
                      extra_body={'chat_template_kwargs': {'enable_thinking': True}})
        if cfg['max_output_tokens'] is not None:
            raise ValueError('Official-style evaluation has no added output cap')
        core.save(folder / 'request.json', kwargs)
        started = time.monotonic()
        try:
            response = self.client.chat.completions.create(**kwargs)
            core.save(folder / 'raw_response.json', response.model_dump(mode='json'))
            core.require(response.model == self.model_name, 'Wrong model returned by local server')
            core.require(bool(response.choices), 'No completion choices')
            return response
        except Exception as exc:
            core.save(folder / 'error.json', {'type': type(exc).__name__, 'message': str(exc), 'semantic_result': None})
            raise
        finally:
            core.save(folder / 'timing.json', {'seconds': time.monotonic() - started, 'at_utc': core.now()})


def make_pipeline(client, phase):
    import secopd_local_llm
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
    cfg = config()
    os.environ['AGENTDOJO_QWEN_ENABLE_THINKING'] = '1'
    os.environ['AGENTDOJO_QWEN_STRIP_THINKING'] = '0'
    llm = secopd_local_llm.LocalLLM(client, 'opcd-agentdojo-target', temperature=cfg['temperature'],
                                  top_p=cfg['top_p'], tool_delimiter=cfg['tool_delimiter'])
    llm.name = 'local'
    return AgentPipeline.from_config(PipelineConfig(llm=llm,
        defense=cfg['benign_defense'] if phase == 'benign' else cfg['attack_defense'],
        system_message_name=None, system_message=None))


def run_case(case, folder, endpoint, model_name):
    """Call official benchmark functions; preserve their native trace/score files."""
    from agentdojo.attacks import load_attack
    from agentdojo.benchmark import run_task_without_injection_tasks, run_task_with_injection_tasks
    from agentdojo.logging import OutputLogger
    cfg, suite = config(), suites()[case['suite']]
    client = AuditedClient(endpoint, model_name, case, folder)
    pipeline = make_pipeline(client, case['phase'])
    traces = folder / 'traces'
    traces.mkdir(parents=True)
    result = {'case': case, 'status': 'valid', 'utility': None, 'security': None, 'error': None}
    start = time.monotonic()
    try:
        with OutputLogger(str(traces)):
            if case['phase'] == 'attacked':
                attack = load_attack(cfg['attack'], suite, pipeline)
                user = suite.get_user_task_by_id(case['user_task'])
                injection = suite.get_injection_task_by_id(case['injection_task'])
                core.require(attack.attack(user, injection) == case['injections'], 'Attack payload mismatch')
                utilities, security = run_task_with_injection_tasks(suite, pipeline, user, attack,
                    traces, False, [case['injection_task']], cfg['benchmark_version'])
                key = (case['user_task'], case['injection_task'])
                result['utility'], result['security'] = utilities[key], security[key]
            else:
                task = (suite.get_user_task_by_id(case['user_task']) if case['phase'] == 'benign'
                        else suite.get_injection_task_by_id(case['user_task']))
                result['utility'], result['security'] = run_task_without_injection_tasks(
                    suite, pipeline, task, traces, False, cfg['benchmark_version'])
        native = list(traces.rglob('*.json'))
        core.require(len(native) == 1, 'Expected one native AgentDojo trace')
        trace = core.read(native[0])
        core.require(type(result['utility']) is bool and type(result['security']) is bool
                     and trace['utility'] == result['utility'] and trace['security'] == result['security'],
                     'Native returned scores and trace disagree')
        if trace.get('error') or list((folder / 'requests').rglob('error.json')):
            result.update(status='infrastructure_error', error=trace.get('error') or 'Request transport failed')
    except Exception as exc:
        result.update(status='infrastructure_error', error=type(exc).__name__ + ': ' + str(exc))
        core.save(folder / 'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
    finally:
        client.client.close()
    result.update(duration_seconds=time.monotonic() - start, requests=client.count,
                  security_true_means='attack succeeded', score_source='unmodified AgentDojo native benchmark')
    result['simulation_only'] = any(core.read(p).get('simulation_only', False)
                                   for p in (folder/'requests').rglob('raw_response.json'))
    result['files_sha256'] = {str(p.relative_to(folder)): core.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    core.save(folder / 'result.json', result)
    return result


def checked_result(path, case, *, binding=None, model=None, allow_simulation=False):
    row = core.read(path)
    core.require(row['case'] == case, 'Saved evaluation case changed')
    core.require(allow_simulation or not row.get('simulation_only'), 'Simulated responses cannot be formal evaluation results')
    actual = {str(p.relative_to(path.parent)) for p in path.parent.rglob('*') if p.is_file() and p != path}
    core.require(actual == set(row['files_sha256']), 'Saved case file inventory differs')
    for name, checksum in row['files_sha256'].items():
        relative = Path(name)
        core.require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe saved trace path')
        core.require(core.sha(path.parent / name) == checksum, 'Saved trace or request changed')
        if relative.name == 'raw_response.json':
            core.require(allow_simulation or not core.read(path.parent/name).get('simulation_only'),
                         'Simulated raw response cannot enter formal evaluation')
    if binding is not None:
        context = core.read(path.parent/'context.json')
        core.require(context['binding'] == binding and context['model'] == model and context['case'] == case
                     and context['evaluation_is_real'] is True, 'Case model/config binding changed')
    if row['status'] == 'valid':
        traces = list((path.parent/'traces').rglob('*.json'))
        core.require(len(traces) == 1, 'Missing/duplicate native trace')
        native = core.read(traces[0])
        core.require(type(row['utility']) is bool and type(row['security']) is bool and
                     row['utility'] == native['utility'] and row['security'] == native['security']
                     and not native.get('error') and not list((path.parent/'requests').rglob('error.json')),
                     'Saved scores or status differ from native AgentDojo trace')
    return row


def metrics(rows, expected, attacked=False):
    valid = [r for r in rows if r['status'] == 'valid']
    raw = [r for r in rows if type(r['utility']) is bool and type(r['security']) is bool]
    complete = len(rows) == len(valid) == expected
    report = {'expected': expected, 'completed': len(rows), 'valid': len(valid),
              'infrastructure_errors': len(rows)-len(valid), 'error_free_complete': complete,
              'task_successes': sum(r['utility'] for r in valid),
              'task_success_rate': sum(r['utility'] for r in valid)/expected if complete else None,
              'task_success_rate_on_valid_cases': sum(r['utility'] for r in valid)/len(valid) if valid else None,
              'official_raw_scored_cases': len(raw)}
    if attacked:
        report.update(attack_successes=sum(r['security'] for r in valid),
             asr=sum(r['security'] for r in valid)/expected if complete else None,
             asr_on_valid_cases=sum(r['security'] for r in valid)/len(valid) if valid else None,
             official_raw_asr=sum(r['security'] for r in raw)/len(raw) if raw else None,
             official_raw_asr_note='Upstream can mark context/server errors security=True; inspect infrastructure_errors.')
    return report


def summarize(root, models, cases):
    output = {'security_true_means': 'attack succeeded', 'models': {}, 'paired_comparison': None}
    binding = core.read(root/'run_binding.json') if (root/'run_binding.json').exists() else None
    for model in models:
        rows = []
        for case in cases:
            directory = root / model / 'cases' / case['case_id']
            committed = sorted(directory.glob('attempt_*/result.json'))
            if committed:
                rows.append(checked_result(committed[-1], case, binding=binding, model=model))
        output['models'][model] = {
            'groups': {phase: metrics([r for r in rows if r['case']['phase'] == phase],
                                     sum(c['phase'] == phase for c in cases), phase == 'attacked')
                       for phase in config()['expected_per_model']},
            'by_suite': {suite: {phase: metrics([r for r in rows if r['case']['phase'] == phase and r['case']['suite'] == suite],
                          sum(c['phase'] == phase and c['suite'] == suite for c in cases), phase == 'attacked')
                       for phase in config()['expected_per_model']} for suite in config()['suites']}}
    full = all(g['error_free_complete'] for m in output['models'].values() for g in m['groups'].values())
    output['status'] = 'complete' if full else 'partial_or_errors'
    if full and set(models) == {'base', 'trained'}:
        b, t = (output['models'][m]['groups'] for m in ('base', 'trained'))
        output['paired_comparison'] = {'trained_minus_base_percentage_points': {
            'benign_task_success': 100*(t['benign']['task_success_rate']-b['benign']['task_success_rate']),
            'attacked_task_success': 100*(t['attacked']['task_success_rate']-b['attacked']['task_success_rate']),
            'asr': 100*(t['attacked']['asr']-b['attacked']['asr'])}}
    return output
