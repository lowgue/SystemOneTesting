"""Serial, resumable comparison runner with frozen input and machine identity."""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import sys
import threading
import time

from PIL import Image
import psutil

from .adapters import REGISTRY, create_adapter, endpoint
from .core import (
    ROOT,
    fingerprint,
    finalize,
    load_json,
    resolve_image,
    sha256,
    validate_manifest,
    write_json,
)
from .protocols import describe, item_contract, output_diagnostics
from .runtime import resolve_device, resolve_thread_count


def environment():
    """Capture hardware, Python, and installed-package details for a reproducible run."""
    packages = {}
    package_names = [
        'torch',
        'torchvision',
        'transformers',
        'huggingface-hub',
        'pillow',
        'psutil',
        'numpy',
        'matplotlib',
    ]
    for package_name in package_names:
        try:
            packages[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            packages[package_name] = None

    cpu = platform.processor()
    cpu_info_path = Path('/proc/cpuinfo')
    if cpu_info_path.exists():
        cpu = next(
            (
                line.split(':', 1)[1].strip()
                for line in cpu_info_path.read_text().splitlines()
                if line.startswith('model name')
            ),
            cpu,
        )

    installed_packages = dict(
        sorted(
            (distribution.metadata['Name'], distribution.version)
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get('Name')
        )
    )
    return dict(
        hostname=platform.node(),
        platform=platform.platform(),
        python=platform.python_version(),
        cpu=cpu,
        logical_cpus=psutil.cpu_count(),
        physical_cpus=psutil.cpu_count(logical=False),
        ram_bytes=psutil.virtual_memory().total,
        packages=packages,
        installed_packages=installed_packages,
    )


def validate_config(config, selected=None):
    """Validate model IDs, adapter names, revisions, and credential configuration."""
    specs = config.get('models', [])
    model_ids = {spec['id'] for spec in specs}
    if not specs or len(model_ids) != len(specs):
        raise ValueError('Config needs models with unique IDs')

    if selected:
        unknown_ids = set(selected) - model_ids
        if unknown_ids:
            raise ValueError(f'Unknown model IDs: {sorted(unknown_ids)}')
        specs = [spec for spec in specs if spec['id'] in selected]

    for spec in specs:
        model_id = spec['id']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', model_id):
            raise ValueError('Model ID must be a filename-safe label')

        adapter = spec['adapter']
        if adapter not in REGISTRY and not adapter.startswith('python:'):
            raise ValueError(f'Unsupported adapter: {adapter}')

        credential_fields = ['token', 'api_key', 'password', 'authorization']
        if any(field in spec for field in credential_fields):
            raise ValueError(
                'Use token_env, never literal credentials in the configuration'
            )

        local_hub_adapters = {'blip', 'laya', 'd1_omni'}
        if adapter in local_hub_adapters and not re.fullmatch(
            r'[a-f0-9]{40}', spec.get('revision', '')
        ):
            raise ValueError('Pin local Hub models to a full commit SHA')

        token_env = spec.get('token_env')
        if token_env and not re.fullmatch(r'[A-Z_][A-Z0-9_]*', token_env):
            raise ValueError('token_env must name an environment variable')

        image_encoding = spec.get('image_encoding', 'data_url')
        if image_encoding not in {'data_url', 'base64'}:
            raise ValueError('image_encoding must be data_url or base64')

    return specs


def preflight(
    config,
    manifest,
    contract,
    data_root,
    selected=None,
    require_dependencies=True,
):
    """Check model configuration, dataset integrity, dependencies, endpoints, and known limits."""
    specs = validate_config(config, selected)
    document = validate_manifest(manifest, data_root, contract)
    issues = []
    current_environment = environment()

    for spec in specs:
        adapter = spec['adapter']
        choice_limit = spec.get('max_choices')
        if choice_limit:
            for domain in contract['domains'].values():
                for question_key, question in domain['questions'].items():
                    if (
                        question['type'] == 'choice'
                        and len(question['criteria']) > choice_limit
                    ):
                        issues.append(
                            f'{spec["id"]}: {question_key} exceeds {choice_limit} '
                            'choices; use the shared description_ollama_v1.json '
                            'contract for all compared models'
                        )

        local_hub_adapters = {'blip', 'laya', 'd1_omni'}
        if adapter in local_hub_adapters:
            if require_dependencies:
                required_packages = [
                    'torch',
                    'torchvision',
                    'transformers',
                    'huggingface-hub',
                ]
                for package_name in required_packages:
                    if not current_environment['packages'].get(package_name):
                        issues.append(
                            f'{spec["id"]}: missing {package_name}; install '
                            'requirements/cpu.txt or GPU environment'
                        )
            if adapter == 'laya' and not (
                ROOT / 'modelos/laya/laya_vision/laya/vlm.py'
            ).is_file():
                issues.append(
                    'Initialize modelos/laya/laya_vision with '
                    'git submodule update --init --recursive'
                )

        if adapter == 'systemone_http':
            try:
                endpoint(spec)
            except ValueError as exc:
                issues.append(f'{spec["id"]}: {exc}')
            token_env = spec.get('token_env')
            if token_env and not os.environ.get(token_env):
                issues.append(f'{spec["id"]}: set {token_env}')

    return specs, document, issues


class Resources:
    """Sample CPU time and resident memory for the client and optional local server processes."""
    def __init__(self, server_pid=None):
        """Initialize the object and store the configuration it needs."""
        self.processes = [psutil.Process()]
        if server_pid is not None:
            server_process = psutil.Process(int(server_pid))
            self.processes.extend(
                [server_process, *server_process.children(recursive=True)]
            )
        self.stop = threading.Event()
        self.peak = 0
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def cpu(self):
        """Return accumulated CPU time for the monitored processes."""
        total_cpu_seconds = 0
        for process in self.processes:
            try:
                times = process.cpu_times()
                total_cpu_seconds += times.user + times.system
            except psutil.NoSuchProcess:
                pass
        return total_cpu_seconds

    def rss(self):
        """Return the combined resident memory of the monitored processes."""
        total_rss_bytes = 0
        for process in self.processes:
            try:
                total_rss_bytes += process.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return total_rss_bytes

    def poll(self):
        """Sample the monitored processes until the stop signal is set."""
        while not self.stop.is_set():
            self.peak = max(self.peak, self.rss())
            self.stop.wait(.05)

    def __enter__(self):
        """Start the managed operation and return its active resource."""
        self.peak = self.rss()
        self.before = self.cpu()
        self.thread.start()
        return self

    def __exit__(self, *args):
        """Release resources and finalize the managed operation."""
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.rss())
        self.cpu_s = max(0, self.cpu() - self.before)


def server_metadata(spec):
    """Load and verify provenance for a locally managed HTTP inference server."""
    metadata_env = spec.get('server_metadata_env')
    metadata_path = os.environ.get(metadata_env, '') if metadata_env else ''
    if not metadata_path:
        return None

    metadata = load_json(metadata_path)
    if metadata.get('revision') != spec.get('revision'):
        raise ValueError('Clef server metadata revision differs from model config')

    from urllib.parse import urlsplit

    requested_endpoint = urlsplit(endpoint(spec))
    observed_endpoint = urlsplit(metadata.get('endpoint', ''))
    is_localhost = requested_endpoint.hostname in {'localhost', '127.0.0.1', '::1'}
    endpoint_matches = (
        requested_endpoint.port == observed_endpoint.port
        and requested_endpoint.path == observed_endpoint.path
    )
    if not is_localhost or not endpoint_matches:
        raise ValueError(
            'Local server resource metadata does not match the requested endpoint'
        )

    if metadata.get('pid'):
        process = psutil.Process(int(metadata['pid']))
        # Reject stale PID metadata after PID reuse.
        if abs(process.create_time() - metadata.get('process_created_s', 0)) > .5:
            raise ValueError('Server PID metadata is stale')
    return metadata


def _validate_run_arguments(args):
    """Reject invalid repetition, thread, and item-limit settings."""
    if args.repetitions < 1 or args.threads < 1:
        raise ValueError('threads and repetitions must be positive')
    if args.limit is not None and args.limit < 1:
        raise ValueError('limit must be positive')


def _items_for_run(manifest, limit):
    """Apply the optional item limit to the validated manifest."""
    return manifest['items'][:limit] if limit else manifest['items']


def _dry_run_result(specs, items, repetitions):
    """Summarize the planned requests without loading models or sending inference."""
    return dict(
        status='validated-no-inference',
        models=[spec['id'] for spec in specs],
        events=len(items),
        requests=len(items) * len(specs) * repetitions,
        domains=sorted({item['domain'] for item in items}),
        note=(
            'No weights loaded and no HTTP inference sent. Runtime protocol '
            'support is checked by an actual smoke run.'
        ),
    )


def _enable_offline_mode(offline):
    """Set offline model-cache flags when the run requests offline operation."""
    if offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def _code_fingerprints():
    """Hash project Python files so a resumed run can detect code changes."""
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted((ROOT / 'vision_bench').rglob('*.py'))
    }


def _frozen_configuration(args, specs, contract, manifest, items):
    """Freeze model, dataset, machine, protocol, and environment inputs for a run."""
    current_environment = environment()
    machine_id = args.machine_id or current_environment['hostname']
    code = _code_fingerprints()

    local_hub_adapters = {'blip', 'laya', 'd1_omni'}
    if args.device.startswith('cuda') and any(
        spec['adapter'] in local_hub_adapters for spec in specs
    ):
        import torch

        current_environment['cuda'] = dict(
            torch_runtime=torch.version.cuda,
            tf32_matmul=True,
            tf32_cudnn=True,
            devices=[
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        )

    external_metadata = {
        spec['id']: server_metadata(spec)
        for spec in specs
        if spec['adapter'] == 'systemone_http'
    }
    frozen = dict(
        schema_version=1,
        models=specs,
        contract=contract,
        manifest=manifest,
        item_ids=[item['id'] for item in items],
        repetitions=args.repetitions,
        threads=args.threads,
        device=args.device,
        max_edge=args.max_edge,
        offline=args.offline,
        seed=args.seed,
        machine_id=machine_id,
        environment=current_environment,
        code=code,
        resolved_http_endpoints={
            spec['id']: endpoint(spec)
            for spec in specs
            if spec['adapter'] == 'systemone_http'
        },
        external_provenance={
            model_id: {
                key: value
                for key, value in metadata.items()
                if key not in {'pid', 'process_created_s'}
            }
            if metadata
            else None
            for model_id, metadata in external_metadata.items()
        },
    )
    return machine_id, current_environment, frozen


def _create_output_directory(output_path, resume, frozen, identity):
    """Create a new run directory or verify that an existing run is safe to resume."""
    output_directory = Path(output_path).resolve()
    if output_directory.exists() and not resume:
        raise ValueError('Output exists; use a new run directory or --resume')
    if resume and not (output_directory / 'frozen_config.json').exists():
        raise ValueError('Cannot resume: frozen config missing')

    output_directory.mkdir(parents=True, exist_ok=True)
    if resume:
        existing_identity = fingerprint(
            load_json(output_directory / 'frozen_config.json')
        )
        if existing_identity != identity:
            raise ValueError(
                'Resume rejected: code, data, machine, dependencies or configuration changed'
            )
    else:
        write_json(output_directory / 'frozen_config.json', frozen)
    return output_directory


def _write_run_metadata(output_directory, identity, machine_id, environment_data,
                        contract, manifest, items, specs):
    """Write the immutable run identity, hardware snapshot, and dataset hashes."""
    dataset_fields = {
        'id',
        'domain',
        'sha256',
        'expected',
        'annotations',
        'group_id',
    }
    dataset_hash = fingerprint(
        [
            {key: value for key, value in item.items() if key in dataset_fields}
            for item in items
        ]
    )
    write_json(
        output_directory / 'run.json',
        dict(
            started_utc=datetime.now(timezone.utc).isoformat(),
            identity=identity,
            machine_id=machine_id,
            hardware=environment_data,
            contract_hash=fingerprint(contract),
            dataset_hash=dataset_hash,
            label_status=manifest.get(
                'label_status', 'exploratory_unadjudicated'
            ),
            resource_note=(
                'CPU/RSS scope is explicit per record; HTTP client resources are '
                'not server resources. Energy and monetary cost not measured.'
            ),
            simulated=any(spec['adapter'] == 'mock' for spec in specs),
        ),
    )


def _load_existing_results(output_directory):
    """Read completed request keys and error totals before resuming a run."""
    predictions_path = output_directory / 'predictions.jsonl'
    completed_requests = set()
    measured_errors = 0
    if predictions_path.exists():
        for line in predictions_path.read_text().splitlines():
            row = json.loads(line)
            request_key = row['request_key']
            if request_key in completed_requests:
                raise ValueError('Duplicate request key in existing results')
            completed_requests.add(request_key)
            measured_errors += bool(row.get('error'))
    return completed_requests, measured_errors


def _append_jsonl(output_directory, filename, row):
    """Append and flush one durable JSON record to a run artifact."""
    with (output_directory / filename).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _initialize_torch(specs, threads, seed, device):
    """Set local Torch thread count and random seed when a Torch adapter is used."""
    local_hub_adapters = {'laya', 'blip', 'd1_omni'}
    if any(spec['adapter'] in local_hub_adapters for spec in specs):
        import torch

        torch.set_num_threads(threads)
        torch.manual_seed(seed)
        if device.startswith('cuda'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


def _ordered_specs(specs, block):
    """Rotate and reverse adapter order to reduce order-related measurement bias."""
    rotation = block % len(specs)
    ordered = specs[rotation:] + specs[:rotation]
    if block % 2:
        ordered = list(reversed(ordered))
    return ordered


def _warm_up_adapter(adapter, spec, contract, data_root, items, max_edge):
    """Warm each adapter once per domain and record domains whose warm-up failed."""
    warmup_errors = []
    domains = sorted({item['domain'] for item in items})
    for domain in domains:
        item = next(item for item in items if item['domain'] == domain)
        try:
            image_path = resolve_image(data_root, item['rel_path'])
            with Image.open(image_path) as source:
                image = source.convert('RGB')
            adapter.contract = item_contract(contract, item)
            image.thumbnail((max_edge, max_edge))
            adapter.predict(image, domain)
            adapter.synchronize()
        except Exception:
            warmup_errors.append(domain)
            print(
                f'Warm-up failed for {spec["id"]}/{domain}; '
                'recording all measured attempts.',
                flush=True,
            )
    return warmup_errors


def _redact_error(error, specs):
    """Remove configured secret values from an error message before persisting it."""
    for spec in specs:
        token = os.environ.get(spec.get('token_env', ''), '')
        if token:
            error = error.replace(token, '[REDACTED]')
    return error


def _run_measured_request(
    adapter,
    spec,
    item,
    effective_contract,
    args,
    specs,
    block,
    resource_scope,
    server_pid,
):
    """Run one timed request and collect predictions, resource metrics, and diagnostics."""
    adapter.contract = effective_contract
    request_key = f'{block}/{spec["id"]}/{item["id"]}'
    uses_gpu = (
        args.device.startswith('cuda')
        and spec['adapter'] in {'blip', 'laya', 'd1_omni'}
    )
    if uses_gpu:
        sys.modules['torch'].cuda.reset_peak_memory_stats(args.device)

    adapter.synchronize()
    timer_start = time.perf_counter()
    with Resources(server_pid) as resources:
        try:
            image_path = resolve_image(args.data_root, item['rel_path'])
            with Image.open(image_path) as source:
                image = source.convert('RGB')
            image.thumbnail((args.max_edge, args.max_edge))
            answers, raw = adapter.predict(image, item['domain'])
            result = finalize(effective_contract, item['domain'], answers)
            adapter.synchronize()
            error = None
        except Exception as exc:
            raw = {}
            result = finalize(effective_contract, item['domain'], {})
            error = _redact_error(f'{type(exc).__name__}: {exc}', specs)

    row = dict(
        request_key=request_key,
        model=spec['id'],
        adapter=spec['adapter'],
        block=block,
        id=item['id'],
        domain=item['domain'],
        image_sha256=item['sha256'],
        elapsed_ms=(time.perf_counter() - timer_start) * 1000,
        cpu_s=resources.cpu_s,
        peak_rss_bytes=resources.peak,
        resource_scope=resource_scope,
        error=error,
        raw=raw,
        **result,
    )
    row['protocol_metrics'] = (
        dict(getattr(adapter, 'last_metrics', {})) if not error else {}
    )
    row['request'] = describe(spec, effective_contract, item['domain'])
    if spec['adapter'] in {'blip', 'laya', 'd1_omni'}:
        row['output_diagnostics'] = output_diagnostics(
            spec, effective_contract, item['domain'], raw
        )
    if uses_gpu:
        row['gpu_peak_allocated_bytes'] = (
            sys.modules['torch'].cuda.max_memory_allocated(args.device)
        )
    if item.get('expected') is not None:
        row.update(
            expected=item['expected'],
            exploratory_exact=(
                not error and result['predicted'] == item['expected']
            ),
        )
    return request_key, row, error


def _run_model_block(
    output_directory,
    args,
    specs,
    contract,
    items,
    block,
    ordered_specs,
    completed_requests,
    measured_errors,
    total_requests,
):
    """Load one adapter, warm it up, run its pending items, and persist progress."""
    for spec in ordered_specs:
        pending_items = [
            item
            for item in items
            if f'{block}/{spec["id"]}/{item["id"]}' not in completed_requests
        ]
        if not pending_items:
            continue

        metadata = (
            server_metadata(spec) if spec['adapter'] == 'systemone_http' else None
        )
        server_pid = metadata.get('pid') if metadata else None
        if server_pid:
            resource_scope = 'client_plus_local_server'
        elif spec['adapter'] == 'systemone_http':
            resource_scope = 'client_only'
        else:
            resource_scope = 'in_process_model'

        write_json(
            output_directory / 'status.json',
            dict(
                status='running',
                phase='warmup',
                model=spec['id'],
                completed=len(completed_requests),
                planned=total_requests,
                errors=measured_errors,
            ),
        )
        load_start = time.perf_counter()
        adapter = create_adapter(
            spec,
            contract,
            device=args.device,
            offline=args.offline,
        )
        try:
            adapter.synchronize()
            load_seconds = time.perf_counter() - load_start
            warmup_start = time.perf_counter()
            warmup_errors = _warm_up_adapter(
                adapter, spec, contract, args.data_root, items, args.max_edge
            )
            _append_jsonl(
                output_directory,
                'loads.jsonl',
                dict(
                    model=spec['id'],
                    block=block,
                    load_or_client_init_s=load_seconds,
                    warmup_s=time.perf_counter() - warmup_start,
                    warmup_failed_domains=warmup_errors,
                    server_metadata=metadata,
                    resource_scope=resource_scope,
                ),
            )

            random.Random(args.seed + block).shuffle(pending_items)
            for item in pending_items:
                effective_contract = item_contract(contract, item)
                request_key, row, error = _run_measured_request(
                    adapter,
                    spec,
                    item,
                    effective_contract,
                    args,
                    specs,
                    block,
                    resource_scope,
                    server_pid,
                )
                _append_jsonl(output_directory, 'predictions.jsonl', row)
                completed_requests.add(request_key)
                measured_errors += bool(error)
                write_json(
                    output_directory / 'status.json',
                    dict(
                        status='running',
                        phase='predict',
                        model=spec['id'],
                        current_filename=item.get('filename', item['id']),
                        completed=len(completed_requests),
                        planned=total_requests,
                        errors=measured_errors,
                    ),
                )
                error_text = f' ERROR: {error}' if error else ''
                print(
                    f'{len(completed_requests)}/{total_requests} '
                    f'{spec["id"]} {item["id"]}{error_text}',
                    flush=True,
                )
        finally:
            adapter.close()
            del adapter
            gc.collect()
            if 'torch' in sys.modules and args.device.startswith('cuda'):
                sys.modules['torch'].cuda.empty_cache()
    return measured_errors


def run(args):
    """Execute or resume a serial benchmark and persist its frozen inputs and results."""
    contract = load_json(args.contract)
    config = load_json(args.config)
    specs, manifest, issues = preflight(
        config, args.manifest, contract, args.data_root, args.models
    )
    if issues:
        raise ValueError('\n'.join(issues))

    local_torch_adapters = {'blip', 'laya', 'd1_omni'}
    has_local_torch_models = any(
        spec['adapter'] in local_torch_adapters for spec in specs
    )
    args.device = resolve_device(args.device, has_local_torch_models)
    args.threads = resolve_thread_count(args.threads)
    print(
        f"Runtime selecionado: dispositivo={args.device}, "
        f"threads PyTorch={args.threads}"
    )

    _validate_run_arguments(args)
    items = _items_for_run(manifest, args.limit)
    if args.dry_run:
        return _dry_run_result(specs, items, args.repetitions)

    _enable_offline_mode(args.offline)
    machine_id, current_environment, frozen = _frozen_configuration(
        args, specs, contract, manifest, items
    )
    identity = fingerprint(frozen)
    output_directory = _create_output_directory(
        args.output, args.resume, frozen, identity
    )
    if not args.resume:
        _write_run_metadata(
            output_directory,
            identity,
            machine_id,
            current_environment,
            contract,
            manifest,
            items,
            specs,
        )

    completed_requests, measured_errors = _load_existing_results(output_directory)
    _initialize_torch(specs, args.threads, args.seed, args.device)
    total_requests = len(items) * len(specs) * args.repetitions

    for block in range(args.repetitions):
        measured_errors = _run_model_block(
            output_directory,
            args,
            specs,
            contract,
            items,
            block,
            _ordered_specs(specs, block),
            completed_requests,
            measured_errors,
            total_requests,
        )

    write_json(
        output_directory / 'status.json',
        dict(
            status='complete',
            completed=len(completed_requests),
            planned=total_requests,
            finished_utc=datetime.now(timezone.utc).isoformat(),
        ),
    )
    return dict(
        status='complete',
        output=str(output_directory),
        completed=len(completed_requests),
    )
