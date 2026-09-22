"""One WSI = downstream GradPool + streamed SR backwards + one optimizer step."""

import gc
import math
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from wsi_data import load_images


def autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=device.type == "cuda")


class MemoryMeter:
    """Keep iteration maxima while resetting CUDA peaks between measured phases."""

    def __init__(self, device, verbose):
        self.device = device
        self.verbose = verbose
        self.allocated = 0
        self.reserved = 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

    def record(self, stage):
        if self.device.type != "cuda":
            return
        torch.cuda.synchronize(self.device)
        allocated = torch.cuda.max_memory_allocated(self.device)
        reserved = torch.cuda.max_memory_reserved(self.device)
        self.allocated = max(self.allocated, allocated)
        self.reserved = max(self.reserved, reserved)
        if self.verbose:
            print(f"[memory] {stage}: phase peak allocated={allocated / 1024**3:.3f} GiB, "
                  f"reserved={reserved / 1024**3:.3f} GiB; "
                  f"current allocated={torch.cuda.memory_allocated(self.device) / 1024**3:.3f} GiB",
                  flush=True)
        torch.cuda.reset_peak_memory_stats(self.device)

    def summary(self):
        return {"max_memory_allocated_gb": self.allocated / 1024**3,
                "max_memory_reserved_gb": self.reserved / 1024**3}


def check_loss(loss, name):
    value = loss.item()
    if not math.isfinite(value):
        raise FloatingPointError(f"Non-finite {name}: {value}")
    return value


def wsi_predictions(logits, target):
    """Move WSI labels, Tumor probabilities and argmax classes off the graph/device."""
    labels = target.detach().cpu().tolist()
    probabilities = torch.softmax(logits.detach().float(), dim=1)[:, 1].cpu().tolist()
    predictions = torch.argmax(logits.detach(), dim=1).cpu().tolist()
    return labels, probabilities, predictions


def classification_metrics(labels, probabilities, predictions):
    """Shared sklearn definitions for train and validation WSI predictions."""
    if not labels or len(set(labels)) != 2:
        raise ValueError("Metrics require both Normal and Tumor WSI labels to report AUC")
    if not (len(labels) == len(probabilities) == len(predictions)):
        raise ValueError("Labels, probabilities and predictions must have equal lengths")
    return {
        "auc": roc_auc_score(labels, probabilities),
        "acc": accuracy_score(labels, predictions),
        "bacc": balanced_accuracy_score(labels, predictions),
    }


def binary_class_counts(labels, predictions):
    return {
        "gt_normal": labels.count(0), "gt_tumor": labels.count(1),
        "pred_normal": predictions.count(0), "pred_tumor": predictions.count(1),
    }


def _model_gradient_groups(model):
    groups = {"HAT": list(model.sr.encoder.parameters())}
    groups["ResNet18"] = list(model.region_encoder.parameters())
    groups["Gaussian"] = [parameter for name, parameter in model.sr.named_parameters()
                          if not name.startswith("encoder.")]
    groups.update({name: list(parameters)
                   for name, parameters in model.downstream_gradient_groups().items()})
    return groups


def _gradient_group_stats(parameters):
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    gradients = [parameter.grad.detach() for parameter in trainable
                 if parameter.grad is not None]
    norms = [gradient.norm(2).float() for gradient in gradients]
    total_norm = torch.stack(norms).norm(2).item() if norms else 0.0
    return {"norm": total_norm, "with_grad": len(gradients), "trainable": len(trainable)}


def _tensor_gradient_stats(gradient):
    return {"shape": list(gradient.shape), "norm": gradient.detach().float().norm().item(),
            "finite": bool(torch.isfinite(gradient).all().item())}


def _summarize_hat_gradients(hat_gradients, accumulated_gradient):
    branches, aggregate = {}, {}
    for name, values in hat_gradients.items():
        gradient_sum = values["sum"]
        aggregate[name] = gradient_sum
        branches[name] = {
            "calls": values["calls"], "l2_sum": values["l2_sum"],
            "aggregate_l2": (gradient_sum.norm().item() if gradient_sum is not None else 0.0),
            "finite": values["finite"],
        }
    classification, sr = aggregate["classification"], aggregate["sr"]
    cosine = None
    combined_l2 = 0.0
    consistency_max_abs = None
    if classification is not None and sr is not None:
        combined = classification + sr
        combined_l2 = combined.norm().item()
        denominator = classification.norm() * sr.norm()
        if denominator.item() > 0:
            cosine = (classification.flatten() @ sr.flatten() / denominator).item()
        if accumulated_gradient is not None:
            consistency_max_abs = (accumulated_gradient.detach().float() - combined).abs().max().item()
    return {"branches": branches, "cosine": cosine, "combined_l2": combined_l2,
            "consistency_max_abs": consistency_max_abs}


def _print_gradient_log(model, sample, total_norm, stats, cache_gradients,
                        hat_gradients, context):
    location = " ".join(f"{name}={value}" for name, value in context.items())
    group_text = " | ".join(
        f"{name}:l2={values['norm']:.6e},grad_tensors={values['with_grad']}/{values['trainable']}"
        for name, values in stats.items()
    )
    cache_text = " | ".join(
        f"{name}:shape={values['shape']},l2={values['norm']:.6e},finite={values['finite']}"
        for name, values in cache_gradients.items() if name != "mode"
    )
    branch_text = " | ".join(
        f"{name}:calls={values['calls']},per_backward_l2_sum={values['l2_sum']:.6e},"
        f"aggregate_l2={values['aggregate_l2']:.6e},finite={values['finite']}"
        for name, values in hat_gradients["branches"].items()
    )
    print(f"[gradient] {location} model={model.model_name} slide_id={sample['slide_id']} "
          f"N={sample['n_regions']} total_l2={total_norm:.6e}", flush=True)
    print(f"[gradient] groups {group_text}", flush=True)
    print(f"[gradient] cache_boundary mode={cache_gradients['mode']} {cache_text}", flush=True)
    print(f"[gradient] shared_HAT_conv_first {branch_text} | "
          f"cosine={hat_gradients['cosine']} combined_l2={hat_gradients['combined_l2']:.6e} "
          f"consistency_max_abs={hat_gradients['consistency_max_abs']}", flush=True)


def release_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def capture_rng_state(device):
    """Capture the generators that can affect a CPU/CUDA encoder forward."""
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state, device):
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device=device)


def capture_bn_state(module):
    """Snapshot only mutable BatchNorm buffers so a failed WSI can roll back."""
    state = []
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            for name in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(child, name)
                if value is not None:
                    state.append((value, value.detach().clone()))
    return state


def restore_bn_state(state):
    with torch.no_grad():
        for destination, value in state:
            destination.copy_(value)


def classification_loss_and_backward(model, sample, cls_micro_batch, device, target, verbose,
                                     gradient_diagnostics=None):
    """Gradient-cache any embedding-level downstream model one encoder graph at a time.

    Pass 1 encodes every micro-batch under no_grad, updating real BN buffers once
    and saving its RNG state. The detached [N,D] leaf receives a normal downstream
    backward through model.forward_embeddings(), which trains all model-specific
    aggregation/classification parameters and caches dL/dH. Pass 2 replays each
    encoder chunk with matched RNG and temporary BN buffers, injecting dL/dH.
    """
    n = sample["n_regions"]
    chunks, replay = [], []
    with torch.no_grad():
        for start in range(0, n, cls_micro_batch):
            end = min(start + cls_micro_batch, n)
            lr = load_images(sample["lr_paths"][start:end], 256).to(device)
            replay.append((start, end, capture_rng_state(device)))
            embedding = model.encode_regions(lr, preserve_bn_stats=False).float()
            chunks.append(embedding)
            del lr, embedding
        bag = torch.cat(chunks, dim=0)
        del chunks

    bag_leaf = bag.detach().requires_grad_(True)
    with autocast(device):
        logits = model.forward_embeddings(bag_leaf)
        loss_cls = F.cross_entropy(logits, target)
    cls_value = check_loss(loss_cls, "classification loss")
    labels, probabilities, predictions = wsi_predictions(logits, target)
    loss_cls.backward()
    if bag_leaf.grad is None or not torch.isfinite(bag_leaf.grad).all():
        raise RuntimeError("No finite classification gradient reached cached region embeddings")
    if gradient_diagnostics is not None:
        gradient_diagnostics.update(mode="gradpool",
                                    dL_dH=_tensor_gradient_stats(bag_leaf.grad))
    grad_bag = bag_leaf.grad.detach().clone()
    post_downstream_rng = capture_rng_state(device)
    del bag, bag_leaf, logits, loss_cls

    try:
        for start, end, rng_state in replay:
            lr = load_images(sample["lr_paths"][start:end], 256).to(device)
            restore_rng_state(rng_state, device)
            embedding = model.encode_regions(lr, preserve_bn_stats=True, log_shapes=verbose)
            torch.autograd.backward(embedding, grad_bag[start:end])
            del lr, embedding
            release_cuda(device)
    finally:
        restore_rng_state(post_downstream_rng, device)

    del grad_bag, replay
    if verbose:
        print("WSI classification: gradient caching over N={}; model={}; "
              "HAT HAB/OCAB checkpoint=on; outer encoder checkpoint=off".format(
                  n, model.model_name), flush=True)
    return cls_value, labels[0], probabilities[0], predictions[0]


def full_graph_classification_loss_and_backward(
        model, sample, cls_micro_batch, device, target, verbose, gradient_diagnostics=None):
    """Retain encoder graphs; communicate once after collecting the whole WSI."""
    n = sample["n_regions"]
    chunks = []
    for start in range(0, n, cls_micro_batch):
        end = min(start + cls_micro_batch, n)
        lr = load_images(sample["lr_paths"][start:end], 256).to(device)
        chunks.append(model.encode_regions(lr, log_shapes=verbose).float())
        del lr
    bag = torch.cat(chunks, dim=0)
    del chunks
    with autocast(device):
        logits = model.forward_embeddings(bag)
        loss_cls = F.cross_entropy(logits, target)
    cls_value = check_loss(loss_cls, "classification loss")
    labels, probabilities, predictions = wsi_predictions(logits, target)
    loss_cls.backward()
    if gradient_diagnostics is not None:
        gradient_diagnostics.update(mode="full_graph")
    if verbose:
        print("WSI classification: full graph over N={}; model={}; "
              "HAT HAB/OCAB checkpoint=on".format(n, model.model_name), flush=True)
    return cls_value, labels[0], probabilities[0], predictions[0]


def train_wsi(model, sample, optimizer, device, cls_micro_batch=1,
              sr_micro_batch=1, lambda_sr=1.0, verbose=False,
              gradient_log_context=None):
    """Never step between phases; every original region participates in both losses."""
    device = torch.device(device)
    n = sample["n_regions"]
    if n < 1 or len(sample["lr_paths"]) != n or len(sample["hr_paths"]) != n:
        raise ValueError("WSI must contain all N paired LR/HR paths")
    if cls_micro_batch < 1 or sr_micro_batch < 1:
        raise ValueError("Micro-batch sizes must be positive")
    if not math.isfinite(lambda_sr) or lambda_sr <= 0:
        raise ValueError("lambda_sr must be finite and positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    iteration_rng = capture_rng_state(device)
    bn_state = (capture_bn_state(model.region_encoder)
                if hasattr(model, "region_encoder") else [])
    meter = MemoryMeter(device, verbose)
    started = time.perf_counter()
    stage = "classification"
    branch = "classification"
    hat_gradients = {name: {"calls": 0, "l2_sum": 0.0, "finite": True, "sum": None}
                     for name in ("classification", "sr")}

    def record_hat_gradient(gradient):
        item = hat_gradients[branch]
        detached = gradient.detach().float()
        item["calls"] += 1
        item["l2_sum"] += detached.norm().item()
        item["finite"] &= torch.isfinite(detached).all().item()
        if item["sum"] is None:
            item["sum"] = detached.clone()
        else:
            item["sum"].add_(detached)

    handle = None
    stepped = False
    if gradient_log_context is not None and not isinstance(gradient_log_context, dict):
        raise TypeError("gradient_log_context must be a mapping or None")
    if verbose:
        print(f"slide_id={sample['slide_id']}; WSI label={sample['label']}; N regions={n}; "
              f"cls_micro_batch={cls_micro_batch}; sr_micro_batch={sr_micro_batch}; "
              f"model={model.model_name}", flush=True)
    if verbose or gradient_log_context is not None:
        handle = model.sr.encoder.conv_first.weight.register_hook(record_hat_gradient)
    try:
        gradient_diagnostics = {} if gradient_log_context is not None else None
        target = torch.tensor([sample["label"]], dtype=torch.long, device=device)
        classify = (classification_loss_and_backward
                    if getattr(model, "classification_gradpool", True)
                    else full_graph_classification_loss_and_backward)
        cls_value, label, probability, prediction = classify(
            model, sample, cls_micro_batch, device, target, verbose,
            gradient_diagnostics)
        meter.record("classification forward/backward")
        classification_seconds = time.perf_counter() - started
        release_cuda(device)

        branch = "sr"
        sr_started = time.perf_counter()
        sr_mean = 0.0
        for start in range(0, n, sr_micro_batch):
            end = min(start + sr_micro_batch, n)
            m = end - start
            stage = f"SR micro-batch {start // sr_micro_batch + 1}, regions {start + 1}-{end}/{n}"
            lr = load_images(sample["lr_paths"][start:end], 256).to(device)
            hr = load_images(sample["hr_paths"][start:end], 8192).to(device)
            output = model.forward_sr(lr, log_shapes=verbose)
            loss_sr = F.l1_loss(output, hr)
            sr_value = check_loss(loss_sr, stage + " loss")
            weighted_sr = lambda_sr * (m / n) * loss_sr
            if verbose:
                print(f"{stage}: HR={list(hr.shape)}; loss={sr_value:.8f}; "
                      f"weight=lambda_sr*{m}/{n}", flush=True)
            weighted_sr.backward()
            sr_mean += (m / n) * sr_value
            model.clear_sr_cache()
            del output, weighted_sr, loss_sr, lr, hr
            meter.record(stage + " forward/backward")
            release_cuda(device)
        sr_seconds = time.perf_counter() - sr_started

        stage = "gradient check / optimizer.step"
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"),
                                                   error_if_nonfinite=True).item()
        groups = (_model_gradient_groups(model)
                  if verbose or gradient_log_context is not None else None)
        gradient_stats = ({name: _gradient_group_stats(parameters)
                           for name, parameters in groups.items()}
                          if gradient_log_context is not None else None)
        hat_gradient_stats = (_summarize_hat_gradients(
            hat_gradients, model.sr.encoder.conv_first.weight.grad)
            if verbose or gradient_log_context is not None else None)
        for values in hat_gradients.values():
            values["sum"] = None
        if verbose:
            for name, parameters in groups.items():
                present = any(p.grad is not None for p in parameters)
                print(f"{name} gradient exists: {present}", flush=True)
                if not present:
                    raise RuntimeError(f"No gradient reached {name}")
            for name, values in hat_gradient_stats["branches"].items():
                print(f"{name} -> shared HAT incoming gradient: {values}", flush=True)
                if not values["calls"] or not values["finite"] or values["l2_sum"] <= 0:
                    raise RuntimeError(f"No finite nonzero {name} gradient reached shared HAT")
        optimizer.step()
        stepped = True
        if gradient_log_context is not None:
            _print_gradient_log(model, sample, grad_norm, gradient_stats,
                                gradient_diagnostics, hat_gradient_stats,
                                gradient_log_context)
        meter.record("optimizer.step (including newly allocated Adam state)")
        elapsed = time.perf_counter() - started
        result = {
            "model_name": model.model_name,
            "slide_id": sample["slide_id"], "n_regions": n,
            "label": label, "p_tumor": probability, "prediction": prediction,
            "loss_cls": cls_value, "loss_sr": sr_mean,
            "loss_total": cls_value + lambda_sr * sr_mean,
            "grad_norm": grad_norm, "seconds": elapsed,
            "classification_seconds": classification_seconds, "sr_seconds": sr_seconds,
            "regions_per_second": n / elapsed,
            **meter.summary(),
        }
        if verbose:
            print(f"WSI mean SR loss={sr_mean:.8f}; total loss={result['loss_total']:.8f}")
            print(f"optimizer.step succeeded (exactly once); WSI seconds={elapsed:.3f}; "
                  f"SR seconds={sr_seconds:.3f}; regions/second={n / elapsed:.4f}", flush=True)
        return result
    except Exception as exc:
        if not stepped:
            restore_bn_state(bn_state)
            restore_rng_state(iteration_rng, device)
        meter.record("failed " + stage)
        print(f"FAILED slide_id={sample['slide_id']} at {stage}: {type(exc).__name__}: {exc}",
              flush=True)
        print("No resolution reduction, region dropping, automatic micro-batch retry, or optimizer step after failure.",
              flush=True)
        raise
    finally:
        model.clear_sr_cache()
        if handle is not None:
            handle.remove()
        optimizer.zero_grad(set_to_none=True)
        release_cuda(device)
        if verbose:
            print(f"WSI iteration max_memory_allocated={meter.allocated / 1024**3:.3f} GiB; "
                  f"max_memory_reserved={meter.reserved / 1024**3:.3f} GiB", flush=True)


@torch.no_grad()
def evaluate(model, loader, device, cls_micro_batch=1):
    """Validation is WSI classification only: never read HR or invoke Gaussian SR."""
    device = torch.device(device)
    model.eval()
    labels, scores, decisions, predictions = [], [], [], []
    loss_sum = 0.0
    for sample in loader:
        n = sample["n_regions"]
        chunks = []
        for start in range(0, n, cls_micro_batch):
            lr = load_images(sample["lr_paths"][start:start + cls_micro_batch], 256).to(device)
            embedding = model.encode_regions(lr, preserve_bn_stats=False)
            chunks.append(embedding.float())
            del lr, embedding
        bag = torch.cat(chunks, dim=0)
        del chunks
        target = torch.tensor([sample["label"]], dtype=torch.long, device=device)
        with autocast(device):
            logits = model.forward_embeddings(bag)
            loss = F.cross_entropy(logits, target)
        loss_sum += check_loss(loss, "validation classification loss")
        batch_labels, batch_scores, batch_decisions = wsi_predictions(logits, target)
        labels.extend(batch_labels)
        scores.extend(batch_scores)
        decisions.extend(batch_decisions)
        score, decision = batch_scores[0], batch_decisions[0]
        predictions.append({"slide_id": sample["slide_id"], "label": sample["label"],
                            "p_tumor": score, "prediction": decision, "n_regions": n})
        del bag
        del logits, loss
    metrics = classification_metrics(labels, scores, decisions)
    return {**metrics, "loss": loss_sum / len(labels), "predictions": predictions}
