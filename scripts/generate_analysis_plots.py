"""Generate all analysis plots from saved diagnostics .pt files.

Usage:
  python generate_analysis_plots.py \
    --default_pt path/to/diagnostics_default_T128_seed1.pt \
    --shs_pt path/to/diagnostics_shs_T128_seed1.pt \
    --output_dir path/to/output \
    [--eval_model gpt2-large] \
    [--num_annotation_sentences 10]
"""
import argparse
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm


# ============================================================
#  Analysis 1: Jump Count Histogram (Standard vs SHS overlaid)
# ============================================================
def analysis1_jump_count_histogram(d_std, d_shs, output_dir):
  jc_std = d_std['jump_counts'].numpy().flatten()
  jc_shs = d_shs['jump_counts'].numpy().flatten()
  max_val = max(int(jc_std.max()), int(jc_shs.max()))
  bins = np.arange(-0.5, max_val + 1.5, 1)

  fig, ax = plt.subplots(figsize=(12, 6))
  ax.hist(jc_std, bins=bins, alpha=0.55, color='coral', edgecolor='black',
          linewidth=0.5, label=f'Standard (mean={jc_std.mean():.2f})')
  ax.hist(jc_shs, bins=bins, alpha=0.55, color='steelblue', edgecolor='black',
          linewidth=0.5, label=f'SHS (mean={jc_shs.mean():.2f})')
  ax.set_xlabel('Jump Count (per position)', fontsize=13)
  ax.set_ylabel('Count (number of positions)', fontsize=13)
  steps = d_std.get('sampling_steps', '?')
  n_std = d_std['jump_counts'].shape[0]
  n_shs = d_shs['jump_counts'].shape[0]
  ax.set_title(
    f'Jump Count Distribution — Standard vs SHS\n'
    f'(steps={steps}, std_samples={n_std}, shs_samples={n_shs})',
    fontsize=14)
  ax.legend(fontsize=12)
  ax.set_xticks(np.arange(0, max_val + 1, max(1, max_val // 20)))
  plt.tight_layout()
  path = os.path.join(output_dir, 'analysis_jump_count_histogram.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f"[Analysis 1] Saved: {path}")


# ============================================================
#  Analysis 2: Cumulative Mass (S_i^tot) Distribution
# ============================================================
def analysis2_cumulative_mass(d_std, d_shs, output_dir):
  s_std = d_std['cumulative_mass'].numpy().flatten()
  s_shs = d_shs['cumulative_mass'].numpy().flatten()

  fig, ax = plt.subplots(figsize=(12, 6))
  bins = np.linspace(0, max(s_std.max(), s_shs.max()), 80)
  ax.hist(s_std, bins=bins, alpha=0.55, color='coral', edgecolor='black',
          linewidth=0.3, density=True,
          label=f'Standard (mean={s_std.mean():.2f}, std={s_std.std():.2f})')
  ax.hist(s_shs, bins=bins, alpha=0.55, color='steelblue', edgecolor='black',
          linewidth=0.3, density=True,
          label=f'SHS (mean={s_shs.mean():.2f}, std={s_shs.std():.2f})')
  ax.set_xlabel('$S_i^{tot}$ (Cumulative Mass per position)', fontsize=13)
  ax.set_ylabel('Density', fontsize=13)
  ax.set_title('Cumulative Mass Distribution — Standard vs SHS', fontsize=14)
  ax.legend(fontsize=12)
  plt.tight_layout()
  path = os.path.join(output_dir, 'analysis_cumulative_mass_histogram.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f"[Analysis 2] Saved: {path}")


# ============================================================
#  Analysis 3: Jump Count vs Token-level Quality
# ============================================================
def analysis3_jumpcount_vs_quality(d_std, output_dir, eval_model_name, tokenizer_name):
  """Box plot of token-level NLL grouped by jump count, plus scatter plot."""
  print("[Analysis 3] Computing token-level GPT2 NLL...")

  jc_all = d_std['jump_counts']       # (N, L)
  sm_all = d_std['cumulative_mass']    # (N, L)
  tid_all = d_std['token_ids']         # (N, L)
  sentences = d_std['samples']

  # Load UDLM tokenizer to identify PAD/MASK
  udlm_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
  pad_id = udlm_tokenizer.pad_token_id
  mask_id = getattr(udlm_tokenizer, 'mask_token_id', None)
  vocab_size = udlm_tokenizer.vocab_size

  # Load GPT2
  gpt2_tok = transformers.AutoTokenizer.from_pretrained(eval_model_name)
  if gpt2_tok.pad_token is None:
    gpt2_tok.pad_token = gpt2_tok.eos_token
  gpt2_model = transformers.AutoModelForCausalLM.from_pretrained(
    eval_model_name).eval().to('cuda')

  # Collect (jump_count, S_i_tot, nll) per valid position
  records = []  # list of (J_i, S_i, nll)
  N = min(len(sentences), jc_all.shape[0])
  for idx in tqdm(range(N), desc='Token NLL'):
    sentence = sentences[idx]
    jc = jc_all[idx]
    sm = sm_all[idx]
    tids = tid_all[idx]

    valid_pos = []
    for p in range(len(tids)):
      t = tids[p].item()
      if t == pad_id or (mask_id is not None and t == mask_id) or t == vocab_size:
        continue
      valid_pos.append(p)
    if not valid_pos:
      continue

    enc = gpt2_tok(sentence, return_tensors='pt', truncation=True, max_length=1024)
    gpt2_ids = enc['input_ids'].to('cuda')
    gpt2_len = int(enc['attention_mask'].sum().item())
    if gpt2_len <= 1:
      continue

    with torch.no_grad():
      logits = gpt2_model(gpt2_ids).logits
    nlls = F.cross_entropy(
      logits[0, :gpt2_len - 1], gpt2_ids[0, 1:gpt2_len],
      reduction='none').cpu().numpy()
    nll_len = len(nlls)
    if nll_len == 0:
      continue

    num_valid = len(valid_pos)
    for j_idx, p in enumerate(valid_pos):
      g_s = int(j_idx * nll_len / num_valid)
      g_e = max(int((j_idx + 1) * nll_len / num_valid), g_s + 1)
      g_s = min(g_s, nll_len - 1)
      g_e = min(g_e, nll_len)
      avg_nll = nlls[g_s:g_e].mean()
      records.append((int(jc[p].item()), float(sm[p].item()), float(avg_nll)))

  del gpt2_model
  torch.cuda.empty_cache()

  if not records:
    print("[Analysis 3] No data collected, skipping.")
    return

  arr = np.array(records)  # (M, 3): J_i, S_i, nll

  # --- 3a: Box plot grouped by J_i ---
  jump_vals = sorted(set(arr[:, 0].astype(int)))
  # Group: 0, 1, 2, 3, 4+
  groups = {}
  for jv in jump_vals:
    key = jv if jv < 4 else 4
    mask = arr[:, 0] == jv
    groups.setdefault(key, []).extend(arr[mask, 2].tolist())
  labels = sorted(groups.keys())
  label_names = [str(l) if l < 4 else '4+' for l in labels]
  box_data = [groups[l] for l in labels]

  fig, ax = plt.subplots(figsize=(10, 6))
  bp = ax.boxplot(box_data, labels=label_names, patch_artist=True, showfliers=False)
  colors_bp = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(labels)))
  for patch, c in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(c)
  ax.set_xlabel('Jump Count ($J_i$)', fontsize=13)
  ax.set_ylabel('Token-level NLL (GPT2-large)', fontsize=13)
  ax.set_title('Token Quality by Jump Count (Standard)', fontsize=14)
  # Add mean as diamond markers
  means = [np.mean(d) for d in box_data]
  ax.scatter(range(1, len(labels) + 1), means, marker='D', color='black',
             s=40, zorder=5, label='mean')
  ax.legend()
  plt.tight_layout()
  path = os.path.join(output_dir, 'analysis_jumpcount_vs_token_quality.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f"[Analysis 3a] Saved: {path}")

  # --- 3b: Scatter plot S_i^tot vs J_i, colored by NLL ---
  # Subsample for scatter if too many points
  max_pts = 50000
  if len(arr) > max_pts:
    idx = np.random.choice(len(arr), max_pts, replace=False)
    arr_sub = arr[idx]
  else:
    arr_sub = arr

  fig, ax = plt.subplots(figsize=(10, 8))
  vmin, vmax = np.percentile(arr_sub[:, 2], [5, 95])
  sc = ax.scatter(arr_sub[:, 1], arr_sub[:, 0], c=arr_sub[:, 2],
                  cmap='RdYlGn_r', alpha=0.4, s=3, vmin=vmin, vmax=vmax)
  ax.set_xlabel('$S_i^{tot}$ (Expected jump count)', fontsize=13)
  ax.set_ylabel('$J_i$ (Actual jump count)', fontsize=13)
  ax.set_title('Expected vs Actual Jumps, colored by Token NLL (Standard)', fontsize=14)
  # Diagonal line: J_i = round(S_i)
  max_s = arr_sub[:, 1].max()
  diag = np.arange(0, max_s + 1, 0.1)
  ax.plot(diag, np.round(diag), 'k--', alpha=0.5, label='$J_i = round(S_i^{tot})$')
  ax.legend(fontsize=11)
  plt.colorbar(sc, ax=ax, label='Token NLL')
  plt.tight_layout()
  path = os.path.join(output_dir, 'analysis_expected_vs_actual_jumps_quality.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f"[Analysis 3b] Saved: {path}")


# ============================================================
#  Helper: collect per-token NLL records for any diagnostics
# ============================================================
def _collect_token_nll_records(d, eval_model_name, tokenizer_name):
  """Returns numpy array (M, 3): [J_i, S_i^tot, NLL] per valid position."""
  jc_all = d['jump_counts']
  sm_all = d['cumulative_mass']
  tid_all = d['token_ids']
  sentences = d['samples']

  udlm_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
  pad_id = udlm_tokenizer.pad_token_id
  mask_id = getattr(udlm_tokenizer, 'mask_token_id', None)
  vocab_size = udlm_tokenizer.vocab_size

  gpt2_tok = transformers.AutoTokenizer.from_pretrained(eval_model_name)
  if gpt2_tok.pad_token is None:
    gpt2_tok.pad_token = gpt2_tok.eos_token
  gpt2_model = transformers.AutoModelForCausalLM.from_pretrained(
    eval_model_name).eval().to('cuda')

  records = []
  N = min(len(sentences), jc_all.shape[0])
  for idx in tqdm(range(N), desc='Token NLL'):
    sentence = sentences[idx]
    jc = jc_all[idx]
    sm = sm_all[idx]
    tids = tid_all[idx]

    valid_pos = []
    for p in range(len(tids)):
      t = tids[p].item()
      if t == pad_id or (mask_id is not None and t == mask_id) or t == vocab_size:
        continue
      valid_pos.append(p)
    if not valid_pos:
      continue

    enc = gpt2_tok(sentence, return_tensors='pt', truncation=True, max_length=1024)
    gpt2_ids = enc['input_ids'].to('cuda')
    gpt2_len = int(enc['attention_mask'].sum().item())
    if gpt2_len <= 1:
      continue

    with torch.no_grad():
      logits = gpt2_model(gpt2_ids).logits
    nlls = F.cross_entropy(
      logits[0, :gpt2_len - 1], gpt2_ids[0, 1:gpt2_len],
      reduction='none').cpu().numpy()
    nll_len = len(nlls)
    if nll_len == 0:
      continue

    num_valid = len(valid_pos)
    for j_idx, p in enumerate(valid_pos):
      g_s = int(j_idx * nll_len / num_valid)
      g_e = max(int((j_idx + 1) * nll_len / num_valid), g_s + 1)
      g_s = min(g_s, nll_len - 1)
      g_e = min(g_e, nll_len)
      avg_nll = nlls[g_s:g_e].mean()
      records.append((int(jc[p].item()), float(sm[p].item()), float(avg_nll)))

  del gpt2_model
  torch.cuda.empty_cache()
  return np.array(records) if records else None


# ============================================================
#  Analysis 6: Deviation-based analysis (core paper narrative)
# ============================================================
def analysis6_deviation(arr_std, arr_shs, output_dir):
  """
  deviation = J_i - round(S_i^tot)
  - 6a: Deviation distribution Standard vs SHS
  - 6b: Deviation vs NLL (the U-shape plot)
  """
  # Compute deviation for both
  dev_std = arr_std[:, 0] - np.round(arr_std[:, 1])
  dev_shs = arr_shs[:, 0] - np.round(arr_shs[:, 1])

  # --- 6a: Deviation distribution ---
  fig, ax = plt.subplots(figsize=(12, 6))
  min_d = int(min(dev_std.min(), dev_shs.min()))
  max_d = int(max(dev_std.max(), dev_shs.max()))
  bins = np.arange(min_d - 0.5, max_d + 1.5, 1)
  ax.hist(dev_std, bins=bins, alpha=0.55, color='coral', edgecolor='black',
          linewidth=0.5, density=True,
          label=f'Standard (std={dev_std.std():.2f})')
  ax.hist(dev_shs, bins=bins, alpha=0.55, color='steelblue', edgecolor='black',
          linewidth=0.5, density=True,
          label=f'SHS (std={dev_shs.std():.2f})')
  ax.set_xlabel('Deviation: $J_i - \\mathrm{round}(S_i^{tot})$', fontsize=13)
  ax.set_ylabel('Density', fontsize=13)
  ax.set_title('Edit Deviation Distribution — Standard vs SHS\n'
               '(SHS concentrates on {0, ±1}, Standard has long tails)', fontsize=14)
  ax.legend(fontsize=12)
  plt.tight_layout()
  path = os.path.join(output_dir, 'analysis_deviation_distribution.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f"[Analysis 6a] Saved: {path}")

  # --- 6b: Deviation vs NLL (Standard) --- the U-shape plot
  dev_vals_std = sorted(set(dev_std.astype(int)))
  dev_mean_nll = []
  dev_counts = []
  for dv in dev_vals_std:
    mask = (dev_std.astype(int) == dv)
    dev_mean_nll.append(np.exp(np.mean(arr_std[mask, 2])))
    dev_counts.append(mask.sum())

  fig, ax1 = plt.subplots(figsize=(12, 6))
  ax1.plot(dev_vals_std, dev_mean_nll, 'b-o', markersize=5, linewidth=2,
           label='Mean Token PPL', zorder=3)
  ax1.set_xlabel('Deviation: $J_i - \\mathrm{round}(S_i^{tot})$', fontsize=13)
  ax1.set_ylabel('Token-level PPL (GPT2-large)', fontsize=13, color='b')
  ax1.tick_params(axis='y', labelcolor='b')

  ax2 = ax1.twinx()
  ax2.bar(dev_vals_std, dev_counts, alpha=0.15, color='gray', zorder=1)
  ax2.set_ylabel('# tokens', fontsize=11, color='gray')
  ax2.tick_params(axis='y', labelcolor='gray')

  ax1.axvline(0, color='green', linestyle=':', alpha=0.5, label='No deviation')
  ax1.legend(fontsize=11, loc='upper left')
  ax1.set_title('Edit Deviation vs Token Quality (Standard)\n'
                'Under-edit (left) and Over-edit (right) both degrade quality',
                fontsize=14)
  plt.tight_layout()
  path = os.path.join(output_dir, 'analysis_deviation_vs_quality.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f"[Analysis 6b] Saved: {path}")


# ============================================================
#  Analysis 7: S_i vs J_i scatter — Standard vs SHS side-by-side
# ============================================================
def analysis7_scatter_comparison(arr_std, arr_shs, output_dir):
  """Side-by-side scatter of S_i^tot vs J_i for Standard and SHS."""
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

  max_pts = 50000
  for ax, arr, label in [(ax1, arr_std, 'Standard'), (ax2, arr_shs, 'SHS')]:
    if len(arr) > max_pts:
      idx = np.random.choice(len(arr), max_pts, replace=False)
      sub = arr[idx]
    else:
      sub = arr
    vmin, vmax = np.percentile(sub[:, 2], [5, 95])
    sc = ax.scatter(sub[:, 1], sub[:, 0], c=sub[:, 2],
                    cmap='RdYlGn_r', alpha=0.4, s=3, vmin=vmin, vmax=vmax)
    max_s = sub[:, 1].max()
    diag = np.arange(0, max_s + 1, 0.1)
    ax.plot(diag, np.round(diag), 'k--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('$S_i^{tot}$ (Expected jumps)', fontsize=12)
    ax.set_ylabel('$J_i$ (Actual jumps)', fontsize=12)
    ax.set_title(f'{label}', fontsize=14)
    plt.colorbar(sc, ax=ax, label='Token NLL', shrink=0.8)

  fig.suptitle('Expected vs Actual Jumps — Standard vs SHS\n'
               '(SHS concentrates on diagonal, Standard spreads)',
               fontsize=15, y=1.02)
  plt.tight_layout()
  path = os.path.join(output_dir, 'analysis_scatter_std_vs_shs.png')
  fig.savefig(path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f"[Analysis 7] Saved: {path}")


# ============================================================
#  Analysis 8: MAUVE score
# ============================================================
def analysis8_mauve(d_std, d_shs, output_dir, eval_model_name):
  """Compute MAUVE scores for Standard and SHS."""
  try:
    import mauve
  except ImportError:
    print("[Analysis 8] mauve-text not installed, skipping.")
    return

  # Use Standard samples as "reference" approximation is wrong;
  # we need actual reference text. Use a subset of the generated samples
  # cross-compared isn't right either.
  # MAUVE compares p (generated) vs q (reference).
  # We'll use GPT2-large generations as reference (standard practice).
  print("[Analysis 8] Computing MAUVE scores...")
  print("  (Using GPT2-large greedy generations as reference)")

  # Generate reference text from GPT2
  gpt2_tok = transformers.AutoTokenizer.from_pretrained(eval_model_name)
  if gpt2_tok.pad_token is None:
    gpt2_tok.pad_token = gpt2_tok.eos_token
  gpt2_model = transformers.AutoModelForCausalLM.from_pretrained(
    eval_model_name).eval().to('cuda')

  num_ref = min(500, len(d_std['samples']))
  ref_texts = []
  print(f"  Generating {num_ref} reference sentences from {eval_model_name}...")
  for _ in tqdm(range(num_ref), desc='Ref gen'):
    input_ids = gpt2_tok.encode(gpt2_tok.bos_token, return_tensors='pt').to('cuda')
    with torch.no_grad():
      out = gpt2_model.generate(
        input_ids, max_length=128, do_sample=True,
        top_k=50, top_p=0.95, num_return_sequences=1,
        pad_token_id=gpt2_tok.pad_token_id)
    ref_texts.append(gpt2_tok.decode(out[0], skip_special_tokens=True))
  del gpt2_model
  torch.cuda.empty_cache()

  results = {}
  for label, d in [('Standard', d_std), ('SHS', d_shs)]:
    gen_texts = d['samples'][:num_ref]
    out = mauve.compute_mauve(
      p_text=ref_texts, q_text=gen_texts,
      device_id=0, max_text_length=128, verbose=False)
    results[label] = out.mauve
    print(f"  MAUVE ({label}): {out.mauve:.4f}")

  # Save
  path = os.path.join(output_dir, 'analysis_mauve_scores.txt')
  with open(path, 'w') as f:
    for label, score in results.items():
      f.write(f"{label}: {score:.4f}\n")
  print(f"[Analysis 8] Saved: {path}")
  return results


# ============================================================
#  Analysis 4: Token Annotation HTML (Standard only)
# ============================================================
def analysis4_token_annotation_html(d_std, output_dir, tokenizer_name, num_sentences=10):
  udlm_tok = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
  pad_id = udlm_tok.pad_token_id
  mask_id = getattr(udlm_tok, 'mask_token_id', None)
  vocab_size = udlm_tok.vocab_size

  jc_all = d_std['jump_counts'].numpy()
  sm_all = d_std['cumulative_mass'].numpy()
  tid_all = d_std['token_ids'].numpy()

  html_parts = [
    '<!DOCTYPE html><html><head><meta charset="utf-8">',
    '<style>',
    'body{font-family:monospace;font-size:14px;line-height:1.8;padding:20px;}',
    '.under{background:#ff6b6b;color:white;padding:1px 3px;border-radius:3px;}',
    '.over{background:#4dabf7;color:white;padding:1px 3px;border-radius:3px;}',
    '.normal{padding:1px 1px;}',
    '.legend{margin-bottom:20px;padding:10px;background:#f8f9fa;border-radius:5px;}',
    '.sample{margin:15px 0;padding:10px;border:1px solid #dee2e6;border-radius:5px;}',
    '.meta{color:#868e96;font-size:12px;}',
    '</style></head><body>',
    '<h2>Under/Over-edit Token Annotation (Standard Sampling)</h2>',
    '<div class="legend">',
    '<span class="under">RED</span> = Under-edit: J_i=0 but S_i&ge;1.0 (should have jumped)<br>',
    '<span class="over">BLUE</span> = Over-edit: J_i &ge; ceil(S_i)+2 (jumped too many times)<br>',
    '<span class="normal">Normal</span> = Within expected range',
    '</div>',
  ]

  n = min(num_sentences, jc_all.shape[0])
  for idx in range(n):
    tids = tid_all[idx]
    jcs = jc_all[idx]
    sms = sm_all[idx]
    tokens_html = []
    for pos in range(len(tids)):
      tid = int(tids[pos])
      if tid == pad_id or (mask_id is not None and tid == mask_id) or tid == vocab_size:
        continue
      token_str = udlm_tok.decode([tid]).replace('<', '&lt;').replace('>', '&gt;')
      ji = int(jcs[pos])
      si = float(sms[pos])
      is_under = (ji == 0 and si >= 1.0)
      is_over = (ji >= math.ceil(si) + 2)
      title = f'J={ji}, S={si:.2f}'
      if is_under:
        tokens_html.append(f'<span class="under" title="{title}">{token_str}</span>')
      elif is_over:
        tokens_html.append(f'<span class="over" title="{title}">{token_str}</span>')
      else:
        tokens_html.append(f'<span class="normal" title="{title}">{token_str}</span>')

    mean_j = jcs[tids != pad_id].mean() if (tids != pad_id).any() else 0
    mean_s = sms[tids != pad_id].mean() if (tids != pad_id).any() else 0
    html_parts.append(
      f'<div class="sample">'
      f'<div class="meta">Sample {idx+1} | mean J={mean_j:.2f} | mean S={mean_s:.2f}</div>'
      f'{"".join(tokens_html)}</div>')

  html_parts.append('</body></html>')
  path = os.path.join(output_dir, 'analysis_token_annotation.html')
  with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(html_parts))
  print(f"[Analysis 4] Saved: {path}")


# ============================================================
#  Analysis 5: Temporal Tracking
# ============================================================
def analysis5_temporal(d_std, d_shs, output_dir, tokenizer_name):
  steps_label = d_std.get('sampling_steps', '?')

  # --- 5A: 2D Heatmap of p_ik (Standard vs SHS side-by-side) ---
  for label, d in [('Standard', d_std), ('SHS', d_shs)]:
    if 'detailed_p_ik' not in d:
      continue
    p_ik = d['detailed_p_ik']  # (N, steps, seq_len)
    # Average over detailed sentences
    avg_p = p_ik.numpy().mean(axis=0)  # (steps, seq_len)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(avg_p, aspect='auto', cmap='hot', interpolation='nearest',
                   origin='lower')
    ax.set_xlabel('Token Position', fontsize=13)
    ax.set_ylabel('Sampling Step', fontsize=13)
    ax.set_title(f'Jump Probability Heatmap — {label} (avg over {p_ik.shape[0]} sentences)',
                 fontsize=14)
    plt.colorbar(im, ax=ax, label='$p_{ik}$')
    plt.tight_layout()
    path = os.path.join(output_dir, f'analysis_2d_jump_probability_heatmap_{label.lower()}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Analysis 5A] Saved: {path}")

  # Combined heatmap
  if 'detailed_p_ik' in d_std and 'detailed_p_ik' in d_shs:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    p_std = d_std['detailed_p_ik'].numpy().mean(axis=0)
    p_shs = d_shs['detailed_p_ik'].numpy().mean(axis=0)
    vmax = max(p_std.max(), p_shs.max())
    for ax, p, label in [(ax1, p_std, 'Standard'), (ax2, p_shs, 'SHS')]:
      im = ax.imshow(p, aspect='auto', cmap='hot', interpolation='nearest',
                     origin='lower', vmin=0, vmax=vmax)
      ax.set_xlabel('Token Position')
      ax.set_ylabel('Sampling Step')
      ax.set_title(f'{label}')
    plt.colorbar(im, ax=[ax1, ax2], label='$p_{ik}$', shrink=0.8)
    fig.suptitle('Jump Probability Heatmap — Standard vs SHS', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, 'analysis_2d_jump_probability_heatmap.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Analysis 5A combined] Saved: {path}")

  # --- 5B: Jumps per step ---
  fig, ax = plt.subplots(figsize=(12, 5))
  if 'step_jump_counts' in d_std:
    steps_x = np.arange(len(d_std['step_jump_counts']))
    ax.plot(steps_x, d_std['step_jump_counts'], color='coral',
            label='Standard', linewidth=1.5)
  if 'step_jump_counts' in d_shs:
    steps_x = np.arange(len(d_shs['step_jump_counts']))
    ax.plot(steps_x, d_shs['step_jump_counts'], color='steelblue',
            label='SHS', linewidth=1.5)
  ax.set_xlabel('Sampling Step', fontsize=13)
  ax.set_ylabel('Avg. Jumps per Step', fontsize=13)
  ax.set_title(f'Number of Jumps per Sampling Step (T={steps_label})', fontsize=14)
  ax.legend(fontsize=12)
  plt.tight_layout()
  path = os.path.join(output_dir, 'analysis_jumps_per_step.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f"[Analysis 5B] Saved: {path}")

  # --- 5C: Cumulative edit ratio ---
  fig, ax = plt.subplots(figsize=(12, 5))
  if 'step_cumul_edit_ratio' in d_std:
    steps_x = np.arange(len(d_std['step_cumul_edit_ratio']))
    ax.plot(steps_x, d_std['step_cumul_edit_ratio'], color='coral',
            label='Standard', linewidth=1.5)
  if 'step_cumul_edit_ratio' in d_shs:
    steps_x = np.arange(len(d_shs['step_cumul_edit_ratio']))
    ax.plot(steps_x, d_shs['step_cumul_edit_ratio'], color='steelblue',
            label='SHS', linewidth=1.5)
  ax.set_xlabel('Sampling Step', fontsize=13)
  ax.set_ylabel('Fraction of positions edited at least once', fontsize=13)
  ax.set_title(f'Cumulative Edit Ratio over Steps (T={steps_label})', fontsize=14)
  ax.set_ylim(-0.02, 1.02)
  ax.legend(fontsize=12)
  plt.tight_layout()
  path = os.path.join(output_dir, 'analysis_cumulative_edit_ratio.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f"[Analysis 5C] Saved: {path}")

  # --- 5D: Temporal evolution HTML (Standard, 5 sentences) ---
  if 'detailed_snapshots' in d_std:
    _generate_temporal_html(d_std, output_dir, tokenizer_name, 'standard')
  if 'detailed_snapshots' in d_shs:
    _generate_temporal_html(d_shs, output_dir, tokenizer_name, 'shs')


def _generate_temporal_html(d, output_dir, tokenizer_name, mode_name):
  udlm_tok = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
  pad_id = udlm_tok.pad_token_id

  snaps = d['detailed_snapshots']  # (N, steps+1, seq_len)
  n_sentences = min(5, snaps.shape[0])
  n_steps = snaps.shape[1]
  # Show 5 checkpoints: 0, 25%, 50%, 75%, 100%
  checkpoints = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
  checkpoints = sorted(set(checkpoints))

  html = [
    '<!DOCTYPE html><html><head><meta charset="utf-8">',
    '<style>',
    'body{font-family:monospace;font-size:13px;padding:20px;}',
    '.changed{background:#ffd43b;padding:1px 2px;border-radius:2px;}',
    '.step{margin:5px 0;}',
    '.steplabel{color:#868e96;display:inline-block;width:80px;}',
    '.sentence{margin:20px 0;padding:10px;border:1px solid #ccc;border-radius:5px;}',
    '</style></head><body>',
    f'<h2>Temporal Evolution — {mode_name.upper()}</h2>',
    '<p>Yellow = token changed from previous checkpoint</p>',
  ]

  for si in range(n_sentences):
    html.append(f'<div class="sentence"><b>Sentence {si+1}</b>')
    prev_toks = None
    for ci, step_idx in enumerate(checkpoints):
      toks = snaps[si, step_idx].numpy()
      tokens_html = []
      for pos in range(len(toks)):
        tid = int(toks[pos])
        if tid == pad_id:
          continue
        tok_str = udlm_tok.decode([tid]).replace('<', '&lt;').replace('>', '&gt;')
        changed = prev_toks is not None and int(prev_toks[pos]) != tid
        if changed:
          tokens_html.append(f'<span class="changed">{tok_str}</span>')
        else:
          tokens_html.append(tok_str)
      pct = int(100 * step_idx / max(n_steps - 1, 1))
      html.append(
        f'<div class="step"><span class="steplabel">Step {step_idx} ({pct}%)</span>'
        f'{"".join(tokens_html)}</div>')
      prev_toks = toks
    html.append('</div>')

  html.append('</body></html>')
  path = os.path.join(output_dir, f'analysis_temporal_evolution_{mode_name}.html')
  with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(html))
  print(f"[Analysis 5D] Saved: {path}")


# ============================================================
#  Main
# ============================================================
def main():
  parser = argparse.ArgumentParser(description='Generate analysis plots from diagnostics .pt files')
  parser.add_argument('--default_pt', required=True, help='Path to Standard diagnostics .pt')
  parser.add_argument('--shs_pt', required=True, help='Path to SHS diagnostics .pt')
  parser.add_argument('--output_dir', required=True, help='Output directory for plots')
  parser.add_argument('--eval_model', default='gpt2-large', help='GPT2 model for token-level PPL')
  parser.add_argument('--tokenizer', default='gpt2',
                      help='Tokenizer name/path (gpt2 for FS-DFM, bert-base-uncased for UDLM)')
  parser.add_argument('--num_annotation_sentences', type=int, default=10)
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)

  print(f"Loading Standard diagnostics: {args.default_pt}")
  d_std = torch.load(args.default_pt, map_location='cpu', weights_only=False)
  print(f"Loading SHS diagnostics: {args.shs_pt}")
  d_shs = torch.load(args.shs_pt, map_location='cpu', weights_only=False)

  print(f"\n{'='*60}")
  print(f"Standard: {d_std['jump_counts'].shape[0]} samples, "
        f"SHS: {d_shs['jump_counts'].shape[0]} samples")
  print(f"{'='*60}\n")

  # Analysis 1: Jump Count Histogram
  print("--- Analysis 1: Jump Count Histogram ---")
  analysis1_jump_count_histogram(d_std, d_shs, args.output_dir)

  # Analysis 2: Cumulative Mass Distribution
  print("\n--- Analysis 2: Cumulative Mass Distribution ---")
  analysis2_cumulative_mass(d_std, d_shs, args.output_dir)

  # Analysis 3: Jump Count vs Token Quality (Standard only)
  print("\n--- Analysis 3: Jump Count vs Token Quality ---")
  analysis3_jumpcount_vs_quality(d_std, args.output_dir, args.eval_model, args.tokenizer)

  # Analysis 4: Token Annotation HTML (Standard only)
  print("\n--- Analysis 4: Token Annotation HTML ---")
  analysis4_token_annotation_html(d_std, args.output_dir, args.tokenizer,
                                   args.num_annotation_sentences)

  # Analysis 5: Temporal Tracking
  print("\n--- Analysis 5: Temporal Tracking ---")
  analysis5_temporal(d_std, d_shs, args.output_dir, args.tokenizer)

  # Analysis 6-8: Need token-level NLL for both Standard and SHS
  print("\n--- Collecting token-level NLL (Standard) ---")
  arr_std = _collect_token_nll_records(d_std, args.eval_model, args.tokenizer)
  print("\n--- Collecting token-level NLL (SHS) ---")
  arr_shs = _collect_token_nll_records(d_shs, args.eval_model, args.tokenizer)

  if arr_std is not None and arr_shs is not None:
    # Analysis 6: Deviation-based analysis
    print("\n--- Analysis 6: Deviation Analysis ---")
    analysis6_deviation(arr_std, arr_shs, args.output_dir)

    # Analysis 7: Scatter comparison
    print("\n--- Analysis 7: Scatter Standard vs SHS ---")
    analysis7_scatter_comparison(arr_std, arr_shs, args.output_dir)

  # Analysis 8: MAUVE score
  print("\n--- Analysis 8: MAUVE Score ---")
  analysis8_mauve(d_std, d_shs, args.output_dir, args.eval_model)

  print(f"\n{'='*60}")
  print(f"All analyses complete! Outputs in: {args.output_dir}")
  print(f"{'='*60}")


if __name__ == '__main__':
  main()
