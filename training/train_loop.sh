#!/bin/bash
# Self-driving WMNet training loop.
#
# Each round trains a candidate (fresh, or warm-started from the current
# best when the architecture matches), scores it on the held-out
# detail-preservation eval, and adopts it as models/wmnet.pt ONLY if it
# beats the incumbent.  So the production model can only ever improve, and
# real detail (chip prints, part numbers, connector edges) is the weighted
# objective.  Runs until improvements plateau, then stops.
set -u
cd /home/user/Devicepart-Watermark-Remover
MODELS="models/deviceparts_1500.npz,models/deviceparts_1000_tilted.npz"
BG="all_samples"
BEST="models/wmnet.pt"
LOG="scratch_lama_out/loop.log"
EVAL_N=128
mkdir -p models/candidates scratch_lama_out

score_of () {  # $1 = net path -> prints the scalar score (or empty on fail)
  python3 training/eval_wmnet.py --model "$MODELS" --backgrounds "$BG" \
    --net "$1" --n "$EVAL_N" 2>/dev/null \
    | python3 -c "import sys,json;print(json.load(sys.stdin)['score'])" 2>/dev/null
}
eval_json () {  # $1 = net path -> prints the full json line
  python3 training/eval_wmnet.py --model "$MODELS" --backgrounds "$BG" \
    --net "$1" --n "$EVAL_N" 2>/dev/null
}

# Back up the incumbent before the loop can ever overwrite it.
cp "$BEST" "models/wmnet_baseline.bak" 2>/dev/null
best_score=$(score_of "$BEST")
echo "$(date '+%F %T') START incumbent=$(eval_json "$BEST")" >> "$LOG"

# Round plan: base depth steps grad seed.  First a deep-arch exploration,
# then warm continuations that keep refining whatever arch is winning.
configs=(
  "32 4 8000  0.8 11"
  "32 4 10000 1.0 12"
  "40 4 9000  0.9 13"
  "32 4 12000 1.2 14"
  "40 4 11000 1.0 15"
  "48 4 10000 1.1 16"
  "40 5 12000 1.2 17"
  "32 4 14000 1.3 18"
)

no_improve=0
i=0
for cfg in "${configs[@]}"; do
  set -- $cfg; base=$1; depth=$2; steps=$3; grad=$4; seed=$5
  i=$((i+1))
  cand="models/candidates/wmnet_r${i}_b${base}d${depth}.pt"
  rm -f "$cand"
  echo "$(date '+%F %T') round $i TRAIN base=$base depth=$depth steps=$steps grad=$grad seed=$seed" >> "$LOG"
  # warm-start from the current best; trainer falls back to scratch on arch mismatch
  python3 training/train_wmnet.py --model "$MODELS" --backgrounds "$BG" \
    --out "$cand" --init "$BEST" --steps "$steps" --batch 8 \
    --base "$base" --depth "$depth" --grad-weight "$grad" --seed "$seed" \
    > "scratch_lama_out/train_r${i}.log" 2>&1
  if [ ! -f "$cand" ]; then
    echo "$(date '+%F %T') round $i NO checkpoint saved (under-converged)" >> "$LOG"
    no_improve=$((no_improve+1))
  else
    cj=$(eval_json "$cand"); s=$(echo "$cj" | python3 -c "import sys,json;print(json.load(sys.stdin)['score'])" 2>/dev/null)
    echo "$(date '+%F %T') round $i EVAL $cj" >> "$LOG"
    better=$(python3 -c "print(1 if ('$s' and float('$s')>float('$best_score')) else 0)" 2>/dev/null)
    if [ "$better" = "1" ]; then
      cp "$BEST" "models/wmnet_prev.bak"
      cp "$cand" "$BEST"
      best_score=$s
      no_improve=0
      echo "$(date '+%F %T') round $i ADOPTED new best score=$s" >> "$LOG"
      # Persist immediately so progress survives container reclamation.
      git add "$BEST" 2>/dev/null
      git commit -m "Training loop: adopt improved WMNet (round $i, score $s)

Detail-preservation eval on held-out synthetic pairs:
$cj

Co-Authored-By: Claude <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Exj22XxGEXRzMKUnMTu4SE" >> "$LOG" 2>&1
      for attempt in 1 2 3 4; do
        git push origin claude/watermark-removal-tool-kzy7ka >> "$LOG" 2>&1 && break
        sleep $((attempt * 2))
      done
    else
      no_improve=$((no_improve+1))
      echo "$(date '+%F %T') round $i kept incumbent (best=$best_score, no_improve=$no_improve)" >> "$LOG"
    fi
  fi
  rm -f "$cand"                      # candidates are large; keep only the best
  if [ "$no_improve" -ge 3 ]; then
    echo "$(date '+%F %T') PLATEAU: 3 rounds without improvement, stopping" >> "$LOG"
    break
  fi
done

echo "$(date '+%F %T') LOOP DONE best=$(eval_json "$BEST")" >> "$LOG"
