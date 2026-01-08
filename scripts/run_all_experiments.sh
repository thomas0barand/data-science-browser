#!/bin/bash
# Script pour lancer toutes les expériences et mesurer les temps

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "======================================"
echo "🚀 Data Science Browser - Experiments"
echo "======================================"

# Fonction pour mesurer le temps
run_timed() {
    local name=$1
    local cmd=$2
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶ $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    START=$(date +%s)
    eval $cmd
    END=$(date +%s)
    DURATION=$((END - START))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))
    echo "✓ Completed in ${MINUTES}m ${SECONDS}s"
}

# Fonction pour vérifier si un fichier existe et est récent
file_exists() {
    local file=$1
    if [ -f "$file" ]; then
        echo "✓ Already exists: $file (skipping)"
        return 0
    else
        return 1
    fi
}

# Option 1: TF-IDF corrigé
if [ "${RUN_TFIDF:-1}" = "1" ]; then
    echo ""
    echo "╔════════════════════════════════════╗"
    echo "║  Option 1: TF-IDF Corrigé          ║"
    echo "╚════════════════════════════════════╝"
    
    if ! file_exists "data/sparses_embedding_tfidf_corrected.pkl"; then
        run_timed "1.1 Export TF-IDF corrigé" \
            "python3 scripts/export_tfidf_corrected.py"
    fi
    
    if ! file_exists "outputs/results/preds_tfidf_corrected.csv"; then
        run_timed "1.2 Generate predictions (fast)" \
            "python3 scripts/generate_predictions_fast.py --method tfidf"
    fi
    
    if ! file_exists "outputs/results/metrics_preds_tfidf_corrected.json"; then
        run_timed "1.3 Compute metrics" \
            "poetry run python3 scripts/compute_metrics_single.py outputs/results/preds_tfidf_corrected.csv"
    fi
    
    echo ""
    echo "✓ TF-IDF corrigé terminé!"
fi

# Option 2: Bigram RAW
if [ "${RUN_BIGRAM_RAW:-1}" = "1" ]; then
    echo ""
    echo "╔════════════════════════════════════╗"
    echo "║  Option 2: Bigram RAW              ║"
    echo "╚════════════════════════════════════╝"
    
    if ! file_exists "data/sparses_embedding_bigram_raw.pkl"; then
        run_timed "2.1 Export Bigram RAW" \
            "python3 scripts/export_bigram_raw.py"
    fi
    
    if ! file_exists "data/sparses_embedding_bigram_raw_decimated.pkl"; then
        run_timed "2.2 Decimate Bigram RAW" \
            "python3 scripts/decimate_bigram_raw.py"
    fi
    
    if ! file_exists "outputs/results/preds_bigram_raw.csv"; then
        run_timed "2.3 Generate predictions (fast)" \
            "python3 scripts/generate_predictions_fast.py --method bigram_raw"
    fi
    
    if ! file_exists "outputs/results/metrics_preds_bigram_raw.json"; then
        run_timed "2.4 Compute metrics" \
            "poetry run python3 scripts/compute_metrics_single.py outputs/results/preds_bigram_raw.csv"
    fi
    
    echo ""
    echo "✓ Bigram RAW terminé!"
fi

echo ""
echo "╔════════════════════════════════════╗"
echo "║  📊 Résumé des résultats           ║"
echo "╚════════════════════════════════════╝"

if [ -f "outputs/results/metrics_preds_tfidf_corrected.json" ]; then
    echo ""
    echo "TF-IDF Corrigé:"
    poetry run python3 -c "
import json
with open('outputs/results/metrics_preds_tfidf_corrected.json') as f:
    m = json.load(f)
    print(f\"  Precision: {m['precision']:.4f}\")
    print(f\"  Recall:    {m['recall']:.4f}\")
    print(f\"  F1:        {m['f1']:.4f}\")
    print(f\"  AUC:       {m['auc']:.4f}\")
"
fi

if [ -f "outputs/results/metrics_preds_bigram_raw.json" ]; then
    echo ""
    echo "Bigram RAW:"
    poetry run python3 -c "
import json
with open('outputs/results/metrics_preds_bigram_raw.json') as f:
    m = json.load(f)
    print(f\"  Precision: {m['precision']:.4f}\")
    print(f\"  Recall:    {m['recall']:.4f}\")
    print(f\"  F1:        {m['f1']:.4f}\")
    print(f\"  AUC:       {m['auc']:.4f}\")
"
fi

echo ""
echo "✅ Toutes les expériences sont terminées!"

