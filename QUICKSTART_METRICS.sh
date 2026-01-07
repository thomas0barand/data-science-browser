#!/bin/bash
# Quick script to generate all missing metrics

echo "════════════════════════════════════════════════════════════════"
echo "  GENERATING METRICS FOR ALL METHODS"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Step 1: Generate bigram predictions
echo "📊 Step 1/2: Generating bigram predictions..."
echo "   This may take 5-15 minutes depending on your data size"
echo ""
poetry run python scripts/generate_predictions.py --method bigram

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Error generating bigram predictions"
    echo "   Check that bigram embeddings exist in data/"
    exit 1
fi

echo ""
echo "✓ Bigram predictions generated"
echo ""

# Step 2: Compute metrics for all methods
echo "📈 Step 2/2: Computing metrics for all methods..."
echo "   This may take 5-10 minutes (finding optimal thresholds)"
echo ""
poetry run python scripts/compute_metrics.py --method all

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Error computing metrics"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✓ ALL METRICS COMPUTED SUCCESSFULLY!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📁 Results saved to: outputs/results/metrics_summary.json"
echo ""
echo "View comparison:"
echo "  cat outputs/results/metrics_summary.json"
echo ""

