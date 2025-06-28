conda deactivate
conda activate /Users/johyeonho/SaturdayDinner/.conda
cd ..
for i in {1..8}; do
    python3 main.py specs/spec$i.json
done