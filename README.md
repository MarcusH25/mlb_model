# MLB ML Model

Moneyline win-probability model for MLB games. Workflow inspired by AlphaPy's
sport pipeline, adapted for baseball: starting pitchers as first-class features,
longer rolling windows, moneyline target instead of spread.

## Setup
    conda activate mlb-ml
    pip install -r requirements.txt

## Structure
- config/     YAML configs (sport params + model params)
- data/       raw / processed / features (gitignored)
- src/        pipeline modules
- models/     saved artifacts
- output/     predictions and plots

## Pipeline
1. src/ingest.py        pulls games + pitcher starts via pybaseball
2. src/team_frame.py    rolling team features per season
3. src/pitcher_frame.py rolling pitcher features per season
4. src/game_frame.py    merges team + pitcher into one row per game
5. src/train.py         logreg + XGBoost with time-series CV
6. src/evaluate.py      log loss, Brier, ROI vs closing line