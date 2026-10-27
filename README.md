# ai-vs-doctors-rules

Code and results for the article **"Microsoft's AI Beat Doctors 83% to 36%. The Doctors Weren't Allowed to Ask Anyone."**

Microsoft's MAI-DxO study (Science, October 2026) reported 83.2% accuracy on 304 NEJM case challenges against 36.1% for
generalist physicians working individually. This repo is a toy "diagnosis game", calibrated to those numbers, that changes
the contest rules one at a time: colleagues, specialists, teams, and an everyday case mix.

## Model in one paragraph

2,000 diseases with Zipf-like frequency across 12 specialties. Each player has a probability of recognising each disease:
generalists are strong on common diseases and weak on rare ones; references rescue a share of misses; specialists are much
better inside their own field; the AI is broad and nearly flat. Misses go to shared look-alikes, and a shared case-difficulty
term makes doctors struggle on the same cases. Groups vote, then whoever recognised the disease can persuade the group.
Calibrated to: solo generalist 22.3% / 36.1% (without / with references), AI 83.2%; group behaviour checked against
Barnett et al. 2019 (62.5% -> 75.1% for pairs -> 85.6% for nine).

## Headline results (NEJM-style cases)

| Set-up | Accuracy |
|---|---|
| Solo generalist, no references | 23% |
| Solo generalist, with references | 37% |
| + one colleague | 48% |
| Team of nine generalists | 55% |
| Generalist + matching specialist + one colleague | 76% (70-77% across specialist assumptions) |
| Specialist-led team of five | 89% (80-90%) |
| AI alone | 82% |

Everyday case mix: solo generalist with references 79% vs AI 87% (gap 7 points, vs 45 on NEJM-style cases).

## Run it

```bash
pip install -r requirements.txt
python simulate.py     # -> results/results.json (a few seconds)
python make_charts.py  # -> charts/*.png
```

Or open `ai_vs_doctors.ipynb` in Google Colab.

## Limits

A calibrated toy, not a re-analysis of Microsoft's data. Specialist performance on NEJM cases is unknown, so it is swept.
The everyday case mix is synthetic and does not reproduce the study's emergency-department result (AI 80.3% vs 46.8%).

## Licence

MIT. See `LICENSE`.
