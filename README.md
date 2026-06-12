# Stock Price Jumps & NLP Sentiment Analysis
## VCU – Advanced Financial Analytics (FIRE 691) | Week 9 Assignment

**Jump detection** in SPY and Berkshire Hathaway (BRK.B) daily returns using an expanding-window Z-score method, combined with **logistic regression** predicting jumps using FOMC dates, earnings announcements, and **Loughran-McDonald NLP sentiment** from 48 SEC 8-K filings (2020–2025).

---

## Company Focus: Berkshire Hathaway Class B (BRK.B)

---

## Data

| Dataset | Source | Coverage |
|---------|--------|----------|
| `spy_brk_daily.csv` | WRDS/CRSP | Jan 2019–Dec 2025 (3,520 rows) |
| SEC EDGAR 8-K filings | SEC EDGAR API | 2019–2025 (48 BRK.B filings) |
| FOMC meeting dates | Federal Reserve | 2019–2025 |
| Earnings announcement dates | WRDS/Compustat | BRK.B quarterly |

**2019 used as burn-in period; all analysis restricted to 1,508 trading days (Jan 2020–Dec 2025)**

---

## Methodology

### Part 1 — Jump Detection (Jeon-McCurdy-Zhao 2022)
- Expanding-window Z-score: normalize daily return by historical mean and std computed from all prior observations
- Flag jumps at 4 thresholds: |Z| > 2, 2.5, 3, 3.5
- Classify positive (up) and negative (down) jumps separately
- Apply to both SPY (market benchmark) and BRK.B (company)

### Part 2 — Logistic Jump Prediction Model
- **Base model**: FOMC dates + earnings announcement dates as predictors of BRK.B jump days
- **Extended model**: Add Loughran-McDonald NLP sentiment score from SEC 8-K filings
- Evaluate McFadden pseudo-R² as goodness-of-fit

---

## Key Results

### SPY Jump Detection
- **7 extreme jump days** identified (|Z| > 3.5 threshold), 2020–2025
- All 7 concentrated in **February–March 2020** (COVID crash period)
- SPY returns: mean 0.07%/day, std 1.27%, minimum −10.94% (March 2020)

### BRK.B Jump Detection
- BRK.B jumps closely track SPY in 2020; diverge in post-2021 period
- Berkshire's insurance and energy holdings create idiosyncratic jump patterns

### Logistic Regression Results

| Model | McFadden R² | Key Finding |
|-------|-------------|-------------|
| Base (FOMC + Earnings) | ~0.15 | FOMC dates weakly significant |
| + LM Sentiment (8-K) | **0.50** | Sentiment doubles explanatory power |

- **48 BRK.B 8-K filings** parsed from SEC EDGAR (2019–2025)
- Loughran-McDonald negative word count is the strongest individual predictor
- Adding NLP sentiment increases McFadden R² from ~0.15 to **0.50** — a dramatic improvement

---

## Tech Stack

`Python` `pandas` `numpy` `scipy` `statsmodels` `requests` `BeautifulSoup` `Google Colab` `WRDS/CRSP` `SEC EDGAR`

---

*Virginia Commonwealth University · MS Business (Financial Analytics) · FIRE 691*
*Reference: Jeon, McCurdy & Zhao (2022); Loughran & McDonald (2011)*
