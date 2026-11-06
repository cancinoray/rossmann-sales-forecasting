# Dataset shortlist — sales, logistics, and operations

> **Selected: Rossmann Store Sales.** The project is built in this standalone `rossmann-sales-forecasting/` repository; see its `README.md`. This file is kept as the decision record and the comparison that led to Rossmann.

**Status:** 7 November 2026
**Purpose:** pick a midterm dataset that is public, reproducible, not on the course ban list, and rich enough to score the full 16-point rubric.
**How this was built:** each file below was downloaded and profiled locally. Row counts, targets, date ranges, missingness, and quick baselines are measured, not estimated.

---

## 0. Comparison

| Dataset | Domain | Task | Size | Target | Access | 10-day fit |
|---|---|---|---|---|---|---|
| **DataCo Smart Supply Chain** | Logistics / supply chain | Binary classification | 180,519 × 53 | `Late_delivery_risk` (54.8% late) | [Mendeley DOI](https://data.mendeley.com/datasets/8gx2fvg2k6/5), CC BY 4.0 | **Best overall** |
| **Rossmann Store Sales** | Retail sales | Regression (daily sales) | 1,017,209 × 9 + store metadata | `Sales` | [Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales), HF mirror | **Best pure sales** |
| **Online Retail II** | Retail transactions | No ready target; forecasting / RFM | 1,067,371 × 8 | none (derive) | [UCI 502](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | Medium |
| **Olist Brazilian E-commerce** | E-commerce sales + logistics | Classification / regression | 99,441 orders, 9 tables | late delivery (8.1%), review score | [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), CC BY-NC-SA | Rich but heavy |
| **Bike Sharing** | Demand / operations | Regression | 17,379 hourly + 731 daily | `cnt` | [UCI 275](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) | Safest / fastest |
| **Walmart Store Sales** | Retail sales | Regression (weekly sales) | 421,570 × 5 + features | `Weekly_Sales` | [Kaggle](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting) | Good smaller Rossmann |

**One-line recommendation:** if you want **logistics**, build the DataCo late-delivery risk classifier. If you want **sales**, build Rossmann sales forecasting. If you want the lowest-risk path, use DataCo or Bike Sharing.

---

## 1. DataCo Smart Supply Chain — top pick for logistics

### 1.1 Verified profile

| Check | Result |
|---|---|
| Shape | **180,519 rows × 53 columns** |
| Date range | **2015-01-01 → 2018-01-31** |
| Target | `Late_delivery_risk`: **98,977 late (54.8%)** / 81,542 on time (45.2%) |
| Duplicates | **0** |
| Missing | `Product_Description` 100%; `Order_Zipcode` 86.2%; `Customer_Lname` 8 rows; `Customer_Zipcode` 3 rows |
| Sales range | 9.99 → 1,999.99 (mean 203.77) |
| Shipping modes | Standard Class 107,752; Second Class 35,216; First Class 27,814; Same Day 9,737 |
| Late rate by mode | **First Class 95.3%**, Second Class 76.6%, Same Day 45.7%, Standard Class 38.1% |
| Late rate by month | 51.9% → 56.8% (weak seasonality) |
| Late rate by market | Europe 55.2%, Pacific Asia 55.0%, USCA 54.8%, Africa 54.6%, LATAM 54.4% |
| Late rate by category | ~54.5–55.0% across the top 8 categories (weak) |

Source: [DataCo SMART SUPPLY CHAIN](https://data.mendeley.com/datasets/8gx2fvg2k6/5), DOI `10.17632/8gx2fvg2k6.5`, CC BY 4.0. A 53-column, 180,519-row mirror is also available on [Hugging Face](https://huggingface.co/datasets/alalfi/SupplyChainDataset) if Mendeley's download is slow.

### 1.2 Why it works for this midterm

- **Single table, ready target, balanced classes.** No joins, no label engineering, no 0.2% positive class.
- **The action is concrete.** At order time, flag orders likely to miss their promised window; operations can upgrade shipping, reserve inventory, or set customer expectations.
- **It has real leakage to remove.** `Days_for_shipping_(real)`, `Delivery_Status`, `shipping_date`, and `Order_Status` are post-order. Dropping them is the difference between a real model and a crystal ball.
- **The EDA has a genuine insight.** First Class is 95.3% “late” while Standard is 38.1%, because lateness is measured against the promise. The business lesson is that the **promised window**, not order value or category, drives risk. That is a stronger story than another churn model.

### 1.3 Project framing

> **Question:** at the moment an order is placed, what is the probability it will miss its promised delivery window, and should we expedite or re-promise it?
>
> **Unit:** one order line.
> **Target:** `Late_delivery_risk` (1 = late).
> **Prediction time:** order placement. Everything after the order is excluded.
> **Output:** `late_probability`, `risk_band`, `recommended_action`, `expected_cost`.

**Drop as leakage / post-order**
- `Days_for_shipping_(real)` — actual shipping days
- `Delivery_Status` — derived from the target
- `shipping_date_(DateOrders)` — shipment happens after the order
- `Order_Status` — final status; `CANCELED` and `SUSPECTED_FRAUD` have 0% late
- `Late_delivery_risk` — target

**Drop as PII / noise**
- `Customer_Email`, `Customer_Fname`, `Customer_Lname`, `Customer_Password`, `Customer_Street`, `Customer_Zipcode`
- `Order_Zipcode` (86% missing), `Product_Image`, `Product_Description` (100% missing)
- `Customer_Id`, `Order_Customer_Id`, `Order_Id`, `Order_Item_Id` (IDs, high cardinality)

**Borderline — drop for the clean version**
- `Benefit_per_order`, `Order_Profit_Per_Order`, `Order_Item_Profit_Ratio` (realized after the order; keep only if you argue they are known at order time)

**Keep as features**
- Numeric: `Days_for_shipment_(scheduled)`, `Order_Item_Quantity`, `Sales`, `Order_Item_Discount`, `Order_Item_Discount_Rate`, `Product_Price`, `Latitude`, `Longitude`, order month / day-of-week / hour / year
- Categorical: `Shipping_Mode`, `Market`, `Order_Region`, `Order_Country`, `Order_State`, `Category_Name`, `Department_Name`, `Customer_Segment`, `Type`, `Product_Status`

### 1.4 Quick baselines (measured)

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| `Shipping_Mode` alone | 0.7295 | — |
| `Days_for_shipment_(scheduled)` alone | 0.7295 | — |
| Logistic regression (subset of features) | 0.7412 | 0.8155 |
| Random forest (200 trees) | 0.7773 | 0.8388 |
| XGBoost (untuned, early stopping) | **0.8147** | **0.8631** |

The single-feature baseline is the important one: shipping mode gets 0.73 AUC by itself, and the full model adds **+0.086**. Report that lift explicitly — it preempts the reviewer objection that “the model just learned First Class is always late.” The honest framing is: “the promise window explains most of the risk; the model adds the remaining lift from region, category, value, and timing.”

### 1.5 Cost-based decision

If expediting costs `C_E` and a late delivery costs `C_L`, escalate when:

```text
p × C_L > C_E    →    p > C_E / C_L
```

With `C_E = $10` and `C_L = $30`, the threshold is **0.33**. Report the threshold curve and the cost per 1,000 orders at 0.5 vs 0.33. That is module 4 applied to a real decision, not a toy.

### 1.6 Pros and cons

**Pros:** single table; balanced target; clear action; rich EDA; no duplicates; CC BY 4.0; AUC ~0.81 without tuning; FastAPI maps naturally to an order payload.

**Cons:** 82–96 MB raw (clean it down to ~20 columns and commit a compressed/Parquet version); many PII/noise columns; the signal is concentrated in shipping mode/scheduled days, so the model is not a deep “hidden pattern” story; the “sales” angle is secondary.

---

## 2. Rossmann Store Sales — top pick for pure sales

### 2.1 Verified profile

| Check | Result |
|---|---|
| Shape | **1,017,209 rows × 9 columns** + `store.csv` (1,115 stores) |
| Date range | **2013-01-01 → 2015-07-31** |
| Target | `Sales` (daily, per store) |
| Sales (open stores) | mean **6,956**, median **6,369**, max **41,551** |
| Closed days | **172,817 rows** with `Open = 0` and `Sales = 0` |
| Promo share | 38.2% of store-days |
| Store metadata | `StoreType`, `Assortment`, `CompetitionDistance`, `CompetitionOpenSince`, `Promo2`, `Promo2Since`, `PromoInterval` |
| Missing | `CompetitionDistance` 3; `Promo2Since*` / `PromoInterval` 544 each |

Source: [Kaggle Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales). A public mirror is on [Hugging Face](https://huggingface.co/datasets/AiiN-aini/rossmann-store-sales) (`train.csv` ~38 MB, `store.csv` ~45 KB).

### 2.2 Project framing

> **Question:** forecast daily sales for each store over the next 6 weeks, so operations can plan staffing, stock, and promotions.
>
> **Split:** time-based. Train on data before 2015-06-01; validate/test on the last 6–8 weeks. Never random-split a sales series.
>
> **Features:** day-of-week, month, day, week-of-year, year, days-to-Christmas; store type, assortment, competition distance (log), months since competitor opened, `Promo2` status; `Promo`, `StateHoliday`, `SchoolHoliday`; `Open` only if you can defend that it is known in advance.
>
> **Models:** seasonal-naive baseline (same weekday last year / last 4 weeks), Ridge, random forest, XGBoost. Tune `max_depth`, `min_samples_leaf`, `n_estimators`, `eta`, `subsample`.
>
> **Metrics:** RMSE and MAE on open store-days; report the baseline gap. Avoid MAPE because `Sales` can be zero on closed days.
>
> **Deployment:** `POST /predict` with store ID, date, promo/holiday flags, and store metadata; returns forecast and a confidence band. The API does not need the sales history at request time if you use calendar + store features.

### 2.3 Why it works

- Ready target, 1M rows, 1,115 stores, strong seasonality and promo effects.
- Time-based split is **natural** and more honest than a random split — a good story for reviewers.
- Rich EDA: promo lift, store-type differences, competition distance, holiday effects, day-of-week.
- Applies modules 2 (regression, regularization), 4 (validation, RMSE), 6 (RF/XGBoost), 5 (deployment).
- Highly transferable to demand planning, retail analytics, and supply-chain roles.

### 2.4 Pros and cons

**Pros:** clean schema; ready target; strong business framing; time-series discipline; great feature-importance discussion; 38 MB is manageable.

**Cons:** 1M rows and 1,115 stores take a bit more compute; `Open=0` rows need handling; the dataset is a well-known Kaggle competition, so **do not copy a public solution** — build your own pipeline and write your own EDA; `StateHoliday` has mixed types (`0`, `a`, `b`, `c`) that need explicit handling.

**If Rossmann feels too big:** Walmart Store Sales is the same shape but smaller — **421,570 rows**, 45 stores, 81 departments, 2010-02-05 → 2012-10-26, weekly `Weekly_Sales` (mean 15,981, median 7,612, max 693,099), 7.0% holiday weeks. It needs `features.csv` and `stores.csv` for external features.

---

## 3. Online Retail II — sales transactions, no ready target

### 3.1 Verified profile

| Check | Result |
|---|---|
| Shape | **1,067,371 rows × 8 columns** |
| Date range | **2009-12-01 → 2011-12-09** |
| Revenue | **£19,287,250.57** (`Quantity × Price`) |
| Missing `Customer ID` | **243,007 (22.8%)** |
| Negative `Quantity` (returns) | **22,950** |
| `Price <= 0` | **6,207** |
| Duplicate rows | **34,335** |
| Countries | 43; United Kingdom 981,330 rows (~92%) |

Source: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), DOI `10.24432/C5CG6D`. A combined CSV mirror is on [Hugging Face](https://huggingface.co/datasets/attik/Online-Retail-II-UCI).

### 3.2 The honest warning about forecasting

This dataset has **no product category, no cost, and no ready target**. The natural project is to aggregate to a daily/weekly revenue series and forecast it. I tested that quickly:

- Weekly revenue, train before 2011, test 2011.
- Lag/rolling + calendar + trend features, Ridge and random forest.
- Best quick result: **R² ≈ 0.22**, MAE ≈ £50,859; several variants were worse than the mean.
- The test year contains a regime shift (the Christmas peak and growth), and tree models cannot extrapolate a trend.

That does **not** mean it is impossible — a proper SARIMA/Prophet/linear-trend model, or a shorter horizon, could work — but it is a **time-series project**, not a plug-in tabular classification. If you choose it, budget extra time for the split, lags, and baselines.

### 3.3 Better framings if you want sales/customer analytics

1. **RFM + next-purchase propensity (classification).** Pick a cutoff date; label customers who buy again within 90 days; features: recency, frequency, monetary, country, product diversity, return rate. Models: logistic regression, RF, XGBoost; AUC/PR-AUC. This is customer analytics, not forecasting, and it reuses modules 3–4. It is churn-adjacent — a different dataset and definition from the banned telco churn, but a reviewer may still see the resemblance.
2. **Sales driver analysis + a simple model.** Aggregate weekly revenue by country/product; predict weekly revenue from calendar and lag features. Good for EDA, weaker as a deployment story.
3. **Market-basket analysis.** Association rules are interesting but not a predictive model; it may not satisfy the rubric's “train multiple models” criterion.

**Verdict:** only choose Online Retail II if you specifically want a time-series challenge or a customer-analytics project. For a pure sales forecast, Rossmann is easier to make credible.

---

## 4. Olist Brazilian E-commerce — richest, heaviest

### 4.1 Verified profile

| Check | Result |
|---|---|
| Orders | **99,441** (96,478 delivered; 96,476 with both delivery dates) |
| Late-delivery rate | **8.11%** (actual > estimated) |
| Reviews | **99,224**; 57.8% 5-star, 11.5% 1-star |
| Date range | **2016-09-04 → 2018-10-17** |
| Tables | 9: orders, order_items, payments, reviews, customers, products, sellers, geolocation, category translation |
| Size | ~120 MB total |
| License | CC BY-NC-SA 4.0 (non-commercial) |

Source: [Kaggle Brazilian E-commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Raw CSVs are mirrored on [Hugging Face](https://huggingface.co/datasets/bulutttt/olist-raw-data).

### 4.2 Project framing

- **Late-delivery classification:** predict whether `order_delivered_customer_date > order_estimated_delivery_date`. Drop both actual delivery dates and the review fields; use only order-time features: price, freight, product category, seller/customer state, payment installments, estimated days, item count, purchase month.
- **Review-score prediction:** 1–5 stars or “bad review (≤2) vs good (≥4)”; use delivery delay, price, freight, category, seller state. Careful: delivery delay is post-hoc if you are predicting at order time.
- **Delivery-time regression:** predict `delivery_days`; regression metrics.

**Why it is attractive:** realistic e-commerce joins, multiple business questions, strong portfolio story, recognized dataset.

**Why it is risky for 10 days:** 9 tables, ~120 MB, 8.1% positive class, and every join is a chance to leak or mis-aggregate. If you pick it, use only `orders + order_items + products + customers + payments` first; add reviews/sellers only if time.

---

## 5. Bike Sharing — safest fallback

### 5.1 Verified profile

| Check | Result |
|---|---|
| Shape | **17,379 hourly** + **731 daily** rows |
| Date range | **2011-01-01 → 2012-12-31** |
| Target | `cnt` (hourly 1–977; daily 22–8,714) |
| Missing / duplicates | 0 / 0 |
| Features | season, year, month, hour, holiday, weekday, workingday, weather situation, temp, feels-like temp, humidity, wind speed |

Source: [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), DOI `10.24432/C5W894`. Direct zip; ~1.2 MB.

### 5.2 Project framing

> Hourly demand forecasting for a bike-share operator: staffing, rebalancing, and capacity planning.

- Drop `casual` and `registered` (they sum to `cnt` — leakage).
- Time-based split: last 3 months as test.
- Features: hour, day-of-week, month, season, holiday, workingday, weather, temp, humidity, wind.
- Models: Ridge, random forest, XGBoost; RMSE/MAE; compare with a seasonal-naive baseline.
- Deployment: `POST /predict` with date/time + weather → expected rentals.

**Pros:** tiny, clean, no missing values, fast to train, easy to finish with time for tests/CI/cloud. **Cons:** less “sales/logistics” than DataCo/Rossmann; simpler story; lower ceiling for a standout portfolio.

---

## 6. If you have the Kaggle API

These are stronger sales-forecasting datasets but need `~/.kaggle/kaggle.json`:

| Dataset | Size | Target | Why |
|---|---|---|---|
| [Rossmann](https://www.kaggle.com/competitions/rossmann-store-sales) | 1.0M rows | daily `Sales` | Best all-round sales forecasting |
| [Walmart Store Sales](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting) | 421k rows | weekly `Weekly_Sales` | Smaller, same skills |
| [M5 Forecasting](https://www.kaggle.com/competitions/m5-forecasting-accuracy) | 30,490 series × 1,941 days | daily unit sales | Too big for a 10-day midterm; great later |
| [Store Sales (Favorita)](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) | ~3M rows | daily `sales` | Rich but heavy |

Kaggle competition data has its own terms; check before redistributing. For a course project, committing a subset or a cleaned aggregate is usually fine, but read the rules.

---

## 7. Decision guide

| If you want… | Pick | Why |
|---|---|---|
| The best logistics project in 10 days | **DataCo** | Single table, balanced target, concrete action, AUC ~0.81 |
| The best pure sales project | **Rossmann** | Ready target, store metadata, natural time split, 1M rows |
| A sales project without Kaggle | **Online Retail II** | Direct UCI download; but no ready target and forecasting is harder |
| The richest e-commerce portfolio | **Olist** | Real joins, multiple targets; but 9 tables and 8.1% positive |
| The lowest-risk finish | **Bike Sharing** | Tiny, clean, fast; less domain relevance |
| A smaller sales dataset | **Walmart** | 421k rows, same skills as Rossmann |

**My recommendation:** **DataCo** if you want logistics, **Rossmann** if you want sales. If you have no strong preference, DataCo is the safer path to a strong rubric score because the target is balanced and the table is already flat.

---

## 8. Data access notes

- **DataCo:** official [Mendeley DOI](https://data.mendeley.com/datasets/8gx2fvg2k6/5); the page's download is a manual click, and the CSV is ~96 MB. A 53-column mirror is on [Hugging Face](https://huggingface.co/datasets/alalfi/SupplyChainDataset). Verify the row count (180,519) and columns after download.
- **Rossmann:** official [Kaggle competition](https://www.kaggle.com/competitions/rossmann-store-sales); mirror on [Hugging Face](https://huggingface.co/datasets/AiiN-aini/rossmann-store-sales) (`train.csv`, `store.csv`). Verify 1,017,209 rows and 1,115 stores.
- **Online Retail II:** official [UCI 502](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (two Excel sheets, ~45 MB zip); combined CSV mirror on [Hugging Face](https://huggingface.co/datasets/attik/Online-Retail-II-UCI). The UCI server was slow from here; the mirror downloaded in seconds.
- **Olist:** official [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce); raw mirror on [Hugging Face](https://huggingface.co/datasets/bulutttt/olist-raw-data).
- **Bike Sharing:** official [UCI 275](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset); direct zip.
- **Walmart:** official [Kaggle](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting); mirror on [Hugging Face](https://huggingface.co/datasets/large-traversaal/Walmart-sales).

For whichever you choose, commit a **cleaned** dataset (drop PII/leaks) plus a `download_data.py` and a SHA-256 in `model_metadata.json`. Reviewers should not need a Kaggle account to run the project.

---

## 9. Next step

Pick **DataCo** (logistics) or **Rossmann** (sales) — or tell me if you prefer Olist, Online Retail II, or Bike Sharing. I will then rewrite `PLAN.md` around that dataset (problem framing, leakage list, features, models, metrics, folder tree) and scaffold the project.
