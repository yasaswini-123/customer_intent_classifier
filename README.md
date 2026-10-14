# Customer Service Intent Classifier

> **Mini project built during my internship at SURE ProEd** (a unit of SURE Trust, Skill Upgradation for Rural-youth Empowerment).
> Mentor: `Mr. Vamsi Krishna` | Author: `Balla Yasaswini`

Type a customer message such as *"I want to reset my password"* and the app predicts the **intent** behind it (for example `recover_password`), shows how confident the model is, and lists the top 3 predictions. A **TF-IDF + Logistic Regression** model is trained on the Bitext sample customer service dataset and served through a white-themed **Streamlit** web app.

## What it does

- Predicts one of **27 customer service intents** from a short text message.
- Shows the intent in readable form, its **category** (for example Account, Order, Refund) and the confidence.
- Shows the **top 3** most likely intents with probability bars.
- Shows an **"inconclusive"** message when the confidence is below 25%, instead of presenting a weak guess as an answer.
- Offers example messages you can click, and lists all intents the model knows.

## How it works

```
 Customer message
        |
        v
 TF-IDF (single words + two-word phrases, lowercase)
        |
        v
 Logistic Regression (balanced class weights)
        |
        v
 Probability for each of the 27 intents
        |
        v
 Highest probability >= 25%  ->  predicted intent + category
 Highest probability <  25%  ->  "inconclusive" message
```

The Streamlit app trains the model once when it starts (a few seconds) and keeps it in memory, so no saved model file is needed.

## Project structure

```
customer_service_intent_classifier/
├── .streamlit/
│   └── config.toml                  # forces the white (light) theme
├── datasets/
│   ├── Bitext_Sample_Customer_Service_Training_Dataset.csv
│   ├── Bitext_Sample_Customer_Service_Validation_Dataset.csv
│   └── Bitext_Sample_Customer_Service_Testing_Dataset.csv
├── app.py                           # Streamlit application
├── assignment2_customer_support.ipynb   # data checks, training and evaluation
├── requirements.txt
└── README.md
```

## Setup and run

Tested with Python 3.10.

```bash
# 1. clone the repository
git clone <your-repository-url>
cd customer_service_intent_classifier

# 2. create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. install the requirements
pip install -r requirements.txt

# 4. start the app (run it from inside this folder)
streamlit run app.py
```

Then open http://localhost:8501.

Notes:
- Run `streamlit run app.py` **from the project folder**. The app looks for the CSV files in the `datasets` folder next to `app.py`.
- The three CSV files must stay inside `datasets/`, and their names must contain "training", "validation" and "testing".
- The white theme comes from `.streamlit/config.toml`. If you delete that file, the app still uses white styling, but the page may look slightly different in dark mode.

## Dataset

The project uses the **Bitext Sample Customer Service** dataset, provided as three CSV files with the columns `utterance`, `intent`, `category` and `tags`.

| File | Rows |
|---|---|
| Training | 6,539 |
| Validation | 818 |
| Testing | 818 |
| **Total** | **8,175** |

- **27 intents**, with roughly 290 to 324 messages each (a balanced dataset).
- **11 categories** that group the intents.
- No missing values and no duplicate rows were found.

| Category | Intents |
|---|---|
| Account | create_account, delete_account, edit_account, recover_password, registration_problems, switch_account |
| Cancellation fee | check_cancellation_fee |
| Contact | contact_customer_service, contact_human_agent |
| Delivery | delivery_options, delivery_period |
| Feedback | complaint, review |
| Invoice | check_invoice, get_invoice |
| Newsletter | newsletter_subscription |
| Order | cancel_order, change_order, place_order, track_order |
| Payment | check_payment_methods, payment_issue |
| Refund | check_refund_policy, get_refund, track_refund |
| Shipping address | change_shipping_address, set_up_shipping_address |

Source: `[add the dataset page link here, for example the Kaggle or Hugging Face page you downloaded it from]` | Licence: `[add the licence shown on that page]`

## Model

| Step | Component | Settings |
|---|---|---|
| Features | `TfidfVectorizer` | lowercase, word n-grams `(1, 2)`, `sublinear_tf=True` |
| Classifier | `LogisticRegression` | `max_iter=3000`, `class_weight="balanced"`, `random_state=42` |

Logistic Regression was chosen because it trains quickly on sparse TF-IDF features and gives a probability for every intent, which the app uses for its confidence score and the inconclusive message.

## Results

In the notebook, the model was trained and evaluated on the training CSV (6,539 messages) with a stratified 80/20 split: 5,231 for training and **1,308 held-out messages** for testing.

**Accuracy: 99.39%** (macro and weighted F1: 0.99)

The weakest intents on the held-out messages:

| Intent | Precision | Recall | F1-score |
|---|---|---|---|
| registration_problems | 0.94 | 0.98 | 0.96 |
| create_account | 0.98 | 0.96 | 0.97 |
| cancel_order | 0.98 | 0.98 | 0.98 |

All other intents score between 0.99 and 1.00 F1. Only 8 of the 1,308 messages were misclassified. The most frequent mix-up was `create_account` predicted as `registration_problems` (2 messages), two closely related account intents.

The Streamlit app combines all three CSV files (8,175 messages), splits them 80/20 itself and shows the accuracy on its own held-out 20%, so the number on screen can differ slightly from the notebook (around 99%).

**Please read these numbers with care.** The Bitext messages are generated variations of the same intents, so messages in the test set closely resemble those in training. Real customers use wording the dataset does not contain. For example, "where is my package" is not recognised as an order-tracking request. The app gives it only about 9% confidence, so it is reported as inconclusive. Accuracy on real customer messages will be lower than 99%.

## Limitations and future work

- **Synthetic-style data.** The dataset covers a limited range of wording. Collecting real customer messages would give a more realistic test.
- **Separate test file not used for evaluation.** The provided validation and testing CSVs are combined into the training pool. A cleaner evaluation would train on the training file, tune on the validation file and report the final score on the testing file once.
- **English only**, and one intent per message (a message asking for two things gets one answer).
- **Out-of-scope messages.** Anything unrelated to the 27 intents is still mapped to the closest one unless the confidence falls below 25%.
- Ideas: compare other models (Linear SVM, Naive Bayes, a transformer such as DistilBERT), add an "other / out of scope" class, save the trained model to a file so the app starts instantly, and tune the 25% confidence threshold on real messages.

## Libraries used

Streamlit, scikit-learn, pandas.

## Acknowledgements

- **SURE ProEd** (a unit of SURE Trust) and my mentor `Mr. Vamsi Krishna` for guidance during the internship.
- Bitext, for the sample customer service dataset.
- Libraries: Streamlit, scikit-learn, pandas.

## Author

`Balla Yasaswini` | [GitHub](https://github.com/yasaswini-123) | [LinkedIn](https:linkedin.com/in/ballayasaswini)

## Licence

To be confirmed with the mentor. The dataset licence is listed above and applies to the data, not to this code.
