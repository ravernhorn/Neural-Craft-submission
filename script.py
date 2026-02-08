
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, Ridge, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


class EnhancedDomainFeatureExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        # Financial terms - EXPANDED
        self.financial_patterns = [
            r'\$\s*\d+', r'\d+\s*dollar', r'\d+\s*thousand', r'\d+\s*hundred',
            r'\d+k', r'\d+\.\d+', 'payment', 'fee', 'charge', 'balance', 'debt'
        ]
        
        # Legal terms - EXPANDED
        self.legal_terms = [
            'lawsuit', 'sued', 'court', 'attorney', 'lawyer', 'legal action',
            'litigation', 'judgment', 'garnishment', 'summons', 'fdcpa',
            'cease and desist', 'violation', 'rights', 'federal', 'law'
        ]
        
        # Urgency indicators - EXPANDED
        self.urgency_terms = [
            'urgent', 'emergency', 'immediate', 'asap', 'critical', 'severe',
            'desperate', 'help', 'please', 'now', 'today', 'immediately'
        ]
        
        # Company names - EXPANDED
        self.company_patterns = [
            'equifax', 'experian', 'transunion', 'visa', 'mastercard',
            'amex', 'chase', 'wells fargo', 'bank of america', 'capital one',
            'citibank', 'discover', 'paypal', 'synchrony'
        ]
        
        # Time indicators - EXPANDED
        self.time_patterns = [
            r'\d+\s*year', r'\d+\s*month', r'\d+\s*week', r'\d+\s*day',
            'years', 'months', 'weeks', 'days', 'ago', 'since', 'ongoing'
        ]
        
        # Fraud indicators - EXPANDED
        self.fraud_terms = [
            'fraud', 'fraudulent', 'stolen', 'identity theft', 'unauthorized',
            'hacked', 'breach', 'scam', 'compromised', 'fake', 'forged'
        ]
        
        # Credit-specific terms - NEW
        self.credit_terms = [
            'credit report', 'credit score', 'fico', 'dispute', 'inquiry',
            'hard pull', 'soft pull', 'credit bureau', 'credit file'
        ]
        
        # Debt collection terms - NEW
        self.collection_terms = [
            'collector', 'collection', 'debt', 'owe', 'pay', 'validation',
            'verification', 'notice', 'letter', 'harassment'
        ]
        
        # Mortgage terms - NEW
        self.mortgage_terms = [
            'mortgage', 'foreclosure', 'escrow', 'refinance', 'home loan',
            'property', 'modification', 'appraisal', 'title'
        ]
        
        # Account/Card terms - NEW
        self.account_terms = [
            'account', 'card', 'checking', 'savings', 'overdraft', 'nsf',
            'debit', 'atm', 'withdrawal', 'deposit'
        ]
        
        self.scaler = MinMaxScaler()
        self.fitted = False
    
    def fit(self, X, y=None):
        features = self._extract_raw_features(X)
        self.scaler.fit(features)
        self.fitted = True
        return self
    
    def _extract_raw_features(self, X):
        
        features = []
        
        for text in X:
            text_lower = str(text).lower()
            feat = []
            
            # Basic text statistics
            text_len = len(text_lower)
            words = text_lower.split()
            word_count = len(words)
            avg_word_len = text_len / max(word_count, 1)
            
            feat.extend([text_len, word_count, avg_word_len])
            
            # Financial indicators 
            has_financial = int(any(re.search(p, text_lower) for p in self.financial_patterns))
            financial_count = sum(len(re.findall(p, text_lower)) for p in self.financial_patterns)
            # Extract dollar amounts
            dollar_amounts = re.findall(r'\$\s*(\d+)', text_lower)
            max_amount = max([int(a) for a in dollar_amounts], default=0)
            has_large_amount = int(max_amount > 1000)
            
            feat.extend([has_financial, financial_count, max_amount, has_large_amount])
            
            # Legal indicators
            has_legal = int(any(term in text_lower for term in self.legal_terms))
            legal_count = sum(text_lower.count(term) for term in self.legal_terms)
            feat.extend([has_legal, legal_count])
            
            # Urgency indicators
            has_urgency = int(any(term in text_lower for term in self.urgency_terms))
            urgency_count = sum(text_lower.count(term) for term in self.urgency_terms)
            feat.extend([has_urgency, urgency_count])
            
            # Company mentions
            has_company = int(any(comp in text_lower for comp in self.company_patterns))
            company_count = sum(text_lower.count(comp) for comp in self.company_patterns)
            feat.extend([has_company, company_count])
            
            # Time indicators
            has_time = int(any(re.search(p, text_lower) for p in self.time_patterns))
            time_count = sum(len(re.findall(p, text_lower)) for p in self.time_patterns)
            feat.extend([has_time, time_count])
            
            # Fraud indicators
            has_fraud = int(any(term in text_lower for term in self.fraud_terms))
            fraud_count = sum(text_lower.count(term) for term in self.fraud_terms)
            feat.extend([has_fraud, fraud_count])
            
            
            credit_count = sum(text_lower.count(term) for term in self.credit_terms)
            collection_count = sum(text_lower.count(term) for term in self.collection_terms)
            mortgage_count = sum(text_lower.count(term) for term in self.mortgage_terms)
            account_count = sum(text_lower.count(term) for term in self.account_terms)
            
            feat.extend([credit_count, collection_count, mortgage_count, account_count])
            
            # Negation count
            negation_count = sum(text_lower.count(neg) for neg in ['not_', 'never_', 'no_', 'donot_'])
            feat.append(negation_count)
            
            # Emphasis markers
            exclamation_count = text_lower.count('emphasis')
            question_count = text_lower.count('question')
            feat.extend([exclamation_count, question_count])
            
            # Capitalization
            if len(str(text)) > 0:
                caps_ratio = sum(1 for c in str(text) if c.isupper()) / len(str(text))
            else:
                caps_ratio = 0
            feat.append(caps_ratio)
            
            # Repetition indicators
            unique_word_ratio = len(set(words)) / max(len(words), 1)
            feat.append(unique_word_ratio)
            
            
            sentence_count = text_lower.count('.') + text_lower.count('!') + text_lower.count('?')
            avg_sentence_len = word_count / max(sentence_count, 1)
            feat.extend([sentence_count, avg_sentence_len])
            
            
            number_count = len(re.findall(r'\d+', text_lower))
            feat.append(number_count)
            
            features.append(feat)
        
        return np.array(features)
    
    def transform(self, X):
        """Extract and scale features"""
        features = self._extract_raw_features(X)
        
        if self.fitted:
            return self.scaler.transform(features)
        else:
            return features


class ComplaintClassifier:
    def __init__(self):
        self.primary_to_secondary = {}
        self.secondary_to_primary = {}
        self.tfidf_primary = None
        self.tfidf_secondary = {}
        self.tfidf_severity = None
        self.primary_model = None
        self.secondary_models = {}
        self.severity_model = None
        self.primary_classes = []
        self.secondary_classes = []
        self.domain_features = EnhancedDomainFeatureExtractor()
        
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Expand contractions - EXPANDED LIST
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "i'm": "i am", "it's": "it is", "that's": "that is",
            "what's": "what is", "there's": "there is",
            "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
            "i'd": "i would", "you'd": "you would", "he'd": "he would", "we'd": "we would",
            "they'd": "they would", "shouldn't": "should not", "wouldn't": "would not",
            "couldn't": "could not", "haven't": "have not", "hasn't": "has not",
            "hadn't": "had not", "doesn't": "does not", "didn't": "did not",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Preserve negations with underscore
        text = re.sub(r'\bnot\s+', 'not_', text)
        text = re.sub(r'\bnever\s+', 'never_', text)
        text = re.sub(r'\bno\s+', 'no_', text)
        text = re.sub(r'\bcannot\s+', 'cannot_', text)
        text = re.sub(r'\bwill\s+not\s+', 'willnot_', text)
        text = re.sub(r'\bdo\s+not\s+', 'donot_', text)
        text = re.sub(r'\bdoes\s+not\s+', 'doesnot_', text)
        text = re.sub(r'\bdid\s+not\s+', 'didnot_', text)
        
        # Keep punctuation markers
        text = text.replace('!', ' emphasis ')
        text = text.replace('?', ' question ')
        
        # Remove special characters
        text = re.sub(r'[^a-z0-9\s_]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def create_advanced_vectorizer(self, max_features=5000):
        """Create enhanced vectorizer with better parameters"""
        
        # Word-level TF-IDF - IMPROVED
        word_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 5),  # INCREASED to 5-grams
            min_df=2,
            max_df=0.85,  # REDUCED from 0.9
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\b\w+\b',
            use_idf=True,
            smooth_idf=True,
            norm='l2'
        )
        
        # Character-level TF-IDF - IMPROVED
        char_vectorizer = TfidfVectorizer(
            max_features=max_features // 3,  # INCREASED from //4
            ngram_range=(3, 6),  # INCREASED to 6
            min_df=2,  # REDUCED from 3
            max_df=0.85,  # REDUCED from 0.9
            sublinear_tf=True,
            analyzer='char',
            use_idf=True,
            norm='l2'
        )
        
        return word_vectorizer, char_vectorizer
    
    def train_primary_model(self, df):
        """Train ENHANCED stacking ensemble with XGBoost/LightGBM"""
        print("Training primary category model with BOOSTED ensemble...")
        
        # Create feature extractors
        word_vec, char_vec = self.create_advanced_vectorizer(max_features=5000)
        
        self.tfidf_primary = FeatureUnion([
            ('word', word_vec),
            ('char', char_vec)
        ])
        
        X_text = self.tfidf_primary.fit_transform(df['complaint_text_clean'])
        X_domain = self.domain_features.fit_transform(df['complaint_text_clean'])
        
        from scipy.sparse import hstack, csr_matrix
        X = hstack([X_text, csr_matrix(X_domain)])
        
        y = df['primary_category']
        self.primary_classes = df['primary_category'].unique().tolist()
        
        # ENHANCED: Add XGBoost and LightGBM to base estimators
        base_estimators = [
            ('svm', LinearSVC(C=1.5, class_weight='balanced', max_iter=2500, random_state=42, dual=False, loss='squared_hinge')),
            ('lr', LogisticRegression(C=2.0, max_iter=2500, class_weight='balanced', solver='lbfgs', random_state=42, n_jobs=-1)),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=30, min_samples_split=4, class_weight='balanced', random_state=42, n_jobs=-1)),
            ('xgb', xgb.XGBClassifier(n_estimators=150, max_depth=7, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric='mlogloss')),
            ('lgb', lgb.LGBMClassifier(n_estimators=150, max_depth=7, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1))
        ]
        
        # Stronger meta-learner
        meta_learner = LogisticRegression(C=1.5, max_iter=1500, solver='lbfgs', random_state=42)
        
        print("  Creating enhanced stacking classifier...")
        self.primary_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5,  # INCREASED from 3
            n_jobs=-1,
            passthrough=False
        )
        
        print("  Fitting stacking model...")
        self.primary_model.fit(X, y)
        print("  Primary model trained successfully!")
    
    def train_secondary_models(self, df):
        """Train ENHANCED hierarchical models with boosting"""
        print("Training secondary category models...")
        
        # Build mappings
        for _, row in df.iterrows():
            primary = row['primary_category']
            secondary = row['secondary_category']
            if secondary not in self.secondary_to_primary:
                self.secondary_to_primary[secondary] = primary
        
        # Group by primary category
        for primary in df['primary_category'].unique():
            primary_df = df[df['primary_category'] == primary].copy()
            
            if len(primary_df) < 5:
                continue
            
            secondary_counts = primary_df['secondary_category'].value_counts()
            if primary not in self.primary_to_secondary:
                self.primary_to_secondary[primary] = secondary_counts.index[0]
            
            if primary_df['secondary_category'].nunique() < 2:
                continue
            
            # Create feature extractors
            word_vec, char_vec = self.create_advanced_vectorizer(max_features=2500)
            
            vectorizer = FeatureUnion([
                ('word', word_vec),
                ('char', char_vec)
            ])
            
            X_text = vectorizer.fit_transform(primary_df['complaint_text_clean'])
            X_domain = self.domain_features.transform(primary_df['complaint_text_clean'])
            
            from scipy.sparse import hstack, csr_matrix
            X = hstack([X_text, csr_matrix(X_domain)])
            
            y = primary_df['secondary_category']
            
            # ENHANCED: Use XGBoost/LightGBM ensemble for secondary
            if len(primary_df) > 50:  # Enough data for boosting
                svm = LinearSVC(C=1.2, class_weight='balanced', max_iter=2000, random_state=42, dual=False)
                lr = LogisticRegression(C=1.8, max_iter=2000, class_weight='balanced', solver='lbfgs', random_state=42)
                xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='mlogloss')
                
                model = VotingClassifier(
                    estimators=[('svm', svm), ('lr', lr), ('xgb', xgb_model)],
                    voting='hard'
                )
            else:
                # Simpler model for small datasets
                svm = LinearSVC(C=1.0, class_weight='balanced', max_iter=1500, random_state=42, dual=False)
                lr = LogisticRegression(C=1.5, max_iter=1500, class_weight='balanced', solver='lbfgs', random_state=42)
                
                model = VotingClassifier(
                    estimators=[('svm', svm), ('lr', lr)],
                    voting='hard'
                )
            
            model.fit(X, y)
            
            self.tfidf_secondary[primary] = vectorizer
            self.secondary_models[primary] = model
            
            print(f"  Trained for '{primary[:40]}...' ({len(primary_df)} samples)")
    
    def train_severity_model(self, df):
        """Train ENHANCED ensemble regressor with XGBoost"""
        print("Training severity model with XGBoost ensemble...")
        
        word_vec, char_vec = self.create_advanced_vectorizer(max_features=3500)
        
        self.tfidf_severity = FeatureUnion([
            ('word', word_vec),
            ('char', char_vec)
        ])
        
        X_text = self.tfidf_severity.fit_transform(df['complaint_text_clean'])
        X_domain = self.domain_features.transform(df['complaint_text_clean'])
        
        from scipy.sparse import hstack, csr_matrix
        X = hstack([X_text, csr_matrix(X_domain)])
        
        y = df['severity']
        
        # ENHANCED: Use XGBoost instead of Ridge
        self.severity_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror'
        )
        
        self.severity_model.fit(X, y)
        
        # Evaluate
        from sklearn.metrics import r2_score
        y_pred = self.severity_model.predict(X)
        y_pred_clipped = np.clip(y_pred, 1, 5)
        r2 = r2_score(y, y_pred_clipped)
        print(f"  Severity model R²: {r2:.4f}")
    
    def fit(self, df):
        """Build all models from training data"""
        print("Preprocessing training data...")
        df = df.copy()
        df['complaint_text_clean'] = df['complaint_text'].apply(self.preprocess_text)
        
        self.train_primary_model(df)
        self.train_secondary_models(df)
        self.train_severity_model(df)
        
        self.secondary_classes = df['secondary_category'].unique().tolist()
        
        print("Training complete!")
        return self
    
    def predict_primary(self, texts):
        """Batch predict primary category"""
        X_text = self.tfidf_primary.transform(texts)
        X_domain = self.domain_features.transform(texts)
        
        from scipy.sparse import hstack, csr_matrix
        X = hstack([X_text, csr_matrix(X_domain)])
        
        preds = self.primary_model.predict(X)
        return preds
    
    def predict_secondary(self, text, primary):
        """Predict secondary category"""
        if primary in self.secondary_models:
            try:
                X_text = self.tfidf_secondary[primary].transform([text])
                X_domain = self.domain_features.transform([text])
                
                from scipy.sparse import hstack, csr_matrix
                X = hstack([X_text, csr_matrix(X_domain)])
                
                pred = self.secondary_models[primary].predict(X)[0]
                return pred
            except:
                pass
        
        if primary in self.primary_to_secondary:
            return self.primary_to_secondary[primary]
        
        return self.secondary_classes[0] if self.secondary_classes else 'Unknown'
    
    def predict_severity(self, texts):
        """Batch predict severity"""
        X_text = self.tfidf_severity.transform(texts)
        X_domain = self.domain_features.transform(texts)
        
        from scipy.sparse import hstack, csr_matrix
        X = hstack([X_text, csr_matrix(X_domain)])
        
        preds = self.severity_model.predict(X)
        preds = np.clip(preds, 1, 5)
        
        return preds
    
    def predict(self, df):
        """Predict all three outputs - BATCH PROCESSING"""
        print("Making predictions...")
        df = df.copy()
        df['complaint_text_clean'] = df['complaint_text'].apply(self.preprocess_text)
        
        texts = df['complaint_text_clean'].tolist()
        
        print("  Predicting primary categories...")
        primary_preds = self.predict_primary(texts)
        
        print("  Predicting severity scores...")
        severity_preds = self.predict_severity(texts)
        
        print("  Predicting secondary categories...")
        secondary_preds = []
        for text, primary in zip(texts, primary_preds):
            secondary = self.predict_secondary(text, primary)
            secondary_preds.append(secondary)
        
        predictions = pd.DataFrame({
            'complaint_id': df['complaint_id'],
            'primary_category': primary_preds,
            'secondary_category': secondary_preds,
            'severity': np.round(severity_preds, 2)
        })
        
        return predictions


def main():
    """Main execution pipeline"""
    
    TRAIN_PATH = '/kaggle/input/neural-craft/neural-craft-2026/train_complaints.csv'
    TEST_PATH = '/kaggle/input/neural-craft/neural-craft-2026/test_complaints.csv'
    OUTPUT_PATH = 'predictions.csv'
    
    print("=" * 60)
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"Loaded {len(train_df)} training samples")
    print(f"Primary categories: {train_df['primary_category'].nunique()}")
    print(f"Secondary categories: {train_df['secondary_category'].nunique()}")
    
    print("=" * 60)
    classifier = ComplaintClassifier()
    classifier.fit(train_df)
    
    print("=" * 60)
    print("Loading test data...")
    test_df = pd.read_csv(TEST_PATH)
    print(f"Loaded {len(test_df)} test samples")
    
    print("=" * 60)
    predictions = classifier.predict(test_df)
    
    print("=" * 60)
    print("Saving predictions...")
    predictions.to_csv(OUTPUT_PATH, index=False)
    print(f"Done! Predictions saved to {OUTPUT_PATH}")
    
    print("\nSample predictions:")
    print(predictions.head(10))
    
    if 'primary_category' in test_df.columns:
        from sklearn.metrics import accuracy_score, r2_score
        
        print("=" * 60)
        print("Evaluation Results:")
        
        primary_acc = accuracy_score(test_df['primary_category'], predictions['primary_category'])
        secondary_acc = accuracy_score(test_df['secondary_category'], predictions['secondary_category'])
        severity_r2 = r2_score(test_df['severity'], predictions['severity'])
        
        final_score = 0.3 * primary_acc + 0.4 * secondary_acc + 0.3 * severity_r2
        
        print(f"Primary Accuracy:     {primary_acc:.4f} (weight: 0.3)")
        print(f"Secondary Accuracy:   {secondary_acc:.4f} (weight: 0.4)")
        print(f"Severity R²:          {severity_r2:.4f} (weight: 0.3)")
        print(f"{'=' * 60}")
        print(f"FINAL WEIGHTED SCORE: {final_score:.4f}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()