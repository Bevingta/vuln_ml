import json, re, gc, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

class VulnFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vulnerability_patterns = [
            ('strcpy_unbounded', r'\bstrcpy\s*\([^;]*;(?!.*if)'),
            ('strcat_unbounded', r'\bstrcat\s*\([^;]*;(?!.*if)'),
            ('gets_usage', r'\bgets\s*\('),
            ('fixed_buffer', r'\[\s*\d+\s*\]'),
            ('printf_direct_var', r'\bprintf\s*\(\s*\w+\s*\)'),
            ('malloc_without_size_check', r'\bmalloc\s*\([^;]*;(?!.*if)'),
            ('double_free_pattern', r'free\s*\([^;]*;.*free\s*\('),
            ('use_after_free', r'free\s*\([^;]*;\s*[^{]*\w+\s*(?:=|==|!=|>=|<=|>|<|\.|\->|\[)'),
            ('system_with_var', r'\bsystem\s*\(\s*\w+\s*\)'),
            ('null_deref_risk', r'\*\s*\w+(?!.*?if)')
        ]
        self.safety_patterns = [
            ('bounds_check', r'if\s*\(\s*\w+\s*(?:<|<=|>=|>)\s*(?:\w+|sizeof)'),
            ('null_check', r'if\s*\(\s*\w+\s*==\s*(?:NULL|nullptr|0)\)'),
            ('size_check', r'if\s*\(\s*(?:strlen|sizeof)\s*\(')
        ]

    def fit(self, X, y=None): return self

    def transform(self, X):
        features = []
        for code in X:
            feat = {'code_length': len(code), 'line_count': code.count('\n') + 1}

            for name, pattern in self.vulnerability_patterns:
                matches = len(re.findall(pattern, code))
                feat[f'has_{name}'] = 1 if matches > 0 else 0
                feat[f'{name}_count'] = min(matches, 5)

            for name, pattern in self.safety_patterns:
                matches = len(re.findall(pattern, code))
                feat[f'has_{name}'] = 1 if matches > 0 else 0
                feat[f'{name}_count'] = min(matches, 5)

            features.append(list(feat.values()))
        return np.array(features)

class BalancedVulnDetector:
    def __init__(self, batch_size=500, text_features=1000):
        self.batch_size = batch_size
        self.text_features = text_features
        self.text_vectorizer = HashingVectorizer(n_features=text_features, ngram_range=(1, 2))
        self.code_feature_extractor = VulnFeatureExtractor()
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=4,
            max_features='sqrt', class_weight='balanced', n_jobs=-1, random_state=42
        )
        self.best_threshold = 0.5

    def preprocess_code(self, code):
        if not isinstance(code, str): return ""
        code = code.lower()
        code = re.sub(r'\/\/.*?$|\/\*.*?\*\/', ' ', code, flags=re.MULTILINE)
        code = re.sub(r'\s+', ' ', code).strip()
        return code

    def load_chunks(self, data_path):
        with open(data_path, 'r') as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == '[':
                data = json.loads(f.read())
                for i in range(0, len(data), self.batch_size):
                    yield pd.DataFrame(data[i:i+self.batch_size])
            else:
                chunk = []
                for line in f:
                    if line.strip():
                        try:
                            obj = json.loads(line.strip())
                            chunk.append(obj)
                            if len(chunk) >= self.batch_size:
                                yield pd.DataFrame(chunk)
                                chunk = []
                        except json.JSONDecodeError: pass
                if chunk: yield pd.DataFrame(chunk)

    def train(self, train_path, valid_path=None):
        total_samples = vuln_samples = non_vuln_samples = 0
        first_batch = True

        # Count class distribution
        for df_chunk in tqdm(self.load_chunks(train_path), desc="Counting class distribution"):
            if 'target' in df_chunk.columns:
                chunk_vuln = df_chunk['target'].sum()
                vuln_samples += chunk_vuln
                non_vuln_samples += len(df_chunk) - chunk_vuln
                total_samples += len(df_chunk)

        # Update model with class weights
        class_weight = {
            0: 1.0,
            1: (total_samples / (2 * vuln_samples)) if vuln_samples > 0 else 3.0
        }
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=4,
            max_features='sqrt', class_weight=class_weight, n_jobs=-1, random_state=42
        )

        # Training data preparation
        all_X_train, all_y_train = [], []

        for df_chunk in tqdm(self.load_chunks(train_path), desc="Training batches"):
            X_text = np.array([self.preprocess_code(x) for x in df_chunk['func'].values])

            if first_batch:
                X_text_vec = self.text_vectorizer.fit_transform(X_text)
                first_batch = False
            else:
                X_text_vec = self.text_vectorizer.transform(X_text)

            X_code_features = self.code_feature_extractor.transform(df_chunk['func'].values)
            X_combined = np.hstack((X_text_vec.toarray(), X_code_features))
            y = df_chunk['target'].values if 'target' in df_chunk.columns else None

            all_X_train.append(X_combined)
            all_y_train.append(y)

            del X_text, X_text_vec, X_code_features
            gc.collect()

        X_train = np.vstack(all_X_train)
        y_train = np.concatenate(all_y_train)
        self.model.fit(X_train, y_train)

        del X_train, y_train, all_X_train, all_y_train
        gc.collect()

        # Find balanced threshold
        if valid_path:
            self.optimize_threshold(valid_path)

    def optimize_threshold(self, test_path):
        all_probs, all_targets = [], []

        for df_chunk in self.load_chunks(test_path):
            if 'target' in df_chunk.columns:
                X_text = np.array([self.preprocess_code(x) for x in df_chunk['func'].values])
                X_text_vec = self.text_vectorizer.transform(X_text)
                X_code_features = self.code_feature_extractor.transform(df_chunk['func'].values)
                X_combined = np.hstack((X_text_vec.toarray(), X_code_features))

                probs = self.model.predict_proba(X_combined)[:, 1]
                all_probs.extend(probs)
                all_targets.extend(df_chunk['target'].values)

                del X_text, X_text_vec, X_code_features, X_combined
                gc.collect()

        y_proba = np.array(all_probs)
        y_test = np.array(all_targets)

        best_score, best_threshold = -1, 0.5
        for threshold in np.linspace(0.3, 0.9, 13):
            y_pred = (y_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            review_burden = (tp + fp) / len(y_test)

            # Balance score: 70% recall, 30% low review burden
            balance_score = (0.7 * recall) - (0.3 * review_burden)
            adjusted_score = balance_score if recall >= 0.8 else balance_score * (recall / 0.8)

            if adjusted_score > best_score:
                best_score = adjusted_score
                best_threshold = threshold

        self.best_threshold = best_threshold

    def predict(self, code, threshold=None):
        if threshold is None: threshold = self.best_threshold

        preprocessed_code = self.preprocess_code(code)
        X_text = self.text_vectorizer.transform([preprocessed_code])
        X_code = self.code_feature_extractor.transform([code])
        X_combined = np.hstack((X_text.toarray(), X_code))

        prob = self.model.predict_proba(X_combined)[0, 1]
        pred = 1 if prob >= threshold else 0

        return pred, prob

    def analyze(self, code):
        pred, prob = self.predict(code)

        issues = []
        if re.search(r'\bstrcpy\s*\([^;]*;', code) and not re.search(r'if\s*\(\s*\w+\s*<', code):
            issues.append(("Buffer Overflow", "Usage of strcpy without bounds checking"))

        if re.search(r'\bgets\s*\(', code):
            issues.append(("Buffer Overflow", "Usage of gets() function (always unsafe)"))

        if re.search(r'printf\s*\(\s*\w+\s*\)', code):
            issues.append(("Format String", "Direct use of variable in printf without format specifier"))

        if re.search(r'\bmalloc\s*\(', code) and not re.search(r'\bfree\s*\(', code):
            issues.append(("Memory Leak", "Memory allocation without corresponding deallocation"))

        if re.search(r'free\s*\([^;]*;.*free\s*\(', code):
            issues.append(("Double Free", "Potential double free vulnerability"))

        if re.search(r'free\s*\([^;]*;\s*[^{]*\w+\s*(?:=|==|!=|>=|<=|>|<|\.|\->|\[)', code):
            issues.append(("Use-After-Free", "Potential use-after-free vulnerability"))

        if not issues and pred == 1:
            issues.append(("Unknown", "Potential vulnerability detected by the model"))

        return {
            'is_vulnerable': bool(pred),
            'probability': float(prob),
            'issues': issues
        }

from tqdm import tqdm

def evaluate(self, test_path):
    all_preds, all_probs, all_targets = [], [], []
    vuln_types = {'buffer': 0, 'format': 0, 'memory': 0, 'integer': 0, 'command': 0, 'other': 0}
    detected = {'buffer': 0, 'format': 0, 'memory': 0, 'integer': 0, 'command': 0, 'other': 0}

    for df_chunk in tqdm(self.load_chunks(test_path), desc="Evaluating"):
        if 'target' in df_chunk.columns:
            X_text = np.array([self.preprocess_code(x) for x in df_chunk['func'].values])
            X_text_vec = self.text_vectorizer.transform(X_text)
            X_code_features = self.code_feature_extractor.transform(df_chunk['func'].values)
            X_combined = np.hstack((X_text_vec.toarray(), X_code_features))

            probs = self.model.predict_proba(X_combined)[:, 1]
            preds = (probs >= self.best_threshold).astype(int)

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_targets.extend(df_chunk['target'].values)

            # Categorize vulnerabilities if CWE info is available
            if 'cwe' in df_chunk.columns:
                for i, (code, cwe, pred) in enumerate(zip(df_chunk['func'].values, df_chunk['cwe'].values, preds)):
                    if not isinstance(cwe, list): continue

                    # Skip if no CWE or not vulnerable
                    if not cwe or df_chunk['target'].values[i] == 0: continue

                    # Categorize vulnerability
                    vuln_category = 'other'
                    for c in cwe:
                        if c in ['119', '120', '121', '122', '124', '126', '127', '129', '131', '190', '680']:
                            vuln_category = 'buffer'  # Buffer-related
                        elif c in ['134', '789']:
                            vuln_category = 'format'  # Format string
                        elif c in ['401', '476', '415', '416', '562', '761']:
                            vuln_category = 'memory'  # Memory management
                        elif c in ['190', '191', '680']:
                            vuln_category = 'integer'  # Integer issues
                        elif c in ['77', '78', '88']:
                            vuln_category = 'command'  # Command injection

                    vuln_types[vuln_category] += 1
                    if pred == 1:
                        detected[vuln_category] += 1

            del X_text, X_text_vec, X_code_features, X_combined
            gc.collect()

    # Calculate core metrics
    y_pred = np.array(all_preds)
    y_test = np.array(all_targets)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        'samples': len(y_test),
        'accuracy': (tp + tn) / len(y_test),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'review_burden': (tp + fp) / len(y_test),
        'detection_by_type': {k: {'total': vuln_types[k], 'detected': detected[k],
                                 'recall': detected[k]/vuln_types[k] if vuln_types[k] > 0 else 0}
                             for k in vuln_types},
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'threshold': self.best_threshold
    }

    return results

def main():
    print("Training model...")

    # Train the model
    detector = BalancedVulnDetector(batch_size=500)
    detector.train('../data/train.json', '../data/test.json')

    print("Testing model...")

    # Evaluate on test set to get performance metrics
    results = detector.evaluate('test.json')

    # Print concise performance report
    print(f"MODEL PERFORMANCE SUMMARY:")
    print(f"Overall Metrics:")
    print(f"- Accuracy: {results['accuracy']:.4f}")
    print(f"- Precision: {results['precision']:.4f}")
    print(f"- Recall: {results['recall']:.4f}")
    print(f"- F1 Score: {results['f1']:.4f}")
    print(f"- False Positive Rate: {results['false_positive_rate']:.4f}")
    print(f"- Review Burden: {results['review_burden']*100:.1f}%")

    # Print performance by vulnerability type
    print(f"\nPerformance by Vulnerability Type:")
    for vuln_type, data in results['detection_by_type'].items():
        if data['total'] > 0:
            print(f"- {vuln_type.capitalize()}: {data['detected']}/{data['total']} detected ({data['recall']*100:.1f}%)")

    # Confusion matrix
    cm = results['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"True Positives: {cm['tp']} | False Negatives: {cm['fn']}")
    print(f"False Positives: {cm['fp']} | True Negatives: {cm['tn']}")

    # Test on examples
    examples = [
        """static int vulnerable_function(char *input) {
            char buffer[10];
            strcpy(buffer, input);
            return 0;
        }""",

        """void log_message(char *user_input) {
            printf(user_input);
        }""",

        """void process_data(char *data) {
            char *ptr = malloc(100);
            free(ptr);
            ptr[0] = 'A';
        }""",

        """static int safe_function(const char *input) {
            char buffer[10];
            size_t input_len = strlen(input);
            if (input_len >= sizeof(buffer)) {
                return -1;
            }
            strncpy(buffer, input, sizeof(buffer) - 1);
            buffer[sizeof(buffer) - 1] = '\0';
            return 0;
        }"""
    ]

    # Analyze each example
    print("\nEXAMPLE ANALYSES:")
    for i, code in enumerate(examples):
        analysis = detector.analyze(code)
        print(f"Example {i+1}:")
        print(f"- Vulnerable: {analysis['is_vulnerable']}, Probability: {analysis['probability']:.4f}")
        for issue_type, description in analysis['issues']:
            print(f"- {issue_type}: {description}")
        print()

if __name__ == "__main__":
    main()