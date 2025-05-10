# Minimal code to train and use the balanced vulnerability detector

print("Training model...")

# Train the model
detector = BalancedVulnDetector(batch_size=500)
detector.train('train.json', 'test.json')

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
