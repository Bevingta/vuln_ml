import re
import json
from collections import defaultdict
import numpy as np
from google.colab import files

class SmartVulnerabilityDetector:
    """Rule-based detector for C/C++ code vulnerabilities without machine learning."""

    def __init__(self):
        # Risk functions and their properties
        self.risky_functions = {
            # Buffer overflow risk functions
            'strcpy': {'risk_level': 'high', 'category': 'buffer_overflow', 'needs_size_check': True},
            'strcat': {'risk_level': 'high', 'category': 'buffer_overflow', 'needs_size_check': True},
            'gets': {'risk_level': 'high', 'category': 'buffer_overflow', 'needs_size_check': True},
            'sprintf': {'risk_level': 'high', 'category': 'buffer_overflow', 'needs_size_check': True},
            'vsprintf': {'risk_level': 'high', 'category': 'buffer_overflow', 'needs_size_check': True},
            'strncpy': {'risk_level': 'medium', 'category': 'buffer_overflow', 'needs_null_termination': True},
            'strncat': {'risk_level': 'medium', 'category': 'buffer_overflow', 'needs_size_check': True},
            'snprintf': {'risk_level': 'low', 'category': 'buffer_overflow', 'needs_size_check': False},

            # Memory management
            'malloc': {'risk_level': 'medium', 'category': 'memory_management', 'needs_null_check': True},
            'calloc': {'risk_level': 'medium', 'category': 'memory_management', 'needs_null_check': True},
            'realloc': {'risk_level': 'medium', 'category': 'memory_management', 'needs_null_check': True},
            'free': {'risk_level': 'medium', 'category': 'memory_management', 'check_use_after': True},

            # Format string vulnerabilities
            'printf': {'risk_level': 'medium', 'category': 'format_string', 'direct_arg_risky': True},
            'fprintf': {'risk_level': 'medium', 'category': 'format_string', 'direct_arg_risky': True},
            'scanf': {'risk_level': 'medium', 'category': 'format_string', 'needs_size_check': True},
            'sscanf': {'risk_level': 'medium', 'category': 'format_string', 'needs_size_check': True},
        }

        # Risk weights
        self.category_weights = {
            'buffer_overflow': 10,
            'memory_management': 8,
            'format_string': 7,
            'integer_overflow': 6,
            'array_bounds': 9,
            'race_condition': 5
        }

        # Risk level multipliers
        self.risk_multipliers = {'high': 1.0, 'medium': 0.7, 'low': 0.3}

    def extract_identifiers(self, code):
        """Extract variable and function identifiers from code"""
        # Extract variable declarations
        var_pattern = r'\b(int|char|float|double|void|unsigned|long|short|size_t|struct|const)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        var_matches = re.findall(var_pattern, code)
        variables = [match[1] for match in var_matches]

        # Extract array declarations
        array_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\[\s*([^\]]*)\s*\]'
        array_matches = re.findall(array_pattern, code)
        arrays = [match[0] for match in array_matches]

        # Extract function names
        func_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        functions = re.findall(func_pattern, code)

        # Extract function parameters
        param_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[,\)]'
        param_matches = re.findall(param_pattern, code)
        parameters = [match[1] for match in param_matches]

        # Combine all identifiers
        identifiers = set(variables + arrays + parameters)

        return identifiers, functions, arrays

    def check_buffer_overflow(self, code, identifiers, arrays):
        """Check for potential buffer overflow vulnerabilities"""
        issues = []

        # Check for dangerous string functions
        for func, props in self.risky_functions.items():
            if props['category'] != 'buffer_overflow':
                continue

            # Find function usages
            pattern = r'\b' + re.escape(func) + r'\s*\(([^)]*)\)'
            matches = re.findall(pattern, code)

            for args in matches:
                if func in ['strcpy', 'strcat', 'gets']:
                    # Check if target is a fixed-size buffer
                    args_list = [a.strip() for a in args.split(',')]
                    if len(args_list) > 0:
                        target = args_list[0]
                        if target in arrays:
                            # Check if a size check exists
                            size_check_pattern = r'(strlen|sizeof)\s*\(\s*' + re.escape(args_list[-1] if len(args_list) > 1 else '') + r'\s*\)'
                            if not re.search(size_check_pattern, code):
                                issues.append({
                                    'type': 'Buffer Overflow Risk',
                                    'severity': props['risk_level'],
                                    'function': func,
                                    'details': f"Unbounded {func}() used with buffer '{target}' without size checking"
                                })

                elif func in ['strncpy']:
                    # Check for null termination after strncpy
                    args_list = [a.strip() for a in args.split(',')]
                    if len(args_list) > 0:
                        target = args_list[0]
                        # Look for explicit null termination
                        null_term_pattern = re.escape(target) + r'\s*\[\s*([^]]*)\s*\]\s*=\s*[\'"\\]0[\'"]'
                        if not re.search(null_term_pattern, code):
                            issues.append({
                                'type': 'Missing Null Termination',
                                'severity': 'medium',
                                'function': func,
                                'details': f"strncpy() used with buffer '{target}' without explicit null termination"
                            })

        # Check for array access without bounds checking
        for array in arrays:
            # Find array accesses
            access_pattern = re.escape(array) + r'\s*\[\s*([^]]*)\s*\]'
            accesses = re.findall(access_pattern, code)

            for index in accesses:
                # Check if index is a variable and not a constant
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', index.strip()):
                    # Look for bounds checking
                    bounds_check_pattern = r'if\s*\(\s*' + re.escape(index.strip()) + r'\s*[<>=]+\s*([^){}]*)\s*\)'
                    if not re.search(bounds_check_pattern, code):
                        issues.append({
                            'type': 'Array Bounds Check Missing',
                            'severity': 'medium',
                            'function': 'array_access',
                            'details': f"Array '{array}' accessed with variable index '{index}' without bounds checking"
                        })

        return issues

    def check_memory_issues(self, code, identifiers):
        """Check for memory-related issues like leaks and use-after-free"""
        issues = []

        # Track memory allocation
        alloc_pattern = r'\b(malloc|calloc|realloc)\s*\(([^)]*)\)'
        alloc_matches = re.findall(alloc_pattern, code)

        # Extract variables that store allocated memory
        allocated_vars = set()
        for func, args in alloc_matches:
            # Find assignment target
            assign_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*' + func + r'\s*\(([^)]*)\)'
            assign_matches = re.findall(assign_pattern, code)
            if assign_matches:
                for var, _ in assign_matches:
                    allocated_vars.add(var)

                    # Check for NULL check after allocation
                    null_check_pattern = r'if\s*\(\s*' + re.escape(var) + r'\s*==\s*NULL\s*\)|if\s*\(\s*!\s*' + re.escape(var) + r'\s*\)'
                    if not re.search(null_check_pattern, code):
                        issues.append({
                            'type': 'Missing NULL Check',
                            'severity': 'high',
                            'function': func,
                            'details': f"Memory allocation using {func}() not checked for NULL for pointer '{var}'"
                        })

        # Check for free() usage
        free_pattern = r'\bfree\s*\(\s*([^)]*)\s*\)'
        free_matches = re.findall(free_pattern, code)
        freed_vars = set(free_matches)

        # Check for use after free
        for var in freed_vars:
            # Get code after free statement
            free_loc = code.find('free(' + var + ')')
            if free_loc != -1:
                code_after_free = code[free_loc + len('free(' + var + ')'):]

                # Check for usage after free
                use_pattern = r'\b' + re.escape(var) + r'\b[^=]'  # Match but not if being reassigned
                if re.search(use_pattern, code_after_free):
                    issues.append({
                        'type': 'Use After Free',
                        'severity': 'high',
                        'function': 'free',
                        'details': f"Pointer '{var}' is used after being freed"
                    })

        # Check for memory leaks (allocated but not freed)
        potentially_leaked = allocated_vars - freed_vars
        if potentially_leaked and not re.search(r'\breturn\s+[a-zA-Z_][a-zA-Z0-9_]*', code):
            for var in potentially_leaked:
                issues.append({
                    'type': 'Potential Memory Leak',
                    'severity': 'medium',
                    'function': 'malloc/calloc/realloc',
                    'details': f"Memory allocated to '{var}' may not be freed"
                })

        return issues

    def check_format_string_vulnerabilities(self, code):
        """Check for format string vulnerabilities"""
        issues = []

        # Check printf-family functions for format string vulnerabilities
        for func in ['printf', 'fprintf', 'sprintf']:
            pattern = r'\b' + re.escape(func) + r'\s*\(\s*([^,)]*)'
            matches = re.findall(pattern, code)

            for format_arg in matches:
                format_arg = format_arg.strip()

                # Skip if empty
                if not format_arg:
                    continue

                # If first argument is a variable and not a string literal, it's potentially vulnerable
                if not (format_arg.startswith('"') or format_arg.startswith("'")):
                    # Check that it's not an obvious fmt, format, etc. variable
                    if not any(format_name in format_arg.lower() for format_name in ['fmt', 'format']):
                        issues.append({
                            'type': 'Format String Vulnerability',
                            'severity': 'high',
                            'function': func,
                            'details': f"Function {func}() uses '{format_arg}' directly as format string"
                        })

        return issues

    def check_integer_issues(self, code):
        """Check for integer overflow/underflow issues"""
        issues = []

        # Look for arithmetic on loop counters or array indices
        for op in ['+', '-', '*', '/', '%']:
            # Find operations that could overflow
            pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*' + re.escape(op) + r'\s*([^;]*)'
            matches = re.findall(pattern, code)

            for target, var1, var2 in matches:
                # If target is used as an array index, check for overflow protection
                array_index_pattern = r'\[[^]]*\b' + re.escape(target) + r'\b[^]]*\]'
                if re.search(array_index_pattern, code):
                    # Look for bounds checking or safe casts
                    bounds_check_pattern = r'if\s*\(\s*' + re.escape(target) + r'\s*[<>=]'
                    safe_cast_pattern = r'(size_t|int|long|unsigned)\s+' + re.escape(target)

                    if not (re.search(bounds_check_pattern, code) or re.search(safe_cast_pattern, code)):
                        issues.append({
                            'type': 'Potential Integer Overflow',
                            'severity': 'medium',
                            'function': 'arithmetic',
                            'details': f"Integer '{target}' used in arithmetic without overflow checking"
                        })

        return issues

    def check_race_conditions(self, code):
        """Check for potential race conditions in multithreaded code"""
        issues = []

        # Check if code appears to be multithreaded
        thread_indicators = ['pthread_', 'thread', 'mutex', 'lock', 'unlock', 'atomic']
        is_multithreaded = any(indicator in code for indicator in thread_indicators)

        if is_multithreaded:
            # Check for shared variable access without locks
            identifiers, _, _ = self.extract_identifiers(code)

            for var in identifiers:
                # Look for lock acquisition before variable usage
                lock_pattern = r'(pthread_mutex_lock|lock)\s*\([^)]*\)[^;]*' + re.escape(var)
                var_usage_pattern = r'\b' + re.escape(var) + r'\b\s*[=\.\-\+\*\/\[\]]'

                if re.search(var_usage_pattern, code) and not re.search(lock_pattern, code):
                    # Check if the variable appears to be shared across threads
                    if re.search(r'(pthread_create|thread)\s*\([^)]*\)', code):
                        issues.append({
                            'type': 'Potential Race Condition',
                            'severity': 'medium',
                            'function': 'thread',
                            'details': f"Variable '{var}' may be accessed from multiple threads without synchronization"
                        })

        return issues

    def calculate_risk_score(self, issues):
        """Calculate an overall risk score from identified issues"""
        if not issues:
            return 0.0

        # Initialize category scores
        category_scores = defaultdict(float)

        # Aggregate scores by category
        for issue in issues:
            category = next((props['category'] for func, props in self.risky_functions.items()
                          if func == issue.get('function')), 'other')

            # Use specific category from issue if available
            if issue['function'] == 'array_access':
                category = 'array_bounds'
            elif issue['function'] == 'arithmetic':
                category = 'integer_overflow'
            elif issue['function'] == 'thread':
                category = 'race_condition'

            risk_multiplier = self.risk_multipliers.get(issue['severity'], 0.5)
            category_scores[category] += risk_multiplier

        # Calculate weighted score
        total_score = 0.0
        for category, score in category_scores.items():
            weight = self.category_weights.get(category, 5)  # Default weight for unknown categories
            total_score += score * weight

        # Normalize to 0-1 range
        max_possible_score = sum(self.category_weights.values())  # Maximum possible weighted score
        normalized_score = min(1.0, total_score / max_possible_score)

        return normalized_score

    def analyze_code(self, code):
        """Main function to analyze code for vulnerabilities"""
        # Preprocess code
        code = code.replace('\t', '    ')  # Standardize indentation

        # Extract identifiers and arrays
        identifiers, functions, arrays = self.extract_identifiers(code)

        # Perform different vulnerability checks
        buffer_issues = self.check_buffer_overflow(code, identifiers, arrays)
        memory_issues = self.check_memory_issues(code, identifiers)
        format_issues = self.check_format_string_vulnerabilities(code)
        integer_issues = self.check_integer_issues(code)
        race_issues = self.check_race_conditions(code)

        # Combine all issues
        all_issues = buffer_issues + memory_issues + format_issues + integer_issues + race_issues

        # Calculate risk score
        risk_score = self.calculate_risk_score(all_issues)

        # Sort issues by severity
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        all_issues.sort(key=lambda x: severity_order.get(x['severity'], 3))

        # Make final vulnerability determination
        is_vulnerable = risk_score > 0.25 or any(issue['severity'] == 'high' for issue in all_issues)
        confidence = 'high' if risk_score > 0.6 else 'medium' if risk_score > 0.3 else 'low'

        return {
            'prediction': 'Vulnerable' if is_vulnerable else 'Safe',
            'confidence': confidence,
            'risk_score': risk_score,
            'issues': all_issues,
            'issue_count': len(all_issues)
        }


def run_vulnerability_test(test_data, sample_size=None):
    """Run the vulnerability detector on test data and report results"""
    detector = SmartVulnerabilityDetector()

    # Sample data if requested
    if sample_size and sample_size < len(test_data):
        import random
        random.seed(42)
        sampled_data = random.sample(test_data, sample_size)
    else:
        sampled_data = test_data

    # Tracking metrics
    total = len(sampled_data)
    correct = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    # Category tracking
    cwe_performance = defaultdict(lambda: {'total': 0, 'detected': 0})
    issue_types = defaultdict(int)

    print(f"Testing on {total} code samples...")

    # Run analysis on each sample
    for i, item in enumerate(sampled_data):
        if i % 50 == 0:
            print(f"Progress: {i}/{total}")

        code = item.get("func", "")
        true_label = item.get("target", 0) == 1  # 1 = vulnerable, 0 = safe
        cwes = item.get("cwe", [])

        # Run the detector
        analysis = detector.analyze_code(code)
        prediction = analysis["prediction"] == "Vulnerable"

        # Update metrics
        if prediction == true_label:
            correct += 1
            if prediction:
                true_pos += 1
            else:
                true_neg += 1
        else:
            if prediction:
                false_pos += 1
            else:
                false_neg += 1

        # Track CWE performance
        for cwe in cwes:
            cwe_performance[cwe]['total'] += 1
            if prediction:
                cwe_performance[cwe]['detected'] += 1

        # Track issue types
        for issue in analysis['issues']:
            issue_types[issue['type']] += 1

    # Calculate overall metrics
    accuracy = correct / total * 100 if total > 0 else 0
    precision = true_pos / (true_pos + false_pos) * 100 if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) * 100 if (true_pos + false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Print results
    print("\n===== PERFORMANCE SUMMARY =====")
    print(f"Total samples tested: {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}")
    print(f"\nTrue Positives: {true_pos}")
    print(f"True Negatives: {true_neg}")
    print(f"False Positives: {false_pos}")
    print(f"False Negatives: {false_neg}")

    # Print CWE performance if relevant
    if cwe_performance:
        print("\n===== CWE PERFORMANCE =====")
        print(f"{'CWE':<15} | {'Total':<8} | {'Detected':<10} | {'Rate':<10}")
        print("-" * 50)

        for cwe, stats in sorted(cwe_performance.items()):
            if stats['total'] > 0:
                detection_rate = stats['detected'] / stats['total'] * 100
                print(f"{cwe:<15} | {stats['total']:<8} | {stats['detected']:<10} | {detection_rate:.2f}%")

    # Print issue type summary
    if issue_types:
        print("\n===== DETECTED ISSUE TYPES =====")
        print(f"{'Issue Type':<30} | {'Count':<8}")
        print("-" * 40)

        for issue_type, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
            print(f"{issue_type:<30} | {count:<8}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_pos,
        'true_negatives': true_neg,
        'false_positives': false_pos,
        'false_negatives': false_neg
    }

# Main execution code - for Colab
def main():
    print("=== C/C++ Vulnerability Detector ===")
    print("1. Upload test data file")
    print("2. Use the provided sample data")
    choice = input("Enter your choice (1/2): ")

    if choice == '1':
        print("Please upload your JSON test file...")
        uploaded = files.upload()
        test_file = list(uploaded.keys())[0]
        with open(test_file, 'r') as f:
            test_data = json.load(f)
    else:
        # Sample data from the prompt
        test_data = [
            {
                "idx": 274442,
                "func": "void AppModalDialog::CloseModalDialog() {\n  DCHECK(native_dialog_);\n  native_dialog_->CloseAppModalDialog();\n }\n",
                "target": 0,
                "cwe": ["CWE-20"],
                "cve": "CVE-2012-2877"
            },
            {
                "idx": 124642,
                "func": "ldap_start_tls( LDAP *ld,\n\tLDAPControl **serverctrls,\n\tLDAPControl **clientctrls,\n\tint *msgidp )\n{\n\treturn ldap_extended_operation( ld, LDAP_EXOP_START_TLS,\n\t\tNULL, serverctrls, clientctrls, msgidp );\n}",
                "target": 0,
                "cwe": ["CWE-617"],
                "cve": "CVE-2020-36230"
            },
            # Additional examples would be here
        ]

    print(f"Loaded {len(test_data)} test samples")

    # Ask for sample size
    sample_input = input("How many samples to test? (Enter for all): ")
    sample_size = int(sample_input) if sample_input.strip() else None

    # Run the test
    results = run_vulnerability_test(test_data, sample_size)

    # Ask if user wants to save results
    save_choice = input("Save results to file? (y/n): ")
    if save_choice.lower() == 'y':
        with open('vulnerability_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        files.download('vulnerability_results.json')
        print("Results saved and downloaded.")

if __name__ == "__main__":
    main()
