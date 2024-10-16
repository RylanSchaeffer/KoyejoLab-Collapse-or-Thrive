import json

# Initialize counts
count_A = 0
count_B = 0
count_C = 0

# Specify the path to your jsonl file
file_path = "results_helpsteer2_iter1_iter2.jsonl"

# Open the jsonl file and process each line
count = 0
with open(file_path, "r") as file:
    for line in file:
        count += 1
        try:
            # Load each line as a JSON object
            data = json.loads(line)[
                1
            ]  # Since it's a jsonl file, each line is a JSON object
            if data:  # Ensure choices list is not empty
                # Get the content of the first choice
                response = data["choices"][0]["message"]["content"]
                # Count occurrences of A, B, and C

                if response == "A":
                    count_A += 1
                elif response == "B":
                    count_B += 1
                elif response == "C":
                    count_C += 1
        except json.JSONDecodeError:
            print(f"Error decoding JSON for line: {line}")
print(count)
# Print the results
print(f"Count of A: {count_A}")
print(f"Count of B: {count_B}")
print(f"Count of C: {count_C}")
