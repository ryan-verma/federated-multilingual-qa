from transformers import AutoTokenizer

MODEL_NAME = "xlm-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def preprocess_validation_examples(examples):

    questions = [q.strip() for q in examples["question"]]

    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")

    example_ids = []

    for i in range(len(inputs["input_ids"])):

        sample_idx = sample_map[i]

        example_ids.append(
            examples["id"][sample_idx]
        )

        sequence_ids = inputs.sequence_ids(i)

        offsets = inputs["offset_mapping"][i]

        inputs["offset_mapping"][i] = [
            offset if sequence_ids[k] == 1 else None
            for k, offset in enumerate(offsets)
        ]

    inputs["example_id"] = example_ids

    return inputs