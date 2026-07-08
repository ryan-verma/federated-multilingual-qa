from transformers import AutoTokenizer

MODEL_NAME = "xlm-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def preprocess_with_labels(examples):
    """Like preprocess_validation_examples, but keeps start/end position
    labels so Trainer.predict() can compute loss. Used only for the
    convergence plot -- NOT for EM/F1, since it drops the offset mapping
    needed for answer-span post-processing."""

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

    offset_mapping = inputs.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):

        sample_idx = sample_map[i]

        answer = examples["answers"][sample_idx]

        if len(answer["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        sequence_ids = inputs.sequence_ids(i)

        # find start/end of context within the tokenized sequence
        context_start = 0
        while sequence_ids[context_start] != 1:
            context_start += 1

        context_end = len(sequence_ids) - 1
        while sequence_ids[context_end] != 1:
            context_end -= 1

        if (
            offsets[context_start][0] > start_char
            or offsets[context_end][1] < end_char
        ):
            # answer not fully inside this feature's window
            start_positions.append(0)
            end_positions.append(0)
        else:
            token_start = context_start
            while (
                token_start <= context_end
                and offsets[token_start][0] <= start_char
            ):
                token_start += 1
            start_positions.append(token_start - 1)

            token_end = context_end
            while (
                token_end >= context_start
                and offsets[token_end][1] >= end_char
            ):
                token_end -= 1
            end_positions.append(token_end + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs