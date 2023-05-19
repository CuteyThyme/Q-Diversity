from transformers import BertTokenizer


def check_data(s1, s2, label, labels_type):
    if len(s1.split()) == 0 or len(s2.split()) == 0:
        return False
    elif label in labels_type:
        return True
    else:
        return False


def processSentences(examples, max_seq_len):
    input_ids, attention_masks, segment_ids, label_ids, confounder_ids = [], [], [], [], []
   
    MAX_SEQ_LENGTH = max_seq_len
    BERT_PATH = "/home/v-wuting/Desktop/llm_models/bert-base-uncased/"
    
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    for example in examples:
        tokens_1 = tokenizer.tokenize(example.s1)
        tokens_2 = None
        if example.s2:
            tokens_2 = tokenizer.tokenize(example.s2)
            _truncate_seq_pair(tokens_1, tokens_2, MAX_SEQ_LENGTH - 3)
        else:
            if len(tokens_1) > MAX_SEQ_LENGTH - 2:
                tokens_1 = tokens_1[:(MAX_SEQ_LENGTH - 2)]

        tokens = ["[CLS]"] + tokens_1 + ["[SEP]"]
        segment_id = [0] * len(tokens) 
        attention_mask = [1] * len(tokens)

        if tokens_2:
            tokens += tokens_2 + ["[SEP]"]
            segment_id += [1] * (len(tokens_2) + 1)
            attention_mask += [1] * (len(tokens_2) + 1)

        input_id = tokenizer.convert_tokens_to_ids(tokens)

        if len(input_id) < MAX_SEQ_LENGTH: 
            padding_length = MAX_SEQ_LENGTH - len(input_id)
            input_id = input_id + ([tokenizer.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            segment_id = segment_id + ([0] * padding_length)
         
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        segment_ids.append(segment_id)
        label_ids.append(example.label)
        confounder_ids.append(example.confounder)
    
    return input_ids, attention_masks, segment_ids, label_ids, confounder_ids
   


def _truncate_seq_pair(tokens_1, tokens_2, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_1) + len(tokens_2)
        if total_length <= max_length:
            break
        if len(tokens_1) > len(tokens_2):
            tokens_1.pop()
        else:
            tokens_2.pop()