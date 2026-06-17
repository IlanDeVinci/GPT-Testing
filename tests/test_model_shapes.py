def test_tiny_gpt_forward_shape() -> None:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=64,
            n_positions=16,
            n_ctx=16,
            n_embd=32,
            n_layer=1,
            n_head=4,
        )
    )
    output = model(input_ids=torch.randint(0, 64, (2, 8)))
    assert output.logits.shape == (2, 8, 64)


def test_tinybert_mlm_and_classifier_shapes() -> None:
    import torch
    from transformers import (
        BertConfig,
        BertForMaskedLM,
        BertForSequenceClassification,
    )

    config = BertConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    input_ids = torch.randint(0, 64, (2, 8))
    mlm = BertForMaskedLM(config)
    classifier = BertForSequenceClassification(config)
    assert mlm(input_ids=input_ids).logits.shape == (2, 8, 64)
    assert classifier(input_ids=input_ids).logits.shape == (2, 2)
