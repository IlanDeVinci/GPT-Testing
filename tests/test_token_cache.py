def test_token_accuracy_perfect_prediction():
    import torch
    from scripts.train_model import token_accuracy

    ids = torch.tensor([[1, 2, 3, 4]])
    logits = torch.zeros(1, 4, 10)
    for position in range(3):
        logits[0, position, ids[0, position + 1]] = 10.0
    assert token_accuracy(logits, ids) == 1.0


def test_build_cache_and_lazy_dataset(tmp_path):
    from transformers import PreTrainedTokenizerFast
    from scripts.train_model import LazyBlockDataset, build_token_cache

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="tokenizer-v3/tokenizer.json",
        pad_token="[PAD]", unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]",
    )
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join("Une phrase de test bien formée ici." for _ in range(50)) + "\n",
        encoding="utf-8",
    )

    cache = build_token_cache(corpus, tokenizer, "continuous")
    assert cache.exists()

    dataset = LazyBlockDataset(cache, block_size=16)
    assert len(dataset) >= 1
    item = dataset[0]
    assert item["input_ids"].shape[0] == 16
    assert item["labels"].shape[0] == 16
    assert item["attention_mask"].shape[0] == 16
